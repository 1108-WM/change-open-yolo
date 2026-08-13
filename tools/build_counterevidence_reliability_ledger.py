#!/usr/bin/env python3
"""构建“二维未覆盖是否可信”为反证的无 GT superpoint 特征账本。

输入为固定的自动 SAM 多视角观测组、冻结 YOLO-World+SAM 观测和既有正反
可见性账本。对观测组中的每个已有 ScanNet superpoint，记录反证频率、相机视角
独立性、深度/二维 mask 边界可靠性、冻结语义、局部三维几何和 native 候选关系。
本工具不读取 GT，不删除点，不产生候选、类别、分数、融合或 AP。
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_prediction(root, scene_name):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 native mask 维度异常：{masks.shape}")
    if masks.shape[0] < masks.shape[1]:
        masks = masks.T
    return np.asarray(masks, dtype=bool)


def _load_observations(scene_root):
    result = {}
    with (scene_root / "observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["points"] = np.unique(np.asarray(np.load(row["point_indices_path"])["point_indices"], dtype=np.int64))
            result[int(row["observation_id"])] = row
    return result


def _geometry(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return {"sp_bbox_diagonal_m": 0.0, "sp_planarity": 0.0, "sp_linearity": 0.0, "sp_scattering": 0.0}
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    values = np.linalg.eigvalsh(np.cov(points.T))
    values = np.maximum(values, 0.0)
    largest = max(float(values[2]), 1e-12)
    return {
        "sp_bbox_diagonal_m": diagonal,
        "sp_planarity": float((values[1] - values[0]) / largest),
        "sp_linearity": float((values[2] - values[1]) / largest),
        "sp_scattering": float(values[0] / largest),
    }


def _native_relation(points, masks):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if len(points) == 0 or masks.shape[1] == 0:
        return {"sp_top_native_iou": 0.0, "sp_inside_top_native_ratio": 0.0, "sp_top_native_covered_ratio": 0.0, "sp_native_overlap_count": 0}
    sizes = masks.sum(axis=0, dtype=np.int64)
    intersections = masks[points].sum(axis=0, dtype=np.int64)
    ious = intersections / np.maximum(1, len(points) + sizes - intersections)
    top = int(np.argmax(ious))
    return {
        "sp_top_native_iou": float(ious[top]),
        "sp_inside_top_native_ratio": float(intersections[top] / len(points)),
        "sp_top_native_covered_ratio": float(intersections[top] / max(1, sizes[top])),
        "sp_native_overlap_count": int(np.sum(ious >= 0.10)),
    }


def _mask_sample_features(mask, coords):
    """采样二维 mask 内部/边界，而非把 3D 点邻近错误当作二维边界。"""
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.asarray(mask, dtype=bool)
    coords = np.asarray(coords, dtype=np.int64)
    if len(coords) == 0:
        return 0.0, 0.0, 0.0
    xs, ys = coords[:, 0], coords[:, 1]
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not valid.any():
        return 0.0, 0.0, 0.0
    xs, ys = xs[valid], ys[valid]
    inside = mask[ys, xs]
    padded = np.pad(mask, 2, mode="edge")
    boundary = np.zeros(len(xs), dtype=bool)
    for offset_y in range(5):
        for offset_x in range(5):
            boundary |= padded[ys + offset_y, xs + offset_x] != inside
    return (
        float(inside.mean()),
        float(((~inside) & boundary).mean()),
        float((inside & (~boundary)).mean()),
    )


def _depth_sample_features(depth, projected, point_depth, depth_scale):
    if len(projected) == 0:
        return 0.0, 0.0
    xs, ys = projected[:, 0], projected[:, 1]
    valid = (xs >= 1) & (xs + 1 < depth.shape[1]) & (ys >= 1) & (ys + 1 < depth.shape[0])
    if not valid.any():
        return 0.0, 0.0
    xs, ys, point_depth = xs[valid], ys[valid], point_depth[valid]
    observed = depth[ys, xs].astype(np.float32) / float(depth_scale)
    valid_depth = observed > 0
    if not valid_depth.any():
        return 0.0, 0.0
    xs, ys, observed, point_depth = xs[valid_depth], ys[valid_depth], observed[valid_depth], point_depth[valid_depth]
    residual = np.abs(observed - point_depth)
    jumps = np.maximum.reduce([
        np.abs(observed - depth[ys - 1, xs].astype(np.float32) / float(depth_scale)),
        np.abs(observed - depth[ys + 1, xs].astype(np.float32) / float(depth_scale)),
        np.abs(observed - depth[ys, xs - 1].astype(np.float32) / float(depth_scale)),
        np.abs(observed - depth[ys, xs + 1].astype(np.float32) / float(depth_scale)),
    ])
    return float(np.mean(residual)), float(np.median(jumps))


def _pairwise_view_features(camera_centers, centroid):
    if len(camera_centers) < 2:
        return 0.0, 0.0
    centers = np.asarray(camera_centers, dtype=np.float64)
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    directions = centroid[None, :] - centers
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    return float(pairwise.max()), float(np.degrees(np.arccos(cosine)).max())


def _sp_point_depth(points, pose):
    homo = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    return (np.linalg.inv(pose) @ homo.T).T[:, 2]


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    semantic_rows = json.loads((args.semantic_root / scene_name / "automatic_track_yoloworld_semantics.json").read_text())
    semantic_by_track = {int(row["track_id"]): row for row in semantic_rows}
    evidence_rows = json.loads((args.evidence_root / scene_name / "visibility_counterevidence_ledger.json").read_text())
    evidence_by_track = {int(row["track_id"]): row for row in evidence_rows}
    observations = _load_observations(args.yoloworld_sam_root / scene_name)
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    xyz = np.asarray(processed[:, :3], dtype=np.float32)
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    masks = _load_prediction(args.prediction_cache_dir, scene_name)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.numpy().astype(np.int64, copy=False)
    visibility = visibility.numpy().astype(bool, copy=False)
    scaling = (world.depth_resolution[0] / world.image_resolution[0], world.depth_resolution[1] / world.image_resolution[1])
    poses = [np.loadtxt(path) for path in world.poses]
    camera_centers = [pose[:3, 3] for pose in poses]
    depth_cache = {}
    mask_cache = {}
    global_points_by_sp = {int(sp): np.flatnonzero(superpoints == sp) for sp in np.unique(superpoints)}
    track_sp_sets = {}
    for track in tracks:
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(superpoints))]
        track_sp_sets[int(track["track_id"])] = {int(sp): points[superpoints[points] == sp] for sp in np.unique(superpoints[points])}
    # 仅作为明确标注的“质心 KNN 连通性代理”，不冒充 mesh 邻接关系。
    used_sp_ids = sorted({sp for mapping in track_sp_sets.values() for sp in mapping})
    centroids = {sp: xyz[global_points_by_sp[sp]].mean(axis=0) for sp in used_sp_ids}
    if len(used_sp_ids) > 1:
        ids_array = np.asarray(used_sp_ids, dtype=np.int64)
        tree = cKDTree(np.stack([centroids[sp] for sp in used_sp_ids]))
        _, knn = tree.query(np.stack([centroids[sp] for sp in used_sp_ids]), k=min(7, len(used_sp_ids)))
        knn = np.atleast_2d(knn)
        knn_by_sp = {int(sp): set(int(ids_array[idx]) for idx in np.atleast_1d(knn[row]) if int(ids_array[idx]) != int(sp)) for row, sp in enumerate(ids_array)}
    else:
        knn_by_sp = {sp: set() for sp in used_sp_ids}
    accumulators = {}
    tasks_by_observation = defaultdict(list)
    for track in tracks:
        track_id = int(track["track_id"])
        evidence = evidence_by_track[track_id]
        npz = np.load(evidence["superpoint_evidence_path"])
        ids = np.asarray(npz["superpoint_ids"], dtype=np.int64)
        if len(ids) == 0:
            continue
        index_by_sp = {int(sp): index for index, sp in enumerate(ids)}
        for sp, points in track_sp_sets[track_id].items():
            if sp not in index_by_sp:
                continue
            idx = index_by_sp[sp]
            accumulators[(track_id, sp)] = {
                "selected_coverages": [], "uncovered_boundary": [], "covered_interior": [],
                "depth_residuals": [], "depth_jumps": [], "counter_camera_centers": [],
                "matched_views": 0, "eligible_views": 0,
                "visible_view_count": int(npz["visible_view_counts"][idx]),
                "positive_weight": float(npz["positive_weights"][idx]),
                "negative_weight": float(npz["negative_weights"][idx]),
                "evidence_margin": float(npz["evidence_margins"][idx]),
            }
        for frame in evidence["frame_evidence"]:
            observation_id = int(frame["selected_observation_id"])
            if observation_id < 0 or observation_id not in observations:
                continue
            frame_index = int(frame["frame_index"])
            for sp, points in track_sp_sets[track_id].items():
                key = (track_id, sp)
                if key not in accumulators:
                    continue
                visible_points = points[visibility[frame_index, points]]
                if len(visible_points):
                    tasks_by_observation[observation_id].append((key, frame_index, visible_points, bool(frame["counterevidence_eligible"])))
    for observation_id, tasks in tasks_by_observation.items():
        observation = observations[observation_id]
        mask = mask_cache.get(observation_id)
        if mask is None:
            mask = np.asarray(imageio.imread(observation["mask_path"]), dtype=bool)
            mask_cache[observation_id] = mask
        frame_index = int(observation["frame_index"])
        depth = depth_cache.get(frame_index)
        if depth is None:
            depth = np.asarray(imageio.imread(world.depth_maps_paths[frame_index]))
            depth_cache[frame_index] = depth
        for key, frame_index, visible_points, eligible in tasks:
            # 固定均匀采样，控制账本成本；不依赖 GT 或结果标签。
            if len(visible_points) > args.max_points_per_sp_view:
                sample = visible_points[np.linspace(0, len(visible_points) - 1, args.max_points_per_sp_view, dtype=np.int64)]
            else:
                sample = visible_points
            projected = projections[frame_index, sample]
            rgb_coords = np.stack([
                np.rint(projected[:, 0] / scaling[1]).astype(np.int64),
                np.rint(projected[:, 1] / scaling[0]).astype(np.int64),
            ], axis=1)
            coverage, near_boundary, interior = _mask_sample_features(mask, rgb_coords)
            depths = _sp_point_depth(xyz[sample].astype(np.float64), poses[frame_index])
            residual, jump = _depth_sample_features(depth, projected, depths, args.depth_scale)
            item = accumulators[key]
            item["matched_views"] += 1
            item["selected_coverages"].append(coverage)
            item["uncovered_boundary"].append(near_boundary)
            item["covered_interior"].append(interior)
            item["depth_residuals"].append(residual)
            item["depth_jumps"].append(jump)
            if eligible:
                item["eligible_views"] += 1
                item["counter_camera_centers"].append(camera_centers[frame_index])
    records = []
    for track in tracks:
        track_id = int(track["track_id"])
        semantic = semantic_by_track.get(track_id, {})
        current_track_sps = set(track_sp_sets[track_id])
        for sp, track_points in track_sp_sets[track_id].items():
            item = accumulators.get((track_id, sp))
            if item is None:
                continue
            global_points = global_points_by_sp[sp]
            baseline, angle = _pairwise_view_features(item["counter_camera_centers"], centroids[sp])
            positive, negative = item["positive_weight"], item["negative_weight"]
            denominator = max(positive + negative, 1e-8)
            record = {
                "scene_name": scene_name, "track_id": track_id, "superpoint_id": int(sp),
                "track_superpoint_point_count": int(len(track_points)), "scene_superpoint_point_count": int(len(global_points)),
                "track_superpoint_coverage": float(len(track_points) / max(1, len(global_points))),
                "visible_view_count": item["visible_view_count"], "matched_observation_view_count": item["matched_views"],
                "counterevidence_eligible_view_count": item["eligible_views"],
                "counterevidence_eligible_view_rate": float(item["eligible_views"] / max(1, item["visible_view_count"])),
                "positive_weight": positive, "negative_weight": negative, "evidence_margin": item["evidence_margin"],
                "negative_weight_ratio": float(negative / denominator),
                "is_negative_margin": bool(item["evidence_margin"] < 0.0),
                "mask_coverage_mean": float(np.mean(item["selected_coverages"])) if item["selected_coverages"] else 0.0,
                "mask_coverage_std": float(np.std(item["selected_coverages"])) if item["selected_coverages"] else 0.0,
                "mask_coverage_min": float(np.min(item["selected_coverages"])) if item["selected_coverages"] else 0.0,
                "uncovered_near_2px_mask_boundary_ratio": float(np.mean(item["uncovered_boundary"])) if item["uncovered_boundary"] else 0.0,
                "covered_mask_2px_interior_ratio": float(np.mean(item["covered_interior"])) if item["covered_interior"] else 0.0,
                "counter_view_max_camera_baseline_m": baseline, "counter_view_max_view_angle_deg": angle,
                "depth_residual_mean_m": float(np.mean(item["depth_residuals"])) if item["depth_residuals"] else 0.0,
                "depth_discontinuity_median_m": float(np.median(item["depth_jumps"])) if item["depth_jumps"] else 0.0,
                "yoloworld_voted_class_index": int(semantic.get("voted_class_index", -1)),
                "yoloworld_vote_margin": float(semantic.get("vote_margin", 0.0)),
                "yoloworld_voted_class_support_views": int(semantic.get("voted_class_support_views", 0)),
                "alphaclip_track_evidence_available": False,
                "centroid_knn_track_connectivity_proxy": float(len(knn_by_sp.get(sp, set()) & current_track_sps) / max(1, len(knn_by_sp.get(sp, set())))),
                **_geometry(xyz[global_points]), **_native_relation(track_points, masks),
            }
            records.append(record)
    del world, projections, visibility, processed, masks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--evidence_root", type=Path, required=True)
    parser.add_argument("--yoloworld_sam_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--max_points_per_sp_view", type=int, default=128)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "semantic_root", "evidence_root", "yoloworld_sam_root", "prediction_cache_dir", "processed_scene_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for index, scene in enumerate(scenes, start=1):
        records = _scene_records(scene, args)
        all_records.extend(records)
        scene_root = args.output_root / scene
        scene_root.mkdir()
        with (scene_root / "counterevidence_superpoint_reliability.jsonl").open("w") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[场景完成] {index}/{len(scenes)} {scene}: {len(records)} 条 superpoint 证据", flush=True)
    summary = {
        "gt_usage": "不读取 GT；不删除 superpoint，不产生候选、类别、分数、融合或 AP。",
        "decision_state": "仅固定并导出反证据可靠性特征，尚未定义任何接受、拒绝或删点规则。",
        "missing_feature_note": "当前自动轨迹没有逐轨迹 Alpha-CLIP 缓存；该字段显式标为不可用，未以其他结果替代。",
        "scene_count": len(scenes), "superpoint_record_count": len(all_records),
        "negative_margin_record_count": sum(row["is_negative_margin"] for row in all_records),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "counterevidence_superpoint_reliability_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

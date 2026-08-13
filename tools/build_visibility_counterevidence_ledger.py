#!/usr/bin/env python3
"""构建基线条件下的多视角正反可见性证据账本，不读取 GT、不生成候选。

每条类别无关自动 SAM 轨迹是一个待判定区域。脚本在其每个可见帧中查询同一
冻结 YOLO-World 类别的独立 ``YOLO-World + SAM`` 观测：被 mask 覆盖记为
正证据；当一个整体上匹配该轨迹的 mask 却稳定排除某 superpoint 时，记为
可见反证据。输出是供后续 GT-only 可分性审计使用的特征和 superpoint 账本，
绝不输出最终三维候选、类别、分数或 AP。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


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
    return masks


def _load_track_semantics(path):
    rows = json.loads(path.read_text())
    return {int(row["track_id"]): int(row["voted_class_index"]) for row in rows}


def _load_detection_observations(scene_root):
    """按(帧, 类别)索引冻结 YOLO-World+SAM mask 的三维点。"""
    by_frame_class = defaultdict(list)
    with (scene_root / "observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            points = np.unique(np.asarray(np.load(raw["point_indices_path"])["point_indices"], dtype=np.int64))
            by_frame_class[(int(raw["frame_index"]), int(raw["class_id"]))].append({
                "points": points,
                "quality": float(raw["score"]) * float(raw["sam_score"]),
                "score": float(raw["score"]),
                "sam_score": float(raw["sam_score"]),
                "observation_id": int(raw["observation_id"]),
            })
    return by_frame_class


def _best_observation(visible_track_points, observations):
    """只按与当前轨迹可见部分的覆盖选本帧同类观测，未做候选接受决定。"""
    visible_track_points = np.unique(np.asarray(visible_track_points, dtype=np.int64))
    if len(visible_track_points) == 0 or not observations:
        return None
    best = None
    for observation in observations:
        covered = np.intersect1d(visible_track_points, observation["points"], assume_unique=True)
        coverage = float(len(covered) / len(visible_track_points))
        # 覆盖优先；质量和 ID 仅用于确定性打破并列。
        rank = (coverage, observation["quality"], -observation["observation_id"])
        if best is None or rank > best[0]:
            best = (rank, observation, covered, coverage)
    if best is None:
        return None
    _, observation, covered, coverage = best
    return {**observation, "covered_points": covered, "coverage": coverage}


def _candidate_relation(track_points, masks):
    """记录与 native 候选的原始重叠特征，刻意不执行旧的硬路由规则。"""
    track_points = np.unique(np.asarray(track_points, dtype=np.int64))
    if len(track_points) == 0 or masks.shape[1] == 0:
        return {
            "top_native_candidate_id": -1,
            "top_native_iou": 0.0,
            "track_inside_top_native_ratio": 0.0,
            "top_native_covered_ratio": 0.0,
            "native_candidate_overlap_count": 0,
        }
    candidate_sizes = masks.sum(axis=0, dtype=np.int64)
    intersections = masks[track_points].sum(axis=0, dtype=np.int64)
    ious = intersections / np.maximum(1, len(track_points) + candidate_sizes - intersections)
    top = int(np.argmax(ious))
    return {
        "top_native_candidate_id": top,
        "top_native_iou": float(ious[top]),
        "track_inside_top_native_ratio": float(intersections[top] / len(track_points)),
        "top_native_covered_ratio": float(intersections[top] / max(1, candidate_sizes[top])),
        "native_candidate_overlap_count": int(np.sum(ious >= 0.10)),
    }


def _track_evidence(
    track_points,
    class_id,
    superpoints,
    visibility,
    observations_by_frame_class,
    min_visible_points,
    min_counterevidence_anchor_coverage,
):
    """统计轨迹每个 superpoint 的可见、正支持与可见反证据权重。"""
    track_points = np.unique(np.asarray(track_points, dtype=np.int64))
    point_to_sp = superpoints[track_points]
    track_superpoints = np.unique(point_to_sp)
    sp_to_index = {int(sp): index for index, sp in enumerate(track_superpoints)}
    visible_counts = np.zeros(len(track_superpoints), dtype=np.int32)
    positive_weights = np.zeros(len(track_superpoints), dtype=np.float32)
    negative_weights = np.zeros(len(track_superpoints), dtype=np.float32)
    frame_rows = []
    for frame_index in range(visibility.shape[0]):
        visible_mask = visibility[frame_index, track_points]
        visible_points = track_points[visible_mask]
        if len(visible_points) < min_visible_points:
            continue
        visible_sp = np.unique(superpoints[visible_points])
        for sp in visible_sp:
            visible_counts[sp_to_index[int(sp)]] += 1
        selected = _best_observation(
            visible_points, observations_by_frame_class.get((frame_index, int(class_id)), []),
        )
        row = {
            "frame_index": int(frame_index),
            "visible_track_point_count": int(len(visible_points)),
            "selected_observation_id": -1,
            "selected_coverage": 0.0,
            "selected_quality": 0.0,
            "counterevidence_eligible": False,
        }
        if selected is None:
            frame_rows.append(row)
            continue
        row.update({
            "selected_observation_id": int(selected["observation_id"]),
            "selected_coverage": float(selected["coverage"]),
            "selected_quality": float(selected["quality"]),
            "counterevidence_eligible": bool(selected["coverage"] >= min_counterevidence_anchor_coverage),
        })
        quality = float(selected["quality"])
        covered_set = selected["covered_points"]
        for sp in visible_sp:
            index = sp_to_index[int(sp)]
            sp_visible = visible_points[superpoints[visible_points] == sp]
            covered = np.intersect1d(sp_visible, covered_set, assume_unique=True)
            coverage = float(len(covered) / max(1, len(sp_visible)))
            positive_weights[index] += quality * coverage
            if row["counterevidence_eligible"]:
                negative_weights[index] += quality * (1.0 - coverage)
        frame_rows.append(row)
    total_visible = int(sum(row["visible_track_point_count"] > 0 for row in frame_rows))
    matched = [row for row in frame_rows if row["selected_observation_id"] >= 0]
    eligible = [row for row in matched if row["counterevidence_eligible"]]
    support = [row for row in matched if row["selected_coverage"] > 0]
    sp_margin = positive_weights - negative_weights
    return {
        "frame_rows": frame_rows,
        "superpoint_ids": track_superpoints.astype(np.int64),
        "visible_view_counts": visible_counts,
        "positive_weights": positive_weights,
        "negative_weights": negative_weights,
        "margins": sp_margin,
        "visible_frame_count": total_visible,
        "matched_class_observation_frame_count": len(matched),
        "positive_support_frame_count": len(support),
        "counterevidence_eligible_frame_count": len(eligible),
        "mean_selected_coverage": float(np.mean([row["selected_coverage"] for row in matched])) if matched else 0.0,
        "mean_selected_quality": float(np.mean([row["selected_quality"] for row in matched])) if matched else 0.0,
        "positive_superpoint_count": int(np.sum(positive_weights > 0)),
        "negative_margin_superpoint_count": int(np.sum(sp_margin < 0)),
        "nonnegative_margin_superpoint_count": int(np.sum(sp_margin >= 0)),
        "mean_superpoint_evidence_margin": float(np.mean(sp_margin)) if len(sp_margin) else 0.0,
    }


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    semantics = _load_track_semantics(
        args.semantic_root / scene_name / "automatic_track_yoloworld_semantics.json"
    )
    observations = _load_detection_observations(args.yoloworld_sam_root / scene_name)
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    masks = _load_prediction(args.prediction_cache_dir, scene_name)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, visibility = world.get_mesh_projections()
    visibility = visibility.detach().cpu().numpy().astype(bool, copy=False)
    evidence_dir = args.output_root / scene_name / "superpoint_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for track in tracks:
        track_id = int(track["track_id"])
        class_id = int(semantics.get(track_id, -1))
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(superpoints))]
        evidence = _track_evidence(
            points, class_id, superpoints, visibility, observations,
            args.min_visible_points, args.min_counterevidence_anchor_coverage,
        ) if class_id >= 0 else None
        path = evidence_dir / f"track{track_id:04d}_superpoint_evidence.npz"
        if evidence is None:
            np.savez_compressed(path, superpoint_ids=np.empty(0, dtype=np.int64))
            feature = {
                "visible_frame_count": 0,
                "matched_class_observation_frame_count": 0,
                "positive_support_frame_count": 0,
                "counterevidence_eligible_frame_count": 0,
                "mean_selected_coverage": 0.0,
                "mean_selected_quality": 0.0,
                "positive_superpoint_count": 0,
                "negative_margin_superpoint_count": 0,
                "nonnegative_margin_superpoint_count": 0,
                "mean_superpoint_evidence_margin": 0.0,
                "frame_evidence": [],
            }
        else:
            np.savez_compressed(
                path,
                superpoint_ids=evidence["superpoint_ids"],
                visible_view_counts=evidence["visible_view_counts"],
                positive_weights=evidence["positive_weights"],
                negative_weights=evidence["negative_weights"],
                evidence_margins=evidence["margins"],
            )
            feature = {key: value for key, value in evidence.items() if key not in {
                "superpoint_ids", "visible_view_counts", "positive_weights", "negative_weights", "margins", "frame_rows",
            }}
            feature["frame_evidence"] = evidence["frame_rows"]
        records.append({
            "scene_name": scene_name,
            "track_id": track_id,
            "voted_class_index": class_id,
            "track_point_count": int(len(points)),
            "track_support_view_count": int(track["support_view_count"]),
            "track_mean_node_quality": float(track["mean_node_quality"]),
            "track_mean_edge_score": float(track["mean_edge_score"]),
            "superpoint_evidence_path": str(path),
            **_candidate_relation(points, masks),
            **feature,
        })
    del world, visibility, processed, masks
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--yoloworld_sam_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--min_counterevidence_anchor_coverage", type=float, default=0.30)
    args = parser.parse_args()
    if not 0.0 <= args.min_counterevidence_anchor_coverage <= 1.0:
        raise SystemExit("--min_counterevidence_anchor_coverage 必须在零到一之间。")
    for name in (
        "scene_list", "track_root", "semantic_root", "yoloworld_sam_root", "prediction_cache_dir",
        "processed_scene_root", "dataset_root", "config_path", "output_root",
    ):
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
    for index, scene_name in enumerate(scenes, start=1):
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        records = _scene_records(scene_name, args)
        all_records.extend(records)
        (scene_root / "visibility_counterevidence_ledger.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条轨迹证据", flush=True)
    summary = {
        "gt_usage": "不读取 GT；不生成候选、不融合、不评分、不评测 AP。",
        "decision_state": "仅构建候选级正反可见性与 native 关系证据，尚未定义接受或拒绝规则。",
        "scene_count": len(scenes),
        "track_count": len(all_records),
        "with_semantic_class_count": sum(row["voted_class_index"] >= 0 for row in all_records),
        "with_positive_support_count": sum(row["positive_support_frame_count"] > 0 for row in all_records),
        "with_counterevidence_count": sum(row["counterevidence_eligible_frame_count"] > 0 for row in all_records),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "visibility_counterevidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

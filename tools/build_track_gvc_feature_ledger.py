#!/usr/bin/env python3
"""为自动多视角轨迹建立 GVC 启发的候选级二维--三维一致性账本。

对每条既有自动 SAM 轨迹，本工具仅在冻结 YOLO-World+SAM 已导出的均匀视角中：
投影深度可见三维点、匹配同类别二维框、计算可见点被对应二维 mask 支持的比例，
再按可见点数选择最多 K 个视角汇总。与 native 候选的关系直接从冻结预测缓存
计算；CER/PES 为可选的历史辅助特征，本轮 even96 安全验证不依赖它们。

不读取 GT，不修改轨迹、superpoint 或 native 候选，不形成候选、不做 NMS 或 AP。
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


def _box_iou(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    inter = width * height
    union = max(1e-8, (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - inter)
    return float(inter / union)


def _load_observations(scene_root):
    by_frame_class = defaultdict(list)
    path = scene_root / "observations.jsonl"
    if not path.is_file():
        return by_frame_class
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            points = np.unique(np.asarray(np.load(row["point_indices_path"])["point_indices"], dtype=np.int64))
            by_frame_class[(int(row["frame_index"]), int(row["class_id"]))].append({
                "observation_id": int(row["observation_id"]),
                "bbox": np.asarray(row["bbox_xyxy"], dtype=np.float64),
                "points": points,
            })
    return by_frame_class


def _load_selected_frames(scene_root):
    """读取自动 SAM 导出时固定的输入帧；没有同类 2D 观测的帧也必须保留。"""
    payload = json.loads((scene_root / "summary.json").read_text())
    return sorted(int(row["frame_index"]) for row in payload["frames"])


def _load_scores(scene_root):
    by_track = defaultdict(list)
    path = scene_root / "counterevidence_reliability_scores.jsonl"
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_track[int(row["track_id"])].append(row)
    return by_track


def _load_native_masks(root, scene_name, point_count):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 native mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的 native mask 点数不匹配")
    return np.asarray(masks, dtype=bool)


def _native_relation(points, masks):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if len(points) == 0 or masks.shape[1] == 0:
        return {
            "native_top_candidate_id": -1, "native_top_iou": 0.0,
            "track_inside_top_native_ratio": 0.0, "native_overlap_count": 0,
        }
    sizes = masks.sum(axis=0, dtype=np.int64)
    intersection = masks[points].sum(axis=0, dtype=np.int64)
    iou = intersection / np.maximum(1, len(points) + sizes - intersection)
    top = int(np.argmax(iou))
    return {
        "native_top_candidate_id": top,
        "native_top_iou": float(iou[top]),
        "track_inside_top_native_ratio": float(intersection[top] / len(points)),
        "native_overlap_count": int(np.sum(iou >= 0.10)),
    }


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "p90": 0.0}
    return {"mean": float(values.mean()), "std": float(values.std()), "p90": float(np.quantile(values, 0.90))}


def gvc_from_views(view_rows, max_views):
    """固定公式：每帧为 projected-box IoU 与 mask 点支持率之积。"""
    selected = sorted(view_rows, key=lambda row: (-row["visible_point_count"], row["frame_index"]))[:max_views]
    gvc = [row["gvc_frame_score"] for row in selected]
    boxes = [row["box_iou"] for row in selected]
    masks = [row["mask_point_support"] for row in selected]
    return {
        "gvc_eligible_view_count": len(view_rows),
        "gvc_selected_view_count": len(selected),
        "gvc_matched_view_count": sum(row["matched_observation_id"] >= 0 for row in selected),
        "gvc_selected_match_ratio": float(sum(row["matched_observation_id"] >= 0 for row in selected) / max(1, len(selected))),
        "gvc_score": float(np.mean(gvc)) if gvc else 0.0,
        "gvc_box_iou": _summary(boxes),
        "gvc_mask_point_support": _summary(masks),
        "gvc_frame_score": _summary(gvc),
        "gvc_selected_frames": selected,
    }


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    semantics = {int(row["track_id"]): row for row in json.loads(
        (args.semantic_root / scene_name / "automatic_track_yoloworld_semantics.json").read_text()
    )}
    evidence = {}
    if args.evidence_root is not None:
        evidence = {int(row["track_id"]): row for row in json.loads(
            (args.evidence_root / scene_name / "visibility_counterevidence_ledger.json").read_text()
        )}
    observations = _load_observations(args.yoloworld_sam_root / scene_name)
    selected_frames = _load_selected_frames(args.automatic_root / scene_name)
    scores = _load_scores(args.score_root / scene_name) if args.score_root is not None else {}
    processed = np.load(
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r"
    )
    native_masks = _load_native_masks(args.native_prediction_cache, scene_name, len(processed))
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.float64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    scaling = (world.depth_resolution[0] / world.image_resolution[0], world.depth_resolution[1] / world.image_resolution[1])
    records = []
    for track in tracks:
        track_id = int(track["track_id"])
        semantic = semantics.get(track_id, {})
        class_id = int(semantic.get("voted_class_index", -1))
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < visibility.shape[1])]
        view_rows = []
        if class_id >= 0:
            for frame_index in selected_frames:
                visible_points = points[visibility[frame_index, points]]
                if len(visible_points) < args.min_visible_points:
                    continue
                coords = projections[frame_index, visible_points]
                xs, ys = coords[:, 0] / scaling[1], coords[:, 1] / scaling[0]
                projected_box = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
                candidates = observations[(frame_index, class_id)]
                if not candidates:
                    view_rows.append({
                        "frame_index": int(frame_index), "visible_point_count": int(len(visible_points)),
                        "matched_observation_id": -1, "box_iou": 0.0,
                        "mask_point_support": 0.0, "gvc_frame_score": 0.0,
                    })
                    continue
                selected = max(candidates, key=lambda item: (_box_iou(projected_box, item["bbox"]), -item["observation_id"]))
                box_iou = _box_iou(projected_box, selected["bbox"])
                support = float(len(np.intersect1d(visible_points, selected["points"], assume_unique=True)) / len(visible_points))
                view_rows.append({
                    "frame_index": int(frame_index), "visible_point_count": int(len(visible_points)),
                    "matched_observation_id": int(selected["observation_id"]), "box_iou": box_iou,
                    "mask_point_support": support, "gvc_frame_score": float(box_iou * support),
                })
        gvc = gvc_from_views(view_rows, args.max_views)
        track_scores = scores.get(track_id, [])
        cer = _summary([row["counterevidence_reliability_score"] for row in track_scores])
        pes = _summary([row["positiveevidence_reliability_score"] for row in track_scores])
        relation = _native_relation(points, native_masks)
        records.append({
            "scene_name": scene_name, "track_id": track_id, "candidate_source": "automatic_sam_track",
            "track_point_count": int(len(points)), "voted_class_index": class_id,
            "yoloworld_vote_margin": float(semantic.get("vote_margin", 0.0)),
            "track_support_view_count": int(track["support_view_count"]),
            "track_mean_node_quality": float(track["mean_node_quality"]),
            "native_top_candidate_id": int(relation["native_top_candidate_id"]),
            "native_top_iou": float(relation["native_top_iou"]),
            "track_inside_top_native_ratio": float(relation["track_inside_top_native_ratio"]),
            "native_overlap_count": int(relation["native_overlap_count"]),
            "counterevidence_available": bool(track_scores),
            "cer": cer, "pes": pes, **gvc,
        })
    del world, projections, visibility, processed, native_masks
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--native_prediction_cache", type=Path, required=True)
    parser.add_argument("--evidence_root", type=Path, default=None, help="可选历史可见性账本；本轮不需要。")
    parser.add_argument("--score_root", type=Path, default=None, help="可选历史 CER/PES 账本；本轮不需要。")
    parser.add_argument("--yoloworld_sam_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_views", type=int, default=10)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--resume", action="store_true", help="仅跳过完整场景，用于执行通道中断后的安全续跑。")
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "automatic_root", "semantic_root", "native_prediction_cache", "yoloworld_sam_root", "dataset_root", "processed_scene_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    for name in ("evidence_root", "score_root"):
        value = getattr(args, name)
        setattr(args, name, _resolve(value) if value is not None else None)
    if args.max_views <= 0 or args.min_visible_points <= 0:
        raise SystemExit("--max_views 与 --min_visible_points 必须为正数。")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    all_rows = []
    for index, scene_name in enumerate(scenes, start=1):
        existing_path = args.output_root / scene_name / "track_gvc_feature_ledger.json"
        if existing_path.is_file():
            if not args.resume:
                raise SystemExit(f"输出场景已存在：{scene_name}")
            rows = json.loads(existing_path.read_text())
            all_rows.extend(rows)
            print(f"[跳过已有] {index}/{len(scenes)} {scene_name}: {len(rows)} 条轨迹特征", flush=True)
            continue
        rows = _scene_records(scene_name, args)
        all_rows.extend(rows)
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        (scene_root / "track_gvc_feature_ledger.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(rows)} 条轨迹特征", flush=True)
    payload = {
        "gt_usage": "不读取 GT；不生成候选、不修改 native 候选、不做 NMS 或 AP。",
        "decision_state": "仅固定 GVC 启发的候选级二维--三维一致性与 CER/PES 汇总特征，尚未定义接受规则。",
        "scene_count": len(scenes), "track_count": len(all_rows),
        "with_gvc_match_count": sum(row["gvc_matched_view_count"] > 0 for row in all_rows),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "track_gvc_feature_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

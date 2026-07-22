#!/usr/bin/env python3
"""仅离线真值诊断：检查 YOLO-World 是否看见 Mask3D 漏检的实例。

对每个 Mask3D 未覆盖的 GT 实例，统计同类别二维检测框对其可见三维点的
覆盖率，而不是使用二维框 IoU。结果只用于决定是否值得比较新的二维检测器，
绝不参与推理、候选生成、融合、打分或阈值选择。
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid_classes:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) < min_region_size:
            continue
        instances.append(
            {
                "gt_instance_id": instance_id,
                "gt_class": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
                "indices": indices,
                "point_count": int(len(indices)),
            }
        )
    return gt_ids, instances


def _load_masks(root, scene_name, num_points):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = _as_numpy(masks).astype(bool, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的基础 mask 维度异常：{masks.shape}")
    if masks.shape[0] != num_points and masks.shape[1] == num_points:
        masks = masks.T
    if masks.shape[0] != num_points:
        raise ValueError(f"{scene_name} 的点数不一致：{masks.shape[0]} 与 {num_points}")
    return masks


def _best_mask_iou(masks, mask_sizes, indices, point_count):
    if masks.shape[1] == 0:
        return 0.0
    intersections = masks[indices].sum(axis=0, dtype=np.int64)
    return float(np.max(intersections / np.maximum(1, mask_sizes + point_count - intersections)))


def _load_predictions(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        raise ValueError(f"二维缓存格式异常：{path}")
    return payload


def _frame_coverage(coords_depth, boxes, labels, scores, class_id, scaling_params):
    eligible = labels == class_id
    if not np.any(eligible):
        return 0.0, 0.0, 0
    boxes = boxes[eligible]
    scores = scores[eligible]
    xs = coords_depth[:, 0].astype(np.float32) / float(scaling_params[1])
    ys = coords_depth[:, 1].astype(np.float32) / float(scaling_params[0])
    inside = (
        (xs[:, None] >= boxes[None, :, 0])
        & (xs[:, None] <= boxes[None, :, 2])
        & (ys[:, None] >= boxes[None, :, 1])
        & (ys[:, None] <= boxes[None, :, 3])
    )
    coverages = inside.mean(axis=0)
    best = int(np.argmax(coverages))
    return float(coverages[best]), float(scores[best]), int(len(boxes))


def _diagnose_scene(scene_name, args, prompts, prompt_to_id):
    # 延迟导入可避免纯参数检查时加载二维检测依赖。
    from utils import WORLD_2_CAM

    gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    predictions = _load_predictions(args.bboxes_2d_root / f"{scene_name}.pt")
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = _as_numpy(projections)
    visibility = _as_numpy(visibility).astype(bool, copy=False)
    frame_count = min(int(args.max_frames), len(world.color_paths), projections.shape[0])
    scaling_params = [
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    ]
    rows = []
    for instance in instances:
        class_id = prompt_to_id.get(instance["gt_class"])
        frame_stats = []
        if class_id is not None:
            for frame_index in range(frame_count):
                visible_indices = instance["indices"][visibility[frame_index, instance["indices"]]]
                if len(visible_indices) < args.min_visible_points:
                    continue
                frame_id = Path(world.color_paths[frame_index]).stem
                prediction = predictions.get(frame_id)
                if prediction is None:
                    continue
                coverage, score, box_count = _frame_coverage(
                    projections[frame_index, visible_indices],
                    _as_numpy(prediction["bbox"]).astype(np.float32, copy=False),
                    _as_numpy(prediction["labels"]).astype(np.int64, copy=False),
                    _as_numpy(prediction["scores"]).astype(np.float32, copy=False),
                    class_id,
                    scaling_params,
                )
                frame_stats.append((coverage, score, box_count))
        visible_frames = len(frame_stats)
        matched = [item for item in frame_stats if item[0] >= args.min_box_point_coverage]
        baseline_iou = _best_mask_iou(masks, mask_sizes, instance["indices"], instance["point_count"])
        for threshold, suffix in ((0.25, "iou25"), (0.50, "iou50")):
            if baseline_iou >= threshold:
                continue
            rows.append(
                {
                    "scene_name": scene_name,
                    "gt_instance_id": instance["gt_instance_id"],
                    "gt_class": instance["gt_class"],
                    "gt_point_count": instance["point_count"],
                    "mask3d_best_iou": baseline_iou,
                    "threshold": threshold,
                    "prompt_class_id": class_id if class_id is not None else -1,
                    "class_in_yoloworld_prompts": class_id is not None,
                    "usable_visible_frame_count": visible_frames,
                    "box_covered_frame_count": len(matched),
                    "best_box_point_coverage": max((item[0] for item in frame_stats), default=0.0),
                    "mean_box_point_coverage": float(np.mean([item[0] for item in frame_stats])) if frame_stats else 0.0,
                    "best_box_score": max((item[1] for item in frame_stats), default=0.0),
                    "same_class_box_count": int(sum(item[2] for item in frame_stats)),
                    "reliable_2d_box": bool(len(matched) >= args.min_matched_frames),
                }
            )
    del world, projections, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--bboxes_2d_root", type=Path, default=Path("output/scannet200/bboxes_2d"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--min_box_point_coverage", type=float, default=0.50)
    parser.add_argument("--min_matched_frames", type=int, default=2)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断。")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    prompts = [str(item) for item in args.config["network2d"]["text_prompts"]]
    prompt_to_id = {name: index for index, name in enumerate(prompts)}
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    rows = []
    for scene_name in scenes:
        scene_rows = _diagnose_scene(scene_name, args, prompts, prompt_to_id)
        rows.extend(scene_rows)
        print(f"[场景完成] {scene_name}: 新增 {len(scene_rows)} 条漏检实例记录", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene_name", "gt_instance_id", "gt_class", "gt_point_count", "mask3d_best_iou", "threshold",
        "prompt_class_id", "class_in_yoloworld_prompts", "usable_visible_frame_count", "box_covered_frame_count",
        "best_box_point_coverage", "mean_box_point_coverage", "best_box_score", "same_class_box_count", "reliable_2d_box",
    ]
    with (args.output_dir / "mask3d_missed_yoloworld_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    thresholds = {}
    for threshold in (0.25, 0.50):
        group = [row for row in rows if row["threshold"] == threshold]
        reliable = [row for row in group if row["reliable_2d_box"]]
        absent = [row for row in group if not row["class_in_yoloworld_prompts"]]
        thresholds[f"iou{int(threshold * 100)}"] = {
            "mask3d_missed_gt_count": len(group),
            "reliable_2d_box_count": len(reliable),
            "reliable_2d_box_rate": float(len(reliable) / max(1, len(group))),
            "class_absent_from_yoloworld_prompts_count": len(absent),
            "class_absent_from_yoloworld_prompts_rate": float(len(absent) / max(1, len(group))),
            "class_counts": dict(sorted(Counter(row["gt_class"] for row in group).items())),
        }
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或阈值选择。",
        "decision_rule": "若 Mask3D 漏检实例大多没有可靠二维框，才值得比较新的二维检测器；已有充分框覆盖则优先研究三维实例形成。",
        "scene_count": len(scenes),
        "max_frames": args.max_frames,
        "min_visible_points": args.min_visible_points,
        "min_box_point_coverage": args.min_box_point_coverage,
        "min_matched_frames": args.min_matched_frames,
        "thresholds": thresholds,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

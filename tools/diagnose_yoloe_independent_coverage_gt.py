#!/usr/bin/env python3
"""GT-only：比较 YOLOE 与冻结 YOLO-World 对 native 强基线残差的二维覆盖。

GT 仅用于事后账本。YOLOE 输出、比较结果和任意阈值均不得回流到推理、候选
生成、融合、打分或 AP 评测。所有二维条件固定为同一 f30 帧、同一类别和同一
至少两帧可见点覆盖过半规则。
"""

import argparse
import csv
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid:
            continue
        indices = np.flatnonzero(ids == instance_id).astype(np.int32)
        if len(indices) >= min_region_size:
            instances.append({
                "instance_id": instance_id,
                "class_name": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
                "indices": indices,
                "point_count": int(len(indices)),
            })
    return ids, instances


def _load_native_masks(root, scene_name, point_count):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 native 预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的 native 预测点数不一致")
    # 保持内存映射，避免 even48 逐场景诊断时复制数百 MB 的预测数组。
    return masks


def _best_iou(masks, mask_sizes, indices, point_count):
    if masks.shape[1] == 0:
        return 0.0
    intersection = masks[indices].sum(axis=0, dtype=np.int64)
    return float(np.max(intersection / np.maximum(1, mask_sizes + point_count - intersection)))


def _load_yoloworld(root, scene_name):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    predictions = payload.get("predictions", payload) if isinstance(payload, dict) else payload
    if not isinstance(predictions, dict):
        raise ValueError(f"{scene_name} 的 YOLO-World 缓存格式异常")
    result = {}
    for frame_id, item in predictions.items():
        result[str(frame_id)] = {
            "boxes_xyxy": _as_numpy(item["bbox"]).astype(np.float32, copy=False),
            "labels": _as_numpy(item["labels"]).astype(np.int64, copy=False),
            "scores": _as_numpy(item["scores"]).astype(np.float32, copy=False),
        }
    return result


def _load_yoloe(root, scene_name):
    payload = json.loads((root / scene_name / "yoloe_boxes.json").read_text())
    result = {}
    for item in payload["frames"]:
        result[str(item["frame_id"])] = {
            "boxes_xyxy": np.asarray(item["boxes_xyxy"], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray(item["labels"], dtype=np.int64),
            "scores": np.asarray(item["scores"], dtype=np.float32),
        }
    return result


def _frame_coverage(coords, prediction, class_id, scaling):
    labels = prediction["labels"]
    eligible = labels == class_id
    if not np.any(eligible):
        return 0.0, 0.0, 0
    boxes = prediction["boxes_xyxy"][eligible]
    scores = prediction["scores"][eligible]
    xs = coords[:, 0].astype(np.float32) / float(scaling[1])
    ys = coords[:, 1].astype(np.float32) / float(scaling[0])
    inside = (
        (xs[:, None] >= boxes[None, :, 0])
        & (xs[:, None] <= boxes[None, :, 2])
        & (ys[:, None] >= boxes[None, :, 1])
        & (ys[:, None] <= boxes[None, :, 3])
    )
    coverage = inside.mean(axis=0)
    best = int(np.argmax(coverage))
    return float(coverage[best]), float(scores[best]), int(len(boxes))


def _source_stats(instance, world, projections, visibility, scaling, predictions, class_id, args):
    rows = []
    frame_count = min(args.max_frames, len(world.color_paths), projections.shape[0])
    for frame_index in range(frame_count):
        visible = instance["indices"][visibility[frame_index, instance["indices"]]]
        if len(visible) < args.min_visible_points:
            continue
        frame_id = Path(world.color_paths[frame_index]).stem
        prediction = predictions.get(frame_id)
        if prediction is None:
            continue
        coverage, score, count = _frame_coverage(projections[frame_index, visible], prediction, class_id, scaling)
        rows.append((coverage, score, count))
    matched = [item for item in rows if item[0] >= args.min_box_point_coverage]
    return {
        "usable_frames": len(rows),
        "covered_frames": len(matched),
        "best_coverage": max((item[0] for item in rows), default=0.0),
        "best_score": max((item[1] for item in rows), default=0.0),
        "same_class_box_count": int(sum(item[2] for item in rows)),
        "reliable": len(matched) >= args.min_matched_frames,
    }


def _comparison_label(yoloworld, yoloe):
    if yoloe["reliable"] and not yoloworld["reliable"]:
        return "仅 YOLOE 有可靠二维证据"
    if yoloworld["reliable"] and not yoloe["reliable"]:
        return "仅 YOLO-World 有可靠二维证据"
    if yoloe["reliable"]:
        return "两者均有可靠二维证据"
    return "两者均无可靠二维证据"


def _scene_rows(scene_name, args, prompt_to_id):
    from utils import WORLD_2_CAM

    gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks = _load_native_masks(args.prediction_cache_dir, scene_name, len(gt_ids))
    sizes = masks.sum(axis=0, dtype=np.int64)
    yolo_world = _load_yoloworld(args.yoloworld_root, scene_name)
    yoloe = _load_yoloe(args.yoloe_root, scene_name)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = _as_numpy(projections)
    visibility = _as_numpy(visibility).astype(bool, copy=False)
    scaling = (world.depth_resolution[0] / world.image_resolution[0], world.depth_resolution[1] / world.image_resolution[1])
    rows = []
    for instance in instances:
        native_iou = _best_iou(masks, sizes, instance["indices"], instance["point_count"])
        if native_iou >= args.native_iou_threshold:
            continue
        class_id = prompt_to_id.get(instance["class_name"])
        if class_id is None:
            continue
        world_stats = _source_stats(instance, world, projections, visibility, scaling, yolo_world, class_id, args)
        yoloe_stats = _source_stats(instance, world, projections, visibility, scaling, yoloe, class_id, args)
        rows.append({
            "scene_name": scene_name,
            "gt_instance_id": instance["instance_id"],
            "gt_class": instance["class_name"],
            "gt_point_count": instance["point_count"],
            "native_best_iou": native_iou,
            "yoloworld_usable_frame_count": world_stats["usable_frames"],
            "yoloworld_covered_frame_count": world_stats["covered_frames"],
            "yoloworld_best_box_point_coverage": world_stats["best_coverage"],
            "yoloworld_best_box_score": world_stats["best_score"],
            "yoloworld_same_class_box_count": world_stats["same_class_box_count"],
            "yoloworld_reliable": world_stats["reliable"],
            "yoloe_usable_frame_count": yoloe_stats["usable_frames"],
            "yoloe_covered_frame_count": yoloe_stats["covered_frames"],
            "yoloe_best_box_point_coverage": yoloe_stats["best_coverage"],
            "yoloe_best_box_score": yoloe_stats["best_score"],
            "yoloe_same_class_box_count": yoloe_stats["same_class_box_count"],
            "yoloe_reliable": yoloe_stats["reliable"],
            "comparison": _comparison_label(world_stats, yoloe_stats),
        })
    del masks, sizes, yolo_world, yoloe, world, projections, visibility
    gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--yoloe_root", type=Path, required=True)
    parser.add_argument("--yoloworld_root", type=Path, default=Path("output/scannet200/bboxes_2d"))
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--scene_offset", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--native_iou_threshold", type=float, default=0.50)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--min_box_point_coverage", type=float, default=0.50)
    parser.add_argument("--min_matched_frames", type=int, default=2)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in ("scene_list", "yoloe_root", "yoloworld_root", "prediction_cache_dir", "dataset_root", "gt_instance_dir", "config_path", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    prompt_to_id = {str(name): index for index, name in enumerate(args.config["network2d"]["text_prompts"])}
    scenes = _read_scenes(args.scene_list)
    scenes = scenes[max(0, args.scene_offset):]
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    rows = []
    for scene_name in scenes:
        scene_rows = _scene_rows(scene_name, args, prompt_to_id)
        rows.extend(scene_rows)
        print(f"[场景完成] {scene_name}: {len(scene_rows)} 条 native 残差", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "gt_instance_id", "gt_class"]
    with (args.output_dir / "yoloe_independent_coverage_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter(row["comparison"] for row in rows)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或 AP 评测。",
        "decision_rule": "仅当‘仅 YOLOE 有可靠二维证据’数量明确增加时，才考虑将 YOLOE 作为并行候选源；否则保持 YOLO-World 强基线不变。",
        "native_iou_threshold": args.native_iou_threshold,
        "scene_count": len(scenes),
        "scenes": scenes,
        "native_residual_instance_count": len(rows),
        "comparison_counts": dict(sorted(groups.items())),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

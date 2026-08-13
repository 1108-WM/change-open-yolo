#!/usr/bin/env python3
"""仅离线 GT 账本：评估锚点引导扩展区域的覆盖收益与误吸收。"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(ids):
        instance_id = int(instance_id)
        if instance_id > 0 and instance_id // 1000 in valid:
            size = int(np.sum(ids == instance_id))
            if size >= min_region_size:
                sizes[instance_id] = size
    return ids, sizes


def _load_prediction(root, scene_name, point_count):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的最终预测点数不匹配")
    return np.asarray(masks, dtype=bool)


def _best_gt(points, gt_ids, gt_sizes):
    points = np.unique(np.asarray(points, dtype=np.int64))
    values, counts = np.unique(gt_ids[points], return_counts=True) if len(points) else ([], [])
    candidates = []
    for instance_id, intersection in zip(values, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        size = gt_sizes[instance_id]
        intersection = int(intersection)
        candidates.append({
            "instance_id": instance_id,
            "iou": float(intersection / max(1, len(points) + size - intersection)),
            "precision": float(intersection / max(1, len(points))),
            "coverage": float(intersection / max(1, size)),
        })
    return max(candidates, key=lambda item: item["iou"], default={"instance_id": -1, "iou": 0.0, "precision": 0.0, "coverage": 0.0})


def _against_instance(points, instance_id, gt_ids, gt_sizes):
    if instance_id not in gt_sizes:
        return {"iou": 0.0, "precision": 0.0, "coverage": 0.0}
    points = np.unique(np.asarray(points, dtype=np.int64))
    intersection = int(np.sum(gt_ids[points] == instance_id))
    size = gt_sizes[instance_id]
    return {
        "iou": float(intersection / max(1, len(points) + size - intersection)),
        "precision": float(intersection / max(1, len(points))),
        "coverage": float(intersection / max(1, size)),
    }


def _final_iou_by_gt(masks, gt_ids, gt_sizes):
    sizes = masks.sum(axis=0, dtype=np.int64)
    result = {}
    for instance_id, gt_size in gt_sizes.items():
        indices = np.flatnonzero(gt_ids == instance_id)
        intersections = masks[indices].sum(axis=0, dtype=np.int64) if masks.shape[1] else np.asarray([])
        result[instance_id] = float(np.max(intersections / np.maximum(1, sizes + gt_size - intersections))) if len(intersections) else 0.0
    return result


def _residual_type(iou):
    if iou < 0.25:
        return "无合格三维候选"
    if iou < 0.50:
        return "边界不足"
    return "已有严格三维候选"


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    final_iou = _final_iou_by_gt(_load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids)), gt_ids, gt_sizes)
    records = json.loads((args.expansion_root / scene_name / "anchor_guided_expansions.json").read_text())
    rows = []
    for record in records:
        anchor = np.asarray(np.load(record["anchor_points_path"])["point_indices"], dtype=np.int64)
        expanded = np.asarray(np.load(record["expanded_points_path"])["point_indices"], dtype=np.int64)
        anchor_match = _best_gt(anchor, gt_ids, gt_sizes)
        instance_id = int(anchor_match["instance_id"])
        expanded_same = _against_instance(expanded, instance_id, gt_ids, gt_sizes)
        expanded_match = _best_gt(expanded, gt_ids, gt_sizes)
        native_iou = float(final_iou.get(instance_id, 0.0)) if instance_id > 0 else 0.0
        rows.append({
            "scene_name": scene_name,
            "track_id": int(record["track_id"]),
            "selected_frame_count": int(record["selected_frame_count"]),
            "added_point_count": int(record["added_point_count"]),
            "meets_minimum_support": bool(record["meets_minimum_support"]),
            "anchor_gt_instance_id": instance_id,
            "anchor_gt_class": ID_TO_LABEL.get(instance_id // 1000, "无有效实例") if instance_id > 0 else "无有效实例",
            "anchor_iou": float(anchor_match["iou"]),
            "anchor_precision": float(anchor_match["precision"]),
            "anchor_coverage": float(anchor_match["coverage"]),
            "expanded_same_instance_iou": float(expanded_same["iou"]),
            "expanded_same_instance_precision": float(expanded_same["precision"]),
            "expanded_same_instance_coverage": float(expanded_same["coverage"]),
            "expanded_best_gt_instance_id": int(expanded_match["instance_id"]),
            "expanded_best_iou": float(expanded_match["iou"]),
            "same_instance_iou_change": float(expanded_same["iou"] - anchor_match["iou"]),
            "same_instance_precision_change": float(expanded_same["precision"] - anchor_match["precision"]),
            "native_matched_gt_iou": native_iou,
            "matched_gt_residual_type": _residual_type(native_iou) if instance_id > 0 else "无有效 GT",
        })
    return rows


def _summary(rows):
    supported = [row for row in rows if row["meets_minimum_support"]]
    target = [row for row in supported if row["matched_gt_residual_type"] in {"无合格三维候选", "边界不足"}]
    def instances(items, field):
        return {(row["scene_name"], row["anchor_gt_instance_id"]) for row in items if row["anchor_gt_instance_id"] > 0 and row[field] >= 0.25}
    original = instances(target, "anchor_iou")
    expanded = instances(target, "expanded_same_instance_iou")
    return {
        "至少两帧支持的扩展区域数": len(supported),
        "同一 GT 实例 IoU 提高的扩展区域数": sum(row["same_instance_iou_change"] > 0.0 for row in supported),
        "同一 GT 实例点精度降低的扩展区域数": sum(row["same_instance_precision_change"] < 0.0 for row in supported),
        "目标残差几何合格独立实例数": {"扩展前": len(original), "扩展后": len(expanded), "新增": len(expanded - original)},
        "目标残差中由不足 IoU 25% 提升到不低于 25% 的扩展区域数": sum(
            row["anchor_iou"] < 0.25 <= row["expanded_same_instance_iou"] for row in target
        ),
        "说明": "GT 仅用于离线比较扩展前后，不参与锚点、帧级 mask 选择、扩展、阈值、候选或评分。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--expansion_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线扩展区域评估。")
    for name in ("scene_list", "expansion_root", "prediction_cache_dir", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条扩展区域", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "anchor_guided_track_expansions_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线比较固定的无 GT 锚点扩展，不反向修改规则。",
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

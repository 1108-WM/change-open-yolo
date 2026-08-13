#!/usr/bin/env python3
"""仅离线 GT 账本：评估自动 mask 几何轨迹的冻结 YOLO-World 语义。"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id > 0 and instance_id // 1000 in valid:
            size = int(np.sum(gt_ids == instance_id))
            if size >= min_region_size:
                sizes[instance_id] = size
    return gt_ids, sizes


def _load_prediction(root, scene_name, point_count):
    masks = np.load(f"{root / scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的最终预测点数不匹配")
    return np.asarray(masks, dtype=bool)


def _final_iou_by_gt(masks, gt_ids, gt_sizes):
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    values = {}
    for instance_id, size in gt_sizes.items():
        indices = np.flatnonzero(gt_ids == instance_id)
        intersection = masks[indices].sum(axis=0, dtype=np.int64) if masks.shape[1] else np.asarray([])
        values[instance_id] = float(np.max(intersection / np.maximum(1, mask_sizes + size - intersection))) if len(intersection) else 0.0
    return values


def _best_gt(points, gt_ids, gt_sizes):
    points = np.unique(np.asarray(points, dtype=np.int64))
    ids, counts = np.unique(gt_ids[points], return_counts=True) if len(points) else ([], [])
    candidates = []
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        size = gt_sizes[instance_id]
        intersection = int(intersection)
        candidates.append((instance_id, float(intersection / max(1, len(points) + size - intersection))))
    return max(candidates, key=lambda item: item[1], default=(-1, 0.0))


def _residual_type(iou):
    if iou < 0.25:
        return "无合格三维候选"
    if iou < 0.50:
        return "边界不足"
    return "已有严格三维候选"


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    final_iou = _final_iou_by_gt(_load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids)), gt_ids, gt_sizes)
    records = json.loads((args.semantic_root / scene_name / "automatic_track_yoloworld_semantics.json").read_text())
    rows = []
    for record in records:
        points = np.asarray(np.load(record["points_path"])["point_indices"], dtype=np.int64)
        instance_id, geometric_iou = _best_gt(points, gt_ids, gt_sizes)
        predicted_class_id = int(PRED_ID_TO_ID.get(int(record["voted_class_index"]), -1))
        gt_class_id = instance_id // 1000 if instance_id > 0 else -1
        baseline_iou = float(final_iou.get(instance_id, 0.0)) if instance_id > 0 else 0.0
        rows.append({
            "scene_name": scene_name,
            "track_id": int(record["track_id"]),
            "support_view_count": int(record["support_view_count"]),
            "point_count": int(record["point_count"]),
            "best_gt_instance_id": int(instance_id),
            "best_gt_iou": float(geometric_iou),
            "best_gt_class": ID_TO_LABEL.get(gt_class_id, "无有效实例"),
            "voted_class": ID_TO_LABEL.get(predicted_class_id, "无语义票"),
            "voted_class_index": int(record["voted_class_index"]),
            "voted_class_id": predicted_class_id,
            "semantic_correct": bool(instance_id > 0 and predicted_class_id == gt_class_id),
            "semantic_frame_count": int(record["semantic_frame_count"]),
            "voted_class_support_views": int(record["voted_class_support_views"]),
            "top_vote": float(record["top_vote"]),
            "vote_margin": float(record["vote_margin"]),
            "matched_gt_final_iou": baseline_iou,
            "matched_gt_residual_type": _residual_type(baseline_iou) if instance_id > 0 else "无有效 GT",
        })
    return rows


def _summary(rows):
    geometric = [row for row in rows if row["best_gt_iou"] >= 0.25]
    strict = [row for row in rows if row["best_gt_iou"] >= 0.50]
    def accuracy(items):
        return float(sum(row["semantic_correct"] for row in items) / max(1, len(items)))
    recovered = defaultdict(set)
    for row in geometric:
        if row["semantic_correct"] and row["best_gt_instance_id"] > 0:
            recovered[row["matched_gt_residual_type"]].add((row["scene_name"], row["best_gt_instance_id"]))
    by_margin = {}
    for name, predicate in {"类别间隔至少 50%": lambda row: row["vote_margin"] >= 0.50, "类别间隔不足 50%": lambda row: row["vote_margin"] < 0.50}.items():
        subset = [row for row in geometric if predicate(row)]
        by_margin[name] = {"几何合格轨迹数": len(subset), "语义正确率": accuracy(subset)}
    return {
        "全部类别无关轨迹数": len(rows),
        "几何 IoU 不低于 25% 的轨迹数": len(geometric),
        "几何 IoU 不低于 25% 的语义正确率": accuracy(geometric),
        "几何 IoU 不低于 50% 的轨迹数": len(strict),
        "几何 IoU 不低于 50% 的语义正确率": accuracy(strict),
        "几何 IoU 不低于 25% 且语义正确的独立实例数": {key: len(value) for key, value in sorted(recovered.items())},
        "按类别间隔分组": by_margin,
        "说明": "GT 仅用于离线语义归因。类别间隔可作为后续无 GT 质量特征，但本账本不能用于选择阈值。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--semantic_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线语义归因。")
    for name in ("scene_list", "semantic_root", "prediction_cache_dir", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条语义轨迹", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "automatic_track_yoloworld_semantics_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线语义归因，绝不进入自动 mask 导出、关联、语义投票、候选、融合、评分或阈值。",
        "scene_count": len(scenes),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

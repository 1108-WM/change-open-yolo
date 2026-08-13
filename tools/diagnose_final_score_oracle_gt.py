#!/usr/bin/env python3
"""仅离线 GT 诊断：测量固定 mask 与类别下的理想质量排序上限。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import evaluate_scannet200
from evaluate.scannet200.eval_semantic_instance import PRED_ID_TO_ID
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(ids):
        instance_id = int(instance_id)
        class_id = instance_id // 1000
        if instance_id <= 0 or class_id not in valid:
            continue
        indices = np.flatnonzero(ids == instance_id).astype(np.int32)
        if len(indices) >= min_region_size:
            instances.append((class_id, indices))
    return ids, instances


def _load_prediction(root, scene_name, point_count):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    scores = np.load(f"{prefix}_pred_scores.npy", mmap_mode="r")
    classes = np.load(f"{prefix}_pred_classes.npy", mmap_mode="r")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.ndim != 2 or masks.shape[0] != point_count or masks.shape[1] != len(classes):
        raise ValueError(f"{scene_name} 的预测缓存形状异常")
    semantic = np.asarray([int(PRED_ID_TO_ID.get(int(value), -1)) for value in classes], dtype=np.int64)
    return masks, np.asarray(scores, dtype=np.float32), np.asarray(classes, dtype=np.int64), semantic


def _metric_summary(path):
    result = {"ap": [], "ap50": [], "ap25": []}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            for name in result:
                value = float(row[name])
                if not np.isnan(value):
                    result[name].append(value)
    return {name: float(np.mean(values)) if values else float("nan") for name, values in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in ("scene_list", "prediction_cache_dir", "output_dir", "gt_instance_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_predictions = {}
    oracle_predictions = {}
    positive_quality_predictions = 0
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks, scores, class_indices, semantic_classes = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
        mask_sizes = np.asarray(masks.sum(axis=0), dtype=np.int64)
        oracle_scores = np.zeros(masks.shape[1], dtype=np.float32)
        for class_id, indices in instances:
            selected = np.flatnonzero(semantic_classes == class_id)
            if not len(selected):
                continue
            intersections = np.asarray(masks[indices][:, selected].sum(axis=0), dtype=np.int64)
            ious = intersections / np.maximum(1, mask_sizes[selected] + len(indices) - intersections)
            oracle_scores[selected] = np.maximum(oracle_scores[selected], ious.astype(np.float32))
        positive_quality_predictions += int((oracle_scores > 0.0).sum())
        baseline_predictions[scene_name] = {"pred_masks": masks, "pred_scores": scores, "pred_classes": class_indices}
        oracle_predictions[scene_name] = {"pred_masks": masks, "pred_scores": oracle_scores, "pred_classes": class_indices}

    baseline_csv = args.output_dir / "baseline_from_cache.csv"
    oracle_csv = args.output_dir / "oracle_quality_ranking.csv"
    evaluate_scannet200(baseline_predictions, str(args.gt_instance_dir), output_file=str(baseline_csv), dataset="scannet200")
    evaluate_scannet200(oracle_predictions, str(args.gt_instance_dir), output_file=str(oracle_csv), dataset="scannet200")
    baseline = _metric_summary(baseline_csv)
    oracle = _metric_summary(oracle_csv)
    summary = {
        "diagnostic_only": True,
        "oracle_definition": "类别与mask固定；每条预测仅以同类别GT的最佳IoU作为离线理想质量分数。",
        "positive_quality_prediction_count": positive_quality_predictions,
        "baseline_metrics": baseline,
        "oracle_ranking_metrics": oracle,
        "oracle_metric_gains": {name: float(oracle[name] - baseline[name]) for name in baseline},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

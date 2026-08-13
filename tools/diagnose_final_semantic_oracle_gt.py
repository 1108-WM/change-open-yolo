#!/usr/bin/env python3
"""仅离线 GT 诊断：测量固定最终 mask 下的逐预测语义 Oracle 上限。

每条预测只在与某个有效 GT 的最佳几何 IoU 不低于指定门槛时，才临时改为
该 GT 的类别。该改动仅用于估计“几何不变时语义可带来的最大收益”，绝不写回
推理缓存、候选或模型参数。
"""

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


def _load_valid_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_class_ids = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        class_id = instance_id // 1000
        if instance_id <= 0 or class_id not in valid_class_ids:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) >= min_region_size:
            instances.append((class_id, indices))
    return gt_ids, instances


def _load_prediction(root, scene_name, point_count):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    scores = np.load(f"{prefix}_pred_scores.npy", mmap_mode="r")
    classes = np.load(f"{prefix}_pred_classes.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的预测点数不一致：{masks.shape[0]} 与 {point_count}")
    if masks.shape[1] != len(scores) or len(scores) != len(classes):
        raise ValueError(f"{scene_name} 的预测 mask、分数和类别数量不一致")
    return masks, np.asarray(scores, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def _metric_summary(path):
    values = {"ap": [], "ap50": [], "ap25": []}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            for name in values:
                value = float(row[name])
                if not np.isnan(value):
                    values[name].append(value)
    return {name: float(np.mean(items)) if items else float("nan") for name, items in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--oracle_min_iou", type=float, default=0.25)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    if not 0.0 < args.oracle_min_iou <= 1.0:
        raise SystemExit("--oracle_min_iou 必须在 (0, 1] 内。")
    args.scene_list = _resolve(args.scene_list)
    args.prediction_cache_dir = _resolve(args.prediction_cache_dir)
    args.output_dir = _resolve(args.output_dir)
    args.gt_instance_dir = _resolve(args.gt_instance_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    semantic_to_prediction = {int(semantic): int(prediction) for prediction, semantic in PRED_ID_TO_ID.items() if semantic >= 0}
    baseline_predictions = {}
    oracle_predictions = {}
    total_predictions = 0
    relabeled = 0
    geometric_predictions = {0.25: 0, 0.50: 0}
    semantic_correct = {0.25: 0, 0.50: 0}
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, gt_instances = _load_valid_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks, scores, classes = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
        oracle_classes = classes.copy()
        mask_sizes = np.asarray(masks.sum(axis=0), dtype=np.int64)
        total_predictions += int(masks.shape[1])
        best_ious = np.zeros(masks.shape[1], dtype=np.float32)
        best_semantic_ids = np.full(masks.shape[1], -1, dtype=np.int64)
        for semantic_id, indices in gt_instances:
            intersections = np.asarray(masks[indices].sum(axis=0), dtype=np.int64)
            ious = intersections / np.maximum(1, mask_sizes + len(indices) - intersections)
            better = ious > best_ious
            best_ious[better] = ious[better]
            best_semantic_ids[better] = semantic_id
        original_semantic_ids = np.asarray([PRED_ID_TO_ID.get(int(value), -1) for value in classes], dtype=np.int64)
        for threshold in geometric_predictions:
            valid = best_ious >= threshold
            geometric_predictions[threshold] += int(valid.sum())
            semantic_correct[threshold] += int((valid & (original_semantic_ids == best_semantic_ids)).sum())
        oracle_valid = (best_ious >= args.oracle_min_iou) & (best_semantic_ids >= 0)
        for prediction_index in np.flatnonzero(oracle_valid):
            semantic_id = int(best_semantic_ids[prediction_index])
            oracle_class = semantic_to_prediction.get(semantic_id)
            if oracle_class is not None:
                relabeled += int(oracle_classes[prediction_index] != oracle_class)
                oracle_classes[prediction_index] = oracle_class
        baseline_predictions[scene_name] = {
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_classes": classes,
        }
        oracle_predictions[scene_name] = {
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_classes": oracle_classes,
        }

    baseline_csv = args.output_dir / "baseline_from_cache.csv"
    oracle_csv = args.output_dir / f"oracle_semantics_iou{int(args.oracle_min_iou * 100):02d}.csv"
    evaluate_scannet200(
        baseline_predictions,
        str(args.gt_instance_dir),
        output_file=str(baseline_csv),
        dataset="scannet200",
    )
    evaluate_scannet200(
        oracle_predictions,
        str(args.gt_instance_dir),
        output_file=str(oracle_csv),
        dataset="scannet200",
    )
    baseline_metrics = _metric_summary(baseline_csv)
    oracle_metrics = _metric_summary(oracle_csv)
    summary = {
        "diagnostic_only": True,
        "oracle_definition": "每条预测在最佳有效GT IoU达到门槛时，临时赋为该GT类别；不改变mask或分数。",
        "oracle_min_iou": args.oracle_min_iou,
        "prediction_count": total_predictions,
        "oracle_relabeled_prediction_count": relabeled,
        "geometric_prediction_counts": {str(key): value for key, value in geometric_predictions.items()},
        "semantic_correct_prediction_counts": {str(key): value for key, value in semantic_correct.items()},
        "baseline_metrics": baseline_metrics,
        "oracle_metrics": oracle_metrics,
        "oracle_metric_gains": {
            name: float(oracle_metrics[name] - baseline_metrics[name]) for name in baseline_metrics
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""仅离线 GT 诊断：归因最终强基线预测的几何、语义和重复竞争误差。

输入必须是 ``run_evaluation.py --eval_prediction_cache_dir`` 写出的最终预测。
GT 只在本脚本中用于事后对账，绝不参与候选生成、融合、排序或阈值选择。
"""

import argparse
import csv
import json
import sys
from collections import Counter
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
                "instance_id": instance_id,
                "class_id": semantic_id,
                "class_name": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
                "indices": indices,
                "size": int(len(indices)),
            }
        )
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
    mapped_classes = np.asarray([int(PRED_ID_TO_ID.get(int(value), -1)) for value in classes], dtype=np.int64)
    return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=np.float32), mapped_classes


def _iou_matrix(masks, gt_ids, instances):
    if not len(instances) or masks.shape[1] == 0:
        return np.zeros((len(instances), masks.shape[1]), dtype=np.float32)
    sizes = masks.sum(axis=0, dtype=np.int64)
    matrix = np.zeros((len(instances), masks.shape[1]), dtype=np.float32)
    for gt_index, gt in enumerate(instances):
        intersections = masks[gt["indices"]].sum(axis=0, dtype=np.int64)
        matrix[gt_index] = intersections / np.maximum(1, sizes + gt["size"] - intersections)
    return matrix


def _maximum_matching(adjacency, prediction_count):
    """Kuhn 最大匹配，返回每个 GT 的已匹配预测编号。"""
    prediction_to_gt = np.full(prediction_count, -1, dtype=np.int32)

    def search(gt_index, visited):
        for prediction_index in adjacency[gt_index]:
            if visited[prediction_index]:
                continue
            visited[prediction_index] = True
            other_gt = prediction_to_gt[prediction_index]
            if other_gt < 0 or search(int(other_gt), visited):
                prediction_to_gt[prediction_index] = gt_index
                return True
        return False

    for gt_index in range(len(adjacency)):
        search(gt_index, np.zeros(prediction_count, dtype=bool))
    gt_to_prediction = np.full(len(adjacency), -1, dtype=np.int32)
    for prediction_index, gt_index in enumerate(prediction_to_gt):
        if gt_index >= 0:
            gt_to_prediction[gt_index] = prediction_index
    return gt_to_prediction, prediction_to_gt


def _threshold_diagnostic(scene_name, instances, ious, prediction_classes, prediction_scores, threshold):
    rows = []
    valid_prediction = prediction_classes >= 0
    semantic_edges = []
    for gt_index, gt in enumerate(instances):
        geometric = np.flatnonzero((ious[gt_index] >= threshold) & valid_prediction)
        semantic = geometric[prediction_classes[geometric] == gt["class_id"]]
        semantic_edges.append(semantic.tolist())
    gt_to_prediction, prediction_to_gt = _maximum_matching(semantic_edges, len(prediction_classes))

    for gt_index, gt in enumerate(instances):
        best_prediction = int(np.argmax(ious[gt_index])) if ious.shape[1] else -1
        best_iou = float(ious[gt_index, best_prediction]) if best_prediction >= 0 else 0.0
        geometric = semantic_edges[gt_index]
        any_geometric = bool(np.any((ious[gt_index] >= threshold) & valid_prediction))
        if not any_geometric:
            category = "无合格几何候选"
        elif not geometric:
            category = "几何正确但语义错误"
        elif gt_to_prediction[gt_index] < 0:
            category = "同类候选重复竞争"
        else:
            category = "可一对一匹配"
        rows.append(
            {
                "scene_name": scene_name,
                "gt_instance_id": gt["instance_id"],
                "gt_class": gt["class_name"],
                "gt_point_count": gt["size"],
                "iou_threshold": threshold,
                "best_final_prediction_iou": best_iou,
                "best_final_prediction_class": (
                    str(ID_TO_LABEL.get(int(prediction_classes[best_prediction]), "无有效类别"))
                    if best_prediction >= 0 else "无预测"
                ),
                "best_final_prediction_score": float(prediction_scores[best_prediction]) if best_prediction >= 0 else 0.0,
                "semantic_candidate_count": len(geometric),
                "matched_prediction_index": int(gt_to_prediction[gt_index]),
                "error_group": category,
            }
        )

    matched_predictions = prediction_to_gt >= 0
    geometric_overlap = (ious >= threshold).any(axis=0) if len(instances) else np.zeros(len(prediction_classes), dtype=bool)
    semantic_overlap = np.zeros(len(prediction_classes), dtype=bool)
    for gt_index, gt in enumerate(instances):
        semantic_overlap |= (ious[gt_index] >= threshold) & (prediction_classes == gt["class_id"])
    prediction_groups = Counter()
    for prediction_index in range(len(prediction_classes)):
        if prediction_classes[prediction_index] < 0:
            prediction_groups["无效类别预测"] += 1
        elif not geometric_overlap[prediction_index]:
            prediction_groups["无几何重叠预测"] += 1
        elif not semantic_overlap[prediction_index]:
            prediction_groups["几何重叠但类别错误预测"] += 1
        elif not matched_predictions[prediction_index]:
            prediction_groups["同类重复预测"] += 1
        else:
            prediction_groups["一对一匹配预测"] += 1
    return rows, prediction_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--thresholds", default="0.25,0.50")
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    args.scene_list = _resolve(args.scene_list)
    args.prediction_cache_dir = _resolve(args.prediction_cache_dir)
    args.gt_instance_dir = _resolve(args.gt_instance_dir)
    args.output_dir = _resolve(args.output_dir)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    if not thresholds or any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise SystemExit("--thresholds 必须是 (0, 1] 的逗号分隔数值。")

    all_rows = []
    prediction_group_totals = {threshold: Counter() for threshold in thresholds}
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks, scores, classes = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
        ious = _iou_matrix(masks, gt_ids, instances)
        for threshold in thresholds:
            rows, prediction_groups = _threshold_diagnostic(scene_name, instances, ious, classes, scores, threshold)
            all_rows.extend(rows)
            prediction_group_totals[threshold].update(prediction_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0]) if all_rows else []
    with (args.output_dir / "gt_error_attribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "diagnostic_only": True,
        "scene_count": len(_read_scenes(args.scene_list)),
        "gt_instance_count": len({(row["scene_name"], row["gt_instance_id"]) for row in all_rows}),
        "thresholds": {},
    }
    for threshold in thresholds:
        rows = [row for row in all_rows if row["iou_threshold"] == threshold]
        gt_groups = Counter(row["error_group"] for row in rows)
        summary["thresholds"][f"{threshold:.2f}"] = {
            "gt_error_groups": dict(sorted(gt_groups.items())),
            "gt_error_group_rates": {
                name: float(count / max(1, len(rows))) for name, count in sorted(gt_groups.items())
            },
            "prediction_groups": dict(sorted(prediction_group_totals[threshold].items())),
        }
    if 0.25 in thresholds and 0.50 in thresholds:
        low_groups = {
            (row["scene_name"], row["gt_instance_id"]): row["error_group"]
            for row in all_rows if row["iou_threshold"] == 0.25
        }
        high_groups = {
            (row["scene_name"], row["gt_instance_id"]): row["error_group"]
            for row in all_rows if row["iou_threshold"] == 0.50
        }
        transitions = Counter(
            f"25%：{low_groups[key]} -> 50%：{high_groups[key]}"
            for key in low_groups.keys() & high_groups.keys()
        )
        summary["iou_25_to_50_transitions"] = dict(sorted(transitions.items()))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

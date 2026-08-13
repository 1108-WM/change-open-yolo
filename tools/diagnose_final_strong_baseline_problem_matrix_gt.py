#!/usr/bin/env python3
"""仅离线 GT 诊断：统一盘点强基线的几何、边界、语义、重复和来源问题。

GT 仅在本脚本中用于事后误差归因。不会改变缓存、候选、融合、类别、分数或阈值。
"""

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
            instances.append(
                {
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "class_name": str(ID_TO_LABEL.get(class_id, class_id)),
                    "indices": indices,
                    "point_count": int(len(indices)),
                }
            )
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
    semantic_classes = np.asarray([int(PRED_ID_TO_ID.get(int(item), -1)) for item in classes], dtype=np.int64)
    return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=np.float32), semantic_classes


def _load_applied_sources(report_path):
    report = json.loads(report_path.read_text())
    by_scene = {}
    for scene_name, scene_report in report.get("scene_reports", {}).items():
        detail = scene_report.get("backprojection", scene_report)
        by_scene[scene_name] = [str(item.get("source_kind", "external_unknown")) for item in detail.get("applied", [])]
    return by_scene


def _maximum_matching(adjacency, prediction_count):
    prediction_to_gt = np.full(prediction_count, -1, dtype=np.int32)

    def search(gt_index, seen):
        for prediction_index in adjacency[gt_index]:
            if seen[prediction_index]:
                continue
            seen[prediction_index] = True
            other = prediction_to_gt[prediction_index]
            if other < 0 or search(int(other), seen):
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


def _size_bin(value, cutoffs):
    if value <= cutoffs[0]:
        return "大小最低四分位"
    if value <= cutoffs[1]:
        return "大小第二四分位"
    if value <= cutoffs[2]:
        return "大小第三四分位"
    return "大小最高四分位"


def _count_by(rows, *keys):
    counts = Counter(tuple(row[key] for key in keys) for row in rows)
    result = []
    for grouped_key, value in sorted(counts.items()):
        item = dict(zip(keys, grouped_key))
        item["count"] = value
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--base_prediction_count", type=int, default=600)
    parser.add_argument("--fusion_report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--semantic_oracle_summary", type=Path, default=None)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in ("scene_list", "prediction_cache_dir", "fusion_report", "output_dir", "gt_instance_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.semantic_oracle_summary is not None:
        args.semantic_oracle_summary = _resolve(args.semantic_oracle_summary)
    scenes = _read_scenes(args.scene_list)
    all_gt = []
    gt_payload = {}
    for scene_name in scenes:
        ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        gt_payload[scene_name] = (ids, instances)
        all_gt.extend(instance["point_count"] for instance in instances)
    cutoffs = [int(value) for value in np.quantile(all_gt, (0.25, 0.50, 0.75))]
    applied_sources = _load_applied_sources(args.fusion_report)
    gt_rows = []
    prediction_rows = []
    source_mapping_notes = Counter()
    score_values = []

    for scene_name in scenes:
        gt_ids, instances = gt_payload[scene_name]
        masks, scores, classes = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
        original_count = int(args.base_prediction_count)
        if original_count > masks.shape[1]:
            raise ValueError(f"{scene_name} 的基础预测数 {original_count} 超过冻结预测总数 {masks.shape[1]}")
        sources = ["mask3d"] * min(original_count, masks.shape[1])
        expected_external = max(0, masks.shape[1] - original_count)
        external_sources = applied_sources.get(scene_name, [])
        if len(external_sources) != expected_external:
            source_mapping_notes["外部候选来源记录数量不匹配"] += 1
            external_sources = ["external_unknown"] * expected_external
        sources.extend(external_sources)
        if len(sources) != masks.shape[1]:
            raise ValueError(f"{scene_name} 的来源映射长度异常")
        score_values.extend(float(value) for value in scores)
        sizes = np.asarray(masks.sum(axis=0), dtype=np.int64)
        ious = np.zeros((len(instances), masks.shape[1]), dtype=np.float32)
        for gt_index, instance in enumerate(instances):
            intersections = np.asarray(masks[instance["indices"]].sum(axis=0), dtype=np.int64)
            ious[gt_index] = intersections / np.maximum(1, sizes + instance["point_count"] - intersections)

        per_threshold = {}
        for threshold in (0.25, 0.50):
            semantic_edges = []
            geometric_edges = []
            for gt_index, instance in enumerate(instances):
                geometric = np.flatnonzero((ious[gt_index] >= threshold) & (classes >= 0))
                semantic = geometric[classes[geometric] == instance["class_id"]]
                geometric_edges.append(geometric)
                semantic_edges.append(semantic.tolist())
            gt_to_prediction, prediction_to_gt = _maximum_matching(semantic_edges, masks.shape[1])
            per_threshold[threshold] = (geometric_edges, semantic_edges, gt_to_prediction, prediction_to_gt)
            for gt_index, instance in enumerate(instances):
                best_prediction = int(np.argmax(ious[gt_index])) if masks.shape[1] else -1
                best_iou = float(ious[gt_index, best_prediction]) if best_prediction >= 0 else 0.0
                geometric = geometric_edges[gt_index]
                semantic = semantic_edges[gt_index]
                if len(geometric) == 0:
                    group = "无合格几何候选"
                elif not semantic:
                    group = "几何正确但语义错误"
                elif gt_to_prediction[gt_index] < 0:
                    group = "同类候选重复竞争"
                else:
                    group = "可一对一匹配"
                gt_rows.append(
                    {
                        "scene_name": scene_name,
                        "gt_instance_id": instance["instance_id"],
                        "gt_class": instance["class_name"],
                        "gt_point_count": instance["point_count"],
                        "size_bin": _size_bin(instance["point_count"], cutoffs),
                        "iou_threshold": threshold,
                        "error_group": group,
                        "best_prediction_iou": best_iou,
                        "best_prediction_source": sources[best_prediction] if best_prediction >= 0 else "无预测",
                    }
                )
            geometric_overlap = (ious >= threshold).any(axis=0) if instances else np.zeros(masks.shape[1], dtype=bool)
            semantic_overlap = np.zeros(masks.shape[1], dtype=bool)
            for gt_index, instance in enumerate(instances):
                semantic_overlap |= (ious[gt_index] >= threshold) & (classes == instance["class_id"])
            for prediction_index in range(masks.shape[1]):
                if classes[prediction_index] < 0:
                    group = "无效类别预测"
                elif not geometric_overlap[prediction_index]:
                    group = "无几何重叠预测"
                elif not semantic_overlap[prediction_index]:
                    group = "几何重叠但类别错误预测"
                elif prediction_to_gt[prediction_index] < 0:
                    group = "同类重复预测"
                else:
                    group = "一对一匹配预测"
                prediction_rows.append(
                    {
                        "scene_name": scene_name,
                        "prediction_id": prediction_index,
                        "source": sources[prediction_index],
                        "iou_threshold": threshold,
                        "prediction_group": group,
                    }
                )

    transitions = {}
    low = {(row["scene_name"], row["gt_instance_id"]): row for row in gt_rows if row["iou_threshold"] == 0.25}
    high = {(row["scene_name"], row["gt_instance_id"]): row for row in gt_rows if row["iou_threshold"] == 0.50}
    transition_counter = Counter()
    for key in low.keys() & high.keys():
        transition_counter[f"25%：{low[key]['error_group']} -> 50%：{high[key]['error_group']}"] += 1
    transitions = dict(sorted(transition_counter.items()))
    high_boundary_limited = sum(
        low[key]["error_group"] != "无合格几何候选" and high[key]["error_group"] == "无合格几何候选"
        for key in low.keys() & high.keys()
    )
    semantic_oracle = None
    if args.semantic_oracle_summary and args.semantic_oracle_summary.exists():
        semantic_oracle = json.loads(args.semantic_oracle_summary.read_text()).get("oracle_metric_gains")
    summary = {
        "diagnostic_only": True,
        "scene_count": len(scenes),
        "gt_instance_count": len(all_gt),
        "size_quartile_point_cutoffs": cutoffs,
        "source_mapping_notes": dict(sorted(source_mapping_notes.items())),
        "final_score_summary": {
            "prediction_count": len(score_values),
            "unique_values": sorted(set(score_values)),
            "has_ranking_signal": len(set(score_values)) > 1,
        },
        "gt_error_by_threshold": {},
        "prediction_error_by_threshold_and_source": {},
        "gt_error_by_class": {},
        "gt_error_by_size": {},
        "iou_25_to_50_transitions": transitions,
        "headroom_interpretation": {
            "hard_geometry_missing_at_25_count": sum(
                row["error_group"] == "无合格几何候选" for row in gt_rows if row["iou_threshold"] == 0.25
            ),
            "boundary_limited_count": int(high_boundary_limited),
            "semantic_error_at_50_count": sum(
                row["error_group"] == "几何正确但语义错误" for row in gt_rows if row["iou_threshold"] == 0.50
            ),
            "semantic_oracle_metric_gains": semantic_oracle,
            "geometry_oracle_note": "不以GT替换或修改mask伪造几何AP上限；几何问题只报告真实实例覆盖缺口。",
        },
    }
    for threshold in (0.25, 0.50):
        threshold_rows = [row for row in gt_rows if row["iou_threshold"] == threshold]
        summary["gt_error_by_threshold"][f"{threshold:.2f}"] = dict(
            sorted(Counter(row["error_group"] for row in threshold_rows).items())
        )
        summary["prediction_error_by_threshold_and_source"][f"{threshold:.2f}"] = _count_by(
            [row for row in prediction_rows if row["iou_threshold"] == threshold], "source", "prediction_group"
        )
        summary["gt_error_by_class"][f"{threshold:.2f}"] = _count_by(threshold_rows, "gt_class", "error_group")
        summary["gt_error_by_size"][f"{threshold:.2f}"] = _count_by(threshold_rows, "size_bin", "error_group")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("gt_error_ledger.csv", gt_rows), ("prediction_error_ledger.csv", prediction_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

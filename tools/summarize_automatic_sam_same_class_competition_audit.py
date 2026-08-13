#!/usr/bin/env python3
"""汇总同类候选竞争审计的连续几何分布，不产生竞争决策。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复项")
    return scenes


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p50": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)), "max": float(values.max()),
    }


def summarize(records):
    """按审计类别保留 IoU/双向覆盖的连续描述，不导出选择阈值。"""
    grouped = defaultdict(lambda: {"iou": [], "min_coverage": [], "max_coverage": [], "semantic_js": []})
    for record in records:
        geometry = record["geometry"]
        bucket = grouped[record["competition_audit_category"]]
        bucket["iou"].append(float(geometry["iou"]))
        coverages = (float(geometry["left_coverage"]), float(geometry["right_coverage"]))
        bucket["min_coverage"].append(min(coverages))
        bucket["max_coverage"].append(max(coverages))
        bucket["semantic_js"].append(float(record["semantic_js_divergence"]))
    return {
        name: {metric: _summary(values) for metric, values in metrics.items()}
        for name, metrics in sorted(grouped.items())
    }


def dominance_containment_orientation(record):
    """区分质量支配者是几何容器还是局部 mask，避免二者被错误等同。"""
    direction = record["pareto_dominance_direction"]
    containment = record["containment"]["state"]
    if direction == "neither_candidate_dominates":
        return "no_pareto_dominance"
    if containment == "exact_geometric_duplicate":
        return "pareto_dominance_on_exact_duplicate"
    if containment == "left_strictly_contained_by_right":
        return (
            "dominant_candidate_is_geometric_container"
            if direction == "right_dominates_left"
            else "dominant_candidate_is_geometrically_contained"
        )
    if containment == "right_strictly_contained_by_left":
        return (
            "dominant_candidate_is_geometric_container"
            if direction == "left_dominates_right"
            else "dominant_candidate_is_geometrically_contained"
        )
    return "pareto_dominance_without_strict_containment"


def dominance_native_summary(records):
    """仅描述支配/被支配端与 native 的既有连续关系，不推导删留建议。"""
    grouped = defaultdict(lambda: {"dominant_native_iou": [], "dominated_native_iou": []})
    for record in records:
        direction = record["pareto_dominance_direction"]
        if direction == "neither_candidate_dominates":
            continue
        orientation = dominance_containment_orientation(record)
        if direction == "left_dominates_right":
            dominant, dominated = record["left_native_relation_summary"], record["right_native_relation_summary"]
        else:
            dominant, dominated = record["right_native_relation_summary"], record["left_native_relation_summary"]
        grouped[orientation]["dominant_native_iou"].append(float(dominant["max_any_iou"]))
        grouped[orientation]["dominated_native_iou"].append(float(dominated["max_any_iou"]))
    return {
        orientation: {metric: _summary(values) for metric, values in metrics.items()}
        for orientation, metrics in sorted(grouped.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "audit_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    args.output_dir.mkdir(parents=True)
    records = []
    for scene in _scenes(args.scene_list):
        records.extend(_jsonl(args.audit_root / scene / "same_class_competition_audit.jsonl"))
    categories = Counter(record["competition_audit_category"] for record in records)
    orientations = Counter(dominance_containment_orientation(record) for record in records)
    payload = {
        "purpose": "连续检查 Pareto 支配关系是否伴随严格包含或更高的双向覆盖。",
        "gt_usage": "none",
        "decision_state": "只读汇总；不产生接受、抑制、合并、删除或预测决定。",
        "same_class_relation_count": len(records),
        "competition_audit_category_counts": dict(sorted(categories.items())),
        "dominance_containment_orientation_counts": dict(sorted(orientations.items())),
        "continuous_geometry_by_audit_category": summarize(records),
        "dominance_native_relation_continuous_summary": dominance_native_summary(records),
        "params": vars(args),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

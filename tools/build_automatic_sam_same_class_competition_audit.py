#!/usr/bin/env python3
"""审计自动 SAM 同类候选的包含和 Pareto 竞争证据，不产生筛选决定。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def _write_jsonl(path, records):
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def dominates(left, right, epsilon=1e-12):
    """严格 Pareto 支配：四个既有连续目标均不差，至少一个严格更好。"""
    objectives = (
        (float(left["gvc_score"]), float(right["gvc_score"]), True),
        (float(left["semantic_vote_margin"]), float(right["semantic_vote_margin"]), True),
        (float(left["semantic_normalized_entropy"]), float(right["semantic_normalized_entropy"]), False),
        (float(left["independent_from_native_ratio"]), float(right["independent_from_native_ratio"]), True),
    )
    no_worse = all(
        left_value >= right_value - epsilon if maximize else left_value <= right_value + epsilon
        for left_value, right_value, maximize in objectives
    )
    strictly_better = any(
        left_value > right_value + epsilon if maximize else left_value < right_value - epsilon
        for left_value, right_value, maximize in objectives
    )
    return bool(no_worse and strictly_better)


def containment_state(geometry, left_size, right_size):
    """从精确整数点计数得到集合包含事实，不设置任何连续几何门槛。"""
    intersection = int(geometry["intersection_point_count"])
    left_residual = int(left_size) - intersection
    right_residual = int(right_size) - intersection
    if intersection == 0:
        state = "disjoint_geometry"
    elif left_residual == 0 and right_residual == 0:
        state = "exact_geometric_duplicate"
    elif left_residual == 0:
        state = "left_strictly_contained_by_right"
    elif right_residual == 0:
        state = "right_strictly_contained_by_left"
    else:
        state = "partial_geometric_overlap"
    return {
        "state": state,
        "left_exclusive_point_count": left_residual,
        "right_exclusive_point_count": right_residual,
        "has_strict_set_containment": state in {
            "exact_geometric_duplicate",
            "left_strictly_contained_by_right",
            "right_strictly_contained_by_left",
        },
    }


def _native_summary(relations):
    summary = {
        "relation_count": len(relations),
        "same_class_relation_count": sum(bool(item["same_class"]) for item in relations),
        "cross_class_relation_count": sum(not bool(item["same_class"]) for item in relations),
    }
    for suffix, subset in (
        ("any", relations),
        ("same_class", [item for item in relations if item["same_class"]]),
        ("cross_class", [item for item in relations if not item["same_class"]]),
    ):
        best = max(subset, key=lambda item: item["geometry"]["iou"], default=None)
        summary[f"max_{suffix}_iou"] = float(best["geometry"]["iou"]) if best else 0.0
        summary[f"max_{suffix}_automatic_coverage"] = float(best["geometry"]["left_coverage"]) if best else 0.0
        summary[f"best_{suffix}_native_candidate_index"] = int(best["native_candidate_index"]) if best else -1
    return summary


def _dominance_direction(left, right):
    if dominates(left, right):
        return "left_dominates_right"
    if dominates(right, left):
        return "right_dominates_left"
    return "neither_candidate_dominates"


def _category(dominance_direction, containment):
    has_dominance = dominance_direction != "neither_candidate_dominates"
    if has_dominance and containment["has_strict_set_containment"]:
        return "pareto_dominance_with_strict_containment"
    if has_dominance:
        return "pareto_dominance_without_strict_containment"
    if containment["has_strict_set_containment"]:
        return "strict_containment_without_pareto_dominance"
    return "unresolved_same_class_relation"


def build_scene_audit(nodes, auto_relations, native_relations):
    """为每条同类关系附加完整质量、包含和 native 上下文，绝不作删留。"""
    node_by_id = {int(node["candidate_id"]): node for node in nodes}
    native_by_candidate = defaultdict(list)
    for relation in native_relations:
        native_by_candidate[int(relation["automatic_candidate_id"])].append(relation)

    records, excluded_cross_class = [], 0
    for relation in auto_relations:
        if not relation["same_class"]:
            excluded_cross_class += 1
            continue
        left_id, right_id = int(relation["left_candidate_id"]), int(relation["right_candidate_id"])
        left, right = node_by_id[left_id], node_by_id[right_id]
        containment = containment_state(relation["geometry"], left["point_count"], right["point_count"])
        direction = _dominance_direction(left, right)
        records.append({
            "relation_kind": "same_class_automatic_candidate_competition_audit",
            "left_candidate_id": left_id,
            "right_candidate_id": right_id,
            "left_source_track_id": int(left["source_track_id"]),
            "right_source_track_id": int(right["source_track_id"]),
            "class_id": int(left["class_id"]),
            "geometry": relation["geometry"],
            "containment": containment,
            "semantic_js_divergence": float(relation["semantic_js_divergence"]),
            "cross_view_and_granularity_evidence": relation["cross_view_and_granularity_evidence"],
            "left_quality": {
                key: float(left[key]) for key in (
                    "gvc_score", "semantic_vote_margin", "semantic_normalized_entropy", "independent_from_native_ratio"
                )
            },
            "right_quality": {
                key: float(right[key]) for key in (
                    "gvc_score", "semantic_vote_margin", "semantic_normalized_entropy", "independent_from_native_ratio"
                )
            },
            "pareto_dominance_direction": direction,
            "left_native_relation_summary": _native_summary(native_by_candidate[left_id]),
            "right_native_relation_summary": _native_summary(native_by_candidate[right_id]),
            "competition_audit_category": _category(direction, containment),
            "decision_state": "只读审计；不抑制、合并、删除、改类别、改分数或输出预测。",
        })
    return records, excluded_cross_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--family-graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "family_graph_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)

    totals, categories = Counter(), Counter()
    for index, scene in enumerate(_scenes(args.scene_list), start=1):
        scene_root = args.family_graph_root / scene
        records, excluded = build_scene_audit(
            _jsonl(scene_root / "automatic_candidate_nodes.jsonl"),
            _jsonl(scene_root / "automatic_to_automatic_relations.jsonl"),
            _jsonl(scene_root / "automatic_to_native_relations.jsonl"),
        )
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "same_class_competition_audit.jsonl", records)
        totals["same_class_relation_count"] += len(records)
        totals["cross_class_relation_excluded_count"] += excluded
        categories.update(item["competition_audit_category"] for item in records)
        print(f"[场景完成] {index}: {scene}，同类审计关系 {len(records)}，排除跨类关系 {excluded}", flush=True)
    payload = {
        "purpose": "审计同类自动候选的 Pareto 支配是否同时具有严格集合包含证据。",
        "gt_usage": "none",
        "decision_state": "不筛除、不合并、不改类别、不改分数、不改 native、不输出预测或 AP。",
        "scene_count": len(_scenes(args.scene_list)),
        **dict(totals),
        "competition_audit_category_counts": dict(sorted(categories.items())),
        "params": vars(args),
    }
    (args.output_root / "same_class_competition_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

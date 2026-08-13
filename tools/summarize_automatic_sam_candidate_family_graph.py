#!/usr/bin/env python3
"""汇总自动 SAM 候选族关系图的无 GT 连续证据，不产生竞争决策。"""

import argparse
import json
from collections import Counter
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
        "count": int(len(values)), "mean": float(values.mean()), "p50": float(np.quantile(values, .5)),
        "p90": float(np.quantile(values, .9)), "max": float(values.max()),
    }


def dominates(left, right, epsilon=1e-12):
    """只记录四项连续质量中严格 Pareto 支配，不据此作任何竞争决定。"""
    objectives = (
        (float(left["gvc_score"]), float(right["gvc_score"]), True),
        (float(left["semantic_vote_margin"]), float(right["semantic_vote_margin"]), True),
        (float(left["semantic_normalized_entropy"]), float(right["semantic_normalized_entropy"]), False),
        (float(left["independent_from_native_ratio"]), float(right["independent_from_native_ratio"]), True),
    )
    no_worse = all(left_value >= right_value - epsilon if maximize else left_value <= right_value + epsilon for left_value, right_value, maximize in objectives)
    strictly_better = any(left_value > right_value + epsilon if maximize else left_value < right_value - epsilon for left_value, right_value, maximize in objectives)
    return bool(no_worse and strictly_better)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--family-graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "family_graph_root", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    args.output_dir.mkdir(parents=True)

    auto_iou, native_iou, independent, family_auto_size, family_native_size = [], [], [], [], []
    relation_tags, family_states = Counter(), Counter()
    counts = Counter()
    for scene in _scenes(args.scene_list):
        root = args.family_graph_root / scene
        nodes = _jsonl(root / "automatic_candidate_nodes.jsonl")
        auto_relations = _jsonl(root / "automatic_to_automatic_relations.jsonl")
        native_relations = _jsonl(root / "automatic_to_native_relations.jsonl")
        families = json.loads((root / "local_candidate_families.json").read_text())
        node_by_id = {int(node["candidate_id"]): node for node in nodes}
        independent.extend(float(node["independent_from_native_ratio"]) for node in nodes)
        counts["automatic_candidate_count"] += len(nodes)
        counts["automatic_relation_count"] += len(auto_relations)
        counts["native_relation_count"] += len(native_relations)
        counts["local_family_count"] += len(families)
        for relation in auto_relations:
            tags = relation["relation_tags"]
            relation_tags.update(tags)
            geometry = relation["geometry"]
            if geometry["intersection_point_count"]:
                auto_iou.append(float(geometry["iou"]))
            left, right = node_by_id[int(relation["left_candidate_id"])], node_by_id[int(relation["right_candidate_id"])]
            if dominates(left, right) or dominates(right, left):
                counts["pareto_dominance_relation_count"] += 1
                if relation["same_class"]:
                    counts["same_class_pareto_dominance_relation_count"] += 1
                else:
                    counts["cross_class_pareto_dominance_relation_count"] += 1
        for relation in native_relations:
            relation_tags.update(relation["relation_tags"])
            native_iou.append(float(relation["geometry"]["iou"]))
        for family in families:
            family_auto_size.append(len(family["automatic_neighbor_candidate_ids"]))
            family_native_size.append(len(family["native_neighbor_candidate_indices"]))
            family_states.update(family["audit_states"])
    payload = {
        "purpose": "只读汇总候选族连续关系，为后续预注册规则提供审计，不产生竞争决定。",
        "gt_usage": "none",
        "decision_state": "不筛除、不合并、不改类别、不改分数、不改 native、不输出预测或 AP。",
        **dict(counts),
        "automatic_geometric_iou": _summary(auto_iou),
        "automatic_to_native_geometric_iou": _summary(native_iou),
        "automatic_independent_from_native_ratio": _summary(independent),
        "automatic_neighbor_count_per_local_family": _summary(family_auto_size),
        "native_neighbor_count_per_local_family": _summary(family_native_size),
        "relation_tag_counts": dict(sorted(relation_tags.items())),
        "local_family_audit_state_counts": dict(sorted(family_states.items())),
        "params": vars(args),
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

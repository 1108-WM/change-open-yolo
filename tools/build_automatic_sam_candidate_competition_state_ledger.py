#!/usr/bin/env python3
"""合成自动 SAM 候选的多视图增量和候选族状态，不作竞争决定。"""

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


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(values)), "mean": float(values.mean()), "p50": float(np.quantile(values, .5)),
        "p90": float(np.quantile(values, .9)), "max": float(values.max()),
    }


def increment_features(row):
    views = row["view_records"]
    observed = sum(int(item["candidate_observation_point_count"]) for item in views)
    independent = sum(int(item["candidate_independent_point_count"]) for item in views)
    same_class = sum(int(item["candidate_same_class_native_explained_point_count"]) for item in views)
    any_class = sum(int(item["candidate_any_native_explained_point_count"]) for item in views)
    return {
        "track_observation_count": int(row["track_observation_count"]),
        "candidate_supported_observation_count": int(row["candidate_supported_observation_count"]),
        "candidate_independent_observation_count": int(row["candidate_independent_observation_count"]),
        "all_track_observations_have_independent_points": bool(
            row["track_observation_count"] > 0
            and row["candidate_independent_observation_count"] == row["track_observation_count"]
        ),
        "weighted_independent_ratio": float(independent / max(1, observed)),
        "weighted_same_class_native_explained_ratio": float(same_class / max(1, observed)),
        "weighted_any_native_explained_ratio": float(any_class / max(1, observed)),
    }


def _role_for_candidate(relation, candidate_id):
    direction, containment = relation["pareto_dominance_direction"], relation["containment"]["state"]
    left, right = int(relation["left_candidate_id"]), int(relation["right_candidate_id"])
    if direction == "neither_candidate_dominates" or containment not in {
        "left_strictly_contained_by_right", "right_strictly_contained_by_left"
    }:
        return None
    dominant = left if direction == "left_dominates_right" else right
    container = right if containment == "left_strictly_contained_by_right" else left
    if dominant == container:
        return "dominant_geometric_container" if candidate_id == dominant else "contained_by_dominant_container"
    return "dominant_geometric_part" if candidate_id == dominant else "container_dominated_by_part"


def build_scene_state(increment_rows, audit_rows, families):
    """将连续证据和关系角色组织为候选状态，不给出保留或抑制建议。"""
    family_by_candidate = {int(item["anchor_candidate_id"]): item for item in families}
    relations_by_candidate = defaultdict(list)
    for relation in audit_rows:
        relations_by_candidate[int(relation["left_candidate_id"])].append(relation)
        relations_by_candidate[int(relation["right_candidate_id"])].append(relation)
    rows = []
    for source in increment_rows:
        candidate_id = int(source["candidate_id"])
        related = relations_by_candidate[candidate_id]
        features = increment_features(source)
        roles = Counter(role for relation in related if (role := _role_for_candidate(relation, candidate_id)) is not None)
        semantic_js = [float(relation["semantic_js_divergence"]) for relation in related]
        family = family_by_candidate.get(candidate_id, {})
        states = []
        if features["candidate_independent_observation_count"] == 0:
            states.append("no_native_independent_observation")
        elif features["candidate_independent_observation_count"] == 1:
            states.append("single_view_native_increment_observed")
        else:
            states.append("multiview_native_increment_observed")
        if features["all_track_observations_have_independent_points"]:
            states.append("increment_observed_in_all_track_views")
        if roles["dominant_geometric_container"]:
            states.append("dominant_geometric_container_relation_observed")
        if roles["contained_by_dominant_container"]:
            states.append("contained_by_dominant_container_relation_observed")
        if roles["dominant_geometric_part"] or roles["container_dominated_by_part"]:
            states.append("part_container_quality_direction_conflict_observed")
        if "cross_class_relation_observed" in family.get("audit_states", []):
            states.append("cross_class_competition_kept_as_dual_hypothesis")
        rows.append({
            "scene_name": source["scene_name"],
            "candidate_id": candidate_id,
            "source_track_id": int(source["source_track_id"]),
            "class_id": int(source["class_id"]),
            "class_name": source["class_name"],
            "multiview_increment": features,
            "same_class_competition": {
                "relation_count": len(related),
                "semantic_js_divergence": _summary(semantic_js),
                "relation_role_counts": dict(sorted(roles.items())),
            },
            "candidate_family_context": {
                "automatic_neighbor_count": len(family.get("automatic_neighbor_candidate_ids", [])),
                "native_neighbor_count": len(family.get("native_neighbor_candidate_indices", [])),
                "existing_audit_states": family.get("audit_states", []),
            },
            "candidate_evidence_states": states,
            "decision_state": "候选状态账本；不保留、不抑制、不合并、不删除、不改类别、不改分数、不输出预测。",
            "gt_usage": "none",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--increment-ledger-root", type=Path, required=True)
    parser.add_argument("--same-class-audit-root", type=Path, required=True)
    parser.add_argument("--family-graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "increment_ledger_root", "same_class_audit_root", "family_graph_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    totals, states = Counter(), Counter()
    for index, scene in enumerate(_scenes(args.scene_list), start=1):
        rows = build_scene_state(
            _jsonl(args.increment_ledger_root / scene / "automatic_sam_candidate_multiview_increment_ledger.jsonl"),
            _jsonl(args.same_class_audit_root / scene / "same_class_competition_audit.jsonl"),
            json.loads((args.family_graph_root / scene / "local_candidate_families.json").read_text()),
        )
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "automatic_sam_candidate_competition_state_ledger.jsonl", rows)
        totals["candidate_count"] += len(rows)
        states.update(state for row in rows for state in row["candidate_evidence_states"])
        print(f"[场景完成] {index}: {scene}，候选状态 {len(rows)}", flush=True)
    payload = {
        "purpose": "合成候选自身多视图增量、同类竞争方向、跨类关系与 native 关系状态。",
        "gt_usage": "none",
        "decision_state": "不筛除、不合并、不改类别、不改分数、不改 native、不输出预测或 AP。",
        "scene_count": len(_scenes(args.scene_list)), **dict(totals),
        "candidate_evidence_state_counts": dict(sorted(states.items())), "params": vars(args),
    }
    (args.output_root / "automatic_sam_candidate_competition_state_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

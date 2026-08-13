#!/usr/bin/env python3
"""记录同类自动候选的互补聚合与边界竞争证据，不合并候选。"""

import argparse
import json
from collections import Counter
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


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def complementarity_states(relation, left, right):
    """用已记录的连续证据标注聚合/边界竞争假设，不给出合并决定。"""
    containment = relation["containment"]
    evidence = relation["cross_view_and_granularity_evidence"]
    left_increment = left["multiview_increment"]
    right_increment = right["multiview_increment"]
    left_multi = left_increment["candidate_independent_observation_count"] >= 2
    right_multi = right_increment["candidate_independent_observation_count"] >= 2
    states = []
    if left_multi and right_multi:
        states.append("both_candidates_have_multiview_native_increment")
    if containment["state"] == "partial_geometric_overlap":
        states.append("two_sided_exclusive_geometry_observed")
    if evidence["cross_view_edge_count"]:
        states.append("cross_view_pair_evidence_observed")
    if evidence["same_frame_granularity_relation_count"]:
        states.append("same_frame_granularity_evidence_observed")
    if (
        left_multi and right_multi
        and containment["state"] == "partial_geometric_overlap"
        and evidence["cross_view_edge_count"]
    ):
        states.append("multiview_complementarity_hypothesis")
    if containment["has_strict_set_containment"]:
        states.append("containment_boundary_competition_hypothesis")
    return states


def build_scene_ledger(state_rows, audit_rows):
    states_by_id = {int(row["candidate_id"]): row for row in state_rows}
    records = []
    for relation in audit_rows:
        left_id, right_id = int(relation["left_candidate_id"]), int(relation["right_candidate_id"])
        left, right = states_by_id[left_id], states_by_id[right_id]
        states = complementarity_states(relation, left, right)
        records.append({
            "relation_kind": "same_class_candidate_complementarity_audit",
            "left_candidate_id": left_id,
            "right_candidate_id": right_id,
            "left_source_track_id": int(left["source_track_id"]),
            "right_source_track_id": int(right["source_track_id"]),
            "class_id": int(left["class_id"]),
            "geometry": relation["geometry"],
            "containment": relation["containment"],
            "semantic_js_divergence": float(relation["semantic_js_divergence"]),
            "cross_view_and_granularity_evidence": relation["cross_view_and_granularity_evidence"],
            "left_multiview_increment": left["multiview_increment"],
            "right_multiview_increment": right["multiview_increment"],
            "pair_evidence_states": states,
            "decision_state": "关系账本；不聚合、不替换、不抑制、不删除、不改类别、不输出预测。",
            "gt_usage": "none",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-state-root", type=Path, required=True)
    parser.add_argument("--same-class-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "candidate_state_root", "same_class_audit_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    totals, states = Counter(), Counter()
    for index, scene in enumerate(_scenes(args.scene_list), start=1):
        rows = build_scene_ledger(
            _jsonl(args.candidate_state_root / scene / "automatic_sam_candidate_competition_state_ledger.jsonl"),
            _jsonl(args.same_class_audit_root / scene / "same_class_competition_audit.jsonl"),
        )
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "automatic_sam_candidate_complementarity_ledger.jsonl", rows)
        totals["same_class_relation_count"] += len(rows)
        states.update(state for row in rows for state in row["pair_evidence_states"])
        print(f"[场景完成] {index}: {scene}，同类互补关系 {len(rows)}", flush=True)
    payload = {
        "purpose": "记录同类候选的多视图互补聚合与边界竞争证据。",
        "gt_usage": "none",
        "decision_state": "不聚合、不抑制、不删除、不改类别、不改分数、不改 native、不输出预测或 AP。",
        "scene_count": len(_scenes(args.scene_list)), **dict(totals),
        "pair_evidence_state_counts": dict(sorted(states.items())), "params": vars(args),
    }
    (args.output_root / "automatic_sam_candidate_complementarity_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

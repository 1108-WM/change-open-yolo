#!/usr/bin/env python3
"""将同类自动候选互补账本和真实二维层级证据组织为变体计划，不生成预测。"""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEAR_FULL_COVERAGE = 0.95
MIN_MUTUAL_EXACT_COVERAGE = 0.10


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或包含重复场景")
    return scenes


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _track_pair(left_track_id, right_track_id):
    return tuple(sorted((int(left_track_id), int(right_track_id))))


def exact_2d_state(evidence):
    """将连续精确关系归为计划状态，阈值不构成候选接受或拒绝规则。"""
    if evidence is None:
        return "no_exact_2d_evidence"
    if max(float(evidence["max_left_coverage"]), float(evidence["max_right_coverage"])) >= NEAR_FULL_COVERAGE:
        return "near_containment_exact_2d_evidence"
    mutual_max_coverage = min(float(evidence["max_left_coverage"]), float(evidence["max_right_coverage"]))
    if int(evidence["distinct_frame_count"]) >= 2 and mutual_max_coverage >= MIN_MUTUAL_EXACT_COVERAGE:
        return "multiframe_exact_2d_evidence"
    if int(evidence["distinct_frame_count"]) >= 2:
        return "weak_multiframe_exact_2d_boundary_touch_evidence"
    return "single_frame_exact_2d_evidence"


def plan_state(relation, exact_evidence):
    """只预注册应产生何种可回退变体，不决定该变体是否胜出。"""
    complement = "multiview_complementarity_hypothesis" in relation["pair_evidence_states"]
    exact_state = exact_2d_state(exact_evidence)
    if not complement:
        return "keep_originals_only_insufficient_multiview_complementarity", exact_state
    if exact_state == "multiframe_exact_2d_evidence":
        return "aggregation_variant_hypothesis_keep_originals", exact_state
    if exact_state == "near_containment_exact_2d_evidence":
        return "boundary_competition_variant_hypothesis_keep_originals", exact_state
    return "keep_originals_only_insufficient_exact_2d_support", exact_state


def build_scene_plan(complementarity_rows, exact_hierarchy_rows):
    exact_by_track_pair = {
        _track_pair(row["left_source_track_id"], row["right_source_track_id"]): row
        for row in exact_hierarchy_rows
    }
    records = []
    for relation in complementarity_rows:
        track_pair = _track_pair(relation["left_source_track_id"], relation["right_source_track_id"])
        exact_evidence = exact_by_track_pair.get(track_pair)
        state, exact_state = plan_state(relation, exact_evidence)
        records.append({
            "relation_kind": "same_class_candidate_aggregation_boundary_variant_plan",
            "left_candidate_id": int(relation["left_candidate_id"]),
            "right_candidate_id": int(relation["right_candidate_id"]),
            "left_source_track_id": int(relation["left_source_track_id"]),
            "right_source_track_id": int(relation["right_source_track_id"]),
            "class_id": int(relation["class_id"]),
            "semantic_js_divergence": float(relation["semantic_js_divergence"]),
            "geometry": relation["geometry"],
            "containment": relation["containment"],
            "pair_evidence_states": relation["pair_evidence_states"],
            "exact_2d_evidence_state": exact_state,
            "exact_2d_evidence": exact_evidence,
            "variant_plan_state": state,
            "original_candidate_policy": "left/right 原候选及冻结 native 均保留为回退，不替换、不删除。",
            "decision_state": "变体计划账本；不创建三维并集、不改变 mask、类别或分数，不输出预测或 AP。",
            "gt_usage": "none",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--complementarity-root", type=Path, required=True)
    parser.add_argument("--exact-hierarchy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "complementarity_root", "exact_hierarchy_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    state_counts, exact_state_counts = Counter(), Counter()
    actionable_count = 0
    scenes = _scenes(args.scene_list)
    for index, scene in enumerate(scenes, start=1):
        all_rows = build_scene_plan(
            _jsonl(args.complementarity_root / scene / "automatic_sam_candidate_complementarity_ledger.jsonl"),
            _jsonl(args.exact_hierarchy_root / scene / "automatic_track_pair_exact_hierarchy_evidence.jsonl"),
        )
        rows = [row for row in all_rows if row["variant_plan_state"] in {
            "aggregation_variant_hypothesis_keep_originals",
            "boundary_competition_variant_hypothesis_keep_originals",
        }]
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "automatic_sam_aggregation_boundary_variant_plan.jsonl", rows)
        state_counts.update(row["variant_plan_state"] for row in all_rows)
        exact_state_counts.update(row["exact_2d_evidence_state"] for row in all_rows)
        actionable_count += len(rows)
        print(f"[场景完成] {index}: {scene}，可行动变体关系 {len(rows)}/{len(all_rows)}", flush=True)
    payload = {
        "purpose": "用真实二维连续关系分流同类候选的聚合/边界竞争变体假设。",
        "gt_usage": "none",
        "decision_state": "只生成计划；原候选与 native 保留，不创建变体、不聚合、不抑制、不删除、不输出预测或 AP。",
        "near_full_coverage": NEAR_FULL_COVERAGE,
        "min_mutual_exact_coverage": MIN_MUTUAL_EXACT_COVERAGE,
        "scene_count": len(scenes),
        "relation_count": sum(state_counts.values()),
        "actionable_relation_count": actionable_count,
        "stored_relation_policy": "仅持久化聚合/边界竞争变体假设；非行动关系仅计入汇总。",
        "variant_plan_state_counts": dict(sorted(state_counts.items())),
        "exact_2d_evidence_state_counts": dict(sorted(exact_state_counts.items())),
        "params": vars(args),
    }
    (args.output_root / "automatic_sam_aggregation_boundary_variant_plan_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

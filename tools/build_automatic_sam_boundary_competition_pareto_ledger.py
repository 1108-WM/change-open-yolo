#!/usr/bin/env python3
"""将二维层级边界竞争关系组织为无权重 Pareto 审计账本。"""

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
        raise ValueError("场景列表为空或包含重复场景")
    return scenes


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _key(row):
    return tuple(sorted((int(row["left_candidate_id"]), int(row["right_candidate_id"]))))


def dominates(left, right, epsilon=1e-12):
    """四项单候选连续质量均不差且至少一项更好，才构成无权重 Pareto 支配。"""
    objectives = (
        (float(left["gvc_score"]), float(right["gvc_score"]), True),
        (float(left["semantic_vote_margin"]), float(right["semantic_vote_margin"]), True),
        (float(left["semantic_normalized_entropy"]), float(right["semantic_normalized_entropy"]), False),
        (float(left["weighted_independent_ratio"]), float(right["weighted_independent_ratio"]), True),
    )
    no_worse = all(a >= b - epsilon if maximize else a <= b + epsilon for a, b, maximize in objectives)
    strictly_better = any(a > b + epsilon if maximize else a < b - epsilon for a, b, maximize in objectives)
    return bool(no_worse and strictly_better)


def direction(left, right):
    if dominates(left, right):
        return "left_pareto_dominates_right"
    if dominates(right, left):
        return "right_pareto_dominates_left"
    return "pareto_incomparable"


def _quality(candidate, state):
    return {
        "gvc_score": float(candidate["gvc_score"]),
        "semantic_vote_margin": float(candidate["semantic_vote_margin"]),
        "semantic_normalized_entropy": float(candidate["semantic_normalized_entropy"]),
        "weighted_independent_ratio": float(state["multiview_increment"]["weighted_independent_ratio"]),
        "candidate_independent_observation_count": int(state["multiview_increment"]["candidate_independent_observation_count"]),
        "support_view_count": int(candidate["support_view_count"]),
        "native_relation_diagnostic": candidate["native_relation_diagnostic"],
    }


def build_scene_ledger(plan_rows, dino_rows, candidate_rows, state_rows):
    plans = {_key(row): row for row in plan_rows if row["variant_plan_state"] == "boundary_competition_variant_hypothesis_keep_originals"}
    dinos = {_key(row): row for row in dino_rows}
    candidates = {int(row["candidate_id"]): row for row in candidate_rows}
    states = {int(row["candidate_id"]): row for row in state_rows}
    records = []
    for key, plan in sorted(plans.items()):
        if key not in dinos:
            raise ValueError(f"边界关系 {key} 缺少 DINOv2 账本")
        left_id, right_id = int(plan["left_candidate_id"]), int(plan["right_candidate_id"])
        left, right = _quality(candidates[left_id], states[left_id]), _quality(candidates[right_id], states[right_id])
        records.append({
            "relation_kind": "automatic_sam_boundary_competition_pareto_ledger",
            "left_candidate_id": left_id, "right_candidate_id": right_id,
            "left_source_track_id": int(plan["left_source_track_id"]), "right_source_track_id": int(plan["right_source_track_id"]),
            "class_id": int(plan["class_id"]), "semantic_js_divergence": float(plan["semantic_js_divergence"]),
            "exact_2d_evidence": plan["exact_2d_evidence"],
            "dino_vits14_masked_crop_cosine": dinos[key]["dino_vits14_masked_crop_cosine"],
            "dino_frame_pair_count": len(dinos[key]["frame_records"]),
            "left_quality": left, "right_quality": right,
            "unweighted_pareto_direction": direction(left, right),
            "decision_state": "无权重 Pareto 审计；支配仅是潜在偏好，不抑制、不合并、不删除、不改类别或分数，不输出预测或 AP。",
            "gt_usage": "none",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--variant-plan-root", required=True, type=Path)
    parser.add_argument("--dino-ledger-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--candidate-state-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for name in ("scene_list", "variant_plan_root", "dino_ledger_root", "candidate_root", "candidate_state_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    totals = Counter()
    scenes = _scenes(args.scene_list)
    for index, scene in enumerate(scenes, start=1):
        candidates = json.loads((args.candidate_root / scene / "backprojection_candidates.json").read_text())["candidates"]
        rows = build_scene_ledger(
            _jsonl(args.variant_plan_root / scene / "automatic_sam_aggregation_boundary_variant_plan.jsonl"),
            _jsonl(args.dino_ledger_root / scene / "automatic_sam_boundary_dino_appearance_ledger.jsonl"),
            candidates,
            _jsonl(args.candidate_state_root / scene / "automatic_sam_candidate_competition_state_ledger.jsonl"),
        )
        target = args.output_root / scene
        target.mkdir()
        _write_jsonl(target / "automatic_sam_boundary_competition_pareto_ledger.jsonl", rows)
        totals["boundary_relation_count"] += len(rows)
        totals.update(row["unweighted_pareto_direction"] for row in rows)
        print(f"[场景完成] {index}: {scene}，边界竞争关系 {len(rows)}", flush=True)
    summary = {
        "purpose": "联合 DINOv2 外观、二维层级、语义分歧和单候选连续质量，审计边界竞争关系的无权重 Pareto 状态。",
        "gt_usage": "none",
        "decision_state": "无权重 Pareto 审计；不抑制、不合并、不删除、不改类别或分数，不输出预测或 AP。",
        "scene_count": len(scenes), **dict(totals), "params": vars(args),
    }
    (args.output_root / "automatic_sam_boundary_competition_pareto_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

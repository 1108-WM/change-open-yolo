#!/usr/bin/env python3
"""Plan unified A/B/unknown ownership for frozen boundary superpoints.

Each candidate region must have complete affinity evidence on all of its
contact edges.  A unique owner is planned only when one candidate Pareto
dominates every competitor on both the mean and minimum complete edge
affinity.  Otherwise the boundary superpoint remains explicitly unknown.

This baseline-adapted plan is not an exact reproduction of unpublished
MV3DIS region aggregation.  It has no tunable margin and reads no GT,
semantics, native predictions, scores, or AP.  It writes a plan only.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_ADAPTER = "mv3dis_relative_depth_rle_reference"
ASSIGN_ACTION = "assign_unique_pareto_owner"
KEEP_ACTION = "keep_unique_pareto_owner"
UNKNOWN_ACTION = "defer_to_unknown_boundary_pool"
DECISION_CONTRACT = (
    "complete candidate edges; unique mean-and-minimum affinity Pareto owner; "
    "otherwise unknown"
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def complete_candidate_summary(candidate):
    pairs = candidate.get("pair_evidence", [])
    values = [row.get("pair_affinity") for row in pairs]
    complete = bool(pairs) and all(value is not None for value in values)
    if not complete:
        return {
            "proposal_id": int(candidate["proposal_id"]),
            "neighbor_pair_count": len(pairs),
            "complete_edge_evidence": False,
            "mean_pair_affinity": None,
            "minimum_pair_affinity": None,
        }
    values = list(map(float, values))
    return {
        "proposal_id": int(candidate["proposal_id"]),
        "neighbor_pair_count": len(values),
        "complete_edge_evidence": True,
        "mean_pair_affinity": float(np.mean(values)),
        "minimum_pair_affinity": float(np.min(values)),
    }


def unique_complete_pareto_owner(candidate_rows):
    summaries = [complete_candidate_summary(row) for row in candidate_rows]
    if len(summaries) < 2:
        return None, "fewer_than_two_candidate_regions", summaries
    if any(not row["complete_edge_evidence"] for row in summaries):
        return None, "incomplete_candidate_edge_evidence", summaries

    winners = []
    for candidate in summaries:
        dominates_all = True
        for other in summaries:
            if other["proposal_id"] == candidate["proposal_id"]:
                continue
            mean_no_worse = (
                candidate["mean_pair_affinity"] >= other["mean_pair_affinity"]
            )
            minimum_no_worse = (
                candidate["minimum_pair_affinity"] >= other["minimum_pair_affinity"]
            )
            strictly_better = (
                candidate["mean_pair_affinity"] > other["mean_pair_affinity"]
                or candidate["minimum_pair_affinity"] > other["minimum_pair_affinity"]
            )
            if not (mean_no_worse and minimum_no_worse and strictly_better):
                dominates_all = False
                break
        if dominates_all:
            winners.append(candidate["proposal_id"])
    if len(winners) != 1:
        return None, "no_unique_complete_mean_min_pareto_owner", summaries
    return int(winners[0]), "unique_complete_mean_min_pareto_owner", summaries


def plan_boundary_superpoint(row):
    if row.get("assignment_action") != "none_preassignment_ledger_only":
        raise ValueError("preassignment row already contains an action")
    if row.get("gt_usage") != "none":
        raise ValueError("preassignment row used ground truth")
    if not bool(row.get("unknown_allowed")):
        raise ValueError("preassignment row does not allow unknown")
    current_owners = sorted(set(map(int, row["current_owner_proposal_ids"])))
    candidates = sorted(set(map(int, row["adjacent_candidate_proposal_ids"])))
    evidence = row.get("candidate_region_evidence", [])
    if candidates != sorted(int(item["proposal_id"]) for item in evidence):
        raise ValueError("candidate proposal/evidence identities differ")
    winner, evidence_state, summaries = unique_complete_pareto_owner(evidence)

    if winner is None:
        action = UNKNOWN_ACTION
        target_state = "unknown"
        remove_from = []
        add_to = []
    elif current_owners == [winner]:
        action = KEEP_ACTION
        target_state = "unique_candidate_owner"
        remove_from = []
        add_to = []
    else:
        action = ASSIGN_ACTION
        target_state = "unique_candidate_owner"
        remove_from = [item for item in current_owners if item != winner]
        add_to = [] if winner in current_owners else [winner]
        if not remove_from and not add_to:
            raise ValueError("unique owner assignment has no ownership delta")
    return {
        "superpoint_id": int(row["superpoint_id"]),
        "competition_state": str(row["competition_state"]),
        "current_owner_proposal_ids": current_owners,
        "adjacent_candidate_proposal_ids": candidates,
        "planned_owner_proposal_id": winner,
        "planned_target_state": target_state,
        "planned_action": action,
        "remove_from_proposal_ids": remove_from,
        "add_to_proposal_ids": add_to,
        "candidate_affinity_summaries": summaries,
        "evidence_state": evidence_state,
        "unknown_allowed": True,
        "assignment_applied": False,
        "proposal_mutation_applied": False,
        "score_used_for_decision": False,
        "decision_contract": DECISION_CONTRACT,
        "ground_truth_usage": "none",
    }


def build_scene_plan(rows):
    plans = [
        plan_boundary_superpoint(row)
        for row in sorted(rows, key=lambda item: int(item["superpoint_id"]))
    ]
    ids = [row["superpoint_id"] for row in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("boundary superpoint IDs are not unique")
    return plans


def _build_scene(scene_name, args):
    source_root = args.preassignment_root / scene_name
    source_summary = json.loads((source_root / "summary.json").read_text())
    if source_summary.get("depth_weight_adapter") != DEPTH_ADAPTER:
        raise ValueError(f"{scene_name} does not use the relative-depth adapter")
    if source_summary.get("ground_truth_usage") != "none":
        raise ValueError(f"{scene_name} preassignment ledger used ground truth")
    if int(source_summary.get("assignment_action_count", -1)) != 0:
        raise ValueError(f"{scene_name} preassignment ledger contains actions")
    rows = _read_jsonl(source_root / "boundary_competition.jsonl")
    if len(rows) != int(source_summary["boundary_competition_superpoint_count"]):
        raise ValueError(f"{scene_name} boundary row count differs")
    plans = build_scene_plan(rows)
    actions = Counter(row["planned_action"] for row in plans)
    evidence = Counter(row["evidence_state"] for row in plans)
    summary = {
        "scene_name": scene_name,
        "boundary_superpoint_count": len(plans),
        "unique_owner_assignment_plan_count": actions[ASSIGN_ACTION],
        "unique_owner_keep_plan_count": actions[KEEP_ACTION],
        "unknown_boundary_plan_count": actions[UNKNOWN_ACTION],
        "complete_pareto_owner_count": sum(
            row["planned_owner_proposal_id"] is not None for row in plans
        ),
        "evidence_state_counts": dict(sorted(evidence.items())),
        "assignment_applied_count": 0,
        "proposal_mutation_count": 0,
        "score_used_for_decision_count": 0,
        "decision_contract": DECISION_CONTRACT,
        "ground_truth_usage": "none",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "global_boundary_assignment_plan.jsonl", plans)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--preassignment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "preassignment_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("decision_contract") != DECISION_CONTRACT:
                raise SystemExit(f"resume contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: assign "
                f"{summary['unique_owner_assignment_plan_count']}, keep "
                f"{summary['unique_owner_keep_plan_count']}, unknown "
                f"{summary['unknown_boundary_plan_count']}",
                flush=True,
            )
        summaries.append(summary)

    sum_keys = (
        "boundary_superpoint_count", "unique_owner_assignment_plan_count",
        "unique_owner_keep_plan_count", "unknown_boundary_plan_count",
        "complete_pareto_owner_count", "assignment_applied_count",
        "proposal_mutation_count", "score_used_for_decision_count",
    )
    evidence = Counter()
    for row in summaries:
        evidence.update(row["evidence_state_counts"])
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in sum_keys
        },
        "evidence_state_counts": dict(sorted(evidence.items())),
        "decision_contract": DECISION_CONTRACT,
        "depth_weight_adapter": DEPTH_ADAPTER,
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "mv3dis_global_boundary_assignment_plan_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "boundary_superpoint_count",
        "unique_owner_assignment_plan_count", "unique_owner_keep_plan_count",
        "unknown_boundary_plan_count", "complete_pareto_owner_count",
        "assignment_applied_count", "proposal_mutation_count",
        "score_used_for_decision_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

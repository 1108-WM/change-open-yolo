#!/usr/bin/env python3
"""Plan parameter-free boundary actions for the Open-YOLO D2b structure.

This ``baseline_adapted`` stage is intentionally not presented as MV3DIS
paper reproduction.  A candidate strictly dominates only when every one of
its fully observed neighboring pair affinities is greater than every observed
pair affinity of every competing candidate.  Resolve, move, and grow actions
are separated into independent ablation families.  This tool writes a plan
only and never mutates a proposal.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def strict_all_edge_dominant_candidate(candidate_rows):
    """Return one dominant proposal only under complete strict range separation."""
    ranges = []
    for candidate in candidate_rows:
        pair_rows = candidate.get("pair_evidence", [])
        values = [row.get("pair_affinity") for row in pair_rows]
        if not pair_rows or any(value is None for value in values):
            return None, "incomplete_candidate_pair_evidence", []
        values = [float(value) for value in values]
        ranges.append({
            "proposal_id": int(candidate["proposal_id"]),
            "pair_affinity_min": min(values),
            "pair_affinity_max": max(values),
            "pair_affinity_count": len(values),
        })
    if len(ranges) < 2:
        return None, "fewer_than_two_candidates", ranges
    winners = []
    for candidate in ranges:
        other_max = max(
            row["pair_affinity_max"] for row in ranges
            if row["proposal_id"] != candidate["proposal_id"]
        )
        if candidate["pair_affinity_min"] > other_max:
            winners.append(candidate["proposal_id"])
    if len(winners) != 1:
        return None, "no_unique_strict_all_edge_dominance", ranges
    return int(winners[0]), "unique_strict_all_edge_dominance", ranges


def plan_boundary_row(row):
    current_owners = sorted(set(map(int, row["current_owner_proposal_ids"])))
    candidates = sorted(set(map(int, row["adjacent_candidate_proposal_ids"])))
    if len(candidates) != len(row["candidate_region_evidence"]):
        raise ValueError("candidate proposal/evidence counts differ")
    winner, dominance_state, ranges = strict_all_edge_dominant_candidate(
        row["candidate_region_evidence"]
    )
    action = "fallback_no_action"
    family = "fallback"
    remove_from = []
    add_to = []
    if winner is not None:
        state = row["competition_state"]
        if state == "current_multi_owner_conflict" and winner in current_owners:
            action = "resolve_multi_owner_to_dominant_current_owner"
            family = "resolve"
            remove_from = [item for item in current_owners if item != winner]
        elif state == "owned_boundary_competition" and winner not in current_owners:
            action = "move_owned_boundary_to_dominant_neighbor"
            family = "move"
            remove_from = list(current_owners)
            add_to = [winner]
        elif state == "unowned_between_regions" and not current_owners:
            action = "grow_unowned_boundary_to_dominant_neighbor"
            family = "grow"
            add_to = [winner]
        elif winner in current_owners:
            action = "keep_dominant_current_owner"
            family = "keep"
        else:
            dominance_state = "dominant_candidate_incompatible_with_ownership_state"
            winner = None
    return {
        "superpoint_id": int(row["superpoint_id"]),
        "competition_state": str(row["competition_state"]),
        "current_owner_proposal_ids": current_owners,
        "adjacent_candidate_proposal_ids": candidates,
        "dominant_proposal_id": winner,
        "dominance_state": dominance_state,
        "candidate_affinity_ranges": ranges,
        "planned_action": action,
        "ablation_family": family,
        "remove_from_proposal_ids": remove_from,
        "add_to_proposal_ids": add_to,
        "proposal_mutation_applied": False,
        "assignment_applied": False,
        "method_contract": (
            "baseline_adapted strict all-edge dominance; not MV3DIS paper region refinement"
        ),
        "gt_usage": "none",
    }


def build_scene_plan(boundary_rows):
    planned = [
        plan_boundary_row(row)
        for row in sorted(boundary_rows, key=lambda item: int(item["superpoint_id"]))
    ]
    ids = [row["superpoint_id"] for row in planned]
    if len(ids) != len(set(ids)):
        raise ValueError("boundary superpoint IDs are not unique")
    return planned


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--preassignment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "preassignment_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    scenes = _read_scenes(args.scene_list)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        source_summary = json.loads(
            (args.preassignment_root / scene_name / "summary.json").read_text()
        )
        if source_summary.get("depth_weight_adapter") != "mv3dis_relative_depth_rle_reference":
            raise ValueError(f"{scene_name} is not the relative-depth preassignment ledger")
        rows = build_scene_plan(
            _read_jsonl(args.preassignment_root / scene_name / "boundary_competition.jsonl")
        )
        counts = Counter(row["ablation_family"] for row in rows)
        actions = Counter(row["planned_action"] for row in rows)
        summary = {
            "scene_name": scene_name,
            "boundary_superpoint_count": len(rows),
            "ablation_family_counts": dict(counts),
            "planned_action_counts": dict(actions),
            "resolve_action_count": counts["resolve"],
            "move_action_count": counts["move"],
            "grow_action_count": counts["grow"],
            "proposal_mutation_count": 0,
            "assignment_action_count": 0,
            "ground_truth_usage": "none",
            "decision_state": (
                "baseline_adapted action plan only; resolve/move/grow remain separate and unapplied"
            ),
        }
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        _write_jsonl(scene_root / "assignment_plan.jsonl", rows)
        (scene_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        summaries.append(summary)
        print(
            f"[done] {index}/{len(scenes)} {scene_name}: resolve "
            f"{counts['resolve']}, move {counts['move']}, grow {counts['grow']}",
            flush=True,
        )
    keys = (
        "boundary_superpoint_count", "resolve_action_count", "move_action_count",
        "grow_action_count", "proposal_mutation_count", "assignment_action_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{key: sum(int(row[key]) for row in summaries) for key in keys},
        "method_contract": (
            "baseline_adapted strict all-edge dominance; not MV3DIS paper region refinement"
        ),
        "ablation_contract": "resolve, move, and grow must be materialized and evaluated separately",
        "ground_truth_usage": "none",
        "params": {key: value for key, value in vars(args).items()},
        "scene_summaries": summaries,
    }
    (args.output_root / "mv3dis_baseline_adapted_assignment_plan_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "boundary_superpoint_count", "resolve_action_count",
        "move_action_count", "grow_action_count", "proposal_mutation_count",
        "assignment_action_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

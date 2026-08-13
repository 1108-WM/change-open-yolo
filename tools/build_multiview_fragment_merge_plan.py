#!/usr/bin/env python3
"""Plan conservative multiview fragment merges without mutating D2b.

A pair is eligible only when it has bridge evidence in at least two independent
uniform30 frames, no same-frame separation counterexample, frozen raw-superpoint
contact, and no strict Details inclusion observation.  Among eligible pairs,
each proposal ranks partners only by bridge-frame count.  A merge is planned
only for mutual unique best partners; tied evidence always falls back.

The plan uses no GT, native proposal, class, semantic value, candidate quality
score, AP result, or tunable CLI threshold.  It applies no merge itself.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_BRIDGE_FRAMES = 2


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


def eligibility_reasons(row):
    reasons = []
    if int(row["bridge_frame_count"]) < MIN_BRIDGE_FRAMES:
        reasons.append("insufficient_bridge_frames")
    if int(row["separation_frame_count"]) > 0:
        reasons.append("separation_counterevidence")
    if not bool(row["has_spatial_contact"]):
        reasons.append("no_raw_superpoint_contact")
    if str(row["containment_direction"]) != "none":
        reasons.append("strict_inclusion_observed")
    return reasons


def _unique_best_partners(eligible_rows, proposal_ids):
    candidates = defaultdict(list)
    for row in eligible_rows:
        left = int(row["left_proposal_id"])
        right = int(row["right_proposal_id"])
        bridge_frames = int(row["bridge_frame_count"])
        candidates[left].append((right, bridge_frames))
        candidates[right].append((left, bridge_frames))

    result = {}
    for proposal_id in proposal_ids:
        rows = candidates.get(int(proposal_id), [])
        if not rows:
            result[int(proposal_id)] = {
                "unique_best_partner_id": None,
                "best_bridge_frame_count": 0,
                "best_partner_tie_count": 0,
            }
            continue
        best_count = max(count for _, count in rows)
        best_partners = sorted(partner for partner, count in rows if count == best_count)
        result[int(proposal_id)] = {
            "unique_best_partner_id": (
                best_partners[0] if len(best_partners) == 1 else None
            ),
            "best_bridge_frame_count": best_count,
            "best_partner_tie_count": len(best_partners),
        }
    return result


def build_fragment_merge_plan(scene_name, nodes, pair_rows):
    proposal_ids = [int(row["proposal_id"]) for row in nodes]
    if proposal_ids != sorted(proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("proposal nodes are not uniquely ordered")
    expected_pairs = len(proposal_ids) * (len(proposal_ids) - 1) // 2
    if len(pair_rows) != expected_pairs:
        raise ValueError("fragment relation ledger does not conserve all pairs")

    decisions = []
    eligible = []
    for row in pair_rows:
        reasons = eligibility_reasons(row)
        decision = {
            "scene_name": scene_name,
            "left_proposal_id": int(row["left_proposal_id"]),
            "right_proposal_id": int(row["right_proposal_id"]),
            "bridge_frame_count": int(row["bridge_frame_count"]),
            "bridge_observation_count": int(row["bridge_observation_count"]),
            "separation_frame_count": int(row["separation_frame_count"]),
            "has_spatial_contact": bool(row["has_spatial_contact"]),
            "containment_direction": str(row["containment_direction"]),
            "eligible_for_mutual_best": not reasons,
            "ineligibility_reasons": reasons,
            "merge_plan_state": "ineligible" if reasons else "eligible_not_selected",
            "merge_action_applied": False,
            "score_used_for_decision": False,
            "gt_usage": "none",
        }
        decisions.append(decision)
        if not reasons:
            eligible.append(decision)

    best = _unique_best_partners(eligible, proposal_ids)
    actions = []
    for row in eligible:
        left = int(row["left_proposal_id"])
        right = int(row["right_proposal_id"])
        mutual = (
            best[left]["unique_best_partner_id"] == right
            and best[right]["unique_best_partner_id"] == left
        )
        row["left_unique_best_partner_id"] = best[left]["unique_best_partner_id"]
        row["right_unique_best_partner_id"] = best[right]["unique_best_partner_id"]
        row["mutual_unique_best"] = mutual
        if mutual:
            row["merge_plan_state"] = "planned_mutual_unique_best"
            actions.append({
                "scene_name": scene_name,
                "action_index": len(actions),
                "anchor_proposal_id": left,
                "absorbed_proposal_id": right,
                "bridge_frame_count": int(row["bridge_frame_count"]),
                "bridge_observation_count": int(row["bridge_observation_count"]),
                "separation_frame_count": 0,
                "selection_contract": (
                    "at least two bridge frames; no separation; raw-superpoint "
                    "contact; no strict inclusion; mutual unique maximum bridge-frame count"
                ),
                "merge_action_applied": False,
                "score_used_for_decision": False,
                "gt_usage": "none",
            })

    action_ids = [
        item
        for row in actions
        for item in (int(row["anchor_proposal_id"]), int(row["absorbed_proposal_id"]))
    ]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("planned fragment actions are not proposal-disjoint")
    action_partner = {}
    for row in actions:
        left = int(row["anchor_proposal_id"])
        right = int(row["absorbed_proposal_id"])
        action_partner[left] = right
        action_partner[right] = left
    proposal_states = []
    for proposal_id in proposal_ids:
        state = best[proposal_id]
        proposal_states.append({
            "scene_name": scene_name,
            "proposal_id": proposal_id,
            **state,
            "planned_merge_partner_id": action_partner.get(proposal_id),
            "proposal_plan_state": (
                "planned_fragment_merge"
                if proposal_id in action_partner
                else "keep_frozen_d2b"
            ),
            "proposal_mutated": False,
            "score_used_for_decision": False,
            "gt_usage": "none",
        })
    return decisions, actions, proposal_states


def _build_scene(scene_name, args):
    ledger_scene = args.fragment_ledger_root / scene_name
    nodes = _read_jsonl(ledger_scene / "proposal_nodes.jsonl")
    pairs = _read_jsonl(ledger_scene / "fragment_pair_evidence.jsonl")
    decisions, actions, proposal_states = build_fragment_merge_plan(
        scene_name, nodes, pairs
    )
    reason_counts = Counter(
        reason for row in decisions for reason in row["ineligibility_reasons"]
    )
    summary = {
        "scene_name": scene_name,
        "proposal_count": len(nodes),
        "pair_decision_count": len(decisions),
        "eligible_pair_count": sum(row["eligible_for_mutual_best"] for row in decisions),
        "planned_merge_action_count": len(actions),
        "planned_proposal_count": 2 * len(actions),
        "proposal_plan_state_count": len(proposal_states),
        "ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        "minimum_bridge_frame_count": MIN_BRIDGE_FRAMES,
        "score_used_for_decision_count": 0,
        "proposal_mutation_count": 0,
        "merge_action_applied_count": 0,
        "ground_truth_usage": "none",
        "decision_state": "Fragment merge plan only; frozen D2b remains unchanged.",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "fragment_pair_decisions.jsonl", decisions)
        _write_jsonl(staging / "fragment_merge_actions.jsonl", actions)
        _write_jsonl(staging / "proposal_plan_states.jsonl", proposal_states)
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
    parser.add_argument("--fragment-ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "fragment_ledger_root", "output_root"):
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
            if summary.get("minimum_bridge_frame_count") != MIN_BRIDGE_FRAMES:
                raise SystemExit(f"resume contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: eligible "
                f"{summary['eligible_pair_count']}, planned "
                f"{summary['planned_merge_action_count']}",
                flush=True,
            )
        summaries.append(summary)

    additive_keys = (
        "proposal_count", "pair_decision_count", "eligible_pair_count",
        "planned_merge_action_count", "planned_proposal_count",
        "proposal_plan_state_count", "score_used_for_decision_count",
        "proposal_mutation_count", "merge_action_applied_count",
    )
    reasons = Counter()
    for row in summaries:
        reasons.update(row["ineligibility_reason_counts"])
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in additive_keys
        },
        "ineligibility_reason_counts": dict(sorted(reasons.items())),
        "minimum_bridge_frame_count": MIN_BRIDGE_FRAMES,
        "ranking_contract": "unique maximum bridge-frame count only",
        "score_used_for_decision_count": 0,
        "proposal_mutation_count": 0,
        "merge_action_applied_count": 0,
        "ground_truth_usage": "none",
        "decision_state": "Fragment merge plan only; frozen D2b remains unchanged.",
        "params": vars(args),
        "scene_summaries": summaries,
    }
    (args.output_root / "multiview_fragment_merge_plan_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "proposal_count": payload["proposal_count"],
        "eligible_pair_count": payload["eligible_pair_count"],
        "planned_merge_action_count": payload["planned_merge_action_count"],
        "score_used_for_decision_count": payload["score_used_for_decision_count"],
        "proposal_mutation_count": payload["proposal_mutation_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

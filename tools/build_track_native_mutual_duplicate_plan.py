#!/usr/bin/env python3
"""Plan conservative track suppression for native mutual duplicates.

The same GT-free geometry contract is applied to D2b and grow tracks.  An
appended track is planned for suppression only when at least one native mask
strictly covers more than 0.99 of the track and the track strictly covers
more than 0.99 of that native mask.  Scores are deliberately excluded because
native semantic-vote scores and automatic-SAM track quality are not calibrated
to the same scale.  This tool writes an action plan only and mutates nothing.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRICT_MUTUAL_COVERAGE = 0.99
SOURCE_VARIANT = "hierarchy_safe_d2b"
GROW_VARIANT = "mv3dis_baseline_adapted_grow"


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


def is_strict_mutual_duplicate(relation):
    """Use geometry only; scores and observed score relations are ignored."""
    return bool(
        float(relation["track_inside_native_ratio"]) > STRICT_MUTUAL_COVERAGE
        and float(relation["native_inside_track_ratio"]) > STRICT_MUTUAL_COVERAGE
    )


def choose_mutual_duplicate(relations):
    duplicates = [row for row in relations if is_strict_mutual_duplicate(row)]
    if not duplicates:
        return None
    return min(
        duplicates,
        key=lambda row: (-float(row["point_iou"]), int(row["native_candidate_id"])),
    )


def plan_proposal(proposal_summary, relations, geometry_variant):
    proposal_id = int(proposal_summary["proposal_id"])
    if any(int(row["proposal_id"]) != proposal_id for row in relations):
        raise ValueError(f"proposal {proposal_id} relation IDs differ")
    winner = choose_mutual_duplicate(relations)
    suppress = winner is not None
    return {
        "proposal_id": proposal_id,
        "geometry_variant": geometry_variant,
        "planned_action": (
            "suppress_appended_track_as_native_mutual_duplicate"
            if suppress else "keep_appended_track"
        ),
        "action_family": "mutual_duplicate_suppression" if suppress else "keep",
        "selected_native_candidate_id": (
            int(winner["native_candidate_id"]) if winner else None
        ),
        "selected_point_iou": float(winner["point_iou"]) if winner else None,
        "selected_track_inside_native_ratio": (
            float(winner["track_inside_native_ratio"]) if winner else None
        ),
        "selected_native_inside_track_ratio": (
            float(winner["native_inside_track_ratio"]) if winner else None
        ),
        "strict_mutual_duplicate_relation_count": sum(
            is_strict_mutual_duplicate(row) for row in relations
        ),
        "overlap_native_candidate_count": int(
            proposal_summary["overlap_native_candidate_count"]
        ),
        "score_used_for_decision": False,
        "ground_truth_usage": "none",
        "track_suppression_applied": False,
        "native_candidate_mutation_applied": False,
        "decision_contract": "strict bidirectional point coverage > 0.99",
        "method_contract": (
            "baseline_adapted conservative mutual-duplicate plan; "
            "not Details score-based inclusion cleanup"
        ),
    }


def _load_relations(path, expected_variant, expected_count):
    by_proposal = defaultdict(list)
    count = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if row.get("geometry_variant") != expected_variant:
                raise ValueError(f"unexpected geometry variant at {path}")
            if row.get("ground_truth_usage") != "none":
                raise ValueError(f"relation used ground truth at {path}")
            by_proposal[int(row["proposal_id"])].append(row)
    if count != int(expected_count):
        raise ValueError(f"nonzero relation count differs at {path}")
    return by_proposal


def _plan_variant(comparisons, relations_by_proposal, side, geometry_variant):
    plans = []
    seen = set()
    summary_key = f"{side}_native_relation"
    for comparison in sorted(comparisons, key=lambda row: int(row["proposal_id"])):
        proposal_id = int(comparison["proposal_id"])
        if proposal_id in seen:
            raise ValueError(f"duplicate proposal ID {proposal_id}")
        seen.add(proposal_id)
        summary = comparison[summary_key]
        relations = relations_by_proposal.get(proposal_id, [])
        if len(relations) != int(summary["overlap_native_candidate_count"]):
            raise ValueError(f"proposal {proposal_id} overlap relation count differs")
        plans.append(plan_proposal(summary, relations, geometry_variant))
    return plans


def compare_variant_plans(source_plans, grow_plans):
    source = {int(row["proposal_id"]): row for row in source_plans}
    grow = {int(row["proposal_id"]): row for row in grow_plans}
    if source.keys() != grow.keys():
        raise ValueError("source/grow proposal IDs differ")
    rows = []
    for proposal_id in sorted(source):
        source_suppress = source[proposal_id]["action_family"] == "mutual_duplicate_suppression"
        grow_suppress = grow[proposal_id]["action_family"] == "mutual_duplicate_suppression"
        state = (
            "shared_suppression" if source_suppress and grow_suppress else
            "source_only_suppression" if source_suppress else
            "grow_only_suppression" if grow_suppress else
            "shared_keep"
        )
        rows.append({
            "proposal_id": proposal_id,
            "source_planned_action": source[proposal_id]["planned_action"],
            "grow_planned_action": grow[proposal_id]["planned_action"],
            "cross_variant_action_state": state,
            "ground_truth_usage": "none",
            "action_applied": False,
        })
    return rows


def _build_scene(scene_name, args):
    root = args.competition_ledger_root / scene_name
    source_summary = json.loads((root / "summary.json").read_text())
    if source_summary.get("ground_truth_usage") != "none":
        raise ValueError(f"{scene_name} competition ledger used ground truth")
    if int(source_summary.get("candidate_action_count", -1)) != 0:
        raise ValueError(f"{scene_name} competition ledger already contains actions")
    comparisons = _read_jsonl(root / "proposal_competition_comparison.jsonl")
    if len(comparisons) != int(source_summary["proposal_count"]):
        raise ValueError(f"{scene_name} proposal comparison count differs")
    source_relations = _load_relations(
        root / "source_track_native_relations.jsonl",
        SOURCE_VARIANT,
        source_summary["source_nonzero_relation_count"],
    )
    grow_relations = _load_relations(
        root / "grow_track_native_relations.jsonl",
        GROW_VARIANT,
        source_summary["grow_nonzero_relation_count"],
    )
    source_plans = _plan_variant(
        comparisons, source_relations, "source", SOURCE_VARIANT
    )
    grow_plans = _plan_variant(comparisons, grow_relations, "grow", GROW_VARIANT)
    comparison_plans = compare_variant_plans(source_plans, grow_plans)
    transition_counts = Counter(
        row["cross_variant_action_state"] for row in comparison_plans
    )
    summary = {
        "scene_name": scene_name,
        "proposal_count": len(comparisons),
        "source_suppression_plan_count": sum(
            row["action_family"] == "mutual_duplicate_suppression"
            for row in source_plans
        ),
        "grow_suppression_plan_count": sum(
            row["action_family"] == "mutual_duplicate_suppression"
            for row in grow_plans
        ),
        "cross_variant_action_state_counts": dict(transition_counts),
        "score_used_for_decision_count": 0,
        "ground_truth_usage": "none",
        "track_suppression_applied_count": 0,
        "native_candidate_mutation_count": 0,
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "source_competition_plan.jsonl", source_plans)
        _write_jsonl(staging / "grow_competition_plan.jsonl", grow_plans)
        _write_jsonl(staging / "cross_variant_plan_comparison.jsonl", comparison_plans)
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
    parser.add_argument("--competition-ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "competition_ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[done] {index}/{len(scenes)} {scene_name}: source suppress "
            f"{summary['source_suppression_plan_count']}, grow suppress "
            f"{summary['grow_suppression_plan_count']}",
            flush=True,
        )
    transition_counts = sum(
        (Counter(row["cross_variant_action_state_counts"]) for row in summaries),
        Counter(),
    )
    sum_keys = (
        "proposal_count", "source_suppression_plan_count",
        "grow_suppression_plan_count", "score_used_for_decision_count",
        "track_suppression_applied_count", "native_candidate_mutation_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{key: sum(int(row[key]) for row in summaries) for key in sum_keys},
        "cross_variant_action_state_counts": dict(transition_counts),
        "coverage_contract": "strict bidirectional point coverage > 0.99",
        "score_contract": "scores observed in source ledger but excluded from decisions",
        "method_contract": (
            "baseline_adapted conservative mutual-duplicate plan; "
            "not Details score-based inclusion cleanup"
        ),
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "track_native_mutual_duplicate_plan_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "proposal_count", "source_suppression_plan_count",
        "grow_suppression_plan_count", "cross_variant_action_state_counts",
        "score_used_for_decision_count", "track_suppression_applied_count",
        "native_candidate_mutation_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

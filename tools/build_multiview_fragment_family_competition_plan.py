#!/usr/bin/env python3
"""Plan conservative A/B/M fragment-family competition from a frozen ledger.

M replaces A+B only when at least two common non-bridge frames are available
and M Pareto-dominates A and B independently on mean best sIoU and support
frame rate. Every undefined, tied, or mixed case falls back to A+B.

This tool reads no observations, GT, native predictions, semantics, scores, or
AP results. It writes a plan only and mutates no candidate.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USE_MERGED_ACTION = "use_merged_candidate"
KEEP_ORIGINAL_PAIR_ACTION = "keep_original_pair"
DECISION_CONTRACT = (
    "M replaces A+B only with reliable common non-bridge evidence and "
    "independent Pareto dominance over both A and B"
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


def plan_family(row):
    if row.get("candidate_action") != "none_ledger_only":
        raise ValueError("family quality ledger already contains an action")
    if row.get("ground_truth_usage") != "none":
        raise ValueError("family quality ledger used ground truth")
    if bool(row.get("score_used_for_decision")):
        raise ValueError("family quality ledger used a score")
    reliable = bool(row["quality_evidence_reliable"])
    jointly_dominant = bool(row["merged_jointly_dominates"])
    if jointly_dominant and not reliable:
        raise ValueError("unreliable family cannot contain joint dominance")
    if jointly_dominant != bool(
        row["merged_dominates_anchor"]
        and row["merged_dominates_absorbed"]
    ):
        raise ValueError("joint dominance does not match pairwise facts")

    if not reliable:
        action = KEEP_ORIGINAL_PAIR_ACTION
        state = "fallback_insufficient_nonbridge_frames"
    elif jointly_dominant:
        action = USE_MERGED_ACTION
        state = "merged_jointly_pareto_dominant"
    else:
        action = KEEP_ORIGINAL_PAIR_ACTION
        state = "fallback_no_joint_pareto_dominance"
    return {
        "scene_name": str(row["scene_name"]),
        "action_index": int(row["action_index"]),
        "anchor_proposal_id": int(row["anchor_proposal_id"]),
        "absorbed_proposal_id": int(row["absorbed_proposal_id"]),
        "planned_action": action,
        "planning_state": state,
        "common_nonbridge_visible_frame_count": int(
            row["common_nonbridge_visible_frame_count"]
        ),
        "quality_evidence_reliable": reliable,
        "merged_dominates_anchor": bool(row["merged_dominates_anchor"]),
        "merged_dominates_absorbed": bool(row["merged_dominates_absorbed"]),
        "merged_jointly_dominates": jointly_dominant,
        "decision_contract": DECISION_CONTRACT,
        "score_used_for_decision": False,
        "candidate_action_applied": False,
        "ground_truth_usage": "none",
    }


def _build_scene(scene_name, args):
    ledger_summary = json.loads(
        (args.family_quality_root / scene_name / "summary.json").read_text()
    )
    if ledger_summary.get("ground_truth_usage") != "none":
        raise ValueError(f"{scene_name} family ledger used ground truth")
    if int(ledger_summary.get("candidate_action_count", -1)) != 0:
        raise ValueError(f"{scene_name} family ledger already contains actions")
    rows = _read_jsonl(
        args.family_quality_root / scene_name / "fragment_family_quality.jsonl"
    )
    if len(rows) != int(ledger_summary["family_count"]):
        raise ValueError(f"{scene_name} family count differs")
    plans = [plan_family(row) for row in rows]
    identities = [
        (int(row["anchor_proposal_id"]), int(row["absorbed_proposal_id"]))
        for row in plans
    ]
    involved = [item for pair in identities for item in pair]
    if len(involved) != len(set(involved)):
        raise ValueError(f"{scene_name} family plans are not proposal-disjoint")
    if [row["action_index"] for row in plans] != list(range(len(plans))):
        raise ValueError(f"{scene_name} family action indices are not deterministic")

    summary = {
        "scene_name": scene_name,
        "family_count": len(plans),
        "use_merged_plan_count": sum(
            row["planned_action"] == USE_MERGED_ACTION for row in plans
        ),
        "keep_original_pair_plan_count": sum(
            row["planned_action"] == KEEP_ORIGINAL_PAIR_ACTION for row in plans
        ),
        "insufficient_evidence_fallback_count": sum(
            row["planning_state"] == "fallback_insufficient_nonbridge_frames"
            for row in plans
        ),
        "no_joint_dominance_fallback_count": sum(
            row["planning_state"] == "fallback_no_joint_pareto_dominance"
            for row in plans
        ),
        "candidate_action_applied_count": 0,
        "candidate_mutation_count": 0,
        "score_used_for_decision_count": 0,
        "ground_truth_usage": "none",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "fragment_family_plan.jsonl", plans)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def _validate_manifest(root, expected_scene_count):
    path = root / "multiview_fragment_family_quality_summary.json"
    payload = json.loads(path.read_text())
    if int(payload.get("scene_count", 0)) != int(expected_scene_count):
        raise ValueError("family quality manifest scene count differs")
    if payload.get("ground_truth_usage") != "none":
        raise ValueError("family quality manifest used ground truth")
    if int(payload.get("candidate_action_count", -1)) != 0:
        raise ValueError("family quality manifest already contains actions")
    return path, payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--family-quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "family_quality_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list)
    manifest_path, manifest = _validate_manifest(
        args.family_quality_root, len(all_scenes)
    )
    scenes = all_scenes
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
            f"[done] {index}/{len(scenes)} {scene_name}: use merged "
            f"{summary['use_merged_plan_count']}/{summary['family_count']}",
            flush=True,
        )
    additive_keys = (
        "family_count", "use_merged_plan_count",
        "keep_original_pair_plan_count", "insufficient_evidence_fallback_count",
        "no_joint_dominance_fallback_count", "candidate_action_applied_count",
        "candidate_mutation_count", "score_used_for_decision_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in additive_keys
        },
        "decision_contract": DECISION_CONTRACT,
        "source_family_count": int(manifest["family_count"]),
        "family_quality_manifest": str(manifest_path),
        "candidate_action_applied_count": 0,
        "candidate_mutation_count": 0,
        "score_used_for_decision_count": 0,
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "multiview_fragment_family_competition_plan_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "family_count", "use_merged_plan_count",
        "keep_original_pair_plan_count", "insufficient_evidence_fallback_count",
        "no_joint_dominance_fallback_count", "candidate_action_applied_count",
        "candidate_mutation_count", "score_used_for_decision_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

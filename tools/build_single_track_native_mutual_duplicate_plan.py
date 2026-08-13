#!/usr/bin/env python3
"""Plan strict native mutual-duplicate suppression for one track variant."""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_track_native_competition_ledger import _read_scenes, _resolve, _write_jsonl
from tools.build_track_native_mutual_duplicate_plan import plan_proposal


COVERAGE_CONTRACT = "strict bidirectional point coverage > 0.99"


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_manifest(root, expected_scene_count):
    path = root / "single_track_native_competition_ledger_summary.json"
    payload = json.loads(path.read_text())
    if int(payload.get("scene_count", 0)) != int(expected_scene_count):
        raise ValueError("competition ledger scene count differs")
    if payload.get("ground_truth_usage") != "none":
        raise ValueError("competition ledger used ground truth")
    if int(payload.get("candidate_action_count", -1)) != 0:
        raise ValueError("competition ledger already contains candidate actions")
    variant = payload.get("geometry_variant")
    if not variant:
        raise ValueError("competition ledger lacks geometry variant")
    return path, payload


def _build_scene(scene_name, args):
    root = args.competition_ledger_root / scene_name
    scene_summary = json.loads((root / "summary.json").read_text())
    if scene_summary.get("geometry_variant") != args.geometry_variant:
        raise ValueError(f"{scene_name} geometry variant differs")
    if scene_summary.get("ground_truth_usage") != "none":
        raise ValueError(f"{scene_name} competition ledger used ground truth")
    if int(scene_summary.get("candidate_action_count", -1)) != 0:
        raise ValueError(f"{scene_name} competition ledger contains actions")
    proposal_summaries = _read_jsonl(root / "proposal_native_summaries.jsonl")
    if len(proposal_summaries) != int(scene_summary["proposal_count"]):
        raise ValueError(f"{scene_name} proposal summary count differs")
    relations_by_proposal = defaultdict(list)
    relation_count = 0
    for row in _read_jsonl(root / "track_native_relations.jsonl"):
        if row.get("geometry_variant") != args.geometry_variant:
            raise ValueError(f"{scene_name} relation geometry variant differs")
        if row.get("ground_truth_usage") != "none":
            raise ValueError(f"{scene_name} relation used ground truth")
        relations_by_proposal[int(row["proposal_id"])].append(row)
        relation_count += 1
    if relation_count != int(scene_summary["nonzero_relation_count"]):
        raise ValueError(f"{scene_name} relation count differs")

    plans = []
    for summary in proposal_summaries:
        proposal_id = int(summary["proposal_id"])
        relations = relations_by_proposal.get(proposal_id, [])
        if len(relations) != int(summary["overlap_native_candidate_count"]):
            raise ValueError(f"{scene_name} proposal {proposal_id} relation count differs")
        plans.append(plan_proposal(summary, relations, args.geometry_variant))
    if [int(row["proposal_id"]) for row in plans] != sorted(
        int(row["proposal_id"]) for row in plans
    ):
        raise ValueError(f"{scene_name} proposal plan order differs")
    summary = {
        "scene_name": scene_name,
        "geometry_variant": args.geometry_variant,
        "proposal_count": len(plans),
        "suppression_plan_count": sum(
            row["action_family"] == "mutual_duplicate_suppression"
            for row in plans
        ),
        "keep_plan_count": sum(row["action_family"] == "keep" for row in plans),
        "score_used_for_decision_count": 0,
        "track_suppression_applied_count": 0,
        "native_candidate_mutation_count": 0,
        "ground_truth_usage": "none",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "competition_plan.jsonl", plans)
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
    all_scenes = _read_scenes(args.scene_list)
    manifest_path, manifest = _load_manifest(
        args.competition_ledger_root, len(all_scenes)
    )
    args.geometry_variant = manifest["geometry_variant"]
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
            f"[done] {index}/{len(scenes)} {scene_name}: suppress "
            f"{summary['suppression_plan_count']}/{summary['proposal_count']}",
            flush=True,
        )
    sum_keys = (
        "proposal_count", "suppression_plan_count", "keep_plan_count",
        "score_used_for_decision_count", "track_suppression_applied_count",
        "native_candidate_mutation_count",
    )
    payload = {
        "scene_count": len(summaries),
        "geometry_variant": args.geometry_variant,
        **{key: sum(int(row[key]) for row in summaries) for key in sum_keys},
        "coverage_contract": COVERAGE_CONTRACT,
        "score_contract": "scores observed in ledger but excluded from decisions",
        "method_contract": (
            "baseline_adapted conservative mutual-duplicate plan; "
            "not Details score-based inclusion cleanup"
        ),
        "ground_truth_usage": "none",
        "competition_ledger_manifest": str(manifest_path),
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "track_native_mutual_duplicate_plan_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "proposal_count", "suppression_plan_count",
        "keep_plan_count", "score_used_for_decision_count",
        "track_suppression_applied_count", "native_candidate_mutation_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

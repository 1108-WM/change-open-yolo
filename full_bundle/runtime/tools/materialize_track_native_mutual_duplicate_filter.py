#!/usr/bin/env python3
"""Materialize one frozen track/native mutual-duplicate filter plan.

Only appended tracks planned as strict bidirectional-coverage duplicates are
removed.  Every retained track dictionary, including geometry, point path,
lineage, and scores, is copied without modification.  Native predictions are
not an input and cannot be mutated by this tool.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_SIDES = ("source", "grow")
EXPECTED_VARIANTS = {
    "source": "hierarchy_safe_d2b",
    "grow": "mv3dis_baseline_adapted_grow",
}
KEEP_ACTION = "keep_appended_track"
SUPPRESS_ACTION = "suppress_appended_track_as_native_mutual_duplicate"
DECISION_CONTRACT = "strict bidirectional point coverage > 0.99"


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


def _validate_source_tracks(tracks):
    proposal_ids = [int(row["proposal_id"]) for row in tracks]
    if proposal_ids != sorted(proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("source proposal IDs are invalid")
    for track in tracks:
        proposal_id = int(track["proposal_id"])
        if int(track.get("track_id", proposal_id)) != proposal_id:
            raise ValueError(f"proposal/track identity differs: {proposal_id}")
        point_path = Path(track["points_path"])
        if not point_path.is_file():
            raise FileNotFoundError(f"proposal {proposal_id} point file is missing")
        with np.load(point_path) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if not len(points) or len(points) != int(track.get("point_count", len(points))):
            raise ValueError(f"proposal {proposal_id} point file differs from metadata")
    return proposal_ids


def _expected_variant(plan_side=None, geometry_variant=None):
    if (plan_side is None) == (geometry_variant is None):
        raise ValueError("provide exactly one of plan_side or geometry_variant")
    if geometry_variant is not None:
        if not str(geometry_variant).strip():
            raise ValueError("geometry_variant must be non-empty")
        return str(geometry_variant)
    if plan_side not in PLAN_SIDES:
        raise ValueError(f"unknown plan side: {plan_side}")
    return EXPECTED_VARIANTS[plan_side]


def filter_tracks(source_tracks, plan_rows, plan_side=None, geometry_variant=None):
    """Return exact retained records plus a one-to-one filter ledger."""
    expected_variant = _expected_variant(plan_side, geometry_variant)
    source_ids = [int(row["proposal_id"]) for row in source_tracks]
    plan_ids = [int(row["proposal_id"]) for row in plan_rows]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("plan proposal IDs contain duplicates")
    if plan_ids != source_ids:
        raise ValueError("plan/source proposal count, order, or IDs differ")

    retained = []
    ledger = []
    actions = []
    for source, plan in zip(source_tracks, plan_rows):
        proposal_id = int(source["proposal_id"])
        if plan.get("geometry_variant") != expected_variant:
            raise ValueError(f"proposal {proposal_id} geometry variant differs")
        if plan.get("decision_contract") != DECISION_CONTRACT:
            raise ValueError(f"proposal {proposal_id} decision contract differs")
        if plan.get("ground_truth_usage") != "none":
            raise ValueError(f"proposal {proposal_id} plan used ground truth")
        if bool(plan.get("score_used_for_decision")):
            raise ValueError(f"proposal {proposal_id} plan used a score")
        if bool(plan.get("track_suppression_applied")):
            raise ValueError(f"proposal {proposal_id} plan was already applied")
        action = plan.get("planned_action")
        if action not in (KEEP_ACTION, SUPPRESS_ACTION):
            raise ValueError(f"proposal {proposal_id} has unknown action: {action}")
        suppressed = action == SUPPRESS_ACTION
        if suppressed:
            if plan.get("action_family") != "mutual_duplicate_suppression":
                raise ValueError(f"proposal {proposal_id} suppression family differs")
            if int(plan.get("strict_mutual_duplicate_relation_count", 0)) <= 0:
                raise ValueError(f"proposal {proposal_id} lacks a mutual duplicate")
            if plan.get("selected_native_candidate_id") is None:
                raise ValueError(f"proposal {proposal_id} lacks selected native candidate")
            actions.append({
                "proposal_id": proposal_id,
                "track_id": int(source.get("track_id", proposal_id)),
                "lineage_proposal_ids": list(
                    source.get("lineage_proposal_ids", [proposal_id])
                ),
                "materialization_action": SUPPRESS_ACTION,
                "selected_native_candidate_id": int(
                    plan["selected_native_candidate_id"]
                ),
                "selected_point_iou": float(plan["selected_point_iou"]),
                "selected_track_inside_native_ratio": float(
                    plan["selected_track_inside_native_ratio"]
                ),
                "selected_native_inside_track_ratio": float(
                    plan["selected_native_inside_track_ratio"]
                ),
                "score_used_for_decision": False,
                "ground_truth_usage": "none",
                "track_suppression_applied": True,
                "native_candidate_mutation_applied": False,
            })
        else:
            if plan.get("action_family") != "keep":
                raise ValueError(f"proposal {proposal_id} keep family differs")
            retained.append(source)
        ledger.append({
            "proposal_id": proposal_id,
            "track_id": int(source.get("track_id", proposal_id)),
            "lineage_proposal_ids": list(
                source.get("lineage_proposal_ids", [proposal_id])
            ),
            "source_point_count": int(source.get("point_count", 0)),
            "source_points_path": str(source["points_path"]),
            "planned_action": action,
            "retained": not suppressed,
            "track_suppression_applied": suppressed,
            "retained_track_field_mutation_count": 0,
            "native_candidate_mutation_applied": False,
            "score_used_for_decision": False,
            "ground_truth_usage": "none",
        })

    retained_by_id = {int(row["proposal_id"]): row for row in retained}
    for source in source_tracks:
        proposal_id = int(source["proposal_id"])
        if proposal_id in retained_by_id and retained_by_id[proposal_id] != source:
            raise ValueError(f"retained proposal {proposal_id} changed fields")
    if len(retained) + len(actions) != len(source_tracks):
        raise ValueError("source proposals are not conserved by keep/suppress partition")
    return retained, actions, ledger


def _validate_plan_manifest(
    plan_root, expected_scene_count, plan_side=None, geometry_variant=None
):
    path = plan_root / "track_native_mutual_duplicate_plan_summary.json"
    payload = json.loads(path.read_text())
    if int(payload.get("scene_count", 0)) != int(expected_scene_count):
        raise ValueError("plan scene count differs from requested split")
    if payload.get("coverage_contract") != DECISION_CONTRACT:
        raise ValueError("plan coverage contract differs")
    if payload.get("ground_truth_usage") != "none":
        raise ValueError("plan manifest used ground truth")
    if int(payload.get("track_suppression_applied_count", -1)) != 0:
        raise ValueError("plan manifest already applied suppression")
    expected_variant = _expected_variant(plan_side, geometry_variant)
    manifest_variant = payload.get("geometry_variant")
    if geometry_variant is not None and manifest_variant != expected_variant:
        raise ValueError("plan manifest geometry variant differs")
    return path


def _build_scene(scene_name, args):
    source_payload = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    source_tracks = source_payload.get("tracks", [])
    if int(source_payload.get("track_count", len(source_tracks))) != len(source_tracks):
        raise ValueError(f"{scene_name} source track count differs")
    _validate_source_tracks(source_tracks)
    plan_path = args.plan_root / scene_name / (
        f"{args.plan_side}_competition_plan.jsonl"
        if args.plan_side is not None else "competition_plan.jsonl"
    )
    plan_rows = _read_jsonl(plan_path)
    retained, actions, ledger = filter_tracks(
        source_tracks,
        plan_rows,
        plan_side=args.plan_side,
        geometry_variant=args.geometry_variant,
    )
    expected_variant = _expected_variant(args.plan_side, args.geometry_variant)
    summary = {
        "scene_name": scene_name,
        "plan_side": args.plan_side or "single_variant",
        "geometry_variant": expected_variant,
        "source_proposal_count": len(source_tracks),
        "final_proposal_count": len(retained),
        "suppressed_proposal_count": len(actions),
        "retained_proposal_count": len(retained),
        "retained_track_field_mutation_count": 0,
        "retained_geometry_mutation_count": 0,
        "retained_score_mutation_count": 0,
        "retained_lineage_mutation_count": 0,
        "native_candidate_mutation_count": 0,
        "score_used_for_decision_count": 0,
        "ground_truth_usage": "none",
        "decision_contract": DECISION_CONTRACT,
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        output_payload = dict(source_payload)
        output_payload.update({
            "track_count": len(retained),
            "tracks": retained,
            "source_track_count": len(source_tracks),
            "track_native_competition_materialization": (
                "baseline_adapted_mutual_duplicate_filter"
            ),
        })
        (staging / "automatic_tracks.json").write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _write_jsonl(staging / "suppression_actions.jsonl", actions)
        _write_jsonl(staging / "proposal_filter_ledger.jsonl", ledger)
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
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    variant = parser.add_mutually_exclusive_group(required=True)
    variant.add_argument("--plan-side", choices=PLAN_SIDES)
    variant.add_argument("--geometry-variant")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "track_root", "plan_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list)
    plan_manifest = _validate_plan_manifest(
        args.plan_root,
        len(all_scenes),
        plan_side=args.plan_side,
        geometry_variant=args.geometry_variant,
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
            f"[done] {index}/{len(scenes)} {scene_name}: retained "
            f"{summary['final_proposal_count']}/{summary['source_proposal_count']}, "
            f"suppressed {summary['suppressed_proposal_count']}",
            flush=True,
        )
    sum_keys = (
        "source_proposal_count", "final_proposal_count",
        "suppressed_proposal_count", "retained_proposal_count",
        "retained_track_field_mutation_count", "retained_geometry_mutation_count",
        "retained_score_mutation_count", "retained_lineage_mutation_count",
        "native_candidate_mutation_count", "score_used_for_decision_count",
    )
    payload = {
        "scene_count": len(summaries),
        "plan_side": args.plan_side or "single_variant",
        "geometry_variant": _expected_variant(args.plan_side, args.geometry_variant),
        **{key: sum(int(row[key]) for row in summaries) for key in sum_keys},
        "decision_contract": DECISION_CONTRACT,
        "method_contract": (
            "baseline_adapted mutual-duplicate track filter; native unchanged"
        ),
        "ground_truth_usage": "none",
        "plan_manifest": str(plan_manifest),
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "track_native_mutual_duplicate_filter_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "plan_side", "source_proposal_count",
        "final_proposal_count", "suppressed_proposal_count",
        "retained_track_field_mutation_count", "native_candidate_mutation_count",
        "score_used_for_decision_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

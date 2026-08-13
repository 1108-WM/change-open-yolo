#!/usr/bin/env python3
"""Materialize frozen A/B/M fragment-family competition decisions.

The complete D2b proposal set is the source.  For every family planned as
``use_merged_candidate``, the F1 proposal M at the anchor ID replaces A and
the absorbed proposal B is removed.  Every fallback family keeps the exact
D2b A and B records.  Proposals outside all families are also copied exactly
from D2b.  Only the output ``points_path`` is rewritten.

This tool does not read observations, GT, native predictions, semantics,
classes, scores, or AP results.  It does not refine candidates or recompute
scores.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


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


def _ordered_unique_ints(row, key):
    values = list(map(int, row.get(key, [])))
    if not values or values != sorted(set(values)):
        raise ValueError(f"proposal {row.get('proposal_id')} has invalid {key}")
    return values


def _validate_tracks(tracks, label, validate_point_files=False):
    proposal_ids = [int(row["proposal_id"]) for row in tracks]
    if proposal_ids != sorted(proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError(f"{label} proposal IDs are invalid")
    for row in tracks:
        proposal_id = int(row["proposal_id"])
        if int(row.get("track_id", proposal_id)) != proposal_id:
            raise ValueError(f"{label} proposal {proposal_id} track identity differs")
        _ordered_unique_ints(row, "lineage_proposal_ids")
        _ordered_unique_ints(row, "observation_ids")
        _ordered_unique_ints(row, "superpoint_ids")
        if int(row.get("superpoint_count", -1)) != len(row["superpoint_ids"]):
            raise ValueError(f"{label} proposal {proposal_id} superpoint count differs")
        if validate_point_files:
            point_path = Path(row["points_path"])
            if not point_path.is_file():
                raise FileNotFoundError(
                    f"{label} proposal {proposal_id} point file is missing"
                )
            with np.load(point_path) as payload:
                points = np.asarray(payload["point_indices"], dtype=np.int64)
            if len(points) == 0 or len(points) != len(np.unique(points)):
                raise ValueError(f"{label} proposal {proposal_id} point file is invalid")
            if int(row.get("point_count", -1)) != len(points):
                raise ValueError(f"{label} proposal {proposal_id} point count differs")
    return proposal_ids


def _validate_plan_row(row, expected_index):
    if int(row["action_index"]) != expected_index:
        raise ValueError("family plan action indices are not deterministic")
    if row.get("decision_contract") != DECISION_CONTRACT:
        raise ValueError("family plan decision contract differs")
    if row.get("ground_truth_usage") != "none":
        raise ValueError("family plan used ground truth")
    if bool(row.get("score_used_for_decision")):
        raise ValueError("family plan used a score")
    if bool(row.get("candidate_action_applied")):
        raise ValueError("family plan action was already applied")
    action = row.get("planned_action")
    if action not in (USE_MERGED_ACTION, KEEP_ORIGINAL_PAIR_ACTION):
        raise ValueError(f"unknown family action: {action}")
    anchor_id = int(row["anchor_proposal_id"])
    absorbed_id = int(row["absorbed_proposal_id"])
    if anchor_id >= absorbed_id:
        raise ValueError("family anchor must be the lower proposal ID")
    if action == USE_MERGED_ACTION:
        if not bool(row.get("quality_evidence_reliable")):
            raise ValueError("merged action lacks reliable evidence")
        if not bool(row.get("merged_jointly_dominates")):
            raise ValueError("merged action lacks joint Pareto dominance")
    return anchor_id, absorbed_id, action


def _union_field(anchor, absorbed, key):
    return sorted(set(map(int, anchor[key])) | set(map(int, absorbed[key])))


def materialize_family_competition(source_tracks, merged_tracks, plan_rows):
    """Select exact D2b or F1 records according to a frozen family plan."""
    source_ids = _validate_tracks(source_tracks, "source")
    _validate_tracks(merged_tracks, "merged")
    source_by_id = {int(row["proposal_id"]): row for row in source_tracks}
    merged_by_id = {int(row["proposal_id"]): row for row in merged_tracks}

    family_ids = []
    parsed = []
    for index, row in enumerate(plan_rows):
        anchor_id, absorbed_id, action = _validate_plan_row(row, index)
        if str(row.get("scene_name", "")).strip() == "":
            raise ValueError("family plan lacks scene name")
        family_ids.extend((anchor_id, absorbed_id))
        parsed.append((row, anchor_id, absorbed_id, action))
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("family plan is not proposal-disjoint")
    if set(family_ids) - set(source_ids):
        raise ValueError("family plan references unknown D2b proposals")

    selected = dict(source_by_id)
    ledger = []
    for row, anchor_id, absorbed_id, action in parsed:
        anchor = source_by_id[anchor_id]
        absorbed = source_by_id[absorbed_id]
        merged = merged_by_id.get(anchor_id)
        if merged is None or absorbed_id in merged_by_id:
            raise ValueError("F1 source does not contain the expected merged family")
        for key in ("lineage_proposal_ids", "observation_ids", "source_track_ids"):
            if list(map(int, merged.get(key, []))) != _union_field(anchor, absorbed, key):
                raise ValueError(
                    f"F1 proposal {anchor_id} does not conserve family {key}"
                )

        use_merged = action == USE_MERGED_ACTION
        if use_merged:
            selected[anchor_id] = merged
            del selected[absorbed_id]
            state = "selected_exact_f1_merged_candidate"
        else:
            state = "retained_exact_d2b_original_pair"
        ledger.append({
            **row,
            "materialization_state": state,
            "selected_proposal_ids": [anchor_id] if use_merged else [anchor_id, absorbed_id],
            "merged_candidate_selected": use_merged,
            "absorbed_proposal_removed": use_merged,
            "candidate_action_applied": use_merged,
            "candidate_field_mutation_count": 0,
            "score_recomputed": False,
            "score_used_for_decision": False,
            "ground_truth_usage": "none",
        })

    final = [selected[item] for item in sorted(selected)]
    selected_merged_count = sum(
        action == USE_MERGED_ACTION for _, _, _, action in parsed
    )
    if len(final) != len(source_tracks) - selected_merged_count:
        raise ValueError("final proposal count does not match selected merges")

    source_lineage = sorted(
        item for row in source_tracks for item in map(int, row["lineage_proposal_ids"])
    )
    final_lineage = sorted(
        item for row in final for item in map(int, row["lineage_proposal_ids"])
    )
    source_observations = sorted(
        item for row in source_tracks for item in map(int, row["observation_ids"])
    )
    final_observations = sorted(
        item for row in final for item in map(int, row["observation_ids"])
    )
    if len(source_lineage) != len(set(source_lineage)):
        raise ValueError("D2b lineage is not proposal-disjoint")
    if len(source_observations) != len(set(source_observations)):
        raise ValueError("D2b observations are not proposal-disjoint")
    if final_lineage != source_lineage:
        raise ValueError("family competition does not conserve D2b lineage")
    if final_observations != source_observations:
        raise ValueError("family competition does not conserve D2b observations")
    return final, ledger


def _validate_plan_manifest(plan_root, expected_scene_count):
    path = plan_root / "multiview_fragment_family_competition_plan_summary.json"
    payload = json.loads(path.read_text())
    if int(payload.get("scene_count", 0)) != int(expected_scene_count):
        raise ValueError("family plan manifest scene count differs")
    if payload.get("decision_contract") != DECISION_CONTRACT:
        raise ValueError("family plan manifest decision contract differs")
    if payload.get("ground_truth_usage") != "none":
        raise ValueError("family plan manifest used ground truth")
    if int(payload.get("candidate_action_applied_count", -1)) != 0:
        raise ValueError("family plan manifest already contains applied actions")
    return path, payload


def _load_track_payload(root, scene_name, label):
    payload = json.loads((root / scene_name / "automatic_tracks.json").read_text())
    tracks = payload.get("tracks", [])
    if int(payload.get("track_count", len(tracks))) != len(tracks):
        raise ValueError(f"{scene_name} {label} track count differs")
    _validate_tracks(tracks, label, validate_point_files=True)
    return payload, tracks


def _build_scene(scene_name, args):
    source_payload, source_tracks = _load_track_payload(
        args.source_track_root, scene_name, "source"
    )
    _, merged_tracks = _load_track_payload(
        args.merged_track_root, scene_name, "merged"
    )
    plans = _read_jsonl(args.plan_root / scene_name / "fragment_family_plan.jsonl")
    if any(row.get("scene_name") != scene_name for row in plans):
        raise ValueError(f"{scene_name} plan contains another scene")
    final, ledger = materialize_family_competition(
        source_tracks, merged_tracks, plans
    )

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        point_root = staging / "track_points"
        point_root.mkdir()
        public_tracks = []
        for track in final:
            proposal_id = int(track["proposal_id"])
            filename = f"track{proposal_id:04d}_points.npz"
            shutil.copyfile(Path(track["points_path"]), point_root / filename)
            public = dict(track)
            public["points_path"] = str(
                args.output_root / scene_name / "track_points" / filename
            )
            public_tracks.append(public)

        selected_count = sum(row["merged_candidate_selected"] for row in ledger)
        keep_count = len(ledger) - selected_count
        output_payload = dict(source_payload)
        output_payload.update({
            "source_track_count": len(source_tracks),
            "track_count": len(public_tracks),
            "tracks": public_tracks,
            "fragment_family_competition_materialization": "frozen_f2",
        })
        (staging / "automatic_tracks.json").write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _write_jsonl(staging / "fragment_family_materialization.jsonl", ledger)
        summary = {
            "scene_name": scene_name,
            "source_proposal_count": len(source_tracks),
            "final_proposal_count": len(public_tracks),
            "family_count": len(ledger),
            "selected_merged_candidate_count": selected_count,
            "kept_original_pair_count": keep_count,
            "removed_absorbed_proposal_count": selected_count,
            "retained_source_proposal_count": len(public_tracks) - selected_count,
            "candidate_field_mutation_count": 0,
            "score_recomputed_count": 0,
            "score_used_for_decision_count": 0,
            "native_candidate_mutation_count": 0,
            "proposal_lineage_conserved": True,
            "proposal_observation_conserved": True,
            "ground_truth_usage": "none",
            "decision_contract": DECISION_CONTRACT,
        }
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
    parser.add_argument("--source-track-root", type=Path, required=True)
    parser.add_argument("--merged-track-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "source_track_root", "merged_track_root", "plan_root",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list)
    plan_manifest_path, plan_manifest = _validate_plan_manifest(
        args.plan_root, len(all_scenes)
    )
    scenes = all_scenes
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
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
                f"[done] {index}/{len(scenes)} {scene_name}: selected M "
                f"{summary['selected_merged_candidate_count']}/"
                f"{summary['family_count']}, final {summary['final_proposal_count']}",
                flush=True,
            )
        summaries.append(summary)

    additive_keys = (
        "source_proposal_count", "final_proposal_count", "family_count",
        "selected_merged_candidate_count", "kept_original_pair_count",
        "removed_absorbed_proposal_count", "retained_source_proposal_count",
        "candidate_field_mutation_count", "score_recomputed_count",
        "score_used_for_decision_count", "native_candidate_mutation_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in additive_keys
        },
        "source_plan_family_count": int(plan_manifest["family_count"]),
        "plan_manifest": str(plan_manifest_path),
        "proposal_lineage_conserved": all(
            row["proposal_lineage_conserved"] for row in summaries
        ),
        "proposal_observation_conserved": all(
            row["proposal_observation_conserved"] for row in summaries
        ),
        "decision_contract": DECISION_CONTRACT,
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "multiview_fragment_family_competition_materialization_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "source_proposal_count", "final_proposal_count",
        "family_count", "selected_merged_candidate_count",
        "kept_original_pair_count", "candidate_field_mutation_count",
        "score_recomputed_count", "score_used_for_decision_count",
        "native_candidate_mutation_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

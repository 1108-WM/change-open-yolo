#!/usr/bin/env python3
"""Materialize one GT-free MV3DIS baseline-adapted boundary ablation.

Resolve, move, and grow are deliberately separate geometry variants.  The
selected family is applied atomically per boundary superpoint to frozen D2b
proposals.  If the batch would empty a proposal, every action removing from
that proposal falls back, and the batch is recomputed until all proposals are
non-empty.  Proposal identity, lineage, scores, and all non-geometry metadata
are inherited from D2b.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("resolve", "move", "grow")
GEOMETRY_KEYS = {
    "superpoint_ids", "superpoint_count", "point_count", "points_path",
    "decision_state",
}


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


def _point_count(superpoint_ids, superpoint_sizes):
    return int(sum(int(superpoint_sizes[item]) for item in superpoint_ids))


def _source_geometry(source_tracks, superpoint_sizes):
    geometries = {}
    for track in source_tracks:
        proposal_id = int(track["proposal_id"])
        if proposal_id in geometries:
            raise ValueError(f"duplicate proposal ID: {proposal_id}")
        if int(track.get("track_id", proposal_id)) != proposal_id:
            raise ValueError(f"proposal/track identity differs: {proposal_id}")
        ids = list(map(int, track.get("superpoint_ids", [])))
        if not ids or ids != sorted(set(ids)):
            raise ValueError(f"proposal {proposal_id} has invalid superpoint IDs")
        unknown = set(ids) - set(superpoint_sizes)
        if unknown:
            raise ValueError(f"proposal {proposal_id} references unknown superpoints")
        expected_count = _point_count(ids, superpoint_sizes)
        if int(track.get("superpoint_count", len(ids))) != len(ids):
            raise ValueError(f"proposal {proposal_id} superpoint_count is inconsistent")
        if int(track.get("point_count", expected_count)) != expected_count:
            raise ValueError(f"proposal {proposal_id} point_count is inconsistent")
        geometries[proposal_id] = set(ids)
    return geometries


def _owners_by_superpoint(geometries):
    owners = defaultdict(set)
    for proposal_id, superpoint_ids in geometries.items():
        for superpoint_id in superpoint_ids:
            owners[int(superpoint_id)].add(int(proposal_id))
    return owners


def _selected_actions(plan_rows, family, source_geometry):
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    owners = _owners_by_superpoint(source_geometry)
    selected = []
    seen_superpoints = set()
    for row in sorted(plan_rows, key=lambda item: int(item["superpoint_id"])):
        if row.get("ablation_family") != family:
            continue
        superpoint_id = int(row["superpoint_id"])
        if superpoint_id in seen_superpoints:
            raise ValueError(f"duplicate selected superpoint: {superpoint_id}")
        seen_superpoints.add(superpoint_id)
        remove_from = sorted(set(map(int, row.get("remove_from_proposal_ids", []))))
        add_to = sorted(set(map(int, row.get("add_to_proposal_ids", []))))
        referenced = set(remove_from) | set(add_to)
        unknown = referenced - set(source_geometry)
        if unknown:
            raise ValueError(f"action references unknown proposals: {sorted(unknown)}")
        expected_owners = sorted(set(map(int, row["current_owner_proposal_ids"])))
        actual_owners = sorted(owners.get(superpoint_id, set()))
        if actual_owners != expected_owners:
            raise ValueError(
                f"superpoint {superpoint_id} ownership differs: "
                f"plan={expected_owners}, D2b={actual_owners}"
            )
        if any(superpoint_id not in source_geometry[item] for item in remove_from):
            raise ValueError(f"action removes non-owned superpoint {superpoint_id}")
        if any(superpoint_id in source_geometry[item] for item in add_to):
            raise ValueError(f"action redundantly adds owned superpoint {superpoint_id}")
        if not remove_from and not add_to:
            raise ValueError(f"selected action has no geometry delta: {superpoint_id}")
        selected.append({
            **row,
            "materialization_action_index": len(selected),
            "superpoint_id": superpoint_id,
            "remove_from_proposal_ids": remove_from,
            "add_to_proposal_ids": add_to,
        })
    return selected


def _apply_action_subset(source_geometry, actions, active_indices):
    result = {proposal_id: set(ids) for proposal_id, ids in source_geometry.items()}
    for action in actions:
        if action["materialization_action_index"] not in active_indices:
            continue
        superpoint_id = int(action["superpoint_id"])
        for proposal_id in action["remove_from_proposal_ids"]:
            result[proposal_id].remove(superpoint_id)
        for proposal_id in action["add_to_proposal_ids"]:
            result[proposal_id].add(superpoint_id)
    return result


def materialize_family(source_tracks, plan_rows, superpoint_sizes, family):
    """Apply one family and return tracks, action ledger, and proposal ledger."""
    source_geometry = _source_geometry(source_tracks, superpoint_sizes)
    actions = _selected_actions(plan_rows, family, source_geometry)
    active_indices = {
        int(action["materialization_action_index"]) for action in actions
    }
    fallback_reasons = defaultdict(list)
    while True:
        candidate_geometry = _apply_action_subset(
            source_geometry, actions, active_indices
        )
        empty_ids = sorted(
            proposal_id for proposal_id, ids in candidate_geometry.items() if not ids
        )
        if not empty_ids:
            break
        newly_fallback = set()
        for proposal_id in empty_ids:
            for action in actions:
                action_index = int(action["materialization_action_index"])
                if (
                    action_index in active_indices
                    and proposal_id in action["remove_from_proposal_ids"]
                ):
                    newly_fallback.add(action_index)
                    fallback_reasons[action_index].append(
                        f"would_empty_proposal:{proposal_id}"
                    )
        if not newly_fallback:
            raise ValueError(f"empty proposals cannot be recovered: {empty_ids}")
        active_indices -= newly_fallback

    final_geometry = _apply_action_subset(source_geometry, actions, active_indices)
    source_owners = _owners_by_superpoint(source_geometry)
    final_owners = _owners_by_superpoint(final_geometry)
    action_ledger = []
    for action in actions:
        action_index = int(action["materialization_action_index"])
        applied = action_index in active_indices
        superpoint_id = int(action["superpoint_id"])
        ledger = dict(action)
        ledger.update({
            "materialization_status": "applied" if applied else "fallback_would_empty",
            "assignment_applied": applied,
            "proposal_mutation_applied": applied,
            "fallback_reasons": sorted(set(fallback_reasons[action_index])),
            "source_owner_proposal_ids": sorted(source_owners.get(superpoint_id, set())),
            "final_owner_proposal_ids": sorted(final_owners.get(superpoint_id, set())),
            "ground_truth_usage": "none",
        })
        action_ledger.append(ledger)

    actions_by_proposal = defaultdict(list)
    fallback_by_proposal = defaultdict(list)
    for action in action_ledger:
        action_index = int(action["materialization_action_index"])
        involved = set(action["remove_from_proposal_ids"]) | set(
            action["add_to_proposal_ids"]
        )
        target = actions_by_proposal if action["assignment_applied"] else fallback_by_proposal
        for proposal_id in involved:
            target[proposal_id].append(action_index)

    final_tracks = []
    proposal_ledger = []
    for source in source_tracks:
        proposal_id = int(source["proposal_id"])
        source_ids = source_geometry[proposal_id]
        final_ids = final_geometry[proposal_id]
        added = sorted(final_ids - source_ids)
        removed = sorted(source_ids - final_ids)
        changed = bool(added or removed)
        result = dict(source)
        result.update({
            "superpoint_ids": sorted(final_ids),
            "superpoint_count": len(final_ids),
            "point_count": _point_count(final_ids, superpoint_sizes),
            "decision_state": (
                f"MV3DIS baseline-adapted {family} geometry variant."
                if changed else "Unchanged frozen D2b proposal fallback."
            ),
        })
        final_tracks.append(result)
        proposal_ledger.append({
            "proposal_id": proposal_id,
            "track_id": int(source.get("track_id", proposal_id)),
            "lineage_proposal_ids": list(source.get("lineage_proposal_ids", [proposal_id])),
            "materialization_family": family,
            "geometry_changed": changed,
            "source_superpoint_ids": sorted(source_ids),
            "final_superpoint_ids": sorted(final_ids),
            "added_superpoint_ids": added,
            "removed_superpoint_ids": removed,
            "source_point_count": _point_count(source_ids, superpoint_sizes),
            "final_point_count": _point_count(final_ids, superpoint_sizes),
            "applied_action_indices": sorted(actions_by_proposal[proposal_id]),
            "fallback_action_indices": sorted(fallback_by_proposal[proposal_id]),
            "ground_truth_usage": "none",
        })
    validate_materialization(
        source_tracks, final_tracks, action_ledger, proposal_ledger, family
    )
    return final_tracks, action_ledger, proposal_ledger


def validate_materialization(source, final, actions, proposal_ledger, family):
    source_by_id = {int(row["proposal_id"]): row for row in source}
    final_by_id = {int(row["proposal_id"]): row for row in final}
    if list(source_by_id) != list(final_by_id) or len(source_by_id) != len(source):
        raise ValueError("proposal count, order, or IDs changed")
    for proposal_id, source_row in source_by_id.items():
        final_row = final_by_id[proposal_id]
        if not final_row["superpoint_ids"]:
            raise ValueError(f"proposal {proposal_id} became empty")
        for key, value in source_row.items():
            if key not in GEOMETRY_KEYS and final_row.get(key) != value:
                raise ValueError(f"proposal {proposal_id} changed non-geometry field {key}")
    if len(proposal_ledger) != len(source):
        raise ValueError("proposal ledger is not one-to-one with D2b")
    if any(row.get("ablation_family") != family for row in actions):
        raise ValueError("materialization mixed action families")
    if any(row.get("ground_truth_usage") != "none" for row in actions + proposal_ledger):
        raise ValueError("ground-truth contract is missing")


def _load_superpoints(scene_name, processed_scene_root):
    path = processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    points = {
        int(item): np.flatnonzero(superpoints == item).astype(np.int64)
        for item in ids
    }
    return superpoints, sizes, points


def _validate_source_point_files(tracks, superpoints):
    for track in tracks:
        path = Path(track["points_path"])
        with np.load(path) as payload:
            actual = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        expected = np.flatnonzero(np.isin(superpoints, track["superpoint_ids"]))
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"source proposal {track['proposal_id']} point file is inconsistent"
            )


def _build_and_publish_scene(scene_name, args):
    superpoints, superpoint_sizes, points_by_superpoint = _load_superpoints(
        scene_name, args.processed_scene_root
    )
    payload = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    source_tracks = payload.get("tracks", [])
    if int(payload.get("track_count", len(source_tracks))) != len(source_tracks):
        raise ValueError(f"{scene_name} source track count is inconsistent")
    _validate_source_point_files(source_tracks, superpoints)
    plan_rows = _read_jsonl(args.plan_root / scene_name / "assignment_plan.jsonl")
    final, actions, proposal_ledger = materialize_family(
        source_tracks, plan_rows, superpoint_sizes, args.family
    )

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        point_root = staging / "track_points"
        point_root.mkdir()
        for track in final:
            proposal_id = int(track["proposal_id"])
            filename = f"track{proposal_id:04d}_points.npz"
            chunks = [points_by_superpoint[item] for item in track["superpoint_ids"]]
            point_indices = np.sort(np.concatenate(chunks).astype(np.int64, copy=False))
            np.savez_compressed(point_root / filename, point_indices=point_indices)
            track["points_path"] = str(
                args.output_root / scene_name / "track_points" / filename
            )
        counts = Counter(row["materialization_status"] for row in actions)
        changed_count = sum(row["geometry_changed"] for row in proposal_ledger)
        added_count = sum(len(row["added_superpoint_ids"]) for row in proposal_ledger)
        removed_count = sum(len(row["removed_superpoint_ids"]) for row in proposal_ledger)
        summary = {
            "scene_name": scene_name,
            "materialization_family": args.family,
            "source_proposal_count": len(source_tracks),
            "final_proposal_count": len(final),
            "planned_action_count": len(actions),
            "applied_action_count": counts["applied"],
            "fallback_action_count": len(actions) - counts["applied"],
            "changed_proposal_count": changed_count,
            "added_proposal_superpoint_membership_count": added_count,
            "removed_proposal_superpoint_membership_count": removed_count,
            "empty_proposal_count": 0,
            "proposal_id_mutation_count": 0,
            "lineage_mutation_count": 0,
            "score_mutation_count": 0,
            "ground_truth_usage": "none",
            "method_contract": (
                "baseline_adapted strict all-edge dominance; "
                "not MV3DIS paper region refinement"
            ),
        }
        output_payload = dict(payload)
        output_payload.update({
            "track_count": len(final),
            "tracks": final,
            "mv3dis_baseline_adapted_family": args.family,
        })
        (staging / "automatic_tracks.json").write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _write_jsonl(staging / "materialization_actions.jsonl", actions)
        _write_jsonl(staging / "proposal_geometry_ledger.jsonl", proposal_ledger)
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
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "track_root", "plan_root", "processed_scene_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_and_publish_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[done] {index}/{len(scenes)} {scene_name}: {args.family} "
            f"{summary['applied_action_count']}/{summary['planned_action_count']}, "
            f"changed {summary['changed_proposal_count']}",
            flush=True,
        )
    sum_keys = (
        "source_proposal_count", "final_proposal_count", "planned_action_count",
        "applied_action_count", "fallback_action_count", "changed_proposal_count",
        "added_proposal_superpoint_membership_count",
        "removed_proposal_superpoint_membership_count", "empty_proposal_count",
        "proposal_id_mutation_count", "lineage_mutation_count", "score_mutation_count",
    )
    result = {
        "scene_count": len(summaries),
        "materialization_family": args.family,
        **{key: sum(int(row[key]) for row in summaries) for key in sum_keys},
        "ground_truth_usage": "none",
        "method_contract": (
            "baseline_adapted strict all-edge dominance; "
            "not MV3DIS paper region refinement"
        ),
        "ablation_contract": "this output contains exactly one action family",
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scene_summaries": summaries,
    }
    summary_path = args.output_root / "mv3dis_baseline_adapted_variant_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: result[key] for key in (
        "scene_count", "materialization_family", "final_proposal_count",
        "planned_action_count", "applied_action_count", "fallback_action_count",
        "changed_proposal_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

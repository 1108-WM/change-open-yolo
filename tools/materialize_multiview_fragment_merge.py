#!/usr/bin/env python3
"""Materialize a frozen conservative multiview fragment merge plan.

Planned pairs are proposal-disjoint.  Each pair combines its frozen D2b
observations and lineage, then immediately reuses the exact D2b multiview
consensus refinement over the union of the two proposal superpoint domains.
Pairs do not trigger further iterative merges.  If refinement would empty a
proposal, that pair atomically falls back to both original proposals.

The plan is the only action input.  This tool reads no GT, native predictions,
semantics, classes, or AP results.  Scores are never used for selection; the
score fields of an applied merge are recomputed with D2b's existing
observation-weighted aggregation contract.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_details_iterative_proposal_merges import (
    CONSENSUS_PARAMS,
    _merge_proposals,
    _validate_source_point_files,
)
from tools.refine_details_automatic_tracks_consensus import (
    _load_observations,
    _superpoint_points,
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


def _point_count(superpoint_ids, superpoint_sizes):
    return int(sum(int(superpoint_sizes[item]) for item in superpoint_ids))


def prepare_d2b_proposals(source_tracks, superpoint_sizes):
    """Restore D2b's private aggregation accumulators without changing fields."""
    prepared = []
    proposal_ids = []
    observation_owner = {}
    lineage_owner = {}
    known_superpoints = set(map(int, superpoint_sizes))
    for source in sorted(source_tracks, key=lambda row: int(row["proposal_id"])):
        proposal_id = int(source["proposal_id"])
        if int(source.get("track_id", proposal_id)) != proposal_id:
            raise ValueError(f"proposal {proposal_id} track identity mismatch")
        superpoint_ids = list(map(int, source.get("superpoint_ids", [])))
        if not superpoint_ids or superpoint_ids != sorted(set(superpoint_ids)):
            raise ValueError(f"proposal {proposal_id} has invalid superpoints")
        if set(superpoint_ids) - known_superpoints:
            raise ValueError(f"proposal {proposal_id} references unknown superpoints")
        point_count = _point_count(superpoint_ids, superpoint_sizes)
        if int(source.get("point_count", point_count)) != point_count:
            raise ValueError(f"proposal {proposal_id} point count mismatch")
        observation_ids = sorted(set(map(int, source.get("observation_ids", []))))
        lineage = sorted(set(map(int, source.get("lineage_proposal_ids", []))))
        source_track_ids = sorted(
            set(map(int, source.get("source_track_ids", [source.get("source_track_id", proposal_id)])))
        )
        if not observation_ids or not lineage or not source_track_ids:
            raise ValueError(f"proposal {proposal_id} lacks provenance")
        for observation_id in observation_ids:
            if observation_id in observation_owner:
                raise ValueError(f"observation {observation_id} has multiple D2b owners")
            observation_owner[observation_id] = proposal_id
        for lineage_id in lineage:
            if lineage_id in lineage_owner:
                raise ValueError(f"lineage {lineage_id} has multiple D2b owners")
            lineage_owner[lineage_id] = proposal_id

        observation_weight = len(observation_ids)
        edge_weight = max(0, observation_weight - len(source_track_ids))
        proposal = dict(source)
        proposal.update({
            "proposal_id": proposal_id,
            "track_id": proposal_id,
            "source_track_ids": source_track_ids,
            "lineage_proposal_ids": lineage,
            "observation_ids": observation_ids,
            "superpoint_ids": superpoint_ids,
            "superpoint_count": len(superpoint_ids),
            "point_count": point_count,
            "_quality_sum": float(source.get("mean_node_quality", 0.0)) * observation_weight,
            "_predicted_iou_sum": float(source.get("mean_predicted_iou", 0.0)) * observation_weight,
            "_stability_sum": float(source.get("mean_stability_score", 0.0)) * observation_weight,
            "_observation_weight": observation_weight,
            "_edge_score_sum": float(source.get("mean_edge_score", 0.0)) * edge_weight,
            "_edge_weight": edge_weight,
        })
        proposal_ids.append(proposal_id)
        prepared.append(proposal)
    if proposal_ids != sorted(proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("D2b proposal IDs are invalid")
    return prepared


def materialize_fragment_merges(
    source_tracks,
    planned_actions,
    observations,
    superpoint_sizes,
    merge_function=_merge_proposals,
):
    prepared = prepare_d2b_proposals(source_tracks, superpoint_sizes)
    active = {int(row["proposal_id"]): row for row in prepared}
    source_lineage = sorted(
        lineage
        for row in prepared
        for lineage in map(int, row["lineage_proposal_ids"])
    )
    action_ids = []
    for expected_index, action in enumerate(planned_actions):
        if int(action["action_index"]) != expected_index:
            raise ValueError("fragment plan action indices are not deterministic")
        anchor_id = int(action["anchor_proposal_id"])
        absorbed_id = int(action["absorbed_proposal_id"])
        if anchor_id >= absorbed_id:
            raise ValueError("fragment plan anchor must be the lower proposal ID")
        action_ids.extend((anchor_id, absorbed_id))
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("fragment plan actions are not proposal-disjoint")
    if set(action_ids) - set(active):
        raise ValueError("fragment plan references unknown D2b proposals")

    ledger = []
    applied_anchor_ids = set()
    for action in planned_actions:
        anchor_id = int(action["anchor_proposal_id"])
        absorbed_id = int(action["absorbed_proposal_id"])
        anchor = active[anchor_id]
        absorbed = active[absorbed_id]
        try:
            merged, refinement = merge_function(
                anchor,
                absorbed,
                observations,
                superpoint_sizes,
                int(action["action_index"]),
            )
        except ValueError as error:
            if "became empty after frozen consensus refinement" not in str(error):
                raise
            ledger.append({
                **action,
                "materialization_state": "atomic_fallback_empty_refinement",
                "merge_action_applied": False,
                "anchor_lineage_before": list(anchor["lineage_proposal_ids"]),
                "absorbed_lineage": list(absorbed["lineage_proposal_ids"]),
                "score_used_for_decision": False,
                "gt_usage": "none",
            })
            continue
        active[anchor_id] = merged
        del active[absorbed_id]
        applied_anchor_ids.add(anchor_id)
        ledger.append({
            **action,
            "materialization_state": "applied_d2b_consensus_refined_merge",
            "merge_action_applied": True,
            "anchor_lineage_before": list(anchor["lineage_proposal_ids"]),
            "absorbed_lineage": list(absorbed["lineage_proposal_ids"]),
            "anchor_lineage_after": list(merged["lineage_proposal_ids"]),
            **refinement,
            "score_used_for_decision": False,
            "gt_usage": "none",
        })

    final = [active[item] for item in sorted(active)]
    final_lineage = sorted(
        lineage
        for row in final
        for lineage in map(int, row["lineage_proposal_ids"])
    )
    applied_count = sum(row["merge_action_applied"] for row in ledger)
    if final_lineage != source_lineage:
        raise ValueError("fragment materialization does not conserve D2b lineage")
    if len(final) != len(source_tracks) - applied_count:
        raise ValueError("fragment proposal count does not match applied actions")
    if any(not row["superpoint_ids"] for row in final):
        raise ValueError("fragment materialization produced an empty proposal")
    return final, ledger, applied_anchor_ids


def _public_track(proposal, source_by_id, applied_anchor_ids, points_path):
    proposal_id = int(proposal["proposal_id"])
    if proposal_id not in applied_anchor_ids:
        result = dict(source_by_id[proposal_id])
    else:
        result = {key: value for key, value in proposal.items() if not key.startswith("_")}
        result["decision_state"] = (
            "Multiview fragment merge followed by frozen D2b consensus refinement."
        )
    result["points_path"] = str(points_path)
    return result


def _build_scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {
        int(item): int(count) for item, count in zip(ids, counts)
    }
    points_by_superpoint = _superpoint_points(superpoints)

    source_payload = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    source_tracks = source_payload.get("tracks", [])
    source_by_id = {int(row["proposal_id"]): row for row in source_tracks}
    _validate_source_point_files(source_tracks, superpoints)
    actions = _read_jsonl(
        args.plan_root / scene_name / "fragment_merge_actions.jsonl"
    )

    world = WORLD_2_CAM(
        str(args.dataset_root / scene_name), args.depth_scale, args.config
    )
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations = _load_observations(
        args.automatic_root / scene_name, superpoints, visibility
    )
    final, materialization_ledger, applied_anchor_ids = materialize_fragment_merges(
        source_tracks, actions, observations, superpoint_sizes
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
        for proposal in final:
            proposal_id = int(proposal["proposal_id"])
            filename = f"track{proposal_id:04d}_points.npz"
            chunks = [points_by_superpoint[item] for item in proposal["superpoint_ids"]]
            points = np.sort(np.concatenate(chunks).astype(np.int64, copy=False))
            np.savez_compressed(point_root / filename, point_indices=points)
            public_tracks.append(_public_track(
                proposal,
                source_by_id,
                applied_anchor_ids,
                args.output_root / scene_name / "track_points" / filename,
            ))
        applied_count = sum(
            row["merge_action_applied"] for row in materialization_ledger
        )
        fallback_count = len(materialization_ledger) - applied_count
        payload = {
            "scene_name": scene_name,
            "source_track_count": len(source_tracks),
            "track_count": len(public_tracks),
            "planned_merge_action_count": len(actions),
            "applied_merge_action_count": applied_count,
            "fallback_merge_action_count": fallback_count,
            "tracks": public_tracks,
        }
        (staging / "automatic_tracks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _write_jsonl(
            staging / "fragment_merge_materialization.jsonl", materialization_ledger
        )
        summary = {
            "scene_name": scene_name,
            "source_proposal_count": len(source_tracks),
            "final_proposal_count": len(public_tracks),
            "planned_merge_action_count": len(actions),
            "applied_merge_action_count": applied_count,
            "fallback_merge_action_count": fallback_count,
            "changed_final_proposal_count": len(applied_anchor_ids),
            "refinement_removed_superpoint_count": sum(
                int(row.get("removed_by_refinement_count", 0))
                for row in materialization_ledger
            ),
            "score_used_for_decision_count": 0,
            "score_recomputed_for_merged_proposal_count": len(applied_anchor_ids),
            "consensus_params": CONSENSUS_PARAMS,
            "proposal_lineage_conserved": True,
            "ground_truth_usage": "none",
            "decision_state": (
                "Frozen mutual-unique-best fragment plan materialized with exact "
                "D2b consensus refinement and atomic empty fallback."
            ),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    del world, raw_visibility, visibility, observations, processed
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument(
        "--processed-scene-root", type=Path, default=Path("data/scannet200")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument(
        "--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "track_root", "automatic_root", "plan_root",
        "processed_scene_root", "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("consensus_params") != CONSENSUS_PARAMS:
                raise SystemExit(f"resume consensus mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: applied "
                f"{summary['applied_merge_action_count']}/"
                f"{summary['planned_merge_action_count']}, final "
                f"{summary['final_proposal_count']}",
                flush=True,
            )
        summaries.append(summary)

    additive_keys = (
        "source_proposal_count", "final_proposal_count",
        "planned_merge_action_count", "applied_merge_action_count",
        "fallback_merge_action_count", "changed_final_proposal_count",
        "refinement_removed_superpoint_count", "score_used_for_decision_count",
        "score_recomputed_for_merged_proposal_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in additive_keys
        },
        "consensus_params": CONSENSUS_PARAMS,
        "proposal_lineage_conserved": all(
            row["proposal_lineage_conserved"] for row in summaries
        ),
        "score_used_for_decision_count": 0,
        "ground_truth_usage": "none",
        "decision_state": (
            "Frozen fragment plan materialized; no iterative propagation, native "
            "competition, semantics, or AP."
        ),
        "params": vars(args),
        "scene_summaries": summaries,
    }
    (args.output_root / "multiview_fragment_merge_materialization_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "source_proposal_count": payload["source_proposal_count"],
        "final_proposal_count": payload["final_proposal_count"],
        "planned_merge_action_count": payload["planned_merge_action_count"],
        "applied_merge_action_count": payload["applied_merge_action_count"],
        "fallback_merge_action_count": payload["fallback_merge_action_count"],
        "refinement_removed_superpoint_count": payload["refinement_removed_superpoint_count"],
        "score_used_for_decision_count": payload["score_used_for_decision_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

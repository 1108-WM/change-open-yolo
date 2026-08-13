#!/usr/bin/env python3
"""Run the Details iterative merge loop on frozen D1 consensus proposals.

This D2b stage implements only Algorithm 1's merge/refine loop.  It uses the
paper's strict point-IoU > 0.3 criterion, a frozen upper-triangular matrix per
round, immediate multi-view consensus refinement after every absorption, and a
full relationship/action ledger.  It does not perform duplicate suppression,
inclusion cleanup, boundary growth, scoring changes, semantic assignment, or AP.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.refine_details_automatic_tracks_consensus import (
    _load_observations,
    _superpoint_points,
    refine_tracklet_superpoints,
)


DETAILS_MERGE_IOU = 0.30
CONSENSUS_PARAMS = {
    "min_visible_ratio": 0.10,
    "min_mask_support": 0.30,
    "min_visible_points": 3,
    "frame_superpoint_coverage": 0.50,
    "min_support_frames": 2,
    "min_consensus_rate": 0.30,
    "mean_superpoint_coverage": 0.55,
}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _point_count(superpoint_ids, superpoint_sizes):
    return int(sum(int(superpoint_sizes[item]) for item in superpoint_ids))


def proposal_geometry(left, right, superpoint_sizes):
    left_ids = set(map(int, left["superpoint_ids"]))
    right_ids = set(map(int, right["superpoint_ids"]))
    left_count = _point_count(left_ids, superpoint_sizes)
    right_count = _point_count(right_ids, superpoint_sizes)
    intersection = _point_count(left_ids & right_ids, superpoint_sizes)
    union = left_count + right_count - intersection
    return {
        "left_point_count": left_count,
        "right_point_count": right_count,
        "point_intersection_count": intersection,
        "point_iou": float(intersection / max(1, union)),
        "left_point_coverage": float(intersection / max(1, left_count)),
        "right_point_coverage": float(intersection / max(1, right_count)),
    }


def proposal_pair_rows(proposals, superpoint_sizes, round_index):
    """Build a strict-upper-triangular point relationship snapshot."""
    ordered = sorted(proposals, key=lambda row: int(row["proposal_id"]))
    rows = []
    for left, right in combinations(ordered, 2):
        geometry = proposal_geometry(left, right, superpoint_sizes)
        rows.append({
            "round_index": int(round_index),
            "left_proposal_id": int(left["proposal_id"]),
            "right_proposal_id": int(right["proposal_id"]),
            **geometry,
            "details_merge_eligible_observed": bool(
                geometry["point_iou"] > DETAILS_MERGE_IOU
            ),
            "decision_state": "Frozen start-of-round relation; action uses strict point IoU > 0.3.",
        })
    expected = len(ordered) * (len(ordered) - 1) // 2
    if len(rows) != expected:
        raise ValueError("round relationship count is not conserved")
    return rows


def initialize_proposals(source_tracks, superpoint_sizes):
    proposals = []
    track_ids = [int(track["track_id"]) for track in source_tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("source track IDs must be unique")
    observation_owner = {}
    for track in sorted(source_tracks, key=lambda row: int(row["track_id"])):
        proposal_id = int(track["track_id"])
        superpoint_ids = list(map(int, track.get("superpoint_ids", [])))
        if not superpoint_ids or superpoint_ids != sorted(set(superpoint_ids)):
            raise ValueError(f"track {proposal_id} has invalid superpoint_ids")
        if any(item not in superpoint_sizes for item in superpoint_ids):
            raise ValueError(f"track {proposal_id} references unknown superpoints")
        point_count = _point_count(superpoint_ids, superpoint_sizes)
        if int(track.get("point_count", point_count)) != point_count:
            raise ValueError(f"track {proposal_id} point_count is inconsistent")
        observation_ids = sorted(set(map(int, track.get("observation_ids", []))))
        if not observation_ids:
            raise ValueError(f"track {proposal_id} has no observations")
        for observation_id in observation_ids:
            if observation_id in observation_owner:
                raise ValueError(
                    f"observation {observation_id} belongs to tracks "
                    f"{observation_owner[observation_id]} and {proposal_id}"
                )
            observation_owner[observation_id] = proposal_id
        observation_weight = len(observation_ids)
        edge_weight = max(0, observation_weight - 1)
        proposal = dict(track)
        proposal.update({
            "proposal_id": proposal_id,
            "track_id": proposal_id,
            "source_track_id": int(track.get("source_track_id", proposal_id)),
            "source_track_ids": [int(track.get("source_track_id", proposal_id))],
            "lineage_proposal_ids": [proposal_id],
            "observation_ids": observation_ids,
            "node_ids": sorted(set(map(int, track.get("node_ids", [])))),
            "frame_ids": sorted(
                set(map(str, track.get("frame_ids", []))),
                key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
            ),
            "superpoint_ids": superpoint_ids,
            "superpoint_count": len(superpoint_ids),
            "point_count": point_count,
            "merge_action_count": 0,
            "last_merge_round": None,
            "_quality_sum": float(track.get("mean_node_quality", 0.0)) * observation_weight,
            "_predicted_iou_sum": float(track.get("mean_predicted_iou", 0.0)) * observation_weight,
            "_stability_sum": float(track.get("mean_stability_score", 0.0)) * observation_weight,
            "_observation_weight": observation_weight,
            "_edge_score_sum": float(track.get("mean_edge_score", 0.0)) * edge_weight,
            "_edge_weight": edge_weight,
        })
        proposals.append(proposal)
    return proposals


def _merge_proposals(
    anchor,
    absorbed,
    observations,
    superpoint_sizes,
    round_index,
):
    if set(anchor["observation_ids"]) & set(absorbed["observation_ids"]):
        raise ValueError("merged proposals share observation IDs")
    observation_ids = sorted(
        set(map(int, anchor["observation_ids"]))
        | set(map(int, absorbed["observation_ids"]))
    )
    union_superpoints = set(map(int, anchor["superpoint_ids"])) | set(
        map(int, absorbed["superpoint_ids"])
    )
    kept, diagnostics, frame_rows, initial = refine_tracklet_superpoints(
        observation_ids,
        observations,
        superpoint_sizes,
        **CONSENSUS_PARAMS,
        candidate_superpoints=union_superpoints,
    )
    if not kept:
        raise ValueError(
            f"merge {anchor['proposal_id']} <- {absorbed['proposal_id']} "
            "became empty after frozen consensus refinement"
        )
    if not kept <= union_superpoints:
        raise ValueError("merge refinement introduced superpoints outside proposal union")

    result = dict(anchor)
    for key in (
        "_quality_sum", "_predicted_iou_sum", "_stability_sum",
        "_observation_weight", "_edge_score_sum", "_edge_weight",
    ):
        result[key] = anchor[key] + absorbed[key]
    result.update({
        "source_track_ids": sorted(
            set(map(int, anchor["source_track_ids"]))
            | set(map(int, absorbed["source_track_ids"]))
        ),
        "lineage_proposal_ids": sorted(
            set(map(int, anchor["lineage_proposal_ids"]))
            | set(map(int, absorbed["lineage_proposal_ids"]))
        ),
        "observation_ids": observation_ids,
        "node_ids": sorted(
            set(map(int, anchor.get("node_ids", [])))
            | set(map(int, absorbed.get("node_ids", [])))
        ),
        "frame_ids": sorted(
            set(map(str, anchor.get("frame_ids", [])))
            | set(map(str, absorbed.get("frame_ids", []))),
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        ),
        "support_view_count": len(frame_rows),
        "superpoint_ids": sorted(map(int, kept)),
        "superpoint_count": len(kept),
        "point_count": _point_count(kept, superpoint_sizes),
        "initial_superpoint_count": len(initial),
        "removed_superpoint_count": len(initial - kept),
        "mean_consensus_rate": float(
            np.mean([diagnostics[item]["consensus_rate"] for item in kept])
        ),
        "mean_supported_coverage": float(
            np.mean([diagnostics[item]["mean_supported_coverage"] for item in kept])
        ),
        "support_score": float(
            sum(diagnostics[item]["support_frames"] for item in kept)
        ),
        "merge_action_count": int(anchor["merge_action_count"])
        + int(absorbed["merge_action_count"])
        + 1,
        "last_merge_round": int(round_index),
    })
    weight = max(1, int(result["_observation_weight"]))
    result["mean_node_quality"] = float(result["_quality_sum"] / weight)
    result["mean_predicted_iou"] = float(result["_predicted_iou_sum"] / weight)
    result["mean_stability_score"] = float(result["_stability_sum"] / weight)
    result["mean_edge_score"] = float(
        result["_edge_score_sum"] / max(1, int(result["_edge_weight"]))
    )
    return result, {
        "union_superpoint_count": len(union_superpoints),
        "refined_superpoint_count": len(kept),
        "removed_by_refinement_count": len(union_superpoints - kept),
        "refined_point_count": _point_count(kept, superpoint_sizes),
        "merged_observation_count": len(observation_ids),
        "merged_frame_count": len(frame_rows),
    }


def iterative_merge_proposals(source_tracks, observations, superpoint_sizes):
    """Faithfully execute the supplement's frozen-matrix row/column loop."""
    active = {
        int(row["proposal_id"]): row
        for row in initialize_proposals(source_tracks, superpoint_sizes)
    }
    all_relations, actions, rounds = [], [], []
    round_index = 0
    while True:
        ordered = [active[key] for key in sorted(active)]
        relation_rows = proposal_pair_rows(ordered, superpoint_sizes, round_index)
        all_relations.extend(relation_rows)
        relation_by_pair = {
            (int(row["left_proposal_id"]), int(row["right_proposal_id"])): row
            for row in relation_rows
        }
        eligible_count = sum(
            bool(row["details_merge_eligible_observed"]) for row in relation_rows
        )
        round_summary = {
            "round_index": round_index,
            "start_proposal_count": len(ordered),
            "start_relation_count": len(relation_rows),
            "eligible_pair_count": eligible_count,
            "merge_action_count": 0,
            "absorbed_proposal_count": 0,
            "end_proposal_count": len(ordered),
            "terminal": eligible_count == 0,
        }
        if eligible_count == 0:
            rounds.append(round_summary)
            break

        visited, absorbed_ids = set(), set()
        ordered_ids = sorted(active)
        for row_position, anchor_id in enumerate(ordered_ids):
            if anchor_id in visited or anchor_id in absorbed_ids:
                continue
            for absorbed_id in ordered_ids[row_position + 1 :]:
                if absorbed_id in visited or absorbed_id in absorbed_ids:
                    continue
                relation = relation_by_pair[(anchor_id, absorbed_id)]
                if not relation["details_merge_eligible_observed"]:
                    continue
                anchor_before = active[anchor_id]
                absorbed = active[absorbed_id]
                refined, refinement = _merge_proposals(
                    anchor_before,
                    absorbed,
                    observations,
                    superpoint_sizes,
                    round_index,
                )
                active[anchor_id] = refined
                absorbed_ids.add(absorbed_id)
                visited.add(absorbed_id)
                actions.append({
                    "action_index": len(actions),
                    "round_index": round_index,
                    "anchor_proposal_id": anchor_id,
                    "absorbed_proposal_id": absorbed_id,
                    "frozen_round_point_iou": float(relation["point_iou"]),
                    "anchor_lineage_before": list(anchor_before["lineage_proposal_ids"]),
                    "absorbed_lineage": list(absorbed["lineage_proposal_ids"]),
                    "anchor_lineage_after": list(refined["lineage_proposal_ids"]),
                    "anchor_superpoint_count_before": len(anchor_before["superpoint_ids"]),
                    "absorbed_superpoint_count": len(absorbed["superpoint_ids"]),
                    **refinement,
                    "decision_state": "Details Algorithm 1 merge followed immediately by frozen consensus refinement.",
                })
            visited.add(anchor_id)
        if not absorbed_ids:
            raise ValueError("eligible relationships exist but the merge round made no progress")
        for proposal_id in absorbed_ids:
            del active[proposal_id]
        round_summary.update({
            "merge_action_count": len(absorbed_ids),
            "absorbed_proposal_count": len(absorbed_ids),
            "end_proposal_count": len(active),
            "terminal": False,
        })
        rounds.append(round_summary)
        round_index += 1

    final = [active[key] for key in sorted(active)]
    validate_merge_result(source_tracks, final, actions, rounds)
    return final, actions, rounds, all_relations


def validate_merge_result(source_tracks, final, actions, rounds):
    source_ids = sorted(int(row["track_id"]) for row in source_tracks)
    lineage = sorted(
        item for proposal in final for item in proposal["lineage_proposal_ids"]
    )
    if lineage != source_ids:
        raise ValueError("source proposal lineage is not conserved exactly once")
    if len(final) + len(actions) != len(source_tracks):
        raise ValueError("proposal count does not conserve one removal per merge action")
    if not rounds or not rounds[-1]["terminal"]:
        raise ValueError("merge loop did not record a terminal round")
    if any(not row["superpoint_ids"] for row in final):
        raise ValueError("final merge output contains an empty proposal")
    final_ids = [int(row["proposal_id"]) for row in final]
    if final_ids != sorted(final_ids) or len(final_ids) != len(set(final_ids)):
        raise ValueError("final proposal IDs are invalid")


def _validate_source_point_files(tracks, superpoints):
    for track in tracks:
        path = Path(track["points_path"])
        with np.load(path) as payload:
            actual = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        expected = np.flatnonzero(
            np.isin(superpoints, np.asarray(track["superpoint_ids"], dtype=np.int64))
        )
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"source track {track['track_id']} point file is not its superpoint union"
            )


def _public_proposal(proposal, output_points_path):
    result = {key: value for key, value in proposal.items() if not key.startswith("_")}
    result["points_path"] = str(output_points_path)
    result["decision_state"] = (
        "Unchanged frozen D1 proposal."
        if int(result["merge_action_count"]) == 0
        else "Details iterative merge/refine output; no suppress or inclusion cleanup applied."
    )
    return result


def _scene_summary(scene_name, source_count, final, actions, rounds, relations):
    return {
        "scene_name": scene_name,
        "source_proposal_count": source_count,
        "final_proposal_count": len(final),
        "merge_action_count": len(actions),
        "changed_final_proposal_count": sum(
            int(row["merge_action_count"]) > 0 for row in final
        ),
        "absorbed_source_proposal_count": source_count - len(final),
        "executed_round_count": sum(not row["terminal"] for row in rounds),
        "recorded_round_count_including_terminal": len(rounds),
        "round_relation_count": len(relations),
        "refinement_removed_superpoint_count": sum(
            row["removed_by_refinement_count"] for row in actions
        ),
        "merge_iou_contract": "strict point IoU > 0.3; frozen upper triangle per round",
        "consensus_params": CONSENSUS_PARAMS,
        "ground_truth_usage": "none",
        "decision_state": "D2b merge/refine only; no suppress, inclusion cleanup, semantics, or AP.",
    }


def _build_and_publish_scene(scene_name, args):
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

    source = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    source_tracks = source.get("tracks", [])
    _validate_source_point_files(source_tracks, superpoints)
    world = WORLD_2_CAM(
        str(args.dataset_root / scene_name), args.depth_scale, args.config
    )
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations = _load_observations(
        args.automatic_root / scene_name, superpoints, visibility
    )
    final, actions, rounds, relations = iterative_merge_proposals(
        source_tracks, observations, superpoint_sizes
    )
    summary = _scene_summary(
        scene_name, len(source_tracks), final, actions, rounds, relations
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
            chunks = [
                points_by_superpoint[item]
                for item in proposal["superpoint_ids"]
            ]
            points = np.sort(np.concatenate(chunks).astype(np.int64, copy=False))
            np.savez_compressed(point_root / filename, point_indices=points)
            public_tracks.append(
                _public_proposal(
                    proposal,
                    args.output_root / scene_name / "track_points" / filename,
                )
            )
        payload = {
            "scene_name": scene_name,
            "source_track_count": len(source_tracks),
            "track_count": len(public_tracks),
            "merge_action_count": len(actions),
            "tracks": public_tracks,
        }
        (staging / "automatic_tracks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _write_jsonl(staging / "merge_actions.jsonl", actions)
        _write_jsonl(staging / "merge_rounds.jsonl", rounds)
        _write_jsonl(staging / "merge_round_relations.jsonl", relations)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list", "track_root", "automatic_root", "processed_scene_root",
        "dataset_root", "config_path", "output_root",
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
                raise SystemExit(f"resume consensus parameter mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_and_publish_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"{summary['source_proposal_count']} -> {summary['final_proposal_count']}, "
                f"merges {summary['merge_action_count']}",
                flush=True,
            )
        summaries.append(summary)

    totals = Counter()
    for row in summaries:
        totals.update({
            "source_proposal_count": row["source_proposal_count"],
            "final_proposal_count": row["final_proposal_count"],
            "merge_action_count": row["merge_action_count"],
            "changed_final_proposal_count": row["changed_final_proposal_count"],
            "executed_round_count": row["executed_round_count"],
            "refinement_removed_superpoint_count": row["refinement_removed_superpoint_count"],
        })
    payload = {
        "scene_count": len(summaries),
        **dict(totals),
        "merge_iou_contract": "strict point IoU > 0.3; frozen upper triangle per round",
        "consensus_params": CONSENSUS_PARAMS,
        "ground_truth_usage": "none",
        "decision_state": "D2b merge/refine only; no suppress, inclusion cleanup, semantics, or AP.",
        "params": {key: value for key, value in vars(args).items() if key != "config"},
        "scene_summaries": summaries,
    }
    (args.output_root / "details_iterative_merge_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "source_proposal_count": payload.get("source_proposal_count", 0),
        "final_proposal_count": payload.get("final_proposal_count", 0),
        "merge_action_count": payload.get("merge_action_count", 0),
        "executed_round_count": payload.get("executed_round_count", 0),
        "refinement_removed_superpoint_count": payload.get("refinement_removed_superpoint_count", 0),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

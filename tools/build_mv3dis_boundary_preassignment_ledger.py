#!/usr/bin/env python3
"""Build an auditable MV3DIS-style boundary pre-assignment ledger.

This M1b stage consumes the frozen M1a mask-matching ledger.  It resolves only
points whose covering masks have defined consistency scores and a unique
highest score, builds refined-label histograms, and applies MV3DIS Eqs. (6)-(9)
to frozen raw-superpoint contact pairs with the same binary-depth adapter used
by M1a.  It records boundary ownership competition but never assigns, grows,
removes, or mutates a proposal.

Mask NMS, the exact boundary definition, region-affinity aggregation, the
reassignment margin, and stopping rules are not public in the paper.  They are
therefore intentionally absent from this tool.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_details_consensus_proposal_relation_graph import (
    build_superpoint_contact_map,
)
from tools.build_mv3dis_3d_guide_mask_matching_ledger import (
    DEPTH_WEIGHT_ADAPTER,
    _normalize_proposals,
)


ADJACENCY_KNN = 12
ADJACENCY_MAX_DISTANCE = 0.05
MIN_CONTACT_POINTS = 3
MIN_CONTACT_RATIO = 0.02


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


def resolve_pre_nms_point_labels(mask_rows, observation_points):
    """Resolve point labels conservatively from final M1a consistency scores."""
    by_frame = defaultdict(list)
    seen = set()
    for raw in sorted(mask_rows, key=lambda row: int(row["observation_id"])):
        observation_id = int(raw["observation_id"])
        if observation_id in seen:
            raise ValueError("duplicate mask consistency row")
        if observation_id not in observation_points:
            raise ValueError(f"missing point support for observation {observation_id}")
        by_frame[int(raw["frame_index"])].append({
            "observation_id": observation_id,
            "score": raw.get("final_consistency_score"),
            "points": np.unique(
                np.asarray(observation_points[observation_id], dtype=np.int64)
            ),
        })
        seen.add(observation_id)

    resolved_by_frame = {}
    frame_rows = []
    for frame_index in sorted(by_frame):
        candidates = defaultdict(list)
        for mask in by_frame[frame_index]:
            for point_index in mask["points"]:
                candidates[int(point_index)].append(
                    (mask["score"], int(mask["observation_id"]))
                )
        resolved = {}
        states = Counter()
        for point_index in sorted(candidates):
            options = candidates[point_index]
            if any(score is None for score, _ in options):
                states["unknown_undefined_score"] += 1
                continue
            maximum = max(float(score) for score, _ in options)
            winners = sorted(
                observation_id
                for score, observation_id in options
                if float(score) == maximum
            )
            if len(winners) != 1:
                states["unknown_exact_top_score_tie"] += 1
                continue
            resolved[point_index] = winners[0]
            states["resolved_unique_highest_score"] += 1
        resolved_by_frame[frame_index] = resolved
        frame_rows.append({
            "frame_index": frame_index,
            "matched_mask_count": len(by_frame[frame_index]),
            "covered_point_count": len(candidates),
            "resolved_point_count": len(resolved),
            "unknown_undefined_score_point_count": states[
                "unknown_undefined_score"
            ],
            "unknown_exact_top_score_tie_point_count": states[
                "unknown_exact_top_score_tie"
            ],
            "resolution_contract": (
                "pre-NMS unique highest defined consistency; undefined or exact tie is unknown"
            ),
            "gt_usage": "none",
        })
    return resolved_by_frame, frame_rows


def build_superpoint_label_histograms(resolved_by_frame, superpoints):
    histograms = {}
    for frame_index in sorted(resolved_by_frame):
        grouped = defaultdict(Counter)
        for point_index, observation_id in resolved_by_frame[frame_index].items():
            if point_index < 0 or point_index >= len(superpoints):
                raise ValueError("resolved point index is outside the scene")
            grouped[int(superpoints[point_index])][int(observation_id)] += 1
        for superpoint_id in sorted(grouped):
            histograms[(frame_index, superpoint_id)] = dict(grouped[superpoint_id])
    return histograms


def histogram_cosine(left, right):
    left_norm = float(sum(value * value for value in left.values()))
    right_norm = float(sum(value * value for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    dot = float(sum(value * right.get(key, 0) for key, value in left.items()))
    return float(dot / np.sqrt(left_norm * right_norm))


def build_pair_affinity_rows(
    contact_map,
    histograms,
    frame_visible_counts,
    superpoint_sizes,
    frame_depth_weight_means=None,
    depth_weight_adapter=DEPTH_WEIGHT_ADAPTER,
):
    """Apply MV3DIS Eqs. (6)-(9) with binary depth weights."""
    rows = []
    frames = sorted(frame_visible_counts)
    if frame_depth_weight_means is None:
        frame_depth_weight_means = {
            frame_index: {
                int(superpoint_id): 1.0
                for superpoint_id, count in frame_visible_counts[frame_index].items()
                if int(count) > 0
            }
            for frame_index in frames
        }
    for left_id, right_id in sorted(contact_map):
        weighted_sum = 0.0
        weight_sum = 0.0
        frame_evidence = []
        for frame_index in frames:
            left_hist = histograms.get((frame_index, left_id), {})
            right_hist = histograms.get((frame_index, right_id), {})
            frame_affinity = histogram_cosine(left_hist, right_hist)
            if frame_affinity is None:
                continue
            left_visibility = float(
                frame_visible_counts[frame_index].get(left_id, 0)
                / max(1, superpoint_sizes[left_id])
            )
            right_visibility = float(
                frame_visible_counts[frame_index].get(right_id, 0)
                / max(1, superpoint_sizes[right_id])
            )
            left_depth_weight = float(
                frame_depth_weight_means.get(frame_index, {}).get(left_id, 0.0)
            )
            right_depth_weight = float(
                frame_depth_weight_means.get(frame_index, {}).get(right_id, 0.0)
            )
            depth_weight_product = left_depth_weight * right_depth_weight
            weight = left_visibility * right_visibility * depth_weight_product
            if weight <= 0.0:
                continue
            weighted_sum += weight * frame_affinity
            weight_sum += weight
            frame_evidence.append({
                "frame_index": frame_index,
                "frame_affinity": frame_affinity,
                "left_visibility_weight": left_visibility,
                "right_visibility_weight": right_visibility,
                "left_mean_depth_weight": left_depth_weight,
                "right_mean_depth_weight": right_depth_weight,
                "depth_weight_product": depth_weight_product,
                "combined_weight": weight,
            })
        rows.append({
            "left_superpoint_id": int(left_id),
            "right_superpoint_id": int(right_id),
            "boundary_contact_count": int(
                contact_map[(left_id, right_id)]["boundary_contact_count"]
            ),
            "boundary_contact_ratio": float(
                contact_map[(left_id, right_id)]["boundary_contact_ratio"]
            ),
            "defined_frame_count": len(frame_evidence),
            "affinity": (
                float(weighted_sum / weight_sum) if weight_sum > 0.0 else None
            ),
            "affinity_state": (
                "defined_weighted_frame_mean"
                if weight_sum > 0.0
                else "undefined_no_joint_resolved_label_frame"
            ),
            "frame_evidence": frame_evidence,
            "depth_weight_adapter": depth_weight_adapter,
            "gt_usage": "none",
        })
    return rows


def build_boundary_competition_rows(proposals, contact_map, affinity_rows):
    owners = defaultdict(set)
    for proposal in proposals:
        proposal_id = int(proposal["proposal_id"])
        for superpoint_id in proposal["superpoint_ids"]:
            owners[int(superpoint_id)].add(proposal_id)

    neighbors = defaultdict(set)
    for left_id, right_id in contact_map:
        neighbors[int(left_id)].add(int(right_id))
        neighbors[int(right_id)].add(int(left_id))
    affinity_by_pair = {
        (int(row["left_superpoint_id"]), int(row["right_superpoint_id"])): row
        for row in affinity_rows
    }

    rows = []
    all_superpoints = sorted(set(owners) | set(neighbors))
    for superpoint_id in all_superpoints:
        current_owners = sorted(owners.get(superpoint_id, set()))
        adjacent_candidates = set(current_owners)
        for neighbor_id in neighbors.get(superpoint_id, set()):
            adjacent_candidates.update(owners.get(neighbor_id, set()))
        adjacent_candidates = sorted(adjacent_candidates)
        if len(current_owners) > 1:
            state = "current_multi_owner_conflict"
        elif len(current_owners) == 1 and len(adjacent_candidates) > 1:
            state = "owned_boundary_competition"
        elif not current_owners and len(adjacent_candidates) > 1:
            state = "unowned_between_regions"
        else:
            continue

        candidate_evidence = []
        for proposal_id in adjacent_candidates:
            pair_evidence = []
            for neighbor_id in sorted(neighbors.get(superpoint_id, set())):
                if proposal_id not in owners.get(neighbor_id, set()):
                    continue
                key = tuple(sorted((superpoint_id, neighbor_id)))
                affinity = affinity_by_pair[key]["affinity"]
                pair_evidence.append({
                    "neighbor_superpoint_id": neighbor_id,
                    "pair_affinity": affinity,
                    "pair_affinity_defined": affinity is not None,
                })
            defined = [
                row["pair_affinity"] for row in pair_evidence
                if row["pair_affinity"] is not None
            ]
            candidate_evidence.append({
                "proposal_id": proposal_id,
                "neighbor_pair_count": len(pair_evidence),
                "defined_pair_affinity_count": len(defined),
                "observed_pair_affinity_mean": (
                    float(np.mean(defined)) if defined else None
                ),
                "observed_pair_affinity_max": (
                    float(np.max(defined)) if defined else None
                ),
                "pair_evidence": pair_evidence,
                "decision_state": (
                    "diagnostic summaries only; unpublished region aggregation not applied"
                ),
            })
        rows.append({
            "superpoint_id": superpoint_id,
            "current_owner_proposal_ids": current_owners,
            "adjacent_candidate_proposal_ids": adjacent_candidates,
            "competition_state": state,
            "candidate_region_evidence": candidate_evidence,
            "unknown_allowed": True,
            "assignment_action": "none_preassignment_ledger_only",
            "boundary_definition": (
                "adapter: current multi-owner or raw-contact adjacency to multiple D2b proposals"
            ),
            "gt_usage": "none",
        })
    return rows


def _load_observation_points(scene_root, required_ids, point_count, visibility):
    required_ids = set(map(int, required_ids))
    result = {}
    with (scene_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            observation_id = int(row["observation_id"])
            if observation_id not in required_ids:
                continue
            frame_index = int(row["frame_index"])
            with np.load(row["point_indices_path"]) as payload:
                points = np.unique(
                    np.asarray(payload["point_indices"], dtype=np.int64)
                )
            if np.any(points < 0) or np.any(points >= point_count):
                raise ValueError(f"observation {observation_id} has invalid points")
            if visibility is not None and np.any(~visibility[frame_index, points]):
                raise ValueError(f"observation {observation_id} contains non-visible points")
            result[observation_id] = points
    missing = required_ids - set(result)
    if missing:
        raise ValueError(f"missing matched observations: {sorted(missing)[:5]}")
    return result


def _visible_counts(mask_rows, superpoints, visibility):
    result = {}
    for frame_index in sorted({int(row["frame_index"]) for row in mask_rows}):
        ids, counts = np.unique(
            superpoints[np.flatnonzero(visibility[frame_index])], return_counts=True
        )
        result[frame_index] = {
            int(item): int(count) for item, count in zip(ids, counts)
        }
    return result


def _relative_visible_counts(scene_root, superpoints, required_frames):
    counts_by_frame = {}
    depth_means_by_frame = {}
    rows = _read_jsonl(scene_root / "relative_visibility_frames.jsonl")
    by_frame = {int(row["frame_index"]): row for row in rows}
    for frame_index in sorted(required_frames):
        if frame_index not in by_frame:
            raise ValueError(f"relative visibility is missing frame {frame_index}")
        with np.load(by_frame[frame_index]["visibility_path"]) as payload:
            points = np.asarray(payload["point_indices"], dtype=np.int64)
            weights = np.asarray(payload["depth_weights"], dtype=np.float64)
        if points.shape != weights.shape or np.any(weights <= 0.0) or np.any(weights > 1.0):
            raise ValueError(f"relative visibility frame {frame_index} is invalid")
        ids, inverse, counts = np.unique(
            superpoints[points], return_inverse=True, return_counts=True
        )
        weight_sums = np.bincount(inverse, weights=weights, minlength=len(ids))
        counts_by_frame[frame_index] = {
            int(item): int(count) for item, count in zip(ids, counts)
        }
        depth_means_by_frame[frame_index] = {
            int(item): float(total / count)
            for item, total, count in zip(ids, weight_sums, counts)
        }
    return counts_by_frame, depth_means_by_frame


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
    sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    source = json.loads(
        (args.proposal_root / scene_name / "automatic_tracks.json").read_text()
    )
    proposals = _normalize_proposals(source.get("tracks", []), sizes)
    mask_rows = _read_jsonl(
        args.matching_root / scene_name / "mask_coverage_vectors.jsonl"
    )

    world = None
    raw_visibility = None
    visibility = None
    if args.relative_observation_root is None:
        world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
        _, raw_visibility = world.get_mesh_projections()
        visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
        observation_scene_root = args.automatic_root / scene_name
        expected_adapter = DEPTH_WEIGHT_ADAPTER
    else:
        observation_scene_root = args.relative_observation_root / scene_name
        expected_adapter = "mv3dis_relative_depth_rle_reference"
    adapters = {row.get("depth_weight_adapter") for row in mask_rows}
    if adapters != {expected_adapter}:
        raise ValueError(f"{scene_name} matching/visibility adapter mismatch: {adapters}")
    observation_points = _load_observation_points(
        observation_scene_root,
        [row["observation_id"] for row in mask_rows],
        len(superpoints),
        visibility,
    )
    resolved, frame_rows = resolve_pre_nms_point_labels(mask_rows, observation_points)
    histograms = build_superpoint_label_histograms(resolved, superpoints)
    required_frames = {int(row["frame_index"]) for row in mask_rows}
    if args.relative_observation_root is None:
        visible_counts = _visible_counts(mask_rows, superpoints, visibility)
        depth_weight_means = None
    else:
        visible_counts, depth_weight_means = _relative_visible_counts(
            observation_scene_root, superpoints, required_frames
        )
    contact_map = build_superpoint_contact_map(
        processed,
        ADJACENCY_KNN,
        ADJACENCY_MAX_DISTANCE,
        MIN_CONTACT_POINTS,
        MIN_CONTACT_RATIO,
    )
    affinity_rows = build_pair_affinity_rows(
        contact_map,
        histograms,
        visible_counts,
        sizes,
        frame_depth_weight_means=depth_weight_means,
        depth_weight_adapter=expected_adapter,
    )
    competition_rows = build_boundary_competition_rows(
        proposals, contact_map, affinity_rows
    )
    state_counts = Counter(row["competition_state"] for row in competition_rows)
    summary = {
        "scene_name": scene_name,
        "source_proposal_count": len(proposals),
        "source_matched_mask_count": len(mask_rows),
        "resolved_point_count": sum(row["resolved_point_count"] for row in frame_rows),
        "unknown_undefined_score_point_count": sum(
            row["unknown_undefined_score_point_count"] for row in frame_rows
        ),
        "unknown_exact_top_score_tie_point_count": sum(
            row["unknown_exact_top_score_tie_point_count"] for row in frame_rows
        ),
        "contact_pair_count": len(contact_map),
        "defined_affinity_pair_count": sum(
            row["affinity"] is not None for row in affinity_rows
        ),
        "undefined_affinity_pair_count": sum(
            row["affinity"] is None for row in affinity_rows
        ),
        "boundary_competition_superpoint_count": len(competition_rows),
        "competition_state_counts": dict(state_counts),
        "proposal_mutation_count": 0,
        "assignment_action_count": 0,
        "depth_weight_adapter": expected_adapter,
        "ground_truth_usage": "none",
        "decision_state": (
            "M1b pre-assignment ledger only; pre-NMS point explanation and pair affinity, "
            "no region assignment or proposal mutation."
        ),
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "frame_point_resolution_summary.jsonl", frame_rows)
        _write_jsonl(staging / "superpoint_pair_affinity.jsonl", affinity_rows)
        _write_jsonl(staging / "boundary_competition.jsonl", competition_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    del world, raw_visibility, visibility, observation_points, processed
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--relative-observation-root", type=Path)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "proposal_root", "automatic_root", "matching_root",
        "processed_scene_root", "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.relative_observation_root is not None:
        args.relative_observation_root = _resolve(args.relative_observation_root)
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
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"affinity {summary['defined_affinity_pair_count']}/"
                f"{summary['contact_pair_count']}, boundary "
                f"{summary['boundary_competition_superpoint_count']}",
                flush=True,
            )
        summaries.append(summary)
    keys = (
        "source_proposal_count", "source_matched_mask_count", "resolved_point_count",
        "unknown_undefined_score_point_count", "unknown_exact_top_score_tie_point_count",
        "contact_pair_count", "defined_affinity_pair_count", "undefined_affinity_pair_count",
        "boundary_competition_superpoint_count", "proposal_mutation_count",
        "assignment_action_count",
    )
    totals = {key: sum(int(row[key]) for row in summaries) for key in keys}
    states = Counter()
    for row in summaries:
        states.update(row["competition_state_counts"])
    payload = {
        "scene_count": len(summaries),
        **totals,
        "competition_state_counts": dict(states),
        "depth_weight_adapters": sorted(
            {row["depth_weight_adapter"] for row in summaries}
        ),
        "adjacency_contract": {
            "knn": ADJACENCY_KNN,
            "max_distance": ADJACENCY_MAX_DISTANCE,
            "min_contact_points": MIN_CONTACT_POINTS,
            "min_contact_ratio": MIN_CONTACT_RATIO,
        },
        "ground_truth_usage": "none",
        "decision_state": (
            "M1b pre-assignment ledger only; no NMS, region assignment, proposal mutation, "
            "semantics, or AP."
        ),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
        "scene_summaries": summaries,
    }
    (args.output_root / "mv3dis_boundary_preassignment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "defined_affinity_pair_count": payload["defined_affinity_pair_count"],
        "boundary_competition_superpoint_count": payload[
            "boundary_competition_superpoint_count"
        ],
        "proposal_mutation_count": payload["proposal_mutation_count"],
        "assignment_action_count": payload["assignment_action_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

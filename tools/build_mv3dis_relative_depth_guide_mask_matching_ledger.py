#!/usr/bin/env python3
"""Build MV3DIS guide/mask matching from the relative-depth RLE cache.

This is the M1-depth ``paper_reference`` counterpart of the binary-visibility
M1a ledger.  It consumes only the parallel relative-depth cache, frozen D2b
proposals, and raw superpoint IDs.  It applies the same published strict
visibility thresholds and consistency equations without NMS or assignment.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_mv3dis_3d_guide_mask_matching_ledger import (
    _normalize_proposals,
    _validate_and_build_fallback_rows,
    _write_jsonl,
    build_guide_mask_matching,
)
from tools.export_mv3dis_relative_depth_observations import PROJECTION_CONTRACT


DEPTH_WEIGHT_ADAPTER = PROJECTION_CONTRACT


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


def aggregate_weighted_superpoint_support(points, weights, superpoints):
    points = np.asarray(points, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    superpoints = np.asarray(superpoints, dtype=np.int64)
    if points.ndim != 1 or weights.shape != points.shape:
        raise ValueError("point indices and depth weights must be aligned vectors")
    if len(points) and (np.any(points < 0) or np.any(points >= len(superpoints))):
        raise ValueError("point indices are outside the scene")
    if np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("depth weights must lie in (0, 1]")
    ids, inverse, counts = np.unique(
        superpoints[points], return_inverse=True, return_counts=True
    )
    weight_sums = np.bincount(inverse, weights=weights, minlength=len(ids))
    return (
        {int(item): int(count) for item, count in zip(ids, counts)},
        {int(item): float(value) for item, value in zip(ids, weight_sums)},
    )


def _load_relative_cache(scene_name, scene_root, superpoints):
    cache_summary = json.loads((scene_root / "summary.json").read_text())
    if cache_summary.get("projection_contract") != PROJECTION_CONTRACT:
        raise ValueError(f"{scene_name} relative-depth cache contract mismatch")
    frame_visible_counts = {}
    visible_points_by_frame = {}
    for row in _read_jsonl(scene_root / "relative_visibility_frames.jsonl"):
        frame_index = int(row["frame_index"])
        with np.load(row["visibility_path"]) as payload:
            points = np.asarray(payload["point_indices"], dtype=np.int64)
            weights = np.asarray(payload["depth_weights"], dtype=np.float64)
        counts, _ = aggregate_weighted_superpoint_support(points, weights, superpoints)
        if frame_index in frame_visible_counts:
            raise ValueError(f"{scene_name} duplicate relative visibility frame")
        frame_visible_counts[frame_index] = counts
        visible_points_by_frame[frame_index] = set(points.tolist())

    observations = []
    seen = set()
    for row in _read_jsonl(scene_root / "automatic_observations.jsonl"):
        observation_id = int(row["observation_id"])
        frame_index = int(row["frame_index"])
        if observation_id in seen or frame_index not in frame_visible_counts:
            raise ValueError(f"{scene_name} invalid relative observation identity")
        if row.get("projection_contract") != PROJECTION_CONTRACT:
            raise ValueError(f"observation {observation_id} projection contract mismatch")
        with np.load(row["point_indices_path"]) as payload:
            points = np.asarray(payload["point_indices"], dtype=np.int64)
            weights = np.asarray(payload["depth_weights"], dtype=np.float64)
        if not set(points.tolist()) <= visible_points_by_frame[frame_index]:
            raise ValueError(f"observation {observation_id} is outside frame visibility")
        inside_counts, inside_weight_sums = aggregate_weighted_superpoint_support(
            points, weights, superpoints
        )
        observations.append({
            "observation_id": observation_id,
            "frame_id": str(row["frame_id"]),
            "frame_index": frame_index,
            "inside_counts": inside_counts,
            "inside_weight_sums": inside_weight_sums,
        })
        seen.add(observation_id)
    return observations, frame_visible_counts, cache_summary


def _build_scene(scene_name, args):
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
    observations, visible_counts, cache_summary = _load_relative_cache(
        scene_name, args.relative_observation_root / scene_name, superpoints
    )
    ledger = build_guide_mask_matching(
        proposals,
        observations,
        visible_counts,
        sizes,
        depth_weight_adapter=DEPTH_WEIGHT_ADAPTER,
    )
    fallback_rows = _validate_and_build_fallback_rows(
        proposals, superpoints, ledger["guide_rows"]
    )
    state_counts = Counter(row["consistency_state"] for row in ledger["guide_rows"])
    summary = {
        "scene_name": scene_name,
        "source_proposal_count": len(proposals),
        "fallback_proposal_count": len(fallback_rows),
        "source_observation_count": len(observations),
        "matched_mask_count": len(ledger["mask_rows"]),
        "guide_mask_match_count": len(ledger["match_rows"]),
        "guide_state_counts": dict(state_counts),
        "defined_final_mask_score_count": sum(
            row["final_consistency_score"] is not None for row in ledger["mask_rows"]
        ),
        "undefined_final_mask_score_count": sum(
            row["final_consistency_score"] is None for row in ledger["mask_rows"]
        ),
        "relative_observation_point_count": int(
            cache_summary["relative_observation_point_count"]
        ),
        "frame_visibility_contract": "strict > 0.3",
        "mask_visibility_contract": "strict > 0.9",
        "visibility_contract": "strict |zc-d| < 0.05*d",
        "depth_weight_contract": "wpd = 1-|zc-d|/(0.05*d)",
        "depth_weight_adapter": DEPTH_WEIGHT_ADAPTER,
        "proposal_mutation_count": 0,
        "ground_truth_usage": "none",
        "decision_state": (
            "M1-depth paper-reference mask matching only; no NMS, assignment, "
            "proposal mutation, semantics, or AP."
        ),
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "guide_mask_matches.jsonl", ledger["match_rows"])
        _write_jsonl(staging / "mask_coverage_vectors.jsonl", ledger["mask_rows"])
        _write_jsonl(staging / "guide_summaries.jsonl", ledger["guide_rows"])
        _write_jsonl(staging / "proposal_fallback_ledger.jsonl", fallback_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    del processed
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--relative-observation-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "proposal_root", "relative_observation_root",
        "processed_scene_root", "output_root",
    ):
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
            if summary.get("depth_weight_adapter") != DEPTH_WEIGHT_ADAPTER:
                raise SystemExit(f"resume adapter mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: guides "
                f"{summary['source_proposal_count']}, matched masks "
                f"{summary['matched_mask_count']}, links "
                f"{summary['guide_mask_match_count']}",
                flush=True,
            )
        summaries.append(summary)
    keys = (
        "source_proposal_count", "fallback_proposal_count", "source_observation_count",
        "matched_mask_count", "guide_mask_match_count", "defined_final_mask_score_count",
        "undefined_final_mask_score_count", "relative_observation_point_count",
        "proposal_mutation_count",
    )
    states = Counter()
    for row in summaries:
        states.update(row["guide_state_counts"])
    payload = {
        "scene_count": len(summaries),
        **{key: sum(int(row[key]) for row in summaries) for key in keys},
        "guide_state_counts": dict(states),
        "visibility_contract": "strict |zc-d| < 0.05*d",
        "depth_weight_contract": "wpd = 1-|zc-d|/(0.05*d)",
        "depth_weight_adapter": DEPTH_WEIGHT_ADAPTER,
        "ground_truth_usage": "none",
        "decision_state": (
            "M1-depth paper-reference mask matching only; no NMS, assignment, "
            "proposal mutation, semantics, or AP."
        ),
        "params": {key: value for key, value in vars(args).items()},
        "scene_summaries": summaries,
    }
    (args.output_root / "mv3dis_relative_depth_matching_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "source_proposal_count": payload["source_proposal_count"],
        "fallback_proposal_count": payload["fallback_proposal_count"],
        "matched_mask_count": payload["matched_mask_count"],
        "guide_mask_match_count": payload["guide_mask_match_count"],
        "proposal_mutation_count": payload["proposal_mutation_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

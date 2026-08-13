#!/usr/bin/env python3
"""Build a GT-free competition ledger for one track geometry variant.

Every track/native pair is conserved. Positive-overlap relations are written
explicitly and zero-overlap relations are conserved by count. This tool only
records geometry and score facts; it never suppresses, reorders, or mutates a
track or native candidate.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_track_native_competition_ledger import (
    _load_tracks,
    _native_cache_contract,
    _read_scenes,
    _resolve,
    _track_points,
    _write_jsonl,
    track_native_relations,
)


def _build_scene(scene_name, args):
    tracks = _load_tracks(args.track_root, scene_name)
    native_masks = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_masks.npy",
        mmap_mode="r",
    )
    native_scores = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_scores.npy",
        mmap_mode="r",
    )
    if native_masks.ndim != 2:
        raise ValueError(f"{scene_name} native masks are not two-dimensional")
    if len(native_scores) != native_masks.shape[1]:
        raise ValueError(f"{scene_name} native score count differs")
    native_sizes = np.count_nonzero(native_masks, axis=0).astype(np.int64)

    relations = []
    proposal_summaries = []
    for track in tracks:
        points = _track_points(track, native_masks.shape[0])
        rows, summary = track_native_relations(
            points,
            native_masks,
            native_sizes,
            native_scores,
            track["proposal_id"],
            float(track.get("mean_node_quality", 0.0)),
            args.geometry_variant,
        )
        relations.extend(rows)
        proposal_summaries.append(summary)

    complete_pair_count = len(tracks) * native_masks.shape[1]
    if sum(
        int(row["overlap_native_candidate_count"])
        + int(row["disjoint_native_candidate_count"])
        for row in proposal_summaries
    ) != complete_pair_count:
        raise ValueError(f"{scene_name} track/native pairs are not conserved")
    summary = {
        "scene_name": scene_name,
        "geometry_variant": args.geometry_variant,
        "proposal_count": len(tracks),
        "native_candidate_count": int(native_masks.shape[1]),
        "complete_pair_count": complete_pair_count,
        "nonzero_relation_count": len(relations),
        "zero_relation_count": complete_pair_count - len(relations),
        "no_native_overlap_proposal_count": sum(
            int(row["overlap_native_candidate_count"]) == 0
            for row in proposal_summaries
        ),
        "strict_mutual_duplicate_proposal_count": sum(
            int(row["strict_099_mutual_duplicate_count"]) > 0
            for row in proposal_summaries
        ),
        "proposal_mutation_count": 0,
        "native_candidate_mutation_count": 0,
        "candidate_action_count": 0,
        "ground_truth_usage": "none",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "track_native_relations.jsonl", relations)
        _write_jsonl(staging / "proposal_native_summaries.jsonl", proposal_summaries)
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
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--geometry-variant", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "track_root", "native_prediction_cache", "output_root"
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list)
    scenes = all_scenes
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    native_contract = _native_cache_contract(
        args.native_prediction_cache, len(all_scenes), all_scenes
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[done] {index}/{len(scenes)} {scene_name}: proposals "
            f"{summary['proposal_count']}, nonzero relations "
            f"{summary['nonzero_relation_count']}",
            flush=True,
        )
    sum_keys = (
        "proposal_count", "native_candidate_count", "complete_pair_count",
        "nonzero_relation_count", "zero_relation_count",
        "no_native_overlap_proposal_count",
        "strict_mutual_duplicate_proposal_count", "proposal_mutation_count",
        "native_candidate_mutation_count", "candidate_action_count",
    )
    payload = {
        "scene_count": len(summaries),
        "geometry_variant": args.geometry_variant,
        **{key: sum(int(row[key]) for row in summaries) for key in sum_keys},
        "native_cache_contract": native_contract,
        "observation_contract": "strict directional point coverage > 0.99",
        "decision_contract": (
            "ledger only; no suppression, reorder, rescore, or selection"
        ),
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "single_track_native_competition_ledger_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "proposal_count", "complete_pair_count",
        "nonzero_relation_count", "zero_relation_count",
        "no_native_overlap_proposal_count",
        "strict_mutual_duplicate_proposal_count", "candidate_action_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

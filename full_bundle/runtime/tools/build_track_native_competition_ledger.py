#!/usr/bin/env python3
"""Build a GT-free D2b/grow competition ledger against native proposals.

The source and grow variants use exactly the same point-overlap and score
contract.  Every nonzero track/native relation is saved; zero-overlap pairs
are conserved by count.  Details' strict 0.99 directional-coverage facts are
observed but never used here to suppress, reorder, rescore, or select a track.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETAILS_INCLUSION_COVERAGE = 0.99
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


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _native_cache_contract(root, expected_scene_count, scene_names=None):
    manifest_path = root / "native_cache_no_gt_manifest.json"
    if not manifest_path.is_file():
        stream_manifest_path = root.parent / "native_export_manifest.json"
        if not stream_manifest_path.is_file():
            raise ValueError(
                "native cache lacks either the split manifest or the single-scene "
                f"stream manifest: {manifest_path}, {stream_manifest_path}"
            )
        if int(expected_scene_count) != 1:
            raise ValueError(
                "single-scene stream manifest cannot authorize a multi-scene cache"
            )
        if scene_names is None or len(scene_names) != 1:
            raise ValueError(
                "single-scene stream manifest requires exactly one requested scene"
            )
        manifest = json.loads(stream_manifest_path.read_text())
        scene_name = str(scene_names[0])
        if manifest.get("scene_name") != scene_name:
            raise ValueError("single-scene native manifest scene differs")
        if manifest.get("split") != "official_scannet200_train":
            raise ValueError("single-scene native manifest is not official train data")
        if manifest.get("cache_contract") != "Mask3D + YOLO-World only":
            raise ValueError("single-scene native cache contract differs")
        if manifest.get("ground_truth_usage") != "none":
            raise ValueError("single-scene native cache unexpectedly used ground truth")
        if bool(manifest.get("sam_inference")) or bool(manifest.get("d2b_inference")):
            raise ValueError("single-scene native cache contains non-native inference")
        return {
            "mode": "mask3d_yoloworld_only",
            "manifest_path": str(stream_manifest_path),
            "decision_state": "single-scene official-train stream cache",
        }
    manifest = json.loads(manifest_path.read_text())
    mode = manifest.get("candidate_inputs", {}).get("mode")
    if mode != "mask3d_yoloworld_only":
        raise ValueError(f"native cache mode is {mode}, not mask3d_yoloworld_only")
    if int(manifest.get("scene_count", 0)) != int(expected_scene_count):
        raise ValueError("native cache scene count differs from the requested split")
    return {
        "mode": mode,
        "manifest_path": str(manifest_path),
        "decision_state": manifest.get("decision_state"),
    }


def _load_tracks(root, scene_name):
    payload = json.loads((root / scene_name / "automatic_tracks.json").read_text())
    tracks = payload.get("tracks", [])
    ids = [int(row["proposal_id"]) for row in tracks]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{scene_name} proposal IDs are invalid at {root}")
    return tracks


def _track_points(track, point_count):
    path = Path(track["points_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    points = np.unique(np.asarray(np.load(path)["point_indices"], dtype=np.int64))
    if len(points) != int(track.get("point_count", len(points))):
        raise ValueError(f"proposal {track['proposal_id']} point_count differs")
    if not len(points) or points[0] < 0 or points[-1] >= point_count:
        raise ValueError(f"proposal {track['proposal_id']} points are empty or invalid")
    return points


def _score_relation(native_score, track_score):
    if native_score > track_score:
        return "native_higher"
    if native_score < track_score:
        return "track_higher"
    return "equal"


def track_native_relations(
    points,
    native_masks,
    native_sizes,
    native_scores,
    proposal_id,
    track_score,
    geometry_variant,
):
    """Return all positive relations and a threshold-free proposal summary."""
    if native_masks.ndim != 2 or native_masks.shape[0] <= int(points[-1]):
        raise ValueError("native mask shape differs from track points")
    candidate_count = native_masks.shape[1]
    if len(native_sizes) != candidate_count or len(native_scores) != candidate_count:
        raise ValueError("native size/score count differs from masks")
    intersections = np.count_nonzero(native_masks[points], axis=0).astype(np.int64)
    positive = np.flatnonzero(intersections > 0)
    rows = []
    for native_id in positive:
        intersection = int(intersections[native_id])
        native_size = int(native_sizes[native_id])
        union = len(points) + native_size - intersection
        track_coverage = float(intersection / len(points))
        native_coverage = float(intersection / max(1, native_size))
        native_score = float(native_scores[native_id])
        track_inside = track_coverage > DETAILS_INCLUSION_COVERAGE
        native_inside = native_coverage > DETAILS_INCLUSION_COVERAGE
        rows.append({
            "proposal_id": int(proposal_id),
            "geometry_variant": geometry_variant,
            "native_candidate_id": int(native_id),
            "track_point_count": int(len(points)),
            "native_point_count": native_size,
            "intersection_point_count": intersection,
            "union_point_count": int(union),
            "point_iou": float(intersection / max(1, union)),
            "track_inside_native_ratio": track_coverage,
            "native_inside_track_ratio": native_coverage,
            "track_inside_native_strict_099": track_inside,
            "native_inside_track_strict_099": native_inside,
            "mutual_duplicate_strict_099": bool(track_inside and native_inside),
            "track_score": float(track_score),
            "native_score": native_score,
            "score_relation": _score_relation(native_score, track_score),
            "native_contains_track_with_higher_or_equal_score_observed": bool(
                track_inside and native_score >= track_score
            ),
            "ground_truth_usage": "none",
            "decision_state": "observed relation only; no suppression, reorder, or rescore",
        })
    rows.sort(key=lambda row: (row["proposal_id"], row["native_candidate_id"]))

    def best(field):
        if not rows:
            return None
        return min(rows, key=lambda row: (-float(row[field]), row["native_candidate_id"]))

    best_iou = best("point_iou")
    best_track_coverage = best("track_inside_native_ratio")
    best_native_coverage = best("native_inside_track_ratio")
    summary = {
        "proposal_id": int(proposal_id),
        "geometry_variant": geometry_variant,
        "track_point_count": int(len(points)),
        "track_score": float(track_score),
        "native_candidate_count": int(candidate_count),
        "overlap_native_candidate_count": len(rows),
        "disjoint_native_candidate_count": int(candidate_count - len(rows)),
        "best_point_iou": float(best_iou["point_iou"]) if best_iou else 0.0,
        "best_point_iou_native_candidate_id": (
            int(best_iou["native_candidate_id"]) if best_iou else None
        ),
        "best_track_inside_native_ratio": (
            float(best_track_coverage["track_inside_native_ratio"])
            if best_track_coverage else 0.0
        ),
        "best_track_inside_native_candidate_id": (
            int(best_track_coverage["native_candidate_id"])
            if best_track_coverage else None
        ),
        "best_native_inside_track_ratio": (
            float(best_native_coverage["native_inside_track_ratio"])
            if best_native_coverage else 0.0
        ),
        "best_native_inside_track_candidate_id": (
            int(best_native_coverage["native_candidate_id"])
            if best_native_coverage else None
        ),
        "strict_099_track_inside_native_count": sum(
            row["track_inside_native_strict_099"] for row in rows
        ),
        "strict_099_native_inside_track_count": sum(
            row["native_inside_track_strict_099"] for row in rows
        ),
        "strict_099_mutual_duplicate_count": sum(
            row["mutual_duplicate_strict_099"] for row in rows
        ),
        "native_contains_track_with_higher_or_equal_score_count": sum(
            row["native_contains_track_with_higher_or_equal_score_observed"]
            for row in rows
        ),
        "higher_or_equal_score_overlapping_native_count": sum(
            row["native_score"] >= track_score for row in rows
        ),
        "max_overlapping_native_score": (
            max(row["native_score"] for row in rows) if rows else None
        ),
        "ground_truth_usage": "none",
    }
    if summary["overlap_native_candidate_count"] + summary["disjoint_native_candidate_count"] != candidate_count:
        raise ValueError("track/native pair count is not conserved")
    return rows, summary


def compare_proposal(
    source_track,
    grow_track,
    source_points,
    grow_points,
    source_summary,
    grow_summary,
    source_overlap_ids,
    grow_overlap_ids,
):
    proposal_id = int(source_track["proposal_id"])
    if int(grow_track["proposal_id"]) != proposal_id:
        raise ValueError("source/grow proposal IDs differ")
    for key, value in source_track.items():
        if key not in GEOMETRY_KEYS and grow_track.get(key) != value:
            raise ValueError(f"proposal {proposal_id} changed non-geometry field {key}")
    added = np.setdiff1d(grow_points, source_points, assume_unique=True)
    removed = np.setdiff1d(source_points, grow_points, assume_unique=True)
    source_best = float(source_summary["best_point_iou"])
    grow_best = float(grow_summary["best_point_iou"])
    if grow_best > source_best:
        best_iou_change = "increased"
    elif grow_best < source_best:
        best_iou_change = "decreased"
    else:
        best_iou_change = "equal"
    return {
        "proposal_id": proposal_id,
        "track_id": int(source_track.get("track_id", proposal_id)),
        "lineage_proposal_ids": list(
            source_track.get("lineage_proposal_ids", [proposal_id])
        ),
        "track_score": float(source_track.get("mean_node_quality", 0.0)),
        "grow_geometry_changed": bool(len(added) or len(removed)),
        "source_point_count": int(len(source_points)),
        "grow_point_count": int(len(grow_points)),
        "grow_added_point_count": int(len(added)),
        "grow_removed_point_count": int(len(removed)),
        "source_native_relation": source_summary,
        "grow_native_relation": grow_summary,
        "best_native_iou_change": best_iou_change,
        "best_native_iou_delta": grow_best - source_best,
        "added_overlap_native_candidate_ids": sorted(grow_overlap_ids - source_overlap_ids),
        "removed_overlap_native_candidate_ids": sorted(source_overlap_ids - grow_overlap_ids),
        "shared_overlap_native_candidate_count": len(source_overlap_ids & grow_overlap_ids),
        "ground_truth_usage": "none",
        "decision_state": "aligned competition facts only; no candidate action applied",
    }


def _build_scene(scene_name, args):
    source_tracks = _load_tracks(args.source_track_root, scene_name)
    grow_tracks = _load_tracks(args.grow_track_root, scene_name)
    if [row["proposal_id"] for row in source_tracks] != [row["proposal_id"] for row in grow_tracks]:
        raise ValueError(f"{scene_name} source/grow proposal IDs differ")
    native_masks = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
    )
    native_scores = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_scores.npy", mmap_mode="r"
    )
    if native_masks.ndim != 2:
        raise ValueError(f"{scene_name} native masks are not two-dimensional")
    native_sizes = np.count_nonzero(native_masks, axis=0).astype(np.int64)
    if len(native_scores) != native_masks.shape[1]:
        raise ValueError(f"{scene_name} native score count differs")

    source_relations, grow_relations, comparisons = [], [], []
    for source_track, grow_track in zip(source_tracks, grow_tracks):
        source_points = _track_points(source_track, native_masks.shape[0])
        grow_points = _track_points(grow_track, native_masks.shape[0])
        track_score = float(source_track.get("mean_node_quality", 0.0))
        source_rows, source_summary = track_native_relations(
            source_points, native_masks, native_sizes, native_scores,
            source_track["proposal_id"], track_score, "hierarchy_safe_d2b",
        )
        grow_rows, grow_summary = track_native_relations(
            grow_points, native_masks, native_sizes, native_scores,
            grow_track["proposal_id"], track_score, "mv3dis_baseline_adapted_grow",
        )
        source_relations.extend(source_rows)
        grow_relations.extend(grow_rows)
        comparisons.append(compare_proposal(
            source_track,
            grow_track,
            source_points,
            grow_points,
            source_summary,
            grow_summary,
            {row["native_candidate_id"] for row in source_rows},
            {row["native_candidate_id"] for row in grow_rows},
        ))

    pair_count = len(source_tracks) * native_masks.shape[1]
    changed = [row for row in comparisons if row["grow_geometry_changed"]]
    summary = {
        "scene_name": scene_name,
        "proposal_count": len(source_tracks),
        "native_candidate_count": int(native_masks.shape[1]),
        "source_complete_pair_count": pair_count,
        "grow_complete_pair_count": pair_count,
        "source_nonzero_relation_count": len(source_relations),
        "grow_nonzero_relation_count": len(grow_relations),
        "source_zero_relation_count": pair_count - len(source_relations),
        "grow_zero_relation_count": pair_count - len(grow_relations),
        "grow_changed_proposal_count": len(changed),
        "source_no_native_overlap_proposal_count": sum(
            row["source_native_relation"]["overlap_native_candidate_count"] == 0
            for row in comparisons
        ),
        "grow_no_native_overlap_proposal_count": sum(
            row["grow_native_relation"]["overlap_native_candidate_count"] == 0
            for row in comparisons
        ),
        "source_native_dominated_observed_proposal_count": sum(
            row["source_native_relation"]["native_contains_track_with_higher_or_equal_score_count"] > 0
            for row in comparisons
        ),
        "grow_native_dominated_observed_proposal_count": sum(
            row["grow_native_relation"]["native_contains_track_with_higher_or_equal_score_count"] > 0
            for row in comparisons
        ),
        "best_native_iou_change_counts": dict(Counter(
            row["best_native_iou_change"] for row in comparisons
        )),
        "changed_best_native_iou_change_counts": dict(Counter(
            row["best_native_iou_change"] for row in changed
        )),
        "proposal_id_mutation_count": 0,
        "lineage_mutation_count": 0,
        "score_mutation_count": 0,
        "candidate_action_count": 0,
        "ground_truth_usage": "none",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "source_track_native_relations.jsonl", source_relations)
        _write_jsonl(staging / "grow_track_native_relations.jsonl", grow_relations)
        _write_jsonl(staging / "proposal_competition_comparison.jsonl", comparisons)
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
    parser.add_argument("--grow-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "source_track_root", "grow_track_root",
        "native_prediction_cache", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    native_contract = _native_cache_contract(
        args.native_prediction_cache, len(_read_scenes(args.scene_list))
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[done] {index}/{len(scenes)} {scene_name}: proposals "
            f"{summary['proposal_count']}, changed {summary['grow_changed_proposal_count']}",
            flush=True,
        )
    sum_keys = (
        "proposal_count", "native_candidate_count", "source_complete_pair_count",
        "grow_complete_pair_count", "source_nonzero_relation_count",
        "grow_nonzero_relation_count", "source_zero_relation_count",
        "grow_zero_relation_count", "grow_changed_proposal_count",
        "source_no_native_overlap_proposal_count",
        "grow_no_native_overlap_proposal_count",
        "source_native_dominated_observed_proposal_count",
        "grow_native_dominated_observed_proposal_count", "proposal_id_mutation_count",
        "lineage_mutation_count", "score_mutation_count", "candidate_action_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{key: sum(int(row[key]) for row in summaries) for key in sum_keys},
        "best_native_iou_change_counts": dict(sum(
            (Counter(row["best_native_iou_change_counts"]) for row in summaries),
            Counter(),
        )),
        "changed_best_native_iou_change_counts": dict(sum(
            (Counter(row["changed_best_native_iou_change_counts"]) for row in summaries),
            Counter(),
        )),
        "native_cache_contract": native_contract,
        "details_observation_contract": "strict directional point coverage > 0.99",
        "decision_contract": "ledger only; no suppression, reorder, rescore, or selection",
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    (args.output_root / "track_native_competition_ledger_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "proposal_count", "native_candidate_count",
        "source_complete_pair_count", "source_nonzero_relation_count",
        "grow_nonzero_relation_count", "grow_changed_proposal_count",
        "source_no_native_overlap_proposal_count",
        "grow_no_native_overlap_proposal_count",
        "source_native_dominated_observed_proposal_count",
        "grow_native_dominated_observed_proposal_count",
        "changed_best_native_iou_change_counts", "candidate_action_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

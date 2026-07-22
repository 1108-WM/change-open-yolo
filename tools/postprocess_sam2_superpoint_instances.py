#!/usr/bin/env python3
"""Details-Matter-style post-track cleanup for GT-free SAM2 superpoint instances.

SAM2 remains the only cross-view associator.  This tool operates after tracks
have been lifted to IBSp: it iteratively merges substantially overlapping 3D
candidates, recomputes multi-view consensus after every merge, removes
contained instances, and resolves competing superpoints conservatively.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_any3dis_sam2_tracks import _load_scene_arrays, _load_visibility
from lift_sam2_tracks_to_superpoints import _unpack_masks


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _set_overlap(left, right):
    intersection = len(left & right)
    if not intersection:
        return 0.0, 0.0
    return (
        float(intersection / max(1, len(left | right))),
        float(intersection / max(1, min(len(left), len(right)))),
    )


def _candidate_consensus(track_ids, candidate_segments, masks_by_track, visible_points, pixels, superpoints, args):
    """Recompute Details Matter's support-frame / visible-frame consensus after a merge."""
    support = defaultdict(int)
    visible = defaultdict(int)
    coverage_sum = defaultdict(float)
    segments = set(candidate_segments)
    if not segments:
        return set(), {}, {}, {}, 0.0
    for track_id in track_ids:
        for mask, point_indices, frame_pixels in zip(
            masks_by_track[int(track_id)], visible_points, pixels
        ):
            if not len(point_indices) or float(mask.mean()) > float(args.max_frame_area_ratio):
                continue
            frame_segments = superpoints[point_indices]
            membership = np.isin(frame_segments, np.fromiter(segments, dtype=np.int64))
            if not np.any(membership):
                continue
            visible_counts = np.bincount(frame_segments[membership])
            inside = mask[frame_pixels[:, 1], frame_pixels[:, 0]] & membership
            inside_counts = np.bincount(frame_segments[inside], minlength=len(visible_counts))
            for segment_id in np.flatnonzero(visible_counts >= int(args.min_visible_points_per_superpoint)):
                ratio = float(inside_counts[segment_id] / visible_counts[segment_id])
                visible[int(segment_id)] += 1
                if ratio >= float(args.frame_superpoint_coverage):
                    support[int(segment_id)] += 1
                    coverage_sum[int(segment_id)] += ratio
    kept = {
        segment_id
        for segment_id in segments
        if support[segment_id] >= int(args.min_support_frames)
        and support[segment_id] / max(1, visible[segment_id]) >= float(args.min_consensus_rate)
        and coverage_sum[segment_id] / max(1, support[segment_id]) >= float(args.mean_superpoint_coverage)
    }
    score = float(sum(coverage_sum[segment_id] for segment_id in kept))
    return kept, dict(support), dict(visible), dict(coverage_sum), score


def _refine(candidate, masks_by_track, visible_points, pixels, superpoints, args):
    segments, support, visible, coverage, score = _candidate_consensus(
        candidate["track_ids"], candidate["superpoint_ids"], masks_by_track,
        visible_points, pixels, superpoints, args
    )
    result = dict(candidate)
    result.update(
        superpoint_ids=set(segments),
        support=support,
        visible=visible,
        coverage=coverage,
        support_score=score,
    )
    return result


def _is_valid(candidate, sizes, args):
    return (
        len(candidate["superpoint_ids"]) >= int(args.min_superpoints_per_instance)
        and sum(int(sizes[item]) for item in candidate["superpoint_ids"]) >= int(args.min_instance_points)
    )


def _iterative_merge(candidates, masks_by_track, visible_points, pixels, superpoints, sizes, args):
    merges = []
    candidates = [_refine(item, masks_by_track, visible_points, pixels, superpoints, args) for item in candidates]
    candidates = [item for item in candidates if _is_valid(item, sizes, args)]
    while True:
        selected_pair = None
        best_iou = float(args.merge_iou)
        for left_id in range(len(candidates)):
            for right_id in range(left_id + 1, len(candidates)):
                iou, _ = _set_overlap(candidates[left_id]["superpoint_ids"], candidates[right_id]["superpoint_ids"])
                if iou >= best_iou:
                    selected_pair, best_iou = (left_id, right_id), iou
        if selected_pair is None:
            break
        left_id, right_id = selected_pair
        left, right = candidates[left_id], candidates[right_id]
        merged = {
            "track_ids": sorted(set(left["track_ids"]) | set(right["track_ids"])),
            "superpoint_ids": set(left["superpoint_ids"]) | set(right["superpoint_ids"]),
        }
        merged = _refine(merged, masks_by_track, visible_points, pixels, superpoints, args)
        merges.append(
            {
                "left_track_ids": left["track_ids"],
                "right_track_ids": right["track_ids"],
                "iou_before_merge": float(best_iou),
                "merged_superpoint_count": len(merged["superpoint_ids"]),
            }
        )
        candidates = [item for index, item in enumerate(candidates) if index not in selected_pair]
        if _is_valid(merged, sizes, args):
            candidates.append(merged)
    return candidates, merges


def _remove_contained(candidates, args):
    removed = []
    keep = [True] * len(candidates)
    for small_id, small in enumerate(candidates):
        if not keep[small_id]:
            continue
        for large_id, large in enumerate(candidates):
            if small_id == large_id or not keep[large_id]:
                continue
            if len(small["superpoint_ids"]) > len(large["superpoint_ids"]):
                continue
            _, inclusion = _set_overlap(small["superpoint_ids"], large["superpoint_ids"])
            if inclusion >= float(args.containment_threshold):
                keep[small_id] = False
                removed.append(
                    {
                        "removed_track_ids": small["track_ids"],
                        "container_track_ids": large["track_ids"],
                        "inclusion": float(inclusion),
                    }
                )
                break
    return [item for index, item in enumerate(candidates) if keep[index]], removed


def _clean_competing(candidates, masks_by_track, visible_points, pixels, superpoints, sizes, args):
    owners = defaultdict(list)
    for candidate_id, candidate in enumerate(candidates):
        for segment_id in candidate["superpoint_ids"]:
            owners[int(segment_id)].append(candidate_id)
    ambiguous = []
    for segment_id, candidate_ids in owners.items():
        if len(candidate_ids) < 2:
            continue
        ranked = sorted(
            candidate_ids,
            key=lambda item: candidates[item]["coverage"].get(segment_id, 0.0),
            reverse=True,
        )
        best, second = ranked[:2]
        best_score = candidates[best]["coverage"].get(segment_id, 0.0)
        second_score = candidates[second]["coverage"].get(segment_id, 0.0)
        if second_score >= best_score * float(args.ambiguity_ratio):
            ambiguous.append(int(segment_id))
            for candidate_id in candidate_ids:
                candidates[candidate_id]["superpoint_ids"].discard(segment_id)
        else:
            for candidate_id in ranked[1:]:
                candidates[candidate_id]["superpoint_ids"].discard(segment_id)
    refined = [_refine(item, masks_by_track, visible_points, pixels, superpoints, args) for item in candidates]
    return [item for item in refined if _is_valid(item, sizes, args)], ambiguous


def process_scene(scene_name, args):
    lift_scene = Path(args.lift_root) / scene_name
    track_scene = Path(args.track_root) / scene_name
    lift_summary = json.loads((lift_scene / "summary.json").read_text())
    track_summary = json.loads((track_scene / "summary.json").read_text())
    initial = _read_jsonl(lift_scene / "instances.jsonl")
    track_records = {int(item["track_id"]): item for item in _read_jsonl(track_scene / "tracks.jsonl")}
    points, superpoints, intrinsics, source_array = _load_scene_arrays(
        Path(args.dataset_root) / scene_name, args.superpoint_root, scene_name
    )
    visible_points, pixels, _ = _load_visibility(
        Path(args.dataset_root) / scene_name, track_summary["frame_ids"], points, intrinsics, args.depth_tolerance
    )
    sizes = np.bincount(superpoints, minlength=int(superpoints.max(initial=-1)) + 1).astype(np.int64)
    masks_by_track = {track_id: _unpack_masks(record["mask_path"]) for track_id, record in track_records.items()}
    candidates = [
        {"track_ids": [int(record["track_id"])], "superpoint_ids": set(record["superpoint_ids"])}
        for record in initial
    ]
    merged, merges = _iterative_merge(
        candidates, masks_by_track, visible_points, pixels, superpoints, sizes, args
    )
    uncontained, contained = _remove_contained(merged, args)
    cleaned, ambiguous = _clean_competing(
        uncontained, masks_by_track, visible_points, pixels, superpoints, sizes, args
    )
    cleaned.sort(key=lambda item: (-item["support_score"], -len(item["superpoint_ids"]), item["track_ids"]))
    output_scene = Path(args.output_root) / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)
    records = []
    for instance_id, candidate in enumerate(cleaned):
        point_indices = np.flatnonzero(np.isin(superpoints, list(candidate["superpoint_ids"]))).astype(np.int32)
        point_path = output_scene / f"instance{instance_id:04d}_points.npz"
        np.savez_compressed(point_path, point_indices=point_indices)
        records.append(
            {
                "instance_id": instance_id,
                "source_track_ids": candidate["track_ids"],
                "superpoint_ids": sorted(int(item) for item in candidate["superpoint_ids"]),
                "superpoint_count": len(candidate["superpoint_ids"]),
                "point_count": int(len(point_indices)),
                "support_score": float(candidate["support_score"]),
                "point_indices_path": str(point_path),
            }
        )
    (output_scene / "instances.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    payload = {
        "scene_name": scene_name,
        "source_lift_root": str(lift_scene),
        "source_superpoint_array": str(source_array),
        "superpoint_column": 9,
        "input_instance_count": len(initial),
        "merge_count": len(merges),
        "merge_operations": merges,
        "contained_removals": contained,
        "ambiguous_superpoint_count": len(ambiguous),
        "output_instance_count": len(records),
        "params": vars(args),
    }
    (output_scene / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Apply Details-Matter-style cleanup to SAM2-lifted IBSp instances.")
    parser.add_argument("--lift_root", required=True)
    parser.add_argument("--track_root", required=True)
    parser.add_argument("--superpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_root", default="data/scannet200")
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--depth_tolerance", type=float, default=0.10)
    parser.add_argument("--frame_superpoint_coverage", type=float, default=0.50)
    parser.add_argument("--mean_superpoint_coverage", type=float, default=0.55)
    parser.add_argument("--min_visible_points_per_superpoint", type=int, default=3)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--min_consensus_rate", type=float, default=0.30)
    parser.add_argument("--max_frame_area_ratio", type=float, default=0.65)
    parser.add_argument("--min_superpoints_per_instance", type=int, default=1)
    parser.add_argument("--min_instance_points", type=int, default=100)
    parser.add_argument("--merge_iou", type=float, default=0.30)
    parser.add_argument("--containment_threshold", type=float, default=0.90)
    parser.add_argument("--ambiguity_ratio", type=float, default=0.90)
    args = parser.parse_args()
    if args.scene_names:
        scenes = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.scene_split:
        scenes = [line.strip() for line in Path(args.scene_split).read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Provide --scene_names or --scene_split.")
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    summaries = [process_scene(scene_name, args) for scene_name in tqdm(scenes)]
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "postprocess_summary.json").write_text(
        json.dumps({"scenes": summaries, "params": vars(args)}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

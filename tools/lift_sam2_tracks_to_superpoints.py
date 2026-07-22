#!/usr/bin/env python3
"""Lift GT-free SAM2 tracks to IBSp superpoint consensus instances.

This is the bridge between Any3DIS-style 2D tracking and the later
Details-Matter-style instance cleanup.  It reads only exported SAM2 masks and
inference-time geometry/superpoints; no semantic or instance GT columns are
read from the ScanNet processed arrays.
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


def _unpack_masks(path):
    data = np.load(path)
    shape = tuple(int(value) for value in data["mask_shape"])
    pixels = int(np.prod(shape[1:]))
    return np.unpackbits(data["packed_masks"], axis=1, count=pixels).reshape(shape).astype(bool)


def _remove_same_frame_track_overlaps(masks_by_track, min_overlap_pixels):
    """删除同帧多轨迹共有像素，避免在三维提升前把歧义区域归属给任一实例。"""
    if not masks_by_track:
        return [], {"total_removed_pixels": 0, "affected_frame_count": 0, "per_track": []}
    masks = np.stack(masks_by_track, axis=0).astype(bool, copy=True)
    if masks.ndim != 4:
        raise ValueError(f"Expected [track, frame, height, width] masks, got {masks.shape}")
    per_track_removed = np.zeros(masks.shape[0], dtype=np.int64)
    per_track_frames = np.zeros(masks.shape[0], dtype=np.int64)
    affected_frames = 0
    for frame_index in range(masks.shape[1]):
        overlap = masks[:, frame_index].sum(axis=0) >= 2
        if int(overlap.sum()) < int(min_overlap_pixels):
            continue
        affected_frames += 1
        for track_index in range(masks.shape[0]):
            removed = masks[track_index, frame_index] & overlap
            removed_count = int(removed.sum())
            if removed_count:
                masks[track_index, frame_index, removed] = False
                per_track_removed[track_index] += removed_count
                per_track_frames[track_index] += 1
    return [masks[index] for index in range(len(masks))], {
        "total_removed_pixels": int(per_track_removed.sum()),
        "affected_frame_count": int(affected_frames),
        "per_track": [
            {
                "removed_pixel_count": int(removed),
                "affected_frame_count": int(frames),
            }
            for removed, frames in zip(per_track_removed, per_track_frames)
        ],
    }


def _set_overlap(left, right):
    intersection = len(left & right)
    if not intersection:
        return 0.0, 0.0
    return (
        float(intersection / max(1, len(left | right))),
        float(intersection / max(1, min(len(left), len(right)))),
    )


def _deduplicate(records, iou_threshold, containment_threshold):
    kept = []
    removed = []
    order = sorted(
        records,
        key=lambda item: (-float(item["support_score"]), -int(item["point_count"]), int(item["track_id"])),
    )
    for record in order:
        segments = set(record["superpoint_ids"])
        duplicate_of = None
        for other in kept:
            iou, containment = _set_overlap(segments, set(other["superpoint_ids"]))
            if iou >= float(iou_threshold) or containment >= float(containment_threshold):
                duplicate_of = int(other["track_id"])
                break
        if duplicate_of is None:
            kept.append(record)
        else:
            removed.append({"track_id": int(record["track_id"]), "duplicate_of": duplicate_of})
    return kept, removed


def _track_superpoint_support(masks, visible_points, pixels, superpoints, args, frame_weights=None):
    """Build per-frame lifted candidates and Details-Matter visibility statistics."""
    support_frames = defaultdict(float)
    visible_frames = defaultdict(float)
    coverage_sum = defaultdict(float)
    candidates_by_frame = []
    for frame_index, mask in enumerate(masks):
        weight = 1.0 if frame_weights is None else float(frame_weights[frame_index])
        if weight <= 0.0:
            candidates_by_frame.append(set())
            continue
        if mask.mean() > float(args.max_frame_area_ratio):
            candidates_by_frame.append(set())
            continue
        indices = visible_points[frame_index]
        if not len(indices):
            candidates_by_frame.append(set())
            continue
        frame_pixels = pixels[frame_index]
        segment_ids = superpoints[indices]
        visible_counts = np.bincount(segment_ids)
        selected = mask[frame_pixels[:, 1], frame_pixels[:, 0]]
        masked_counts = np.bincount(segment_ids[selected], minlength=len(visible_counts))
        frame_candidates = set()
        for segment_id in np.flatnonzero(visible_counts >= int(args.min_visible_points_per_superpoint)):
            ratio = float(masked_counts[segment_id] / visible_counts[segment_id])
            visible_frames[int(segment_id)] += weight
            if ratio >= float(args.frame_superpoint_coverage):
                support_frames[int(segment_id)] += weight
                coverage_sum[int(segment_id)] += ratio * weight
                frame_candidates.add(int(segment_id))
        candidates_by_frame.append(frame_candidates)
    return candidates_by_frame, support_frames, visible_frames, coverage_sum


def _any3dis_greedy_maskopt(candidates_by_frame, masks, visible_points, pixels, superpoints, frame_weights=None):
    """Any3DIS Algorithm 1: accept a frame's new lifted SPs only if global mask alignment improves."""
    selected = set()

    def objective(segment_ids):
        if not segment_ids:
            return 0
        value = 0
        for frame_index, (mask, indices, frame_pixels) in enumerate(zip(masks, visible_points, pixels)):
            weight = 1.0 if frame_weights is None else float(frame_weights[frame_index])
            if weight <= 0.0:
                continue
            if not len(indices):
                continue
            belongs = np.isin(superpoints[indices], np.fromiter(segment_ids, dtype=np.int64))
            if not np.any(belongs):
                continue
            inside = mask[frame_pixels[belongs, 1], frame_pixels[belongs, 0]].sum()
            value += weight * (int(inside) - int(belongs.sum() - inside))
        return value

    current_score = 0
    accepted_frames = []
    for frame_index, candidates in enumerate(candidates_by_frame):
        proposal = selected | candidates
        proposal_score = objective(proposal)
        if proposal_score > current_score:
            selected = proposal
            current_score = proposal_score
            accepted_frames.append(frame_index)
    return selected, current_score, accepted_frames


def _details_consensus_filter(candidate_segments, support_frames, visible_frames, coverage_sum, args):
    """Keep only superpoints with reliable support among the frames where they are visible."""
    selected = {
        segment_id
        for segment_id in candidate_segments
        if support_frames[segment_id] >= int(args.min_support_frames)
        and support_frames[segment_id] / max(1, visible_frames[segment_id]) >= float(args.min_consensus_rate)
        and coverage_sum[segment_id] / max(1, support_frames[segment_id]) >= float(args.mean_superpoint_coverage)
    }
    return selected


def _track_superpoint_consensus(masks, visible_points, pixels, superpoints, args, frame_weights=None):
    candidates_by_frame, support_frames, visible_frames, coverage_sum = _track_superpoint_support(
        masks, visible_points, pixels, superpoints, args, frame_weights
    )
    if args.mask_optimization == "any3dis_dp":
        candidates, objective_score, accepted_frames = _any3dis_greedy_maskopt(
            candidates_by_frame, masks, visible_points, pixels, superpoints, frame_weights
        )
    else:
        candidates = set().union(*candidates_by_frame) if candidates_by_frame else set()
        objective_score, accepted_frames = None, list(range(len(candidates_by_frame)))
    selected = _details_consensus_filter(candidates, support_frames, visible_frames, coverage_sum, args)
    return selected, support_frames, visible_frames, coverage_sum, objective_score, accepted_frames


def _reobservation_frame_weights(frame_count, track, enabled, rejected_weight):
    """将独立重观测分歧转为软证据权重，避免一次分歧直接删除轨迹。"""
    rejected = [int(value) for value in track.get("reobservation_rejected_frame_indices", [])]
    weights = np.ones(frame_count, dtype=np.float32)
    rejected = [index for index in rejected if 0 <= index < frame_count]
    if enabled:
        weights[rejected] = float(rejected_weight)
    return weights, rejected


def _read_tracks(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def lift_scene(scene_name, args):
    track_scene = Path(args.track_root) / scene_name
    summary_path = track_scene / "summary.json"
    tracks_path = track_scene / "tracks.jsonl"
    if not summary_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError(f"Missing SAM2 track metadata for {scene_name}: {track_scene}")
    source = json.loads(summary_path.read_text())
    frame_ids = source["frame_ids"]
    points, superpoints, intrinsics, source_array = _load_scene_arrays(
        Path(args.dataset_root) / scene_name, args.superpoint_root, scene_name
    )
    visible_points, pixels, _ = _load_visibility(
        Path(args.dataset_root) / scene_name, frame_ids, points, intrinsics, args.depth_tolerance
    )
    sizes = np.bincount(superpoints, minlength=int(superpoints.max(initial=-1)) + 1).astype(np.int64)
    output_scene = Path(args.output_root) / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)
    tracks = _read_tracks(tracks_path)
    masks_by_track = [_unpack_masks(track["mask_path"]) for track in tracks]
    if any(masks.shape[0] != len(frame_ids) for masks in masks_by_track):
        raise ValueError(f"Track frame count mismatch in {track_scene}")
    if args.same_frame_overlap_cleanup:
        masks_by_track, overlap_summary = _remove_same_frame_track_overlaps(
            masks_by_track, args.same_frame_overlap_min_pixels
        )
    else:
        overlap_summary = {
            "total_removed_pixels": 0,
            "affected_frame_count": 0,
            "per_track": [{"removed_pixel_count": 0, "affected_frame_count": 0} for _ in tracks],
        }
    raw_records = []
    for track, masks, overlap_stats in tqdm(
        zip(tracks, masks_by_track, overlap_summary["per_track"]),
        total=len(tracks),
        desc=f"lift {scene_name}",
        leave=False,
    ):
        if masks.shape[0] != len(frame_ids):
            raise ValueError(f"Track frame count mismatch: {track['mask_path']}")
        frame_weights, rejected_reobservation_frames = _reobservation_frame_weights(
            len(masks), track, args.use_reobservation_confirmation, args.reobservation_rejected_frame_weight
        )
        area_ratios = masks.mean(axis=(1, 2))
        nonempty = int((masks.sum(axis=(1, 2)) >= int(args.min_mask_area)).sum())
        segments, support_frames, visible_frames, coverage_sum, objective_score, accepted_frames = _track_superpoint_consensus(
            masks, visible_points, pixels, superpoints, args, frame_weights
        )
        point_mask = np.isin(superpoints, np.fromiter(segments, dtype=np.int64)) if segments else np.zeros(len(points), dtype=bool)
        point_indices = np.flatnonzero(point_mask).astype(np.int32)
        support_score = float(sum(coverage_sum[segment] for segment in segments))
        record = {
            "track_id": int(track["track_id"]),
            "seed_superpoint_id": int(track["seed_superpoint_id"]),
            "pivot_frame_id": track["pivot_frame_id"],
            "nonempty_frames": nonempty,
            "max_frame_area_ratio": float(area_ratios.max(initial=0.0)),
            "same_frame_overlap_removed_pixels": int(overlap_stats["removed_pixel_count"]),
            "same_frame_overlap_affected_frames": int(overlap_stats["affected_frame_count"]),
            "reobservation_checked_frame_count": int(len(track.get("reobservations", []))),
            "reobservation_rejected_frame_count": int(len(rejected_reobservation_frames)),
            "reobservation_rejected_frame_indices": rejected_reobservation_frames,
            "superpoint_ids": sorted(int(segment) for segment in segments),
            "superpoint_support_frames": {str(key): float(value) for key, value in sorted(support_frames.items()) if key in segments},
            "superpoint_visible_frames": {str(key): float(value) for key, value in sorted(visible_frames.items()) if key in segments},
            "superpoint_consensus_rate": {
                str(key): float(support_frames[key] / max(1, visible_frames[key]))
                for key in sorted(segments)
            },
            "mask_optimization_objective": objective_score,
            "mask_optimization_accepted_frame_indices": accepted_frames,
            "support_score": support_score,
            "point_count": int(len(point_indices)),
            "accepted_before_dedup": bool(
                nonempty >= int(args.min_track_frames)
                and float(area_ratios.max(initial=0.0)) <= float(args.max_frame_area_ratio)
                and len(segments) >= int(args.min_superpoints_per_instance)
                and len(point_indices) >= int(args.min_instance_points)
            ),
        }
        raw_records.append(record)
        np.savez_compressed(output_scene / f"track{record['track_id']:04d}_points.npz", point_indices=point_indices)
    eligible = [record for record in raw_records if record["accepted_before_dedup"]]
    kept, removed = _deduplicate(eligible, args.dedup_iou, args.dedup_containment)
    for record in kept:
        record["point_indices_path"] = str(output_scene / f"track{record['track_id']:04d}_points.npz")
    (output_scene / "lifted_tracks.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in raw_records)
    )
    (output_scene / "instances.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept)
    )
    payload = {
        "scene_name": scene_name,
        "source_track_root": str(track_scene),
        "source_superpoint_array": str(source_array),
        "superpoint_column": 9,
        "raw_track_count": len(raw_records),
        "eligible_track_count": len(eligible),
        "kept_instance_count": len(kept),
        "deduplicated_tracks": removed,
        "same_frame_overlap_cleanup": {
            "enabled": bool(args.same_frame_overlap_cleanup),
            "min_overlap_pixels": int(args.same_frame_overlap_min_pixels),
            **{key: value for key, value in overlap_summary.items() if key != "per_track"},
        },
        "params": vars(args),
    }
    (output_scene / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Lift SAM2 masks to GT-free IBSp consensus instances.")
    parser.add_argument("--track_root", required=True)
    parser.add_argument("--superpoint_root", required=True)
    parser.add_argument("--dataset_root", default="data/scannet200")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--depth_tolerance", type=float, default=0.10)
    parser.add_argument("--frame_superpoint_coverage", type=float, default=0.50)
    parser.add_argument("--mean_superpoint_coverage", type=float, default=0.55)
    parser.add_argument("--min_visible_points_per_superpoint", type=int, default=3)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--min_consensus_rate", type=float, default=0.30)
    parser.add_argument("--mask_optimization", choices=("any3dis_dp", "union"), default="any3dis_dp")
    parser.add_argument("--same_frame_overlap_cleanup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--same_frame_overlap_min_pixels", type=int, default=32)
    parser.add_argument("--use_reobservation_confirmation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reobservation_rejected_frame_weight", type=float, default=0.50)
    parser.add_argument("--min_mask_area", type=int, default=64)
    parser.add_argument("--min_track_frames", type=int, default=3)
    parser.add_argument("--max_frame_area_ratio", type=float, default=0.65)
    parser.add_argument("--min_superpoints_per_instance", type=int, default=1)
    parser.add_argument("--min_instance_points", type=int, default=100)
    parser.add_argument("--dedup_iou", type=float, default=0.70)
    parser.add_argument("--dedup_containment", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 <= args.reobservation_rejected_frame_weight <= 1.0:
        raise ValueError("--reobservation_rejected_frame_weight 必须在 0 到 1 之间。")
    if args.scene_names:
        scenes = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.scene_split:
        scenes = [line.strip() for line in Path(args.scene_split).read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Provide --scene_names or --scene_split.")
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    summaries = [lift_scene(scene, args) for scene in scenes]
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "sam2_superpoint_lift_summary.json").write_text(
        json.dumps({"scenes": summaries, "params": vars(args)}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

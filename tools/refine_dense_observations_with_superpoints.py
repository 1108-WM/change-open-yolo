#!/usr/bin/env python3
"""Details-Matter-style superpoint instance refinement for dense observations.

The input is exported evidence, not final predictions.  The tool builds
category-agnostic observation tracks from shared reliable superpoints, removes
duplicate tracks, assigns ambiguous superpoints conservatively, and writes
cleaned 3D instance seeds for diagnosis or later fusion experiments.
"""

import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


class UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _superpoint_file(superpoint_root, scene_name):
    scene_id = scene_name.replace("scene", "")
    return Path(superpoint_root) / scene_name / f"{scene_id}.npy"


def _support_from_points(point_indices, point_superpoints, segment_sizes, core_coverage, partial_coverage):
    valid = np.asarray(point_indices, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < len(point_superpoints))]
    if len(valid) == 0:
        return {}, set(), set()
    segment_ids, counts = np.unique(point_superpoints[valid], return_counts=True)
    coverage = {
        int(segment_id): float(count / max(1, segment_sizes[int(segment_id)]))
        for segment_id, count in zip(segment_ids, counts)
    }
    core = {segment_id for segment_id, ratio in coverage.items() if ratio >= float(core_coverage)}
    partial = {
        segment_id
        for segment_id, ratio in coverage.items()
        if float(partial_coverage) <= ratio < float(core_coverage)
    }
    return coverage, core, partial


def prepare_observation_support(observations, point_superpoints, core_coverage, partial_coverage):
    """Attach superpoint coverage to observations with existing point-index files."""
    num_segments = int(point_superpoints.max(initial=-1)) + 1
    segment_sizes = np.bincount(point_superpoints, minlength=num_segments).astype(np.int64)
    prepared = []
    for observation in observations:
        point_path = observation.get("point_indices_path")
        if not point_path or not osp.exists(point_path):
            continue
        data = np.load(point_path)
        if "point_indices" not in data:
            continue
        coverage, core, partial = _support_from_points(
            data["point_indices"], point_superpoints, segment_sizes, core_coverage, partial_coverage
        )
        if not core:
            continue
        item = dict(observation)
        item["superpoint_coverage"] = coverage
        item["core_superpoints"] = core
        item["partial_superpoints"] = partial
        item["confidence"] = float(item.get("priority", 0.0))
        prepared.append(item)
    return prepared, segment_sizes


def _set_overlap(left, right):
    intersection = len(left & right)
    if not intersection:
        return 0.0, 0.0
    return float(intersection / max(1, len(left | right))), float(intersection / max(1, min(len(left), len(right))))


def match_observation_tracks(observations, min_core_iou, min_core_containment, max_candidates_per_superpoint):
    """Connect cross-view observations that have strong shared core superpoints."""
    union_find = UnionFind(len(observations))
    bucket = defaultdict(list)
    checked_pairs = set()
    relation_count = 0
    for current_id, current in enumerate(observations):
        candidate_ids = set()
        for segment_id in current["core_superpoints"]:
            candidate_ids.update(bucket[int(segment_id)])
        for other_id in candidate_ids:
            if observations[other_id].get("frame_index") == current.get("frame_index"):
                continue
            pair = (min(other_id, current_id), max(other_id, current_id))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            iou, containment = _set_overlap(
                observations[other_id]["core_superpoints"], current["core_superpoints"]
            )
            if iou >= float(min_core_iou) or containment >= float(min_core_containment):
                union_find.union(other_id, current_id)
                relation_count += 1
        for segment_id in current["core_superpoints"]:
            segment_bucket = bucket[int(segment_id)]
            segment_bucket.append(current_id)
            if len(segment_bucket) > int(max_candidates_per_superpoint):
                segment_bucket.sort(key=lambda item: observations[item]["confidence"], reverse=True)
                del segment_bucket[int(max_candidates_per_superpoint) :]
    tracks = defaultdict(list)
    for observation_id in range(len(observations)):
        tracks[union_find.find(observation_id)].append(observation_id)
    return list(tracks.values()), relation_count


def _aggregate_track(track_ids, observations):
    support = defaultdict(float)
    frame_support = defaultdict(set)
    class_votes = defaultdict(float)
    observation_ids = []
    for observation_id in track_ids:
        observation = observations[observation_id]
        confidence = max(1e-5, float(observation["confidence"]))
        observation_ids.append(int(observation.get("observation_id", observation_id)))
        class_votes[str(observation.get("class_name", "unknown"))] += confidence
        for segment_id, coverage in observation["superpoint_coverage"].items():
            support[int(segment_id)] += confidence * float(coverage)
            frame_support[int(segment_id)].add(int(observation.get("frame_index", -1)))
    score = float(sum(max(1e-5, observations[item]["confidence"]) for item in track_ids))
    return {
        "observation_indices": track_ids,
        "observation_ids": sorted(observation_ids),
        "support": support,
        "frame_support": frame_support,
        "class_votes": class_votes,
        "score": score,
        "frame_count": len({int(observations[item].get("frame_index", -1)) for item in track_ids}),
    }


def select_track_superpoints(track, observations, min_support_views, singleton_min_confidence):
    selected = set()
    for segment_id, support in track["support"].items():
        frame_count = len({frame for frame in track["frame_support"][segment_id] if frame >= 0})
        if frame_count >= int(min_support_views):
            selected.add(int(segment_id))
            continue
        if any(
            int(segment_id) in observations[item]["core_superpoints"]
            and float(observations[item]["confidence"]) >= float(singleton_min_confidence)
            for item in track["observation_indices"]
        ):
            selected.add(int(segment_id))
    return selected


def deduplicate_tracks(tracks, selected_segments, duplicate_iou):
    """Keep the highest-confidence of near-identical 3D superpoint tracks."""
    order = sorted(range(len(tracks)), key=lambda item: (-tracks[item]["score"], -tracks[item]["frame_count"], item))
    kept = []
    removed = set()
    for track_id in order:
        if not selected_segments[track_id]:
            removed.add(track_id)
            continue
        duplicate = False
        for kept_id in kept:
            iou, _ = _set_overlap(selected_segments[track_id], selected_segments[kept_id])
            if iou >= float(duplicate_iou):
                duplicate = True
                break
        if duplicate:
            removed.add(track_id)
        else:
            kept.append(track_id)
    return kept, removed


def clean_competing_superpoints(track_ids, tracks, selected_segments, ambiguity_ratio):
    """Assign each superpoint to one track, dropping ties rather than guessing."""
    owners = defaultdict(list)
    for track_id in track_ids:
        for segment_id in selected_segments[track_id]:
            owners[int(segment_id)].append(track_id)
    cleaned = {track_id: set(selected_segments[track_id]) for track_id in track_ids}
    ambiguous = set()
    for segment_id, candidates in owners.items():
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: tracks[item]["support"][segment_id], reverse=True)
        best, second = ranked[:2]
        best_score = tracks[best]["support"][segment_id]
        second_score = tracks[second]["support"][segment_id]
        if second_score >= best_score * float(ambiguity_ratio):
            ambiguous.add(segment_id)
            for track_id in candidates:
                cleaned[track_id].discard(segment_id)
        else:
            for track_id in ranked[1:]:
                cleaned[track_id].discard(segment_id)
    return cleaned, ambiguous


def build_refined_instances(
    observations,
    point_superpoints,
    core_coverage=0.50,
    partial_coverage=0.15,
    min_core_iou=0.20,
    min_core_containment=0.50,
    max_candidates_per_superpoint=40,
    min_support_views=2,
    singleton_min_confidence=4.0,
    duplicate_iou=0.70,
    ambiguity_ratio=0.90,
):
    prepared, segment_sizes = prepare_observation_support(
        observations, point_superpoints, core_coverage, partial_coverage
    )
    grouped, relation_count = match_observation_tracks(
        prepared, min_core_iou, min_core_containment, max_candidates_per_superpoint
    )
    tracks = [_aggregate_track(track_ids, prepared) for track_ids in grouped]
    selected = [
        select_track_superpoints(track, prepared, min_support_views, singleton_min_confidence)
        for track in tracks
    ]
    kept, removed = deduplicate_tracks(tracks, selected, duplicate_iou)
    cleaned, ambiguous = clean_competing_superpoints(kept, tracks, selected, ambiguity_ratio)
    instances = []
    for track_id in kept:
        segments = cleaned[track_id]
        if not segments:
            continue
        class_votes = tracks[track_id]["class_votes"]
        class_name, semantic_score = max(class_votes.items(), key=lambda item: (item[1], item[0]))
        instances.append(
            {
                "track_id": int(track_id),
                "observation_ids": tracks[track_id]["observation_ids"],
                "frame_count": int(tracks[track_id]["frame_count"]),
                "score": float(tracks[track_id]["score"]),
                "class_name": class_name,
                "semantic_vote_score": float(semantic_score),
                "superpoint_ids": sorted(int(item) for item in segments),
                "superpoint_count": len(segments),
                "point_count": int(sum(segment_sizes[item] for item in segments)),
            }
        )
    instances.sort(key=lambda item: (-item["score"], -item["point_count"], item["track_id"]))
    diagnostics = {
        "input_observation_count": len(observations),
        "reliable_observation_count": len(prepared),
        "raw_track_count": len(tracks),
        "match_relation_count": int(relation_count),
        "duplicate_track_count": len(removed),
        "ambiguous_superpoint_count": len(ambiguous),
        "output_instance_count": len(instances),
    }
    return instances, diagnostics


def process_scene(scene_name, args):
    source_scene = Path(args.observation_root) / scene_name
    observations_path = source_scene / "observations.jsonl"
    if not observations_path.is_file():
        raise FileNotFoundError(observations_path)
    superpoint_path = _superpoint_file(args.superpoint_root, scene_name)
    if not superpoint_path.is_file():
        raise FileNotFoundError(superpoint_path)
    superpoint_array = np.load(superpoint_path)
    if superpoint_array.ndim != 2 or superpoint_array.shape[1] < 10:
        raise ValueError(f"Expected processed scene with superpoint labels in column 9: {superpoint_path}")
    point_superpoints = superpoint_array[:, 9].astype(np.int64)
    observations = _load_jsonl(observations_path)
    instances, diagnostics = build_refined_instances(
        observations,
        point_superpoints,
        core_coverage=args.core_coverage,
        partial_coverage=args.partial_coverage,
        min_core_iou=args.min_core_iou,
        min_core_containment=args.min_core_containment,
        max_candidates_per_superpoint=args.max_candidates_per_superpoint,
        min_support_views=args.min_support_views,
        singleton_min_confidence=args.singleton_min_confidence,
        duplicate_iou=args.duplicate_iou,
        ambiguity_ratio=args.ambiguity_ratio,
    )
    output_scene = Path(args.output_root) / scene_name
    if output_scene.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_scene}; use --overwrite to replace it")
    point_dir = output_scene / "instance_points"
    point_dir.mkdir(parents=True, exist_ok=True)
    for instance_id, instance in enumerate(instances):
        segment_mask = np.isin(point_superpoints, instance["superpoint_ids"])
        point_indices = np.flatnonzero(segment_mask).astype(np.int64)
        point_path = point_dir / f"instance{instance_id:05d}_points.npz"
        np.savez_compressed(point_path, point_indices=point_indices)
        instance["instance_id"] = instance_id
        instance["point_indices_path"] = str(point_path)
    payload = {
        "scene_name": scene_name,
        "source_observations": str(observations_path),
        "superpoint_path": str(superpoint_path),
        "params": {key: value for key, value in vars(args).items() if key not in {"scene_names", "scene_split", "max_scenes", "overwrite"}},
        "diagnostics": diagnostics,
        "instances": instances,
    }
    output_scene.mkdir(parents=True, exist_ok=True)
    with (output_scene / "refined_instances.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"scene_name": scene_name, **diagnostics}


def _read_scene_names(args):
    if args.scene_names:
        names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.scene_split:
        with open(args.scene_split) as handle:
            names = [line.strip() for line in handle if line.strip()]
    else:
        names = sorted(path.name for path in Path(args.observation_root).iterdir() if path.is_dir())
    return names[: args.max_scenes] if args.max_scenes is not None else names


def main():
    parser = argparse.ArgumentParser(description="Refine dense frame observations with IBSp superpoints.")
    parser.add_argument("--observation_root", required=True)
    parser.add_argument("--superpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)
    parser.add_argument("--core_coverage", default=0.50, type=float)
    parser.add_argument("--partial_coverage", default=0.15, type=float)
    parser.add_argument("--min_core_iou", default=0.20, type=float)
    parser.add_argument("--min_core_containment", default=0.50, type=float)
    parser.add_argument("--max_candidates_per_superpoint", default=40, type=int)
    parser.add_argument("--min_support_views", default=2, type=int)
    parser.add_argument("--singleton_min_confidence", default=4.0, type=float)
    parser.add_argument("--duplicate_iou", default=0.70, type=float)
    parser.add_argument("--ambiguity_ratio", default=0.90, type=float)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    summaries = [process_scene(scene_name, args) for scene_name in tqdm(_read_scene_names(args))]
    with open(osp.join(args.output_root, "refined_instances_summary.json"), "w") as handle:
        json.dump({"params": vars(args), "scenes": summaries}, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

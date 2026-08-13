#!/usr/bin/env python3
"""Build a no-GT relation graph over frozen Details-consensus proposals.

The graph is an observation ledger for the later iterative Details merge stage.  It
does not merge, suppress, clean up, score, relabel, or export predictions.  Every
unordered proposal pair is retained so that overlap, spatial contact, and disjoint
states are conserved explicitly.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETAILS_MERGE_IOU = 0.30
DETAILS_INCLUSION_COVERAGE = 0.99


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


def geometry_metrics(left_size, right_size, intersection):
    """Return exact IoU and directional coverage for two finite sets."""
    left_size, right_size, intersection = map(int, (left_size, right_size, intersection))
    if left_size <= 0 or right_size <= 0:
        raise ValueError("proposal sets must be non-empty")
    if intersection < 0 or intersection > min(left_size, right_size):
        raise ValueError("invalid set intersection")
    union = left_size + right_size - intersection
    return {
        "intersection_count": intersection,
        "iou": float(intersection / union),
        "left_coverage": float(intersection / left_size),
        "right_coverage": float(intersection / right_size),
    }


def containment_direction(left_coverage, right_coverage):
    """Describe paper-threshold inclusion without applying cleanup."""
    left_in_right = float(left_coverage) > DETAILS_INCLUSION_COVERAGE
    right_in_left = float(right_coverage) > DETAILS_INCLUSION_COVERAGE
    if left_in_right and right_in_left:
        return "mutual"
    if left_in_right:
        return "left_in_right"
    if right_in_left:
        return "right_in_left"
    return "none"


def temporal_metrics(left, right):
    left_observations = set(map(int, left.get("observation_ids", [])))
    right_observations = set(map(int, right.get("observation_ids", [])))
    left_frames = set(map(str, left.get("frame_ids", [])))
    right_frames = set(map(str, right.get("frame_ids", [])))
    return {
        "shared_observation_count": len(left_observations & right_observations),
        "shared_frame_count": len(left_frames & right_frames),
        "frame_union_count": len(left_frames | right_frames),
    }


def build_superpoint_contact_map(
    processed,
    adjacency_knn=12,
    adjacency_max_distance=0.05,
    min_contact_points=3,
    min_contact_ratio=0.02,
):
    """Build the frozen raw-superpoint contact ledger used by earlier growth audits."""
    processed = np.asarray(processed)
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError("processed scene must contain xyz and raw superpoint columns")
    if adjacency_knn <= 0 or adjacency_max_distance <= 0:
        raise ValueError("adjacency parameters must be positive")
    if min_contact_points <= 0 or not 0.0 <= min_contact_ratio <= 1.0:
        raise ValueError("contact filters are invalid")

    points = np.asarray(processed[:, :3], dtype=np.float32)
    raw_ids, inverse = np.unique(
        np.asarray(processed[:, 9], dtype=np.int64), return_inverse=True
    )
    sizes = np.bincount(inverse, minlength=len(raw_ids)).astype(np.int64)
    if len(points) < 2:
        return {}

    neighbor_count = min(int(adjacency_knn), len(points) - 1)
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=neighbor_count + 1, workers=-1)
    left_points = np.repeat(np.arange(len(points), dtype=np.int64), neighbor_count)
    right_points = np.asarray(neighbors, dtype=np.int64)[:, 1:].reshape(-1)
    distances = np.asarray(distances, dtype=np.float32)[:, 1:].reshape(-1)
    valid = (left_points != right_points) & (distances <= float(adjacency_max_distance))
    left_segments = inverse[left_points[valid]]
    right_segments = inverse[right_points[valid]]
    cross = left_segments != right_segments
    left_segments, right_segments = left_segments[cross], right_segments[cross]
    if not len(left_segments):
        return {}

    low = np.minimum(left_segments, right_segments)
    high = np.maximum(left_segments, right_segments)
    packed = low * len(raw_ids) + high
    unique_pairs, pair_inverse = np.unique(packed, return_inverse=True)
    counts = np.bincount(pair_inverse, minlength=len(unique_pairs)).astype(np.int64)
    result = {}
    for packed_pair, count in zip(unique_pairs, counts):
        low_index, high_index = divmod(int(packed_pair), len(raw_ids))
        ratio = float(count / max(1, min(sizes[low_index], sizes[high_index])))
        if int(count) < int(min_contact_points) or ratio < float(min_contact_ratio):
            continue
        key = (int(raw_ids[low_index]), int(raw_ids[high_index]))
        result[key] = {
            "boundary_contact_count": int(count),
            "boundary_contact_ratio": ratio,
        }
    return result


def _load_proposals(scene_name, tracks, processed, track_scene_root):
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError("processed scene must contain raw superpoint IDs in column 9")
    raw_superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    point_count = len(raw_superpoints)
    proposals, nodes = [], []
    track_ids = [int(track["track_id"]) for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("track IDs must be unique")

    for track in sorted(tracks, key=lambda row: int(row["track_id"])):
        track_id = int(track["track_id"])
        superpoint_ids = list(map(int, track.get("superpoint_ids", [])))
        if not superpoint_ids or superpoint_ids != sorted(set(superpoint_ids)):
            raise ValueError(f"track {track_id} has invalid superpoint_ids")
        path = Path(track["points_path"])
        if not path.is_absolute():
            path = track_scene_root / path
        with np.load(path) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if not len(points) or points[0] < 0 or points[-1] >= point_count:
            raise ValueError(f"track {track_id} has invalid point indices")
        if int(track.get("point_count", len(points))) != len(points):
            raise ValueError(f"track {track_id} point_count does not match its point file")
        expected = np.flatnonzero(np.isin(raw_superpoints, superpoint_ids))
        if not np.array_equal(points, expected):
            raise ValueError(f"track {track_id} is not the exact union of its raw superpoints")

        observation_ids = sorted(set(map(int, track.get("observation_ids", []))))
        frame_ids = sorted(
            set(map(str, track.get("frame_ids", []))),
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
        node = {
            "scene_name": scene_name,
            "proposal_id": track_id,
            "track_id": track_id,
            "source_track_id": int(track.get("source_track_id", track_id)),
            "observation_ids": observation_ids,
            "frame_ids": frame_ids,
            "superpoint_ids": superpoint_ids,
            "superpoint_count": len(superpoint_ids),
            "point_count": len(points),
            "support_view_count": int(track.get("support_view_count", len(frame_ids))),
            "mean_node_quality": float(track.get("mean_node_quality", 0.0)),
            "points_path": str(track["points_path"]),
            "decision_state": "Frozen D1 proposal; relation graph does not modify it.",
        }
        nodes.append(node)
        proposals.append({"node": node, "points": points, "superpoints": set(superpoint_ids), "track": track})
    return nodes, proposals


def _point_intersections(proposals, scene_point_count):
    rows, columns = [], []
    for row, proposal in enumerate(proposals):
        rows.extend([row] * len(proposal["points"]))
        columns.extend(proposal["points"].tolist())
    matrix = sparse.csr_matrix(
        (np.ones(len(columns), dtype=np.int32), (rows, columns)),
        shape=(len(proposals), scene_point_count),
    )
    product = sparse.triu(matrix @ matrix.T, k=1).tocoo()
    return {
        (int(left), int(right)): int(value)
        for left, right, value in zip(product.row, product.col, product.data)
    }


def _contact_metrics(left_superpoints, right_superpoints, contact_map):
    pairs = set()
    total = 0
    for first in left_superpoints:
        for second in right_superpoints:
            if first == second:
                continue
            key = tuple(sorted((int(first), int(second))))
            if key in contact_map and key not in pairs:
                pairs.add(key)
                total += int(contact_map[key]["boundary_contact_count"])
    return len(pairs), total


def build_scene_graph(scene_name, tracks, processed, track_scene_root, contact_map):
    """Build all nodes and unordered relations without mutating source records."""
    nodes, proposals = _load_proposals(
        scene_name, tracks, np.asarray(processed), Path(track_scene_root)
    )
    intersections = _point_intersections(proposals, len(processed))
    relations = []
    for left_index, right_index in combinations(range(len(proposals)), 2):
        left, right = proposals[left_index], proposals[right_index]
        point_geometry = geometry_metrics(
            len(left["points"]), len(right["points"]),
            intersections.get((left_index, right_index), 0),
        )
        shared_superpoints = len(left["superpoints"] & right["superpoints"])
        superpoint_geometry = geometry_metrics(
            len(left["superpoints"]), len(right["superpoints"]), shared_superpoints
        )
        adjacency_count, boundary_count = _contact_metrics(
            left["superpoints"], right["superpoints"], contact_map
        )
        if point_geometry["intersection_count"]:
            relation_kind = "overlap"
        elif adjacency_count:
            relation_kind = "contact_only"
        else:
            relation_kind = "disjoint"
        direction = containment_direction(
            point_geometry["left_coverage"], point_geometry["right_coverage"]
        )
        relations.append({
            "scene_name": scene_name,
            "left_proposal_id": int(left["node"]["proposal_id"]),
            "right_proposal_id": int(right["node"]["proposal_id"]),
            "point_intersection_count": point_geometry["intersection_count"],
            "point_iou": point_geometry["iou"],
            "left_point_coverage": point_geometry["left_coverage"],
            "right_point_coverage": point_geometry["right_coverage"],
            "superpoint_intersection_count": superpoint_geometry["intersection_count"],
            "superpoint_iou": superpoint_geometry["iou"],
            "left_superpoint_coverage": superpoint_geometry["left_coverage"],
            "right_superpoint_coverage": superpoint_geometry["right_coverage"],
            **temporal_metrics(left["track"], right["track"]),
            "adjacent_superpoint_pair_count": adjacency_count,
            "boundary_contact_count": boundary_count,
            "has_point_overlap": bool(point_geometry["intersection_count"]),
            "has_spatial_contact": bool(adjacency_count),
            "relation_kind": relation_kind,
            "containment_direction": direction,
            "details_merge_eligible_observed": point_geometry["iou"] > DETAILS_MERGE_IOU,
            "details_inclusion_observed": direction != "none",
            "decision_state": "Observed relation only; no merge, suppression, or cleanup applied.",
        })
    validate_scene_graph(nodes, relations)
    return nodes, relations


def validate_scene_graph(nodes, relations):
    proposal_ids = [int(node["proposal_id"]) for node in nodes]
    if proposal_ids != sorted(proposal_ids) or len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("proposal nodes are not uniquely and deterministically ordered")
    expected_count = len(nodes) * (len(nodes) - 1) // 2
    if len(relations) != expected_count:
        raise ValueError("proposal pair count is not conserved")
    expected_pairs = list(combinations(proposal_ids, 2))
    actual_pairs = [
        (int(row["left_proposal_id"]), int(row["right_proposal_id"]))
        for row in relations
    ]
    if actual_pairs != expected_pairs:
        raise ValueError("relation pairs are invalid or non-deterministic")
    valid_kinds = {"overlap", "contact_only", "disjoint"}
    if any(row["relation_kind"] not in valid_kinds for row in relations):
        raise ValueError("unknown relation kind")
    for row in relations:
        expected_kind = (
            "overlap" if row["has_point_overlap"]
            else "contact_only" if row["has_spatial_contact"]
            else "disjoint"
        )
        if row["relation_kind"] != expected_kind:
            raise ValueError("overlap/contact/disjoint states are inconsistent")


def _scene_summary(scene_name, nodes, relations, args):
    kinds = Counter(row["relation_kind"] for row in relations)
    return {
        "scene_name": scene_name,
        "proposal_count": len(nodes),
        "relation_count": len(relations),
        "expected_relation_count": len(nodes) * (len(nodes) - 1) // 2,
        "relation_counts": {kind: int(kinds[kind]) for kind in ("overlap", "contact_only", "disjoint")},
        "details_merge_eligible_observed_count": sum(bool(row["details_merge_eligible_observed"]) for row in relations),
        "details_inclusion_observed_count": sum(bool(row["details_inclusion_observed"]) for row in relations),
        "contact_parameters": {
            "adjacency_knn": args.adjacency_knn,
            "adjacency_max_distance": args.adjacency_max_distance,
            "min_contact_points": args.min_contact_points,
            "min_contact_ratio": args.min_contact_ratio,
        },
        "ground_truth_usage": "none",
        "decision_state": "Relation ledger only; frozen proposals and point files are unchanged.",
    }


def _build_and_publish_scene(scene_name, args):
    track_scene_root = args.track_root / scene_name
    source = json.loads((track_scene_root / "automatic_tracks.json").read_text())
    tracks = source.get("tracks", [])
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    contact_map = build_superpoint_contact_map(
        processed, args.adjacency_knn, args.adjacency_max_distance,
        args.min_contact_points, args.min_contact_ratio,
    )
    nodes, relations = build_scene_graph(
        scene_name, tracks, processed, track_scene_root, contact_map
    )
    summary = _scene_summary(scene_name, nodes, relations, args)

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "proposal_nodes.jsonl", nodes)
        _write_jsonl(staging / "proposal_relations.jsonl", relations)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adjacency-knn", type=int, default=12)
    parser.add_argument("--adjacency-max-distance", type=float, default=0.05)
    parser.add_argument("--min-contact-points", type=int, default=3)
    parser.add_argument("--min-contact-ratio", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "processed_scene_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("contact_parameters") != {
                "adjacency_knn": args.adjacency_knn,
                "adjacency_max_distance": args.adjacency_max_distance,
                "min_contact_points": args.min_contact_points,
                "min_contact_ratio": args.min_contact_ratio,
            }:
                raise SystemExit(f"resume parameter mismatch for {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_and_publish_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"{summary['proposal_count']} nodes, {summary['relation_count']} relations",
                flush=True,
            )
        summaries.append(summary)

    relation_counts = Counter()
    for summary in summaries:
        relation_counts.update(summary["relation_counts"])
    payload = {
        "scene_count": len(summaries),
        "proposal_count": sum(row["proposal_count"] for row in summaries),
        "relation_count": sum(row["relation_count"] for row in summaries),
        "expected_relation_count": sum(row["expected_relation_count"] for row in summaries),
        "relation_counts": {kind: int(relation_counts[kind]) for kind in ("overlap", "contact_only", "disjoint")},
        "details_merge_eligible_observed_count": sum(row["details_merge_eligible_observed_count"] for row in summaries),
        "details_inclusion_observed_count": sum(row["details_inclusion_observed_count"] for row in summaries),
        "paper_observation_thresholds": {
            "details_merge_point_iou_strictly_greater_than": DETAILS_MERGE_IOU,
            "details_inclusion_directional_coverage_strictly_greater_than": DETAILS_INCLUSION_COVERAGE,
        },
        "ground_truth_usage": "none",
        "decision_state": "D2 relation graph only; no proposal action and no evaluation.",
        "params": vars(args),
        "scene_summaries": summaries,
    }
    if payload["relation_count"] != payload["expected_relation_count"]:
        raise ValueError("global proposal pair count is not conserved")
    (args.output_root / "proposal_relation_graph_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "proposal_count": payload["proposal_count"],
        "relation_count": payload["relation_count"],
        "relation_counts": payload["relation_counts"],
        "details_merge_eligible_observed_count": payload["details_merge_eligible_observed_count"],
        "details_inclusion_observed_count": payload["details_inclusion_observed_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

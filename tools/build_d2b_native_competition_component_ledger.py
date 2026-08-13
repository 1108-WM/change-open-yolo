#!/usr/bin/env python3
"""Build the frozen no-GT D2b/native relation-component ledger for C1c.

The input is the strict ``>0.99``-filtered D2b track cache and the unchanged
Mask3D+YOLO-World cache.  It records every positive point-overlap relation
between tracks and native candidates, every positive track--track relation,
and their connected components.  A component is a *diagnostic relation graph*
only: no candidate is suppressed, merged, rescored, or materialized.

Components use the preregistered threshold-free edge rule ``intersection >
0``.  Point-IoU, both directional coverages, strict containment facts, and
the absence of an independently available spatial-contact measurement are
saved for a later GT-only C1c action oracle.  They are not used to make an
inference decision here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_track_native_competition_ledger import (  # noqa: E402
    DETAILS_INCLUSION_COVERAGE,
    _native_cache_contract,
    _read_scenes,
    _resolve,
    _track_points,
    _write_jsonl,
    track_native_relations,
)


CONTRACT = (
    "Frozen no-GT C1c relation ledger: any positive point overlap connects a "
    "diagnostic component. Relations/components do not suppress, merge, "
    "rescore, rank, materialize, or otherwise modify tracks or native masks."
)


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            if left > right:
                left, right = right, left
            self.parent[right] = left


def _load_tracks(root: Path, scene: str) -> dict[int, dict]:
    rows = json.loads((root / scene / "automatic_tracks.json").read_text())["tracks"]
    result = {int(row["track_id"]): row for row in rows}
    if len(rows) != len(result) or list(result) != sorted(result):
        raise ValueError(f"{scene}: D2b track IDs are not unique/sorted")
    for track_id, row in result.items():
        if int(row.get("proposal_id", track_id)) != track_id:
            raise ValueError(f"{scene}: proposal/track ID differs for {track_id}")
    return result


def _track_track_relations(points_by_track: dict[int, np.ndarray]):
    rows = []
    ids = sorted(points_by_track)
    for left_index, left_id in enumerate(ids):
        left = points_by_track[left_id]
        for right_id in ids[left_index + 1:]:
            right = points_by_track[right_id]
            intersection = int(np.intersect1d(left, right, assume_unique=True).size)
            if not intersection:
                continue
            union = len(left) + len(right) - intersection
            left_coverage = float(intersection / len(left))
            right_coverage = float(intersection / len(right))
            rows.append({
                "left_track_id": left_id,
                "right_track_id": right_id,
                "left_point_count": int(len(left)),
                "right_point_count": int(len(right)),
                "intersection_point_count": intersection,
                "union_point_count": int(union),
                "point_iou": float(intersection / max(1, union)),
                "left_inside_right_ratio": left_coverage,
                "right_inside_left_ratio": right_coverage,
                "left_inside_right_strict_099": left_coverage > DETAILS_INCLUSION_COVERAGE,
                "right_inside_left_strict_099": right_coverage > DETAILS_INCLUSION_COVERAGE,
                "mutual_duplicate_strict_099": bool(
                    left_coverage > DETAILS_INCLUSION_COVERAGE
                    and right_coverage > DETAILS_INCLUSION_COVERAGE
                ),
                "spatial_contact_measurement_available": False,
                "spatial_contact_observed": None,
                "ground_truth_usage": "none",
                "decision_state": "observed relation only; no candidate action applied",
            })
    return rows


def _components(track_ids, native_relation_rows, track_relation_rows):
    union_find = _UnionFind()
    for track_id in track_ids:
        union_find.add(("track", int(track_id)))
    for row in native_relation_rows:
        track = ("track", int(row["proposal_id"]))
        native = ("native", int(row["native_candidate_id"]))
        union_find.add(native)
        union_find.union(track, native)
    for row in track_relation_rows:
        union_find.union(
            ("track", int(row["left_track_id"])),
            ("track", int(row["right_track_id"])),
        )
    grouped = defaultdict(list)
    for item in union_find.parent:
        grouped[union_find.find(item)].append(item)
    component_by_node = {}
    component_rows = []
    for component_id, (_, nodes) in enumerate(sorted(grouped.items()), start=0):
        nodes = sorted(nodes)
        tracks = [identifier for kind, identifier in nodes if kind == "track"]
        natives = [identifier for kind, identifier in nodes if kind == "native"]
        for node in nodes:
            component_by_node[node] = component_id
        component_rows.append({
            "component_id": component_id,
            "track_ids": tracks,
            "native_candidate_ids": natives,
            "track_count": len(tracks),
            "native_candidate_count": len(natives),
            "relation_edge_rule": "positive_point_intersection_only",
            "ground_truth_usage": "none",
            "decision_state": "diagnostic component only; no candidate action applied",
        })
    return component_rows, component_by_node


def _build_scene(scene: str, args):
    tracks = _load_tracks(args.filtered_d2b_track_root, scene)
    native_masks = np.load(
        args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r"
    )
    native_scores = np.asarray(
        np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    if native_masks.ndim != 2 or native_masks.shape[1] != len(native_scores):
        raise ValueError(f"{scene}: native masks/scores differ")
    native_sizes = np.count_nonzero(native_masks, axis=0).astype(np.int64)
    points_by_track = {
        track_id: _track_points(track, native_masks.shape[0])
        for track_id, track in tracks.items()
    }
    native_rows = []
    for track_id, track in tracks.items():
        rows, _ = track_native_relations(
            points_by_track[track_id], native_masks, native_sizes, native_scores,
            track_id, float(track["mean_node_quality"]), "filtered_d2b",
        )
        native_rows.extend(rows)
    track_rows = _track_track_relations(points_by_track)
    components, component_by_node = _components(tracks, native_rows, track_rows)
    native_edge_counts = Counter()
    track_edge_counts = Counter()
    for row in native_rows:
        component_id = component_by_node[("track", int(row["proposal_id"]))]
        if component_id != component_by_node[("native", int(row["native_candidate_id"]))]:
            raise ValueError(f"{scene}: track/native component disagreement")
        row["component_id"] = component_id
        native_edge_counts[component_id] += 1
    for row in track_rows:
        component_id = component_by_node[("track", int(row["left_track_id"]))]
        if component_id != component_by_node[("track", int(row["right_track_id"]))]:
            raise ValueError(f"{scene}: track/track component disagreement")
        row["component_id"] = component_id
        track_edge_counts[component_id] += 1
    for row in components:
        component_id = row["component_id"]
        row["track_native_positive_relation_count"] = int(native_edge_counts[component_id])
        row["track_track_positive_relation_count"] = int(track_edge_counts[component_id])
        row["candidate_action_count"] = 0
        row["track_suppression_applied"] = False
        row["native_mutation_applied"] = False
    component_sizes = Counter(
        (row["track_count"], row["native_candidate_count"]) for row in components
    )
    summary = {
        "scene_name": scene,
        "filtered_d2b_track_count": len(tracks),
        "native_candidate_count": int(native_masks.shape[1]),
        "track_native_positive_relation_count": len(native_rows),
        "track_track_positive_relation_count": len(track_rows),
        "component_count": len(components),
        "track_only_component_count": sum(
            row["track_count"] > 0 and row["native_candidate_count"] == 0
            for row in components
        ),
        "mixed_component_count": sum(
            row["track_count"] > 0 and row["native_candidate_count"] > 0
            for row in components
        ),
        "component_size_histogram": {
            f"tracks_{tracks}_native_{natives}": count
            for (tracks, natives), count in sorted(component_sizes.items())
        },
        "candidate_action_count": 0,
        "proposal_materialization_applied": False,
        "ground_truth_usage": "none",
    }
    published = args.output_root / scene
    staging = args.output_root / f".{scene}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"{scene}: output exists")
    staging.mkdir()
    try:
        _write_jsonl(staging / "track_native_relations.jsonl", native_rows)
        _write_jsonl(staging / "track_track_relations.jsonl", track_rows)
        _write_jsonl(staging / "relation_components.jsonl", components)
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
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in (
        "scene_list", "filtered_d2b_track_root", "native_prediction_cache", "output_root"
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list)
    scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    if not scenes:
        raise SystemExit("--max-scenes must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    native_contract = _native_cache_contract(args.native_prediction_cache, len(all_scenes))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ordinal, scene in enumerate(scenes, 1):
        summary = _build_scene(scene, args)
        summaries.append(summary)
        print(f"[C1c relation ledger] {ordinal}/{len(scenes)} {scene}", flush=True)
    keys = (
        "filtered_d2b_track_count", "native_candidate_count",
        "track_native_positive_relation_count", "track_track_positive_relation_count",
        "component_count", "track_only_component_count", "mixed_component_count",
        "candidate_action_count",
    )
    root = {
        "diagnostic_type": "no-GT C1c filtered-D2b/native relation-component ledger",
        "decision_constraint": CONTRACT,
        "native_cache_contract": native_contract,
        "scene_count": len(summaries),
        **{key: sum(int(row[key]) for row in summaries) for key in keys},
        "proposal_materialization_applied": False,
        "ground_truth_usage": "none",
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

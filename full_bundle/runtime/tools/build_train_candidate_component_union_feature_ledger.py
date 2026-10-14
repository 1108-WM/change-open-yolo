#!/usr/bin/env python3
"""Build no-GT component-union geometry features for official100 tracks.

For each relation component, native exact-geometry representatives are merged
into one point-set union.  Every track is described relative to that union,
to its individual native neighbours, and to peer tracks in the component.
Only immutable candidate geometry and existing no-GT relation components are
read; labels and ground truth are never accessed or written.
"""
from __future__ import annotations

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

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    candidate_ledger_path,
    read_jsonl,
    read_scene_list,
)
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402


VERSION = "official100_component_union_track_features_v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _track_points(track: dict, point_count: int) -> np.ndarray:
    path = Path(track["points_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or points[0] < 0 or points[-1] >= point_count:
        raise ValueError(f"invalid track points: {path}")
    if len(points) != int(track.get("point_count", len(points))):
        raise ValueError(f"track point count mismatch: {path}")
    return points


def _intersection_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(len(np.intersect1d(left, right, assume_unique=True)))


def _pair_geometry(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    intersection = _intersection_count(left, right)
    union = len(left) + len(right) - intersection
    return (
        float(intersection / max(1, union)),
        float(intersection / max(1, len(left))),
        float(intersection / max(1, len(right))),
    )


def _aggregate(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        array = np.asarray([0.0], dtype=np.float64)
    ordered = np.sort(array)[::-1]
    return {
        f"{prefix}__max": float(ordered[0]),
        f"{prefix}__second": float(ordered[1] if len(ordered) > 1 else 0.0),
        f"{prefix}__mean": float(array.mean()),
        f"{prefix}__sum": float(array.sum()),
    }


def component_track_features(
    native_points: dict[str, np.ndarray], track_points: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Return pure geometry features for every track in one component."""
    if not native_points or not track_points:
        raise ValueError("relation component must contain native groups and tracks")
    union_native = np.unique(np.concatenate(list(native_points.values())))
    component_native_total = sum(len(points) for points in native_points.values())
    native_union_redundancy = float(
        (component_native_total - len(union_native)) / max(1, component_native_total)
    )
    result = {}
    for track_id, points in track_points.items():
        union_iou, track_inside_union, union_inside_track = _pair_geometry(points, union_native)
        native_pair = [_pair_geometry(points, native) for native in native_points.values()]
        native_ious = [value[0] for value in native_pair]
        track_inside = [value[1] for value in native_pair]
        native_inside = [value[2] for value in native_pair]
        peer_pair = [
            _pair_geometry(points, other)
            for other_id, other in track_points.items() if other_id != track_id
        ]
        peer_ious = [value[0] for value in peer_pair]
        peer_inside = [value[1] for value in peer_pair]
        other_inside = [value[2] for value in peer_pair]
        overlap_native_count = sum(value > 0.0 for value in native_ious)
        overlap_peer_count = sum(value > 0.0 for value in peer_ious)
        best_native = max(native_ious, default=0.0)
        features = {
            "component_native_group_count": float(len(native_points)),
            "component_track_count": float(len(track_points)),
            "component_native_union_point_count": float(len(union_native)),
            "component_native_union_redundancy": native_union_redundancy,
            "track_point_count": float(len(points)),
            "track_to_native_union_point_ratio": float(len(points) / max(1, len(union_native))),
            "track_native_union_point_iou": union_iou,
            "track_inside_native_union_ratio": track_inside_union,
            "native_union_inside_track_ratio": union_inside_track,
            "track_residual_outside_native_union_fraction": float(1.0 - track_inside_union),
            "native_union_residual_outside_track_fraction": float(1.0 - union_inside_track),
            "overlapped_native_group_count": float(overlap_native_count),
            "overlapped_native_group_fraction": float(
                overlap_native_count / max(1, len(native_points))
            ),
            "best_native_iou_share_of_sum": float(
                best_native / max(1e-12, sum(native_ious)) if native_ious else 0.0
            ),
            "positive_overlap_peer_track_count": float(overlap_peer_count),
            "positive_overlap_peer_track_fraction": float(
                overlap_peer_count / max(1, len(track_points) - 1)
            ),
        }
        features.update(_aggregate(native_ious, "native_pair_point_iou"))
        features.update(_aggregate(track_inside, "native_pair_track_inside_ratio"))
        features.update(_aggregate(native_inside, "native_pair_native_inside_ratio"))
        features.update(_aggregate(peer_ious, "peer_track_point_iou"))
        features.update(_aggregate(peer_inside, "peer_track_candidate_inside_ratio"))
        features.update(_aggregate(other_inside, "peer_track_other_inside_ratio"))
        result[int(track_id)] = features
    return result


def _scene(scene: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    candidate_rows = read_jsonl(candidate_ledger_path(args.records_root, scene))
    native_rows = {
        int(row["candidate_id"]): row for row in candidate_rows
        if row["candidate_source"] == NATIVE_SOURCE
    }
    track_rows = {
        int(row["candidate_id"]): row for row in candidate_rows
        if row["candidate_source"] == TRACK_SOURCE
    }
    cache_root = args.records_root / scene / "native_cache"
    masks_path = cache_root / f"{scene}_pred_masks.npy"
    scores_path = cache_root / f"{scene}_pred_scores.npy"
    masks = np.load(masks_path, mmap_mode="r")
    scores = np.asarray(np.load(scores_path, mmap_mode="r"), dtype=np.float64)
    if masks.shape[1] != len(native_rows) or len(scores) != len(native_rows):
        raise ValueError(f"{scene}: native cache differs from candidate ledger")
    track_path = args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    tracks = json.loads(track_path.read_text()).get("tracks", [])
    track_by_id = {int(row["track_id"]): row for row in tracks}
    if set(track_by_id) != set(track_rows):
        raise ValueError(f"{scene}: filtered tracks differ from candidate ledger")
    all_track_points = {
        track_id: _track_points(track, masks.shape[0])
        for track_id, track in track_by_id.items()
    }
    relation_rows = read_jsonl(
        args.relation_feature_ledger_root / scene / "relation_features.jsonl"
    )
    components = read_jsonl(
        args.relation_feature_ledger_root / scene / "relation_components.jsonl"
    )
    relation_by_component = {}
    for row in relation_rows:
        relation_by_component.setdefault(int(row["relation_component_id"]), []).append(row)
    output = []
    for component in sorted(components, key=lambda row: int(row["relation_component_id"])):
        component_id = int(component["relation_component_id"])
        raw = relation_by_component[component_id]
        group_members = {}
        for row in raw:
            group_id = str(row["native_exact_geometry_group_id"])
            members = sorted(map(int, row["native_member_candidate_ids"]))
            previous = group_members.setdefault(group_id, members)
            if previous != members:
                raise ValueError(f"{scene}: inconsistent group members {group_id}")
        native_points = {}
        native_representatives = {}
        for group_id, members in group_members.items():
            representative = min(members, key=lambda candidate_id: (
                -float(scores[candidate_id]), candidate_id,
            ))
            native_representatives[group_id] = representative
            native_points[group_id] = np.flatnonzero(
                np.asarray(masks[:, representative], dtype=bool)
            ).astype(np.int64)
        component_track_ids = sorted(map(int, component["track_ids"]))
        selected_track_points = {
            track_id: all_track_points[track_id] for track_id in component_track_ids
        }
        features = component_track_features(native_points, selected_track_points)
        for track_id in component_track_ids:
            output.append({
                "scene_name": scene,
                "relation_component_id": component_id,
                "track_id": track_id,
                "native_exact_geometry_group_ids": sorted(native_points),
                "native_representative_candidate_ids": [
                    native_representatives[group_id] for group_id in sorted(native_points)
                ],
                "model_features": features[track_id],
                "contracts": {
                    "feature_ground_truth_usage": "none",
                    "candidate_geometry_modified": False,
                    "candidate_score_modified": False,
                    "threshold_scanning": False,
                },
            })
    controlled_tracks = {int(row["track_id"]) for row in output}
    expected_controlled = {
        int(track_id) for component in components for track_id in component["track_ids"]
    }
    if controlled_tracks != expected_controlled or len(output) != len(controlled_tracks):
        raise ValueError(f"{scene}: component track feature coverage mismatch")
    summary = {
        "scene_name": scene,
        "relation_component_count": len(components),
        "relation_count": len(relation_rows),
        "track_feature_row_count": len(output),
        "feature_count": len(output[0]["model_features"]) if output else 0,
        "feature_ground_truth_usage": "none",
        "input_provenance": {
            "candidate_ledger_sha256": _sha256(candidate_ledger_path(args.records_root, scene)),
            "native_masks_sha256": _sha256(masks_path),
            "native_scores_sha256": _sha256(scores_path),
            "filtered_tracks_sha256": _sha256(track_path),
            "relation_features_sha256": _sha256(
                args.relation_feature_ledger_root / scene / "relation_features.jsonl"
            ),
            "relation_components_sha256": _sha256(
                args.relation_feature_ledger_root / scene / "relation_components.jsonl"
            ),
        },
    }
    return output, summary


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != args.expected_scene_count:
        raise ValueError(f"expected {args.expected_scene_count} scenes, got {len(scenes)}")
    relation_summary = json.loads(
        (args.relation_feature_ledger_root / "summary.json").read_text()
    )
    if relation_summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("relation feature ledger violates no-GT contract")
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    all_rows = []
    summaries = []
    try:
        for index, scene in enumerate(scenes, start=1):
            rows, summary = _scene(scene, args)
            scene_root = staging / scene
            scene_root.mkdir()
            _write_jsonl(scene_root / "component_union_track_features.jsonl", rows)
            (scene_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(f"[component union features] {index}/{len(scenes)} {scene}", flush=True)
        feature_names = sorted(all_rows[0]["model_features"]) if all_rows else []
        if any(sorted(row["model_features"]) != feature_names for row in all_rows):
            raise ValueError("component union feature schema differs across tracks")
        _write_jsonl(staging / "component_union_track_features.jsonl", all_rows)
        output = {
            "version": VERSION,
            "scene_count": len(scenes),
            "relation_component_count": sum(row["relation_component_count"] for row in summaries),
            "relation_count": sum(row["relation_count"] for row in summaries),
            "track_feature_row_count": len(all_rows),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "feature_ground_truth_usage": "none",
            "candidate_files_modified": False,
            "ap_evaluation_run": False,
            "threshold_scanning": False,
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "relation_summary_sha256": _sha256(
                    args.relation_feature_ledger_root / "summary.json"
                ),
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "records_root", "relation_feature_ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

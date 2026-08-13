#!/usr/bin/env python3
"""Append no-GT residual-region evidence to component-union track features.

The residual is the part of a track outside the union of the native Mask3D
geometry representatives in its frozen relation component.  Features use only
candidate point sets, SAM automatic observations, and columns 0:10 of the
prepared scene (xyz, rgb, normal, raw superpoint id).  Semantic and instance
label columns are never indexed.
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

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.build_automatic_sam_track_growth_ledger import (  # noqa: E402
    _normalize_normals,
    _normalize_rgb,
    _raw_superpoint_context,
)
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402
from tools.build_train_candidate_component_union_feature_ledger import (  # noqa: E402
    _resolve,
    _track_points,
    _write_jsonl,
)


VERSION = "official100_component_union_residual_track_features_v1"


def _connected_components(
    active_ids: set[int], neighbors: dict[int, list[dict]], weights: dict[int, int],
) -> tuple[int, float]:
    unseen = set(active_ids)
    component_weights = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        total = 0
        while stack:
            current = stack.pop()
            total += int(weights[current])
            adjacent = {
                int(row["neighbor_superpoint_id"])
                for row in neighbors.get(current, [])
            }
            reached = unseen.intersection(adjacent)
            unseen.difference_update(reached)
            stack.extend(reached)
        component_weights.append(total)
    total_weight = sum(component_weights)
    return len(component_weights), float(max(component_weights, default=0) / max(1, total_weight))


def residual_region_features(
    processed_inference: np.ndarray,
    context: dict,
    track_points: np.ndarray,
    native_union_points: np.ndarray,
    observations: list[tuple[np.ndarray, float, float]] | None = None,
) -> dict[str, float]:
    """Describe a track residual without reading semantic or instance labels."""
    processed = np.asarray(processed_inference)
    if processed.ndim != 2 or processed.shape[1] != 10:
        raise ValueError("residual features require exactly inference columns 0:10")
    track = np.unique(np.asarray(track_points, dtype=np.int64))
    native = np.unique(np.asarray(native_union_points, dtype=np.int64))
    residual = np.setdiff1d(track, native, assume_unique=True)
    residual_count = len(residual)
    empty = residual_count == 0
    output = {
        "residual_point_count": float(residual_count),
        "residual_point_fraction_of_track": float(residual_count / max(1, len(track))),
        "residual_empty": float(empty),
    }
    zero_features = (
        "residual_superpoint_count",
        "residual_dominant_superpoint_fraction",
        "residual_top2_superpoint_fraction",
        "residual_mean_superpoint_occupancy",
        "residual_max_superpoint_occupancy",
        "residual_full_superpoint_fraction",
        "residual_shared_native_superpoint_fraction",
        "residual_connected_component_count",
        "residual_largest_connected_component_fraction",
        "residual_boundary_native_superpoint_fraction",
        "residual_boundary_contact_count_per_point",
        "residual_mean_color_difference_to_native",
        "residual_mean_normal_difference_to_native",
        "residual_observation_support_count",
        "residual_observation_support_fraction",
        "residual_point_observation_support_mean",
        "residual_point_observation_support_fraction_mean",
        "residual_point_multiview_fraction",
        "residual_point_single_view_fraction",
        "residual_point_unobserved_fraction",
        "residual_support_predicted_iou_mean",
        "residual_support_stability_score_mean",
    )
    if empty:
        output.update({name: 0.0 for name in zero_features})
        output["residual_observation_missing"] = float(not observations)
        return output

    raw_point_ids = np.asarray(processed[:, 9], dtype=np.int64)
    residual_ids, residual_counts = np.unique(raw_point_ids[residual], return_counts=True)
    native_ids = set(map(int, np.unique(raw_point_ids[native])))
    active_ids = set(map(int, residual_ids))
    residual_by_id = {
        int(superpoint_id): int(count)
        for superpoint_id, count in zip(residual_ids, residual_counts)
    }
    size_by_id = {
        int(superpoint_id): int(size)
        for superpoint_id, size in zip(context["raw_ids"], context["sizes"])
    }
    occupancies = np.asarray([
        residual_by_id[superpoint_id] / max(1, size_by_id[superpoint_id])
        for superpoint_id in sorted(active_ids)
    ], dtype=np.float64)
    ordered_counts = np.sort(residual_counts)[::-1]
    component_count, largest_component_fraction = _connected_components(
        active_ids, context["neighbors"], residual_by_id,
    )
    boundary_ids = set()
    boundary_contact_count = 0
    for superpoint_id in active_ids:
        if superpoint_id in native_ids:
            boundary_ids.add(superpoint_id)
        for adjacency in context["neighbors"].get(superpoint_id, []):
            if int(adjacency["neighbor_superpoint_id"]) in native_ids:
                boundary_ids.add(superpoint_id)
                boundary_contact_count += int(adjacency["boundary_contact_count"])

    colors = _normalize_rgb(processed[:, 3:6])
    normals = _normalize_normals(processed[:, 6:9])
    residual_color = colors[residual].mean(axis=0)
    native_color = colors[native].mean(axis=0)
    residual_normal = _normalize_normals(normals[residual].mean(axis=0, keepdims=True))[0]
    native_normal = _normalize_normals(normals[native].mean(axis=0, keepdims=True))[0]
    output.update({
        "residual_superpoint_count": float(len(residual_ids)),
        "residual_dominant_superpoint_fraction": float(ordered_counts[0] / residual_count),
        "residual_top2_superpoint_fraction": float(ordered_counts[:2].sum() / residual_count),
        "residual_mean_superpoint_occupancy": float(occupancies.mean()),
        "residual_max_superpoint_occupancy": float(occupancies.max()),
        "residual_full_superpoint_fraction": float(np.mean(occupancies >= 0.95)),
        "residual_shared_native_superpoint_fraction": float(
            len(active_ids.intersection(native_ids)) / max(1, len(active_ids))
        ),
        "residual_connected_component_count": float(component_count),
        "residual_largest_connected_component_fraction": largest_component_fraction,
        "residual_boundary_native_superpoint_fraction": float(
            len(boundary_ids) / max(1, len(active_ids))
        ),
        "residual_boundary_contact_count_per_point": float(
            boundary_contact_count / residual_count
        ),
        "residual_mean_color_difference_to_native": float(
            np.linalg.norm(residual_color - native_color) / np.sqrt(3.0)
        ),
        "residual_mean_normal_difference_to_native": float(
            1.0 - abs(float(np.dot(residual_normal, native_normal)))
        ),
    })

    observations = observations or []
    support = np.zeros(residual_count, dtype=np.int64)
    supported_observations = 0
    overlap_weights = []
    predicted_ious = []
    stability_scores = []
    for points, predicted_iou, stability_score in observations:
        _, residual_positions, _ = np.intersect1d(
            residual, np.unique(np.asarray(points, dtype=np.int64)),
            assume_unique=True, return_indices=True,
        )
        overlap = len(residual_positions)
        if not overlap:
            continue
        support[residual_positions] += 1
        supported_observations += 1
        overlap_weights.append(overlap)
        predicted_ious.append(float(predicted_iou))
        stability_scores.append(float(stability_score))
    observation_count = len(observations)
    output.update({
        "residual_observation_missing": float(observation_count == 0),
        "residual_observation_support_count": float(supported_observations),
        "residual_observation_support_fraction": float(
            supported_observations / max(1, observation_count)
        ),
        "residual_point_observation_support_mean": float(support.mean()),
        "residual_point_observation_support_fraction_mean": float(
            support.mean() / max(1, observation_count)
        ),
        "residual_point_multiview_fraction": float(np.mean(support >= 2)),
        "residual_point_single_view_fraction": float(np.mean(support == 1)),
        "residual_point_unobserved_fraction": float(np.mean(support == 0)),
        "residual_support_predicted_iou_mean": float(
            np.average(predicted_ious, weights=overlap_weights) if overlap_weights else 0.0
        ),
        "residual_support_stability_score_mean": float(
            np.average(stability_scores, weights=overlap_weights) if overlap_weights else 0.0
        ),
    })
    if sorted(output) != sorted(("residual_point_count", "residual_point_fraction_of_track", "residual_empty", "residual_observation_missing", *zero_features)):
        raise AssertionError("residual feature schema is incomplete")
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("non-finite residual feature")
    return output


def _load_observations(
    root: Path, track: dict, by_id: dict[int, dict],
) -> list[tuple[np.ndarray, float, float]]:
    output = []
    for observation_id in map(int, track.get("observation_ids", [])):
        row = by_id[observation_id]
        points_path = root / "points" / f"obs{observation_id:06d}_points.npz"
        with np.load(points_path) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        output.append((
            points,
            float(row.get("predicted_iou", 0.0)),
            float(row.get("stability_score", 0.0)),
        ))
    return output


def _scene(scene: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    base_path = args.base_component_union_feature_ledger_root / scene / "component_union_track_features.jsonl"
    base_rows = read_jsonl(base_path)
    if not base_rows:
        return [], {"scene_name": scene, "track_feature_row_count": 0, "feature_count": 0}
    masks_path = args.records_root / scene / "native_cache" / f"{scene}_pred_masks.npy"
    masks = np.load(masks_path, mmap_mode="r")
    track_path = args.records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    tracks = json.loads(track_path.read_text()).get("tracks", [])
    track_by_id = {int(row["track_id"]): row for row in tracks}
    track_points = {
        track_id: _track_points(track, masks.shape[0])
        for track_id, track in track_by_id.items()
    }
    processed_path = args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy"
    processed_raw = np.load(processed_path, mmap_mode="r")
    if processed_raw.ndim != 2 or processed_raw.shape[0] != masks.shape[0] or processed_raw.shape[1] < 10:
        raise ValueError(f"{scene}: invalid prepared inference point array")
    processed_inference = np.asarray(processed_raw[:, :10])
    context = _raw_superpoint_context(
        processed_inference, args.adjacency_knn, args.adjacency_max_distance,
        args.min_contact_points, args.min_contact_ratio,
    )
    observation_root = args.records_root / scene / "sam_automatic_uniform30" / scene
    observation_ledger_path = observation_root / "automatic_observations.jsonl"
    observation_by_id = {
        int(row["observation_id"]): row for row in read_jsonl(observation_ledger_path)
    }
    controlled_track_ids = {int(row["track_id"]) for row in base_rows}
    observations = {
        track_id: _load_observations(
            observation_root, track_by_id[track_id], observation_by_id,
        )
        for track_id in controlled_track_ids
    }
    output = []
    for row in base_rows:
        track_id = int(row["track_id"])
        representatives = list(map(int, row["native_representative_candidate_ids"]))
        native_union = np.flatnonzero(
            np.any(np.asarray(masks[:, representatives], dtype=bool), axis=1)
        ).astype(np.int64)
        residual = residual_region_features(
            processed_inference, context, track_points[track_id], native_union,
            observations[track_id],
        )
        overlap = set(row["model_features"]).intersection(residual)
        if overlap:
            raise ValueError(f"{scene}: residual feature collision: {sorted(overlap)}")
        output.append({
            **row,
            "model_features": {**row["model_features"], **residual},
            "contracts": {
                **row.get("contracts", {}),
                "processed_scene_columns_accessed": "0:10 only",
                "semantic_instance_label_columns_accessed": False,
                "sam_observation_geometry_only": True,
            },
        })
    feature_names = sorted(output[0]["model_features"])
    if any(sorted(row["model_features"]) != feature_names for row in output):
        raise ValueError(f"{scene}: residual feature schema differs across tracks")
    summary = {
        "scene_name": scene,
        "track_feature_row_count": len(output),
        "feature_count": len(feature_names),
        "feature_ground_truth_usage": "none",
        "input_provenance": {
            "base_component_union_features_sha256": _sha256(base_path),
            "native_masks_sha256": _sha256(masks_path),
            "filtered_tracks_sha256": _sha256(track_path),
            "prepared_scene_sha256": _sha256(processed_path),
            "automatic_observations_sha256": _sha256(observation_ledger_path),
        },
    }
    return output, summary


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != args.expected_scene_count:
        raise ValueError(f"expected {args.expected_scene_count} scenes, got {len(scenes)}")
    base_summary = json.loads(
        (args.base_component_union_feature_ledger_root / "summary.json").read_text()
    )
    if base_summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("base component-union ledger violates no-GT contract")
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
            print(f"[component residual features] {index}/{len(scenes)} {scene}", flush=True)
        feature_names = sorted(all_rows[0]["model_features"]) if all_rows else []
        output = {
            "version": VERSION,
            "scene_count": len(scenes),
            "relation_component_count": int(base_summary["relation_component_count"]),
            "relation_count": int(base_summary["relation_count"]),
            "track_feature_row_count": len(all_rows),
            "feature_count": len(feature_names),
            "base_feature_count": int(base_summary["feature_count"]),
            "residual_feature_count": len(feature_names) - int(base_summary["feature_count"]),
            "feature_names": feature_names,
            "feature_ground_truth_usage": "none",
            "processed_scene_columns_accessed": "0:10 only",
            "semantic_instance_label_columns_accessed": False,
            "candidate_files_modified": False,
            "ap_evaluation_run": False,
            "threshold_scanning": False,
            "adjacency_parameters": {
                "knn": args.adjacency_knn,
                "max_distance": args.adjacency_max_distance,
                "min_contact_points": args.min_contact_points,
                "min_contact_ratio": args.min_contact_ratio,
            },
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "base_component_union_summary_sha256": _sha256(
                    args.base_component_union_feature_ledger_root / "summary.json"
                ),
                "base_component_union_version": base_summary["version"],
            },
        }
        _write_jsonl(staging / "component_union_track_features.jsonl", all_rows)
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
    parser.add_argument("--processed-scene-root", type=Path, required=True)
    parser.add_argument("--base-component-union-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, required=True)
    parser.add_argument("--adjacency-knn", type=int, default=12)
    parser.add_argument("--adjacency-max-distance", type=float, default=0.05)
    parser.add_argument("--min-contact-points", type=int, default=3)
    parser.add_argument("--min-contact-ratio", type=float, default=0.02)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "records_root", "processed_scene_root",
        "base_component_union_feature_ledger_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

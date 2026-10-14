"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

from collections import defaultdict

from pathlib import Path

import numpy as np

from tools.fi1_shared.evaluate_candidate_quality_reranking_class_agnostic_ap import FEATURE_GROUP, PROTOCOL_NAME

from tools.fi1_shared.train_candidate_component_action_head_structured_oof import RELATIVE_FEATURES, TARGET_FEATURES

from tools.fi1_shared.train_candidate_quality_head_oof import canonicalize_predictions, feature_matrix

def _track_points(track: dict, point_count: int) -> tuple[np.ndarray, Path]:
    path = Path(track["points_path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen track points: {path}")
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or points[0] < 0 or points[-1] >= point_count:
        raise ValueError(f"invalid track points: {path}")
    if int(track.get("point_count", len(points))) != len(points):
        raise ValueError(f"track point count mismatch: {path}")
    return points, path

def _native_groups(scene: str, masks: np.ndarray, scores: np.ndarray) -> tuple[list[dict], dict[int, str]]:
    by_digest: dict[str, list[int]] = defaultdict(list)
    for candidate_id in range(masks.shape[1]):
        packed = np.packbits(np.asarray(masks[:, candidate_id], dtype=np.uint8)).tobytes()
        by_digest[hashlib.sha256(packed).hexdigest()].append(candidate_id)
    groups = []
    member_to_group = {}
    for index, members in enumerate(sorted(by_digest.values(), key=lambda values: values[0])):
        group_id = f"{scene}:native_geometry:{index:04d}"
        groups.append({
            "native_exact_geometry_group_id": group_id,
            "native_member_candidate_ids": members,
            "native_exact_geometry_group_size": len(members),
            "native_anchor_candidate_id": members[0],
            "native_original_score_median": float(np.median(scores[members])),
        })
        for member in members:
            member_to_group[member] = group_id
    if len(member_to_group) != masks.shape[1]:
        raise AssertionError(f"{scene}: exact geometry grouping lost candidates")
    return groups, member_to_group

def _positive_pairs(
    scene: str, masks: np.ndarray, scores: np.ndarray, tracks: list[dict],
) -> tuple[list[dict], dict[int, np.ndarray], dict[str, np.ndarray], list[Path]]:
    groups, _ = _native_groups(scene, masks, scores)
    group_by_id = {row["native_exact_geometry_group_id"]: row for row in groups}
    anchors = np.asarray([row["native_anchor_candidate_id"] for row in groups], dtype=np.int64)
    native_sizes = np.count_nonzero(masks[:, anchors], axis=0).astype(np.int64)
    native_points = {
        row["native_exact_geometry_group_id"]: np.flatnonzero(
            np.asarray(masks[:, row["native_anchor_candidate_id"]], dtype=bool)
        ).astype(np.int64)
        for row in groups
    }
    track_points = {}
    point_paths = []
    pairs = []
    for track in sorted(tracks, key=lambda row: int(row["track_id"])):
        track_id = int(track["track_id"])
        points, path = _track_points(track, masks.shape[0])
        point_paths.append(path)
        track_points[track_id] = points
        intersections = np.count_nonzero(masks[points][:, anchors], axis=0).astype(np.int64)
        for group_index in np.flatnonzero(intersections > 0):
            group_id = groups[int(group_index)]["native_exact_geometry_group_id"]
            group = group_by_id[group_id]
            intersection = int(intersections[group_index])
            native_size = int(native_sizes[group_index])
            union = len(points) + native_size - intersection
            track_coverage = float(intersection / len(points))
            native_coverage = float(intersection / max(1, native_size))
            pairs.append({
                "scene_name": scene,
                "track_id": track_id,
                "native_exact_geometry_group_id": group_id,
                "native_member_candidate_ids": group["native_member_candidate_ids"],
                "native_exact_geometry_group_size": group["native_exact_geometry_group_size"],
                "track_original_score": float(track["mean_node_quality"]),
                "native_original_score_median": group["native_original_score_median"],
                "point_iou": float(intersection / max(1, union)),
                "track_inside_native_ratio": track_coverage,
                "native_inside_track_ratio": native_coverage,
                "mutual_duplicate_strict_099": bool(
                    track_coverage > 0.99 and native_coverage > 0.99
                ),
            })
    involved_groups = {str(row["native_exact_geometry_group_id"]) for row in pairs}
    involved_tracks = {int(row["track_id"]) for row in pairs}
    return (
        pairs,
        {key: value for key, value in track_points.items() if key in involved_tracks},
        {key: value for key, value in native_points.items() if key in involved_groups},
        point_paths,
    )

def _quality_predictions(package: dict, feature_rows: list[dict]) -> tuple[dict, dict]:
    matrix, names = feature_matrix(feature_rows, FEATURE_GROUP, PROTOCOL_NAME)
    expected = package["metadata"]["feature_contract"]["candidate_quality_feature_names"]
    if names != expected:
        raise ValueError("safety60 candidate quality feature schema differs from frozen model")
    predictions = {}
    for target, model in package["quality_models"].items():
        raw = model.predict_proba(matrix)[:, 1] if hasattr(model, "predict_proba") else model.predict(matrix)
        predictions[target] = canonicalize_predictions(target, raw)
    return predictions, {"feature_names": names, "candidate_count": len(feature_rows)}

def _relation_predictions(package: dict, rows: list[dict]) -> list[dict]:
    target_names = package["metadata"]["feature_contract"]["target_relation_feature_names"]
    relative_names = package["metadata"]["feature_contract"]["relative_relation_feature_names"]
    if target_names != list(TARGET_FEATURES) or relative_names != list(RELATIVE_FEATURES):
        raise ValueError("frozen relation schema differs from code contract")
    target_matrix = np.asarray([
        [float(row["features"][name]) for name in target_names] for row in rows
    ], dtype=np.float64)
    relative_matrix = np.asarray([
        [float(row["features"][name]) for name in relative_names] for row in rows
    ], dtype=np.float64)
    same = package["target_relation_model"].predict_proba(target_matrix)[:, 1]
    better = package["relative_relation_model"].predict_proba(relative_matrix)[:, 1]
    return [{
        "same_target_score": float(same[index]),
        "different_target_score": float(1.0 - same[index]),
        "track_better_score": float(better[index]),
        "baseline_better_score": float(1.0 - better[index]),
        "track_win_relation_score": float(same[index] * better[index]),
        "baseline_win_relation_score": float(same[index] * (1.0 - better[index])),
        "coexist_relation_score": float(1.0 - same[index]),
    } for index in range(len(rows))]

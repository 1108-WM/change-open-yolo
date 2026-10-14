"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from tools.fi1_shared.build_train_candidate_relation_feature_ledger import EXCLUSIVE_PAIR_MODEL_FEATURES

RANDOM_SEED = 20260809

MODEL_FEATURES = {
    "A_point_iou_only": (
        "point_iou",
    ),
    "B_overlap_geometry": (
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance",
        "centroid_distance_normalized",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
    ),
    "C_plus_topology_appearance": (
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance",
        "centroid_distance_normalized",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
        "shared_superpoint_count",
        "track_shared_superpoint_fraction",
        "native_shared_superpoint_fraction",
        "track_superpoint_count",
        "native_superpoint_count",
        "track_superpoint_component_count",
        "native_superpoint_component_count",
        "native_superpoint_full_occupancy_fraction",
        "native_superpoint_mean_occupancy",
        "mean_rgb_distance",
        "mean_normal_absolute_dot",
        "mean_normal_difference",
        "track_normal_coherence",
        "native_normal_coherence",
        "exclusive_boundary_superpoint_edge_count",
        "exclusive_boundary_contact_point_count",
        "exclusive_boundary_contact_ratio_mean",
        "exclusive_boundary_contact_ratio_max",
        "exclusive_boundary_distance_weighted_mean",
        "exclusive_boundary_normal_difference_weighted_mean",
        "exclusive_boundary_color_difference_weighted_mean",
    ),
    "D_plus_public_component": (
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance",
        "centroid_distance_normalized",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
        "shared_superpoint_count",
        "track_shared_superpoint_fraction",
        "native_shared_superpoint_fraction",
        "track_superpoint_count",
        "native_superpoint_count",
        "track_superpoint_component_count",
        "native_superpoint_component_count",
        "native_superpoint_full_occupancy_fraction",
        "native_superpoint_mean_occupancy",
        "mean_rgb_distance",
        "mean_normal_absolute_dot",
        "mean_normal_difference",
        "track_normal_coherence",
        "native_normal_coherence",
        "exclusive_boundary_superpoint_edge_count",
        "exclusive_boundary_contact_point_count",
        "exclusive_boundary_contact_ratio_mean",
        "exclusive_boundary_contact_ratio_max",
        "exclusive_boundary_distance_weighted_mean",
        "exclusive_boundary_normal_difference_weighted_mean",
        "exclusive_boundary_color_difference_weighted_mean",
        "public_common_eligible_view_count",
        "public_common_selected_view_count",
        "public_common_view_available",
        "public_projected_box_iou_mean",
        "public_projected_box_iou_min",
        "public_projected_box_iou_max",
        "public_same_matched_observation_count",
        "public_same_matched_observation_fraction",
        "public_different_matched_observation_count",
        "public_different_matched_observation_fraction",
        "public_both_matched_observation_count",
        "public_track_gvc_mean",
        "public_native_gvc_mean",
        "public_track_minus_native_gvc",
        "public_track_visible_fraction_mean",
        "public_native_visible_fraction_mean",
        "public_centroid_camera_range_absolute_delta_mean",
        "public_centroid_camera_range_absolute_delta_max",
        "public_centroid_camera_range_order_consistency",
        "relation_component_track_count",
        "relation_component_native_group_count",
        "relation_component_native_member_candidate_count",
        "relation_component_relation_count",
        "track_overlapping_native_group_count",
        "track_overlapping_native_member_candidate_count",
        "native_group_overlapping_track_count",
        "hypothetical_pair_native_member_deletion_count",
        "hypothetical_track_wide_native_member_deletion_count",
        "track_original_score",
        "native_original_score_median",
        "original_score_delta_track_minus_native",
    ),
}

MODEL_FEATURES["E_plus_exclusive_pair"] = (
    MODEL_FEATURES["D_plus_public_component"] + EXCLUSIVE_PAIR_MODEL_FEATURES
)

def scene_track_balanced_weights(rows: list[dict], indexes: np.ndarray | None = None) -> np.ndarray:
    """Give every scene equal mass and every track equal mass within a scene."""
    if indexes is None:
        indexes = np.arange(len(rows), dtype=np.int64)
    indexes = np.asarray(indexes, dtype=np.int64)
    if not len(indexes):
        raise ValueError("cannot weight an empty relation population")
    scene_tracks: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indexes:
        row = rows[int(index)]
        scene_tracks[str(row["scene_name"])][int(row["track_id"])].append(int(index))
    weights = np.zeros(len(rows), dtype=np.float64)
    scene_mass = len(indexes) / len(scene_tracks)
    for tracks in scene_tracks.values():
        track_mass = scene_mass / len(tracks)
        for relation_indexes in tracks.values():
            relation_weight = track_mass / len(relation_indexes)
            weights[relation_indexes] = relation_weight
    if not np.isclose(weights[indexes].sum(), len(indexes), atol=1e-8):
        raise AssertionError("scene-track weights do not preserve total mass")
    return weights

def class_balanced_weights(labels: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("class-balanced fitting requires both classes")
    total = float(weights.sum())
    for label in (0, 1):
        mask = labels == label
        mass = float(weights[mask].sum())
        if mass <= 0:
            raise ValueError(f"class {label} has no positive weight")
        weights[mask] *= 0.5 * total / mass
    return weights

def seeded_scene_folds(scene_names: list[str], fold_count: int, seed: int) -> list[dict]:
    unique = sorted(set(scene_names))
    if len(unique) < fold_count or len(unique) % fold_count:
        raise ValueError("scene count must be divisible by inner fold count")
    shuffled = np.random.default_rng(seed).permutation(unique).tolist()
    size = len(unique) // fold_count
    folds = []
    for fold_index in range(fold_count):
        validation = set(shuffled[fold_index * size:(fold_index + 1) * size])
        folds.append({
            "fold_index": fold_index,
            "train_scenes": sorted(set(unique) - validation),
            "validation_scenes": sorted(validation),
        })
    return folds

def _make_base_model(seed: int) -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )),
    ])

def _fit_base(
    matrix: np.ndarray,
    labels: np.ndarray,
    base_weights: np.ndarray,
    seed: int,
) -> Pipeline:
    model = _make_base_model(seed)
    fit_weights = class_balanced_weights(labels, base_weights)
    model.fit(matrix, labels, classifier__sample_weight=fit_weights)
    return model

def _fit_platt(decisions: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    if not np.isfinite(decisions).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Platt calibration requires finite decisions and both classes")
    calibrator = LogisticRegression(
        penalty="l2", C=1e6, solver="lbfgs", max_iter=5000,
        random_state=RANDOM_SEED,
    )
    calibrator.fit(decisions.reshape(-1, 1), labels, sample_weight=weights)
    return calibrator

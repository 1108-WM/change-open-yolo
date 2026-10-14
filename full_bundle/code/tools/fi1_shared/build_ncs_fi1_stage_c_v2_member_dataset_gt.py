"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

from pathlib import Path

import imageio.v2 as imageio

import numpy as np

from tools.build_details_same_frame_hierarchy_preprocessor import decode_binary_mask_rle

from tools.fi1_shared.export_mv3dis_relative_depth_observations import mask_points_from_projection, project_relative_depth_frame

MEMBER_EVIDENCE_FEATURE_NAMES = (
    "member_log1p_track_observation_count",
    "member_source_supported_observation_fraction",
    "member_source_coverage_mean",
    "member_source_coverage_min",
    "member_source_coverage_max",
    "member_source_predicted_iou_weighted_coverage",
    "member_source_stability_weighted_coverage",
    "member_source_point_support_mean",
    "member_source_point_multiview_fraction",
    "member_source_point_unobserved_fraction",
    "member_relative_visible_observation_fraction",
    "member_relative_mask_coverage_mean",
    "member_relative_mask_coverage_min",
    "member_relative_mask_coverage_max",
    "member_relative_depth_weighted_coverage_mean",
    "member_relative_depth_weighted_coverage_min",
    "member_relative_depth_weighted_coverage_max",
    "member_relative_inside_depth_weight_mean",
    "member_relative_point_visible_view_mean",
    "member_relative_point_inside_view_mean",
    "member_relative_point_inside_given_visible_mean",
    "member_relative_point_multiview_inside_fraction",
    "member_relative_point_visible_but_outside_fraction",
)

def _summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.min()), float(array.max())

def member_evidence_features(
    atom_local_indexes: np.ndarray,
    source_support: list[np.ndarray],
    relative_visible: list[np.ndarray],
    relative_inside: list[np.ndarray],
    relative_weights: list[np.ndarray],
    predicted_ious: list[float],
    stability_scores: list[float],
) -> dict[str, float]:
    """Aggregate genuinely atom-local evidence across the parent track views."""
    indexes = np.asarray(atom_local_indexes, dtype=np.int64)
    observation_count = len(source_support)
    if not (
        len(relative_visible) == len(relative_inside) == len(relative_weights)
        == len(predicted_ious) == len(stability_scores) == observation_count
    ):
        raise ValueError("member evidence observation arrays differ in length")
    point_count = len(indexes)
    source_coverages = []
    visible_coverages = []
    mask_coverages = []
    weighted_coverages = []
    inside_weight_means = []
    point_source_support = np.zeros(point_count, dtype=np.int32)
    point_visible_support = np.zeros(point_count, dtype=np.int32)
    point_inside_support = np.zeros(point_count, dtype=np.int32)
    for source, visible, inside, weights in zip(
        source_support, relative_visible, relative_inside, relative_weights
    ):
        local_source = np.asarray(source, dtype=bool)[indexes]
        local_visible = np.asarray(visible, dtype=bool)[indexes]
        local_inside = np.asarray(inside, dtype=bool)[indexes]
        local_weights = np.asarray(weights, dtype=np.float64)[indexes]
        if np.any(local_inside & ~local_visible):
            raise ValueError("relative mask support occurs outside relative visibility")
        source_count = int(local_source.sum())
        visible_count = int(local_visible.sum())
        inside_count = int(local_inside.sum())
        source_coverages.append(source_count / max(1, point_count))
        point_source_support += local_source
        point_visible_support += local_visible
        point_inside_support += local_inside
        if visible_count:
            visible_coverages.append(visible_count / max(1, point_count))
            mask_coverages.append(inside_count / visible_count)
            weighted_coverages.append(float(local_weights[local_inside].sum() / visible_count))
            inside_weight_means.append(float(
                local_weights[local_inside].mean() if inside_count else 0.0
            ))
    source_mean, source_min, source_max = _summary(source_coverages)
    mask_mean, mask_min, mask_max = _summary(mask_coverages)
    weighted_mean, weighted_min, weighted_max = _summary(weighted_coverages)
    inside_weight_mean = float(np.mean(inside_weight_means)) if inside_weight_means else 0.0
    predicted_weight = float(sum(predicted_ious))
    stability_weight = float(sum(stability_scores))
    conditional = np.divide(
        point_inside_support,
        np.maximum(point_visible_support, 1),
        dtype=np.float64,
    )
    output = {
        "member_log1p_track_observation_count": math.log1p(observation_count),
        "member_source_supported_observation_fraction": float(
            np.mean(np.asarray(source_coverages) > 0.0) if observation_count else 0.0
        ),
        "member_source_coverage_mean": source_mean,
        "member_source_coverage_min": source_min,
        "member_source_coverage_max": source_max,
        "member_source_predicted_iou_weighted_coverage": float(
            np.dot(source_coverages, predicted_ious) / max(predicted_weight, 1e-12)
            if observation_count else 0.0
        ),
        "member_source_stability_weighted_coverage": float(
            np.dot(source_coverages, stability_scores) / max(stability_weight, 1e-12)
            if observation_count else 0.0
        ),
        "member_source_point_support_mean": float(
            point_source_support.mean() if point_count else 0.0
        ),
        "member_source_point_multiview_fraction": float(
            np.mean(point_source_support >= 2) if point_count else 0.0
        ),
        "member_source_point_unobserved_fraction": float(
            np.mean(point_source_support == 0) if point_count else 0.0
        ),
        "member_relative_visible_observation_fraction": float(
            len(visible_coverages) / max(1, observation_count)
        ),
        "member_relative_mask_coverage_mean": mask_mean,
        "member_relative_mask_coverage_min": mask_min,
        "member_relative_mask_coverage_max": mask_max,
        "member_relative_depth_weighted_coverage_mean": weighted_mean,
        "member_relative_depth_weighted_coverage_min": weighted_min,
        "member_relative_depth_weighted_coverage_max": weighted_max,
        "member_relative_inside_depth_weight_mean": inside_weight_mean,
        "member_relative_point_visible_view_mean": float(
            point_visible_support.mean() if point_count else 0.0
        ),
        "member_relative_point_inside_view_mean": float(
            point_inside_support.mean() if point_count else 0.0
        ),
        "member_relative_point_inside_given_visible_mean": float(
            conditional.mean() if point_count else 0.0
        ),
        "member_relative_point_multiview_inside_fraction": float(
            np.mean(point_inside_support >= 2) if point_count else 0.0
        ),
        "member_relative_point_visible_but_outside_fraction": float(
            np.mean((point_visible_support > 0) & (point_inside_support == 0))
            if point_count else 0.0
        ),
    }
    if tuple(output) != MEMBER_EVIDENCE_FEATURE_NAMES:
        raise AssertionError("member evidence feature order differs")
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("non-finite member evidence feature")
    return output

def _union_observation_arrays(
    *, union_points: np.ndarray, observations: list[dict], world: object,
    intrinsic: np.ndarray, automatic_scene_root: Path, depth_cache: dict[int, np.ndarray],
    depth_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    union = np.asarray(union_points, dtype=np.int64)
    points_h = np.column_stack((
        np.asarray(world.points_xyz[union], dtype=np.float64),
        np.ones(len(union), dtype=np.float64),
    ))
    source_arrays = []
    visible_arrays = []
    inside_arrays = []
    weight_arrays = []
    for observation in observations:
        frame_index = int(observation["frame_index"])
        depth_map = depth_cache.get(frame_index)
        if depth_map is None:
            depth_map = np.asarray(
                imageio.imread(world.depth_maps_paths[frame_index]), dtype=np.float64
            ) / depth_scale
            depth_cache[frame_index] = depth_map
        pixels, visible, weights = project_relative_depth_frame(
            points_h, np.loadtxt(world.poses[frame_index]), intrinsic, depth_map,
        )
        mask = decode_binary_mask_rle(observation["mask_rle"])
        inside_local, inside_weights = mask_points_from_projection(
            mask, pixels, visible, weights, world.depth_resolution,
        )
        inside = np.zeros(len(union), dtype=bool)
        inside[inside_local] = True
        local_weights = np.zeros(len(union), dtype=np.float32)
        local_weights[inside_local] = inside_weights
        points_path = Path(observation["point_indices_path"])
        if not points_path.is_file():
            points_path = automatic_scene_root / "points" / f"obs{int(observation['observation_id']):06d}_points.npz"
        with np.load(points_path) as payload:
            source_points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        positions = np.searchsorted(union, source_points)
        valid = (positions < len(union)) & (union[np.minimum(positions, len(union) - 1)] == source_points)
        source = np.zeros(len(union), dtype=bool)
        source[positions[valid]] = True
        source_arrays.append(source)
        visible_arrays.append(np.asarray(visible, dtype=bool))
        inside_arrays.append(inside)
        weight_arrays.append(local_weights)
    return source_arrays, visible_arrays, inside_arrays, weight_arrays

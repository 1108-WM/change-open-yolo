import argparse
import gc
import json
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = osp.dirname(osp.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from run_evaluation import load_yaml
from export_sam_fused_proposals import (
    _clamp_box,
    _connected_component_summary,
    _erode_binary_mask,
    _existing_mask_metrics,
    _filter_seed_indices_by_adaptive_internal_seed,
    _filter_seed_indices_by_depth_cluster,
    _label_consensus_metrics,
    _load_geometry_discriminator,
    _load_sam_predictor,
    _load_scene_points_xyz,
    _merge_label_consensus,
    _parse_class_names,
    _prepare_image,
    _predict_geometry_discriminator_score,
    _resolve_scene_names,
    _safe_label,
    _sam_mask_discriminator_row,
    _sam_mask_to_indices,
    _sam_seed_geometry_quality,
    _sam_view_quality_score,
    _save_mask,
    _save_overlay,
    _select_2d_nms_indices,
    _to_numpy,
)
from utils import OpenYolo3D
from utils.superpoint_diagnostics import (
    _FrameProjector,
    classify_observation_superpoint_support,
    load_or_build_scene_superpoint_cache,
    save_observation_superpoint_evidence,
    summarize_candidate_superpoints,
)


def _seed_overlap(left, right):
    if len(left) == 0 or len(right) == 0:
        return 0, 0, 0.0, 0.0
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    intersection = int(np.intersect1d(left, right, assume_unique=False).size)
    union = int(len(left) + len(right) - intersection)
    iou = float(intersection / max(1, union))
    containment = float(intersection / max(1, min(len(left), len(right))))
    return intersection, union, iou, containment


def _seed_centroid(points_xyz, seed_indices):
    if points_xyz is None or len(seed_indices) == 0:
        return None
    return points_xyz[np.asarray(seed_indices, dtype=np.int64)].mean(axis=0)


def _safe_mean(values, default=0.0):
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return float(default)
    return float(np.mean(values))


def _graph_quality_score(observation):
    return float(
        0.35 * observation.get("view_quality_score", 0.0)
        + 0.25 * observation.get("sam_score", 0.0)
        + 0.20 * observation.get("score", 0.0)
        + 0.10 * observation.get("seed_depth_support_ratio", 0.0)
        + 0.10 * observation.get("sam_mask_geometry", {}).get("quality_score", 0.0)
    )


def _cluster_reference_key(observation, min_reference_coverage):
    ref_id = observation.get("best_existing_mask_id")
    ref_coverage = float(observation.get("best_existing_seed_coverage", 0.0) or 0.0)
    if ref_id is None or ref_coverage < float(min_reference_coverage):
        return None
    return int(ref_id)


def _visible_seed_stats(seed_indices, visible_mask):
    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    if len(seed_indices) == 0:
        return np.asarray([], dtype=np.int64), 0, 0.0
    visible_seed_indices = seed_indices[visible_mask]
    visible_count = int(len(visible_seed_indices))
    visible_ratio = float(visible_count / max(1, len(seed_indices)))
    return visible_seed_indices.astype(np.int64), visible_count, visible_ratio


def _projection_coords_for_mask(coords, scaling_params=None):
    coords = np.asarray(coords, dtype=np.float32)
    if scaling_params is None:
        return np.round(coords[:, 0]).astype(np.int64), np.round(coords[:, 1]).astype(np.int64)
    return (
        np.round(coords[:, 0] / float(scaling_params[1])).astype(np.int64),
        np.round(coords[:, 1] / float(scaling_params[0])).astype(np.int64),
    )


def _points_inside_observation_mask(point_indices, target_observation, projections_np, visible_np, scaling_params=None):
    point_indices = np.asarray(point_indices, dtype=np.int64)
    if len(point_indices) == 0 or projections_np is None or visible_np is None:
        return np.asarray([], dtype=np.int64), {
            "input_points": int(len(point_indices)),
            "visible_points": 0,
            "inside_points": 0,
            "visible_ratio": 0.0,
            "inside_visible_ratio": 0.0,
        }
    target_frame = int(target_observation["frame_index"])
    if target_frame < 0 or target_frame >= visible_np.shape[0]:
        return np.asarray([], dtype=np.int64), {
            "input_points": int(len(point_indices)),
            "visible_points": 0,
            "inside_points": 0,
            "visible_ratio": 0.0,
            "inside_visible_ratio": 0.0,
        }
    valid_point_mask = (point_indices >= 0) & (point_indices < visible_np.shape[1])
    point_indices = point_indices[valid_point_mask]
    if len(point_indices) == 0:
        return np.asarray([], dtype=np.int64), {
            "input_points": 0,
            "visible_points": 0,
            "inside_points": 0,
            "visible_ratio": 0.0,
            "inside_visible_ratio": 0.0,
        }
    depth_visible = visible_np[target_frame, point_indices].astype(bool)
    visible_indices = point_indices[depth_visible]
    if len(visible_indices) == 0:
        return np.asarray([], dtype=np.int64), {
            "input_points": int(len(point_indices)),
            "visible_points": 0,
            "inside_points": 0,
            "visible_ratio": 0.0,
            "inside_visible_ratio": 0.0,
        }
    sam_mask = target_observation.get("_sam_mask")
    if sam_mask is None:
        return visible_indices.astype(np.int64), {
            "input_points": int(len(point_indices)),
            "visible_points": int(len(visible_indices)),
            "inside_points": int(len(visible_indices)),
            "visible_ratio": float(len(visible_indices) / max(1, len(point_indices))),
            "inside_visible_ratio": 1.0,
        }
    sam_mask = np.asarray(sam_mask, dtype=bool)
    coords = projections_np[target_frame, visible_indices].astype(np.float32)
    xs, ys = _projection_coords_for_mask(coords, scaling_params=scaling_params)
    valid_pixels = (xs >= 0) & (xs < sam_mask.shape[1]) & (ys >= 0) & (ys < sam_mask.shape[0])
    if not valid_pixels.any():
        return np.asarray([], dtype=np.int64), {
            "input_points": int(len(point_indices)),
            "visible_points": int(len(visible_indices)),
            "inside_points": 0,
            "visible_ratio": float(len(visible_indices) / max(1, len(point_indices))),
            "inside_visible_ratio": 0.0,
        }
    visible_indices = visible_indices[valid_pixels]
    xs = xs[valid_pixels]
    ys = ys[valid_pixels]
    inside = sam_mask[ys, xs]
    inside_indices = visible_indices[inside]
    return inside_indices.astype(np.int64), {
        "input_points": int(len(point_indices)),
        "visible_points": int(len(visible_indices)),
        "inside_points": int(len(inside_indices)),
        "visible_ratio": float(len(visible_indices) / max(1, len(point_indices))),
        "inside_visible_ratio": float(len(inside_indices) / max(1, len(visible_indices))),
    }


def _pair_common_visible_seed_sets(left, right, projections_np=None, visible_np=None, scaling_params=None):
    left_seed = np.asarray(left["_seed_indices"], dtype=np.int64)
    right_seed = np.asarray(right["_seed_indices"], dtype=np.int64)
    if projections_np is None or visible_np is None:
        return (
            np.asarray(left.get("_visible_seed_indices", left_seed), dtype=np.int64),
            np.asarray(right.get("_visible_seed_indices", right_seed), dtype=np.int64),
            {
                "enabled": False,
                "fallback": "missing_projection_visibility",
                "left_to_right_visible_ratio": float(left.get("seed_depth_support_ratio", 0.0) or 0.0),
                "right_to_left_visible_ratio": float(right.get("seed_depth_support_ratio", 0.0) or 0.0),
                "left_to_right_inside_visible_ratio": 0.0,
                "right_to_left_inside_visible_ratio": 0.0,
                "bidirectional_depth_consistency": min(
                    float(left.get("seed_depth_support_ratio", 0.0) or 0.0),
                    float(right.get("seed_depth_support_ratio", 0.0) or 0.0),
                ),
                "bidirectional_mask_consistency": 0.0,
            },
        )
    left_in_right, left_stats = _points_inside_observation_mask(left_seed, right, projections_np, visible_np, scaling_params=scaling_params)
    right_in_left, right_stats = _points_inside_observation_mask(right_seed, left, projections_np, visible_np, scaling_params=scaling_params)
    depth_consistency = min(float(left_stats["visible_ratio"]), float(right_stats["visible_ratio"]))
    mask_consistency = min(float(left_stats["inside_visible_ratio"]), float(right_stats["inside_visible_ratio"]))
    return left_in_right, right_in_left, {
        "enabled": True,
        "left_to_right_input_points": int(left_stats["input_points"]),
        "left_to_right_visible_points": int(left_stats["visible_points"]),
        "left_to_right_inside_points": int(left_stats["inside_points"]),
        "left_to_right_visible_ratio": float(left_stats["visible_ratio"]),
        "left_to_right_inside_visible_ratio": float(left_stats["inside_visible_ratio"]),
        "right_to_left_input_points": int(right_stats["input_points"]),
        "right_to_left_visible_points": int(right_stats["visible_points"]),
        "right_to_left_inside_points": int(right_stats["inside_points"]),
        "right_to_left_visible_ratio": float(right_stats["visible_ratio"]),
        "right_to_left_inside_visible_ratio": float(right_stats["inside_visible_ratio"]),
        "bidirectional_depth_consistency": float(depth_consistency),
        "bidirectional_mask_consistency": float(mask_consistency),
    }


def _empty_depth_direction_stats(input_points=0):
    return {
        "input_points": int(input_points),
        "valid_depth_points": 0,
        "depth_consistent_points": 0,
        "depth_consistent_ratio": 0.0,
        "inside_mask_points": 0,
        "inside_depth_consistent_points": 0,
        "inside_depth_consistent_ratio": 0.0,
        "depth_conflict_points": 0,
        "depth_conflict_ratio": 0.0,
        "inside_depth_conflict_points": 0,
        "inside_depth_conflict_ratio": 0.0,
        "depth_error_median": 0.0,
        "depth_error_p90": 0.0,
    }


class _DepthRelationCache:
    def __init__(self, openyolo3d, points_xyz, projections_np, scaling_params=None):
        self.world2cam = getattr(openyolo3d, "world2cam", None)
        self.points_xyz = None if points_xyz is None else np.asarray(points_xyz, dtype=np.float32)
        self.projections_np = projections_np
        self.scaling_params = scaling_params
        self.depth_cache = {}
        self.pose_cache = {}
        self.result_cache = {}

    def _load_depth(self, frame_idx):
        frame_idx = int(frame_idx)
        if frame_idx not in self.depth_cache:
            if self.world2cam is None or frame_idx < 0 or frame_idx >= len(self.world2cam.depth_maps_paths):
                return None
            depth = imageio.imread(self.world2cam.depth_maps_paths[frame_idx]).astype(np.float32)
            depth = depth / float(self.world2cam.depth_scale)
            self.depth_cache[frame_idx] = depth
        return self.depth_cache[frame_idx]

    def _load_extrinsic(self, frame_idx):
        frame_idx = int(frame_idx)
        if frame_idx not in self.pose_cache:
            if self.world2cam is None or frame_idx < 0 or frame_idx >= len(self.world2cam.poses):
                return None
            self.pose_cache[frame_idx] = np.linalg.inv(np.loadtxt(self.world2cam.poses[frame_idx]).astype(np.float64))
        return self.pose_cache[frame_idx]

    def direction_stats(self, source_observation, target_observation):
        source_id = int(source_observation.get("graph_observation_id", -1))
        target_id = int(target_observation.get("graph_observation_id", -1))
        cache_key = (source_id, target_id)
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]

        source_indices = np.asarray(source_observation.get("_seed_indices", []), dtype=np.int64)
        output = _empty_depth_direction_stats(len(source_indices))
        if (
            self.points_xyz is None
            or self.projections_np is None
            or len(source_indices) == 0
            or int(source_indices.max()) >= self.points_xyz.shape[0]
        ):
            self.result_cache[cache_key] = output
            return output

        target_frame = int(target_observation.get("frame_index", -1))
        depth = self._load_depth(target_frame)
        extrinsic = self._load_extrinsic(target_frame)
        if depth is None or extrinsic is None or target_frame < 0 or target_frame >= self.projections_np.shape[0]:
            self.result_cache[cache_key] = output
            return output

        coords = self.projections_np[target_frame, source_indices].astype(np.float32)
        xs_depth = np.round(coords[:, 0]).astype(np.int64)
        ys_depth = np.round(coords[:, 1]).astype(np.int64)
        valid_pixel = (xs_depth >= 0) & (xs_depth < depth.shape[1]) & (ys_depth >= 0) & (ys_depth < depth.shape[0])
        if not valid_pixel.any():
            self.result_cache[cache_key] = output
            return output

        valid_indices = source_indices[valid_pixel]
        xs_depth = xs_depth[valid_pixel]
        ys_depth = ys_depth[valid_pixel]
        measured_depth = depth[ys_depth, xs_depth].astype(np.float32)
        valid_depth = measured_depth > 0.0
        if not valid_depth.any():
            self.result_cache[cache_key] = output
            return output

        valid_indices = valid_indices[valid_depth]
        measured_depth = measured_depth[valid_depth]
        coords_valid = coords[valid_pixel][valid_depth]
        points = self.points_xyz[valid_indices]
        hom_points = np.concatenate([points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1)
        projected_depth = (hom_points @ extrinsic.T)[:, 2].astype(np.float32)
        positive_projected_depth = projected_depth > 0.0
        if not positive_projected_depth.any():
            self.result_cache[cache_key] = output
            return output

        measured_depth = measured_depth[positive_projected_depth]
        projected_depth = projected_depth[positive_projected_depth]
        coords_valid = coords_valid[positive_projected_depth]
        depth_error = np.abs(projected_depth - measured_depth)
        tolerance = np.minimum(0.10, np.maximum(0.04, 0.03 * measured_depth))
        depth_consistent = depth_error <= tolerance
        depth_conflict = ~depth_consistent

        sam_mask = target_observation.get("_sam_mask")
        inside_mask = np.zeros((len(depth_error),), dtype=bool)
        if sam_mask is not None:
            sam_mask = np.asarray(sam_mask, dtype=bool)
            xs_mask, ys_mask = _projection_coords_for_mask(coords_valid, scaling_params=self.scaling_params)
            valid_mask_pixel = (xs_mask >= 0) & (xs_mask < sam_mask.shape[1]) & (ys_mask >= 0) & (ys_mask < sam_mask.shape[0])
            if valid_mask_pixel.any():
                inside_mask[valid_mask_pixel] = sam_mask[ys_mask[valid_mask_pixel], xs_mask[valid_mask_pixel]]

        inside_count = int(inside_mask.sum())
        output = {
            "input_points": int(len(source_indices)),
            "valid_depth_points": int(len(depth_error)),
            "depth_consistent_points": int(depth_consistent.sum()),
            "depth_consistent_ratio": float(depth_consistent.sum() / max(1, len(depth_error))),
            "inside_mask_points": inside_count,
            "inside_depth_consistent_points": int(np.logical_and(inside_mask, depth_consistent).sum()),
            "inside_depth_consistent_ratio": float(np.logical_and(inside_mask, depth_consistent).sum() / max(1, inside_count)),
            "depth_conflict_points": int(depth_conflict.sum()),
            "depth_conflict_ratio": float(depth_conflict.sum() / max(1, len(depth_error))),
            "inside_depth_conflict_points": int(np.logical_and(inside_mask, depth_conflict).sum()),
            "inside_depth_conflict_ratio": float(np.logical_and(inside_mask, depth_conflict).sum() / max(1, inside_count)),
            "depth_error_median": float(np.median(depth_error)) if len(depth_error) else 0.0,
            "depth_error_p90": float(np.percentile(depth_error, 90)) if len(depth_error) else 0.0,
        }
        self.result_cache[cache_key] = output
        return output


def _depth_pair_stats(depth_relation_cache, left, right):
    if depth_relation_cache is None:
        return {
            "enabled": False,
            "left_to_right": _empty_depth_direction_stats(len(left.get("_seed_indices", []))),
            "right_to_left": _empty_depth_direction_stats(len(right.get("_seed_indices", []))),
        }
    return {
        "enabled": True,
        "left_to_right": depth_relation_cache.direction_stats(left, right),
        "right_to_left": depth_relation_cache.direction_stats(right, left),
    }


def _direction_is_judgeable(stats, min_points, min_ratio, min_floor, source_points):
    valid_points = int(stats.get("valid_depth_points", 0))
    ratio_points = int(math.ceil(float(min_ratio) * max(1, int(source_points))))
    return bool(valid_points >= int(min_points) or valid_points >= max(int(min_floor), ratio_points))


def _relation_from_pair(
    left,
    right,
    reference_counts,
    min_seed_iou,
    min_seed_containment,
    min_reference_coverage,
    spatial_sigma,
    view_consensus_scale,
    projections_np=None,
    visible_np=None,
    scaling_params=None,
    depth_relation_cache=None,
    relation_min_valid_points=30,
    relation_min_valid_ratio=0.15,
    relation_min_valid_floor=20,
    independent_depth_consistency=0.75,
    independent_inside_depth_consistency=0.60,
    independent_visible_iou=0.08,
    independent_visible_containment=0.45,
    independent_support_score=0.65,
    reference_min_seed_coverage=0.50,
    reference_depth_consistency=0.70,
    reference_inside_depth_consistency=0.50,
    reference_visible_iou=0.03,
    reference_visible_containment=0.30,
    hard_conflict_inside_points=30,
    hard_conflict_bidirectional_ratio=0.40,
    hard_conflict_single_ratio=0.60,
):
    left_visible, right_visible, visibility_info = _pair_common_visible_seed_sets(
        left,
        right,
        projections_np=projections_np,
        visible_np=visible_np,
        scaling_params=scaling_params,
    )
    visible_intersection, visible_union, visible_iou, visible_containment = _seed_overlap(left_visible, right_visible)

    left_ref = _cluster_reference_key(left, min_reference_coverage)
    right_ref = _cluster_reference_key(right, min_reference_coverage)
    reference_match = left_ref is not None and left_ref == right_ref
    reference_support = 0.0
    if reference_match:
        reference_support = min(
            1.0,
            float(reference_counts.get((left["class_id"], left_ref), 0)) / max(1.0, float(view_consensus_scale)),
        )
        reference_support = max(
            reference_support,
            min(
                1.0,
                min(
                    float(left.get("best_existing_seed_coverage", 0.0) or 0.0),
                    float(right.get("best_existing_seed_coverage", 0.0) or 0.0),
                ),
            ),
        )

    depth_pair_info = _depth_pair_stats(depth_relation_cache, left, right)
    left_depth_stats = depth_pair_info["left_to_right"]
    right_depth_stats = depth_pair_info["right_to_left"]
    left_judgeable = _direction_is_judgeable(
        left_depth_stats,
        relation_min_valid_points,
        relation_min_valid_ratio,
        relation_min_valid_floor,
        len(left.get("_seed_indices", [])),
    )
    right_judgeable = _direction_is_judgeable(
        right_depth_stats,
        relation_min_valid_points,
        relation_min_valid_ratio,
        relation_min_valid_floor,
        len(right.get("_seed_indices", [])),
    )
    pair_judgeable = bool(
        int(left.get("frame_index", -1)) != int(right.get("frame_index", -1))
        and len(left.get("_seed_indices", [])) >= 80
        and len(right.get("_seed_indices", [])) >= 80
        and left_judgeable
        and right_judgeable
    )
    pair_depth_support = min(
        float(left_depth_stats.get("depth_consistent_ratio", 0.0)),
        float(right_depth_stats.get("depth_consistent_ratio", 0.0)),
    )
    pair_mask_support = min(
        float(left_depth_stats.get("inside_depth_consistent_ratio", 0.0)),
        float(right_depth_stats.get("inside_depth_consistent_ratio", 0.0)),
    )

    spatial_consistency = 0.0
    left_centroid = left.get("seed_centroid")
    right_centroid = right.get("seed_centroid")
    if left_centroid is not None and right_centroid is not None:
        distance = float(np.linalg.norm(np.asarray(left_centroid) - np.asarray(right_centroid)))
        spatial_consistency = float(np.exp(-distance / max(1e-6, float(spatial_sigma))))

    view_consensus = max(reference_support, min(1.0, (int(reference_match) + visible_containment) / 2.0))
    support_score = float(
        0.25 * min(1.0, visible_iou / max(1e-6, float(min_seed_iou)))
        + 0.20 * min(1.0, visible_containment / max(1e-6, float(min_seed_containment)))
        + 0.10 * reference_support
        + 0.10 * spatial_consistency
        + 0.25 * pair_depth_support
        + 0.10 * pair_mask_support
    )
    left_inside_conflict = (
        int(left_depth_stats.get("inside_mask_points", 0)) >= int(hard_conflict_inside_points)
        and float(left_depth_stats.get("inside_depth_conflict_ratio", 0.0)) >= float(hard_conflict_single_ratio)
    )
    right_inside_conflict = (
        int(right_depth_stats.get("inside_mask_points", 0)) >= int(hard_conflict_inside_points)
        and float(right_depth_stats.get("inside_depth_conflict_ratio", 0.0)) >= float(hard_conflict_single_ratio)
    )
    bidirectional_depth_conflict = (
        int(left_depth_stats.get("inside_mask_points", 0)) >= int(hard_conflict_inside_points)
        and int(right_depth_stats.get("inside_mask_points", 0)) >= int(hard_conflict_inside_points)
        and float(left_depth_stats.get("inside_depth_conflict_ratio", 0.0)) >= float(hard_conflict_bidirectional_ratio)
        and float(right_depth_stats.get("inside_depth_conflict_ratio", 0.0)) >= float(hard_conflict_bidirectional_ratio)
    )
    hard_conflict = bool(left_inside_conflict or right_inside_conflict or bidirectional_depth_conflict)
    independent_support = (
        pair_judgeable
        and not hard_conflict
        and float(left_depth_stats.get("depth_consistent_ratio", 0.0)) >= float(independent_depth_consistency)
        and float(right_depth_stats.get("depth_consistent_ratio", 0.0)) >= float(independent_depth_consistency)
        and float(left_depth_stats.get("inside_depth_consistent_ratio", 0.0)) >= float(independent_inside_depth_consistency)
        and float(right_depth_stats.get("inside_depth_consistent_ratio", 0.0)) >= float(independent_inside_depth_consistency)
        and (visible_iou >= float(independent_visible_iou) or visible_containment >= float(independent_visible_containment))
        and support_score >= float(independent_support_score)
    )
    reference_support_ok = (
        pair_judgeable
        and reference_match
        and not hard_conflict
        and min(
            float(left.get("best_existing_seed_coverage", 0.0) or 0.0),
            float(right.get("best_existing_seed_coverage", 0.0) or 0.0),
        ) >= float(reference_min_seed_coverage)
        and float(left_depth_stats.get("depth_consistent_ratio", 0.0)) >= float(reference_depth_consistency)
        and float(right_depth_stats.get("depth_consistent_ratio", 0.0)) >= float(reference_depth_consistency)
        and float(left_depth_stats.get("inside_depth_consistent_ratio", 0.0)) >= float(reference_inside_depth_consistency)
        and float(right_depth_stats.get("inside_depth_consistent_ratio", 0.0)) >= float(reference_inside_depth_consistency)
        and (visible_iou >= float(reference_visible_iou) or visible_containment >= float(reference_visible_containment))
    )
    same_object_support = bool(independent_support or reference_support_ok)
    support_kind = "independent" if independent_support else ("mask3d_reference_assisted" if reference_support_ok else "none")

    left_count = len(left_visible)
    right_count = len(right_visible)
    size_ratio = float(max(left_count, right_count) / max(1, min(left_count, right_count)))
    if left_count >= right_count:
        parent_idx, child_idx = "left", "right"
        parent_count, child_count = left_count, right_count
    else:
        parent_idx, child_idx = "right", "left"
        parent_count, child_count = right_count, left_count
    containment_support = visible_containment >= float(min_seed_containment) and size_ratio >= 1.25
    containment_strength = float(max(visible_containment, support_score))

    return {
        "same_object_support": bool(same_object_support),
        "support_kind": support_kind,
        "independent_support": bool(independent_support),
        "reference_assisted_support": bool(reference_support_ok),
        "hard_conflict": bool(hard_conflict),
        "hard_conflict_reason": (
            "bidirectional_depth_conflict"
            if bidirectional_depth_conflict
            else ("left_to_right_depth_conflict" if left_inside_conflict else ("right_to_left_depth_conflict" if right_inside_conflict else ""))
        ),
        "pair_judgeable": bool(pair_judgeable),
        "containment_support": bool(containment_support),
        "containment_parent": parent_idx,
        "containment_parent_count": int(parent_count),
        "containment_child_count": int(child_count),
        "containment_strength": containment_strength,
        "visible_intersection": int(visible_intersection),
        "visible_union": int(visible_union),
        "visible_iou": float(visible_iou),
        "visible_containment": float(visible_containment),
        "support_score": support_score,
        "reference_match": bool(reference_match),
        "has_common_mask3d_reference": bool(reference_match),
        "reference_support": float(reference_support),
        "depth_support": float(pair_depth_support),
        "mask_support": float(pair_mask_support),
        "visibility_info": visibility_info,
        "depth_pair_info": depth_pair_info,
        "spatial_consistency": float(spatial_consistency),
        "view_consensus": float(view_consensus),
        "same_class": int(left["class_id"]) == int(right["class_id"]),
        "class_mismatch": int(left["class_id"]) != int(right["class_id"]),
        "left_reference_id": int(left_ref) if left_ref is not None else None,
        "right_reference_id": int(right_ref) if right_ref is not None else None,
    }


def _build_mask_graph(
    observations,
    points_xyz=None,
    projections_np=None,
    visible_np=None,
    scaling_params=None,
    depth_relation_cache=None,
    same_class_only=True,
    min_seed_iou=0.03,
    min_seed_containment=0.18,
    min_reference_coverage=0.20,
    spatial_sigma=0.35,
    view_consensus_scale=4.0,
    edge_score_threshold=0.35,
    weak_edge_threshold=None,
    conflict_edge_threshold=None,
    cross_view_conflict_min_visible_ratio=0.45,
    cross_view_conflict_max_inside_ratio=0.05,
    relation_min_valid_points=30,
    relation_min_valid_ratio=0.15,
    relation_min_valid_floor=20,
    independent_depth_consistency=0.75,
    independent_inside_depth_consistency=0.60,
    independent_visible_iou=0.08,
    independent_visible_containment=0.45,
    independent_support_score=0.65,
    reference_min_seed_coverage=0.50,
    reference_depth_consistency=0.70,
    reference_inside_depth_consistency=0.50,
    reference_visible_iou=0.03,
    reference_visible_containment=0.30,
    hard_conflict_inside_points=30,
    hard_conflict_bidirectional_ratio=0.40,
    hard_conflict_single_ratio=0.60,
):
    reference_counts = defaultdict(int)
    for obs in observations:
        key = (obs["class_id"], _cluster_reference_key(obs, min_reference_coverage))
        if key[1] is not None:
            reference_counts[key] += 1

    centroids = [_seed_centroid(points_xyz, obs["_seed_indices"]) for obs in observations]
    for obs, centroid in zip(observations, centroids):
        obs["seed_centroid"] = centroid
    relation_kwargs = {
        "reference_counts": reference_counts,
        "min_seed_iou": min_seed_iou,
        "min_seed_containment": min_seed_containment,
        "min_reference_coverage": min_reference_coverage,
        "spatial_sigma": spatial_sigma,
        "view_consensus_scale": view_consensus_scale,
        "projections_np": projections_np,
        "visible_np": visible_np,
        "scaling_params": scaling_params,
        "depth_relation_cache": depth_relation_cache,
        "relation_min_valid_points": relation_min_valid_points,
        "relation_min_valid_ratio": relation_min_valid_ratio,
        "relation_min_valid_floor": relation_min_valid_floor,
        "independent_depth_consistency": independent_depth_consistency,
        "independent_inside_depth_consistency": independent_inside_depth_consistency,
        "independent_visible_iou": independent_visible_iou,
        "independent_visible_containment": independent_visible_containment,
        "independent_support_score": independent_support_score,
        "reference_min_seed_coverage": reference_min_seed_coverage,
        "reference_depth_consistency": reference_depth_consistency,
        "reference_inside_depth_consistency": reference_inside_depth_consistency,
        "reference_visible_iou": reference_visible_iou,
        "reference_visible_containment": reference_visible_containment,
        "hard_conflict_inside_points": hard_conflict_inside_points,
        "hard_conflict_bidirectional_ratio": hard_conflict_bidirectional_ratio,
        "hard_conflict_single_ratio": hard_conflict_single_ratio,
    }

    same_frame_groups = defaultdict(list)
    for obs_index, obs in enumerate(observations):
        same_frame_groups[int(obs["frame_index"])].append(obs_index)
    same_frame_containment_children = defaultdict(set)
    same_frame_conflict_counts = defaultdict(int)
    same_frame_overlap_counts = defaultdict(int)
    same_frame_relation_edges = []
    for frame_indices in same_frame_groups.values():
        for pos, left_idx in enumerate(frame_indices):
            left = observations[left_idx]
            for right_idx in frame_indices[pos + 1:]:
                right = observations[right_idx]
                relation = _relation_from_pair(
                    left,
                    right,
                    **relation_kwargs,
                )
                edge_base = {
                    "left": int(left_idx),
                    "right": int(right_idx),
                    "same_frame": True,
                    "same_class": bool(relation["same_class"]),
                    "seed_intersection": int(relation["visible_intersection"]),
                    "seed_union": int(relation["visible_union"]),
                    "seed_iou": float(relation["visible_iou"]),
                    "seed_containment": float(relation["visible_containment"]),
                    "coarse_reference_overlap": float(relation["reference_support"]),
                    "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                    "depth_consistency": float(relation["depth_support"]),
                    "mask_consistency": float(relation["mask_support"]),
                    "view_consensus_score": float(relation["view_consensus"]),
                    "visibility_info": relation.get("visibility_info", {}),
                    "depth_pair_info": relation.get("depth_pair_info", {}),
                    "support_kind": relation.get("support_kind", "none"),
                    "has_common_mask3d_reference": bool(relation.get("has_common_mask3d_reference", False)),
                    "pair_judgeable": bool(relation.get("pair_judgeable", False)),
                }
                if relation["hard_conflict"]:
                    conflict_score = float(max(
                        relation["depth_pair_info"]["left_to_right"].get("inside_depth_conflict_ratio", 0.0),
                        relation["depth_pair_info"]["right_to_left"].get("inside_depth_conflict_ratio", 0.0),
                    ))
                    same_frame_conflict_counts[left_idx] += 1
                    same_frame_conflict_counts[right_idx] += 1
                    same_frame_relation_edges.append(
                        {
                            **edge_base,
                            "relation_type": "same_frame_depth_conflict",
                            "same_object_score": 0.0,
                            "containment_score": 0.0,
                            "conflict_score": conflict_score,
                            "edge_score": conflict_score,
                            "conflict_reason": relation.get("hard_conflict_reason", "depth_conflict"),
                        }
                    )
                elif relation["containment_support"]:
                    parent = left_idx if relation["containment_parent"] == "left" else right_idx
                    child = right_idx if relation["containment_parent"] == "left" else left_idx
                    same_frame_containment_children[parent].add(child)
                    same_frame_relation_edges.append(
                        {
                            **edge_base,
                            "relation_type": "same_frame_containment",
                            "parent": int(parent),
                            "child": int(child),
                            "containment_score": float(relation["containment_strength"]),
                            "conflict_score": 0.0,
                            "edge_score": float(relation["containment_strength"]),
                        }
                    )
                elif relation["visible_intersection"] > 0:
                    same_frame_overlap_counts[left_idx] += 1
                    same_frame_overlap_counts[right_idx] += 1
                    relation_type = "same_frame_conflict" if relation["class_mismatch"] else "same_frame_mutex"
                    conflict_score = float(max(relation["support_score"], relation["visible_containment"]))
                    same_frame_conflict_counts[left_idx] += 1
                    same_frame_conflict_counts[right_idx] += 1
                    same_frame_relation_edges.append(
                        {
                            **edge_base,
                            "relation_type": relation_type,
                            "same_object_score": 0.0,
                            "containment_score": 0.0,
                            "conflict_score": conflict_score,
                            "edge_score": conflict_score,
                        }
                    )
    for obs_index, obs in enumerate(observations):
        child_count = len(same_frame_containment_children.get(obs_index, set()))
        geometry_info = obs.get("sam_mask_geometry") if isinstance(obs.get("sam_mask_geometry"), dict) else {}
        geometry_multi_region = (
            int(geometry_info.get("geometry_component_count", geometry_info.get("component_count", 0)) or 0) >= 2
            and float(geometry_info.get("geometry_non_largest_component_ratio", 0.0) or 0.0) >= 0.15
        )
        undersegmentation_evidence = {
            "same_frame_parent_child": bool(child_count >= 2),
            "same_frame_mutex": bool(int(same_frame_conflict_counts.get(obs_index, 0)) >= 1),
            "geometry_multi_region": bool(geometry_multi_region),
        }
        obs["same_frame_child_count"] = int(child_count)
        obs["same_frame_conflict_count"] = int(same_frame_conflict_counts.get(obs_index, 0))
        obs["same_frame_overlap_count"] = int(same_frame_overlap_counts.get(obs_index, 0))
        obs["undersegmentation_evidence"] = undersegmentation_evidence
        obs["undersegmentation_bridge_risk"] = bool(sum(1 for value in undersegmentation_evidence.values() if value) >= 2)

    relation_edges = []
    support_edges = []
    weak_edges = []
    conflict_edges = []
    adjacency = [[] for _ in observations]
    weak_threshold = float(max(0.15, edge_score_threshold * 0.70)) if weak_edge_threshold is None else float(weak_edge_threshold)
    conflict_threshold = float(max(0.20, edge_score_threshold * 0.60)) if conflict_edge_threshold is None else float(conflict_edge_threshold)
    for left_idx in range(len(observations)):
        left = observations[left_idx]
        for right_idx in range(left_idx + 1, len(observations)):
            right = observations[right_idx]
            relation = _relation_from_pair(
                left,
                right,
                **relation_kwargs,
            )
            if int(left["frame_index"]) == int(right["frame_index"]):
                continue

            if relation["hard_conflict"]:
                conflict_score = float(max(
                    relation["depth_pair_info"]["left_to_right"].get("inside_depth_conflict_ratio", 0.0),
                    relation["depth_pair_info"]["right_to_left"].get("inside_depth_conflict_ratio", 0.0),
                ))
                conflict_edge = {
                    "left": int(left_idx),
                    "right": int(right_idx),
                    "relation_type": "depth_conflict",
                    "same_object_score": 0.0,
                    "containment_score": 0.0,
                    "conflict_score": conflict_score,
                    "relation_score": conflict_score,
                    "edge_score": conflict_score,
                    "seed_intersection": int(relation["visible_intersection"]),
                    "seed_union": int(relation["visible_union"]),
                    "seed_iou": float(relation["visible_iou"]),
                    "seed_containment": float(relation["visible_containment"]),
                    "class_compatible": bool(relation["same_class"]),
                    "coarse_reference_overlap": float(relation["reference_support"]),
                    "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                    "depth_consistency": float(relation["depth_support"]),
                    "mask_consistency": float(relation["mask_support"]),
                    "view_consensus_score": float(relation["view_consensus"]),
                    "visibility_info": relation.get("visibility_info", {}),
                    "depth_pair_info": relation.get("depth_pair_info", {}),
                    "same_class": bool(relation["same_class"]),
                    "semantic_mismatch": bool(relation["class_mismatch"]),
                    "conflict_reason": relation.get("hard_conflict_reason", "depth_conflict"),
                    "support_kind": "none",
                    "has_common_mask3d_reference": bool(relation.get("has_common_mask3d_reference", False)),
                    "pair_judgeable": bool(relation.get("pair_judgeable", False)),
                }
                conflict_edges.append(conflict_edge)
                relation_edges.append(conflict_edge)
                continue

            if relation["class_mismatch"] and not relation["same_object_support"]:
                if relation["reference_match"] or relation["support_score"] >= weak_threshold:
                    uncertain_edge = {
                        "left": int(left_idx),
                        "right": int(right_idx),
                        "relation_type": "uncertain",
                        "same_object_score": 0.0,
                        "containment_score": 0.0,
                        "conflict_score": 0.0,
                        "uncertainty_score": float(max(relation["support_score"], relation["spatial_consistency"], relation["reference_support"])),
                        "edge_score": float(max(relation["support_score"], relation["spatial_consistency"], relation["reference_support"])),
                        "seed_iou": float(relation["visible_iou"]),
                        "seed_containment": float(relation["visible_containment"]),
                        "coarse_reference_overlap": float(relation["reference_support"]),
                        "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                        "depth_consistency": float(relation["depth_support"]),
                        "mask_consistency": float(relation["mask_support"]),
                        "view_consensus_score": float(relation["view_consensus"]),
                        "visibility_info": relation.get("visibility_info", {}),
                        "depth_pair_info": relation.get("depth_pair_info", {}),
                        "same_class": False,
                        "semantic_mismatch": True,
                        "support_kind": "none",
                        "has_common_mask3d_reference": bool(relation.get("has_common_mask3d_reference", False)),
                        "pair_judgeable": bool(relation.get("pair_judgeable", False)),
                    }
                    weak_edges.append(uncertain_edge)
                    relation_edges.append(uncertain_edge)
                continue

            visibility_info = relation.get("visibility_info", {})
            left_visible_ratio = float(visibility_info.get("left_to_right_visible_ratio", 0.0) or 0.0)
            right_visible_ratio = float(visibility_info.get("right_to_left_visible_ratio", 0.0) or 0.0)
            left_inside_ratio = float(visibility_info.get("left_to_right_inside_visible_ratio", 0.0) or 0.0)
            right_inside_ratio = float(visibility_info.get("right_to_left_inside_visible_ratio", 0.0) or 0.0)
            mutual_visible_ratio = min(left_visible_ratio, right_visible_ratio)
            mutual_inside_ratio = max(left_inside_ratio, right_inside_ratio)
            cross_view_mutex = (
                not relation["reference_match"]
                and mutual_visible_ratio >= float(cross_view_conflict_min_visible_ratio)
                and mutual_inside_ratio <= float(cross_view_conflict_max_inside_ratio)
                and relation["visible_iou"] <= max(1e-6, float(min_seed_iou) * 0.25)
            )
            if cross_view_mutex:
                conflict_score = float(mutual_visible_ratio * (1.0 - mutual_inside_ratio))
                conflict_edge = {
                    "left": int(left_idx),
                    "right": int(right_idx),
                    "relation_type": "cross_view_conflict",
                    "same_object_score": 0.0,
                    "containment_score": 0.0,
                    "conflict_score": conflict_score,
                    "relation_score": conflict_score,
                    "edge_score": conflict_score,
                    "seed_intersection": int(relation["visible_intersection"]),
                    "seed_union": int(relation["visible_union"]),
                    "seed_iou": float(relation["visible_iou"]),
                    "seed_containment": float(relation["visible_containment"]),
                    "class_compatible": bool(not relation["class_mismatch"]),
                    "coarse_reference_overlap": float(relation["reference_support"]),
                    "coarse_reference_id": None,
                    "depth_consistency": float(relation["depth_support"]),
                    "mask_consistency": float(relation["mask_support"]),
                    "view_consensus_score": float(relation["view_consensus"]),
                    "visibility_info": visibility_info,
                    "depth_pair_info": relation.get("depth_pair_info", {}),
                    "same_class": bool(relation["same_class"]),
                    "semantic_mismatch": bool(relation["class_mismatch"]),
                    "conflict_reason": "mutual_visibility_without_mask_agreement",
                    "support_kind": "none",
                    "has_common_mask3d_reference": bool(relation.get("has_common_mask3d_reference", False)),
                    "pair_judgeable": bool(relation.get("pair_judgeable", False)),
                }
                conflict_edges.append(conflict_edge)
                relation_edges.append(conflict_edge)
                continue

            strong_same_object = bool(relation["same_object_support"])
            if not strong_same_object and not relation["containment_support"]:
                if relation["support_score"] >= weak_threshold:
                    weak_edge = {
                        "left": int(left_idx),
                        "right": int(right_idx),
                        "relation_type": "weak",
                        "same_object_score": float(relation["support_score"]),
                        "containment_score": float(relation["containment_strength"]) if relation["containment_support"] else 0.0,
                        "conflict_score": 0.0,
                        "relation_score": float(relation["support_score"]),
                        "edge_score": float(relation["support_score"]),
                        "seed_intersection": int(relation["visible_intersection"]),
                        "seed_union": int(relation["visible_union"]),
                        "seed_iou": float(relation["visible_iou"]),
                        "seed_containment": float(relation["visible_containment"]),
                        "class_compatible": True,
                        "coarse_reference_overlap": float(relation["reference_support"]),
                        "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                        "depth_consistency": float(relation["depth_support"]),
                        "mask_consistency": float(relation["mask_support"]),
                        "view_consensus_score": float(relation["view_consensus"]),
                        "visibility_info": relation.get("visibility_info", {}),
                        "depth_pair_info": relation.get("depth_pair_info", {}),
                        "same_class": True,
                        "support_kind": "none",
                        "has_common_mask3d_reference": bool(relation.get("has_common_mask3d_reference", False)),
                        "pair_judgeable": bool(relation.get("pair_judgeable", False)),
                    }
                    weak_edges.append(weak_edge)
                    relation_edges.append(weak_edge)
                continue

            relation_type = "same_object" if strong_same_object else "containment"
            relation_edge = {
                "left": int(left_idx),
                "right": int(right_idx),
                "relation_type": relation_type,
                "same_object_score": float(relation["support_score"]) if relation["same_object_support"] else 0.0,
                "containment_score": float(relation["containment_strength"]) if relation["containment_support"] else 0.0,
                "conflict_score": 0.0,
                "relation_score": float(relation["support_score"]),
                "edge_score": float(relation["support_score"]),
                "seed_intersection": int(relation["visible_intersection"]),
                "seed_union": int(relation["visible_union"]),
                "seed_iou": float(relation["visible_iou"]),
                "seed_containment": float(relation["visible_containment"]),
                "class_compatible": bool(not relation["class_mismatch"]),
                "coarse_reference_overlap": float(relation["reference_support"]),
                "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                "depth_consistency": float(relation["depth_support"]),
                "mask_consistency": float(relation["mask_support"]),
                "view_consensus_score": float(relation["view_consensus"]),
                "visibility_info": relation.get("visibility_info", {}),
                "depth_pair_info": relation.get("depth_pair_info", {}),
                "same_class": bool(relation["same_class"]),
                "semantic_mismatch": bool(relation["class_mismatch"]),
                "class_pending": bool(relation["class_mismatch"] and relation_type == "same_object"),
                "has_common_mask3d_reference": bool(relation.get("has_common_mask3d_reference", False)),
                "support_kind": relation.get("support_kind", "none"),
                "independent_support": bool(relation.get("independent_support", False)),
                "reference_assisted_support": bool(relation.get("reference_assisted_support", False)),
                "pair_judgeable": bool(relation.get("pair_judgeable", False)),
            }
            if relation["containment_support"]:
                if relation["containment_parent"] == "left":
                    relation_edge["parent"] = int(left_idx)
                    relation_edge["child"] = int(right_idx)
                else:
                    relation_edge["parent"] = int(right_idx)
                    relation_edge["child"] = int(left_idx)
                relation_edge["containment_strength"] = float(relation["containment_strength"])
            relation_edges.append(relation_edge)
            if relation_type == "same_object":
                support_edges.append(relation_edge)
                adjacency[left_idx].append((right_idx, relation_edge))
                adjacency[right_idx].append((left_idx, relation_edge))
    relation_edges = same_frame_relation_edges + relation_edges
    conflict_edges = [
        edge
        for edge in relation_edges
        if "conflict" in str(edge.get("relation_type")) or str(edge.get("relation_type")) in {"same_frame_mutex"}
    ]
    weak_edges = [edge for edge in relation_edges if str(edge.get("relation_type")) in {"weak", "uncertain"}]
    return relation_edges, adjacency, conflict_edges, weak_edges


def _connected_components(num_nodes, adjacency):
    visited = np.zeros((num_nodes,), dtype=bool)
    components = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor, _ in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _build_constrained_instance_hypotheses(
    observations,
    adjacency,
    relation_edges,
    *,
    min_support_edges=1,
    min_join_support_edges=2,
    high_score_independent_support=0.75,
    min_support_member_ratio=0.30,
    seed_quality_top_ratio=0.30,
    allow_undersegmentation_bridge=False,
):
    source_components = _connected_components(len(observations), adjacency)
    source_component_by_node = {}
    for component_id, component in enumerate(source_components):
        for node in component:
            source_component_by_node[int(node)] = int(component_id)

    conflict_pairs = set()
    weak_pairs = set()
    containment_pairs = set()
    for edge in relation_edges:
        left = int(edge.get("left", -1))
        right = int(edge.get("right", -1))
        if left < 0 or right < 0:
            continue
        key = tuple(sorted((left, right)))
        relation_type = str(edge.get("relation_type", ""))
        if "conflict" in relation_type or relation_type in {"same_frame_mutex"}:
            conflict_pairs.add(key)
        elif relation_type == "weak":
            weak_pairs.add(key)
        elif "containment" in relation_type:
            containment_pairs.add(key)

    edge_lookup = {}
    for left, node_edges in enumerate(adjacency):
        for right, edge in node_edges:
            edge_lookup[tuple(sorted((int(left), int(right))))] = edge

    assigned = {}
    hypotheses = []
    skipped = []
    blocked_nodes = set()
    deferred_nodes_by_component = defaultdict(set)
    deferred_dependencies_by_component = defaultdict(dict)
    non_seed_reasons = defaultdict(list)
    hypothesis_attempt_id = 0

    def add_non_seed_reason(node, reason_info):
        reasons = non_seed_reasons[int(node)]
        reason = reason_info.get("reason")
        if not any(item.get("reason") == reason for item in reasons):
            reasons.append(reason_info)

    def node_sort_key(idx):
        return (
            -_graph_quality_score(observations[idx]),
            -len(observations[idx].get("_seed_indices", [])),
            int(observations[idx].get("graph_observation_id", idx)),
        )

    def bridge_risk(node):
        return bool(observations[node].get("undersegmentation_bridge_risk", False))

    def same_object_edge(node, member):
        key = tuple(sorted((int(node), int(member))))
        edge = edge_lookup.get(key)
        if edge is not None and str(edge.get("relation_type")) == "same_object":
            return edge
        return None

    def has_independent_support(node, candidates):
        for member in candidates:
            edge = same_object_edge(node, member)
            if edge is not None and bool(edge.get("independent_support", False)):
                return True
        return False

    def can_add(node, members):
        support_edges = []
        conflict_hits = []
        weak_hits = []
        containment_hits = []
        for member in members:
            key = tuple(sorted((int(node), int(member))))
            if key in conflict_pairs:
                conflict_hits.append(int(member))
            if key in weak_pairs:
                weak_hits.append(int(member))
            if key in containment_pairs:
                containment_hits.append(int(member))
            edge = same_object_edge(node, member)
            if edge is not None:
                support_edges.append(edge)
        if conflict_hits:
            return False, {
                "reason": "conflict_with_hypothesis",
                "conflict_members": conflict_hits,
                "support_edge_count": int(len(support_edges)),
            }
        if not allow_undersegmentation_bridge and bool(observations[node].get("undersegmentation_bridge_risk", False)):
            return False, {
                "reason": "undersegmentation_bridge_risk",
                "support_edge_count": int(len(support_edges)),
                "same_frame_child_count": int(observations[node].get("same_frame_child_count", 0)),
                "same_frame_conflict_count": int(observations[node].get("same_frame_conflict_count", 0)),
            }
        support_member_ratio = float(len(support_edges) / max(1, len(members)))
        support_frames = {
            int(observations[member].get("frame_index", -1))
            for member in members
            if same_object_edge(node, member) is not None
        }
        high_score_independent = any(
            bool(edge.get("independent_support", False))
            and float(edge.get("edge_score", 0.0)) >= float(high_score_independent_support)
            for edge in support_edges
        )
        enough_support = (
            len(support_edges) >= int(max(1, min_join_support_edges))
            and len(support_frames) >= min(2, int(max(1, min_join_support_edges)))
        ) or high_score_independent
        if not enough_support:
            return False, {
                "reason": "insufficient_support_edges",
                "support_edge_count": int(len(support_edges)),
                "support_frame_count": int(len(support_frames)),
                "high_score_independent_support": bool(high_score_independent),
                "weak_hits": weak_hits,
                "containment_hits": containment_hits,
            }
        if support_member_ratio < float(min_support_member_ratio):
            return False, {
                "reason": "low_hypothesis_support_coverage",
                "support_edge_count": int(len(support_edges)),
                "hypothesis_member_count": int(len(members)),
                "support_member_ratio": float(support_member_ratio),
                "min_support_member_ratio": float(min_support_member_ratio),
            }
        return True, {
            "reason": "accepted",
            "support_edge_count": int(len(support_edges)),
            "support_frame_count": int(len(support_frames)),
            "support_member_ratio": float(support_member_ratio),
            "high_score_independent_support": bool(high_score_independent),
            "max_support_score": float(max(float(edge.get("edge_score", 0.0)) for edge in support_edges)),
        }

    def valid_hypothesis(members):
        frames = {int(observations[node].get("frame_index", -1)) for node in members}
        union_seed = np.unique(np.concatenate([np.asarray(observations[node].get("_seed_indices", []), dtype=np.int64) for node in members]))
        independent_edge_count = 0
        for pos, left in enumerate(members):
            for right in members[pos + 1:]:
                edge = same_object_edge(left, right)
                if edge is not None and bool(edge.get("independent_support", False)):
                    independent_edge_count += 1
        bridge_count = sum(1 for node in members if bridge_risk(node))
        reasons = []
        if len(frames) < 2:
            reasons.append("few_views")
        if independent_edge_count < 1:
            reasons.append("no_independent_support")
        if bridge_count > 0:
            reasons.append("contains_undersegmentation_bridge")
        if len(union_seed) < 80:
            reasons.append("few_full_core_points")
        return not reasons, {
            "view_count": int(len(frames)),
            "independent_support_edge_count": int(independent_edge_count),
            "undersegmentation_bridge_count": int(bridge_count),
            "full_core_point_count": int(len(union_seed)),
            "invalid_reasons": reasons,
        }

    for source_component_id, source_component in enumerate(source_components):
        source_set = set(int(node) for node in source_component)
        if len(source_component) <= 1:
            node = int(source_component[0]) if source_component else -1
            if node >= 0:
                skipped.append(
                    {
                        "observation": node,
                        "source_component_id": int(source_component_id),
                        "reason": "singleton_without_cross_view_support",
                    }
                )
                blocked_nodes.add(node)
            continue

        while True:
            remaining = [
                int(node)
                for node in source_component
                if int(node) not in assigned
                and int(node) not in blocked_nodes
                and int(node) not in deferred_nodes_by_component[int(source_component_id)]
            ]
            if not remaining:
                break
            ranked_remaining = sorted([int(node) for node in remaining], key=node_sort_key)
            top_k = max(1, int(math.ceil(len(source_component) * float(seed_quality_top_ratio))))
            top_quality_nodes = set(sorted([int(node) for node in source_component], key=node_sort_key)[:top_k])
            seed_candidates = []
            for node in ranked_remaining:
                if not allow_undersegmentation_bridge and bridge_risk(node):
                    skipped.append(
                        {
                            "observation": node,
                            "source_component_id": int(source_component_id),
                            "reason": "undersegmentation_bridge_seed_blocked",
                            "same_frame_child_count": int(observations[node].get("same_frame_child_count", 0)),
                            "same_frame_conflict_count": int(observations[node].get("same_frame_conflict_count", 0)),
                        }
                    )
                    blocked_nodes.add(node)
                    continue
                if node not in top_quality_nodes:
                    add_non_seed_reason(
                        node,
                        {
                            "reason": "seed_quality_below_component_top_ratio",
                            "seed_quality_top_ratio": float(seed_quality_top_ratio),
                        },
                    )
                    continue
                if not has_independent_support(node, source_set):
                    add_non_seed_reason(
                        node,
                        {
                            "reason": "seed_without_independent_support",
                        },
                    )
                    continue
                seed_candidates.append(node)
            if not seed_candidates:
                break

            seed = sorted(seed_candidates, key=node_sort_key)[0]
            hypothesis_id = len(hypotheses)
            hypothesis_attempt_id += 1
            current_attempt_token = ("attempt", int(hypothesis_attempt_id))
            members = [int(seed)]
            assigned[int(seed)] = hypothesis_id
            trace = [
                {
                    "observation": int(seed),
                    "action": "seed",
                    "reason": "highest_remaining_quality_in_source_component",
                    "source_component_id": int(source_component_id),
                    "source_component_size": int(len(source_component)),
                    "undersegmentation_bridge_risk": bool(observations[seed].get("undersegmentation_bridge_risk", False)),
                }
            ]
            changed = True
            while changed:
                changed = False
                candidate_neighbors = set()
                for member in members:
                    for neighbor, _ in adjacency[member]:
                        neighbor = int(neighbor)
                        if (
                            neighbor in source_set
                            and neighbor not in assigned
                            and neighbor not in deferred_nodes_by_component[int(source_component_id)]
                        ):
                            candidate_neighbors.add(neighbor)
                for neighbor in sorted(candidate_neighbors, key=node_sort_key):
                    compatible_hypotheses = []
                    compatible_tokens = []
                    ok, decision = can_add(neighbor, members)
                    if ok:
                        compatible_hypotheses.append(hypothesis_id)
                        compatible_tokens.append(current_attempt_token)
                    for existing_hypothesis in hypotheses:
                        if int(existing_hypothesis.get("source_component_id", -1)) != int(source_component_id):
                            continue
                        existing_ok, _ = can_add(neighbor, existing_hypothesis["members"])
                        if existing_ok:
                            existing_id = int(existing_hypothesis["graph_hypothesis_id"])
                            compatible_hypotheses.append(existing_id)
                            compatible_tokens.append(("hypothesis", existing_id))
                    if len(set(compatible_tokens)) > 1:
                        deferred_nodes_by_component[int(source_component_id)].add(int(neighbor))
                        deferred_dependencies_by_component[int(source_component_id)][int(neighbor)] = set(compatible_tokens)
                        trace.append(
                            {
                                "observation": int(neighbor),
                                "action": "defer",
                                "reason": "ambiguous_multiple_hypotheses",
                                "compatible_hypotheses": sorted(set(int(item) for item in compatible_hypotheses)),
                            }
                        )
                        continue
                    if not ok:
                        trace.append({"observation": int(neighbor), "action": "reject", **decision})
                        continue
                    members.append(int(neighbor))
                    assigned[int(neighbor)] = hypothesis_id
                    trace.append({"observation": int(neighbor), "action": "join", **decision})
                    changed = True

            is_valid, validity = valid_hypothesis(members)
            if (len(members) <= 1 and int(min_support_edges) > 0) or not is_valid:
                for member in members:
                    assigned.pop(int(member), None)
                released_deferred = []
                reassigned_deferred = []
                component_deps = deferred_dependencies_by_component[int(source_component_id)]
                for node, dependencies in list(component_deps.items()):
                    if current_attempt_token not in dependencies:
                        continue
                    remaining_dependencies = set(dependencies)
                    remaining_dependencies.discard(current_attempt_token)
                    if len(remaining_dependencies) == 1:
                        only_dependency = next(iter(remaining_dependencies))
                        if only_dependency[0] == "hypothesis":
                            target_hypothesis = next(
                                (
                                    item
                                    for item in hypotheses
                                    if int(item.get("graph_hypothesis_id", -1)) == int(only_dependency[1])
                                ),
                                None,
                            )
                            if target_hypothesis is not None:
                                target_ok, target_decision = can_add(int(node), target_hypothesis["members"])
                                if target_ok:
                                    target_hypothesis["members"] = sorted(set(int(member) for member in target_hypothesis["members"]) | {int(node)})
                                    assigned[int(node)] = int(target_hypothesis["graph_hypothesis_id"])
                                    target_hypothesis.setdefault("trace", []).append(
                                        {
                                            "observation": int(node),
                                            "action": "join_after_deferred_release",
                                            "reason": "failed_hypothesis_removed_ambiguity",
                                            **target_decision,
                                        }
                                    )
                                    _, updated_validity = valid_hypothesis(target_hypothesis["members"])
                                    target_hypothesis["validity"] = updated_validity
                                    deferred_nodes_by_component[int(source_component_id)].discard(int(node))
                                    component_deps.pop(int(node), None)
                                    reassigned_deferred.append(int(node))
                                    continue
                    if len(remaining_dependencies) <= 1:
                        deferred_nodes_by_component[int(source_component_id)].discard(int(node))
                        component_deps.pop(int(node), None)
                        released_deferred.append(int(node))
                    else:
                        component_deps[int(node)] = remaining_dependencies
                skipped.append(
                    {
                        "observation": int(seed),
                        "source_component_id": int(source_component_id),
                        "reason": "invalid_hypothesis",
                        "members": sorted(int(node) for node in members),
                        "released_deferred_observations": sorted(released_deferred),
                        "reassigned_deferred_observations": sorted(reassigned_deferred),
                        **validity,
                    }
                )
                blocked_nodes.add(int(seed))
                continue

            component_deps = deferred_dependencies_by_component[int(source_component_id)]
            for node, dependencies in list(component_deps.items()):
                if current_attempt_token not in dependencies:
                    continue
                updated_dependencies = set(dependencies)
                updated_dependencies.discard(current_attempt_token)
                updated_dependencies.add(("hypothesis", int(hypothesis_id)))
                component_deps[int(node)] = updated_dependencies

            hypotheses.append(
                {
                    "graph_hypothesis_id": int(hypothesis_id),
                    "members": sorted(members),
                    "trace": trace,
                    "formation_policy": "constrained_instance_hypothesis",
                    "source_component_id": int(source_component_id),
                    "source_component_size": int(len(source_component)),
                    "validity": validity,
                }
            )

    for idx in range(len(observations)):
        if idx not in assigned:
            if idx in blocked_nodes:
                continue
            skipped.append(
                {
                    "observation": int(idx),
                    "source_component_id": int(source_component_by_node.get(int(idx), -1)),
                    "reason": "deferred_after_hypothesis_building" if idx in deferred_nodes_by_component.get(
                        int(source_component_by_node.get(int(idx), -1)), set()
                    ) else "unassigned_after_hypothesis_building",
                    "non_seed_reasons": non_seed_reasons.get(int(idx), []),
                }
            )
    return hypotheses, skipped


def _hypothesis_partition_stats(connected_components, hypotheses, hypothesis_skipped):
    by_source = defaultdict(list)
    for hypothesis in hypotheses:
        source_id = hypothesis.get("source_component_id")
        if source_id is None:
            continue
        by_source[int(source_id)].append(hypothesis)
    skipped_by_source = defaultdict(set)
    skipped_observations = set()
    for item in hypothesis_skipped:
        source_id = item.get("source_component_id")
        if item.get("observation") is not None:
            observation_id = int(item["observation"])
            skipped_observations.add(observation_id)
            if source_id is not None:
                skipped_by_source[int(source_id)].add(observation_id)

    split_components = 0
    unchanged_components = 0
    dropped_components = 0
    component_rows = []
    for component_id, component in enumerate(connected_components):
        hypothesis_count = len(by_source.get(int(component_id), []))
        if hypothesis_count > 1:
            split_components += 1
        elif hypothesis_count == 1:
            unchanged_components += 1
        else:
            dropped_components += 1
        component_rows.append(
            {
                "source_component_id": int(component_id),
                "source_component_size": int(len(component)),
                "hypothesis_count": int(hypothesis_count),
                "skipped_observation_count": int(len(skipped_by_source.get(int(component_id), set()))),
            }
        )
    return {
        "source_component_count": int(len(connected_components)),
        "hypothesis_count": int(len(hypotheses)),
        "split_source_component_count": int(split_components),
        "unchanged_source_component_count": int(unchanged_components),
        "dropped_source_component_count": int(dropped_components),
        "unassigned_observation_count": int(len(skipped_observations)),
        "components": component_rows,
    }


def _component_edge_stats(component, adjacency, relation_edges=None):
    component_set = set(component)
    edge_scores = []
    seed_ious = []
    containments = []
    depth_scores = []
    consensus_scores = []
    relation_kind_counts = defaultdict(int)
    relation_kind_scores = defaultdict(list)
    support_kind_counts = defaultdict(int)
    containment_parent_count = 0
    containment_child_count = 0
    seen = set()
    for node in component:
        for neighbor, edge in adjacency[node]:
            if neighbor not in component_set:
                continue
            key = tuple(sorted((node, neighbor)))
            if key in seen:
                continue
            seen.add(key)
            edge_scores.append(edge["edge_score"])
            seed_ious.append(edge["seed_iou"])
            containments.append(edge["seed_containment"])
            depth_scores.append(edge["depth_consistency"])
            consensus_scores.append(edge["view_consensus_score"])
            relation_kind = edge.get("relation_type", "same_object")
            relation_kind_counts[relation_kind] += 1
            relation_kind_scores[relation_kind].append(float(edge.get("relation_score", edge.get("edge_score", 0.0))))
            if relation_kind == "same_object":
                support_kind_counts[str(edge.get("support_kind", "none"))] += 1
            if float(edge.get("containment_score", 0.0) or 0.0) > 0.0:
                relation_kind_counts["containment"] += 1
                relation_kind_scores["containment"].append(float(edge.get("containment_score", edge.get("containment_strength", 0.0))))
    internal_conflict_count = 0
    external_conflict_count = 0
    internal_weak_count = 0
    external_weak_count = 0
    internal_uncertain_count = 0
    external_uncertain_count = 0
    if relation_edges is not None:
        for edge in relation_edges:
            left_in = edge["left"] in component_set
            right_in = edge["right"] in component_set
            if not left_in and not right_in:
                continue
            internal_edge = bool(left_in and right_in)
            relation_kind = edge.get("relation_type", "same_object")
            if "conflict" in relation_kind or relation_kind in {"same_frame_mutex"}:
                if internal_edge:
                    internal_conflict_count += 1
                    relation_kind_scores["conflict"].append(float(edge.get("conflict_score", edge.get("edge_score", 0.0))))
                else:
                    external_conflict_count += 1
            elif relation_kind == "weak":
                if internal_edge:
                    internal_weak_count += 1
                    relation_kind_scores[relation_kind].append(float(edge.get("relation_score", edge.get("edge_score", 0.0))))
                else:
                    external_weak_count += 1
            elif relation_kind == "uncertain":
                if internal_edge:
                    internal_uncertain_count += 1
                    relation_kind_scores[relation_kind].append(float(edge.get("uncertainty_score", edge.get("edge_score", 0.0))))
                else:
                    external_uncertain_count += 1
            elif internal_edge and relation_kind == "containment":
                relation_kind_counts["containment"] += 1
                relation_kind_scores["containment"].append(float(edge.get("containment_score", edge.get("containment_strength", 0.0))))
            if internal_edge and edge.get("parent") in component_set:
                containment_parent_count += 1
            if internal_edge and edge.get("child") in component_set:
                containment_child_count += 1
    return {
        "edge_count": int(len(edge_scores)),
        "edge_mean_score": _safe_mean(edge_scores),
        "seed_mean_iou": _safe_mean(seed_ious),
        "seed_mean_containment": _safe_mean(containments),
        "depth_consistency_score": _safe_mean(depth_scores),
        "graph_consensus_score": _safe_mean(consensus_scores),
        "same_object_edge_count": int(relation_kind_counts.get("same_object", 0)),
        "independent_support_edge_count": int(support_kind_counts.get("independent", 0)),
        "reference_assisted_support_edge_count": int(support_kind_counts.get("mask3d_reference_assisted", 0)),
        "containment_edge_count": int(relation_kind_counts.get("containment", 0)),
        "weak_edge_count": int(max(internal_weak_count, relation_kind_counts.get("weak", 0))),
        "conflict_edge_count": int(max(internal_conflict_count, relation_kind_counts.get("conflict", 0))),
        "uncertain_edge_count": int(internal_uncertain_count),
        "external_weak_edge_count": int(external_weak_count),
        "external_conflict_edge_count": int(external_conflict_count),
        "external_uncertain_edge_count": int(external_uncertain_count),
        "same_object_edge_mean_score": _safe_mean(relation_kind_scores.get("same_object", [])),
        "containment_edge_mean_score": _safe_mean(relation_kind_scores.get("containment", [])),
        "weak_edge_mean_score": _safe_mean(relation_kind_scores.get("weak", [])),
        "conflict_edge_mean_score": _safe_mean(relation_kind_scores.get("conflict", [])),
        "uncertain_edge_mean_score": _safe_mean(relation_kind_scores.get("uncertain", [])),
        "containment_parent_edge_count": int(containment_parent_count),
        "containment_child_edge_count": int(containment_child_count),
    }


def _existing_seed_coverage_mask(
    existing_masks,
    seed_indices,
    *,
    existing_classes=None,
    existing_scores=None,
    candidate_class_id=None,
    require_class_compatible=False,
    min_existing_score=0.0,
    min_mask_seed_coverage=0.0,
):
    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    if len(seed_indices) == 0 or existing_masks is None or existing_masks.size == 0:
        return np.zeros((len(seed_indices),), dtype=bool)
    existing_masks = np.asarray(existing_masks, dtype=bool)
    max_seed_index = int(seed_indices.max()) if len(seed_indices) > 0 else -1
    if existing_masks.shape[0] <= max_seed_index and existing_masks.shape[1] > max_seed_index:
        existing_masks = existing_masks.T
    if existing_masks.shape[0] <= max_seed_index:
        return np.zeros((len(seed_indices),), dtype=bool)
    seed_rows = existing_masks[seed_indices]
    keep_masks = np.ones((seed_rows.shape[1],), dtype=bool)
    if require_class_compatible:
        if existing_classes is None or candidate_class_id is None:
            keep_masks[:] = False
        else:
            existing_classes = np.asarray(existing_classes, dtype=np.int64)
            if existing_classes.shape[0] != keep_masks.shape[0]:
                keep_masks[:] = False
            else:
                keep_masks &= existing_classes == int(candidate_class_id)
    if float(min_existing_score) > 0.0:
        if existing_scores is None:
            keep_masks[:] = False
        else:
            existing_scores = np.asarray(existing_scores, dtype=np.float32)
            if existing_scores.shape[0] == keep_masks.shape[0]:
                keep_masks &= existing_scores >= float(min_existing_score)
            else:
                keep_masks[:] = False
    if float(min_mask_seed_coverage) > 0.0 and seed_rows.shape[1] > 0:
        per_mask_seed_coverage = seed_rows.sum(axis=0) / max(1, len(seed_indices))
        keep_masks &= per_mask_seed_coverage >= float(min_mask_seed_coverage)
    if not keep_masks.any():
        return np.zeros((len(seed_indices),), dtype=bool)
    return seed_rows[:, keep_masks].any(axis=1)


def _graph_gap_metrics(
    seed_indices,
    existing_masks,
    points_xyz,
    existing_classes=None,
    existing_scores=None,
    candidate_class_id=None,
    reliable_existing_coverage=False,
    min_existing_score=0.0,
    min_mask_seed_coverage=0.05,
    min_uncovered_points=80,
    min_uncovered_ratio=0.30,
    min_largest_component_ratio=0.50,
    cc_radius=0.03,
    cc_max_points=50000,
):
    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    coverage_mask = _existing_seed_coverage_mask(
        existing_masks,
        seed_indices,
        existing_classes=existing_classes if reliable_existing_coverage else None,
        existing_scores=existing_scores if reliable_existing_coverage else None,
        candidate_class_id=candidate_class_id if reliable_existing_coverage else None,
        require_class_compatible=bool(reliable_existing_coverage),
        min_existing_score=min_existing_score if reliable_existing_coverage else 0.0,
        min_mask_seed_coverage=min_mask_seed_coverage if reliable_existing_coverage else 0.0,
    )
    uncovered_indices = seed_indices[~coverage_mask].astype(np.int64)
    seed_count = int(len(seed_indices))
    uncovered_count = int(len(uncovered_indices))
    uncovered_ratio = float(uncovered_count / max(1, seed_count))
    if points_xyz is not None and uncovered_count > 0:
        local_points = points_xyz[uncovered_indices].astype(np.float32, copy=False)
        cc_info = _connected_component_summary(local_points, cc_radius, cc_max_points)
    else:
        cc_info = {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "small_component_ratio": 0.0,
            "component_skipped": points_xyz is None and uncovered_count > 0,
            "component_radius": float(cc_radius),
        }
    largest_ratio = float(cc_info.get("largest_component_ratio", 0.0) or 0.0)
    largest_points = int(round(uncovered_count * largest_ratio))
    is_new_gap = (
        uncovered_count >= int(min_uncovered_points)
        and uncovered_ratio >= float(min_uncovered_ratio)
        and (
            bool(cc_info.get("component_skipped", False))
            or largest_ratio >= float(min_largest_component_ratio)
        )
    )
    return {
        "candidate_action": "new_candidate" if is_new_gap else "existing_support",
        "gap_uncovered_seed_points": uncovered_count,
        "gap_uncovered_seed_ratio": uncovered_ratio,
        "gap_largest_component_points": largest_points,
        "gap_largest_component_ratio": largest_ratio,
        "gap_component_count": int(cc_info.get("component_count", 0) or 0),
        "gap_small_component_ratio": float(cc_info.get("small_component_ratio", 0.0) or 0.0),
        "gap_component_skipped": bool(cc_info.get("component_skipped", False)),
        "gap_component_radius": float(cc_info.get("component_radius", cc_radius) or cc_radius),
        "gap_min_uncovered_points": int(min_uncovered_points),
        "gap_min_uncovered_ratio": float(min_uncovered_ratio),
        "gap_min_largest_component_ratio": float(min_largest_component_ratio),
        "gap_reliable_existing_coverage": bool(reliable_existing_coverage),
        "gap_min_existing_score": float(min_existing_score),
        "gap_min_mask_seed_coverage": float(min_mask_seed_coverage),
        "gap_candidate_class_id": None if candidate_class_id is None else int(candidate_class_id),
        "_gap_uncovered_seed_indices": uncovered_indices,
    }


def _graph_candidate_seed_indices(selected_seed, gap_info, candidate_action, seed_policy="adaptive"):
    selected_seed = np.asarray(selected_seed, dtype=np.int64)
    gap_info = gap_info or {}
    uncovered_seed = np.asarray(gap_info.get("_gap_uncovered_seed_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
    seed_policy = str(seed_policy or "adaptive")

    if seed_policy == "full_core":
        return selected_seed.copy(), "full_core"
    if seed_policy == "uncovered_core":
        if str(candidate_action) == "new_candidate" and len(uncovered_seed) > 0:
            return np.unique(uncovered_seed).astype(np.int64), "uncovered_core"
        return selected_seed.copy(), "selected_seed_fallback"

    if str(candidate_action) == "new_candidate" and len(uncovered_seed) > 0:
        return np.unique(uncovered_seed).astype(np.int64), "uncovered_core"
    return selected_seed.copy(), "full_core"


def _public_candidate_record(candidate):
    return {
        key: value
        for key, value in candidate.items()
        if not str(key).startswith("_")
    }


def _write_existing_support_diagnostic(candidate, diagnostic_id, diagnostic_dir):
    os.makedirs(diagnostic_dir, exist_ok=True)
    output_seed = np.asarray(candidate.get("_seed_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
    full_core_seed = np.asarray(candidate.get("_full_core_seed_indices", output_seed), dtype=np.int64)
    gap_core_seed = np.asarray(candidate.get("_gap_core_seed_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
    prefix = f"existing_support{int(diagnostic_id):04d}"
    output_seed_path = osp.join(diagnostic_dir, f"{prefix}_output_points.npz")
    full_core_seed_path = osp.join(diagnostic_dir, f"{prefix}_full_core_points.npz")
    gap_core_seed_path = osp.join(diagnostic_dir, f"{prefix}_gap_core_points.npz")
    np.savez_compressed(output_seed_path, point_indices=output_seed)
    np.savez_compressed(full_core_seed_path, point_indices=full_core_seed)
    np.savez_compressed(gap_core_seed_path, point_indices=gap_core_seed)

    record = _public_candidate_record(candidate)
    record.update(
        {
            "diagnostic_id": int(diagnostic_id),
            "diagnostic_kind": "existing_instance_revision",
            "candidate_action": "existing_support",
            "revision_options": ["keep_original_mask3d", "trim_by_multiview_core", "complete_with_multiview_core"],
            "output_seed_points_path": output_seed_path,
            "full_core_seed_points_path": full_core_seed_path,
            "gap_core_seed_points_path": gap_core_seed_path,
            "output_seed_point_count": int(len(output_seed)),
            "full_core_seed_point_count": int(len(full_core_seed)),
            "gap_core_seed_point_count": int(len(gap_core_seed)),
        }
    )
    return record


def _graph_candidate_core_indices(selected_seed, gap_info):
    selected_seed = np.unique(np.asarray(selected_seed, dtype=np.int64)).astype(np.int64)
    gap_info = gap_info or {}
    uncovered_seed = np.unique(
        np.asarray(gap_info.get("_gap_uncovered_seed_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
    ).astype(np.int64)
    return selected_seed, uncovered_seed


def _graph_candidate_competition_quality(candidate):
    conflict_total = int(candidate.get("conflict_edge_count", 0))
    relation_total = (
        int(candidate.get("same_object_edge_count", 0))
        + int(candidate.get("containment_edge_count", 0))
        + int(candidate.get("weak_edge_count", 0))
        + conflict_total
    )
    conflict_ratio = float(conflict_total / max(1, relation_total))
    return (
        1 if str(candidate.get("candidate_action")) == "new_candidate" else 0,
        -float(candidate.get("seed_in_existing_mask_ratio", 1.0)),
        float(candidate.get("graph_consensus_score", 0.0)),
        float(candidate.get("depth_consistency_score", 0.0)),
        -conflict_ratio,
        int(candidate.get("support_view_count", 0)),
        float(candidate.get("proposal_priority", 0.0)),
    )


def _graph_competition_priority_factor(candidate):
    source_kind = str(candidate.get("source_kind") or "").strip().lower()
    if not source_kind.startswith("mask_graph"):
        return 1.0

    graph_consensus = float(candidate.get("graph_consensus_score", 0.0) or 0.0)
    depth_consistency = float(candidate.get("depth_consistency_score", 0.0) or 0.0)
    support_view_count = float(candidate.get("support_view_count", 0.0) or 0.0)
    support_view_norm = min(1.0, support_view_count / 4.0)
    gap_ratio = float((candidate.get("gap_info") or {}).get("gap_uncovered_seed_ratio", 0.0) or 0.0)
    seed_existing_ratio = float(candidate.get("seed_in_existing_mask_ratio", 0.0) or 0.0)
    conflict_total = int(candidate.get("conflict_edge_count", 0))
    relation_total = (
        int(candidate.get("same_object_edge_count", 0))
        + int(candidate.get("containment_edge_count", 0))
        + int(candidate.get("weak_edge_count", 0))
        + conflict_total
    )
    conflict_ratio = float(conflict_total / max(1, relation_total))
    quality = (
        0.30 * graph_consensus
        + 0.25 * depth_consistency
        + 0.15 * support_view_norm
        + 0.15 * gap_ratio
        + 0.15 * (1.0 - conflict_ratio)
    )
    if str(candidate.get("candidate_action")) != "new_candidate":
        quality *= 0.65
    if source_kind == "mask_graph_single_view":
        quality *= 0.85
    factor = 0.90 + 0.70 * quality + 0.10 * (1.0 - seed_existing_ratio)
    return float(min(1.60, max(0.75, factor)))


def _apply_graph_candidate_competition(
    candidates,
    *,
    same_class_iou=0.60,
    cross_class_iou=0.35,
    containment=0.80,
):
    ranked = sorted(candidates, key=_graph_candidate_competition_quality, reverse=True)
    kept = []
    skipped = []
    for candidate in ranked:
        candidate_seed = np.asarray(candidate.get("_seed_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
        reject_record = None
        for kept_candidate in kept:
            kept_seed = np.asarray(kept_candidate.get("_seed_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
            intersection, union, seed_iou, seed_containment = _seed_overlap(candidate_seed, kept_seed)
            same_class = int(candidate.get("class_id", -1)) == int(kept_candidate.get("class_id", -2))
            iou_threshold = float(same_class_iou if same_class else cross_class_iou)
            if seed_iou < iou_threshold and seed_containment < float(containment):
                continue
            reject_record = {
                "reason": "graph_candidate_competition",
                "graph_cluster_id": int(candidate.get("graph_cluster_id", -1)),
                "class_name": candidate.get("class_name"),
                "candidate_action": candidate.get("candidate_action"),
                "blocked_by_graph_cluster_id": int(kept_candidate.get("graph_cluster_id", -1)),
                "blocked_by_class_name": kept_candidate.get("class_name"),
                "same_class": bool(same_class),
                "seed_intersection": int(intersection),
                "seed_union": int(union),
                "seed_iou": float(seed_iou),
                "seed_containment": float(seed_containment),
                "same_class_iou_threshold": float(same_class_iou),
                "cross_class_iou_threshold": float(cross_class_iou),
                "containment_threshold": float(containment),
                "candidate_quality": [float(value) for value in _graph_candidate_competition_quality(candidate)],
                "kept_quality": [float(value) for value in _graph_candidate_competition_quality(kept_candidate)],
            }
            break
        if reject_record is None:
            candidate["graph_competition_quality"] = [float(value) for value in _graph_candidate_competition_quality(candidate)]
            kept.append(candidate)
        else:
            skipped.append(reject_record)
    return kept, skipped


def _graph_candidate_rejection_reasons(
    selected_indices,
    stats,
    *,
    min_selected_views=0,
    min_same_object_edges=0,
    min_independent_support_edges=0,
    min_edge_mean_score=0.0,
    min_consensus_score=0.0,
    min_depth_consistency=0.0,
    max_conflict_edges=None,
    max_conflict_ratio=None,
    core_min_largest_component_ratio=None,
    core_max_second_component_ratio=None,
    min_label_majority=None,
    min_label_margin=None,
):
    reasons = []
    selected_count = int(len(selected_indices))
    if int(min_selected_views or 0) > 0 and selected_count < int(min_selected_views):
        reasons.append(
            {
                "reason": "few_selected_views",
                "selected_view_count": selected_count,
                "min_selected_views": int(min_selected_views),
            }
        )
    if int(min_same_object_edges or 0) > 0 and int(stats["same_object_edge_count"]) < int(min_same_object_edges):
        reasons.append(
            {
                "reason": "few_same_object_edges",
                "same_object_edge_count": int(stats["same_object_edge_count"]),
                "min_same_object_edges": int(min_same_object_edges),
            }
        )
    if int(min_independent_support_edges or 0) > 0 and int(stats.get("independent_support_edge_count", 0)) < int(min_independent_support_edges):
        reasons.append(
            {
                "reason": "few_independent_support_edges",
                "independent_support_edge_count": int(stats.get("independent_support_edge_count", 0)),
                "min_independent_support_edges": int(min_independent_support_edges),
            }
        )
    if float(min_edge_mean_score or 0.0) > 0.0 and float(stats["edge_mean_score"]) < float(min_edge_mean_score):
        reasons.append(
            {
                "reason": "low_edge_mean_score",
                "edge_mean_score": float(stats["edge_mean_score"]),
                "min_edge_mean_score": float(min_edge_mean_score),
            }
        )
    if float(min_consensus_score or 0.0) > 0.0 and float(stats["graph_consensus_score"]) < float(min_consensus_score):
        reasons.append(
            {
                "reason": "low_consensus_score",
                "graph_consensus_score": float(stats["graph_consensus_score"]),
                "min_consensus_score": float(min_consensus_score),
            }
        )
    if float(min_depth_consistency or 0.0) > 0.0 and float(stats["depth_consistency_score"]) < float(min_depth_consistency):
        reasons.append(
            {
                "reason": "low_depth_consistency",
                "depth_consistency_score": float(stats["depth_consistency_score"]),
                "min_depth_consistency": float(min_depth_consistency),
            }
        )
    if max_conflict_edges is not None and int(stats["conflict_edge_count"]) > int(max_conflict_edges):
        reasons.append(
            {
                "reason": "too_many_conflict_edges",
                "conflict_edge_count": int(stats["conflict_edge_count"]),
                "max_conflict_edges": int(max_conflict_edges),
            }
        )
    if max_conflict_ratio is not None:
        relation_total = (
            int(stats["same_object_edge_count"])
            + int(stats["containment_edge_count"])
            + int(stats["weak_edge_count"])
            + int(stats["conflict_edge_count"])
        )
        conflict_ratio = float(stats["conflict_edge_count"] / max(1, relation_total))
        if conflict_ratio > float(max_conflict_ratio):
            reasons.append(
                {
                    "reason": "too_many_conflict_edges_ratio",
                    "conflict_edge_count": int(stats["conflict_edge_count"]),
                    "relation_edge_total": int(relation_total),
                    "conflict_ratio": conflict_ratio,
                    "max_conflict_ratio": float(max_conflict_ratio),
                }
            )
    if str(stats.get("candidate_action", "")) == "new_candidate":
        if (
            core_min_largest_component_ratio is not None
            and float(stats.get("core_largest_component_ratio", 0.0)) < float(core_min_largest_component_ratio)
        ):
            reasons.append(
                {
                    "reason": "spatially_disconnected_core",
                    "core_largest_component_ratio": float(stats.get("core_largest_component_ratio", 0.0)),
                    "core_min_largest_component_ratio": float(core_min_largest_component_ratio),
                }
            )
        if (
            core_max_second_component_ratio is not None
            and float(stats.get("core_second_component_ratio", 0.0)) > float(core_max_second_component_ratio)
        ):
            reasons.append(
                {
                    "reason": "large_secondary_core_component",
                    "core_second_component_ratio": float(stats.get("core_second_component_ratio", 0.0)),
                    "core_max_second_component_ratio": float(core_max_second_component_ratio),
                }
            )
        if min_label_majority is not None or min_label_margin is not None:
            label_majority = float(stats.get("label_consensus_score", 0.0) or 0.0)
            label_margin = float(stats.get("label_margin", 0.0) or 0.0)
            majority_ok = min_label_majority is None or label_majority >= float(min_label_majority)
            margin_ok = min_label_margin is None or label_margin >= float(min_label_margin)
            if not (majority_ok or margin_ok):
                reasons.append(
                    {
                        "reason": "weak_label_majority",
                        "label_consensus_score": label_majority,
                        "label_margin": label_margin,
                        "min_label_majority": None if min_label_majority is None else float(min_label_majority),
                        "min_label_margin": None if min_label_margin is None else float(min_label_margin),
                    }
                )
    return reasons


def _select_cluster_views(
    observations,
    component,
    adjacency,
    max_views=4,
    min_new_seed_ratio=0.05,
    redundancy_penalty=0.20,
):
    ranked = sorted(component, key=lambda idx: (-_graph_quality_score(observations[idx]), -len(observations[idx]["_seed_indices"])))
    selected = []
    selected_seed = np.asarray([], dtype=np.int64)
    max_views = int(max(1, max_views))
    while ranked and len(selected) < max_views:
        best_idx = None
        best_score = None
        best_seed = None
        for idx in ranked:
            seed = observations[idx]["_seed_indices"]
            if len(selected_seed) == 0:
                new_seed_ratio = 1.0
                redundancy = 0.0
            else:
                intersection = np.intersect1d(selected_seed, seed, assume_unique=False).size
                new_seed_ratio = float((len(seed) - intersection) / max(1, len(seed)))
                redundancy = float(intersection / max(1, min(len(selected_seed), len(seed))))
            if selected and new_seed_ratio < float(min_new_seed_ratio):
                continue
            graph_support = 0.0
            for neighbor, edge in adjacency[idx]:
                if neighbor in selected:
                    graph_support = max(graph_support, float(edge.get("edge_score", 0.0)))
            score = (
                0.45 * _graph_quality_score(observations[idx])
                + 0.30 * new_seed_ratio
                + 0.20 * graph_support
                - float(redundancy_penalty) * redundancy
                + 0.05 * min(1.0, math.log1p(len(seed)) / math.log(10000.0))
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
                best_seed = seed
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_seed = np.union1d(selected_seed, best_seed).astype(np.int64)
        ranked = [idx for idx in ranked if idx != best_idx]

    if not selected and component:
        selected = [component[0]]
        selected_seed = observations[component[0]]["_seed_indices"].copy()
    return selected, selected_seed.astype(np.int64)


def _vote_selected_seed_points(
    observations,
    selected_indices,
    fallback_seed,
    min_vote_score=0.35,
    min_support_count=1,
    min_keep_ratio=0.35,
    min_keep_points=0,
    allow_fallback=False,
):
    vote_scores = defaultdict(float)
    support_counts = defaultdict(int)
    max_vote = 0.0
    for idx in selected_indices:
        obs = observations[idx]
        seed = np.asarray(obs["_seed_indices"], dtype=np.int64)
        if len(seed) == 0:
            continue
        view_weight = max(0.05, _graph_quality_score(obs))
        depth_weight = 0.5 + 0.5 * float(obs.get("seed_depth_support_ratio", 0.0))
        weight = float(view_weight * depth_weight)
        max_vote = max(max_vote, weight)
        for point_idx in seed:
            point_idx = int(point_idx)
            vote_scores[point_idx] += weight
            support_counts[point_idx] += 1

    if not vote_scores:
        return np.asarray([], dtype=np.int64), {
            "enabled": False,
            "fallback": "no_votes",
            "min_vote_score": float(min_vote_score),
            "min_support_count": int(min_support_count),
            "kept_points": 0,
            "input_points": int(len(fallback_seed)),
        }

    threshold = float(max(0.0, min_vote_score))
    min_support = int(max(1, min_support_count))
    if len(selected_indices) > 1:
        min_support = max(min_support, 2)
    input_count = int(len(np.unique(fallback_seed)))
    kept = [
        point_idx
        for point_idx, score in vote_scores.items()
        if support_counts[point_idx] >= min_support and float(score) >= threshold
    ]
    if not kept:
        if allow_fallback and len(selected_indices) > 1:
            kept = [
                int(point_idx)
                for point_idx, score in vote_scores.items()
                if support_counts[point_idx] >= 1 and float(score) >= threshold
            ]
            fallback = "single_support_fallback"
        else:
            return np.asarray([], dtype=np.int64), {
                "enabled": True,
                "fallback": "rejected_after_vote",
                "min_vote_score": float(threshold),
                "min_support_count": int(min_support),
                "min_keep_ratio": float(min_keep_ratio),
                "min_keep_points": int(max(0, min_keep_points)),
                "input_points": input_count,
                "kept_points": 0,
                "kept_ratio": 0.0,
                "max_vote_score": float(max_vote),
                "mean_vote_score": float(np.mean(list(vote_scores.values()))) if vote_scores else 0.0,
                "multi_view_point_count": int(sum(1 for value in support_counts.values() if int(value) >= 2)),
            }
    else:
        fallback = None
    kept = np.asarray(sorted(set(kept)), dtype=np.int64)
    kept_ratio = float(len(kept) / max(1, input_count))
    if (
        len(selected_indices) > 1
        and input_count > 0
        and (
            kept_ratio < float(min_keep_ratio)
            or len(kept) < int(max(0, min_keep_points))
        )
    ):
        if allow_fallback:
            kept = np.asarray(sorted(set(int(point_idx) for point_idx in fallback_seed)), dtype=np.int64)
            kept_ratio = float(len(kept) / max(1, input_count))
            fallback = "small_core_fallback"
        else:
            return np.asarray([], dtype=np.int64), {
                "enabled": True,
                "fallback": "rejected_small_core",
                "min_vote_score": float(threshold),
                "min_support_count": int(min_support),
                "min_keep_ratio": float(min_keep_ratio),
                "min_keep_points": int(max(0, min_keep_points)),
                "input_points": input_count,
                "kept_points": 0,
                "kept_ratio": 0.0,
                "max_vote_score": float(max_vote),
                "mean_vote_score": float(np.mean(list(vote_scores.values()))) if vote_scores else 0.0,
                "multi_view_point_count": int(sum(1 for value in support_counts.values() if int(value) >= 2)),
            }
    return kept, {
        "enabled": True,
        "fallback": fallback,
        "min_vote_score": float(threshold),
        "min_support_count": int(min_support),
        "min_keep_ratio": float(min_keep_ratio),
        "min_keep_points": int(max(0, min_keep_points)),
        "input_points": input_count,
        "kept_points": int(len(kept)),
        "kept_ratio": kept_ratio,
        "max_vote_score": float(max_vote),
        "mean_vote_score": float(np.mean(list(vote_scores.values()))) if vote_scores else 0.0,
        "multi_view_point_count": int(sum(1 for value in support_counts.values() if int(value) >= 2)),
    }


def _merge_cluster_label_consensus(cluster_record, observations, selected_indices):
    first = True
    for idx in selected_indices:
        obs = observations[idx]
        label_record = {
            key: obs.get(key)
            for key in (
                "label_consensus_score",
                "label_conflict_score",
                "label_margin",
                "label_entropy",
                "label_consensus_view_count",
                "label_conflict_view_count",
                "label_evidence_view_count",
                "label_target_evidence",
                "label_total_evidence",
                "top_conflicting_class_id",
                "top_conflicting_evidence",
            )
            if key in obs
        }
        if first:
            cluster_record.update(label_record)
            first = False
        else:
            _merge_label_consensus(cluster_record, label_record)
    return cluster_record


def _component_class_vote(observations, component):
    class_votes = defaultdict(float)
    for idx in component:
        obs = observations[idx]
        class_votes[int(obs["class_id"])] += float(obs.get("fusion_score", obs.get("score", 0.0))) * max(
            0.1,
            float(obs.get("sam_score", 0.0)),
        )
    class_id = max(class_votes.items(), key=lambda item: item[1])[0]
    matching_class_observations = [observations[idx] for idx in component if int(observations[idx]["class_id"]) == int(class_id)]
    if matching_class_observations:
        class_name = max(
            matching_class_observations,
            key=lambda obs: (_graph_quality_score(obs), obs.get("proposal_priority", 0.0), len(obs["_seed_indices"])),
        )["class_name"]
    else:
        best_observation = max(
            [observations[idx] for idx in component],
            key=lambda obs: (_graph_quality_score(obs), obs.get("proposal_priority", 0.0), len(obs["_seed_indices"])),
        )
        class_name = best_observation["class_name"]
    return int(class_id), class_name


def _cluster_to_candidate(
    cluster_id,
    observations,
    component,
    selected_indices,
    selected_seed,
    adjacency,
    relation_edges,
    existing_masks,
    points_xyz=None,
    seed_vote_info=None,
    gap_info=None,
    graph_gap_seed_policy="adaptive",
    hypothesis_trace=None,
    hypothesis_formation_policy=None,
):
    selected_observations = [observations[idx] for idx in selected_indices]
    best_observation = max(
        selected_observations,
        key=lambda obs: (_graph_quality_score(obs), obs.get("proposal_priority", 0.0), len(obs["_seed_indices"])),
    )
    class_id, class_name = _component_class_vote(observations, component)
    stats = _component_edge_stats(component, adjacency, relation_edges=relation_edges)
    undersegmentation_bridge_observations = [
        int(observations[idx]["graph_observation_id"])
        for idx in component
        if bool(observations[idx].get("undersegmentation_bridge_risk", False))
    ]
    candidate_action = (gap_info or {}).get("candidate_action", "new_candidate")
    full_core_seed_indices, gap_core_seed_indices = _graph_candidate_core_indices(selected_seed, gap_info)
    if points_xyz is not None and len(full_core_seed_indices) > 0:
        core_cc_info = _connected_component_summary(points_xyz[full_core_seed_indices], 0.03, 50000)
    else:
        core_cc_info = {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "small_component_ratio": 0.0,
            "component_skipped": points_xyz is None and len(full_core_seed_indices) > 0,
            "component_radius": 0.03,
        }
    core_largest_ratio = float(core_cc_info.get("largest_component_ratio", 0.0) or 0.0)
    final_seed_indices, applied_seed_policy = _graph_candidate_seed_indices(
        selected_seed,
        gap_info,
        candidate_action,
        seed_policy=graph_gap_seed_policy,
    )
    existing_metrics = _existing_mask_metrics(existing_masks, final_seed_indices)
    support_views = []
    for rank, obs in enumerate(selected_observations):
        view = dict(obs["support_views"][0])
        view["graph_observation_id"] = int(obs["graph_observation_id"])
        view["graph_selected_rank"] = int(rank)
        support_views.append(view)

    conflict_penalty = max(0.50, 1.0 - 0.20 * min(2, int(stats["conflict_edge_count"])))
    containment_penalty = max(0.70, 1.0 - 0.10 * min(3, int(stats["containment_parent_edge_count"])))
    candidate = {
        "scene_name": best_observation["scene_name"],
        "source_kind": "mask_graph_multi_view" if int(len(component)) >= 2 and int(stats["edge_count"]) > 0 else "mask_graph_single_view",
        "candidate_action": candidate_action,
        "graph_gap_seed_policy": str(graph_gap_seed_policy),
        "graph_gap_seed_policy_applied": str(applied_seed_policy),
        "frame_id": best_observation["frame_id"],
        "frame_index": int(best_observation["frame_index"]),
        "detection_id": int(best_observation["detection_id"]),
        "sam_mask_id": int(best_observation["sam_mask_id"]),
        "sam_mask_rank": int(best_observation.get("sam_mask_rank", 0)),
        "sam_score_rank": int(best_observation.get("sam_score_rank", 0)),
        "class_id": int(class_id),
        "class_name": class_name,
        "score": float(max(obs.get("score", 0.0) for obs in selected_observations)),
        "bbox_xyxy": best_observation["bbox_xyxy"],
        "box_area_ratio": float(best_observation.get("box_area_ratio", 0.0)),
        "fusion_score": float(_safe_mean([obs.get("fusion_score", obs.get("score", 0.0)) for obs in selected_observations])),
        "proposal_priority": float(
            max(0.05, stats["graph_consensus_score"])
            * max(obs.get("proposal_priority", 0.0) for obs in selected_observations)
            * (0.5 + 0.5 * min(1.0, len(selected_indices) / 4.0))
            * conflict_penalty
            * containment_penalty
        ),
        "support_view_count": int(len(selected_indices)),
        "support_mean_iou": float(stats["seed_mean_iou"]),
        "support_best_iou": float(
            max(
                [edge["seed_iou"] for idx in component for _, edge in adjacency[idx] if edge["right"] in component or edge["left"] in component]
                or [0.0]
            )
        ),
        "support_mean_score": float(_safe_mean([obs.get("score", 0.0) for obs in selected_observations])),
        "support_best_score": float(max(obs.get("score", 0.0) for obs in selected_observations)),
        "view_quality_score": float(_safe_mean([obs.get("view_quality_score", 0.0) for obs in selected_observations])),
        "support_views": support_views,
        "merged_observations": int(len(component)),
        "selected_seed_view_count": int(len(selected_indices)),
        "selected_seed_point_count": int(len(selected_seed)),
        "output_seed_point_count": int(len(final_seed_indices)),
        "full_core_seed_point_count": int(len(full_core_seed_indices)),
        "gap_core_seed_point_count": int(len(gap_core_seed_indices)),
        "core_component_count": int(core_cc_info.get("component_count", 0) or 0),
        "core_largest_component_ratio": core_largest_ratio,
        "core_second_component_ratio": float(max(0.0, 1.0 - core_largest_ratio)),
        "core_small_component_ratio": float(core_cc_info.get("small_component_ratio", 0.0) or 0.0),
        "core_component_skipped": bool(core_cc_info.get("component_skipped", False)),
        "core_component_radius": float(core_cc_info.get("component_radius", 0.03) or 0.03),
        "available_seed_view_count": int(len(component)),
        "graph_cluster_id": int(cluster_id),
        "graph_cluster_observation_ids": [int(observations[idx]["graph_observation_id"]) for idx in component],
        "graph_selected_observation_ids": [int(observations[idx]["graph_observation_id"]) for idx in selected_indices],
        "hypothesis_formation_policy": hypothesis_formation_policy or "connected_component",
        "hypothesis_trace": hypothesis_trace or [],
        "cluster_observation_count": int(len(component)),
        "selected_view_count": int(len(selected_indices)),
        "graph_edge_count": int(stats["edge_count"]),
        "graph_edge_mean_score": float(stats["edge_mean_score"]),
        "graph_consensus_score": float(stats["graph_consensus_score"]),
        "depth_consistency_score": float(stats["depth_consistency_score"]),
        "seed_mean_containment": float(stats["seed_mean_containment"]),
        "same_object_edge_count": int(stats["same_object_edge_count"]),
        "independent_support_edge_count": int(stats.get("independent_support_edge_count", 0)),
        "reference_assisted_support_edge_count": int(stats.get("reference_assisted_support_edge_count", 0)),
        "containment_edge_count": int(stats["containment_edge_count"]),
        "weak_edge_count": int(stats["weak_edge_count"]),
        "conflict_edge_count": int(stats["conflict_edge_count"]),
        "uncertain_edge_count": int(stats.get("uncertain_edge_count", 0)),
        "external_weak_edge_count": int(stats.get("external_weak_edge_count", 0)),
        "external_conflict_edge_count": int(stats.get("external_conflict_edge_count", 0)),
        "external_uncertain_edge_count": int(stats.get("external_uncertain_edge_count", 0)),
        "same_object_edge_mean_score": float(stats["same_object_edge_mean_score"]),
        "containment_edge_mean_score": float(stats["containment_edge_mean_score"]),
        "weak_edge_mean_score": float(stats["weak_edge_mean_score"]),
        "conflict_edge_mean_score": float(stats["conflict_edge_mean_score"]),
        "uncertain_edge_mean_score": float(stats.get("uncertain_edge_mean_score", 0.0)),
        "containment_parent_edge_count": int(stats["containment_parent_edge_count"]),
        "containment_child_edge_count": int(stats["containment_child_edge_count"]),
        "undersegmentation_bridge_observation_count": int(len(undersegmentation_bridge_observations)),
        "undersegmentation_bridge_observation_ids": undersegmentation_bridge_observations,
        "same_frame_child_count_sum": int(sum(int(observations[idx].get("same_frame_child_count", 0)) for idx in component)),
        "same_frame_conflict_count_sum": int(sum(int(observations[idx].get("same_frame_conflict_count", 0)) for idx in component)),
        "conflict_penalty": float(conflict_penalty),
        "containment_penalty": float(containment_penalty),
        "seed_vote_info": seed_vote_info or {"enabled": False},
        "gap_info": {
            key: value
            for key, value in (gap_info or {}).items()
            if not key.startswith("_")
        },
        "graph_competition_priority_factor": float(_graph_competition_priority_factor(
            {
                "source_kind": "mask_graph_multi_view" if int(len(component)) >= 2 and int(stats["edge_count"]) > 0 else "mask_graph_single_view",
                "candidate_action": candidate_action,
                "graph_consensus_score": float(stats["graph_consensus_score"]),
                "depth_consistency_score": float(stats["depth_consistency_score"]),
                "support_view_count": int(len(selected_indices)),
                "gap_info": gap_info or {},
                "seed_in_existing_mask_ratio": float(existing_metrics.get("seed_in_existing_mask_ratio", 0.0)),
                "conflict_edge_count": int(stats["conflict_edge_count"]),
                "same_object_edge_count": int(stats["same_object_edge_count"]),
                "containment_edge_count": int(stats["containment_edge_count"]),
                "weak_edge_count": int(stats["weak_edge_count"]),
            }
        )),
        "sam_mask_selection_policy": best_observation.get("sam_mask_selection_policy"),
        "sam_mask_selection_score": best_observation.get("sam_mask_selection_score"),
        "sam_mask_geometry": best_observation.get("sam_mask_geometry"),
        "evidence": best_observation.get("evidence", {}),
        "_seed_indices": final_seed_indices,
        "_full_core_seed_indices": full_core_seed_indices,
        "_gap_core_seed_indices": gap_core_seed_indices,
        **existing_metrics,
    }
    candidate["proposal_priority"] = float(candidate["proposal_priority"] * candidate["graph_competition_priority_factor"])
    _merge_cluster_label_consensus(candidate, observations, selected_indices)
    return candidate


def collect_scene_mask_observations(
    openyolo3d,
    predictor,
    scene_name,
    output_dir,
    detection_score_th=0.45,
    min_seed_points=80,
    max_box_area_ratio=0.30,
    frame_stride=5,
    max_frames=None,
    max_detections_per_frame=8,
    blocked_classes=None,
    sam_multimask_topk=1,
    sam_mask_selection_policy="sam_score",
    sam_mask_geometry_model=None,
    sam_mask_geometry_cc_radius=0.03,
    sam_mask_geometry_plane_threshold=0.02,
    sam_mask_geometry_max_points=50000,
    seed_depth_cluster=False,
    seed_depth_cluster_bin_size=0.10,
    seed_depth_cluster_window_bins=1,
    seed_depth_cluster_min_keep_ratio=0.25,
    seed_depth_cluster_min_removed_ratio=0.0,
    seed_depth_cluster_max_removed_ratio=1.0,
    sam_adaptive_internal_seed=False,
    sam_adaptive_internal_keep_ratio=0.70,
    sam_adaptive_internal_min_keep_ratio=0.35,
    sam_adaptive_internal_boundary_weight=0.45,
    sam_adaptive_internal_depth_weight=0.55,
    sam_adaptive_internal_depth_bin_size=0.10,
    sam_adaptive_internal_depth_window_bins=1,
    sam_mask_erode_pixels=0,
    sam_mask_erode_min_area_ratio=0.15,
    label_consensus_iou_th=0.25,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
):
    labels = openyolo3d.openyolo3d_config["network2d"]["text_prompts"]
    blocked_classes = _parse_class_names(blocked_classes)
    projections, keep_visible_points = openyolo3d.mesh_projections
    projections_np = _to_numpy(projections).astype(np.int64)
    visible_np = _to_numpy(keep_visible_points).astype(bool)
    existing_masks = _to_numpy(openyolo3d.preds_3d[0]).astype(bool)
    if existing_masks.shape[0] != projections_np.shape[1]:
        existing_masks = existing_masks.T
    points_xyz = _load_scene_points_xyz(openyolo3d)
    geometry_model_bundle = sam_mask_geometry_model
    if isinstance(geometry_model_bundle, str):
        geometry_model_bundle = _load_geometry_discriminator(geometry_model_bundle)
    if str(sam_mask_selection_policy) == "learned_geometry" and geometry_model_bundle is None:
        raise ValueError("--sam_mask_geometry_model is required when using learned_geometry selection.")

    scene_dir = osp.join(output_dir, scene_name)
    image_dir = osp.join(scene_dir, "mask_graph_images")
    mask_dir = osp.join(scene_dir, "mask_graph_masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_height, image_width = openyolo3d.world2cam.image_resolution
    frame_indices = list(range(0, len(openyolo3d.world2cam.color_paths), max(1, int(frame_stride))))
    if max_frames is not None:
        frame_indices = frame_indices[: int(max_frames)]

    observations = []
    skipped = []
    raw_observations = 0
    for frame_idx in frame_indices:
        image_path = openyolo3d.world2cam.color_paths[frame_idx]
        frame_id = osp.basename(image_path).split(".")[0]
        frame_pred = openyolo3d.preds_2d.get(frame_id)
        if frame_pred is None:
            continue
        boxes = _to_numpy(frame_pred["bbox"]).astype(np.float32)
        class_ids = _to_numpy(frame_pred["labels"]).astype(np.int64)
        scores = _to_numpy(frame_pred["scores"]).astype(np.float32)
        nms_indices = _select_2d_nms_indices(
            boxes,
            scores,
            class_ids,
            iou_threshold=box_nms_iou,
            same_class_only=box_nms_same_class_only,
        )
        order = nms_indices[np.argsort(-scores[nms_indices])[:max_detections_per_frame]]

        image = _prepare_image(image_path)
        predictor.set_image(image)
        for det_rank, det_id in enumerate(order):
            class_id = int(class_ids[det_id])
            class_name = _safe_label(labels, class_id)
            score = float(scores[det_id])
            if class_name in blocked_classes:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "class_blocked"})
                continue
            if score < detection_score_th:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "low_score"})
                continue
            box = np.asarray(_clamp_box(boxes[det_id], image_width, image_height), dtype=np.float32)
            box_area_ratio = float(((box[2] - box[0]) * (box[3] - box[1])) / max(1, image_width * image_height))
            if box_area_ratio > max_box_area_ratio:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "large_2d_box"})
                continue

            masks, sam_scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            sam_score_order = np.argsort(-sam_scores)
            sam_score_rank_by_id = {int(mask_id): int(rank) for rank, mask_id in enumerate(sam_score_order)}
            mask_order = sam_score_order if str(sam_mask_selection_policy) in {"geometry", "learned_geometry"} else sam_score_order[: max(1, int(sam_multimask_topk))]
            mask_items = []
            for mask_rank, mask_id in enumerate(mask_order):
                mask_id = int(mask_id)
                sam_score = float(sam_scores[mask_id])
                sam_mask = masks[mask_id].astype(bool)
                core_mask, erosion_info = _erode_binary_mask(
                    sam_mask,
                    erode_pixels=sam_mask_erode_pixels,
                    min_area_ratio=sam_mask_erode_min_area_ratio,
                )
                seed_indices = _sam_mask_to_indices(openyolo3d, frame_idx, core_mask, projections_np, visible_np)
                adaptive_internal_info = {"enabled": False}
                if sam_adaptive_internal_seed:
                    seed_indices, adaptive_internal_info = _filter_seed_indices_by_adaptive_internal_seed(
                        openyolo3d,
                        frame_idx,
                        seed_indices,
                        core_mask,
                        projections_np,
                        keep_ratio=sam_adaptive_internal_keep_ratio,
                        min_keep_ratio=sam_adaptive_internal_min_keep_ratio,
                        boundary_weight=sam_adaptive_internal_boundary_weight,
                        depth_weight=sam_adaptive_internal_depth_weight,
                        depth_bin_size=sam_adaptive_internal_depth_bin_size,
                        depth_window_bins=sam_adaptive_internal_depth_window_bins,
                        min_points=min_seed_points,
                    )
                depth_cluster_info = {"enabled": False}
                if seed_depth_cluster:
                    seed_indices, depth_cluster_info = _filter_seed_indices_by_depth_cluster(
                        openyolo3d,
                        frame_idx,
                        seed_indices,
                        projections_np,
                        bin_size=seed_depth_cluster_bin_size,
                        window_bins=seed_depth_cluster_window_bins,
                        min_keep_ratio=seed_depth_cluster_min_keep_ratio,
                        min_removed_ratio=seed_depth_cluster_min_removed_ratio,
                        max_removed_ratio=seed_depth_cluster_max_removed_ratio,
                        min_points=min_seed_points,
                    )
                geometry_info = _sam_seed_geometry_quality(
                    seed_indices,
                    points_xyz,
                    existing_masks,
                    min_seed_points=min_seed_points,
                    cc_radius=sam_mask_geometry_cc_radius,
                    plane_threshold=sam_mask_geometry_plane_threshold,
                    max_points=sam_mask_geometry_max_points,
                )
                learned_geometry_score = None
                if str(sam_mask_selection_policy) == "learned_geometry":
                    existing_metrics = _existing_mask_metrics(existing_masks, seed_indices)
                    feature_row = _sam_mask_discriminator_row(
                        geometry_info,
                        existing_metrics,
                        detection_score=score,
                        sam_score=sam_score,
                        box_area_ratio=box_area_ratio,
                        num_seed_points=len(seed_indices),
                        num_mask_points=len(seed_indices),
                    )
                    learned_geometry_score = _predict_geometry_discriminator_score(feature_row, geometry_model_bundle)
                    selection_score = float(learned_geometry_score)
                else:
                    selection_score = float(0.45 * max(0.0, min(1.0, sam_score)) + 0.55 * float(geometry_info.get("quality_score", 0.0)))
                if len(seed_indices) < min_seed_points:
                    skipped.append(
                        {
                            "frame_id": frame_id,
                            "detection_id": int(det_id),
                            "sam_mask_id": mask_id,
                            "reason": "few_seed_points",
                            "num_seed_points": int(len(seed_indices)),
                            "sam_mask_geometry": geometry_info,
                        }
                    )
                    continue
                visible_seed_mask = visible_np[frame_idx][seed_indices]
                visible_seed_indices, visible_seed_count, visible_seed_ratio = _visible_seed_stats(seed_indices, visible_seed_mask)
                mask_items.append(
                    {
                        "mask_id": mask_id,
                        "sam_score": sam_score,
                        "core_mask": core_mask,
                        "erosion_info": erosion_info,
                        "seed_indices": seed_indices,
                        "adaptive_internal_info": adaptive_internal_info,
                        "depth_cluster_info": depth_cluster_info,
                        "geometry_info": geometry_info,
                        "learned_geometry_score": learned_geometry_score,
                        "selection_score": selection_score,
                        "sam_score_rank": int(sam_score_rank_by_id.get(mask_id, mask_rank)),
                        "sam_mask_rank": int(mask_rank),
                        "visible_seed_indices": visible_seed_indices,
                        "seed_visible_count": int(visible_seed_count),
                        "seed_depth_support_ratio": float(visible_seed_ratio),
                    }
                )

            if str(sam_mask_selection_policy) in {"geometry", "learned_geometry"}:
                mask_items = sorted(
                    mask_items,
                    key=lambda item: (
                        -float(item["selection_score"]),
                        -float(item["geometry_info"].get("quality_score", 0.0)),
                        -float(item["sam_score"]),
                        -len(item["seed_indices"]),
                    ),
                )[: max(1, int(sam_multimask_topk))]
                for rank, item in enumerate(mask_items):
                    item["sam_mask_rank"] = int(rank)

            if not mask_items:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "no_valid_sam_mask"})
                continue

            for item in mask_items:
                mask_id = int(item["mask_id"])
                sam_score = float(item["sam_score"])
                seed_indices = item["seed_indices"]
                visible_seed_indices = item["visible_seed_indices"]
                geometry_info = item["geometry_info"]
                learned_geometry_score = item.get("learned_geometry_score")
                priority_factor = 1.0
                if str(sam_mask_selection_policy) == "geometry":
                    priority_factor = 0.75 + 0.50 * float(geometry_info.get("quality_score", 0.0))
                elif str(sam_mask_selection_policy) == "learned_geometry":
                    priority_factor = 0.75 + 0.50 * float(learned_geometry_score or 0.0)
                view_quality_score = _sam_view_quality_score(
                    geometry_info,
                    detection_score=score,
                    sam_score=sam_score,
                    box_area_ratio=box_area_ratio,
                    num_seed_points=len(seed_indices),
                    min_seed_points=min_seed_points,
                )
                evidence_prefix = osp.join(
                    image_dir,
                    f"obs{raw_observations:05d}_frame{frame_id}_det{int(det_id):03d}_sam{mask_id}",
                )
                mask_prefix = osp.join(
                    mask_dir,
                    f"obs{raw_observations:05d}_frame{frame_id}_det{int(det_id):03d}_sam{mask_id}",
                )
                evidence = _save_overlay(image, item["core_mask"], box, evidence_prefix)
                mask_path = _save_mask(item["core_mask"], mask_prefix)
                observation = {
                    "scene_name": scene_name,
                    "graph_observation_id": int(raw_observations),
                    "frame_id": frame_id,
                    "frame_index": int(frame_idx),
                    "detection_id": int(det_id),
                    "sam_mask_id": mask_id,
                    "sam_mask_rank": int(item["sam_mask_rank"]),
                    "sam_score_rank": int(item["sam_score_rank"]),
                    "sam_score": sam_score,
                    "sam_mask_selection_policy": str(sam_mask_selection_policy),
                    "sam_mask_selection_score": float(item["selection_score"]),
                    "sam_mask_learned_geometry_score": learned_geometry_score,
                    "sam_mask_geometry": geometry_info,
                    "sam_mask_erosion": item["erosion_info"],
                    "sam_adaptive_internal_seed": item["adaptive_internal_info"],
                    "seed_depth_cluster": item["depth_cluster_info"],
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "box_area_ratio": box_area_ratio,
                    "num_seed_points": int(len(seed_indices)),
                    "seed_visible_count": int(item.get("seed_visible_count", 0)),
                    "seed_depth_support_ratio": float(item.get("seed_depth_support_ratio", 0.0)),
                    "proposal_priority": float(score * max(0.1, sam_score) * np.log1p(len(seed_indices)) * priority_factor),
                    "fusion_score": float(score * max(0.1, sam_score)),
                    "support_view_count": 1,
                    "support_mean_iou": 1.0,
                    "support_best_iou": 1.0,
                    "support_mean_score": score,
                    "support_best_score": score,
                    "view_quality_score": view_quality_score,
                    "support_views": [
                        {
                            "frame_id": frame_id,
                            "frame_index": int(frame_idx),
                            "visible_seed_points": int(item.get("seed_visible_count", len(seed_indices))),
                            "iou": 1.0,
                            "score": score,
                            "sam_score": sam_score,
                            "sam_mask_id": mask_id,
                            "sam_mask_rank": int(item["sam_mask_rank"]),
                            "sam_score_rank": int(item["sam_score_rank"]),
                            "bbox_xyxy": [float(v) for v in box.tolist()],
                            "sam_mask_path": mask_path,
                            "sam_mask_selection_policy": str(sam_mask_selection_policy),
                            "sam_mask_selection_score": float(item["selection_score"]),
                            "sam_mask_learned_geometry_score": learned_geometry_score,
                            "sam_mask_geometry_quality_score": float(geometry_info.get("quality_score", 0.0)),
                            "view_quality_score": view_quality_score,
                            "sam_mask_erode_pixels": int(sam_mask_erode_pixels or 0),
                            "sam_mask_core_area_ratio": float(item["erosion_info"].get("area_ratio", 1.0)),
                            "seed_depth_cluster_keep_ratio": float(item["depth_cluster_info"].get("keep_ratio", 1.0)),
                            "seed_depth_support_ratio": float(item.get("seed_depth_support_ratio", 0.0)),
                        }
                    ],
                    **_label_consensus_metrics(box, class_id, boxes, class_ids, scores, label_consensus_iou_th),
                    "evidence": {
                        "color_path": image_path,
                        "bbox_xyxy": [int(round(v)) for v in box.tolist()],
                        "sam_mask_path": mask_path,
                        **evidence,
                    },
                    "_seed_indices": seed_indices,
                    "_visible_seed_indices": visible_seed_indices,
                    "_sam_mask": item["core_mask"],
                    **_existing_mask_metrics(existing_masks, seed_indices),
                }
                observations.append(observation)
                raw_observations += 1

    return observations, skipped, existing_masks, points_xyz


def export_scene_mask_graph_proposals(
    openyolo3d,
    predictor,
    scene_name,
    output_dir,
    processed_scene_path=None,
    detection_score_th=0.45,
    min_seed_points=80,
    max_box_area_ratio=0.30,
    frame_stride=5,
    max_frames=None,
    max_detections_per_frame=8,
    max_candidates=30,
    blocked_classes=None,
    ranking_policy="priority",
    sam_multimask_topk=1,
    sam_mask_selection_policy="sam_score",
    sam_mask_geometry_model=None,
    sam_mask_geometry_cc_radius=0.03,
    sam_mask_geometry_plane_threshold=0.02,
    sam_mask_geometry_max_points=50000,
    seed_depth_cluster=False,
    seed_depth_cluster_bin_size=0.10,
    seed_depth_cluster_window_bins=1,
    seed_depth_cluster_min_keep_ratio=0.25,
    seed_depth_cluster_min_removed_ratio=0.0,
    seed_depth_cluster_max_removed_ratio=1.0,
    sam_adaptive_internal_seed=False,
    sam_adaptive_internal_keep_ratio=0.70,
    sam_adaptive_internal_min_keep_ratio=0.35,
    sam_adaptive_internal_boundary_weight=0.45,
    sam_adaptive_internal_depth_weight=0.55,
    sam_adaptive_internal_depth_bin_size=0.10,
    sam_adaptive_internal_depth_window_bins=1,
    sam_mask_erode_pixels=0,
    sam_mask_erode_min_area_ratio=0.15,
    label_consensus_iou_th=0.25,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
    graph_same_class_only=True,
    graph_min_seed_iou=0.03,
    graph_min_seed_containment=0.18,
    graph_min_reference_coverage=0.20,
    graph_spatial_sigma=0.35,
    graph_view_consensus_scale=4.0,
    graph_edge_score_threshold=0.35,
    graph_min_cluster_observations=2,
    graph_keep_singletons=False,
    graph_max_views_per_cluster=4,
    graph_min_new_seed_ratio=0.05,
    graph_point_vote_allow_fallback=False,
    graph_point_vote_min_score=0.35,
    graph_point_vote_min_support=1,
    graph_point_vote_min_keep_ratio=0.35,
    graph_point_vote_min_keep_points=0,
    graph_min_selected_views=0,
    graph_min_same_object_edges=0,
    graph_min_independent_support_edges=1,
    graph_min_edge_mean_score=0.0,
    graph_min_consensus_score=0.0,
    graph_min_depth_consistency=0.0,
    graph_max_conflict_edges=None,
    graph_max_conflict_ratio=None,
    graph_core_min_largest_component_ratio=0.80,
    graph_core_max_second_component_ratio=0.10,
    graph_min_label_majority=0.65,
    graph_min_label_margin=0.20,
    graph_output_existing_support=False,
    graph_gap_min_uncovered_points=20,
    graph_gap_min_uncovered_ratio=0.60,
    graph_gap_min_largest_component_ratio=0.50,
    graph_gap_cc_radius=0.03,
    graph_gap_cc_max_points=50000,
    graph_gap_seed_policy="adaptive",
    graph_gap_reliable_existing_coverage=True,
    graph_gap_min_existing_score=0.30,
    graph_gap_min_mask_seed_coverage=0.50,
    graph_candidate_competition=True,
    graph_competition_same_class_iou=0.60,
    graph_competition_cross_class_iou=0.35,
    graph_competition_containment=0.80,
    graph_hypothesis_mode="constrained",
    graph_hypothesis_min_support_edges=1,
    graph_hypothesis_min_join_support_edges=2,
    graph_hypothesis_high_score_independent_support=0.75,
    graph_hypothesis_min_support_member_ratio=0.30,
    graph_hypothesis_seed_quality_top_ratio=0.30,
    graph_hypothesis_allow_undersegmentation_bridge=False,
    graph_cross_view_conflict_min_visible_ratio=0.45,
    graph_cross_view_conflict_max_inside_ratio=0.05,
    graph_relation_min_valid_points=30,
    graph_relation_min_valid_ratio=0.15,
    graph_relation_min_valid_floor=20,
    graph_independent_depth_consistency=0.75,
    graph_independent_inside_depth_consistency=0.60,
    graph_independent_visible_iou=0.08,
    graph_independent_visible_containment=0.45,
    graph_independent_support_score=0.65,
    graph_reference_min_seed_coverage=0.50,
    graph_reference_depth_consistency=0.70,
    graph_reference_inside_depth_consistency=0.50,
    graph_reference_visible_iou=0.03,
    graph_reference_visible_containment=0.30,
    graph_hard_conflict_inside_points=30,
    graph_hard_conflict_bidirectional_ratio=0.40,
    graph_hard_conflict_single_ratio=0.60,
    export_max_existing_iou=None,
    export_max_seed_in_existing_mask_ratio=None,
    export_code_version="",
    superpoint_diagnostics=False,
    superpoint_adjacency_knn=12,
    superpoint_adjacency_max_distance=0.05,
    superpoint_adjacency_min_contact_points=3,
    superpoint_adjacency_min_contact_ratio=0.02,
    superpoint_support_min_coverage=0.60,
    superpoint_partial_min_coverage=0.30,
    superpoint_min_visible_points=20,
    superpoint_min_depth_consistency=0.70,
    superpoint_reject_min_depth_conflict=0.60,
    superpoint_reject_min_inside_points=20,
    superpoint_reject_min_conflict_points=20,
    superpoint_outside_reject_min_visible_points=20,
    superpoint_outside_reject_max_inside_ratio=0.10,
    superpoint_outside_reject_min_outside_ratio=0.90,
):
    scene_dir = osp.join(output_dir, scene_name)
    seed_dir = osp.join(scene_dir, "seed_points")
    os.makedirs(seed_dir, exist_ok=True)

    observations, skipped, existing_masks, points_xyz = collect_scene_mask_observations(
        openyolo3d,
        predictor,
        scene_name,
        output_dir,
        detection_score_th=detection_score_th,
        min_seed_points=min_seed_points,
        max_box_area_ratio=max_box_area_ratio,
        frame_stride=frame_stride,
        max_frames=max_frames,
        max_detections_per_frame=max_detections_per_frame,
        blocked_classes=blocked_classes,
        sam_multimask_topk=sam_multimask_topk,
        sam_mask_selection_policy=sam_mask_selection_policy,
        sam_mask_geometry_model=sam_mask_geometry_model,
        sam_mask_geometry_cc_radius=sam_mask_geometry_cc_radius,
        sam_mask_geometry_plane_threshold=sam_mask_geometry_plane_threshold,
        sam_mask_geometry_max_points=sam_mask_geometry_max_points,
        seed_depth_cluster=seed_depth_cluster,
        seed_depth_cluster_bin_size=seed_depth_cluster_bin_size,
        seed_depth_cluster_window_bins=seed_depth_cluster_window_bins,
        seed_depth_cluster_min_keep_ratio=seed_depth_cluster_min_keep_ratio,
        seed_depth_cluster_min_removed_ratio=seed_depth_cluster_min_removed_ratio,
        seed_depth_cluster_max_removed_ratio=seed_depth_cluster_max_removed_ratio,
        sam_adaptive_internal_seed=sam_adaptive_internal_seed,
        sam_adaptive_internal_keep_ratio=sam_adaptive_internal_keep_ratio,
        sam_adaptive_internal_min_keep_ratio=sam_adaptive_internal_min_keep_ratio,
        sam_adaptive_internal_boundary_weight=sam_adaptive_internal_boundary_weight,
        sam_adaptive_internal_depth_weight=sam_adaptive_internal_depth_weight,
        sam_adaptive_internal_depth_bin_size=sam_adaptive_internal_depth_bin_size,
        sam_adaptive_internal_depth_window_bins=sam_adaptive_internal_depth_window_bins,
        sam_mask_erode_pixels=sam_mask_erode_pixels,
        sam_mask_erode_min_area_ratio=sam_mask_erode_min_area_ratio,
        label_consensus_iou_th=label_consensus_iou_th,
        box_nms_iou=box_nms_iou,
        box_nms_same_class_only=box_nms_same_class_only,
    )
    existing_classes = _to_numpy(openyolo3d.predicated_classes).astype(np.int64) if getattr(openyolo3d, "predicated_classes", None) is not None else None
    existing_scores = _to_numpy(openyolo3d.predicated_scores).astype(np.float32) if getattr(openyolo3d, "predicated_scores", None) is not None else None
    if existing_classes is not None and existing_classes.shape[0] != existing_masks.shape[1]:
        existing_classes = None
    if existing_scores is not None and existing_scores.shape[0] != existing_masks.shape[1]:
        existing_scores = None
    effective_reliable_existing_coverage = bool(graph_gap_reliable_existing_coverage)
    projections_np = _to_numpy(openyolo3d.mesh_projections[0]).astype(np.int64)
    visible_np = _to_numpy(openyolo3d.mesh_projections[1]).astype(bool)
    depth_relation_cache = _DepthRelationCache(
        openyolo3d,
        points_xyz,
        projections_np,
        scaling_params=getattr(openyolo3d, "scaling_params", None),
    )
    scene_cache = None
    superpoint_scene_summary = {
        "enabled": False,
        "reason": "disabled",
    }
    observation_superpoint_records = []
    observation_superpoint_paths = {
        "observation_superpoint_summary_path": None,
        "observation_superpoint_items": [],
    }
    observation_superpoint_by_id = {}
    if superpoint_diagnostics:
        if processed_scene_path is None or not osp.exists(processed_scene_path):
            superpoint_scene_summary = {
                "enabled": False,
                "reason": "missing_processed_scene",
                "processed_scene_path": processed_scene_path,
            }
        else:
            scene_cache = load_or_build_scene_superpoint_cache(
                scene_dir,
                processed_scene_path,
                points_xyz=points_xyz,
                adjacency_knn=superpoint_adjacency_knn,
                adjacency_max_distance=superpoint_adjacency_max_distance,
                adjacency_min_contact_points=superpoint_adjacency_min_contact_points,
                adjacency_min_contact_ratio=superpoint_adjacency_min_contact_ratio,
            )
            if not bool(scene_cache["summary"].get("point_order_matches_scene_points", False)):
                superpoint_scene_summary = {
                    "enabled": False,
                    "reason": "point_order_mismatch",
                    **scene_cache["summary"],
                    "cache_npz_path": scene_cache.get("cache_npz_path"),
                    "cache_summary_path": scene_cache.get("cache_summary_path"),
                }
                scene_cache = None
            else:
                frame_projector = _FrameProjector(
                    scene_cache["scene_points"],
                    openyolo3d.world2cam,
                    projections_np,
                    scaling_params=getattr(openyolo3d, "scaling_params", None),
                )
                for obs in observations:
                    evidence = classify_observation_superpoint_support(
                        scene_cache,
                        obs,
                        frame_projector,
                        strong_support_min_coverage=superpoint_support_min_coverage,
                        partial_support_min_coverage=superpoint_partial_min_coverage,
                        min_valid_visible_points=superpoint_min_visible_points,
                        strong_support_min_depth_consistency=superpoint_min_depth_consistency,
                        strong_reject_min_depth_conflict=superpoint_reject_min_depth_conflict,
                        strong_reject_min_inside_points=superpoint_reject_min_inside_points,
                        strong_reject_min_conflict_points=superpoint_reject_min_conflict_points,
                        outside_reject_min_visible_points=superpoint_outside_reject_min_visible_points,
                        outside_reject_max_inside_ratio=superpoint_outside_reject_max_inside_ratio,
                        outside_reject_min_outside_ratio=superpoint_outside_reject_min_outside_ratio,
                    )
                    observation_superpoint_records.append(evidence)
                    observation_superpoint_by_id[int(evidence["graph_observation_id"])] = evidence
                observation_superpoint_paths = save_observation_superpoint_evidence(
                    observation_superpoint_records,
                    scene_dir,
                )
                superpoint_scene_summary = {
                    "enabled": True,
                    **scene_cache["summary"],
                    "cache_npz_path": scene_cache.get("cache_npz_path"),
                    "cache_summary_path": scene_cache.get("cache_summary_path"),
                    **{
                        "observation_superpoint_summary_path": observation_superpoint_paths["observation_superpoint_summary_path"],
                        "observation_superpoint_count": int(len(observation_superpoint_records)),
                    },
                }
    else:
        scene_cache = None

    relation_edges, adjacency, conflict_edges, weak_edges = _build_mask_graph(
        observations,
        points_xyz=points_xyz,
        projections_np=projections_np,
        visible_np=visible_np,
        scaling_params=getattr(openyolo3d, "scaling_params", None),
        depth_relation_cache=depth_relation_cache,
        same_class_only=graph_same_class_only,
        min_seed_iou=graph_min_seed_iou,
        min_seed_containment=graph_min_seed_containment,
        min_reference_coverage=graph_min_reference_coverage,
        spatial_sigma=graph_spatial_sigma,
        view_consensus_scale=graph_view_consensus_scale,
        edge_score_threshold=graph_edge_score_threshold,
        cross_view_conflict_min_visible_ratio=graph_cross_view_conflict_min_visible_ratio,
        cross_view_conflict_max_inside_ratio=graph_cross_view_conflict_max_inside_ratio,
        relation_min_valid_points=graph_relation_min_valid_points,
        relation_min_valid_ratio=graph_relation_min_valid_ratio,
        relation_min_valid_floor=graph_relation_min_valid_floor,
        independent_depth_consistency=graph_independent_depth_consistency,
        independent_inside_depth_consistency=graph_independent_inside_depth_consistency,
        independent_visible_iou=graph_independent_visible_iou,
        independent_visible_containment=graph_independent_visible_containment,
        independent_support_score=graph_independent_support_score,
        reference_min_seed_coverage=graph_reference_min_seed_coverage,
        reference_depth_consistency=graph_reference_depth_consistency,
        reference_inside_depth_consistency=graph_reference_inside_depth_consistency,
        reference_visible_iou=graph_reference_visible_iou,
        reference_visible_containment=graph_reference_visible_containment,
        hard_conflict_inside_points=graph_hard_conflict_inside_points,
        hard_conflict_bidirectional_ratio=graph_hard_conflict_bidirectional_ratio,
        hard_conflict_single_ratio=graph_hard_conflict_single_ratio,
    )
    connected_components = _connected_components(len(observations), adjacency)
    hypothesis_skipped = []
    if str(graph_hypothesis_mode) == "connected_component":
        hypotheses = [
            {
                "graph_hypothesis_id": int(idx),
                "members": component,
                "trace": [],
                "formation_policy": "connected_component",
                "source_component_id": int(idx),
                "source_component_size": int(len(component)),
            }
            for idx, component in enumerate(connected_components)
        ]
    else:
        hypotheses, hypothesis_skipped = _build_constrained_instance_hypotheses(
            observations,
            adjacency,
            relation_edges,
            min_support_edges=graph_hypothesis_min_support_edges,
            min_join_support_edges=graph_hypothesis_min_join_support_edges,
            high_score_independent_support=graph_hypothesis_high_score_independent_support,
            min_support_member_ratio=graph_hypothesis_min_support_member_ratio,
            seed_quality_top_ratio=graph_hypothesis_seed_quality_top_ratio,
            allow_undersegmentation_bridge=graph_hypothesis_allow_undersegmentation_bridge,
        )
    hypothesis_partition_stats = _hypothesis_partition_stats(connected_components, hypotheses, hypothesis_skipped)
    candidates = []
    existing_support_diagnostics = []
    existing_support_diagnostic_dir = osp.join(scene_dir, "existing_support_diagnostics")
    prefilter_skipped = []
    cluster_skipped = []
    for cluster_id, hypothesis in enumerate(hypotheses):
        component = list(hypothesis["members"])
        if len(component) < int(graph_min_cluster_observations):
            if not graph_keep_singletons:
                cluster_skipped.append({"graph_cluster_id": int(cluster_id), "reason": "few_observations", "observation_count": int(len(component))})
                continue
        selected_indices, selected_seed = _select_cluster_views(
            observations,
            component,
            adjacency,
            max_views=graph_max_views_per_cluster,
            min_new_seed_ratio=graph_min_new_seed_ratio,
        )
        selected_seed, seed_vote_info = _vote_selected_seed_points(
            observations,
            selected_indices,
            selected_seed,
            min_vote_score=graph_point_vote_min_score,
            min_support_count=graph_point_vote_min_support,
            min_keep_ratio=graph_point_vote_min_keep_ratio,
            min_keep_points=graph_point_vote_min_keep_points,
            allow_fallback=graph_point_vote_allow_fallback,
        )
        if len(selected_seed) < int(min_seed_points):
            cluster_skipped.append({"graph_cluster_id": int(cluster_id), "reason": "few_cluster_seed_points", "num_seed_points": int(len(selected_seed))})
            continue
        voted_class_id, _ = _component_class_vote(observations, component)
        candidate = _cluster_to_candidate(
            len(candidates),
            observations,
            component,
            selected_indices,
            selected_seed,
            adjacency,
            relation_edges,
            existing_masks,
            points_xyz=points_xyz,
            seed_vote_info=seed_vote_info,
            gap_info=_graph_gap_metrics(
                selected_seed,
                existing_masks,
                points_xyz,
                existing_classes=existing_classes,
                existing_scores=existing_scores,
                candidate_class_id=voted_class_id,
                reliable_existing_coverage=effective_reliable_existing_coverage,
                min_existing_score=graph_gap_min_existing_score,
                min_mask_seed_coverage=graph_gap_min_mask_seed_coverage,
                min_uncovered_points=graph_gap_min_uncovered_points,
                min_uncovered_ratio=graph_gap_min_uncovered_ratio,
                min_largest_component_ratio=graph_gap_min_largest_component_ratio,
                cc_radius=graph_gap_cc_radius,
                cc_max_points=graph_gap_cc_max_points,
            ),
            graph_gap_seed_policy=graph_gap_seed_policy,
            hypothesis_trace=hypothesis.get("trace", []),
            hypothesis_formation_policy=hypothesis.get("formation_policy"),
        )
        if candidate.get("candidate_action") != "new_candidate" and not graph_output_existing_support:
            diagnostic_record = _write_existing_support_diagnostic(
                candidate,
                len(existing_support_diagnostics),
                existing_support_diagnostic_dir,
            )
            existing_support_diagnostics.append(diagnostic_record)
            prefilter_skipped.append(
                {
                    "reason": "graph_existing_support_only",
                    "graph_cluster_id": int(cluster_id),
                    "class_name": candidate.get("class_name"),
                    "candidate_action": candidate.get("candidate_action"),
                    "existing_support_diagnostic_id": int(diagnostic_record["diagnostic_id"]),
                    "gap_info": candidate.get("gap_info", {}),
                }
            )
            continue
        graph_rejection_reasons = _graph_candidate_rejection_reasons(
            selected_indices,
            candidate,
            min_selected_views=graph_min_selected_views,
            min_same_object_edges=graph_min_same_object_edges,
            min_independent_support_edges=graph_min_independent_support_edges,
            min_edge_mean_score=graph_min_edge_mean_score,
            min_consensus_score=graph_min_consensus_score,
            min_depth_consistency=graph_min_depth_consistency,
            max_conflict_edges=graph_max_conflict_edges,
            max_conflict_ratio=graph_max_conflict_ratio,
            core_min_largest_component_ratio=graph_core_min_largest_component_ratio,
            core_max_second_component_ratio=graph_core_max_second_component_ratio,
            min_label_majority=graph_min_label_majority,
            min_label_margin=graph_min_label_margin,
        )
        if graph_rejection_reasons:
            prefilter_skipped.append(
                {
                    "reason": "graph_consistency_rejected",
                    "graph_cluster_id": int(cluster_id),
                    "class_name": candidate.get("class_name"),
                    "rejection_reasons": graph_rejection_reasons,
                    "graph_consensus_score": float(candidate.get("graph_consensus_score", 0.0)),
                    "depth_consistency_score": float(candidate.get("depth_consistency_score", 0.0)),
                    "same_object_edge_count": int(candidate.get("same_object_edge_count", 0)),
                    "conflict_edge_count": int(candidate.get("conflict_edge_count", 0)),
                }
            )
            continue
        if export_max_existing_iou is not None and float(candidate.get("best_existing_iou", 0.0)) > float(export_max_existing_iou):
            prefilter_skipped.append(
                {
                    "reason": "export_matched_existing_3d_mask",
                    "graph_cluster_id": int(cluster_id),
                    "class_name": candidate.get("class_name"),
                    "best_existing_iou": float(candidate.get("best_existing_iou", 0.0)),
                }
            )
            continue
        if (
            export_max_seed_in_existing_mask_ratio is not None
            and float(candidate.get("seed_in_existing_mask_ratio", 0.0)) > float(export_max_seed_in_existing_mask_ratio)
        ):
            prefilter_skipped.append(
                {
                    "reason": "export_mostly_covered_by_existing_masks",
                    "graph_cluster_id": int(cluster_id),
                    "class_name": candidate.get("class_name"),
                    "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", 0.0)),
                }
            )
            continue
        if scene_cache is not None and observation_superpoint_by_id:
            candidate["superpoint_diagnostics"] = summarize_candidate_superpoints(
                scene_cache,
                candidate,
                observation_superpoint_by_id,
            )
        candidates.append(candidate)

    if graph_candidate_competition and len(candidates) > 1:
        candidates, competition_skipped = _apply_graph_candidate_competition(
            candidates,
            same_class_iou=graph_competition_same_class_iou,
            cross_class_iou=graph_competition_cross_class_iou,
            containment=graph_competition_containment,
        )
        prefilter_skipped.extend(competition_skipped)

    if ranking_policy == "graph_priority":
        candidates = sorted(
            candidates,
            key=lambda item: (
                -int(item.get("support_view_count", 0)),
                -float(item.get("graph_consensus_score", 0.0)),
                -float(item.get("graph_edge_mean_score", 0.0)),
                -float(item.get("proposal_priority", 0.0)),
            ),
        )
    elif ranking_policy == "novelty":
        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item.get("seed_in_existing_mask_ratio", 1.0)),
                float(item.get("best_existing_iou", 1.0)),
                -float(item.get("graph_consensus_score", 0.0)),
                -float(item.get("proposal_priority", 0.0)),
            ),
        )
    else:
        candidates = sorted(candidates, key=lambda item: -float(item.get("proposal_priority", 0.0)))

    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]

    output_candidates = []
    for candidate_id, candidate in enumerate(candidates):
        seed_indices = candidate.pop("_seed_indices")
        full_core_seed_indices = candidate.pop("_full_core_seed_indices", seed_indices)
        gap_core_seed_indices = candidate.pop("_gap_core_seed_indices", np.asarray([], dtype=np.int64))
        seed_path = osp.join(seed_dir, f"candidate{candidate_id:04d}_points.npz")
        np.savez_compressed(seed_path, point_indices=seed_indices)
        full_core_seed_path = osp.join(seed_dir, f"candidate{candidate_id:04d}_full_core_points.npz")
        gap_core_seed_path = osp.join(seed_dir, f"candidate{candidate_id:04d}_gap_core_points.npz")
        np.savez_compressed(full_core_seed_path, point_indices=np.asarray(full_core_seed_indices, dtype=np.int64))
        np.savez_compressed(gap_core_seed_path, point_indices=np.asarray(gap_core_seed_indices, dtype=np.int64))
        candidate["candidate_id"] = int(candidate_id)
        candidate["num_seed_points"] = int(len(seed_indices))
        candidate["seed_points_path"] = seed_path
        candidate["full_core_seed_points_path"] = full_core_seed_path
        candidate["gap_core_seed_points_path"] = gap_core_seed_path
        output_candidates.append(candidate)

    observation_seed_dir = osp.join(scene_dir, "observation_seed_points")
    os.makedirs(observation_seed_dir, exist_ok=True)
    trace_path = osp.join(scene_dir, "mask_graph_trace.json")
    observation_summaries = []
    for obs in observations:
        obs_id = int(obs.get("graph_observation_id", len(observation_summaries)))
        obs_seed_path = osp.join(observation_seed_dir, f"observation{obs_id:05d}_points.npz")
        obs_visible_seed_path = osp.join(observation_seed_dir, f"observation{obs_id:05d}_visible_points.npz")
        np.savez_compressed(obs_seed_path, point_indices=np.asarray(obs.get("_seed_indices", []), dtype=np.int64))
        np.savez_compressed(
            obs_visible_seed_path,
            point_indices=np.asarray(obs.get("_visible_seed_indices", []), dtype=np.int64),
        )
        observation_summaries.append(
            {
                **{
                    key: value
                    for key, value in obs.items()
                    if not key.startswith("_") and key not in {"support_views"}
                },
                "observation_seed_points_path": obs_seed_path,
                "observation_visible_seed_points_path": obs_visible_seed_path,
            }
        )
    with open(trace_path, "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "raw_observations": len(observations),
                "observations": observation_summaries,
                "relation_edges": relation_edges,
                "hypotheses": hypotheses,
                "hypothesis_skipped": hypothesis_skipped,
                "hypothesis_partition_stats": hypothesis_partition_stats,
                "existing_support_diagnostics": existing_support_diagnostics,
                "cluster_skipped": cluster_skipped,
                "prefilter_skipped": prefilter_skipped,
                "superpoint_diagnostics": superpoint_scene_summary,
                "observation_superpoint_items": observation_superpoint_paths["observation_superpoint_items"],
            },
            f,
            indent=2,
            default=str,
        )

    json_path = osp.join(scene_dir, "backprojection_candidates.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "source_kind": "mask_graph",
                "num_candidates": len(output_candidates),
                "raw_observations": len(observations),
                "graph_edges": len(relation_edges),
                "graph_support_edges": len(adjacency) and sum(len(node_edges) for node_edges in adjacency) // 2,
                "graph_weak_edges": len(weak_edges),
                "graph_conflict_edges": len(conflict_edges),
                "graph_uncertain_edges": sum(1 for edge in relation_edges if str(edge.get("relation_type")) == "uncertain"),
                "graph_components": len(connected_components),
                "graph_hypotheses": len(hypotheses),
                "graph_split_components": int(hypothesis_partition_stats["split_source_component_count"]),
                "graph_dropped_components": int(hypothesis_partition_stats["dropped_source_component_count"]),
                "graph_unassigned_observations": int(hypothesis_partition_stats["unassigned_observation_count"]),
                "hypothesis_partition_stats": hypothesis_partition_stats,
                "existing_support_diagnostic_count": len(existing_support_diagnostics),
                "superpoint_diagnostics": superpoint_scene_summary,
                "mask_graph_trace_path": trace_path,
                "hypothesis_skipped": hypothesis_skipped,
                "existing_support_diagnostics": existing_support_diagnostics,
                "skipped": skipped,
                "cluster_skipped": cluster_skipped,
                "prefilter_skipped": prefilter_skipped,
                "filters": {
                    "detection_score_th": detection_score_th,
                    "min_seed_points": min_seed_points,
                    "max_box_area_ratio": max_box_area_ratio,
                    "frame_stride": frame_stride,
                    "max_frames": max_frames,
                    "max_detections_per_frame": max_detections_per_frame,
                    "max_candidates": max_candidates,
                    "blocked_classes": sorted(_parse_class_names(blocked_classes)),
                    "ranking_policy": ranking_policy,
                    "sam_multimask_topk": sam_multimask_topk,
                    "sam_mask_selection_policy": sam_mask_selection_policy,
                    "sam_mask_geometry_cc_radius": sam_mask_geometry_cc_radius,
                    "sam_mask_geometry_plane_threshold": sam_mask_geometry_plane_threshold,
                    "sam_mask_geometry_max_points": sam_mask_geometry_max_points,
                    "seed_depth_cluster": seed_depth_cluster,
                    "sam_adaptive_internal_seed": sam_adaptive_internal_seed,
                    "sam_mask_erode_pixels": sam_mask_erode_pixels,
                    "sam_mask_erode_min_area_ratio": sam_mask_erode_min_area_ratio,
                    "label_consensus_iou_th": label_consensus_iou_th,
                    "box_nms_iou": box_nms_iou,
                    "box_nms_same_class_only": box_nms_same_class_only,
                    "graph_same_class_only": graph_same_class_only,
                    "graph_min_seed_iou": graph_min_seed_iou,
                    "graph_min_seed_containment": graph_min_seed_containment,
                    "graph_min_reference_coverage": graph_min_reference_coverage,
                    "graph_spatial_sigma": graph_spatial_sigma,
                    "graph_view_consensus_scale": graph_view_consensus_scale,
                    "graph_edge_score_threshold": graph_edge_score_threshold,
                    "graph_min_cluster_observations": graph_min_cluster_observations,
                    "graph_keep_singletons": graph_keep_singletons,
                    "graph_max_views_per_cluster": graph_max_views_per_cluster,
                    "graph_min_new_seed_ratio": graph_min_new_seed_ratio,
                    "graph_point_vote_allow_fallback": graph_point_vote_allow_fallback,
                    "graph_point_vote_min_score": graph_point_vote_min_score,
                    "graph_point_vote_min_support": graph_point_vote_min_support,
                    "graph_point_vote_min_keep_ratio": graph_point_vote_min_keep_ratio,
                    "graph_point_vote_min_keep_points": graph_point_vote_min_keep_points,
                    "graph_min_selected_views": graph_min_selected_views,
                    "graph_min_same_object_edges": graph_min_same_object_edges,
                    "graph_min_independent_support_edges": graph_min_independent_support_edges,
                    "graph_min_edge_mean_score": graph_min_edge_mean_score,
                    "graph_min_consensus_score": graph_min_consensus_score,
                    "graph_min_depth_consistency": graph_min_depth_consistency,
                    "graph_max_conflict_edges": graph_max_conflict_edges,
                    "graph_max_conflict_ratio": graph_max_conflict_ratio,
                    "graph_core_min_largest_component_ratio": graph_core_min_largest_component_ratio,
                    "graph_core_max_second_component_ratio": graph_core_max_second_component_ratio,
                    "graph_min_label_majority": graph_min_label_majority,
                    "graph_min_label_margin": graph_min_label_margin,
                    "graph_output_existing_support": graph_output_existing_support,
                    "graph_gap_min_uncovered_points": graph_gap_min_uncovered_points,
                    "graph_gap_min_uncovered_ratio": graph_gap_min_uncovered_ratio,
                    "graph_gap_min_largest_component_ratio": graph_gap_min_largest_component_ratio,
                    "graph_gap_cc_radius": graph_gap_cc_radius,
                    "graph_gap_cc_max_points": graph_gap_cc_max_points,
                    "graph_gap_seed_policy": graph_gap_seed_policy,
                    "graph_gap_reliable_existing_coverage": graph_gap_reliable_existing_coverage,
                    "graph_gap_min_existing_score": graph_gap_min_existing_score,
                    "graph_gap_min_mask_seed_coverage": graph_gap_min_mask_seed_coverage,
                    "graph_candidate_competition": graph_candidate_competition,
                    "graph_competition_same_class_iou": graph_competition_same_class_iou,
                    "graph_competition_cross_class_iou": graph_competition_cross_class_iou,
                    "graph_competition_containment": graph_competition_containment,
                    "graph_hypothesis_mode": graph_hypothesis_mode,
                    "graph_hypothesis_min_support_edges": graph_hypothesis_min_support_edges,
                    "graph_hypothesis_min_join_support_edges": graph_hypothesis_min_join_support_edges,
                    "graph_hypothesis_high_score_independent_support": graph_hypothesis_high_score_independent_support,
                    "graph_hypothesis_min_support_member_ratio": graph_hypothesis_min_support_member_ratio,
                    "graph_hypothesis_seed_quality_top_ratio": graph_hypothesis_seed_quality_top_ratio,
                    "graph_hypothesis_allow_undersegmentation_bridge": graph_hypothesis_allow_undersegmentation_bridge,
                    "graph_cross_view_conflict_min_visible_ratio": graph_cross_view_conflict_min_visible_ratio,
                    "graph_cross_view_conflict_max_inside_ratio": graph_cross_view_conflict_max_inside_ratio,
                    "graph_relation_min_valid_points": graph_relation_min_valid_points,
                    "graph_relation_min_valid_ratio": graph_relation_min_valid_ratio,
                    "graph_relation_min_valid_floor": graph_relation_min_valid_floor,
                    "graph_independent_depth_consistency": graph_independent_depth_consistency,
                    "graph_independent_inside_depth_consistency": graph_independent_inside_depth_consistency,
                    "graph_independent_visible_iou": graph_independent_visible_iou,
                    "graph_independent_visible_containment": graph_independent_visible_containment,
                    "graph_independent_support_score": graph_independent_support_score,
                    "graph_reference_min_seed_coverage": graph_reference_min_seed_coverage,
                    "graph_reference_depth_consistency": graph_reference_depth_consistency,
                    "graph_reference_inside_depth_consistency": graph_reference_inside_depth_consistency,
                    "graph_reference_visible_iou": graph_reference_visible_iou,
                    "graph_reference_visible_containment": graph_reference_visible_containment,
                    "graph_hard_conflict_inside_points": graph_hard_conflict_inside_points,
                    "graph_hard_conflict_bidirectional_ratio": graph_hard_conflict_bidirectional_ratio,
                    "graph_hard_conflict_single_ratio": graph_hard_conflict_single_ratio,
                    "export_max_existing_iou": export_max_existing_iou,
                    "export_max_seed_in_existing_mask_ratio": export_max_seed_in_existing_mask_ratio,
                    "export_code_version": export_code_version,
                    "superpoint_diagnostics": superpoint_diagnostics,
                    "superpoint_adjacency_knn": superpoint_adjacency_knn,
                    "superpoint_adjacency_max_distance": superpoint_adjacency_max_distance,
                    "superpoint_adjacency_min_contact_points": superpoint_adjacency_min_contact_points,
                    "superpoint_adjacency_min_contact_ratio": superpoint_adjacency_min_contact_ratio,
                    "superpoint_support_min_coverage": superpoint_support_min_coverage,
                    "superpoint_partial_min_coverage": superpoint_partial_min_coverage,
                    "superpoint_min_visible_points": superpoint_min_visible_points,
                    "superpoint_min_depth_consistency": superpoint_min_depth_consistency,
                    "superpoint_reject_min_depth_conflict": superpoint_reject_min_depth_conflict,
                    "superpoint_reject_min_inside_points": superpoint_reject_min_inside_points,
                    "superpoint_reject_min_conflict_points": superpoint_reject_min_conflict_points,
                    "superpoint_outside_reject_min_visible_points": superpoint_outside_reject_min_visible_points,
                    "superpoint_outside_reject_max_inside_ratio": superpoint_outside_reject_max_inside_ratio,
                    "superpoint_outside_reject_min_outside_ratio": superpoint_outside_reject_min_outside_ratio,
                },
                "graph_edge_preview": relation_edges[:200],
                "candidates": output_candidates,
            },
            f,
            indent=2,
        )
    return json_path, output_candidates, {
        "raw_observations": len(observations),
        "graph_edges": len(relation_edges),
        "graph_support_edges": len(adjacency) and sum(len(node_edges) for node_edges in adjacency) // 2,
        "graph_weak_edges": len(weak_edges),
        "graph_conflict_edges": len(conflict_edges),
        "graph_uncertain_edges": sum(1 for edge in relation_edges if str(edge.get("relation_type")) == "uncertain"),
        "graph_components": len(connected_components),
        "graph_hypotheses": len(hypotheses),
        "graph_split_components": int(hypothesis_partition_stats["split_source_component_count"]),
        "graph_dropped_components": int(hypothesis_partition_stats["dropped_source_component_count"]),
        "graph_unassigned_observations": int(hypothesis_partition_stats["unassigned_observation_count"]),
        "existing_support_diagnostic_count": len(existing_support_diagnostics),
        "num_candidates": len(output_candidates),
        "superpoint_diagnostics_enabled": bool(superpoint_scene_summary.get("enabled", False)),
        "superpoint_count": int(superpoint_scene_summary.get("superpoint_count", 0) or 0),
    }


def export_dataset_mask_graph_proposals(
    dataset_name,
    dataset_root,
    path_to_3d_masks,
    output_dir,
    sam_checkpoint,
    sam_source,
    sam_model_type="vit_b",
    scene_name=None,
    path_to_2d_preds=None,
    reuse_2d_preds=True,
    scene_list=None,
    max_scenes=None,
    **kwargs,
):
    config = load_yaml(osp.join(f"./pretrained/config_{dataset_name}.yaml"))
    dataset_env_key = f"OPENYOLO3D_DATA_ROOT_{dataset_name.upper()}"
    path_2_dataset = dataset_root or os.environ.get(dataset_env_key) or os.environ.get("OPENYOLO3D_DATA_ROOT")
    if path_2_dataset is None:
        path_2_dataset = osp.join("./data", dataset_name)
    depth_scale = config["openyolo3d"]["depth_scale"]

    if dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype = "point cloud"
    elif dataset_name == "scannet200":
        scene_names = SCENE_NAMES_SCANNET200
        datatype = "mesh"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if scene_name is not None:
        scene_names = [scene_name]
    scene_names = _resolve_scene_names(scene_names, scene_list=scene_list, max_scenes=max_scenes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _load_sam_predictor(sam_checkpoint, sam_model_type, device, sam_source)
    geometry_model_bundle = _load_geometry_discriminator(kwargs.get("sam_mask_geometry_model"))
    kwargs["sam_mask_geometry_model"] = geometry_model_bundle
    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_name}.yaml")
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    start = time.time()
    for current_scene in tqdm(scene_names):
        scene_id = current_scene.replace("scene", "")
        processed_file = osp.join(path_2_dataset, current_scene, f"{scene_id}.npy") if dataset_name == "scannet200" else None
        openyolo3d.predict(
            path_2_scene_data=osp.join(path_2_dataset, current_scene),
            depth_scale=depth_scale,
            datatype=datatype,
            processed_scene=processed_file,
            path_to_3d_masks=path_to_3d_masks,
            is_gt=False,
            path_to_2d_preds=path_to_2d_preds,
            save_2d_preds=False,
            reuse_2d_preds=reuse_2d_preds,
        )
        json_path, _, summary = export_scene_mask_graph_proposals(
            openyolo3d,
            predictor,
            current_scene,
            output_dir,
            processed_scene_path=processed_file,
            **kwargs,
        )
        summaries.append({"scene_name": current_scene, "json_path": json_path, **summary})

        for attr in (
            "world2cam",
            "mesh_projections",
            "preds_3d",
            "preds_2d",
            "predicted_masks",
            "predicated_scores",
            "predicated_classes",
        ):
            if hasattr(openyolo3d, attr):
                setattr(openyolo3d, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = osp.join(output_dir, "mask_graph_proposals_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "elapsed_seconds": time.time() - start,
                "params": kwargs,
                "scenes": summaries,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved mask-graph proposal summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--dataset_root", default=None, type=str)
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--output_dir", default="./output/mask_graph_proposals_replica")
    parser.add_argument("--sam_checkpoint", default="./pretrained/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_source", default="./_external/segment-anything/segment-anything-main")
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)

    parser.add_argument("--detection_score_th", default=0.45, type=float)
    parser.add_argument("--min_seed_points", default=80, type=int)
    parser.add_argument("--max_box_area_ratio", default=0.30, type=float)
    parser.add_argument("--frame_stride", default=5, type=int)
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--max_detections_per_frame", default=8, type=int)
    parser.add_argument("--max_candidates_per_scene", default=30, type=int)
    parser.add_argument("--blocked_classes", default="rug")
    parser.add_argument("--ranking_policy", default="priority", choices=["graph_priority", "novelty", "priority"])

    parser.add_argument("--sam_multimask_topk", default=1, type=int)
    parser.add_argument("--sam_mask_selection_policy", default="sam_score", choices=["sam_score", "geometry", "learned_geometry"])
    parser.add_argument("--sam_mask_geometry_model", default=None)
    parser.add_argument("--sam_mask_geometry_cc_radius", default=0.03, type=float)
    parser.add_argument("--sam_mask_geometry_plane_threshold", default=0.02, type=float)
    parser.add_argument("--sam_mask_geometry_max_points", default=50000, type=int)
    parser.add_argument("--seed_depth_cluster", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed_depth_cluster_bin_size", default=0.10, type=float)
    parser.add_argument("--seed_depth_cluster_window_bins", default=1, type=int)
    parser.add_argument("--seed_depth_cluster_min_keep_ratio", default=0.25, type=float)
    parser.add_argument("--seed_depth_cluster_min_removed_ratio", default=0.0, type=float)
    parser.add_argument("--seed_depth_cluster_max_removed_ratio", default=1.0, type=float)
    parser.add_argument("--sam_adaptive_internal_seed", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sam_adaptive_internal_keep_ratio", default=0.70, type=float)
    parser.add_argument("--sam_adaptive_internal_min_keep_ratio", default=0.35, type=float)
    parser.add_argument("--sam_adaptive_internal_boundary_weight", default=0.45, type=float)
    parser.add_argument("--sam_adaptive_internal_depth_weight", default=0.55, type=float)
    parser.add_argument("--sam_adaptive_internal_depth_bin_size", default=0.10, type=float)
    parser.add_argument("--sam_adaptive_internal_depth_window_bins", default=1, type=int)
    parser.add_argument("--sam_mask_erode_pixels", default=0, type=int)
    parser.add_argument("--sam_mask_erode_min_area_ratio", default=0.15, type=float)
    parser.add_argument("--label_consensus_iou_th", default=0.25, type=float)
    parser.add_argument("--box_nms_iou", default=0.0, type=float)
    parser.add_argument("--box_nms_same_class_only", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--graph_same_class_only", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_min_seed_iou", default=0.03, type=float)
    parser.add_argument("--graph_min_seed_containment", default=0.18, type=float)
    parser.add_argument("--graph_min_reference_coverage", default=0.20, type=float)
    parser.add_argument("--graph_spatial_sigma", default=0.35, type=float)
    parser.add_argument("--graph_view_consensus_scale", default=4.0, type=float)
    parser.add_argument("--graph_edge_score_threshold", default=0.35, type=float)
    parser.add_argument("--graph_min_cluster_observations", default=2, type=int)
    parser.add_argument("--graph_keep_singletons", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_max_views_per_cluster", default=4, type=int)
    parser.add_argument("--graph_min_new_seed_ratio", default=0.05, type=float)
    parser.add_argument("--graph_point_vote_allow_fallback", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_point_vote_min_score", default=0.35, type=float)
    parser.add_argument("--graph_point_vote_min_support", default=1, type=int)
    parser.add_argument("--graph_point_vote_min_keep_ratio", default=0.35, type=float)
    parser.add_argument("--graph_point_vote_min_keep_points", default=0, type=int)
    parser.add_argument("--graph_min_selected_views", default=0, type=int)
    parser.add_argument("--graph_min_same_object_edges", default=0, type=int)
    parser.add_argument("--graph_min_independent_support_edges", default=1, type=int)
    parser.add_argument("--graph_min_edge_mean_score", default=0.0, type=float)
    parser.add_argument("--graph_min_consensus_score", default=0.0, type=float)
    parser.add_argument("--graph_min_depth_consistency", default=0.0, type=float)
    parser.add_argument("--graph_max_conflict_edges", default=None, type=int)
    parser.add_argument("--graph_max_conflict_ratio", default=None, type=float)
    parser.add_argument("--graph_core_min_largest_component_ratio", default=0.80, type=float)
    parser.add_argument("--graph_core_max_second_component_ratio", default=0.10, type=float)
    parser.add_argument("--graph_min_label_majority", default=0.65, type=float)
    parser.add_argument("--graph_min_label_margin", default=0.20, type=float)
    parser.add_argument("--graph_output_existing_support", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_gap_min_uncovered_points", default=20, type=int)
    parser.add_argument("--graph_gap_min_uncovered_ratio", default=0.60, type=float)
    parser.add_argument("--graph_gap_min_largest_component_ratio", default=0.50, type=float)
    parser.add_argument("--graph_gap_cc_radius", default=0.03, type=float)
    parser.add_argument("--graph_gap_cc_max_points", default=50000, type=int)
    parser.add_argument("--graph_gap_seed_policy", default="adaptive", choices=["adaptive", "full_core", "uncovered_core"])
    parser.add_argument("--graph_gap_reliable_existing_coverage", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_gap_min_existing_score", default=0.30, type=float)
    parser.add_argument("--graph_gap_min_mask_seed_coverage", default=0.50, type=float)
    parser.add_argument("--graph_candidate_competition", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_competition_same_class_iou", default=0.60, type=float)
    parser.add_argument("--graph_competition_cross_class_iou", default=0.35, type=float)
    parser.add_argument("--graph_competition_containment", default=0.80, type=float)
    parser.add_argument("--graph_hypothesis_mode", default="constrained", choices=["constrained", "connected_component"])
    parser.add_argument("--graph_hypothesis_min_support_edges", default=1, type=int)
    parser.add_argument("--graph_hypothesis_min_join_support_edges", default=2, type=int)
    parser.add_argument("--graph_hypothesis_high_score_independent_support", default=0.75, type=float)
    parser.add_argument("--graph_hypothesis_min_support_member_ratio", default=0.30, type=float)
    parser.add_argument("--graph_hypothesis_seed_quality_top_ratio", default=0.30, type=float)
    parser.add_argument("--graph_hypothesis_allow_undersegmentation_bridge", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_cross_view_conflict_min_visible_ratio", default=0.45, type=float)
    parser.add_argument("--graph_cross_view_conflict_max_inside_ratio", default=0.05, type=float)
    parser.add_argument("--graph_relation_min_valid_points", default=30, type=int)
    parser.add_argument("--graph_relation_min_valid_ratio", default=0.15, type=float)
    parser.add_argument("--graph_relation_min_valid_floor", default=20, type=int)
    parser.add_argument("--graph_independent_depth_consistency", default=0.75, type=float)
    parser.add_argument("--graph_independent_inside_depth_consistency", default=0.60, type=float)
    parser.add_argument("--graph_independent_visible_iou", default=0.08, type=float)
    parser.add_argument("--graph_independent_visible_containment", default=0.45, type=float)
    parser.add_argument("--graph_independent_support_score", default=0.65, type=float)
    parser.add_argument("--graph_reference_min_seed_coverage", default=0.50, type=float)
    parser.add_argument("--graph_reference_depth_consistency", default=0.70, type=float)
    parser.add_argument("--graph_reference_inside_depth_consistency", default=0.50, type=float)
    parser.add_argument("--graph_reference_visible_iou", default=0.03, type=float)
    parser.add_argument("--graph_reference_visible_containment", default=0.30, type=float)
    parser.add_argument("--graph_hard_conflict_inside_points", default=30, type=int)
    parser.add_argument("--graph_hard_conflict_bidirectional_ratio", default=0.40, type=float)
    parser.add_argument("--graph_hard_conflict_single_ratio", default=0.60, type=float)
    parser.add_argument("--export_max_existing_iou", default=None, type=float)
    parser.add_argument("--export_max_seed_in_existing_mask_ratio", default=None, type=float)
    parser.add_argument("--export_code_version", default="", type=str)
    parser.add_argument("--superpoint_diagnostics", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--superpoint_adjacency_knn", default=12, type=int)
    parser.add_argument("--superpoint_adjacency_max_distance", default=0.05, type=float)
    parser.add_argument("--superpoint_adjacency_min_contact_points", default=3, type=int)
    parser.add_argument("--superpoint_adjacency_min_contact_ratio", default=0.02, type=float)
    parser.add_argument("--superpoint_support_min_coverage", default=0.60, type=float)
    parser.add_argument("--superpoint_partial_min_coverage", default=0.30, type=float)
    parser.add_argument("--superpoint_min_visible_points", default=20, type=int)
    parser.add_argument("--superpoint_min_depth_consistency", default=0.70, type=float)
    parser.add_argument("--superpoint_reject_min_depth_conflict", default=0.60, type=float)
    parser.add_argument("--superpoint_reject_min_inside_points", default=20, type=int)
    parser.add_argument("--superpoint_reject_min_conflict_points", default=20, type=int)
    parser.add_argument("--superpoint_outside_reject_min_visible_points", default=20, type=int)
    parser.add_argument("--superpoint_outside_reject_max_inside_ratio", default=0.10, type=float)
    parser.add_argument("--superpoint_outside_reject_min_outside_ratio", default=0.90, type=float)
    args = parser.parse_args()

    kwargs = vars(args).copy()
    dataset_name = kwargs.pop("dataset_name")
    dataset_root = kwargs.pop("dataset_root")
    path_to_3d_masks = kwargs.pop("path_to_3d_masks")
    output_dir = kwargs.pop("output_dir")
    sam_checkpoint = kwargs.pop("sam_checkpoint")
    sam_source = kwargs.pop("sam_source")
    sam_model_type = kwargs.pop("sam_model_type")
    scene_name = kwargs.pop("scene_name")
    path_to_2d_preds = kwargs.pop("path_to_2d_preds")
    reuse_2d_preds = kwargs.pop("reuse_2d_preds")
    scene_list = kwargs.pop("scene_list")
    max_scenes = kwargs.pop("max_scenes")
    kwargs["max_candidates"] = kwargs.pop("max_candidates_per_scene")

    export_dataset_mask_graph_proposals(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        path_to_3d_masks=path_to_3d_masks,
        output_dir=output_dir,
        sam_checkpoint=sam_checkpoint,
        sam_source=sam_source,
        sam_model_type=sam_model_type,
        scene_name=scene_name,
        path_to_2d_preds=path_to_2d_preds,
        reuse_2d_preds=reuse_2d_preds,
        scene_list=scene_list,
        max_scenes=max_scenes,
        **kwargs,
    )


if __name__ == "__main__":
    main()

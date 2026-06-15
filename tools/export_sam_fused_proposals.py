import argparse
import gc
import json
import math
import os
import os.path as osp
import pickle
import sys
import time
from collections import defaultdict

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio.v2 as imageio
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from run_evaluation import load_yaml
from utils import OpenYolo3D


def _add_sam_to_path(sam_source):
    sam_source = osp.abspath(sam_source)
    if sam_source not in sys.path:
        sys.path.insert(0, sam_source)


def _load_sam_predictor(checkpoint, model_type, device, sam_source):
    _add_sam_to_path(sam_source)
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_float(value):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(output) or math.isinf(output):
        return 0.0
    return output


def _frame_key(image_path):
    return osp.basename(image_path).split(".")[0]


def _safe_label(labels, label_id):
    if label_id < 0 or label_id >= len(labels):
        return "unknown"
    return labels[label_id]


def _parse_class_names(value):
    if value is None:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _clamp_box(box, width, height, min_size=2.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + min_size, min(width * 1.0, x2))
    y2 = max(y1 + min_size, min(height * 1.0, y2))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _prepare_image(path):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def _erode_binary_mask(mask, erode_pixels=0, min_area_ratio=0.15):
    input_area = int(mask.sum())
    info = {
        "enabled": int(erode_pixels or 0) > 0,
        "erode_pixels": int(erode_pixels or 0),
        "input_area": input_area,
        "output_area": input_area,
        "area_ratio": 1.0,
        "fallback": None,
    }
    if int(erode_pixels or 0) <= 0:
        return mask.astype(bool), info
    if input_area <= 0:
        info["fallback"] = "empty_input"
        return mask.astype(bool), info

    radius = int(erode_pixels)
    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=structure)
    output_area = int(eroded.sum())
    area_ratio = float(output_area / max(1, input_area))
    info["output_area"] = output_area
    info["area_ratio"] = area_ratio
    if output_area <= 0 or area_ratio < float(min_area_ratio):
        info["fallback"] = "small_eroded_mask"
        info["output_area"] = input_area
        info["area_ratio"] = 1.0
        return mask.astype(bool), info
    return eroded.astype(bool), info


def _sam_mask_to_indices(openyolo3d, frame_idx, sam_mask, projections_np, visible_np):
    image_height, image_width = openyolo3d.world2cam.image_resolution
    visible_indices = np.flatnonzero(visible_np[frame_idx].astype(bool))
    if len(visible_indices) == 0:
        return np.asarray([], dtype=np.int64)

    coords = projections_np[frame_idx, visible_indices].astype(np.float32)
    xs = np.round(coords[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64)
    ys = np.round(coords[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64)
    valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    if not valid.any():
        return np.asarray([], dtype=np.int64)

    visible_indices = visible_indices[valid]
    xs = xs[valid]
    ys = ys[valid]
    inside = sam_mask[ys, xs].astype(bool)
    return np.unique(visible_indices[inside]).astype(np.int64)


def _load_scene_points_xyz(openyolo3d):
    try:
        points, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
    except Exception:
        return None
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        return None
    return points[:, :3].astype(np.float32, copy=False)


def _connected_component_summary(local_points, radius, max_points):
    output = {
        "component_count": 0,
        "largest_component_ratio": 0.0,
        "small_component_ratio": 0.0,
        "component_skipped": False,
        "component_radius": float(radius),
    }
    num_points = int(len(local_points))
    if num_points <= 0 or float(radius) <= 0.0:
        return output
    if max_points is not None and num_points > int(max_points):
        output["component_skipped"] = True
        return output

    tree = cKDTree(local_points)
    effective_radius = float(radius)
    if num_points >= 2:
        nearest = tree.query(local_points, k=2)[0][:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if len(nearest) > 0:
            effective_radius = max(effective_radius, float(np.median(nearest) * 2.5))
    output["component_radius"] = effective_radius
    neighbors = tree.query_ball_point(local_points, r=effective_radius)
    parent = np.arange(num_points, dtype=np.int32)
    rank = np.zeros(num_points, dtype=np.uint8)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    for index, local_neighbors in enumerate(neighbors):
        for neighbor in local_neighbors:
            if neighbor > index:
                union(index, int(neighbor))

    roots = np.asarray([find(index) for index in range(num_points)], dtype=np.int32)
    _, counts = np.unique(roots, return_counts=True)
    largest = int(counts.max(initial=0))
    small_points = int(counts[counts < max(2, int(0.05 * num_points))].sum()) if len(counts) else 0
    output.update(
        {
            "component_count": int(len(counts)),
            "largest_component_ratio": float(largest / max(1, num_points)),
            "small_component_ratio": float(small_points / max(1, num_points)),
        }
    )
    return output


def _sam_seed_geometry_quality(
    seed_indices,
    points_xyz,
    existing_masks,
    min_seed_points,
    cc_radius=0.03,
    plane_threshold=0.02,
    max_points=50000,
):
    output = {
        "enabled": points_xyz is not None,
        "point_count": int(len(seed_indices)),
        "geometry_point_count": int(len(seed_indices)),
        "component_count": 0,
        "largest_component_ratio": 0.0,
        "small_component_ratio": 0.0,
        "geometry_component_count": 0,
        "geometry_largest_component_ratio": 0.0,
        "geometry_non_largest_component_ratio": 0.0,
        "geometry_small_component_ratio": 0.0,
        "geometry_component_skipped": False,
        "geometry_extent_x": 0.0,
        "geometry_extent_y": 0.0,
        "geometry_extent_z": 0.0,
        "extent_max": 0.0,
        "extent_mid": 0.0,
        "extent_min": 0.0,
        "geometry_extent_max": 0.0,
        "geometry_extent_mid": 0.0,
        "geometry_extent_min": 0.0,
        "geometry_aspect_max_mid": 0.0,
        "geometry_aspect_mid_min": 0.0,
        "geometry_aspect_max_min": 0.0,
        "geometry_bbox_volume": 0.0,
        "geometry_bbox_density": 0.0,
        "geometry_pca_linearity": 0.0,
        "geometry_pca_planarity": 0.0,
        "geometry_pca_scattering": 0.0,
        "aspect_mid_min": 0.0,
        "plane_inlier_ratio": 0.0,
        "plane_residual_p90": 0.0,
        "geometry_plane_inlier_ratio": 0.0,
        "geometry_plane_residual_mean": 0.0,
        "geometry_plane_residual_p90": 0.0,
        "seed_in_existing_mask_ratio": 0.0,
        "quality_score": 0.0,
    }

    if len(seed_indices) > 0 and existing_masks is not None and existing_masks.size > 0:
        seed_rows = existing_masks[seed_indices]
        output["seed_in_existing_mask_ratio"] = float(seed_rows.any(axis=1).sum() / max(1, len(seed_indices)))

    if points_xyz is None or len(seed_indices) == 0:
        return output

    local_points = points_xyz[seed_indices].astype(np.float32, copy=False)
    cc_info = _connected_component_summary(local_points, cc_radius, max_points)
    output.update(cc_info)
    output.update(
        {
            "geometry_component_count": int(cc_info.get("component_count", 0) or 0),
            "geometry_largest_component_ratio": float(cc_info.get("largest_component_ratio", 0.0) or 0.0),
            "geometry_non_largest_component_ratio": float(
                max(0.0, 1.0 - float(cc_info.get("largest_component_ratio", 0.0) or 0.0))
            ),
            "geometry_small_component_ratio": float(cc_info.get("small_component_ratio", 0.0) or 0.0),
            "geometry_component_skipped": bool(cc_info.get("component_skipped", False)),
        }
    )

    lower = local_points.min(axis=0)
    upper = local_points.max(axis=0)
    extents_xyz = np.maximum(upper - lower, 0.0)
    extents = np.sort(extents_xyz)[::-1]
    extent_max, extent_mid, extent_min = [float(value) for value in extents]
    bbox_volume = float(np.prod(np.maximum(extents_xyz, 1e-4)))
    aspect_max_mid = float(extent_max / max(extent_mid, 1e-4))
    aspect_mid_min = float(extent_mid / max(extent_min, 1e-4))
    output.update(
        {
            "geometry_extent_x": float(extents_xyz[0]),
            "geometry_extent_y": float(extents_xyz[1]),
            "geometry_extent_z": float(extents_xyz[2]),
            "extent_max": extent_max,
            "extent_mid": extent_mid,
            "extent_min": extent_min,
            "geometry_extent_max": extent_max,
            "geometry_extent_mid": extent_mid,
            "geometry_extent_min": extent_min,
            "geometry_aspect_max_mid": aspect_max_mid,
            "geometry_aspect_mid_min": aspect_mid_min,
            "geometry_aspect_max_min": float(extent_max / max(extent_min, 1e-4)),
            "geometry_bbox_volume": bbox_volume,
            "geometry_bbox_density": float(len(seed_indices) / max(bbox_volume, 1e-4)),
            "aspect_mid_min": aspect_mid_min,
        }
    )

    plane_inlier_ratio = 0.0
    plane_residual_mean = 0.0
    plane_residual_p90 = 0.0
    if len(local_points) >= 3:
        centered = local_points - local_points.mean(axis=0, keepdims=True)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            eigenvectors = eigenvectors[:, order]
            first, second, third = [float(value) for value in eigenvalues]
            denom = max(first, 1e-8)
            output.update(
                {
                    "geometry_pca_linearity": float((first - second) / denom),
                    "geometry_pca_planarity": float((second - third) / denom),
                    "geometry_pca_scattering": float(third / denom),
                }
            )
            normal = eigenvectors[:, -1]
            residuals = np.abs(centered @ normal)
            plane_inlier_ratio = float(np.mean(residuals <= float(plane_threshold)))
            plane_residual_mean = float(np.mean(residuals))
            plane_residual_p90 = float(np.percentile(residuals, 90))
        except np.linalg.LinAlgError:
            pass
    output["plane_inlier_ratio"] = plane_inlier_ratio
    output["plane_residual_p90"] = plane_residual_p90
    output["geometry_plane_inlier_ratio"] = plane_inlier_ratio
    output["geometry_plane_residual_mean"] = plane_residual_mean
    output["geometry_plane_residual_p90"] = plane_residual_p90

    point_score = min(1.0, np.log1p(len(seed_indices)) / max(np.log1p(max(int(min_seed_points) * 8, 1)), 1e-6))
    component_score = float(output["largest_component_ratio"]) if not output.get("component_skipped") else 0.5
    fragmentation_penalty = min(1.0, np.log1p(max(0, int(output["component_count"]) - 1)) / np.log(8.0))
    existing_score = 1.0 - min(1.0, float(output["seed_in_existing_mask_ratio"]))
    thin_penalty = min(1.0, np.log(max(1.0, aspect_mid_min)) / np.log(20.0))
    shape_score = 1.0 - thin_penalty
    large_flat = extent_max > 0.75 and extent_min < 0.08 and plane_inlier_ratio > 0.80
    plane_score = 0.25 if large_flat else 1.0

    quality = (
        0.30 * component_score
        + 0.20 * point_score
        + 0.20 * existing_score
        + 0.15 * shape_score
        + 0.15 * plane_score
        - 0.10 * fragmentation_penalty
        - 0.10 * float(output["small_component_ratio"])
    )
    if large_flat:
        quality -= 0.25
    output["quality_score"] = float(np.clip(quality, 0.0, 1.0))
    return output


def _filter_seed_indices_by_depth_cluster(
    openyolo3d,
    frame_idx,
    seed_indices,
    projections_np,
    bin_size=0.10,
    window_bins=1,
    min_keep_ratio=0.25,
    min_removed_ratio=0.0,
    max_removed_ratio=1.0,
    min_points=80,
):
    info = {
        "enabled": True,
        "input_points": int(len(seed_indices)),
        "output_points": int(len(seed_indices)),
        "bin_size": float(bin_size),
        "window_bins": int(window_bins),
        "min_keep_ratio": float(min_keep_ratio),
        "min_removed_ratio": float(min_removed_ratio),
        "max_removed_ratio": float(max_removed_ratio),
        "fallback": None,
    }
    if len(seed_indices) == 0:
        info["fallback"] = "empty_input"
        return seed_indices, info

    depth_path = openyolo3d.world2cam.depth_maps_paths[frame_idx]
    depth_map = imageio.imread(depth_path).astype(np.float32) / float(openyolo3d.world2cam.depth_scale)
    coords = projections_np[frame_idx, seed_indices].astype(np.int64)
    xs = coords[:, 0]
    ys = coords[:, 1]
    valid = (xs >= 0) & (xs < depth_map.shape[1]) & (ys >= 0) & (ys < depth_map.shape[0])
    if not valid.any():
        info["fallback"] = "no_valid_projected_depth"
        return seed_indices, info

    valid_indices = seed_indices[valid]
    depths = depth_map[ys[valid], xs[valid]]
    valid_depth = np.isfinite(depths) & (depths > 0)
    if not valid_depth.any():
        info["fallback"] = "no_valid_depth"
        return seed_indices, info

    valid_indices = valid_indices[valid_depth]
    depths = depths[valid_depth]
    if len(valid_indices) < int(min_points):
        info["fallback"] = "few_valid_depth_points"
        return seed_indices, info

    bin_size = max(1e-3, float(bin_size))
    min_depth = float(depths.min())
    max_depth = float(depths.max())
    if max_depth <= min_depth:
        info["fallback"] = "flat_depth"
        return seed_indices, info

    bins = np.arange(min_depth, max_depth + bin_size * 1.5, bin_size, dtype=np.float32)
    if len(bins) < 2:
        info["fallback"] = "few_depth_bins"
        return seed_indices, info
    hist, edges = np.histogram(depths, bins=bins)
    if len(hist) == 0 or int(hist.max()) <= 0:
        info["fallback"] = "empty_histogram"
        return seed_indices, info

    best_bin = int(np.argmax(hist))
    window_bins = int(max(0, window_bins))
    left_bin = max(0, best_bin - window_bins)
    right_bin = min(len(hist) - 1, best_bin + window_bins)
    low = float(edges[left_bin])
    high = float(edges[right_bin + 1])
    keep = (depths >= low) & (depths <= high)
    kept_indices = np.unique(valid_indices[keep]).astype(np.int64)
    keep_ratio = float(len(kept_indices) / max(1, len(seed_indices)))
    removed_ratio = float(1.0 - keep_ratio)
    info.update(
        {
            "output_points": int(len(kept_indices)),
            "valid_depth_points": int(len(valid_indices)),
            "keep_ratio": keep_ratio,
            "removed_ratio": removed_ratio,
            "depth_low": low,
            "depth_high": high,
            "hist_peak_points": int(hist[best_bin]),
        }
    )
    if len(kept_indices) < int(min_points) or keep_ratio < float(min_keep_ratio):
        info["fallback"] = "small_depth_cluster"
        info["output_points"] = int(len(seed_indices))
        return seed_indices, info
    if removed_ratio < float(min_removed_ratio):
        info["fallback"] = "weak_depth_cleanup"
        info["output_points"] = int(len(seed_indices))
        return seed_indices, info
    if removed_ratio > float(max_removed_ratio):
        info["fallback"] = "aggressive_depth_cleanup"
        info["output_points"] = int(len(seed_indices))
        return seed_indices, info
    return kept_indices, info


def _filter_seed_indices_by_adaptive_internal_seed(
    openyolo3d,
    frame_idx,
    seed_indices,
    sam_mask,
    projections_np,
    keep_ratio=0.70,
    min_keep_ratio=0.35,
    boundary_weight=0.45,
    depth_weight=0.55,
    depth_bin_size=0.10,
    depth_window_bins=1,
    min_points=80,
):
    info = {
        "enabled": True,
        "input_points": int(len(seed_indices)),
        "output_points": int(len(seed_indices)),
        "keep_ratio": 1.0,
        "target_keep_ratio": float(keep_ratio),
        "min_keep_ratio": float(min_keep_ratio),
        "boundary_weight": float(boundary_weight),
        "depth_weight": float(depth_weight),
        "depth_bin_size": float(depth_bin_size),
        "depth_window_bins": int(depth_window_bins),
        "fallback": None,
    }
    if len(seed_indices) == 0:
        info["fallback"] = "empty_input"
        return seed_indices, info

    mask = sam_mask.astype(bool)
    if not mask.any():
        info["fallback"] = "empty_mask"
        return seed_indices, info

    coords = projections_np[frame_idx, seed_indices].astype(np.float32)
    depth_xs = np.round(coords[:, 0]).astype(np.int64)
    depth_ys = np.round(coords[:, 1]).astype(np.int64)
    image_xs = np.round(coords[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64)
    image_ys = np.round(coords[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64)
    image_valid = (
        (image_xs >= 0)
        & (image_xs < mask.shape[1])
        & (image_ys >= 0)
        & (image_ys < mask.shape[0])
        & mask[image_ys.clip(0, mask.shape[0] - 1), image_xs.clip(0, mask.shape[1] - 1)]
    )
    if not image_valid.any():
        info["fallback"] = "no_valid_mask_projection"
        return seed_indices, info

    valid_indices = seed_indices[image_valid]
    image_xs = image_xs[image_valid]
    image_ys = image_ys[image_valid]
    depth_xs = depth_xs[image_valid]
    depth_ys = depth_ys[image_valid]

    distance_map = ndimage.distance_transform_edt(mask).astype(np.float32)
    mask_distances = distance_map[mask]
    distance_scale = float(np.percentile(mask_distances, 90)) if len(mask_distances) else 0.0
    if distance_scale <= 1e-6:
        boundary_score = np.ones((len(valid_indices),), dtype=np.float32)
    else:
        boundary_score = np.clip(distance_map[image_ys, image_xs] / distance_scale, 0.0, 1.0).astype(np.float32)

    depth_score = np.ones((len(valid_indices),), dtype=np.float32)
    depth_path = openyolo3d.world2cam.depth_maps_paths[frame_idx]
    depth_map = imageio.imread(depth_path).astype(np.float32) / float(openyolo3d.world2cam.depth_scale)
    depth_valid = (
        (depth_xs >= 0)
        & (depth_xs < depth_map.shape[1])
        & (depth_ys >= 0)
        & (depth_ys < depth_map.shape[0])
    )
    depth_values = np.zeros((len(valid_indices),), dtype=np.float32)
    depth_values[depth_valid] = depth_map[depth_ys[depth_valid], depth_xs[depth_valid]]
    depth_valid &= np.isfinite(depth_values) & (depth_values > 0)
    info["valid_depth_points"] = int(depth_valid.sum())
    if depth_valid.sum() >= int(min_points):
        depths = depth_values[depth_valid]
        bin_size = max(1e-3, float(depth_bin_size))
        min_depth = float(depths.min())
        max_depth = float(depths.max())
        if max_depth > min_depth:
            bins = np.arange(min_depth, max_depth + bin_size * 1.5, bin_size, dtype=np.float32)
            hist, edges = np.histogram(depths, bins=bins)
            if len(hist) > 0 and int(hist.max()) > 0:
                best_bin = int(np.argmax(hist))
                window_bins = int(max(0, depth_window_bins))
                left_bin = max(0, best_bin - window_bins)
                right_bin = min(len(hist) - 1, best_bin + window_bins)
                low = float(edges[left_bin])
                high = float(edges[right_bin + 1])
                distance_to_layer = np.zeros_like(depth_values, dtype=np.float32)
                distance_to_layer[depth_values < low] = low - depth_values[depth_values < low]
                distance_to_layer[depth_values > high] = depth_values[depth_values > high] - high
                depth_score = np.exp(-distance_to_layer / bin_size).astype(np.float32)
                depth_score[~depth_valid] = 0.5
                info.update(
                    {
                        "depth_low": low,
                        "depth_high": high,
                        "depth_main_layer_ratio": float(hist[best_bin] / max(1, len(depths))),
                    }
                )
    else:
        info["fallback_depth"] = "few_valid_depth_points"

    boundary_weight = max(0.0, float(boundary_weight))
    depth_weight = max(0.0, float(depth_weight))
    weight_sum = max(1e-6, boundary_weight + depth_weight)
    confidence = (boundary_weight * boundary_score + depth_weight * depth_score) / weight_sum
    target_count = int(math.ceil(float(keep_ratio) * len(valid_indices)))
    target_count = max(int(min_points), target_count)
    target_count = min(len(valid_indices), target_count)
    if target_count >= len(seed_indices):
        info["fallback"] = "no_reduction"
        return seed_indices, info
    order = np.argsort(-confidence, kind="mergesort")
    kept_indices = np.unique(valid_indices[order[:target_count]]).astype(np.int64)
    actual_keep_ratio = float(len(kept_indices) / max(1, len(seed_indices)))
    info.update(
        {
            "output_points": int(len(kept_indices)),
            "keep_ratio": actual_keep_ratio,
            "confidence_mean": float(np.mean(confidence)) if len(confidence) else 0.0,
            "confidence_p25": float(np.percentile(confidence, 25)) if len(confidence) else 0.0,
            "confidence_p75": float(np.percentile(confidence, 75)) if len(confidence) else 0.0,
            "boundary_score_mean": float(np.mean(boundary_score)) if len(boundary_score) else 0.0,
            "depth_score_mean": float(np.mean(depth_score)) if len(depth_score) else 0.0,
        }
    )
    if len(kept_indices) < int(min_points) or actual_keep_ratio < float(min_keep_ratio):
        info["fallback"] = "small_adaptive_seed"
        info["output_points"] = int(len(seed_indices))
        info["keep_ratio"] = 1.0
        return seed_indices, info
    return kept_indices, info


def _seed_iou(left, right):
    if len(left) == 0 or len(right) == 0:
        return 0.0
    intersection = np.intersect1d(left, right, assume_unique=False).size
    union = len(left) + len(right) - intersection
    return float(intersection / max(1, union))


def _box_iou_np(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + boxes_area - inter, 1.0)


def _select_2d_nms_indices(boxes, scores, class_ids, iou_threshold=0.0, same_class_only=True):
    if iou_threshold is None or float(iou_threshold) <= 0.0 or len(boxes) == 0:
        return np.arange(len(boxes), dtype=np.int64)
    order = np.argsort(-scores)
    selected = []
    for det_id in order:
        det_id = int(det_id)
        suppress = False
        for kept_id in selected:
            if same_class_only and int(class_ids[det_id]) != int(class_ids[kept_id]):
                continue
            iou = float(_box_iou_np(boxes[det_id], boxes[np.asarray([kept_id], dtype=np.int64)])[0])
            if iou >= float(iou_threshold):
                suppress = True
                break
        if not suppress:
            selected.append(det_id)
    return np.asarray(selected, dtype=np.int64)


def _label_consensus_metrics(box, class_id, boxes, class_ids, scores, iou_threshold):
    ious = _box_iou_np(box, boxes)
    matched = ious >= float(iou_threshold)
    if not matched.any():
        return {
            "label_consensus_score": 1.0,
            "label_conflict_score": 0.0,
            "label_margin": 0.0,
            "label_entropy": 0.0,
            "label_consensus_view_count": 0,
            "label_conflict_view_count": 0,
            "label_evidence_view_count": 0,
            "label_target_evidence": 0.0,
            "label_total_evidence": 0.0,
            "top_conflicting_class_id": None,
            "top_conflicting_evidence": 0.0,
        }

    evidence_by_label = defaultdict(float)
    evidence = ious[matched] * scores[matched]
    labels = class_ids[matched]
    for label, value in zip(labels, evidence):
        evidence_by_label[int(label)] += float(value)
    total_evidence = float(sum(evidence_by_label.values()))
    target_evidence = float(evidence_by_label.get(int(class_id), 0.0))
    top_conflicting_class_id = None
    top_conflicting_evidence = 0.0
    for label, value in evidence_by_label.items():
        if int(label) == int(class_id):
            continue
        if float(value) > top_conflicting_evidence:
            top_conflicting_class_id = int(label)
            top_conflicting_evidence = float(value)

    probabilities = np.asarray(list(evidence_by_label.values()), dtype=np.float64) / max(total_evidence, 1e-12)
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    entropy /= max(np.log(max(2, len(probabilities))), 1e-12)
    top_idx = int(np.argmax(evidence))
    top_is_target = int(labels[top_idx]) == int(class_id)
    consensus_score = float(target_evidence / max(total_evidence, 1e-12))
    return {
        "label_consensus_score": consensus_score,
        "label_conflict_score": float(max(0.0, 1.0 - consensus_score)),
        "label_margin": float((target_evidence - top_conflicting_evidence) / max(total_evidence, 1e-12)),
        "label_entropy": entropy,
        "label_consensus_view_count": int(top_is_target),
        "label_conflict_view_count": int(not top_is_target),
        "label_evidence_view_count": 1,
        "label_target_evidence": target_evidence,
        "label_total_evidence": total_evidence,
        "top_conflicting_class_id": top_conflicting_class_id,
        "top_conflicting_evidence": top_conflicting_evidence,
    }


def _merge_label_consensus(target, source):
    target_evidence = float(target.get("label_target_evidence", 0.0)) + float(
        source.get("label_target_evidence", 0.0)
    )
    total_evidence = float(target.get("label_total_evidence", 0.0)) + float(
        source.get("label_total_evidence", 0.0)
    )
    if float(source.get("top_conflicting_evidence", 0.0)) > float(target.get("top_conflicting_evidence", 0.0)):
        target["top_conflicting_class_id"] = source.get("top_conflicting_class_id")
        target["top_conflicting_evidence"] = float(source.get("top_conflicting_evidence", 0.0))
    target["label_target_evidence"] = target_evidence
    target["label_total_evidence"] = total_evidence
    target["label_consensus_view_count"] = int(target.get("label_consensus_view_count", 0)) + int(
        source.get("label_consensus_view_count", 0)
    )
    target["label_conflict_view_count"] = int(target.get("label_conflict_view_count", 0)) + int(
        source.get("label_conflict_view_count", 0)
    )
    target["label_evidence_view_count"] = int(target.get("label_evidence_view_count", 0)) + int(
        source.get("label_evidence_view_count", 0)
    )
    if total_evidence > 0.0:
        target["label_consensus_score"] = float(target_evidence / total_evidence)
        target["label_conflict_score"] = float(max(0.0, 1.0 - target["label_consensus_score"]))
        target["label_margin"] = float(
            (target_evidence - float(target.get("top_conflicting_evidence", 0.0))) / total_evidence
        )
    return target


def _resolve_scene_names(scene_names, scene_list=None, max_scenes=None):
    if scene_list is not None:
        raw = str(scene_list).strip()
        if osp.exists(raw):
            with open(raw) as f:
                requested = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        else:
            requested = [item.strip() for item in raw.split(",") if item.strip()]
        allowed = set(scene_names)
        scene_names = [scene for scene in requested if scene in allowed]
    if max_scenes is not None:
        scene_names = scene_names[: int(max_scenes)]
    return scene_names


def _existing_mask_metrics(existing_masks, seed_indices):
    seed_count = int(len(seed_indices))
    if seed_count == 0 or existing_masks.size == 0:
        return {
            "seed_in_existing_mask_ratio": 0.0,
            "best_existing_mask_id": None,
            "best_existing_seed_coverage": 0.0,
            "best_existing_iou": 0.0,
        }

    seed_rows = existing_masks[seed_indices]
    seed_in_existing = seed_rows.any(axis=1)
    intersections = seed_rows.sum(axis=0).astype(np.float64)
    mask_sizes = existing_masks.sum(axis=0).astype(np.float64)
    unions = mask_sizes + seed_count - intersections
    ious = np.divide(intersections, np.maximum(unions, 1.0))
    coverages = intersections / max(1, seed_count)

    best_iou_id = int(np.argmax(ious)) if len(ious) > 0 else None
    best_cov_id = int(np.argmax(coverages)) if len(coverages) > 0 else None
    best_id = best_iou_id if best_iou_id is not None else best_cov_id
    return {
        "seed_in_existing_mask_ratio": float(seed_in_existing.sum() / max(1, seed_count)),
        "best_existing_mask_id": best_id,
        "best_existing_seed_coverage": float(coverages[best_cov_id]) if best_cov_id is not None else 0.0,
        "best_existing_iou": float(ious[best_iou_id]) if best_iou_id is not None else 0.0,
    }


def _load_geometry_discriminator(model_path):
    if model_path is None:
        return None
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid geometry discriminator bundle: {model_path}")
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Geometry discriminator model must provide predict_proba().")
    bundle["_model_path"] = model_path
    return bundle


def _vectorize_geometry_discriminator_row(row, bundle):
    values = []
    for name in bundle.get("numeric_features", ()):
        values.append(_safe_float(row.get(name)))
    categorical_values = bundle.get("categorical_values", {})
    for name in bundle.get("categorical_features", ()):
        value = str(row.get(name, ""))
        for category in categorical_values.get(name, ()):
            values.append(1.0 if value == str(category) else 0.0)
    return np.asarray([values], dtype=np.float32)


def _predict_geometry_discriminator_score(row, bundle):
    if bundle is None:
        return 0.0
    matrix = _vectorize_geometry_discriminator_row(row, bundle)
    scores = bundle["model"].predict_proba(matrix)
    if scores.ndim != 2 or scores.shape[1] < 2:
        return 0.0
    return float(np.clip(scores[0, 1], 0.0, 1.0))


def _sam_mask_discriminator_row(
    geometry_info,
    existing_metrics,
    detection_score,
    sam_score,
    box_area_ratio,
    num_seed_points,
    num_mask_points,
):
    row = {
        "source_kind": "sam_fused",
        "superpoint_refined": False,
        "report_mask_support_enabled": False,
        "report_mask_support_mode": "none",
        "best_existing_iou": float(existing_metrics.get("best_existing_iou", 0.0) or 0.0),
        "seed_in_existing_mask_ratio": float(
            existing_metrics.get(
                "seed_in_existing_mask_ratio",
                geometry_info.get("seed_in_existing_mask_ratio", 0.0),
            )
            or 0.0
        ),
        "score": float(detection_score),
        "fusion_score": float(detection_score) * max(0.1, float(sam_score)),
        "quality_score": float(geometry_info.get("quality_score", 0.0) or 0.0),
        "support_view_count": 1,
        "support_mean_iou": 1.0,
        "support_best_iou": 1.0,
        "consistency_rate": 1.0,
        "box_area_ratio": float(box_area_ratio),
        "num_seed_points": int(num_seed_points),
        "num_mask_points": int(num_mask_points),
        "superpoint_expansion_ratio": 1.0,
        "report_mask_support_mean_positive_ratio": 0.0,
        "report_mask_support_mean_negative_ratio": 0.0,
        "report_mask_support_filtered_segments": 0,
        "report_mask_support_filtered_ratio": 0.0,
        "report_mask_support_usable_view_count": 0,
        "report_cc_component_count": 0,
        "report_cc_largest_component_ratio": 0.0,
        "report_cc_keep_ratio": 0.0,
        "scene_source_quality_z": 0.0,
        "class_source_quality_z": 0.0,
    }
    row.update(geometry_info)
    row.update(
        {
            "seed_in_existing_mask_ratio": float(
                existing_metrics.get(
                    "seed_in_existing_mask_ratio",
                    geometry_info.get("seed_in_existing_mask_ratio", 0.0),
                )
                or 0.0
            ),
            "best_existing_iou": float(existing_metrics.get("best_existing_iou", 0.0) or 0.0),
        }
    )
    return row


def _sam_view_quality_score(
    geometry_info,
    detection_score,
    sam_score,
    box_area_ratio,
    num_seed_points,
    min_seed_points,
):
    geometry_quality = _safe_float(geometry_info.get("quality_score", 0.0))
    existing_penalty = 1.0 - min(1.0, max(0.0, _safe_float(geometry_info.get("seed_in_existing_mask_ratio", 0.0))))
    point_score = min(
        1.0,
        math.log1p(max(0, int(num_seed_points)))
        / max(math.log1p(max(int(min_seed_points) * 8, 1)), 1e-6),
    )
    box_score = 1.0 - min(1.0, max(0.0, float(box_area_ratio)) / 0.30)
    score = (
        0.35 * geometry_quality
        + 0.20 * max(0.0, min(1.0, float(sam_score)))
        + 0.15 * max(0.0, min(1.0, float(detection_score)))
        + 0.15 * point_score
        + 0.10 * existing_penalty
        + 0.05 * box_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _proposal_novelty_score(proposal):
    seed_covered = float(proposal.get("seed_in_existing_mask_ratio", 0.0))
    existing_iou = float(proposal.get("best_existing_iou", 0.0))
    support_views = float(proposal.get("support_view_count", 0.0))
    priority = float(proposal.get("proposal_priority", proposal.get("score", 0.0)))
    novelty = (1.0 - min(1.0, seed_covered)) * (1.0 - min(1.0, existing_iou))
    support_term = 0.5 + 0.5 * min(1.0, support_views / 10.0)
    return float(max(0.0, priority) * novelty * support_term)


def _sort_proposals(proposals, ranking_policy):
    if ranking_policy == "novelty":
        return sorted(
            proposals,
            key=lambda item: (
                float(item.get("seed_in_existing_mask_ratio", 1.0)),
                float(item.get("best_existing_iou", 1.0)),
                -int(item["support_view_count"]),
                -float(item["proposal_priority"]),
                -float(item["score"]),
            ),
        )
    if ranking_policy == "balanced_novelty":
        return sorted(
            proposals,
            key=lambda item: (
                -_proposal_novelty_score(item),
                -int(item["support_view_count"]),
                -float(item["proposal_priority"]),
                -float(item["score"]),
            ),
        )
    return sorted(
        proposals,
        key=lambda item: (
            -int(item["support_view_count"]),
            -float(item["proposal_priority"]),
            -float(item["score"]),
        ),
    )


def _save_overlay(image, sam_mask, box, output_prefix):
    overlay = image.copy()
    red = np.asarray([255, 0, 0], dtype=np.float32)
    current = overlay[sam_mask].astype(np.float32)
    if len(current) > 0:
        overlay[sam_mask] = (0.35 * current + 0.65 * red).astype(np.uint8)

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(image.shape[1] - 1, x1))
    y1 = max(0, min(image.shape[0] - 1, y1))
    x2 = max(x1 + 1, min(image.shape[1], x2))
    y2 = max(y1 + 1, min(image.shape[0], y2))
    overlay[max(0, y1 - 2):min(image.shape[0], y1 + 3), x1:x2] = [0, 255, 0]
    overlay[max(0, y2 - 3):min(image.shape[0], y2 + 2), x1:x2] = [0, 255, 0]
    overlay[y1:y2, max(0, x1 - 2):min(image.shape[1], x1 + 3)] = [0, 255, 0]
    overlay[y1:y2, max(0, x2 - 3):min(image.shape[1], x2 + 2)] = [0, 255, 0]

    context_path = f"{output_prefix}_context.jpg"
    overlay_path = f"{output_prefix}_overlay.jpg"
    crop_path = f"{output_prefix}_crop.jpg"
    imageio.imwrite(context_path, overlay)
    imageio.imwrite(overlay_path, overlay[y1:y2, x1:x2])
    imageio.imwrite(crop_path, image[y1:y2, x1:x2])
    return {
        "context_path": context_path,
        "overlay_path": overlay_path,
        "crop_path": crop_path,
    }


def _save_mask(mask, output_prefix):
    mask_path = f"{output_prefix}_mask.png"
    imageio.imwrite(mask_path, (mask.astype(np.uint8) * 255))
    return mask_path


def _seed_observation_record(observation):
    return {
        "seed_indices": observation["_seed_indices"].copy(),
        "proposal_priority": float(observation.get("proposal_priority", 0.0)),
        "fusion_score": float(observation.get("fusion_score", observation.get("score", 0.0))),
        "score": float(observation.get("score", 0.0)),
        "view_quality_score": float(observation.get("view_quality_score", 0.0)),
        "sam_score": float((observation.get("support_views") or [{}])[0].get("sam_score", 0.0)),
        "frame_index": observation.get("frame_index"),
        "frame_id": observation.get("frame_id"),
        "detection_id": observation.get("detection_id"),
        "sam_mask_id": observation.get("sam_mask_id"),
    }


def _filter_seed_observations_by_view_quality(
    seed_observations,
    enabled=False,
    relative_threshold=0.80,
    min_score=0.0,
    min_keep_ratio=0.50,
):
    info = {
        "enabled": bool(enabled),
        "input_view_count": int(len(seed_observations)),
        "kept_view_count": int(len(seed_observations)),
        "relative_threshold": float(relative_threshold),
        "min_score": float(min_score),
        "min_keep_ratio": float(min_keep_ratio),
        "best_score": 0.0,
        "effective_threshold": 0.0,
    }
    if not enabled or len(seed_observations) <= 1:
        return list(seed_observations), info

    ranked = sorted(
        seed_observations,
        key=lambda item: (
            -float(item.get("view_quality_score", 0.0)),
            -float(item.get("proposal_priority", 0.0)),
            -len(item.get("seed_indices", ())),
        ),
    )
    best_score = float(ranked[0].get("view_quality_score", 0.0))
    threshold = max(float(min_score), best_score * float(relative_threshold))
    min_keep = max(1, int(math.ceil(len(ranked) * float(min_keep_ratio))))
    selected = [item for item in ranked if float(item.get("view_quality_score", 0.0)) >= threshold]
    if len(selected) < min_keep:
        selected = ranked[:min_keep]
    info.update(
        {
            "kept_view_count": int(len(selected)),
            "best_score": best_score,
            "effective_threshold": threshold,
        }
    )
    return selected, info


def _select_seed_indices_from_observations(
    seed_observations,
    seed_merge_policy,
    seed_merge_topk,
    seed_view_quality_gate=False,
    seed_view_quality_relative_threshold=0.80,
    seed_view_quality_min_score=0.0,
    seed_view_quality_min_keep_ratio=0.50,
):
    if not seed_observations:
        return np.asarray([], dtype=np.int64), [], {"enabled": bool(seed_view_quality_gate), "input_view_count": 0}

    seed_observations, quality_gate_info = _filter_seed_observations_by_view_quality(
        seed_observations,
        enabled=seed_view_quality_gate,
        relative_threshold=seed_view_quality_relative_threshold,
        min_score=seed_view_quality_min_score,
        min_keep_ratio=seed_view_quality_min_keep_ratio,
    )
    policy = str(seed_merge_policy or "union").lower()
    if policy == "union":
        selected = seed_observations
    else:
        topk = 1 if policy == "best_view" else int(max(1, seed_merge_topk))
        selected = sorted(
            seed_observations,
            key=lambda item: (
                -float(item.get("proposal_priority", 0.0)),
                -float(item.get("fusion_score", 0.0)),
                -float(item.get("score", 0.0)),
                -len(item.get("seed_indices", ())),
            ),
        )[:topk]

    merged = np.asarray([], dtype=np.int64)
    for item in selected:
        merged = np.union1d(merged, item["seed_indices"])
    return merged.astype(np.int64), selected, quality_gate_info


def _merge_observation(
    proposals,
    observation,
    merge_iou,
    keep_sam_mask_alternatives=False,
    seed_merge_policy="union",
    seed_merge_topk=1,
    seed_view_quality_gate=False,
    seed_view_quality_relative_threshold=0.80,
    seed_view_quality_min_score=0.0,
    seed_view_quality_min_keep_ratio=0.50,
):
    best_idx = None
    best_iou = 0.0
    for idx, proposal in enumerate(proposals):
        if proposal["class_id"] != observation["class_id"]:
            continue
        same_detection = (
            proposal.get("frame_index") == observation.get("frame_index")
            and proposal.get("detection_id") == observation.get("detection_id")
        )
        different_sam_mask = proposal.get("sam_mask_id") != observation.get("sam_mask_id")
        if keep_sam_mask_alternatives and same_detection and different_sam_mask:
            continue
        iou = _seed_iou(proposal["_seed_indices"], observation["_seed_indices"])
        if iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_idx is not None and best_iou >= merge_iou:
        proposal = proposals[best_idx]
        seed_observations = proposal.setdefault(
            "_seed_observations",
            [_seed_observation_record({"_seed_indices": proposal["_seed_indices"], **proposal})],
        )
        seed_observations.append(_seed_observation_record(observation))
        selected_seed_indices, selected_seed_observations, quality_gate_info = _select_seed_indices_from_observations(
            seed_observations,
            seed_merge_policy,
            seed_merge_topk,
            seed_view_quality_gate=seed_view_quality_gate,
            seed_view_quality_relative_threshold=seed_view_quality_relative_threshold,
            seed_view_quality_min_score=seed_view_quality_min_score,
            seed_view_quality_min_keep_ratio=seed_view_quality_min_keep_ratio,
        )
        proposal["_seed_indices"] = selected_seed_indices
        proposal["seed_merge_policy"] = str(seed_merge_policy or "union")
        proposal["seed_view_quality_gate"] = quality_gate_info
        proposal["selected_seed_view_count"] = int(len(selected_seed_observations))
        proposal["available_seed_view_count"] = int(len(seed_observations))
        proposal["score"] = max(float(proposal["score"]), float(observation["score"]))
        proposal["fusion_score"] = float(
            max(proposal.get("fusion_score", proposal["score"]), observation.get("fusion_score", observation["score"]))
        )
        proposal["support_view_count"] += 1
        proposal["support_views"].append(observation["support_views"][0])
        _merge_label_consensus(proposal, observation)
        proposal["proposal_priority"] = max(
            float(proposal.get("proposal_priority", 0.0)),
            float(observation.get("proposal_priority", 0.0)),
        )
        proposal["merged_observations"] += 1
        return

    observation["_seed_observations"] = [_seed_observation_record(observation)]
    observation["seed_merge_policy"] = str(seed_merge_policy or "union")
    observation["seed_view_quality_gate"] = {
        "enabled": bool(seed_view_quality_gate),
        "input_view_count": 1,
        "kept_view_count": 1,
    }
    observation["selected_seed_view_count"] = 1
    observation["available_seed_view_count"] = 1
    proposals.append(observation)


def export_scene_sam_fused_proposals(
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
    merge_iou=0.15,
    max_candidates=30,
    blocked_classes=None,
    ranking_policy="support_priority",
    sam_multimask_topk=1,
    sam_mask_selection_policy="sam_score",
    sam_mask_geometry_model=None,
    sam_mask_geometry_cc_radius=0.03,
    sam_mask_geometry_plane_threshold=0.02,
    sam_mask_geometry_max_points=50000,
    keep_sam_mask_alternatives=False,
    seed_merge_policy="union",
    seed_merge_topk=1,
    seed_view_quality_gate=False,
    seed_view_quality_relative_threshold=0.80,
    seed_view_quality_min_score=0.0,
    seed_view_quality_min_keep_ratio=0.50,
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
    export_max_existing_iou=None,
    export_max_seed_in_existing_mask_ratio=None,
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
    image_dir = osp.join(scene_dir, "sam_fused_images")
    mask_dir = osp.join(scene_dir, "sam_fused_masks")
    seed_dir = osp.join(scene_dir, "seed_points")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(seed_dir, exist_ok=True)

    image_height, image_width = openyolo3d.world2cam.image_resolution
    frame_indices = list(range(0, len(openyolo3d.world2cam.color_paths), max(1, frame_stride)))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    proposals = []
    raw_observations = 0
    skipped = []
    for frame_idx in frame_indices:
        image_path = openyolo3d.world2cam.color_paths[frame_idx]
        frame_id = _frame_key(image_path)
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
            box = _clamp_box(boxes[det_id], image_width, image_height)
            box_area_ratio = float(((box[2] - box[0]) * (box[3] - box[1])) / max(1, image_width * image_height))
            if box_area_ratio > max_box_area_ratio:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "large_2d_box"})
                continue

            masks, sam_scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            sam_score_order = np.argsort(-sam_scores)
            sam_score_rank_by_id = {int(mask_id): int(rank) for rank, mask_id in enumerate(sam_score_order)}
            if str(sam_mask_selection_policy) in {"geometry", "learned_geometry"}:
                mask_order = sam_score_order
            else:
                mask_order = sam_score_order[: max(1, int(sam_multimask_topk))]

            mask_items = []
            accepted_for_detection = 0
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
                    learned_geometry_score = _predict_geometry_discriminator_score(
                        feature_row,
                        geometry_model_bundle,
                    )
                    selection_score = float(learned_geometry_score)
                else:
                    selection_score = float(
                        0.45 * max(0.0, min(1.0, sam_score))
                        + 0.55 * float(geometry_info.get("quality_score", 0.0))
                    )
                if len(seed_indices) < min_seed_points:
                    skipped.append(
                        {
                            "frame_id": frame_id,
                            "detection_id": int(det_id),
                            "sam_mask_id": mask_id,
                            "sam_mask_rank": int(mask_rank),
                            "sam_score_rank": int(sam_score_rank_by_id.get(mask_id, mask_rank)),
                            "reason": "few_seed_points",
                            "num_seed_points": int(len(seed_indices)),
                            "sam_mask_erosion": erosion_info,
                            "sam_adaptive_internal_seed": adaptive_internal_info,
                            "seed_depth_cluster": depth_cluster_info,
                            "sam_mask_geometry": geometry_info,
                            "sam_mask_learned_geometry_score": learned_geometry_score,
                        }
                    )
                    continue
                mask_items.append(
                    {
                        "mask_id": mask_id,
                        "sam_score": sam_score,
                        "sam_mask": sam_mask,
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

            for item in mask_items:
                mask_id = int(item["mask_id"])
                sam_score = float(item["sam_score"])
                core_mask = item["core_mask"]
                erosion_info = item["erosion_info"]
                seed_indices = item["seed_indices"]
                adaptive_internal_info = item["adaptive_internal_info"]
                depth_cluster_info = item["depth_cluster_info"]
                geometry_info = item["geometry_info"]
                selection_score = float(item["selection_score"])
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
                evidence = _save_overlay(image, core_mask, box, evidence_prefix)
                mask_path = _save_mask(core_mask, mask_prefix)
                observation = {
                    "scene_name": scene_name,
                    "frame_id": frame_id,
                    "frame_index": int(frame_idx),
                    "detection_id": int(det_id),
                    "sam_mask_id": mask_id,
                    "sam_mask_rank": int(item["sam_mask_rank"]),
                    "sam_score_rank": int(item["sam_score_rank"]),
                    "sam_mask_selection_policy": str(sam_mask_selection_policy),
                    "sam_mask_selection_score": selection_score,
                    "sam_mask_learned_geometry_score": learned_geometry_score,
                    "sam_mask_geometry": geometry_info,
                    "sam_mask_erosion": erosion_info,
                    "sam_adaptive_internal_seed": adaptive_internal_info,
                    "seed_depth_cluster": depth_cluster_info,
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "box_area_ratio": box_area_ratio,
                    "num_seed_points": int(len(seed_indices)),
                    "proposal_priority": float(
                        score * max(0.1, sam_score) * np.log1p(len(seed_indices)) * priority_factor
                    ),
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
                            "visible_seed_points": int(len(seed_indices)),
                            "iou": 1.0,
                            "score": score,
                            "sam_score": sam_score,
                            "sam_mask_id": mask_id,
                            "sam_mask_rank": int(item["sam_mask_rank"]),
                            "sam_score_rank": int(item["sam_score_rank"]),
                            "bbox_xyxy": [float(v) for v in box.tolist()],
                            "sam_mask_path": mask_path,
                            "sam_mask_selection_policy": str(sam_mask_selection_policy),
                            "sam_mask_selection_score": selection_score,
                            "sam_mask_learned_geometry_score": learned_geometry_score,
                            "sam_mask_geometry_quality_score": float(geometry_info.get("quality_score", 0.0)),
                            "view_quality_score": view_quality_score,
                            "sam_mask_erode_pixels": int(sam_mask_erode_pixels or 0),
                            "sam_mask_core_area_ratio": float(erosion_info.get("area_ratio", 1.0)),
                            "sam_adaptive_internal_seed_keep_ratio": float(
                                adaptive_internal_info.get("keep_ratio", 1.0)
                            ),
                            "seed_depth_cluster_keep_ratio": float(depth_cluster_info.get("keep_ratio", 1.0)),
                        }
                    ],
                    **_label_consensus_metrics(box, class_id, boxes, class_ids, scores, label_consensus_iou_th),
                    "merged_observations": 1,
                    "evidence": {
                        "color_path": image_path,
                        "bbox_xyxy": [int(round(v)) for v in box.tolist()],
                        "sam_mask_path": mask_path,
                        **evidence,
                    },
                    "_seed_indices": seed_indices,
                }
                _merge_observation(
                    proposals,
                    observation,
                    merge_iou,
                    keep_sam_mask_alternatives=keep_sam_mask_alternatives,
                    seed_merge_policy=seed_merge_policy,
                    seed_merge_topk=seed_merge_topk,
                    seed_view_quality_gate=seed_view_quality_gate,
                    seed_view_quality_relative_threshold=seed_view_quality_relative_threshold,
                    seed_view_quality_min_score=seed_view_quality_min_score,
                    seed_view_quality_min_keep_ratio=seed_view_quality_min_keep_ratio,
                )
                raw_observations += 1
                accepted_for_detection += 1

            if accepted_for_detection == 0:
                skipped.append(
                    {
                        "frame_id": frame_id,
                        "detection_id": int(det_id),
                        "reason": "no_valid_sam_mask",
                    }
                )

    prefilter_skipped = []
    filtered_proposals = []
    for proposal in proposals:
        proposal.update(_existing_mask_metrics(existing_masks, proposal["_seed_indices"]))
        if (
            export_max_existing_iou is not None
            and float(proposal.get("best_existing_iou", 0.0)) > float(export_max_existing_iou)
        ):
            prefilter_skipped.append(
                {
                    "reason": "export_matched_existing_3d_mask",
                    "class_name": proposal.get("class_name"),
                    "best_existing_iou": float(proposal.get("best_existing_iou", 0.0)),
                }
            )
            continue
        if (
            export_max_seed_in_existing_mask_ratio is not None
            and float(proposal.get("seed_in_existing_mask_ratio", 0.0))
            > float(export_max_seed_in_existing_mask_ratio)
        ):
            prefilter_skipped.append(
                {
                    "reason": "export_mostly_covered_by_existing_masks",
                    "class_name": proposal.get("class_name"),
                    "seed_in_existing_mask_ratio": float(proposal.get("seed_in_existing_mask_ratio", 0.0)),
                }
            )
            continue
        filtered_proposals.append(proposal)
    proposals = filtered_proposals

    proposals = _sort_proposals(proposals, ranking_policy)
    if max_candidates is not None:
        proposals = proposals[:max_candidates]

    candidates = []
    for candidate_id, proposal in enumerate(proposals):
        seed_indices = proposal.pop("_seed_indices")
        proposal.pop("_seed_observations", None)
        seed_path = osp.join(seed_dir, f"candidate{candidate_id:04d}_points.npz")
        np.savez_compressed(seed_path, point_indices=seed_indices)
        proposal["candidate_id"] = int(candidate_id)
        proposal["num_seed_points"] = int(len(seed_indices))
        proposal["seed_points_path"] = seed_path
        candidates.append(proposal)

    json_path = osp.join(scene_dir, "backprojection_candidates.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "num_candidates": len(candidates),
                "raw_observations": raw_observations,
                "skipped": skipped,
                "prefilter_skipped": prefilter_skipped,
                "filters": {
                    "detection_score_th": detection_score_th,
                    "min_seed_points": min_seed_points,
                    "max_box_area_ratio": max_box_area_ratio,
                    "frame_stride": frame_stride,
                    "max_frames": max_frames,
                    "max_detections_per_frame": max_detections_per_frame,
                    "merge_iou": merge_iou,
                    "max_candidates": max_candidates,
                    "blocked_classes": sorted(blocked_classes),
                    "ranking_policy": ranking_policy,
                    "sam_multimask_topk": sam_multimask_topk,
                    "sam_mask_selection_policy": sam_mask_selection_policy,
                    "sam_mask_geometry_model": (
                        geometry_model_bundle.get("_model_path") if isinstance(geometry_model_bundle, dict) else None
                    ),
                    "sam_mask_geometry_cc_radius": sam_mask_geometry_cc_radius,
                    "sam_mask_geometry_plane_threshold": sam_mask_geometry_plane_threshold,
                    "sam_mask_geometry_max_points": sam_mask_geometry_max_points,
                    "keep_sam_mask_alternatives": keep_sam_mask_alternatives,
                    "seed_merge_policy": seed_merge_policy,
                    "seed_merge_topk": seed_merge_topk,
                    "seed_view_quality_gate": seed_view_quality_gate,
                    "seed_view_quality_relative_threshold": seed_view_quality_relative_threshold,
                    "seed_view_quality_min_score": seed_view_quality_min_score,
                    "seed_view_quality_min_keep_ratio": seed_view_quality_min_keep_ratio,
                    "seed_depth_cluster": seed_depth_cluster,
                    "seed_depth_cluster_bin_size": seed_depth_cluster_bin_size,
                    "seed_depth_cluster_window_bins": seed_depth_cluster_window_bins,
                    "seed_depth_cluster_min_keep_ratio": seed_depth_cluster_min_keep_ratio,
                    "seed_depth_cluster_min_removed_ratio": seed_depth_cluster_min_removed_ratio,
                    "seed_depth_cluster_max_removed_ratio": seed_depth_cluster_max_removed_ratio,
                    "sam_adaptive_internal_seed": sam_adaptive_internal_seed,
                    "sam_adaptive_internal_keep_ratio": sam_adaptive_internal_keep_ratio,
                    "sam_adaptive_internal_min_keep_ratio": sam_adaptive_internal_min_keep_ratio,
                    "sam_adaptive_internal_boundary_weight": sam_adaptive_internal_boundary_weight,
                    "sam_adaptive_internal_depth_weight": sam_adaptive_internal_depth_weight,
                    "sam_adaptive_internal_depth_bin_size": sam_adaptive_internal_depth_bin_size,
                    "sam_adaptive_internal_depth_window_bins": sam_adaptive_internal_depth_window_bins,
                    "sam_mask_erode_pixels": sam_mask_erode_pixels,
                    "sam_mask_erode_min_area_ratio": sam_mask_erode_min_area_ratio,
                    "export_max_existing_iou": export_max_existing_iou,
                    "export_max_seed_in_existing_mask_ratio": export_max_seed_in_existing_mask_ratio,
                    "label_consensus_iou_th": label_consensus_iou_th,
                    "box_nms_iou": box_nms_iou,
                    "box_nms_same_class_only": box_nms_same_class_only,
                },
                "candidates": candidates,
            },
            f,
            indent=2,
        )
    return json_path, candidates, {"raw_observations": raw_observations, "num_candidates": len(candidates)}


def export_dataset_sam_fused_proposals(
    dataset_name,
    path_to_3d_masks,
    output_dir,
    sam_checkpoint,
    sam_source,
    sam_model_type="vit_b",
    scene_name=None,
    detection_score_th=0.45,
    min_seed_points=80,
    max_box_area_ratio=0.30,
    frame_stride=5,
    max_frames=None,
    max_detections_per_frame=8,
    merge_iou=0.15,
    max_candidates_per_scene=30,
    blocked_classes=None,
    ranking_policy="support_priority",
    sam_multimask_topk=1,
    sam_mask_selection_policy="sam_score",
    sam_mask_geometry_model=None,
    sam_mask_geometry_cc_radius=0.03,
    sam_mask_geometry_plane_threshold=0.02,
    sam_mask_geometry_max_points=50000,
    keep_sam_mask_alternatives=False,
    seed_merge_policy="union",
    seed_merge_topk=1,
    seed_view_quality_gate=False,
    seed_view_quality_relative_threshold=0.80,
    seed_view_quality_min_score=0.0,
    seed_view_quality_min_keep_ratio=0.50,
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
    export_max_existing_iou=None,
    export_max_seed_in_existing_mask_ratio=None,
    label_consensus_iou_th=0.25,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
    path_to_2d_preds=None,
    reuse_2d_preds=True,
    scene_list=None,
    max_scenes=None,
):
    config = load_yaml(osp.join(f"./pretrained/config_{dataset_name}.yaml"))
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
    geometry_model_bundle = _load_geometry_discriminator(sam_mask_geometry_model)
    if str(sam_mask_selection_policy) == "learned_geometry" and geometry_model_bundle is None:
        raise ValueError("--sam_mask_geometry_model is required when using learned_geometry selection.")
    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_name}.yaml")
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    start = time.time()
    for current_scene in tqdm(scene_names):
        scene_id = current_scene.replace("scene", "")
        processed_file = (
            osp.join(path_2_dataset, current_scene, f"{scene_id}.npy")
            if dataset_name == "scannet200"
            else None
        )
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
        json_path, _, summary = export_scene_sam_fused_proposals(
            openyolo3d,
            predictor,
            current_scene,
            output_dir,
            detection_score_th=detection_score_th,
            min_seed_points=min_seed_points,
            max_box_area_ratio=max_box_area_ratio,
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_detections_per_frame=max_detections_per_frame,
            merge_iou=merge_iou,
            max_candidates=max_candidates_per_scene,
            blocked_classes=blocked_classes,
            ranking_policy=ranking_policy,
            sam_multimask_topk=sam_multimask_topk,
            sam_mask_selection_policy=sam_mask_selection_policy,
            sam_mask_geometry_model=geometry_model_bundle,
            sam_mask_geometry_cc_radius=sam_mask_geometry_cc_radius,
            sam_mask_geometry_plane_threshold=sam_mask_geometry_plane_threshold,
            sam_mask_geometry_max_points=sam_mask_geometry_max_points,
            keep_sam_mask_alternatives=keep_sam_mask_alternatives,
            seed_merge_policy=seed_merge_policy,
            seed_merge_topk=seed_merge_topk,
            seed_view_quality_gate=seed_view_quality_gate,
            seed_view_quality_relative_threshold=seed_view_quality_relative_threshold,
            seed_view_quality_min_score=seed_view_quality_min_score,
            seed_view_quality_min_keep_ratio=seed_view_quality_min_keep_ratio,
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
            export_max_existing_iou=export_max_existing_iou,
            export_max_seed_in_existing_mask_ratio=export_max_seed_in_existing_mask_ratio,
            label_consensus_iou_th=label_consensus_iou_th,
            box_nms_iou=box_nms_iou,
            box_nms_same_class_only=box_nms_same_class_only,
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

    summary_path = osp.join(output_dir, "sam_fused_proposals_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "elapsed_seconds": time.time() - start,
                "params": {
                    "detection_score_th": detection_score_th,
                    "min_seed_points": min_seed_points,
                    "max_box_area_ratio": max_box_area_ratio,
                    "frame_stride": frame_stride,
                    "max_frames": max_frames,
                    "max_detections_per_frame": max_detections_per_frame,
                    "merge_iou": merge_iou,
                    "max_candidates_per_scene": max_candidates_per_scene,
                    "blocked_classes": blocked_classes,
                    "ranking_policy": ranking_policy,
                    "sam_multimask_topk": sam_multimask_topk,
                    "sam_mask_selection_policy": sam_mask_selection_policy,
                    "sam_mask_geometry_model": (
                        geometry_model_bundle.get("_model_path") if isinstance(geometry_model_bundle, dict) else None
                    ),
                    "sam_mask_geometry_cc_radius": sam_mask_geometry_cc_radius,
                    "sam_mask_geometry_plane_threshold": sam_mask_geometry_plane_threshold,
                    "sam_mask_geometry_max_points": sam_mask_geometry_max_points,
                    "keep_sam_mask_alternatives": keep_sam_mask_alternatives,
                    "seed_merge_policy": seed_merge_policy,
                    "seed_merge_topk": seed_merge_topk,
                    "seed_view_quality_gate": seed_view_quality_gate,
                    "seed_view_quality_relative_threshold": seed_view_quality_relative_threshold,
                    "seed_view_quality_min_score": seed_view_quality_min_score,
                    "seed_view_quality_min_keep_ratio": seed_view_quality_min_keep_ratio,
                    "seed_depth_cluster": seed_depth_cluster,
                    "seed_depth_cluster_bin_size": seed_depth_cluster_bin_size,
                    "seed_depth_cluster_window_bins": seed_depth_cluster_window_bins,
                    "seed_depth_cluster_min_keep_ratio": seed_depth_cluster_min_keep_ratio,
                    "seed_depth_cluster_min_removed_ratio": seed_depth_cluster_min_removed_ratio,
                    "seed_depth_cluster_max_removed_ratio": seed_depth_cluster_max_removed_ratio,
                    "sam_adaptive_internal_seed": sam_adaptive_internal_seed,
                    "sam_adaptive_internal_keep_ratio": sam_adaptive_internal_keep_ratio,
                    "sam_adaptive_internal_min_keep_ratio": sam_adaptive_internal_min_keep_ratio,
                    "sam_adaptive_internal_boundary_weight": sam_adaptive_internal_boundary_weight,
                    "sam_adaptive_internal_depth_weight": sam_adaptive_internal_depth_weight,
                    "sam_adaptive_internal_depth_bin_size": sam_adaptive_internal_depth_bin_size,
                    "sam_adaptive_internal_depth_window_bins": sam_adaptive_internal_depth_window_bins,
                    "sam_mask_erode_pixels": sam_mask_erode_pixels,
                    "sam_mask_erode_min_area_ratio": sam_mask_erode_min_area_ratio,
                    "export_max_existing_iou": export_max_existing_iou,
                    "export_max_seed_in_existing_mask_ratio": export_max_seed_in_existing_mask_ratio,
                    "box_nms_iou": box_nms_iou,
                    "box_nms_same_class_only": box_nms_same_class_only,
                },
                "scenes": summaries,
            },
            f,
            indent=2,
        )
    print(f"Saved SAM-fused proposal summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--output_dir", default="./output/sam_fused_proposals_replica")
    parser.add_argument("--sam_checkpoint", default="./pretrained/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_source", default="./_external/segment-anything/segment-anything-main")
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--detection_score_th", default=0.45, type=float)
    parser.add_argument("--min_seed_points", default=80, type=int)
    parser.add_argument("--max_box_area_ratio", default=0.30, type=float)
    parser.add_argument("--frame_stride", default=5, type=int)
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--max_detections_per_frame", default=8, type=int)
    parser.add_argument("--merge_iou", default=0.15, type=float)
    parser.add_argument("--max_candidates_per_scene", default=30, type=int)
    parser.add_argument("--blocked_classes", default="rug")
    parser.add_argument(
        "--ranking_policy",
        default="support_priority",
        choices=["support_priority", "novelty", "balanced_novelty"],
    )
    parser.add_argument("--sam_multimask_topk", default=1, type=int)
    parser.add_argument(
        "--sam_mask_selection_policy",
        default="sam_score",
        choices=["sam_score", "geometry", "learned_geometry"],
        help="How to rank SAM masks from the same 2D box before back-projection",
    )
    parser.add_argument(
        "--sam_mask_geometry_model",
        default=None,
        help="Path to a trained geometry discriminator model.pkl for learned_geometry selection.",
    )
    parser.add_argument("--sam_mask_geometry_cc_radius", default=0.03, type=float)
    parser.add_argument("--sam_mask_geometry_plane_threshold", default=0.02, type=float)
    parser.add_argument("--sam_mask_geometry_max_points", default=50000, type=int)
    parser.add_argument("--keep_sam_mask_alternatives", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--seed_merge_policy",
        default="union",
        choices=["union", "best_view", "topk_priority"],
        help="How merged multi-view SAM observations produce final 3D seed points",
    )
    parser.add_argument("--seed_merge_topk", default=1, type=int)
    parser.add_argument("--seed_view_quality_gate", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed_view_quality_relative_threshold", default=0.80, type=float)
    parser.add_argument("--seed_view_quality_min_score", default=0.0, type=float)
    parser.add_argument("--seed_view_quality_min_keep_ratio", default=0.50, type=float)
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
    parser.add_argument("--export_max_existing_iou", default=None, type=float)
    parser.add_argument("--export_max_seed_in_existing_mask_ratio", default=None, type=float)
    parser.add_argument("--label_consensus_iou_th", default=0.25, type=float)
    parser.add_argument("--box_nms_iou", default=0.0, type=float, help="Optional 2D box NMS IoU before SAM-fused export; 0 disables")
    parser.add_argument("--box_nms_same_class_only", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--scene_list", default=None, help="Optional comma-separated scene names or file with one scene per line")
    parser.add_argument("--max_scenes", default=None, type=int)
    args = parser.parse_args()

    export_dataset_sam_fused_proposals(
        dataset_name=args.dataset_name,
        path_to_3d_masks=args.path_to_3d_masks,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_source=args.sam_source,
        sam_model_type=args.sam_model_type,
        scene_name=args.scene_name,
        detection_score_th=args.detection_score_th,
        min_seed_points=args.min_seed_points,
        max_box_area_ratio=args.max_box_area_ratio,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_detections_per_frame=args.max_detections_per_frame,
        merge_iou=args.merge_iou,
        max_candidates_per_scene=args.max_candidates_per_scene,
        blocked_classes=args.blocked_classes,
        ranking_policy=args.ranking_policy,
        sam_multimask_topk=args.sam_multimask_topk,
        sam_mask_selection_policy=args.sam_mask_selection_policy,
        sam_mask_geometry_model=args.sam_mask_geometry_model,
        sam_mask_geometry_cc_radius=args.sam_mask_geometry_cc_radius,
        sam_mask_geometry_plane_threshold=args.sam_mask_geometry_plane_threshold,
        sam_mask_geometry_max_points=args.sam_mask_geometry_max_points,
        keep_sam_mask_alternatives=args.keep_sam_mask_alternatives,
        seed_merge_policy=args.seed_merge_policy,
        seed_merge_topk=args.seed_merge_topk,
        seed_view_quality_gate=args.seed_view_quality_gate,
        seed_view_quality_relative_threshold=args.seed_view_quality_relative_threshold,
        seed_view_quality_min_score=args.seed_view_quality_min_score,
        seed_view_quality_min_keep_ratio=args.seed_view_quality_min_keep_ratio,
        seed_depth_cluster=args.seed_depth_cluster,
        seed_depth_cluster_bin_size=args.seed_depth_cluster_bin_size,
        seed_depth_cluster_window_bins=args.seed_depth_cluster_window_bins,
        seed_depth_cluster_min_keep_ratio=args.seed_depth_cluster_min_keep_ratio,
        seed_depth_cluster_min_removed_ratio=args.seed_depth_cluster_min_removed_ratio,
        seed_depth_cluster_max_removed_ratio=args.seed_depth_cluster_max_removed_ratio,
        sam_adaptive_internal_seed=args.sam_adaptive_internal_seed,
        sam_adaptive_internal_keep_ratio=args.sam_adaptive_internal_keep_ratio,
        sam_adaptive_internal_min_keep_ratio=args.sam_adaptive_internal_min_keep_ratio,
        sam_adaptive_internal_boundary_weight=args.sam_adaptive_internal_boundary_weight,
        sam_adaptive_internal_depth_weight=args.sam_adaptive_internal_depth_weight,
        sam_adaptive_internal_depth_bin_size=args.sam_adaptive_internal_depth_bin_size,
        sam_adaptive_internal_depth_window_bins=args.sam_adaptive_internal_depth_window_bins,
        sam_mask_erode_pixels=args.sam_mask_erode_pixels,
        sam_mask_erode_min_area_ratio=args.sam_mask_erode_min_area_ratio,
        export_max_existing_iou=args.export_max_existing_iou,
        export_max_seed_in_existing_mask_ratio=args.export_max_seed_in_existing_mask_ratio,
        label_consensus_iou_th=args.label_consensus_iou_th,
        box_nms_iou=args.box_nms_iou,
        box_nms_same_class_only=args.box_nms_same_class_only,
        path_to_2d_preds=args.path_to_2d_preds,
        reuse_2d_preds=args.reuse_2d_preds,
        scene_list=args.scene_list,
        max_scenes=args.max_scenes,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose ScanNet200 back-projection candidate quality against GT instances.

This script is intentionally offline: it reads exported candidate seed masks and
processed ScanNet200 scene arrays, then reports which candidates look like true
missing-object completions, duplicates, partial masks, or likely false positives.
"""

import argparse
import csv
import json
import math
import os
import os.path as osp
import sys
from collections import Counter, defaultdict

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST
from utils.backprojection_fusion import (
    _candidate_quality_score,
    _candidate_source_kind,
    _candidate_source_name,
    _load_seed_indices,
    _refine_mask_with_superpoints,
)


VALID_INSTANCE_CLASS_IDS = set(int(item) for item in VALID_CLASS_IDS_200_INST)


def _iter_candidate_json_paths(path):
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def _load_candidates(paths):
    grouped = defaultdict(list)
    for root in paths:
        for json_path in _iter_candidate_json_paths(root):
            with open(json_path) as f:
                payload = json.load(f)
            scene_name = payload.get("scene_name")
            source_name = osp.basename(osp.dirname(osp.dirname(json_path))) or osp.basename(osp.dirname(json_path))
            for candidate in payload.get("candidates", []):
                record = dict(candidate)
                record.setdefault("scene_name", scene_name)
                record["_source_json"] = json_path
                record["_source_name"] = source_name
                grouped[str(record["scene_name"])].append(record)
    return dict(grouped)


def _load_report_applied_candidates(path):
    if path is None:
        return None, {}
    with open(path) as f:
        payload = json.load(f)
    keys = set()
    metadata = {}
    for scene_name, report in payload.get("scene_reports", {}).items():
        for item in report.get("applied", []):
            source_json = item.get("source_json")
            candidate_id = item.get("candidate_id")
            if source_json is None or candidate_id is None:
                continue
            key = (str(scene_name), osp.abspath(source_json), int(candidate_id))
            keys.add(key)
            metadata[key] = item
    return keys, metadata


def _read_scene_list(path):
    if path is None:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def _load_processed_scene(dataset_root, scene_name):
    scene_id = scene_name.replace("scene", "")
    path = osp.join(dataset_root, scene_name, f"{scene_id}.npy")
    if not osp.exists(path):
        raise FileNotFoundError(f"Processed scene file not found: {path}")
    return np.load(path, mmap_mode="r")


def _adjust_intrinsic(intrinsic, original_resolution, new_resolution):
    if tuple(original_resolution) == tuple(new_resolution):
        return intrinsic
    resize_width = int(math.floor(new_resolution[1] * float(original_resolution[0]) / float(original_resolution[1])))
    adapted = intrinsic.copy()
    adapted[0, 0] *= float(resize_width) / float(original_resolution[0])
    adapted[1, 1] *= float(new_resolution[1]) / float(original_resolution[1])
    adapted[0, 2] *= float(new_resolution[0] - 1) / float(original_resolution[0] - 1)
    adapted[1, 2] *= float(new_resolution[1] - 1) / float(original_resolution[1] - 1)
    return adapted


def _load_scene_depth_context(dataset_root, scene_name):
    scene_dir = osp.join(dataset_root, scene_name)
    depth_paths = sorted(
        (osp.join(scene_dir, "depth", name) for name in os.listdir(osp.join(scene_dir, "depth")) if name.endswith(".png")),
        key=lambda path: int(osp.splitext(osp.basename(path))[0]),
    )
    color_paths = sorted(
        (osp.join(scene_dir, "color", name) for name in os.listdir(osp.join(scene_dir, "color")) if name.endswith(".jpg")),
        key=lambda path: int(osp.splitext(osp.basename(path))[0]),
    )
    if not depth_paths or not color_paths:
        return None
    depth_resolution = imageio.imread(depth_paths[0]).shape[:2]
    image_resolution = imageio.imread(color_paths[0]).shape[:2]
    intrinsic = np.loadtxt(osp.join(scene_dir, "intrinsics.txt")).astype(np.float64)
    return {
        "scene_dir": scene_dir,
        "intrinsic": _adjust_intrinsic(intrinsic, image_resolution, depth_resolution),
        "depth_resolution": depth_resolution,
        "depth_cache": {},
        "pose_cache": {},
    }


def _candidate_support_frame_ids(candidate, max_views):
    views = candidate.get("support_views")
    if not isinstance(views, list):
        views = []
    views = sorted(
        views,
        key=lambda item: (
            -int(item.get("visible_seed_points", 0) or 0),
            -float(item.get("score", 0.0) or 0.0),
        ),
    )
    frame_ids = []
    for view in views:
        frame_id = view.get("frame_id")
        if frame_id is None:
            continue
        frame_id = str(frame_id)
        if frame_id not in frame_ids:
            frame_ids.append(frame_id)
    if not frame_ids:
        evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
        color_path = evidence.get("color_path")
        if color_path:
            frame_ids.append(osp.splitext(osp.basename(color_path))[0])
    return frame_ids[: max(1, int(max_views))]


def _load_depth_frame(context, frame_id, depth_scale):
    if frame_id not in context["depth_cache"]:
        path = osp.join(context["scene_dir"], "depth", f"{frame_id}.png")
        if not osp.exists(path):
            return None
        context["depth_cache"][frame_id] = imageio.imread(path).astype(np.float32) / float(depth_scale)
    return context["depth_cache"][frame_id]


def _load_world_to_camera(context, frame_id):
    if frame_id not in context["pose_cache"]:
        path = osp.join(context["scene_dir"], "poses", f"{frame_id}.txt")
        if not osp.exists(path):
            return None
        pose = np.loadtxt(path).astype(np.float64)
        if not np.isfinite(pose).all():
            return None
        try:
            context["pose_cache"][frame_id] = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            return None
    return context["pose_cache"][frame_id]


def _candidate_depth_features(
    points_xyz,
    candidate_mask,
    candidate,
    context,
    depth_scale,
    consistency_threshold,
    layer_bin_size,
    max_views,
    max_points,
):
    output = {
        "depth_support_view_count": 0,
        "depth_valid_view_count": 0,
        "depth_projected_point_count": 0,
        "depth_valid_observation_count": 0,
        "depth_valid_projection_ratio": 0.0,
        "depth_consistency_ratio": 0.0,
        "depth_front_gap_ratio": 0.0,
        "depth_behind_surface_ratio": 0.0,
        "depth_residual_mean": 0.0,
        "depth_residual_p50": 0.0,
        "depth_residual_p90": 0.0,
        "depth_camera_span_mean": 0.0,
        "depth_camera_span_relative_mean": 0.0,
        "depth_main_layer_ratio_mean": 0.0,
        "depth_layer_count_mean": 0.0,
    }
    if context is None:
        return output
    indices = np.flatnonzero(candidate_mask)
    if len(indices) == 0:
        return output
    if max_points is not None and len(indices) > int(max_points):
        step = max(1, int(math.ceil(len(indices) / int(max_points))))
        indices = indices[::step][: int(max_points)]
    local_points = points_xyz[indices].astype(np.float64, copy=False)
    points_h = np.concatenate([local_points, np.ones((len(local_points), 1), dtype=np.float64)], axis=1)
    frame_ids = _candidate_support_frame_ids(candidate, max_views)
    output["depth_support_view_count"] = int(len(frame_ids))

    all_residuals = []
    projected_count = 0
    valid_observation_count = 0
    consistent_count = 0
    front_gap_count = 0
    behind_surface_count = 0
    spans = []
    relative_spans = []
    main_layer_ratios = []
    layer_counts = []
    height, width = context["depth_resolution"]
    intrinsic = context["intrinsic"]
    threshold = float(consistency_threshold)
    bin_size = max(float(layer_bin_size), 1e-4)

    for frame_id in frame_ids:
        depth_map = _load_depth_frame(context, frame_id, depth_scale)
        world_to_camera = _load_world_to_camera(context, frame_id)
        if depth_map is None or world_to_camera is None:
            continue
        camera_points = (world_to_camera @ points_h.T).T
        camera_depth = camera_points[:, 2]
        positive = np.isfinite(camera_depth) & (camera_depth > 0)
        if not positive.any():
            continue
        camera_points = camera_points[positive]
        camera_depth = camera_depth[positive]
        projected = (intrinsic @ camera_points.T).T
        xs = np.rint(projected[:, 0] / camera_depth).astype(np.int64)
        ys = np.rint(projected[:, 1] / camera_depth).astype(np.int64)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not inside.any():
            continue
        xs = xs[inside]
        ys = ys[inside]
        camera_depth = camera_depth[inside]
        projected_count += int(len(camera_depth))
        observed_depth = depth_map[ys, xs]
        valid_depth = np.isfinite(observed_depth) & (observed_depth > 0)
        if not valid_depth.any():
            continue
        output["depth_valid_view_count"] += 1
        camera_depth = camera_depth[valid_depth]
        observed_depth = observed_depth[valid_depth]
        residuals = camera_depth - observed_depth
        abs_residuals = np.abs(residuals)
        all_residuals.extend(abs_residuals.tolist())
        valid_observation_count += int(len(residuals))
        consistent_count += int((abs_residuals <= threshold).sum())
        front_gap_count += int((residuals < -threshold).sum())
        behind_surface_count += int((residuals > threshold).sum())

        lower, upper = np.percentile(camera_depth, [10, 90])
        span = float(max(0.0, upper - lower))
        spans.append(span)
        relative_spans.append(float(span / max(float(np.median(camera_depth)), 1e-4)))
        bins = np.floor((camera_depth - float(camera_depth.min())) / bin_size).astype(np.int64)
        counts = np.bincount(bins)
        main_layer_ratios.append(float(counts.max(initial=0) / max(1, len(camera_depth))))
        layer_counts.append(float((counts > max(2, int(0.05 * len(camera_depth)))).sum()))

    output["depth_projected_point_count"] = int(projected_count)
    output["depth_valid_observation_count"] = int(valid_observation_count)
    output["depth_valid_projection_ratio"] = float(valid_observation_count / max(1, projected_count))
    output["depth_consistency_ratio"] = float(consistent_count / max(1, valid_observation_count))
    output["depth_front_gap_ratio"] = float(front_gap_count / max(1, valid_observation_count))
    output["depth_behind_surface_ratio"] = float(behind_surface_count / max(1, valid_observation_count))
    if all_residuals:
        residuals = np.asarray(all_residuals, dtype=np.float32)
        output["depth_residual_mean"] = float(residuals.mean())
        output["depth_residual_p50"] = float(np.percentile(residuals, 50))
        output["depth_residual_p90"] = float(np.percentile(residuals, 90))
    if spans:
        output["depth_camera_span_mean"] = float(np.mean(spans))
        output["depth_camera_span_relative_mean"] = float(np.mean(relative_spans))
        output["depth_main_layer_ratio_mean"] = float(np.mean(main_layer_ratios))
        output["depth_layer_count_mean"] = float(np.mean(layer_counts))
    return output


def _build_gt_instances(processed_scene, min_points):
    semantic_ids = processed_scene[:, 10].astype(np.int64)
    instance_ids = processed_scene[:, 11].astype(np.int64)
    instances = []
    for instance_id in np.unique(instance_ids):
        if instance_id < 0:
            continue
        mask = instance_ids == int(instance_id)
        point_count = int(mask.sum())
        if point_count < int(min_points):
            continue
        labels, counts = np.unique(semantic_ids[mask], return_counts=True)
        class_id = int(labels[int(np.argmax(counts))])
        if class_id not in VALID_INSTANCE_CLASS_IDS:
            continue
        instances.append(
            {
                "instance_id": int(instance_id),
                "class_id": class_id,
                "class_name": ID_TO_LABEL.get(class_id, str(class_id)),
                "mask": mask,
                "point_count": point_count,
            }
        )
    return instances


def _load_baseline_masks(path, scene_name, num_points):
    if path is None:
        return None
    mask_path = osp.join(path, f"{scene_name}.pt")
    if not osp.exists(mask_path):
        return None
    payload = torch.load(mask_path, map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    if masks.shape[0] != num_points and masks.shape[1] == num_points:
        masks = masks.T
    if masks.shape[0] != num_points:
        raise ValueError(f"Unexpected baseline mask shape for {scene_name}: {masks.shape}")
    return masks.astype(bool, copy=False)


def _mask_iou_one_to_many(mask, masks):
    if masks is None or len(masks) == 0:
        return np.zeros((0,), dtype=np.float32)
    mask = mask.astype(bool, copy=False)
    if isinstance(masks, list):
        masks = np.stack(masks, axis=1) if masks else np.zeros((mask.shape[0], 0), dtype=bool)
    masks = masks.astype(bool, copy=False)
    intersections = np.logical_and(masks, mask[:, None]).sum(axis=0)
    unions = masks.sum(axis=0) + int(mask.sum()) - intersections
    return np.divide(intersections, np.maximum(unions, 1), dtype=np.float32)


def _connected_component_geometry(local_points, radius, max_points):
    output = {
        "geometry_component_count": 0,
        "geometry_largest_component_ratio": 0.0,
        "geometry_non_largest_component_ratio": 0.0,
        "geometry_small_component_ratio": 0.0,
        "geometry_component_skipped": False,
    }
    num_points = int(len(local_points))
    if num_points <= 0 or float(radius) <= 0.0:
        return output
    if max_points is not None and num_points > int(max_points):
        output["geometry_component_skipped"] = True
        return output

    tree = cKDTree(local_points)
    neighbors = tree.query_ball_point(local_points, r=float(radius))
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
            "geometry_component_count": int(len(counts)),
            "geometry_largest_component_ratio": float(largest / max(1, num_points)),
            "geometry_non_largest_component_ratio": float((num_points - largest) / max(1, num_points)),
            "geometry_small_component_ratio": float(small_points / max(1, num_points)),
        }
    )
    return output


def _candidate_geometry_features(points_xyz, candidate_mask, cc_radius, plane_threshold, cc_max_points):
    indices = np.flatnonzero(candidate_mask)
    local_points = points_xyz[indices].astype(np.float32, copy=False)
    num_points = int(len(local_points))
    output = {
        "geometry_point_count": num_points,
        "geometry_extent_x": 0.0,
        "geometry_extent_y": 0.0,
        "geometry_extent_z": 0.0,
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
        "geometry_plane_inlier_ratio": 0.0,
        "geometry_plane_residual_mean": 0.0,
        "geometry_plane_residual_p90": 0.0,
    }
    output.update(_connected_component_geometry(local_points, cc_radius, cc_max_points))
    if num_points <= 0:
        return output

    lower = local_points.min(axis=0)
    upper = local_points.max(axis=0)
    extents = np.maximum(upper - lower, 0.0)
    sorted_extents = np.sort(extents)[::-1]
    extent_max, extent_mid, extent_min = [float(value) for value in sorted_extents]
    bbox_volume = float(np.prod(np.maximum(extents, 1e-4)))
    output.update(
        {
            "geometry_extent_x": float(extents[0]),
            "geometry_extent_y": float(extents[1]),
            "geometry_extent_z": float(extents[2]),
            "geometry_extent_max": extent_max,
            "geometry_extent_mid": extent_mid,
            "geometry_extent_min": extent_min,
            "geometry_aspect_max_mid": float(extent_max / max(extent_mid, 1e-4)),
            "geometry_aspect_mid_min": float(extent_mid / max(extent_min, 1e-4)),
            "geometry_aspect_max_min": float(extent_max / max(extent_min, 1e-4)),
            "geometry_bbox_volume": bbox_volume,
            "geometry_bbox_density": float(num_points / max(bbox_volume, 1e-4)),
        }
    )

    if num_points < 3:
        return output
    centered = local_points - local_points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return output
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
    output.update(
        {
            "geometry_plane_inlier_ratio": float(np.mean(residuals <= float(plane_threshold))),
            "geometry_plane_residual_mean": float(np.mean(residuals)),
            "geometry_plane_residual_p90": float(np.percentile(residuals, 90)),
        }
    )
    return output


def _report_geometry_features(report_item):
    superpoint_info = report_item.get("superpoint_refine") if isinstance(report_item, dict) else {}
    if not isinstance(superpoint_info, dict):
        superpoint_info = {}
    support_info = superpoint_info.get("box_support_filter")
    if not isinstance(support_info, dict):
        support_info = {}
    cc_info = report_item.get("cc_cleanup") if isinstance(report_item, dict) else {}
    if not isinstance(cc_info, dict):
        cc_info = {}

    input_segments = int(support_info.get("input_segments", 0) or 0)
    filtered_segments = int(support_info.get("filtered_segments", 0) or 0)
    mean_positive = float(support_info.get("mean_positive_ratio", 0.0) or 0.0)
    output = {
        "report_mask_support_enabled": bool(support_info.get("enabled", False)),
        "report_mask_support_mode": str(support_info.get("support_mode", "none") or "none"),
        "report_mask_support_mean_positive_ratio": mean_positive,
        "report_mask_support_mean_negative_ratio": float(max(0.0, 1.0 - mean_positive)),
        "report_mask_support_filtered_segments": filtered_segments,
        "report_mask_support_filtered_ratio": float(filtered_segments / max(1, input_segments)),
        "report_mask_support_usable_view_count": int(support_info.get("usable_view_count", 0) or 0),
        "report_cc_component_count": int(cc_info.get("num_components", 0) or 0),
        "report_cc_largest_component_ratio": float(
            float(cc_info.get("largest_component_points", 0) or 0)
            / max(1.0, float(cc_info.get("input_points", 0) or 0))
        ),
        "report_cc_keep_ratio": float(
            float(cc_info.get("output_points", 0) or 0)
            / max(1.0, float(cc_info.get("input_points", 0) or 0))
        ),
    }
    return output


def _baseline_gt_coverage(gt_mask, baseline_masks):
    if baseline_masks is None or baseline_masks.shape[1] == 0:
        return float("nan")
    return float(_mask_iou_one_to_many(gt_mask, baseline_masks).max(initial=0.0))


def _candidate_semantic_id(candidate):
    try:
        pred_id = int(candidate.get("class_id"))
    except (TypeError, ValueError):
        return None
    return PRED_ID_TO_ID.get(pred_id)


def _consistency_rate(candidate):
    views = float(candidate.get("support_view_count", 0) or 0)
    merged = float(candidate.get("merged_observations", 0) or 0)
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
    visible = float(
        evidence.get("visible_view_count")
        or evidence.get("candidate_visible_view_count")
        or candidate.get("visible_view_count")
        or max(views, merged, 1.0)
    )
    return float(min(1.0, views / max(visible, 1.0)))


def _standardized_scores(rows, group_key, output_key):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_key(row)].append(index)
    for indices in grouped.values():
        values = np.asarray([rows[index]["quality_score"] for index in indices], dtype=np.float32)
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 0.0
        for index in indices:
            rows[index][output_key] = (
                0.0 if std < 1e-6 else float((rows[index]["quality_score"] - mean) / std)
            )


def _label_candidate(best_same_iou, best_any_iou, best_baseline_gt_iou, best_existing_iou):
    baseline_missed_50 = math.isnan(best_baseline_gt_iou) or best_baseline_gt_iou < 0.50
    if best_same_iou >= 0.50 and baseline_missed_50:
        return "true_completion_50"
    if best_same_iou >= 0.50:
        return "duplicate_or_already_covered_50"
    if best_same_iou >= 0.25 and baseline_missed_50:
        return "partial_completion_25"
    if best_same_iou >= 0.25:
        return "partial_duplicate_25"
    if best_any_iou >= 0.25:
        return "class_error_or_cross_class_overlap"
    if best_existing_iou >= 0.30:
        return "existing_overlap_low_gt"
    return "background_or_bad_geometry"


def _summarize(rows):
    summary = {
        "total_candidates": len(rows),
        "labels": Counter(row["diagnostic_label"] for row in rows),
        "sources": Counter(row["source_kind"] for row in rows),
        "classes": Counter(row["class_name"] for row in rows),
        "by_source": {},
        "by_class": {},
        "by_source_label": {},
    }
    for key_name, key_fn in (
        ("by_source", lambda row: row["source_kind"]),
        ("by_class", lambda row: row["class_name"]),
        ("by_source_label", lambda row: f"{row['source_kind']}::{row['diagnostic_label']}"),
    ):
        buckets = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        for key, items in buckets.items():
            summary[key_name][key] = {
                "count": len(items),
                "label_counts": dict(Counter(item["diagnostic_label"] for item in items)),
                "mean_best_same_class_iou": float(np.mean([item["best_same_class_gt_iou"] for item in items])),
                "mean_best_any_iou": float(np.mean([item["best_any_gt_iou"] for item in items])),
                "mean_quality_score": float(np.mean([item["quality_score"] for item in items])),
                "mean_support_view_count": float(np.mean([item["support_view_count"] for item in items])),
                "mean_consistency_rate": float(np.mean([item["consistency_rate"] for item in items])),
                "mean_label_consensus_score": float(np.mean([item["label_consensus_score"] for item in items])),
                "mean_label_conflict_score": float(np.mean([item["label_conflict_score"] for item in items])),
            }
    for key in ("labels", "sources", "classes"):
        summary[key] = dict(summary[key])
    return summary


def analyze(args):
    scene_list = _read_scene_list(args.scene_list)
    candidate_paths = [item.strip() for item in args.candidates.split(",") if item.strip()]
    candidates_by_scene = _load_candidates(candidate_paths)
    report_applied_keys, report_applied_metadata = _load_report_applied_candidates(args.backprojection_report)
    if scene_list is None:
        scene_names = sorted(candidates_by_scene)
    else:
        scene_names = [scene for scene in scene_list if scene in candidates_by_scene]
    if args.max_scenes is not None:
        scene_names = scene_names[: args.max_scenes]
    if not scene_names:
        raise ValueError("No scenes selected for candidate analysis.")

    rows = []
    for scene_name in tqdm(scene_names, desc="Analyzing candidates"):
        processed_scene = _load_processed_scene(args.dataset_root, scene_name)
        num_points = int(processed_scene.shape[0])
        points_xyz = processed_scene[:, :3].astype(np.float32)
        depth_context = _load_scene_depth_context(args.dataset_root, scene_name) if args.depth_features else None
        point_segments = processed_scene[:, 9].astype(np.int64) if args.superpoint_refine else None
        gt_instances = _build_gt_instances(processed_scene, args.min_gt_points)
        gt_masks = [item["mask"] for item in gt_instances]
        gt_matrix = np.stack(gt_masks, axis=1) if gt_masks else np.zeros((num_points, 0), dtype=bool)
        baseline_masks = _load_baseline_masks(args.baseline_masks, scene_name, num_points)

        for candidate in candidates_by_scene.get(scene_name, []):
            report_key = (
                scene_name,
                osp.abspath(candidate.get("_source_json", "")),
                int(candidate.get("candidate_id", -1)),
            )
            report_item = report_applied_metadata.get(report_key, {})
            if report_applied_keys is not None and report_key not in report_applied_keys:
                continue
            seed_indices = _load_seed_indices(candidate, num_points)
            if seed_indices is None or len(seed_indices) == 0:
                continue
            candidate_mask = np.zeros((num_points,), dtype=bool)
            candidate_mask[seed_indices] = True
            refine_info = {"enabled": False}
            if args.superpoint_refine:
                candidate_mask, refine_info = _refine_mask_with_superpoints(
                candidate_mask,
                point_segments,
                min_coverage=args.superpoint_min_coverage,
                max_expansion_ratio=args.superpoint_max_expansion_ratio,
                max_segment_ratio=args.superpoint_max_segment_ratio,
                large_segment_min_coverage=args.superpoint_large_segment_min_coverage,
                min_seed_retention=args.superpoint_min_seed_retention,
                min_output_points=args.min_seed_points,
            )

            semantic_id = _candidate_semantic_id(candidate)
            ious = _mask_iou_one_to_many(candidate_mask, gt_matrix)
            best_any_index = int(np.argmax(ious)) if len(ious) else -1
            best_any_iou = float(ious[best_any_index]) if best_any_index >= 0 else 0.0
            same_indices = [
                index for index, gt in enumerate(gt_instances) if semantic_id is not None and gt["class_id"] == semantic_id
            ]
            if same_indices:
                same_ious = ious[np.asarray(same_indices, dtype=np.int64)]
                same_local_index = int(np.argmax(same_ious))
                best_same_index = int(same_indices[same_local_index])
                best_same_iou = float(same_ious[same_local_index])
            else:
                best_same_index = -1
                best_same_iou = 0.0

            best_gt_index = best_same_index if best_same_index >= 0 else best_any_index
            if best_gt_index >= 0:
                best_baseline_gt_iou = _baseline_gt_coverage(gt_instances[best_gt_index]["mask"], baseline_masks)
                best_gt_class = gt_instances[best_gt_index]["class_name"]
                best_gt_instance_id = gt_instances[best_gt_index]["instance_id"]
            else:
                best_baseline_gt_iou = float("nan")
                best_gt_class = ""
                best_gt_instance_id = -1

            best_existing_iou = float(candidate.get("best_existing_iou", 0.0) or 0.0)
            geometry_features = _candidate_geometry_features(
                points_xyz,
                candidate_mask,
                args.geometry_cc_radius,
                args.geometry_plane_threshold,
                args.geometry_cc_max_points,
            )
            depth_features = _candidate_depth_features(
                points_xyz,
                candidate_mask,
                candidate,
                depth_context,
                args.depth_scale,
                args.depth_consistency_threshold,
                args.depth_layer_bin_size,
                args.depth_max_views,
                args.depth_max_points,
            )
            report_geometry_features = _report_geometry_features(report_item)
            row = {
                "scene_name": scene_name,
                "candidate_id": int(candidate.get("candidate_id", -1)),
                "source_name": _candidate_source_name(candidate),
                "source_kind": _candidate_source_kind(candidate),
                "class_id": int(candidate.get("class_id", -1)),
                "class_name": candidate.get("class_name", ""),
                "semantic_id": int(semantic_id) if semantic_id is not None else -1,
                "best_same_class_gt_iou": best_same_iou,
                "best_any_gt_iou": best_any_iou,
                "best_gt_class": best_gt_class,
                "best_gt_instance_id": int(best_gt_instance_id),
                "best_baseline_gt_iou": best_baseline_gt_iou,
                "best_existing_iou": best_existing_iou,
                "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", 0.0) or 0.0),
                "score": float(candidate.get("score", 0.0) or 0.0),
                "fusion_score": float(candidate.get("fusion_score", candidate.get("score", 0.0)) or 0.0),
                "quality_score": _candidate_quality_score(candidate),
                "support_view_count": int(candidate.get("support_view_count", 0) or 0),
                "support_mean_iou": float(candidate.get("support_mean_iou", 0.0) or 0.0),
                "support_best_iou": float(candidate.get("support_best_iou", 0.0) or 0.0),
                "consistency_rate": _consistency_rate(candidate),
                "label_consensus_score": float(candidate.get("label_consensus_score", 1.0) or 0.0),
                "label_conflict_score": float(candidate.get("label_conflict_score", 0.0) or 0.0),
                "label_margin": float(candidate.get("label_margin", 0.0) or 0.0),
                "label_evidence_view_count": int(candidate.get("label_evidence_view_count", 0) or 0),
                "label_conflict_view_count": int(candidate.get("label_conflict_view_count", 0) or 0),
                "box_area_ratio": float(candidate.get("box_area_ratio", 0.0) or 0.0),
                "num_seed_points": int(candidate.get("num_seed_points", len(seed_indices)) or len(seed_indices)),
                "num_mask_points": int(candidate_mask.sum()),
                "superpoint_refined": bool(refine_info.get("enabled", False)),
                "superpoint_expansion_ratio": float(refine_info.get("expansion_ratio", 1.0) or 1.0),
                "applied_proposal_score": float(report_item.get("proposal_score", -1.0) or -1.0),
                "applied_score_calibration": float(report_item.get("score_calibration", -1.0) or -1.0),
                "applied_source_score_scale": float(report_item.get("source_score_scale", -1.0) or -1.0),
            }
            row.update(geometry_features)
            row.update(depth_features)
            row.update(report_geometry_features)
            row["diagnostic_label"] = _label_candidate(
                row["best_same_class_gt_iou"],
                row["best_any_gt_iou"],
                row["best_baseline_gt_iou"],
                row["best_existing_iou"],
            )
            rows.append(row)

    _standardized_scores(rows, lambda row: (row["scene_name"], row["source_kind"]), "scene_source_quality_z")
    _standardized_scores(rows, lambda row: (row["class_name"], row["source_kind"]), "class_source_quality_z")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = osp.join(args.output_dir, "candidate_diagnostics.csv")
    json_path = osp.join(args.output_dir, "candidate_diagnostics_summary.json")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(_summarize(rows), f, indent=2, sort_keys=True)
    print(f"[INFO] Wrote {len(rows)} candidate rows to {csv_path}")
    print(f"[INFO] Wrote summary to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate directories or JSON files.")
    parser.add_argument("--dataset_root", default="./data/scannet200", help="ScanNet200 processed dataset root.")
    parser.add_argument("--baseline_masks", default="./output/scannet200/scannet200_masks", help="Optional baseline 3D mask cache.")
    parser.add_argument("--scene_list", default=None, help="Optional scene split txt.")
    parser.add_argument("--max_scenes", default=None, type=int, help="Optional cap for smoke tests.")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV and JSON diagnostics.")
    parser.add_argument("--backprojection_report", default=None, help="Optional run_evaluation BPR report; when set, analyze only applied candidates from that report.")
    parser.add_argument("--min_gt_points", default=100, type=int, help="Minimum GT instance size.")
    parser.add_argument("--min_seed_points", default=80, type=int, help="Minimum refined output points.")
    parser.add_argument("--superpoint_refine", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--superpoint_min_coverage", default=0.30, type=float)
    parser.add_argument("--superpoint_max_expansion_ratio", default=3.0, type=float)
    parser.add_argument("--superpoint_max_segment_ratio", default=None, type=float)
    parser.add_argument("--superpoint_large_segment_min_coverage", default=None, type=float)
    parser.add_argument("--superpoint_min_seed_retention", default=0.0, type=float)
    parser.add_argument("--geometry_cc_radius", default=0.03, type=float, help="Radius for diagnostic 3D connected-component features.")
    parser.add_argument("--geometry_cc_max_points", default=50000, type=int, help="Skip diagnostic connected components above this point count.")
    parser.add_argument("--geometry_plane_threshold", default=0.02, type=float, help="Distance threshold for diagnostic plane inlier ratio.")
    parser.add_argument("--depth_features", default=True, action=argparse.BooleanOptionalAction, help="Add ScanNet depth-consistency diagnostic features.")
    parser.add_argument("--depth_scale", default=1000.0, type=float, help="Raw ScanNet depth scale.")
    parser.add_argument("--depth_consistency_threshold", default=0.05, type=float, help="Depth residual threshold in meters.")
    parser.add_argument("--depth_layer_bin_size", default=0.10, type=float, help="Bin size in meters for depth layer counts.")
    parser.add_argument("--depth_max_views", default=5, type=int, help="Maximum support views used per candidate for depth features.")
    parser.add_argument("--depth_max_points", default=5000, type=int, help="Maximum candidate points sampled for depth features.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

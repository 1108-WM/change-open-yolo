import json
import os
import os.path as osp
from collections import defaultdict

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree


SUPERPOINT_CACHE_VERSION = "mask3d_superpoint_diagnostics_v2"


def _normalize_rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb
    if float(rgb.max(initial=0.0)) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _normalize_normals(normals):
    normals = np.asarray(normals, dtype=np.float32)
    if normals.size == 0:
        return normals
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-6)


def _safe_scaling_params(scaling_params):
    if scaling_params is None or len(scaling_params) < 2:
        return None
    return [float(scaling_params[0]), float(scaling_params[1])]


def _pythonize(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _projected_mask_coords(coords, scaling_params):
    if scaling_params is None:
        return np.round(coords[:, 0]).astype(np.int64), np.round(coords[:, 1]).astype(np.int64)
    return (
        np.round(coords[:, 0] / float(scaling_params[1])).astype(np.int64),
        np.round(coords[:, 1] / float(scaling_params[0])).astype(np.int64),
    )


class _FrameProjector:
    def __init__(self, points_xyz, world2cam, projections_np, scaling_params=None):
        self.points_xyz = np.asarray(points_xyz, dtype=np.float32)
        self.world2cam = world2cam
        self.projections_np = np.asarray(projections_np, dtype=np.float32)
        self.scaling_params = _safe_scaling_params(scaling_params)
        self.depth_cache = {}
        self.extrinsic_cache = {}

    def _load_depth(self, frame_index):
        frame_index = int(frame_index)
        if frame_index not in self.depth_cache:
            if frame_index < 0 or frame_index >= len(self.world2cam.depth_maps_paths):
                return None
            depth = imageio.imread(self.world2cam.depth_maps_paths[frame_index]).astype(np.float32)
            depth = depth / float(self.world2cam.depth_scale)
            self.depth_cache[frame_index] = depth
        return self.depth_cache[frame_index]

    def _load_extrinsic(self, frame_index):
        frame_index = int(frame_index)
        if frame_index not in self.extrinsic_cache:
            if frame_index < 0 or frame_index >= len(self.world2cam.poses):
                return None
            self.extrinsic_cache[frame_index] = np.linalg.inv(
                np.loadtxt(self.world2cam.poses[frame_index]).astype(np.float64)
            )
        return self.extrinsic_cache[frame_index]

    def segment_observation_metrics(self, point_indices, frame_index, sam_mask):
        point_indices = np.asarray(point_indices, dtype=np.int64)
        output = {
            "input_points": int(len(point_indices)),
            "projected_points": 0,
            "valid_depth_points": 0,
            "visible_points": 0,
            "inside_mask_points": 0,
            "inside_depth_consistent_points": 0,
            "inside_depth_conflict_points": 0,
            "outside_visible_points": 0,
            "depth_consistency_ratio": 0.0,
            "depth_conflict_ratio": 0.0,
            "inside_visible_ratio": 0.0,
            "outside_visible_ratio": 0.0,
            "visible_ratio": 0.0,
            "visible_coverage_ratio": 0.0,
            "full_coverage_ratio": 0.0,
            "median_depth_error": 0.0,
            "p90_depth_error": 0.0,
        }
        if len(point_indices) == 0:
            return output

        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self.projections_np.shape[0]:
            return output
        depth = self._load_depth(frame_index)
        extrinsic = self._load_extrinsic(frame_index)
        if depth is None or extrinsic is None:
            return output

        coords = self.projections_np[frame_index, point_indices]
        xs_depth = np.round(coords[:, 0]).astype(np.int64)
        ys_depth = np.round(coords[:, 1]).astype(np.int64)
        valid_pixel = (xs_depth >= 0) & (xs_depth < depth.shape[1]) & (ys_depth >= 0) & (ys_depth < depth.shape[0])
        if not valid_pixel.any():
            return output

        valid_indices = point_indices[valid_pixel]
        output["projected_points"] = int(len(valid_indices))
        coords = coords[valid_pixel]
        xs_depth = xs_depth[valid_pixel]
        ys_depth = ys_depth[valid_pixel]
        measured_depth = depth[ys_depth, xs_depth].astype(np.float32)
        valid_depth = measured_depth > 0.0
        if not valid_depth.any():
            return output

        valid_indices = valid_indices[valid_depth]
        coords = coords[valid_depth]
        measured_depth = measured_depth[valid_depth]
        output["valid_depth_points"] = int(len(valid_indices))

        points = self.points_xyz[valid_indices]
        hom_points = np.concatenate(
            [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
            axis=1,
        )
        projected_depth = (hom_points @ extrinsic.T)[:, 2].astype(np.float32)
        positive_projected = projected_depth > 0.0
        if not positive_projected.any():
            return output

        valid_indices = valid_indices[positive_projected]
        coords = coords[positive_projected]
        measured_depth = measured_depth[positive_projected]
        projected_depth = projected_depth[positive_projected]
        output["visible_points"] = int(len(valid_indices))
        if len(valid_indices) == 0:
            return output

        depth_error = np.abs(projected_depth - measured_depth)
        tolerance = np.minimum(0.10, np.maximum(0.04, 0.03 * measured_depth))
        depth_consistent = depth_error <= tolerance
        depth_conflict = ~depth_consistent

        xs_mask, ys_mask = _projected_mask_coords(coords, self.scaling_params)
        valid_mask = (xs_mask >= 0) & (xs_mask < sam_mask.shape[1]) & (ys_mask >= 0) & (ys_mask < sam_mask.shape[0])
        inside_mask = np.zeros((len(valid_indices),), dtype=bool)
        if valid_mask.any():
            inside_mask[valid_mask] = np.asarray(sam_mask, dtype=bool)[ys_mask[valid_mask], xs_mask[valid_mask]]

        inside_points = int(inside_mask.sum())
        inside_consistent = int(np.logical_and(inside_mask, depth_consistent).sum())
        inside_conflict = int(np.logical_and(inside_mask, depth_conflict).sum())
        outside_visible_points = int(max(0, len(valid_indices) - inside_points))
        output.update(
            {
                "inside_mask_points": inside_points,
                "inside_depth_consistent_points": inside_consistent,
                "inside_depth_conflict_points": inside_conflict,
                "outside_visible_points": outside_visible_points,
                "depth_consistency_ratio": float(inside_consistent / max(1, inside_points)),
                "depth_conflict_ratio": float(inside_conflict / max(1, inside_points)),
                "inside_visible_ratio": float(inside_points / max(1, len(valid_indices))),
                "outside_visible_ratio": float(outside_visible_points / max(1, len(valid_indices))),
                "visible_ratio": float(len(valid_indices) / max(1, len(point_indices))),
                "visible_coverage_ratio": float(inside_consistent / max(1, len(valid_indices))),
                "median_depth_error": float(np.median(depth_error)) if len(depth_error) else 0.0,
                "p90_depth_error": float(np.percentile(depth_error, 90)) if len(depth_error) else 0.0,
            }
        )
        return output


def build_scene_superpoint_cache(
    processed_scene_path,
    points_xyz=None,
    adjacency_knn=12,
    adjacency_max_distance=0.08,
):
    scene = np.load(processed_scene_path, mmap_mode="r")
    if scene.ndim != 2 or scene.shape[1] < 10:
        raise ValueError(f"Processed scene missing expected columns: {processed_scene_path}")

    raw_labels = scene[:, 9].astype(np.int64)
    unique_labels, labels = np.unique(raw_labels, return_inverse=True)
    labels = labels.astype(np.int32)
    num_segments = int(len(unique_labels))

    scene_points = scene[:, :3].astype(np.float32)
    point_order_matches = False
    if points_xyz is not None and len(points_xyz) == len(scene_points):
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        point_order_matches = bool(np.allclose(points_xyz, scene_points, atol=1e-4))
        if point_order_matches:
            points = points_xyz
        else:
            points = scene_points
    else:
        points = scene_points

    colors = _normalize_rgb(scene[:, 3:6])
    normals = _normalize_normals(scene[:, 6:9])

    order = np.argsort(labels, kind="mergesort")
    segment_sizes = np.bincount(labels, minlength=num_segments).astype(np.int32)
    segment_offsets = np.zeros((num_segments + 1,), dtype=np.int64)
    segment_offsets[1:] = np.cumsum(segment_sizes, dtype=np.int64)

    centers = np.zeros((num_segments, 3), dtype=np.float32)
    bbox_min = np.zeros((num_segments, 3), dtype=np.float32)
    bbox_max = np.zeros((num_segments, 3), dtype=np.float32)
    mean_colors = np.zeros((num_segments, 3), dtype=np.float32)
    mean_normals = np.zeros((num_segments, 3), dtype=np.float32)
    planarity = np.zeros((num_segments,), dtype=np.float32)

    segment_records = []
    for segment_id in range(num_segments):
        start = int(segment_offsets[segment_id])
        end = int(segment_offsets[segment_id + 1])
        indices = order[start:end]
        segment_points = points[indices]
        segment_colors = colors[indices]
        segment_normals = normals[indices]
        centers[segment_id] = segment_points.mean(axis=0)
        bbox_min[segment_id] = segment_points.min(axis=0)
        bbox_max[segment_id] = segment_points.max(axis=0)
        mean_colors[segment_id] = segment_colors.mean(axis=0)
        normal_mean = segment_normals.mean(axis=0)
        norm = float(np.linalg.norm(normal_mean))
        if norm > 1e-6:
            normal_mean = normal_mean / norm
        mean_normals[segment_id] = normal_mean.astype(np.float32)
        if len(segment_points) >= 3:
            covariance = np.cov(segment_points.T)
            eigenvalues = np.sort(np.linalg.eigvalsh(covariance).astype(np.float32))
            denom = float(max(eigenvalues.sum(), 1e-6))
            planarity[segment_id] = float(max(0.0, eigenvalues[1] - eigenvalues[0]) / denom)
        segment_records.append(
            {
                "segment_id": int(segment_id),
                "point_count": int(segment_sizes[segment_id]),
                "center_xyz": [float(value) for value in centers[segment_id].tolist()],
                "bbox_min_xyz": [float(value) for value in bbox_min[segment_id].tolist()],
                "bbox_max_xyz": [float(value) for value in bbox_max[segment_id].tolist()],
                "mean_color": [float(value) for value in mean_colors[segment_id].tolist()],
                "mean_normal": [float(value) for value in mean_normals[segment_id].tolist()],
                "planarity": float(planarity[segment_id]),
            }
        )

    neighbor_k = int(max(1, min(int(adjacency_knn), max(1, len(points) - 1))))
    adjacency_max_distance = (
        None if adjacency_max_distance is None else float(max(0.0, adjacency_max_distance))
    )
    pair_stats = defaultdict(lambda: {"contact_count": 0, "distance_sum": 0.0, "normal_sum": 0.0, "color_sum": 0.0})
    if neighbor_k > 0 and len(points) > 1:
        tree = cKDTree(points)
        distances, neighbors = tree.query(points, k=neighbor_k + 1, workers=-1)
        neighbors = np.asarray(neighbors)[:, 1:]
        distances = np.asarray(distances, dtype=np.float32)[:, 1:]
        left = np.repeat(np.arange(len(points), dtype=np.int32), neighbor_k)
        right = neighbors.reshape(-1).astype(np.int32)
        dist = distances.reshape(-1).astype(np.float32)
        valid = (right >= 0) & (right < len(points)) & (left != right)
        if adjacency_max_distance is not None:
            valid &= dist <= float(adjacency_max_distance)
        left = left[valid]
        right = right[valid]
        dist = dist[valid]
        if len(left) > 0:
            undirected_left = np.minimum(left, right).astype(np.int64, copy=False)
            undirected_right = np.maximum(left, right).astype(np.int64, copy=False)
            packed = undirected_left * np.int64(len(points)) + undirected_right
            distance_order = np.argsort(dist, kind="mergesort")
            packed = packed[distance_order]
            left = undirected_left[distance_order].astype(np.int32, copy=False)
            right = undirected_right[distance_order].astype(np.int32, copy=False)
            dist = dist[distance_order]
            _, unique_indices = np.unique(packed, return_index=True)
            left = left[unique_indices]
            right = right[unique_indices]
            dist = dist[unique_indices]
        left_segments = labels[left]
        right_segments = labels[right]
        cross_segment = left_segments != right_segments
        left = left[cross_segment]
        right = right[cross_segment]
        dist = dist[cross_segment]
        left_segments = left_segments[cross_segment]
        right_segments = right_segments[cross_segment]
        if len(left) > 0:
            normal_diff = 1.0 - np.abs(np.sum(normals[left] * normals[right], axis=1))
            color_diff = np.linalg.norm(colors[left] - colors[right], axis=1) / np.sqrt(3.0)
            for idx in range(len(left)):
                seg_left = int(left_segments[idx])
                seg_right = int(right_segments[idx])
                pair = (seg_left, seg_right) if seg_left < seg_right else (seg_right, seg_left)
                stats = pair_stats[pair]
                stats["contact_count"] += 1
                stats["distance_sum"] += float(dist[idx])
                stats["normal_sum"] += float(normal_diff[idx])
                stats["color_sum"] += float(color_diff[idx])

    adjacency_records = []
    segment_neighbors = defaultdict(list)
    for (seg_left, seg_right), stats in sorted(pair_stats.items()):
        count = int(stats["contact_count"])
        if count <= 0:
            continue
        record = {
            "left_segment_id": int(seg_left),
            "right_segment_id": int(seg_right),
            "boundary_contact_count": count,
            "mean_boundary_distance": float(stats["distance_sum"] / count),
            "mean_normal_difference": float(stats["normal_sum"] / count),
            "mean_color_difference": float(stats["color_sum"] / count),
        }
        adjacency_records.append(record)
        segment_neighbors[int(seg_left)].append(int(seg_right))
        segment_neighbors[int(seg_right)].append(int(seg_left))

    summary = {
        "version": SUPERPOINT_CACHE_VERSION,
        "processed_scene_path": osp.abspath(processed_scene_path),
        "point_count": int(len(points)),
        "superpoint_count": int(num_segments),
        "adjacency_edge_count": int(len(adjacency_records)),
        "point_order_matches_scene_points": bool(point_order_matches),
        "raw_superpoint_min": int(raw_labels.min(initial=0)),
        "raw_superpoint_max": int(raw_labels.max(initial=0)),
        "contiguous_ids": bool(num_segments == 0 or (labels.min(initial=0) == 0 and labels.max(initial=0) == num_segments - 1)),
        "adjacency_knn": int(neighbor_k),
        "adjacency_max_distance": adjacency_max_distance,
        "median_superpoint_size": float(np.median(segment_sizes)) if len(segment_sizes) else 0.0,
        "p90_superpoint_size": float(np.percentile(segment_sizes, 90)) if len(segment_sizes) else 0.0,
        "max_superpoint_size": int(segment_sizes.max(initial=0)),
    }

    return {
        "summary": summary,
        "scene_points": points,
        "labels": labels,
        "segment_sizes": segment_sizes,
        "segment_order": order.astype(np.int64),
        "segment_offsets": segment_offsets.astype(np.int64),
        "segment_records": segment_records,
        "adjacency_records": adjacency_records,
        "segment_neighbors": {int(key): sorted(set(value)) for key, value in segment_neighbors.items()},
        "centers": centers,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "mean_colors": mean_colors,
        "mean_normals": mean_normals,
        "planarity": planarity,
    }


def save_scene_superpoint_cache(scene_cache, scene_dir):
    superpoint_dir = osp.join(scene_dir, "superpoint_cache")
    os.makedirs(superpoint_dir, exist_ok=True)
    npz_path = osp.join(superpoint_dir, "superpoint_cache.npz")
    json_path = osp.join(superpoint_dir, "superpoint_cache_summary.json")

    adjacency_left = np.asarray(
        [item["left_segment_id"] for item in scene_cache["adjacency_records"]],
        dtype=np.int32,
    )
    adjacency_right = np.asarray(
        [item["right_segment_id"] for item in scene_cache["adjacency_records"]],
        dtype=np.int32,
    )
    adjacency_contact = np.asarray(
        [item["boundary_contact_count"] for item in scene_cache["adjacency_records"]],
        dtype=np.int32,
    )
    adjacency_distance = np.asarray(
        [item["mean_boundary_distance"] for item in scene_cache["adjacency_records"]],
        dtype=np.float32,
    )
    adjacency_normal = np.asarray(
        [item["mean_normal_difference"] for item in scene_cache["adjacency_records"]],
        dtype=np.float32,
    )
    adjacency_color = np.asarray(
        [item["mean_color_difference"] for item in scene_cache["adjacency_records"]],
        dtype=np.float32,
    )

    np.savez_compressed(
        npz_path,
        scene_points=scene_cache["scene_points"].astype(np.float32),
        point_superpoint_ids=scene_cache["labels"].astype(np.int32),
        superpoint_ids=np.arange(len(scene_cache["segment_sizes"]), dtype=np.int32),
        superpoint_point_offsets=scene_cache["segment_offsets"].astype(np.int64),
        superpoint_point_indices=scene_cache["segment_order"].astype(np.int64),
        superpoint_point_counts=scene_cache["segment_sizes"].astype(np.int32),
        superpoint_centers=scene_cache["centers"].astype(np.float32),
        superpoint_bbox_min=scene_cache["bbox_min"].astype(np.float32),
        superpoint_bbox_max=scene_cache["bbox_max"].astype(np.float32),
        superpoint_mean_colors=scene_cache["mean_colors"].astype(np.float32),
        superpoint_mean_normals=scene_cache["mean_normals"].astype(np.float32),
        superpoint_planarity=scene_cache["planarity"].astype(np.float32),
        adjacency_left=adjacency_left,
        adjacency_right=adjacency_right,
        adjacency_boundary_contact_count=adjacency_contact,
        adjacency_mean_boundary_distance=adjacency_distance,
        adjacency_mean_normal_difference=adjacency_normal,
        adjacency_mean_color_difference=adjacency_color,
    )

    with open(json_path, "w") as f:
        json.dump(
            {
                **scene_cache["summary"],
                "cache_npz_path": npz_path,
                "segments": scene_cache["segment_records"],
                "adjacency": scene_cache["adjacency_records"],
            },
            f,
            indent=2,
        )
    return {
        "cache_npz_path": npz_path,
        "cache_summary_path": json_path,
    }


def load_scene_superpoint_cache(scene_dir, points_xyz=None):
    superpoint_dir = osp.join(scene_dir, "superpoint_cache")
    npz_path = osp.join(superpoint_dir, "superpoint_cache.npz")
    json_path = osp.join(superpoint_dir, "superpoint_cache_summary.json")
    if not osp.exists(npz_path) or not osp.exists(json_path):
        return None

    with open(json_path) as f:
        payload = json.load(f)
    if payload.get("version") != SUPERPOINT_CACHE_VERSION:
        return None

    cache_npz = np.load(npz_path, allow_pickle=False)
    scene_points = cache_npz["scene_points"].astype(np.float32)
    point_order_matches = bool(payload.get("point_order_matches_scene_points", False))
    if points_xyz is not None and len(points_xyz) == len(scene_points):
        point_order_matches = bool(np.allclose(np.asarray(points_xyz, dtype=np.float32), scene_points, atol=1e-4))

    segment_neighbors = defaultdict(list)
    adjacency_records = payload.get("adjacency", [])
    for item in adjacency_records:
        left = int(item["left_segment_id"])
        right = int(item["right_segment_id"])
        segment_neighbors[left].append(right)
        segment_neighbors[right].append(left)

    summary = dict(payload)
    summary["point_order_matches_scene_points"] = bool(point_order_matches)
    return {
        "summary": summary,
        "scene_points": scene_points,
        "labels": cache_npz["point_superpoint_ids"].astype(np.int32),
        "segment_sizes": cache_npz["superpoint_point_counts"].astype(np.int32),
        "segment_order": cache_npz["superpoint_point_indices"].astype(np.int64),
        "segment_offsets": cache_npz["superpoint_point_offsets"].astype(np.int64),
        "segment_records": payload.get("segments", []),
        "adjacency_records": adjacency_records,
        "segment_neighbors": {int(key): sorted(set(value)) for key, value in segment_neighbors.items()},
        "centers": cache_npz["superpoint_centers"].astype(np.float32),
        "bbox_min": cache_npz["superpoint_bbox_min"].astype(np.float32),
        "bbox_max": cache_npz["superpoint_bbox_max"].astype(np.float32),
        "mean_colors": cache_npz["superpoint_mean_colors"].astype(np.float32),
        "mean_normals": cache_npz["superpoint_mean_normals"].astype(np.float32),
        "planarity": cache_npz["superpoint_planarity"].astype(np.float32),
        "cache_npz_path": npz_path,
        "cache_summary_path": json_path,
    }


def load_or_build_scene_superpoint_cache(
    scene_dir,
    processed_scene_path,
    points_xyz=None,
    adjacency_knn=12,
    adjacency_max_distance=0.08,
):
    cached = load_scene_superpoint_cache(scene_dir, points_xyz=points_xyz)
    if cached is not None:
        summary = cached["summary"]
        expected_scene_path = osp.abspath(processed_scene_path)
        if (
            summary.get("processed_scene_path") == expected_scene_path
            and int(summary.get("adjacency_knn", -1)) == int(adjacency_knn)
            and (
                (summary.get("adjacency_max_distance") is None and adjacency_max_distance is None)
                or float(summary.get("adjacency_max_distance", -1.0)) == float(adjacency_max_distance)
            )
        ):
            summary["cache_reused"] = True
            return cached

    built = build_scene_superpoint_cache(
        processed_scene_path,
        points_xyz=points_xyz,
        adjacency_knn=adjacency_knn,
        adjacency_max_distance=adjacency_max_distance,
    )
    built["summary"]["cache_reused"] = False
    scene_cache_paths = save_scene_superpoint_cache(built, scene_dir)
    built.update(scene_cache_paths)
    return built


def classify_observation_superpoint_support(
    scene_cache,
    observation,
    frame_projector,
    strong_support_min_coverage=0.60,
    partial_support_min_coverage=0.30,
    min_valid_visible_points=20,
    strong_support_min_depth_consistency=0.70,
    strong_reject_min_depth_conflict=0.60,
    strong_reject_min_inside_points=20,
    strong_reject_min_conflict_points=20,
    outside_reject_min_visible_points=20,
    outside_reject_max_inside_ratio=0.10,
    outside_reject_min_outside_ratio=0.90,
):
    sam_mask = observation.get("_sam_mask")
    if sam_mask is None:
        return {
            "graph_observation_id": int(observation.get("graph_observation_id", -1)),
            "frame_index": int(observation.get("frame_index", -1)),
            "class_id": int(observation.get("class_id", -1)),
            "class_name": observation.get("class_name"),
            "candidate_superpoint_ids": [],
            "superpoint_evidence": [],
            "summary": {"enabled": False, "reason": "missing_sam_mask"},
        }

    labels = scene_cache["labels"]
    seed_indices = np.asarray(observation.get("_seed_indices", []), dtype=np.int64)
    touched = set(int(item) for item in np.unique(labels[seed_indices]).tolist()) if len(seed_indices) else set()
    candidate_segments = set(touched)
    neighbors = scene_cache["segment_neighbors"]
    for segment_id in list(touched):
        candidate_segments.update(neighbors.get(int(segment_id), []))

    evidence_records = []
    summary_counts = defaultdict(int)
    frame_index = int(observation.get("frame_index", -1))
    segment_order = scene_cache["segment_order"]
    segment_offsets = scene_cache["segment_offsets"]
    segment_sizes = scene_cache["segment_sizes"]
    for segment_id in sorted(candidate_segments):
        start = int(segment_offsets[segment_id])
        end = int(segment_offsets[segment_id + 1])
        point_indices = segment_order[start:end]
        metrics = frame_projector.segment_observation_metrics(point_indices, frame_index, sam_mask)
        metrics["full_coverage_ratio"] = float(
            metrics["inside_depth_consistent_points"] / max(1, int(segment_sizes[segment_id]))
        )
        metrics["coverage_ratio"] = float(metrics["full_coverage_ratio"])
        if (
            (
                metrics["full_coverage_ratio"] >= float(strong_support_min_coverage)
                or float(metrics["visible_coverage_ratio"]) >= float(strong_support_min_coverage)
            )
            and int(metrics["inside_depth_consistent_points"]) >= int(min_valid_visible_points)
            and float(metrics["depth_consistency_ratio"]) >= float(strong_support_min_depth_consistency)
        ):
            label = "strong_support"
        elif (
            int(metrics["inside_mask_points"]) >= int(strong_reject_min_inside_points)
            and int(metrics["inside_depth_conflict_points"]) >= int(strong_reject_min_conflict_points)
            and float(metrics["depth_conflict_ratio"]) >= float(strong_reject_min_depth_conflict)
        ):
            label = "depth_conflict_reject"
        elif (
            segment_id not in touched
            and int(metrics["visible_points"]) >= int(outside_reject_min_visible_points)
            and float(metrics["inside_visible_ratio"]) <= float(outside_reject_max_inside_ratio)
            and float(metrics["outside_visible_ratio"]) >= float(outside_reject_min_outside_ratio)
        ):
            label = "outside_mask_reject"
        elif (
            metrics["full_coverage_ratio"] >= float(partial_support_min_coverage)
            or float(metrics["visible_coverage_ratio"]) >= float(partial_support_min_coverage)
        ):
            label = "partial_support"
        elif segment_id in touched:
            label = "touched_only"
        else:
            label = "no_support"
        record = {
            "segment_id": int(segment_id),
            "support_label": label,
            "touched_by_seed": bool(segment_id in touched),
            "point_count": int(segment_sizes[segment_id]),
            **{key: _pythonize(value) for key, value in metrics.items()},
        }
        evidence_records.append(record)
        summary_counts[label] += 1

    return {
        "graph_observation_id": int(observation.get("graph_observation_id", -1)),
        "frame_index": frame_index,
        "class_id": int(observation.get("class_id", -1)),
        "class_name": observation.get("class_name"),
        "candidate_superpoint_ids": [int(item) for item in sorted(candidate_segments)],
        "superpoint_evidence": evidence_records,
        "summary": {
            "enabled": True,
            "touched_superpoint_count": int(len(touched)),
            "candidate_superpoint_count": int(len(candidate_segments)),
            "strong_support_count": int(summary_counts.get("strong_support", 0)),
            "partial_support_count": int(summary_counts.get("partial_support", 0)),
            "depth_conflict_reject_count": int(summary_counts.get("depth_conflict_reject", 0)),
            "outside_mask_reject_count": int(summary_counts.get("outside_mask_reject", 0)),
            "strong_reject_count": int(
                summary_counts.get("depth_conflict_reject", 0) + summary_counts.get("outside_mask_reject", 0)
            ),
            "touched_only_count": int(summary_counts.get("touched_only", 0)),
        },
    }


def save_observation_superpoint_evidence(evidence_items, scene_dir):
    evidence_dir = osp.join(scene_dir, "superpoint_observation_evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    summary_records = []
    for item in evidence_items:
        observation_id = int(item.get("graph_observation_id", len(summary_records)))
        path = osp.join(evidence_dir, f"observation{observation_id:05d}_superpoints.json")
        with open(path, "w") as f:
            json.dump(item, f, indent=2)
        summary_records.append(
            {
                "graph_observation_id": observation_id,
                "frame_index": int(item.get("frame_index", -1)),
                "class_name": item.get("class_name"),
                "path": path,
                "summary": item.get("summary", {}),
            }
        )
    summary_path = osp.join(evidence_dir, "observation_superpoint_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_records, f, indent=2)
    return {
        "observation_superpoint_summary_path": summary_path,
        "observation_superpoint_items": summary_records,
    }


def summarize_candidate_superpoints(scene_cache, candidate, observation_evidence_by_id):
    selected_observation_ids = [
        int(item)
        for item in candidate.get("graph_selected_observation_ids", [])
        if str(item).strip() != ""
    ]
    all_observation_ids = [
        int(item)
        for item in candidate.get("graph_cluster_observation_ids", [])
        if str(item).strip() != ""
    ]
    if not selected_observation_ids and not all_observation_ids:
        return {"enabled": False, "reason": "missing_candidate_observations"}

    neighbors = scene_cache["segment_neighbors"]
    segment_sizes = scene_cache["segment_sizes"]

    def aggregate_observations(observation_ids):
        aggregate = defaultdict(
            lambda: {
                "strong_support_views": 0,
                "partial_support_views": 0,
                "depth_conflict_reject_views": 0,
                "outside_mask_reject_views": 0,
                "touched_views": 0,
                "best_full_coverage_ratio": 0.0,
                "best_visible_coverage_ratio": 0.0,
                "best_depth_consistency_ratio": 0.0,
                "observation_ids": [],
            }
        )
        for observation_id in observation_ids:
            evidence = observation_evidence_by_id.get(int(observation_id))
            if evidence is None:
                continue
            for record in evidence.get("superpoint_evidence", []):
                segment_id = int(record["segment_id"])
                stats = aggregate[segment_id]
                label = str(record.get("support_label", ""))
                if label == "strong_support":
                    stats["strong_support_views"] += 1
                elif label == "partial_support":
                    stats["partial_support_views"] += 1
                elif label == "depth_conflict_reject":
                    stats["depth_conflict_reject_views"] += 1
                elif label == "outside_mask_reject":
                    stats["outside_mask_reject_views"] += 1
                if bool(record.get("touched_by_seed", False)):
                    stats["touched_views"] += 1
                stats["best_full_coverage_ratio"] = max(
                    stats["best_full_coverage_ratio"],
                    float(record.get("full_coverage_ratio", record.get("coverage_ratio", 0.0))),
                )
                stats["best_visible_coverage_ratio"] = max(
                    stats["best_visible_coverage_ratio"],
                    float(record.get("visible_coverage_ratio", 0.0)),
                )
                stats["best_depth_consistency_ratio"] = max(
                    stats["best_depth_consistency_ratio"],
                    float(record.get("depth_consistency_ratio", 0.0)),
                )
                stats["observation_ids"].append(int(observation_id))
        return aggregate

    aggregate_all = aggregate_observations(all_observation_ids or selected_observation_ids)
    aggregate_selected = aggregate_observations(selected_observation_ids)
    if not aggregate_all:
        return {"enabled": False, "reason": "missing_observation_superpoint_evidence"}

    def reject_count(stats):
        return int(stats["depth_conflict_reject_views"]) + int(stats["outside_mask_reject_views"])

    core_segments = {
        int(segment_id)
        for segment_id, stats in aggregate_all.items()
        if int(stats["strong_support_views"]) >= 2 and reject_count(stats) == 0
    }
    boundary_segments = set()
    conflict_segments = set()
    unresolved_segments = set()
    for segment_id, stats in aggregate_all.items():
        segment_id = int(segment_id)
        if segment_id in core_segments:
            continue
        has_support = int(stats["strong_support_views"]) > 0 or int(stats["partial_support_views"]) > 0
        if reject_count(stats) > 0 and has_support:
            conflict_segments.add(segment_id)
            continue
        adjacent_to_core = any(int(neighbor) in core_segments for neighbor in neighbors.get(segment_id, []))
        if has_support and adjacent_to_core and reject_count(stats) == 0:
            boundary_segments.add(segment_id)
        elif has_support:
            unresolved_segments.add(segment_id)

    connected_core_edges = 0
    for segment_id in core_segments:
        for neighbor in neighbors.get(int(segment_id), []):
            if int(neighbor) in core_segments and int(neighbor) > int(segment_id):
                connected_core_edges += 1

    def point_total(segment_ids):
        return int(sum(int(segment_sizes[int(segment_id)]) for segment_id in segment_ids))

    def connected_components(segment_ids):
        remaining = set(int(item) for item in segment_ids)
        components = []
        while remaining:
            start = remaining.pop()
            queue = [start]
            component = {start}
            while queue:
                current = queue.pop()
                for neighbor in neighbors.get(int(current), []):
                    neighbor = int(neighbor)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    core_components = connected_components(core_segments)
    core_component_point_totals = [point_total(component) for component in core_components]
    total_core_points = point_total(core_segments)
    core_largest_component_ratio = (
        float(max(core_component_point_totals) / max(1, total_core_points)) if core_component_point_totals else 0.0
    )

    def aggregate_summary(aggregate):
        return {
            "supported_superpoint_count": int(len(aggregate)),
            "strong_support_superpoint_count": int(
                sum(1 for stats in aggregate.values() if int(stats["strong_support_views"]) > 0)
            ),
            "partial_support_superpoint_count": int(
                sum(1 for stats in aggregate.values() if int(stats["partial_support_views"]) > 0)
            ),
            "depth_conflict_reject_superpoint_count": int(
                sum(1 for stats in aggregate.values() if int(stats["depth_conflict_reject_views"]) > 0)
            ),
            "outside_mask_reject_superpoint_count": int(
                sum(1 for stats in aggregate.values() if int(stats["outside_mask_reject_views"]) > 0)
            ),
        }

    return {
        "enabled": True,
        "all_reliable_observation_count": int(len(all_observation_ids or selected_observation_ids)),
        "selected_observation_count": int(len(selected_observation_ids)),
        "supported_superpoint_count": int(len(aggregate_all)),
        "core_superpoint_count": int(len(core_segments)),
        "boundary_superpoint_count": int(len(boundary_segments)),
        "conflict_superpoint_count": int(len(conflict_segments)),
        "unresolved_superpoint_count": int(len(unresolved_segments)),
        "core_point_count": total_core_points,
        "boundary_point_count": point_total(boundary_segments),
        "conflict_point_count": point_total(conflict_segments),
        "unresolved_point_count": point_total(unresolved_segments),
        "core_internal_adjacency_edges": int(connected_core_edges),
        "core_connected_component_count": int(len(core_components)),
        "core_largest_component_ratio": float(core_largest_component_ratio),
        "all_observation_summary": aggregate_summary(aggregate_all),
        "selected_observation_summary": aggregate_summary(aggregate_selected),
        "core_superpoint_ids": [int(item) for item in sorted(core_segments)],
        "boundary_superpoint_ids": [int(item) for item in sorted(boundary_segments)],
        "conflict_superpoint_ids": [int(item) for item in sorted(conflict_segments)],
        "unresolved_superpoint_ids": [int(item) for item in sorted(unresolved_segments)],
    }

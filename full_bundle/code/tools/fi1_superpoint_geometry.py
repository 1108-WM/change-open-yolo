"""Raw superpoint normalization and contact geometry used by frozen D-v3."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree


def _normalize_rgb(colors):
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size and float(colors.max(initial=0.0)) > 1.5:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)

def _normalize_normals(normals):
    normals = np.asarray(normals, dtype=np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-6)

def _raw_superpoint_context(processed, adjacency_knn, adjacency_max_distance, min_contact_points, min_contact_ratio):
    """按原始 superpoint 编号构建局部接触图，绝不重编号或修改几何原子。"""
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError("处理点云缺少原始 superpoint 列")
    points = np.asarray(processed[:, :3], dtype=np.float32)
    raw_ids, inverse = np.unique(np.asarray(processed[:, 9], dtype=np.int64), return_inverse=True)
    inverse = inverse.astype(np.int64)
    sizes = np.bincount(inverse, minlength=len(raw_ids)).astype(np.int64)
    colors = _normalize_rgb(processed[:, 3:6])
    normals = _normalize_normals(processed[:, 6:9])
    centers = np.zeros((len(raw_ids), 3), dtype=np.float32)
    mean_colors = np.zeros((len(raw_ids), 3), dtype=np.float32)
    mean_normals = np.zeros((len(raw_ids), 3), dtype=np.float32)
    np.add.at(centers, inverse, points)
    np.add.at(mean_colors, inverse, colors)
    np.add.at(mean_normals, inverse, normals)
    centers /= np.maximum(sizes[:, None], 1)
    mean_colors /= np.maximum(sizes[:, None], 1)
    mean_normals /= np.maximum(sizes[:, None], 1)
    mean_normals = _normalize_normals(mean_normals)

    neighbors = defaultdict(list)
    if len(points) < 2:
        return {
            "raw_ids": raw_ids,
            "sizes": sizes,
            "centers": centers,
            "mean_colors": mean_colors,
            "mean_normals": mean_normals,
            "neighbors": neighbors,
        }
    neighbor_count = min(max(1, int(adjacency_knn)), len(points) - 1)
    tree = cKDTree(points)
    distances, point_neighbors = tree.query(points, k=neighbor_count + 1, workers=-1)
    left = np.repeat(np.arange(len(points), dtype=np.int64), neighbor_count)
    right = np.asarray(point_neighbors, dtype=np.int64)[:, 1:].reshape(-1)
    distances = np.asarray(distances, dtype=np.float32)[:, 1:].reshape(-1)
    valid = (left != right) & (distances <= float(adjacency_max_distance))
    left, right, distances = left[valid], right[valid], distances[valid]
    left_segments, right_segments = inverse[left], inverse[right]
    cross_segment = left_segments != right_segments
    left, right, distances = left[cross_segment], right[cross_segment], distances[cross_segment]
    left_segments, right_segments = left_segments[cross_segment], right_segments[cross_segment]
    if len(left_segments) == 0:
        return {
            "raw_ids": raw_ids,
            "sizes": sizes,
            "centers": centers,
            "mean_colors": mean_colors,
            "mean_normals": mean_normals,
            "neighbors": neighbors,
        }
    low, high = np.minimum(left_segments, right_segments), np.maximum(left_segments, right_segments)
    packed = low * len(raw_ids) + high
    unique_pairs, pair_inverse = np.unique(packed, return_inverse=True)
    contact_counts = np.bincount(pair_inverse, minlength=len(unique_pairs)).astype(np.int64)
    distance_sums = np.bincount(pair_inverse, weights=distances, minlength=len(unique_pairs))
    normal_differences = 1.0 - np.abs(np.sum(normals[left] * normals[right], axis=1))
    color_differences = np.linalg.norm(colors[left] - colors[right], axis=1) / np.sqrt(3.0)
    normal_sums = np.bincount(pair_inverse, weights=normal_differences, minlength=len(unique_pairs))
    color_sums = np.bincount(pair_inverse, weights=color_differences, minlength=len(unique_pairs))
    for packed_pair, count, distance_sum, normal_sum, color_sum in zip(
        unique_pairs, contact_counts, distance_sums, normal_sums, color_sums
    ):
        low_id, high_id = divmod(int(packed_pair), len(raw_ids))
        contact_ratio = float(count / max(1, min(sizes[low_id], sizes[high_id])))
        if int(count) < int(min_contact_points) or contact_ratio < float(min_contact_ratio):
            continue
        record = {
            "neighbor_superpoint_id": int(raw_ids[high_id]),
            "boundary_contact_count": int(count),
            "boundary_contact_ratio": contact_ratio,
            "mean_boundary_distance": float(distance_sum / count),
            "mean_normal_difference": float(normal_sum / count),
            "mean_color_difference": float(color_sum / count),
        }
        reverse = {**record, "neighbor_superpoint_id": int(raw_ids[low_id])}
        neighbors[int(raw_ids[low_id])].append(record)
        neighbors[int(raw_ids[high_id])].append(reverse)
    return {
        "raw_ids": raw_ids,
        "sizes": sizes,
        "centers": centers,
        "mean_colors": mean_colors,
        "mean_normals": mean_normals,
        "neighbors": {key: sorted(value, key=lambda item: item["neighbor_superpoint_id"]) for key, value in neighbors.items()},
    }

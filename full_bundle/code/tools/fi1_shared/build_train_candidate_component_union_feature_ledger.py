"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from pathlib import Path

import numpy as np

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

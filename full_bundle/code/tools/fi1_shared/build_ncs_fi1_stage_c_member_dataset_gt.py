"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

import math

import numpy as np

from tools.fi1_superpoint_geometry import _normalize_normals, _normalize_rgb

ROLES = ("shared", "track_only", "native_only")

ADJACENCY = {
    "knn": 12,
    "max_distance": 0.05,
    "min_contact_points": 3,
    "min_contact_ratio": 0.02,
}

RELATION_FEATURES = (
    "point_iou",
    "native_inside_track_ratio",
    "track_inside_native_ratio",
    "aabb_iou",
    "centroid_distance_normalized",
    "native_shared_superpoint_fraction",
    "track_shared_superpoint_fraction",
    "public_common_selected_view_count",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "public_projected_box_iou_mean",
    "public_native_gvc_mean",
    "public_track_gvc_mean",
    "public_track_minus_native_gvc",
    "public_same_observation_both_depth_consistent_count",
    "exclusive_boundary_contact_ratio_mean",
    "exclusive_boundary_distance_weighted_mean",
    "exclusive_boundary_normal_difference_weighted_mean",
    "exclusive_boundary_color_difference_weighted_mean",
    "mean_rgb_distance",
    "mean_normal_difference",
)

FEATURE_NAMES = (
    "role_shared",
    "role_track_only",
    "role_native_only",
    "log1p_atom_point_count",
    "atom_fraction_of_union",
    "atom_fraction_of_track",
    "atom_fraction_of_native",
    "atom_fraction_of_raw_superpoint",
    "atom_centroid_distance_to_union_normalized",
    "atom_rgb_distance_to_union",
    "atom_normal_difference_to_union",
    "raw_superpoint_degree",
    "raw_superpoint_boundary_contact_count",
    "raw_superpoint_boundary_contact_ratio_mean",
    "raw_superpoint_boundary_distance_mean",
    "raw_superpoint_boundary_color_difference_mean",
    "raw_superpoint_boundary_normal_difference_mean",
    "stage_a_union_quality",
    "stage_a_track_quality",
    "stage_a_native_quality",
    "stage_b_union_score",
    *tuple(f"relation__{name}" for name in RELATION_FEATURES),
)

def _geometry_sha256(points: np.ndarray) -> str:
    values = np.unique(np.asarray(points, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()

def decompose_union_atoms(
    union_points: np.ndarray,
    track_points: np.ndarray,
    native_points: np.ndarray,
    raw_superpoint_ids: np.ndarray,
) -> list[dict]:
    """Partition one union into disjoint raw-superpoint-by-role atoms."""
    union = np.unique(np.asarray(union_points, dtype=np.int64))
    track = np.unique(np.asarray(track_points, dtype=np.int64))
    native = np.unique(np.asarray(native_points, dtype=np.int64))
    raw = np.asarray(raw_superpoint_ids, dtype=np.int64)
    if len(raw) <= int(union[-1]):
        raise ValueError("raw superpoint array does not cover union points")
    if not np.array_equal(union, np.union1d(track, native)):
        raise ValueError("frozen union differs from the union of its parents")
    in_track = np.zeros(len(raw), dtype=bool)
    in_native = np.zeros(len(raw), dtype=bool)
    in_track[track] = True
    in_native[native] = True
    roles = np.full(len(raw), -1, dtype=np.int8)
    roles[in_track & in_native] = 0
    roles[in_track & ~in_native] = 1
    roles[~in_track & in_native] = 2
    atoms = []
    for raw_id in np.unique(raw[union]):
        local = union[raw[union] == raw_id]
        for role_index, role in enumerate(ROLES):
            points = local[roles[local] == role_index]
            if len(points):
                atoms.append({
                    "raw_superpoint_id": int(raw_id),
                    "role": role,
                    "points": points,
                })
    concatenated = np.concatenate([row["points"] for row in atoms]) if atoms else np.empty(0, np.int64)
    if len(concatenated) != len(union) or not np.array_equal(np.sort(concatenated), union):
        raise AssertionError("atom partition does not conserve the frozen union")
    return atoms

def _mean_edge(edges: list[dict], name: str) -> float:
    return float(np.mean([float(row[name]) for row in edges])) if edges else 0.0

def _atom_features(
    atom: dict,
    union: np.ndarray,
    track: np.ndarray,
    native: np.ndarray,
    processed: np.ndarray,
    context: dict,
    qualities: dict,
    relation: dict,
) -> dict[str, float]:
    points = atom["points"]
    raw_id = int(atom["raw_superpoint_id"])
    raw_ids = np.asarray(processed[:, 9], dtype=np.int64)
    raw_size = int(np.count_nonzero(raw_ids == raw_id))
    xyz = np.asarray(processed[:, :3], dtype=np.float64)
    rgb = _normalize_rgb(processed[:, 3:6])
    normals = _normalize_normals(processed[:, 6:9])
    union_extent = xyz[union].max(axis=0) - xyz[union].min(axis=0)
    scale = max(1e-8, float(np.linalg.norm(union_extent)))
    atom_normal = _normalize_normals(normals[points].mean(axis=0, keepdims=True))[0]
    union_normal = _normalize_normals(normals[union].mean(axis=0, keepdims=True))[0]
    edges = list(context["neighbors"].get(raw_id, []))
    relation_raw = relation["features"]
    values = {
        "role_shared": float(atom["role"] == "shared"),
        "role_track_only": float(atom["role"] == "track_only"),
        "role_native_only": float(atom["role"] == "native_only"),
        "log1p_atom_point_count": math.log1p(len(points)),
        "atom_fraction_of_union": len(points) / max(1, len(union)),
        "atom_fraction_of_track": len(points) / max(1, len(track)),
        "atom_fraction_of_native": len(points) / max(1, len(native)),
        "atom_fraction_of_raw_superpoint": len(points) / max(1, raw_size),
        "atom_centroid_distance_to_union_normalized": float(
            np.linalg.norm(xyz[points].mean(axis=0) - xyz[union].mean(axis=0)) / scale
        ),
        "atom_rgb_distance_to_union": float(
            np.linalg.norm(rgb[points].mean(axis=0) - rgb[union].mean(axis=0)) / math.sqrt(3.0)
        ),
        "atom_normal_difference_to_union": float(
            1.0 - abs(float(np.dot(atom_normal, union_normal)))
        ),
        "raw_superpoint_degree": float(len(edges)),
        "raw_superpoint_boundary_contact_count": float(sum(
            int(row["boundary_contact_count"]) for row in edges
        )),
        "raw_superpoint_boundary_contact_ratio_mean": _mean_edge(edges, "boundary_contact_ratio"),
        "raw_superpoint_boundary_distance_mean": _mean_edge(edges, "mean_boundary_distance"),
        "raw_superpoint_boundary_color_difference_mean": _mean_edge(edges, "mean_color_difference"),
        "raw_superpoint_boundary_normal_difference_mean": _mean_edge(edges, "mean_normal_difference"),
        **qualities,
        **{
            f"relation__{name}": float(relation_raw.get(name, 0.0))
            for name in RELATION_FEATURES
        },
    }
    output = {name: float(values[name]) for name in FEATURE_NAMES}
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("non-finite stage-C atom feature")
    return output

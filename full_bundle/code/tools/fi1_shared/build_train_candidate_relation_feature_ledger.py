"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

from collections import defaultdict, deque

import numpy as np

from tools.build_c1_gvc_paper_reference_quality_ledger import _box_iou, _summary

ADJACENCY_KNN = 12

ADJACENCY_MAX_DISTANCE = 0.05

MIN_CONTACT_POINTS = 3

MIN_CONTACT_RATIO = 0.02

MAX_PUBLIC_COMMON_VIEWS = 10

MIN_VISIBLE_POINTS = 30

EXCLUSIVE_PAIR_MODEL_FEATURES = (
    "track_exclusive_fraction_of_track",
    "native_exclusive_fraction_of_native",
    "track_exclusive_fraction_of_pair_exclusive",
    "native_exclusive_fraction_of_pair_exclusive",
    "exclusive_point_balance_log_track_over_native",
    "track_exclusive_empty",
    "native_exclusive_empty",
    "both_exclusive_nonempty",
    "track_exclusive_public_eligible_view_count",
    "track_exclusive_public_selected_view_count",
    "track_exclusive_public_matched_view_count",
    "track_exclusive_public_zero_support_view_fraction",
    "track_exclusive_public_gvc_mean",
    "track_exclusive_public_gvc_variance",
    "track_exclusive_public_mask_support_mean",
    "track_exclusive_public_mask_support_min",
    "native_exclusive_public_eligible_view_count",
    "native_exclusive_public_selected_view_count",
    "native_exclusive_public_matched_view_count",
    "native_exclusive_public_zero_support_view_fraction",
    "native_exclusive_public_gvc_mean",
    "native_exclusive_public_gvc_variance",
    "native_exclusive_public_mask_support_mean",
    "native_exclusive_public_mask_support_min",
    "exclusive_public_common_eligible_view_count",
    "exclusive_public_common_selected_view_count",
    "exclusive_public_common_view_available",
    "exclusive_public_projected_box_iou_mean",
    "exclusive_public_projected_box_iou_min",
    "exclusive_public_same_matched_observation_fraction",
    "exclusive_public_different_matched_observation_fraction",
    "exclusive_public_gvc_margin_track_minus_native_mean",
    "exclusive_public_gvc_margin_track_minus_native_min",
    "exclusive_public_gvc_margin_track_minus_native_max",
    "exclusive_public_gvc_margin_track_minus_native_variance",
    "exclusive_public_track_gvc_win_fraction",
    "exclusive_public_native_gvc_win_fraction",
    "exclusive_public_gvc_order_consistency",
)

def _normalise_rgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size and float(values.max(initial=0.0)) > 1.5:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)

def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(lengths, 1e-8)

def _component_count(superpoint_ids: set[int], neighbors: dict[int, list[dict]]) -> int:
    unseen = set(superpoint_ids)
    count = 0
    while unseen:
        count += 1
        queue = deque([unseen.pop()])
        while queue:
            current = queue.popleft()
            for edge in neighbors.get(current, []):
                other = int(edge["neighbor_superpoint_id"])
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
    return count

def _candidate_stats(
    points: np.ndarray,
    processed: np.ndarray,
    context: dict,
) -> dict:
    if not len(points):
        raise ValueError("candidate point set must be non-empty")
    if points[0] < 0 or points[-1] >= len(processed):
        raise ValueError("candidate point set is outside processed scene")
    xyz = np.asarray(processed[points, :3], dtype=np.float64)
    rgb = _normalise_rgb(processed[points, 3:6])
    normals = _normalise_rows(processed[points, 6:9])
    raw_superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(raw_superpoints[points], return_counts=True)
    size_by_id = {
        int(raw_id): int(size)
        for raw_id, size in zip(context["raw_ids"], context["sizes"])
    }
    occupancies = np.asarray([
        count / max(1, size_by_id[int(raw_id)]) for raw_id, count in zip(ids, counts)
    ], dtype=np.float64)
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    extent = upper - lower
    mean_normal = normals.mean(axis=0)
    mean_normal /= max(1e-8, float(np.linalg.norm(mean_normal)))
    superpoint_ids = set(map(int, ids))
    return {
        "points": points,
        "point_count": int(len(points)),
        "centroid": xyz.mean(axis=0),
        "bbox_min": lower,
        "bbox_max": upper,
        "bbox_extent": extent,
        "bbox_diagonal": float(np.linalg.norm(extent)),
        "bbox_volume": float(np.prod(extent)),
        "mean_rgb": rgb.mean(axis=0),
        "rgb_std": rgb.std(axis=0),
        "mean_normal": mean_normal,
        "normal_coherence": float(np.linalg.norm(normals.mean(axis=0))),
        "superpoint_ids": superpoint_ids,
        "superpoint_count": len(superpoint_ids),
        "superpoint_full_occupancy_fraction": float(np.mean(occupancies >= 1.0 - 1e-8)),
        "superpoint_mean_occupancy": float(occupancies.mean()),
        "superpoint_component_count": _component_count(superpoint_ids, context["neighbors"]),
    }

def _aabb_metrics(left: dict, right: dict) -> dict:
    intersection_extent = np.maximum(
        0.0, np.minimum(left["bbox_max"], right["bbox_max"])
        - np.maximum(left["bbox_min"], right["bbox_min"]),
    )
    intersection = float(np.prod(intersection_extent))
    left_volume, right_volume = left["bbox_volume"], right["bbox_volume"]
    union = left_volume + right_volume - intersection
    return {
        "aabb_intersection_volume": intersection,
        "aabb_iou": float(intersection / max(union, 1e-12)),
        "track_aabb_coverage": float(intersection / max(left_volume, 1e-12)),
        "native_aabb_coverage": float(intersection / max(right_volume, 1e-12)),
    }

def _boundary_metrics(
    left_ids: set[int],
    right_ids: set[int],
    neighbors: dict[int, list[dict]],
) -> dict:
    pairs = {}
    for left in left_ids:
        for edge in neighbors.get(left, []):
            right = int(edge["neighbor_superpoint_id"])
            if right not in right_ids or right == left:
                continue
            key = tuple(sorted((left, right)))
            pairs.setdefault(key, edge)
    values = list(pairs.values())
    total_contact = sum(int(row["boundary_contact_count"]) for row in values)

    def weighted(name: str) -> float:
        if not total_contact:
            return 0.0
        return float(sum(
            float(row[name]) * int(row["boundary_contact_count"]) for row in values
        ) / total_contact)

    return {
        "boundary_superpoint_edge_count": len(values),
        "boundary_contact_point_count": total_contact,
        "boundary_contact_ratio_mean": float(np.mean([
            row["boundary_contact_ratio"] for row in values
        ])) if values else 0.0,
        "boundary_contact_ratio_max": float(max(
            (row["boundary_contact_ratio"] for row in values), default=0.0
        )),
        "boundary_distance_weighted_mean": weighted("mean_boundary_distance"),
        "boundary_normal_difference_weighted_mean": weighted("mean_normal_difference"),
        "boundary_color_difference_weighted_mean": weighted("mean_color_difference"),
    }

def _geometry_features(track: dict, native: dict, context: dict) -> dict:
    shared_superpoints = track["superpoint_ids"] & native["superpoint_ids"]
    track_only = track["superpoint_ids"] - shared_superpoints
    native_only = native["superpoint_ids"] - shared_superpoints
    center_distance = float(np.linalg.norm(track["centroid"] - native["centroid"]))
    scale = max(1e-8, 0.5 * (track["bbox_diagonal"] + native["bbox_diagonal"]))
    normal_dot = float(np.clip(
        np.dot(track["mean_normal"], native["mean_normal"]), -1.0, 1.0
    ))
    color_distance = float(np.linalg.norm(track["mean_rgb"] - native["mean_rgb"]) / math.sqrt(3.0))
    features = {
        "track_point_count": track["point_count"],
        "native_point_count": native["point_count"],
        "log_track_over_native_point_count": float(math.log(
            max(1, track["point_count"]) / max(1, native["point_count"])
        )),
        "track_bbox_diagonal": track["bbox_diagonal"],
        "native_bbox_diagonal": native["bbox_diagonal"],
        "centroid_distance": center_distance,
        "centroid_distance_normalized": float(center_distance / scale),
        "mean_rgb_distance": color_distance,
        "mean_normal_dot": normal_dot,
        "mean_normal_absolute_dot": abs(normal_dot),
        "mean_normal_difference": 1.0 - abs(normal_dot),
        "track_normal_coherence": track["normal_coherence"],
        "native_normal_coherence": native["normal_coherence"],
        "track_superpoint_count": track["superpoint_count"],
        "native_superpoint_count": native["superpoint_count"],
        "shared_superpoint_count": len(shared_superpoints),
        "track_shared_superpoint_fraction": float(len(shared_superpoints) / max(1, track["superpoint_count"])),
        "native_shared_superpoint_fraction": float(len(shared_superpoints) / max(1, native["superpoint_count"])),
        "track_superpoint_component_count": track["superpoint_component_count"],
        "native_superpoint_component_count": native["superpoint_component_count"],
        "track_superpoint_full_occupancy_fraction": track["superpoint_full_occupancy_fraction"],
        "native_superpoint_full_occupancy_fraction": native["superpoint_full_occupancy_fraction"],
        "track_superpoint_mean_occupancy": track["superpoint_mean_occupancy"],
        "native_superpoint_mean_occupancy": native["superpoint_mean_occupancy"],
        **_aabb_metrics(track, native),
    }
    features.update({
        f"exclusive_{name}": value
        for name, value in _boundary_metrics(track_only, native_only, context["neighbors"]).items()
    })
    features.update({
        f"all_cross_{name}": value
        for name, value in _boundary_metrics(
            track["superpoint_ids"], native["superpoint_ids"], context["neighbors"]
        ).items()
    })
    return features

def _candidate_view_rows_with_box(
    points: np.ndarray,
    frame_indices: list[int],
    projections: np.ndarray,
    visibility: np.ndarray,
    scaling: tuple[float, float],
    observations: dict[int, list[dict]],
) -> dict[int, dict]:
    result = {}
    for frame_index in frame_indices:
        visible_points = points[visibility[frame_index, points]]
        if len(visible_points) < MIN_VISIBLE_POINTS:
            continue
        coords = projections[frame_index, visible_points]
        box = np.asarray([
            coords[:, 0].min() / scaling[1],
            coords[:, 1].min() / scaling[0],
            coords[:, 0].max() / scaling[1],
            coords[:, 1].max() / scaling[0],
        ], dtype=np.float64)
        candidates = observations.get(frame_index, [])
        if candidates:
            chosen = max(
                candidates,
                key=lambda row: (_box_iou(box, row["bbox"]), -row["observation_id"]),
            )
            box_iou = _box_iou(box, chosen["bbox"])
            support = float(len(np.intersect1d(
                visible_points, chosen["points"], assume_unique=True
            )) / len(visible_points))
            observation_id = int(chosen["observation_id"])
        else:
            box_iou, support, observation_id = 0.0, 0.0, -1
        result[int(frame_index)] = {
            "frame_index": int(frame_index),
            "visible_point_count": int(len(visible_points)),
            "visible_point_fraction": float(len(visible_points) / max(1, len(points))),
            "projected_box": box,
            "projected_box_area": float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])),
            "matched_observation_id": observation_id,
            "depth_consistent": True,
            "box_iou_to_matched_observation": box_iou,
            "mask_point_support": support,
            "gvc_frame_score": float(box_iou * support),
        }
    return result

def _public_view_features(
    track_rows: dict[int, dict],
    native_rows: dict[int, dict],
    track_centroid: np.ndarray,
    native_centroid: np.ndarray,
    camera_centers: dict[int, np.ndarray],
) -> dict:
    common = sorted(set(track_rows) & set(native_rows))
    selected_ids = sorted(
        common,
        key=lambda frame: (
            -min(track_rows[frame]["visible_point_count"], native_rows[frame]["visible_point_count"]),
            frame,
        ),
    )[:MAX_PUBLIC_COMMON_VIEWS]
    selected = [(track_rows[frame], native_rows[frame]) for frame in selected_ids]
    projected_ious = [_box_iou(left["projected_box"], right["projected_box"]) for left, right in selected]
    same_observation = [
        left["matched_observation_id"] >= 0
        and left["matched_observation_id"] == right["matched_observation_id"]
        for left, right in selected
    ]
    different_observation = [
        left["matched_observation_id"] >= 0
        and right["matched_observation_id"] >= 0
        and left["matched_observation_id"] != right["matched_observation_id"]
        for left, right in selected
    ]
    range_deltas = []
    for frame in selected_ids:
        camera = camera_centers[frame]
        range_deltas.append(float(
            np.linalg.norm(track_centroid - camera) - np.linalg.norm(native_centroid - camera)
        ))
    track_gvc = _summary([left["gvc_frame_score"] for left, _ in selected])
    native_gvc = _summary([right["gvc_frame_score"] for _, right in selected])
    projected = _summary(projected_ious)
    absolute_range = _summary([abs(value) for value in range_deltas])
    return {
        "public_common_eligible_view_count": len(common),
        "public_common_selected_view_count": len(selected),
        "public_common_view_available": bool(selected),
        "public_projected_box_iou_mean": projected["mean"],
        "public_projected_box_iou_min": projected["min"],
        "public_projected_box_iou_max": projected["max"],
        "public_same_matched_observation_count": int(sum(same_observation)),
        "public_different_matched_observation_count": int(sum(different_observation)),
        "public_same_matched_observation_fraction": float(sum(same_observation) / max(1, len(selected))),
        "public_different_matched_observation_fraction": float(sum(different_observation) / max(1, len(selected))),
        "public_both_matched_observation_count": int(sum(
            left["matched_observation_id"] >= 0 and right["matched_observation_id"] >= 0
            for left, right in selected
        )),
        "public_same_observation_both_depth_consistent_count": int(sum(
            same and left["depth_consistent"] and right["depth_consistent"]
            for same, (left, right) in zip(same_observation, selected)
        )),
        "public_track_gvc_mean": track_gvc["mean"],
        "public_native_gvc_mean": native_gvc["mean"],
        "public_track_minus_native_gvc": float(track_gvc["mean"] - native_gvc["mean"]),
        "public_track_visible_fraction_mean": _summary([
            left["visible_point_fraction"] for left, _ in selected
        ])["mean"],
        "public_native_visible_fraction_mean": _summary([
            right["visible_point_fraction"] for _, right in selected
        ])["mean"],
        "public_centroid_camera_range_delta_mean": _summary(range_deltas)["mean"],
        "public_centroid_camera_range_absolute_delta_mean": absolute_range["mean"],
        "public_centroid_camera_range_absolute_delta_max": absolute_range["max"],
        "public_centroid_camera_range_order_consistency": float(
            abs(sum(np.sign(range_deltas))) / max(1, len(range_deltas))
        ),
    }

def _relation_components(pair_rows: list[dict]) -> tuple[dict[tuple[int, str], dict], list[dict]]:
    adjacency: dict[tuple[str, object], set[tuple[str, object]]] = defaultdict(set)
    for row in pair_rows:
        track = ("track", int(row["track_id"]))
        native = ("native", str(row["native_exact_geometry_group_id"]))
        adjacency[track].add(native)
        adjacency[native].add(track)
    node_to_component = {}
    component_rows = []
    visited_nodes = set()
    native_member_counts = {}
    for row in pair_rows:
        group_id = str(row["native_exact_geometry_group_id"])
        size = int(row["native_exact_geometry_group_size"])
        previous = native_member_counts.setdefault(group_id, size)
        if previous != size:
            raise ValueError("native exact-geometry group size differs across relations")
    for start in sorted(adjacency, key=lambda value: (value[0], str(value[1]))):
        if start in visited_nodes:
            continue
        component_id = len(component_rows)
        queue, nodes = deque([start]), set()
        while queue:
            node = queue.popleft()
            if node in nodes:
                continue
            nodes.add(node)
            visited_nodes.add(node)
            queue.extend(sorted(adjacency[node] - nodes, key=lambda value: (value[0], str(value[1]))))
        tracks = sorted(int(value) for kind, value in nodes if kind == "track")
        natives = sorted(str(value) for kind, value in nodes if kind == "native")
        relation_keys = {
            (int(row["track_id"]), str(row["native_exact_geometry_group_id"]))
            for row in pair_rows
            if int(row["track_id"]) in tracks and str(row["native_exact_geometry_group_id"]) in natives
        }
        member_count = sum(native_member_counts[group_id] for group_id in natives)
        component = {
            "relation_component_id": component_id,
            "track_ids": tracks,
            "native_exact_geometry_group_ids": natives,
            "track_count": len(tracks),
            "native_geometry_group_count": len(natives),
            "native_member_candidate_count": member_count,
            "relation_count": len(relation_keys),
        }
        component_rows.append(component)
        for key in relation_keys:
            node_to_component[key] = component
    if len(node_to_component) != len(pair_rows):
        raise AssertionError("relation component construction did not conserve pairs")
    return node_to_component, component_rows

def _component_features(
    pair: dict,
    component: dict,
    pair_rows: list[dict],
) -> dict:
    track_id = int(pair["track_id"])
    group_id = str(pair["native_exact_geometry_group_id"])
    track_rows = [row for row in pair_rows if int(row["track_id"]) == track_id]
    group_rows = [row for row in pair_rows if str(row["native_exact_geometry_group_id"]) == group_id]
    return {
        "relation_component_track_count": int(component["track_count"]),
        "relation_component_native_group_count": int(component["native_geometry_group_count"]),
        "relation_component_native_member_candidate_count": int(component["native_member_candidate_count"]),
        "relation_component_relation_count": int(component["relation_count"]),
        "track_overlapping_native_group_count": len(track_rows),
        "track_overlapping_native_member_candidate_count": sum(
            int(row["native_exact_geometry_group_size"]) for row in track_rows
        ),
        "native_group_overlapping_track_count": len(group_rows),
        "hypothetical_pair_native_member_deletion_count": int(pair["native_exact_geometry_group_size"]),
        "hypothetical_track_wide_native_member_deletion_count": sum(
            int(row["native_exact_geometry_group_size"]) for row in track_rows
        ),
    }

def build_inference_relation_rows(
    scene: str,
    pair_rows: list[dict],
    processed: np.ndarray,
    context: dict,
    track_points: dict[int, np.ndarray],
    native_points: dict[str, np.ndarray],
    track_views: dict[int, dict[int, dict]],
    native_views: dict[str, dict[int, dict]],
    camera_centers: dict[int, np.ndarray],
    fixed_scores: dict[tuple[str, int, str], dict] | None = None,
    label_builder=None,
    exclusive_pair_features: dict[tuple[int, str], dict] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Build relation features without requiring labels or learned scores.

    ``pair_rows`` is an observed, positive-overlap relation ledger.  All
    feature inputs are inference-time quantities.  Training-only labels and
    precomputed learned candidate-quality fields may be attached by callers,
    but neither is required to build a valid relation ledger.
    """
    fixed_scores = {} if fixed_scores is None else fixed_scores
    exclusive_pair_features = {} if exclusive_pair_features is None else exclusive_pair_features
    track_stats = {
        key: _candidate_stats(points, processed, context)
        for key, points in track_points.items()
    }
    native_stats = {
        key: _candidate_stats(points, processed, context)
        for key, points in native_points.items()
    }
    component_lookup, components = _relation_components(pair_rows)
    rows = []
    for pair in pair_rows:
        track_id = int(pair["track_id"])
        group_id = str(pair["native_exact_geometry_group_id"])
        key = (scene, track_id, group_id)
        learned = fixed_scores.get(key, {})
        features = {
            "point_iou": float(pair["point_iou"]),
            "track_inside_native_ratio": float(pair["track_inside_native_ratio"]),
            "native_inside_track_ratio": float(pair["native_inside_track_ratio"]),
            "mutual_duplicate_strict_099": bool(pair["mutual_duplicate_strict_099"]),
            "track_original_score": float(pair["track_original_score"]),
            "native_original_score_median": float(pair["native_original_score_median"]),
            "original_score_delta_track_minus_native": float(
                pair["track_original_score"] - pair["native_original_score_median"]
            ),
            **_geometry_features(track_stats[track_id], native_stats[group_id], context),
            **_public_view_features(
                track_views[track_id], native_views[group_id],
                track_stats[track_id]["centroid"], native_stats[group_id]["centroid"],
                camera_centers,
            ),
            **_component_features(pair, component_lookup[(track_id, group_id)], pair_rows),
            **exclusive_pair_features.get((track_id, group_id), {}),
            **learned,
        }
        if any(name.startswith("label_") or "best_gt" in name for name in features):
            raise AssertionError("GT-derived field leaked into inference feature dictionary")
        row = {
            "scene_name": scene,
            "track_id": track_id,
            "native_exact_geometry_group_id": group_id,
            "native_member_candidate_ids": list(map(int, pair["native_member_candidate_ids"])),
            "relation_component_id": int(
                component_lookup[(track_id, group_id)]["relation_component_id"]
            ),
            "features": features,
            "contracts": {
                "feature_ground_truth_usage": "none",
                "label_ground_truth_usage": "none" if label_builder is None else "official_train_offline_label_only",
                "track_source_frames_excluded_from_public_view_features": True,
                "candidate_geometry_modified": False,
                "relation_model_trained": False,
                "replacement_action_generated": False,
                "ap_evaluation_run": False,
            },
        }
        if label_builder is not None:
            row["labels"] = label_builder(pair)
        rows.append(row)
    if len(rows) != len(pair_rows):
        raise AssertionError(f"{scene}: feature relation conservation failed")
    return rows, components, {
        "scene_name": scene,
        "relation_count": len(rows),
        "relation_component_count": len(components),
        "involved_track_count": len(track_points),
        "involved_native_geometry_group_count": len(native_points),
        "feature_ground_truth_usage": "none",
        "label_ground_truth_usage": "none" if label_builder is None else "official_train_offline_label_only",
        "learned_score_fields_attached": bool(fixed_scores),
        "relation_model_trained": False,
        "replacement_action_generated": False,
    }

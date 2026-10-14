"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

from collections import defaultdict

import numpy as np

from tools.audit_train_candidate_quality_dataset import TRACK_SOURCE

RAW_RELATION_FEATURES = (
    "point_iou",
    "native_inside_track_ratio",
    "track_inside_native_ratio",
    "aabb_iou",
    "centroid_distance_normalized",
    "native_shared_superpoint_fraction",
    "track_shared_superpoint_fraction",
    "mean_rgb_distance",
    "mean_normal_difference",
    "exclusive_boundary_contact_ratio_mean",
    "public_projected_box_iou_mean",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "public_track_minus_native_gvc",
)

CANDIDATE_NUMERIC_FEATURES = (
    "original_source_score",
    "point_count",
    "point_fraction_of_scene",
    "native_exact_geometry_group_size",
    "gvc_excluded_mean",
    "gvc_excluded_max",
    "gvc_excluded_variance",
    "gvc_excluded_selected_view_count",
    "gvc_excluded_matched_view_count",
    "gvc_excluded_zero_support_fraction",
    "support_view_count",
    "superpoint_count",
    "source_frame_count",
    "merge_action_count",
    "mean_consensus_rate",
    "mean_edge_score",
    "mean_node_quality",
)

def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite feature {name}: {value}")
    return result

def _aggregate(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"invalid aggregation for {prefix}")
    return {
        f"{prefix}__min": float(array.min()),
        f"{prefix}__mean": float(array.mean()),
        f"{prefix}__max": float(array.max()),
    }

def _candidate_relation_pairs(
    candidate: dict, raw_rows: list[dict], stacked_rows: list[dict],
) -> list[tuple[dict, dict]]:
    if len(raw_rows) != len(stacked_rows):
        raise ValueError("raw and stacked relation evidence differ")
    if candidate["candidate_source"] == TRACK_SOURCE:
        selected = [
            (raw, stacked) for raw, stacked in zip(raw_rows, stacked_rows)
            if int(raw["track_id"]) == int(candidate["candidate_id"])
        ]
    else:
        selected = [
            (raw, stacked) for raw, stacked in zip(raw_rows, stacked_rows)
            if int(candidate["candidate_id"]) in {
                int(value) for value in raw["native_member_candidate_ids"]
            }
        ]
    if not selected:
        raise ValueError("component candidate has no relation evidence")
    return selected

def build_candidate_feature_rows(
    candidates: list[dict], relation_by_component: dict,
    stacked_by_component: dict,
) -> list[dict]:
    rows = []
    for candidate in candidates:
        component_key = (str(candidate["scene_name"]), int(candidate["relation_component_id"]))
        raw_rows = relation_by_component[component_key]
        stacked_rows = stacked_by_component[component_key]
        pairs = _candidate_relation_pairs(candidate, raw_rows, stacked_rows)
        source_track = candidate["candidate_source"] == TRACK_SOURCE
        features = {
            "candidate_source_is_track": float(source_track),
            "candidate_relation_degree": float(len(pairs)),
            "component_relation_count": float(len(raw_rows)),
            "component_native_count": float(len({
                raw["native_exact_geometry_group_id"] for raw in raw_rows
            })),
            "component_track_count": float(len({int(raw["track_id"]) for raw in raw_rows})),
        }
        for field in CANDIDATE_NUMERIC_FEATURES:
            value = candidate.get(field)
            missing = value is None
            features[f"candidate__{field}"] = 0.0 if missing else _finite(value, field)
            features[f"candidate__{field}__missing"] = float(missing)
        candidate_q_field = "nested_track_q" if source_track else "nested_native_q_median"
        candidate_v25_field = "nested_track_valid25" if source_track else "nested_native_valid25_median"
        candidate_v50_field = "nested_track_valid50" if source_track else "nested_native_valid50_median"
        other_q_field = "nested_native_q_median" if source_track else "nested_track_q"
        win_field = "track_win_relation_score" if source_track else "baseline_win_relation_score"
        other_win_field = "baseline_win_relation_score" if source_track else "track_win_relation_score"
        perspective = {
            "candidate_nested_q": [stacked[candidate_q_field] for _, stacked in pairs],
            "candidate_nested_valid25": [stacked[candidate_v25_field] for _, stacked in pairs],
            "candidate_nested_valid50": [stacked[candidate_v50_field] for _, stacked in pairs],
            "other_nested_q": [stacked[other_q_field] for _, stacked in pairs],
            "candidate_q_advantage": [
                stacked[candidate_q_field] - stacked[other_q_field] for _, stacked in pairs
            ],
            "same_target_score": [stacked["same_target_score"] for _, stacked in pairs],
            "different_target_score": [stacked["different_target_score"] for _, stacked in pairs],
            "candidate_win_relation_score": [stacked[win_field] for _, stacked in pairs],
            "other_win_relation_score": [stacked[other_win_field] for _, stacked in pairs],
            "coexist_relation_score": [stacked["coexist_relation_score"] for _, stacked in pairs],
        }
        auxiliary_fields = (
            "multitask_track_better_score",
            "multitask_baseline_better_score",
            "multitask_track_win_relation_score",
            "multitask_baseline_win_relation_score",
        )
        auxiliary_available = [
            all(field in stacked for field in auxiliary_fields) for _, stacked in pairs
        ]
        if any(auxiliary_available) and not all(auxiliary_available):
            raise ValueError("multitask auxiliary relation evidence is only partially available")
        if all(auxiliary_available):
            auxiliary_win = (
                "multitask_track_win_relation_score" if source_track
                else "multitask_baseline_win_relation_score"
            )
            auxiliary_other_win = (
                "multitask_baseline_win_relation_score" if source_track
                else "multitask_track_win_relation_score"
            )
            perspective.update({
                "multitask_candidate_win_relation_score": [
                    stacked[auxiliary_win] for _, stacked in pairs
                ],
                "multitask_other_win_relation_score": [
                    stacked[auxiliary_other_win] for _, stacked in pairs
                ],
                "multitask_track_better_score": [
                    stacked["multitask_track_better_score"] for _, stacked in pairs
                ],
                "multitask_baseline_better_score": [
                    stacked["multitask_baseline_better_score"] for _, stacked in pairs
                ],
            })
        for name, values in perspective.items():
            features.update(_aggregate([float(value) for value in values], f"relation__{name}"))
        for field in RAW_RELATION_FEATURES:
            features.update(_aggregate(
                [_finite(raw["features"][field], field) for raw, _ in pairs],
                f"raw_relation__{field}",
            ))
        output_row = {
            "scene_name": str(candidate["scene_name"]),
            "relation_component_id": int(candidate["relation_component_id"]),
            "candidate_source": str(candidate["candidate_source"]),
            "candidate_id": int(candidate["candidate_id"]),
            "model_features": features,
        }
        if "label_component_unique_winner" in candidate:
            output_row.update({
                "label_component_unique_winner": int(candidate["label_component_unique_winner"]),
                "label_calibrated_quality": float(candidate["label_calibrated_quality"]),
                "label_official_relevance": float(candidate["label_official_relevance"]),
                "label_best_gt_iou": float(candidate["label_best_gt_iou"]),
                "label_harm_kind": str(candidate["label_harm_kind"]),
            })
        rows.append(output_row)

    by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_component[(row["scene_name"], row["relation_component_id"])].append(row)
    for component_rows in by_component.values():
        q_values = np.asarray([
            row["model_features"]["relation__candidate_nested_q__mean"]
            for row in component_rows
        ], dtype=np.float64)
        for row, q_value in zip(component_rows, q_values):
            same_source = [
                other["model_features"]["relation__candidate_nested_q__mean"]
                for other in component_rows
                if other["candidate_source"] == row["candidate_source"]
            ]
            order = sorted(
                range(len(component_rows)),
                key=lambda index: (-q_values[index], component_rows[index]["candidate_source"], component_rows[index]["candidate_id"]),
            )
            index = component_rows.index(row)
            features = row["model_features"]
            features["context__candidate_q_minus_component_mean"] = float(q_value - q_values.mean())
            features["context__candidate_q_minus_component_max"] = float(q_value - q_values.max())
            features["context__candidate_q_rank_fraction"] = float(order.index(index) / max(1, len(order) - 1))
            features["context__candidate_q_minus_source_mean"] = float(q_value - np.mean(same_source))
            features["context__candidate_q_minus_source_max"] = float(q_value - np.max(same_source))
            features["context__source_candidate_fraction"] = float(len(same_source) / len(component_rows))
    return rows

def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0]["model_features"])
    if any(sorted(row["model_features"]) != names for row in rows):
        raise ValueError("candidate list feature schemas differ")
    forbidden = [name for name in names if "label" in name or "best_gt" in name]
    if forbidden:
        raise AssertionError(f"GT field leaked into model features: {forbidden}")
    matrix = np.asarray([
        [_finite(row["model_features"][name], name) for name in names] for row in rows
    ], dtype=np.float64)
    return matrix, names

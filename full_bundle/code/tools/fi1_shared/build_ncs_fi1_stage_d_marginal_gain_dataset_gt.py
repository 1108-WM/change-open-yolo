"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

from collections import Counter

import numpy as np

from tools.fi1_shared.build_ncs_fi1_stage_a_quality_dataset_gt import UNION_FEATURE_NAMES, _quality_target

from tools.fi1_shared.train_ncs_fi1_stage_c_v2_refinement_oof import QUALITY_FEATURE_NAMES

EXPECTED_BASELINE_COUNT = 8591

EXPECTED_ORIGINAL_COUNT = 1578

EXPECTED_REFINED_COUNT = 85

OFFICIAL_THRESHOLDS = tuple(value / 100.0 for value in range(50, 95, 5))

RELATION_TYPES = (
    "exact_duplicate", "near_duplicate", "containment",
    "complementary", "conflict", "uncertain",
)

BASE_FEATURE_NAMES = (
    "variant_refined",
    "log1p_candidate_point_count",
    "candidate_point_fraction_of_scene",
    "candidate_point_fraction_of_original_union",
    "candidate_removed_point_fraction_from_original_union",
    "stage_a_original_union_quality",
    "stage_b_original_union_score",
    "frozen_threshold_cross_probability",
    "frozen_union_base_quality",
    "frozen_track_quality_q",
    "frozen_native_group_median_quality_q",
    "stage_c_candidate_quality_prediction",
    "stage_c_candidate_quality_lower_bound",
    "geometry_quality_for_scoring",
    "stage_c_confident_removal_fraction",
    "stage_c_connectivity_removal_fraction",
    "stage_c_total_removal_fraction",
    "baseline_max_point_iou",
    "baseline_max_candidate_inside_ratio",
    "baseline_max_baseline_inside_ratio",
    "baseline_max_min_point_count_ratio",
    "baseline_candidate_new_point_fraction",
    "baseline_intersection_count",
    "baseline_iou_ge_005_count",
    "baseline_near_duplicate_count",
    "baseline_containment_count",
    "baseline_overlap_stage_a_quality_max",
    "baseline_overlap_stage_a_quality_mean",
    "baseline_overlap_stage_b_score_max",
    "baseline_overlap_stage_b_score_mean",
    "stage_b_relation_count",
)

FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *tuple(f"stage_a_union__{name}" for name in UNION_FEATURE_NAMES),
    *tuple(f"stage_c__{name}" for name in QUALITY_FEATURE_NAMES),
    *tuple(f"stage_b_relation_count__{name}" for name in RELATION_TYPES),
    *tuple(f"stage_b_relation_fraction__{name}" for name in RELATION_TYPES),
)

def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result

def soft_quality(iou: float) -> float:
    return float(np.clip((float(iou) - 0.50) / 0.45, 0.0, 1.0))

def iou_by_gt(points: np.ndarray, gt: np.ndarray, gt_meta: dict[int, dict]) -> dict[int, float]:
    if not len(points):
        return {}
    labels, intersections = np.unique(gt[points], return_counts=True)
    result = {}
    for encoded, intersection in zip(labels, intersections):
        encoded = int(encoded)
        meta = gt_meta.get(encoded)
        if meta is None:
            continue
        intersection = int(intersection)
        union = len(points) + int(meta["point_count"]) - intersection
        result[encoded] = float(intersection / max(1, union))
    return result

def label_candidate(candidate_iou: dict[int, float], existing_best: dict[int, float]) -> dict:
    gain_candidates = []
    q_gains = []
    soft_gains = []
    threshold_candidates = []
    for encoded, iou in candidate_iou.items():
        base = float(existing_best[encoded])
        gain = max(0.0, float(iou) - base)
        gain_candidates.append((gain, float(iou), -encoded, encoded, base))
        q_gains.append(max(0.0, _quality_target(iou) - _quality_target(base)))
        soft_gains.append(max(0.0, soft_quality(iou) - soft_quality(base)))
        crossings = sum(iou >= threshold and base < threshold for threshold in OFFICIAL_THRESHOLDS)
        threshold_candidates.append((crossings, float(iou) - base, float(iou), -encoded, encoded, base))
    if gain_candidates:
        gain, selected_iou, _, selected_gt, selected_base = max(gain_candidates)
        candidate_best_iou = max(candidate_iou.values())
        threshold = max(threshold_candidates)
    else:
        gain = selected_iou = selected_base = candidate_best_iou = 0.0
        selected_gt = None
        threshold = (0, 0.0, 0.0, 0, None, 0.0)
    return {
        "selected_gain_gt_encoded_id": selected_gt,
        "selected_gain_candidate_iou": float(selected_iou),
        "existing_best_iou_for_selected_gain_target": float(selected_base),
        "candidate_best_iou": float(candidate_best_iou),
        "candidate_quality_q": _quality_target(candidate_best_iou),
        "marginal_iou_gain": float(gain),
        "marginal_q_gain": float(max(q_gains, default=0.0)),
        "marginal_soft_quality_gain": float(max(soft_gains, default=0.0)),
        "threshold_cross_target_gt_encoded_id": threshold[4],
        "official_threshold_cross_count": int(threshold[0]),
        "official_threshold_cross_fraction": float(threshold[0] / len(OFFICIAL_THRESHOLDS)),
        "crosses_any_official_threshold": bool(threshold[0] > 0),
    }

def overlap_features(
    candidate: np.ndarray, baseline: list[tuple[dict, np.ndarray]],
    stage_a_by_key: dict[str, dict], stage_b_by_key: dict[str, dict],
) -> dict[str, float]:
    point_ious = []
    candidate_inside = []
    baseline_inside = []
    min_ratios = []
    stage_a_scores = []
    stage_b_scores = []
    near_count = containment_count = iou_005_count = 0
    for node, points in baseline:
        intersection = int(len(np.intersect1d(candidate, points, assume_unique=True)))
        if not intersection:
            continue
        union = len(candidate) + len(points) - intersection
        iou = intersection / max(1, union)
        inside_candidate = intersection / max(1, len(candidate))
        inside_baseline = intersection / max(1, len(points))
        point_ious.append(iou)
        candidate_inside.append(inside_candidate)
        baseline_inside.append(inside_baseline)
        min_ratios.append(min(len(candidate), len(points)) / max(len(candidate), len(points)))
        key = str(node["geometry_key"])
        stage_a_scores.append(float(stage_a_by_key[key]["oof_unified_quality"]))
        stage_b_scores.append(float(stage_b_by_key[key]["stage_b_score"]))
        iou_005_count += int(iou >= 0.05)
        near_count += int(min(inside_candidate, inside_baseline) >= 0.90)
        containment_count += int(
            max(inside_candidate, inside_baseline) >= 0.90
            and min(inside_candidate, inside_baseline) < 0.90
        )
    return {
        "baseline_max_point_iou": max(point_ious, default=0.0),
        "baseline_max_candidate_inside_ratio": max(candidate_inside, default=0.0),
        "baseline_max_baseline_inside_ratio": max(baseline_inside, default=0.0),
        "baseline_max_min_point_count_ratio": max(min_ratios, default=0.0),
        "baseline_candidate_new_point_fraction": 1.0 - max(candidate_inside, default=0.0),
        "baseline_intersection_count": float(len(point_ious)),
        "baseline_iou_ge_005_count": float(iou_005_count),
        "baseline_near_duplicate_count": float(near_count),
        "baseline_containment_count": float(containment_count),
        "baseline_overlap_stage_a_quality_max": max(stage_a_scores, default=0.0),
        "baseline_overlap_stage_a_quality_mean": float(np.mean(stage_a_scores)) if stage_a_scores else 0.0,
        "baseline_overlap_stage_b_score_max": max(stage_b_scores, default=0.0),
        "baseline_overlap_stage_b_score_mean": float(np.mean(stage_b_scores)) if stage_b_scores else 0.0,
    }

def relation_features(original_key: str, relations_by_geometry: dict[str, Counter]) -> dict[str, float]:
    counts = relations_by_geometry.get(original_key, Counter())
    total = int(sum(counts.values()))
    output = {"stage_b_relation_count": float(total)}
    for relation_type in RELATION_TYPES:
        count = int(counts[relation_type])
        output[f"stage_b_relation_count__{relation_type}"] = float(count)
        output[f"stage_b_relation_fraction__{relation_type}"] = count / max(1, total)
    return output

def candidate_features(
    *, variant: str, points: np.ndarray, original_points: np.ndarray, scene_point_count: int,
    original_union: dict, original_stage_a_row: dict, original_stage_b_row: dict,
    stage_c_row: dict, overlap: dict[str, float], relation: dict[str, float],
) -> dict[str, float]:
    refined = variant == "refined"
    atom_count = max(1, int(stage_c_row["atom_count"]))
    if refined:
        quality_prediction = float(stage_c_row["corrected_oof_temporary_refined_quality"])
        quality_lower = float(stage_c_row["quality_lower_confidence_bound"])
        geometry_quality = float(np.clip(quality_lower, 0.0, 1.0))
    else:
        quality_prediction = float(original_stage_a_row["oof_unified_quality"])
        quality_lower = quality_prediction
        geometry_quality = float(np.clip(original_stage_b_row["stage_b_score"], 0.0, 1.0))
    values = {
        "variant_refined": float(refined),
        "log1p_candidate_point_count": math.log1p(len(points)),
        "candidate_point_fraction_of_scene": len(points) / max(1, scene_point_count),
        "candidate_point_fraction_of_original_union": len(points) / max(1, len(original_points)),
        "candidate_removed_point_fraction_from_original_union": 1.0 - len(points) / max(1, len(original_points)),
        "stage_a_original_union_quality": float(original_stage_a_row["oof_unified_quality"]),
        "stage_b_original_union_score": float(original_stage_b_row["stage_b_score"]),
        "frozen_threshold_cross_probability": float(original_union["threshold_cross_probability"]),
        "frozen_union_base_quality": float(original_union["base_quality"]),
        "frozen_track_quality_q": float(original_union["track_quality_q"]),
        "frozen_native_group_median_quality_q": float(original_union["native_group_median_quality_q"]),
        "stage_c_candidate_quality_prediction": quality_prediction,
        "stage_c_candidate_quality_lower_bound": quality_lower,
        "geometry_quality_for_scoring": geometry_quality,
        "stage_c_confident_removal_fraction": int(stage_c_row["confident_removal_count"]) / atom_count if refined else 0.0,
        "stage_c_connectivity_removal_fraction": int(stage_c_row["connectivity_removal_count"]) / atom_count if refined else 0.0,
        "stage_c_total_removal_fraction": (
            int(stage_c_row["confident_removal_count"]) + int(stage_c_row["connectivity_removal_count"])
        ) / atom_count if refined else 0.0,
        **overlap,
        **relation,
    }
    values.update({
        f"stage_a_union__{name}": float(original_stage_a_row["features"][name])
        for name in UNION_FEATURE_NAMES
    })
    values.update({
        f"stage_c__{name}": float(stage_c_row["quality_features"][name])
        for name in QUALITY_FEATURE_NAMES
    })
    if set(values) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(values))
        extra = sorted(set(values) - set(FEATURE_NAMES))
        raise ValueError(f"feature schema differs; missing={missing}, extra={extra}")
    return {name: _finite(values[name], name) for name in FEATURE_NAMES}

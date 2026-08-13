#!/usr/bin/env python3
"""Train nested scene-disjoint target-consistency models on official100.

The binary task is restricted to reliable relations: same target versus
different targets that must coexist.  Unknown relations are never converted
to negatives.  Every outer fold uses the frozen 80/20 scene split.  Within
the 80 training scenes, four scene-disjoint folds generate decision scores
for Platt calibration before the final outer-validation prediction.

Frozen learned candidate-quality scores are explicitly forbidden because
their original cross-fitting contract is not nested inside this relation
model.  This entry point exports OOF diagnostics only; it does not fit a
deployable all-scene model, choose a threshold, emit a replacement action,
modify candidates, or run AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_train_candidate_relation_feature_ledger import (  # noqa: E402
    EXCLUSIVE_PAIR_MODEL_FEATURES,
)
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)


OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 4
RANDOM_SEED = 20260809
BOOTSTRAP_SEED = 20260810
LEARNED_QUALITY_MARKERS = (
    "C_plus_geometry_track_structure",
    "D_plus_gvc",
)


MODEL_FEATURES = {
    "A_point_iou_only": (
        "point_iou",
    ),
    "B_overlap_geometry": (
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance",
        "centroid_distance_normalized",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
    ),
    "C_plus_topology_appearance": (
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance",
        "centroid_distance_normalized",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
        "shared_superpoint_count",
        "track_shared_superpoint_fraction",
        "native_shared_superpoint_fraction",
        "track_superpoint_count",
        "native_superpoint_count",
        "track_superpoint_component_count",
        "native_superpoint_component_count",
        "native_superpoint_full_occupancy_fraction",
        "native_superpoint_mean_occupancy",
        "mean_rgb_distance",
        "mean_normal_absolute_dot",
        "mean_normal_difference",
        "track_normal_coherence",
        "native_normal_coherence",
        "exclusive_boundary_superpoint_edge_count",
        "exclusive_boundary_contact_point_count",
        "exclusive_boundary_contact_ratio_mean",
        "exclusive_boundary_contact_ratio_max",
        "exclusive_boundary_distance_weighted_mean",
        "exclusive_boundary_normal_difference_weighted_mean",
        "exclusive_boundary_color_difference_weighted_mean",
    ),
    "D_plus_public_component": (
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance",
        "centroid_distance_normalized",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
        "shared_superpoint_count",
        "track_shared_superpoint_fraction",
        "native_shared_superpoint_fraction",
        "track_superpoint_count",
        "native_superpoint_count",
        "track_superpoint_component_count",
        "native_superpoint_component_count",
        "native_superpoint_full_occupancy_fraction",
        "native_superpoint_mean_occupancy",
        "mean_rgb_distance",
        "mean_normal_absolute_dot",
        "mean_normal_difference",
        "track_normal_coherence",
        "native_normal_coherence",
        "exclusive_boundary_superpoint_edge_count",
        "exclusive_boundary_contact_point_count",
        "exclusive_boundary_contact_ratio_mean",
        "exclusive_boundary_contact_ratio_max",
        "exclusive_boundary_distance_weighted_mean",
        "exclusive_boundary_normal_difference_weighted_mean",
        "exclusive_boundary_color_difference_weighted_mean",
        "public_common_eligible_view_count",
        "public_common_selected_view_count",
        "public_common_view_available",
        "public_projected_box_iou_mean",
        "public_projected_box_iou_min",
        "public_projected_box_iou_max",
        "public_same_matched_observation_count",
        "public_same_matched_observation_fraction",
        "public_different_matched_observation_count",
        "public_different_matched_observation_fraction",
        "public_both_matched_observation_count",
        "public_track_gvc_mean",
        "public_native_gvc_mean",
        "public_track_minus_native_gvc",
        "public_track_visible_fraction_mean",
        "public_native_visible_fraction_mean",
        "public_centroid_camera_range_absolute_delta_mean",
        "public_centroid_camera_range_absolute_delta_max",
        "public_centroid_camera_range_order_consistency",
        "relation_component_track_count",
        "relation_component_native_group_count",
        "relation_component_native_member_candidate_count",
        "relation_component_relation_count",
        "track_overlapping_native_group_count",
        "track_overlapping_native_member_candidate_count",
        "native_group_overlapping_track_count",
        "hypothetical_pair_native_member_deletion_count",
        "hypothetical_track_wide_native_member_deletion_count",
        "track_original_score",
        "native_original_score_median",
        "original_score_delta_track_minus_native",
    ),
}
MODEL_FEATURES["E_plus_exclusive_pair"] = (
    MODEL_FEATURES["D_plus_public_component"] + EXCLUSIVE_PAIR_MODEL_FEATURES
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_feature_contract(rows: list[dict]) -> None:
    available = set(rows[0]["features"]) if rows else set()
    for model_name, feature_names in MODEL_FEATURES.items():
        missing = set(feature_names) - available
        if missing:
            raise ValueError(f"{model_name}: missing frozen relation features: {sorted(missing)}")
        forbidden = [
            name for name in feature_names
            if any(marker in name for marker in LEARNED_QUALITY_MARKERS)
            or name.startswith("label_")
            or "gt_" in name
        ]
        if forbidden:
            raise ValueError(f"{model_name}: forbidden learned/GT features: {forbidden}")
    for row in rows:
        for model_name, feature_names in MODEL_FEATURES.items():
            values = [row["features"].get(name) for name in feature_names]
            if any(not isinstance(value, (int, float, bool)) for value in values):
                raise ValueError(f"{model_name}: non-numeric feature in relation row")
            if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                raise ValueError(f"{model_name}: non-finite feature in relation row")


def _reliable_task_rows(rows: list[dict]) -> list[dict]:
    selected = [
        row for row in rows
        if row["labels"]["target_state"] in ("same_target", "different_target_coexist")
    ]
    if not selected or any(not row["labels"]["reliable_pair"] for row in selected):
        raise ValueError("target-consistency training requires reliable labelled relations")
    return selected


def _labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([
        int(row["labels"]["target_state"] == "same_target") for row in rows
    ], dtype=np.int64)


def _matrix(rows: list[dict], feature_names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([
        [float(row["features"][name]) for name in feature_names] for row in rows
    ], dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(rows), len(feature_names)):
        raise AssertionError("relation feature matrix shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("relation feature matrix contains non-finite values")
    return values


def scene_track_balanced_weights(rows: list[dict], indexes: np.ndarray | None = None) -> np.ndarray:
    """Give every scene equal mass and every track equal mass within a scene."""
    if indexes is None:
        indexes = np.arange(len(rows), dtype=np.int64)
    indexes = np.asarray(indexes, dtype=np.int64)
    if not len(indexes):
        raise ValueError("cannot weight an empty relation population")
    scene_tracks: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indexes:
        row = rows[int(index)]
        scene_tracks[str(row["scene_name"])][int(row["track_id"])].append(int(index))
    weights = np.zeros(len(rows), dtype=np.float64)
    scene_mass = len(indexes) / len(scene_tracks)
    for tracks in scene_tracks.values():
        track_mass = scene_mass / len(tracks)
        for relation_indexes in tracks.values():
            relation_weight = track_mass / len(relation_indexes)
            weights[relation_indexes] = relation_weight
    if not np.isclose(weights[indexes].sum(), len(indexes), atol=1e-8):
        raise AssertionError("scene-track weights do not preserve total mass")
    return weights


def class_balanced_weights(labels: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("class-balanced fitting requires both classes")
    total = float(weights.sum())
    for label in (0, 1):
        mask = labels == label
        mass = float(weights[mask].sum())
        if mass <= 0:
            raise ValueError(f"class {label} has no positive weight")
        weights[mask] *= 0.5 * total / mass
    return weights


def seeded_scene_folds(scene_names: list[str], fold_count: int, seed: int) -> list[dict]:
    unique = sorted(set(scene_names))
    if len(unique) < fold_count or len(unique) % fold_count:
        raise ValueError("scene count must be divisible by inner fold count")
    shuffled = np.random.default_rng(seed).permutation(unique).tolist()
    size = len(unique) // fold_count
    folds = []
    for fold_index in range(fold_count):
        validation = set(shuffled[fold_index * size:(fold_index + 1) * size])
        folds.append({
            "fold_index": fold_index,
            "train_scenes": sorted(set(unique) - validation),
            "validation_scenes": sorted(validation),
        })
    return folds


def _make_base_model(seed: int) -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )),
    ])


def _fit_base(
    matrix: np.ndarray,
    labels: np.ndarray,
    base_weights: np.ndarray,
    seed: int,
) -> Pipeline:
    model = _make_base_model(seed)
    fit_weights = class_balanced_weights(labels, base_weights)
    model.fit(matrix, labels, classifier__sample_weight=fit_weights)
    return model


def _fit_platt(decisions: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    if not np.isfinite(decisions).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Platt calibration requires finite decisions and both classes")
    calibrator = LogisticRegression(
        penalty="l2", C=1e6, solver="lbfgs", max_iter=5000,
        random_state=RANDOM_SEED,
    )
    calibrator.fit(decisions.reshape(-1, 1), labels, sample_weight=weights)
    return calibrator


def nested_fit_predict(
    rows: list[dict],
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    outer_fold_index: int,
) -> tuple[np.ndarray, dict]:
    train_indexes = np.asarray(train_indexes, dtype=np.int64)
    validation_indexes = np.asarray(validation_indexes, dtype=np.int64)
    train_scenes = [rows[index]["scene_name"] for index in train_indexes]
    inner_folds = seeded_scene_folds(
        train_scenes, INNER_FOLD_COUNT, RANDOM_SEED + outer_fold_index
    )
    inner_decisions = np.full(len(train_indexes), np.nan, dtype=np.float64)
    train_position = {int(index): position for position, index in enumerate(train_indexes)}
    inner_assignments = np.zeros(len(train_indexes), dtype=np.int64)
    inner_summaries = []
    for inner in inner_folds:
        inner_train = np.asarray([
            index for index in train_indexes
            if rows[index]["scene_name"] in set(inner["train_scenes"])
        ], dtype=np.int64)
        inner_validation = np.asarray([
            index for index in train_indexes
            if rows[index]["scene_name"] in set(inner["validation_scenes"])
        ], dtype=np.int64)
        base = scene_track_balanced_weights(rows, inner_train)[inner_train]
        model = _fit_base(
            matrix[inner_train], labels[inner_train], base,
            RANDOM_SEED + outer_fold_index * 10 + int(inner["fold_index"]),
        )
        decisions = model.decision_function(matrix[inner_validation])
        for index, value in zip(inner_validation, decisions):
            position = train_position[int(index)]
            if not math.isnan(inner_decisions[position]):
                raise AssertionError("inner validation relation predicted more than once")
            inner_decisions[position] = float(value)
            inner_assignments[position] += 1
        inner_summaries.append({
            "inner_fold_index": int(inner["fold_index"]),
            "train_scene_count": len(inner["train_scenes"]),
            "validation_scene_count": len(inner["validation_scenes"]),
            "train_relation_count": len(inner_train),
            "validation_relation_count": len(inner_validation),
        })
    if not np.all(inner_assignments == 1) or not np.isfinite(inner_decisions).all():
        raise AssertionError("every outer-training relation must receive one inner OOF decision")
    calibration_weights = scene_track_balanced_weights(rows, train_indexes)[train_indexes]
    calibrator = _fit_platt(inner_decisions, labels[train_indexes], calibration_weights)
    final_model = _fit_base(
        matrix[train_indexes], labels[train_indexes], calibration_weights,
        RANDOM_SEED + outer_fold_index,
    )
    outer_decisions = final_model.decision_function(matrix[validation_indexes])
    predictions = calibrator.predict_proba(outer_decisions.reshape(-1, 1))[:, 1]
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("calibrated outer predictions are outside [0, 1]")
    classifier = final_model.named_steps["classifier"]
    diagnostics = {
        "outer_fold_index": outer_fold_index,
        "inner_fold_count": INNER_FOLD_COUNT,
        "inner_folds": inner_summaries,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "standardized_coefficients": classifier.coef_[0].astype(float).tolist(),
        "base_intercept": float(classifier.intercept_[0]),
    }
    return predictions.astype(np.float64), diagnostics


def _ece(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, bins: int = 10) -> float:
    total = float(weights.sum())
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & ((scores < high) if index < bins - 1 else (scores <= high))
        if not mask.any():
            continue
        mass = float(weights[mask].sum())
        observed = float(np.average(labels[mask], weights=weights[mask]))
        predicted = float(np.average(scores[mask], weights=weights[mask]))
        error += mass / total * abs(observed - predicted)
    return float(error)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    if not len(labels) or not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError("binary metrics received empty or non-finite inputs")
    result = {
        "count": len(labels),
        "positive_count": int(labels.sum()),
        "positive_rate": float(np.average(labels, weights=weights)),
        "brier": float(brier_score_loss(labels, scores, sample_weight=weights)),
        "ece10": _ece(labels, scores, weights),
        "log_loss": float(log_loss(labels, scores, sample_weight=weights, labels=[0, 1])),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores, sample_weight=weights))
        result["pr_auc"] = float(average_precision_score(labels, scores, sample_weight=weights))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def grouped_metrics(rows: list[dict], labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict:
    return {
        "relation_unweighted": binary_metrics(labels, scores),
        "scene_track_balanced": binary_metrics(labels, scores, weights),
    }


def _prediction_bootstrap(
    rows: list[dict], labels: np.ndarray, scores: np.ndarray,
    cluster_kind: str, repetitions: int, seed: int,
) -> dict:
    if cluster_kind == "scene":
        cluster_ids = [str(row["scene_name"]) for row in rows]
    elif cluster_kind == "track":
        cluster_ids = [f"{row['scene_name']}:{int(row['track_id'])}" for row in rows]
    else:
        raise ValueError(f"unknown bootstrap cluster kind: {cluster_kind}")
    unique = sorted(set(cluster_ids))
    mapping = {value: index for index, value in enumerate(unique)}
    row_clusters = np.asarray([mapping[value] for value in cluster_ids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    aucs, prs, briers = [], [], []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(unique), len(unique))
        cluster_weights = np.bincount(sampled, minlength=len(unique)).astype(np.float64)
        weights = cluster_weights[row_clusters]
        active = weights > 0
        if len(np.unique(labels[active])) < 2:
            continue
        aucs.append(float(roc_auc_score(labels, scores, sample_weight=weights)))
        prs.append(float(average_precision_score(labels, scores, sample_weight=weights)))
        briers.append(float(brier_score_loss(labels, scores, sample_weight=weights)))

    def interval(values: list[float]) -> dict | None:
        if not values:
            return None
        return {
            "median": float(np.median(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    return {
        "cluster_kind": cluster_kind,
        "cluster_count": len(unique),
        "requested_repetitions": repetitions,
        "valid_repetitions": len(aucs),
        "roc_auc": interval(aucs),
        "pr_auc": interval(prs),
        "brier": interval(briers),
    }


def run_oof(
    rows: list[dict],
    scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]],
    bootstrap_repetitions: int,
) -> tuple[dict, list[dict], list[dict], dict]:
    _validate_feature_contract(rows)
    rows = _reliable_task_rows(rows)
    labels = _labels(rows)
    matrices = {name: _matrix(rows, features) for name, features in MODEL_FEATURES.items()}
    predictions = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in MODEL_FEATURES}
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics, model_diagnostics = [], {name: [] for name in MODEL_FEATURES}
    for fold_index in range(OUTER_FOLD_COUNT):
        train_indexes = np.asarray([
            index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] != fold_index
        ], dtype=np.int64)
        validation_indexes = np.asarray([
            index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        train_scenes = {rows[index]["scene_name"] for index in train_indexes}
        validation_scenes = {rows[index]["scene_name"] for index in validation_indexes}
        if len(train_scenes) != 80 or len(validation_scenes) != 20 or train_scenes & validation_scenes:
            raise AssertionError("outer relation rows do not follow frozen 80/20 scene split")
        assignments[validation_indexes] += 1
        evaluation_weights = scene_track_balanced_weights(rows, validation_indexes)[validation_indexes]
        fold = {
            "fold_index": fold_index,
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "train_relation_count": len(train_indexes),
            "validation_relation_count": len(validation_indexes),
            "validation_positive_count": int(labels[validation_indexes].sum()),
            "models": {},
        }
        for model_name, matrix in matrices.items():
            values, diagnostics = nested_fit_predict(
                rows, matrix, labels, train_indexes, validation_indexes, fold_index
            )
            predictions[model_name][validation_indexes] = values
            diagnostics["feature_names"] = list(MODEL_FEATURES[model_name])
            model_diagnostics[model_name].append(diagnostics)
            fold["models"][model_name] = grouped_metrics(
                [rows[index] for index in validation_indexes],
                labels[validation_indexes], values, evaluation_weights,
            )
        fold_metrics.append(fold)
    if not np.all(assignments == 1) or any(not np.isfinite(values).all() for values in predictions.values()):
        raise AssertionError("every reliable relation must receive one outer OOF prediction")
    frozen_evaluation_weights = scene_track_balanced_weights(rows)
    overall = {
        model_name: grouped_metrics(rows, labels, values, frozen_evaluation_weights)
        for model_name, values in predictions.items()
    }
    cohort_metrics = {}
    for cohort_name, cohort_scenes in cohorts.items():
        indexes = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in cohort_scenes
        ], dtype=np.int64)
        cohort_weights = scene_track_balanced_weights(rows, indexes)[indexes]
        cohort_metrics[cohort_name] = {
            "scene_count": len(cohort_scenes),
            "relation_count": len(indexes),
            "positive_count": int(labels[indexes].sum()),
            "models": {
                name: grouped_metrics(
                    [rows[index] for index in indexes], labels[indexes], values[indexes], cohort_weights
                )
                for name, values in predictions.items()
            },
        }
    bootstrap = {
        model_name: {
            kind: _prediction_bootstrap(
                rows, labels, values, kind, bootstrap_repetitions,
                BOOTSTRAP_SEED + model_index * 10 + cluster_index,
            )
            for cluster_index, kind in enumerate(("scene", "track"))
        }
        for model_index, (model_name, values) in enumerate(predictions.items())
    }
    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({
            "scene_name": row["scene_name"],
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "label_target_state": row["labels"]["target_state"],
            "label_same_target": int(labels[index]),
            "evaluation_scene_track_weight": float(frozen_evaluation_weights[index]),
            "predictions": {name: float(values[index]) for name, values in predictions.items()},
            "ground_truth_usage": "official_train_offline_label_only",
        })
    summary = {
        "scene_count": 100,
        "relation_count": len(rows),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "outer_fold_count": OUTER_FOLD_COUNT,
        "inner_fold_count": INNER_FOLD_COUNT,
        "overall_metrics": overall,
        "cohort_metrics": cohort_metrics,
        "model_cluster_bootstrap": bootstrap,
        "model_selection_applied": False,
    }
    return summary, oof_rows, fold_metrics, model_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    actual_split_sha = _sha256(args.split_manifest)
    if actual_split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    all_rows = _load_rows(args.feature_ledger_root, scenes)
    summary_core, oof_rows, fold_metrics, model_diagnostics = run_oof(
        all_rows, scene_to_fold, cohorts, args.bootstrap_repetitions
    )
    summary = {
        "version": f"{args.protocol_name}_candidate_target_consistency_nested_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "feature_groups": {name: list(features) for name, features in MODEL_FEATURES.items()},
        "training_contract": {
            "outer_split": "frozen official100 five-fold scene-disjoint 80/20",
            "inner_split": "four-fold scene-disjoint within each outer 80-scene training partition",
            "base_model": "standardized L2 logistic regression, C=1.0",
            "fit_weighting": "scene-track balanced, then class balanced inside each fit partition",
            "calibration": "Platt logistic calibration on inner-fold OOF decisions with natural scene-track weights",
            "unknown_relations_used_as_negative": False,
            "learned_candidate_quality_features_used": False,
            "learned_candidate_quality_exclusion_reason": "existing candidate-quality OOF scores are not nested inside the relation outer fold",
        },
        "safety_contract": {
            "fit_all_model_exported": False,
            "threshold_selected": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "ap_evaluation_run": False,
            "relative_quality_model_trained": False,
        },
        "input_provenance": {
            "feature_ledger_summary_path": str((args.feature_ledger_root / "summary.json").resolve()),
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": _sha256(args.scene_list),
            "existing20_scene_list_path": str(args.existing20_scene_list.resolve()),
            "existing20_scene_list_sha256": _sha256(args.existing20_scene_list),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": actual_split_sha,
            "split_existing_scene_count": split_payload.get("existing_scene_count"),
            "split_new_scene_count": split_payload.get("new_scene_count"),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "fold_metrics.json").write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "model_diagnostics.json").write_text(
            json.dumps(model_diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "oof_predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in oof_rows
        ))
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

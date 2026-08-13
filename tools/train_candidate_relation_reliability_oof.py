#!/usr/bin/env python3
"""Train a strictly nested relation-reliability reject gate on official100.

The task is defined on every track--native relation: both candidates must have
official-train IoU25 support.  Candidate q/valid25/valid50 predictions are
rebuilt inside every frozen relation outer fold.  Unknown relations are valid
negative labels for this gate, but are never repurposed as target-consistency
labels.  This entry point exports diagnostics only and never selects an action
threshold, mutates candidates, or runs AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.diagnose_candidate_two_stage_cascade_oof import _top_metrics  # noqa: E402
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_quality_head_oof import load_rows as load_candidate_rows  # noqa: E402
from tools.train_candidate_relative_quality_nested_q_oof import (  # noqa: E402
    nested_candidate_quality_predictions,
    relation_nested_quality_evidence,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    INNER_FOLD_COUNT,
    OUTER_FOLD_COUNT,
    RANDOM_SEED,
    _fit_platt,
    binary_metrics,
    scene_track_balanced_weights,
    seeded_scene_folds,
)


MODEL_NAMES = ("R0_valid25_pair_min", "R1_candidate_reliability", "R2_plus_relation")
NESTED_FEATURES = tuple(
    name for target in ("q", "valid25", "valid50") for name in (
        f"nested_track_{target}",
        f"nested_native_{target}_median",
        f"nested_native_{target}_min",
        f"nested_native_{target}_range",
        f"nested_{target}_pair_min",
        f"nested_{target}_pair_product",
        f"nested_{target}_delta_track_minus_native",
    )
)
R1_RAW_FEATURES = (
    "track_original_score",
    "native_original_score_median",
    "track_point_count",
    "native_point_count",
    "track_bbox_diagonal",
    "native_bbox_diagonal",
    "track_superpoint_count",
    "native_superpoint_count",
    "track_superpoint_component_count",
    "native_superpoint_component_count",
    "track_superpoint_full_occupancy_fraction",
    "native_superpoint_full_occupancy_fraction",
    "track_superpoint_mean_occupancy",
    "native_superpoint_mean_occupancy",
    "track_normal_coherence",
    "native_normal_coherence",
    "public_track_gvc_mean",
    "public_native_gvc_mean",
    "public_track_visible_fraction_mean",
    "public_native_visible_fraction_mean",
    "public_common_eligible_view_count",
    "public_common_view_available",
)
R2_EXTRA_FEATURES = (
    "point_iou",
    "track_inside_native_ratio",
    "native_inside_track_ratio",
    "aabb_iou",
    "track_aabb_coverage",
    "native_aabb_coverage",
    "centroid_distance_normalized",
    "log_track_over_native_point_count",
    "shared_superpoint_count",
    "track_shared_superpoint_fraction",
    "native_shared_superpoint_fraction",
    "mean_rgb_distance",
    "mean_normal_difference",
    "exclusive_boundary_contact_ratio_mean",
    "exclusive_boundary_normal_difference_weighted_mean",
    "exclusive_boundary_color_difference_weighted_mean",
    "public_projected_box_iou_mean",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "relation_component_track_count",
    "relation_component_native_group_count",
    "relation_component_relation_count",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reliability_labels(rows: list[dict]) -> np.ndarray:
    labels = np.asarray([int(bool(row["labels"]["reliable_pair"])) for row in rows], dtype=np.int64)
    for row, label in zip(rows, labels):
        expected = row["labels"]["target_state"] in ("same_target", "different_target_coexist")
        if bool(label) != expected:
            raise ValueError("reliable-pair label differs from target-state contract")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("reliability task requires both reliable and unknown relations")
    return labels


def reliability_matrix(
    rows: list[dict], evidence: list[dict], model_name: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if len(rows) != len(evidence):
        raise ValueError("relation rows and nested quality evidence differ in length")
    if model_name == "R1_candidate_reliability":
        raw_names = R1_RAW_FEATURES
    elif model_name == "R2_plus_relation":
        raw_names = (*R1_RAW_FEATURES, *R2_EXTRA_FEATURES)
    else:
        raise ValueError(f"unknown learned reliability model: {model_name}")
    names = (*NESTED_FEATURES, *raw_names)
    values = np.asarray([
        [*(float(item[name]) for name in NESTED_FEATURES),
         *(float(row["features"][name]) for name in raw_names)]
        for row, item in zip(rows, evidence)
    ], dtype=np.float64)
    if values.shape != (len(rows), len(names)) or not np.isfinite(values).all():
        raise ValueError("reliability feature matrix is malformed")
    return values, names


def _make_model(seed: int) -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=seed,
        )),
    ])


def nested_reliability_fit_predict(
    rows: list[dict], matrix: np.ndarray, labels: np.ndarray,
    train_indexes: np.ndarray, validation_indexes: np.ndarray, outer_fold_index: int,
) -> tuple[np.ndarray, dict]:
    train_indexes = np.asarray(train_indexes, dtype=np.int64)
    validation_indexes = np.asarray(validation_indexes, dtype=np.int64)
    folds = seeded_scene_folds(
        [rows[index]["scene_name"] for index in train_indexes],
        INNER_FOLD_COUNT, RANDOM_SEED + outer_fold_index,
    )
    inner_decisions = np.full(len(train_indexes), np.nan, dtype=np.float64)
    positions = {int(index): position for position, index in enumerate(train_indexes)}
    assignments = np.zeros(len(train_indexes), dtype=np.int64)
    summaries = []
    for inner in folds:
        train_scenes = set(inner["train_scenes"])
        validation_scenes = set(inner["validation_scenes"])
        inner_train = np.asarray([
            index for index in train_indexes if rows[index]["scene_name"] in train_scenes
        ], dtype=np.int64)
        inner_validation = np.asarray([
            index for index in train_indexes if rows[index]["scene_name"] in validation_scenes
        ], dtype=np.int64)
        weights = scene_track_balanced_weights(rows, inner_train)[inner_train]
        model = _make_model(RANDOM_SEED + outer_fold_index * 10 + int(inner["fold_index"]))
        model.fit(matrix[inner_train], labels[inner_train], classifier__sample_weight=weights)
        decisions = model.decision_function(matrix[inner_validation])
        for index, value in zip(inner_validation, decisions):
            position = positions[int(index)]
            inner_decisions[position] = float(value)
            assignments[position] += 1
        summaries.append({
            "inner_fold_index": int(inner["fold_index"]),
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "train_relation_count": len(inner_train),
            "validation_relation_count": len(inner_validation),
        })
    if not np.all(assignments == 1) or not np.isfinite(inner_decisions).all():
        raise AssertionError("every outer-training reliability relation needs one inner decision")
    calibration_weights = scene_track_balanced_weights(rows, train_indexes)[train_indexes]
    calibrator = _fit_platt(inner_decisions, labels[train_indexes], calibration_weights)
    final = _make_model(RANDOM_SEED + outer_fold_index)
    final.fit(
        matrix[train_indexes], labels[train_indexes],
        classifier__sample_weight=calibration_weights,
    )
    decisions = final.decision_function(matrix[validation_indexes])
    predictions = calibrator.predict_proba(decisions.reshape(-1, 1))[:, 1]
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("reliability probabilities are outside [0, 1]")
    classifier = final.named_steps["classifier"]
    return predictions.astype(np.float64), {
        "outer_fold_index": outer_fold_index,
        "inner_folds": summaries,
        "fit_class_balanced": False,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "standardized_coefficients": classifier.coef_[0].astype(float).tolist(),
        "base_intercept": float(classifier.intercept_[0]),
    }


def _metrics(rows: list[dict], labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        **binary_metrics(labels, scores),
        "top_selection": _top_metrics(rows, labels, scores),
    }


def run_oof(
    rows: list[dict], candidate_rows: list[dict], scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]], candidate_protocol_name: str,
) -> tuple[dict, list[dict], list[dict], dict]:
    labels = reliability_labels(rows)
    predictions = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in MODEL_NAMES}
    evidence_rows: list[dict | None] = [None] * len(rows)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics, diagnostics = [], []
    feature_names = {}
    for fold_index in range(OUTER_FOLD_COUNT):
        validation = np.asarray([
            index for index, row in enumerate(rows)
            if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        train = np.asarray([
            index for index, row in enumerate(rows)
            if scene_to_fold[row["scene_name"]] != fold_index
        ], dtype=np.int64)
        quality_lookup, quality_diagnostics = nested_candidate_quality_predictions(
            candidate_rows, fold_index, scene_to_fold, candidate_protocol_name
        )
        evidence = relation_nested_quality_evidence(rows, quality_lookup)
        predictions["R0_valid25_pair_min"][validation] = np.asarray([
            evidence[index]["nested_valid25_pair_min"] for index in validation
        ], dtype=np.float64)
        fold_models = {}
        for model_name in MODEL_NAMES[1:]:
            matrix, names = reliability_matrix(rows, evidence, model_name)
            feature_names[model_name] = list(names)
            values, model_diagnostics = nested_reliability_fit_predict(
                rows, matrix, labels, train, validation, fold_index
            )
            predictions[model_name][validation] = values
            fold_models[model_name] = model_diagnostics
        for index in validation:
            evidence_rows[index] = evidence[index]
        assignments[validation] += 1
        fold_rows = [rows[index] for index in validation]
        fold_metrics.append({
            "fold_index": fold_index,
            "validation_relation_count": len(validation),
            "validation_reliable_count": int(labels[validation].sum()),
            "models": {
                name: _metrics(fold_rows, labels[validation], values[validation])
                for name, values in predictions.items()
            },
        })
        diagnostics.append({
            "fold_index": fold_index,
            "candidate_quality": quality_diagnostics,
            "models": fold_models,
        })
    if not np.all(assignments == 1):
        raise AssertionError("every relation must enter one reliability validation fold")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise AssertionError("reliability OOF predictions are incomplete")
    if any(value is None for value in evidence_rows):
        raise AssertionError("nested candidate-quality evidence is incomplete")
    overall = {name: _metrics(rows, labels, values) for name, values in predictions.items()}
    cohort_metrics = {}
    for cohort_name, scenes in cohorts.items():
        indexes = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in scenes
        ], dtype=np.int64)
        cohort_rows = [rows[index] for index in indexes]
        cohort_metrics[cohort_name] = {
            "scene_count": len(scenes),
            "relation_count": len(indexes),
            "reliable_count": int(labels[indexes].sum()),
            "models": {
                name: _metrics(cohort_rows, labels[indexes], values[indexes])
                for name, values in predictions.items()
            },
        }
    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({
            "scene_name": row["scene_name"],
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "label_reliable_pair": int(labels[index]),
            "label_target_state": row["labels"]["target_state"],
            "predictions": {name: float(values[index]) for name, values in predictions.items()},
            "nested_candidate_quality_features": evidence_rows[index],
            "ground_truth_usage": "official_train_offline_label_only",
        })
    summary = {
        "scene_count": len(scene_to_fold),
        "relation_count": len(rows),
        "reliable_count": int(labels.sum()),
        "unknown_count": int((labels == 0).sum()),
        "overall_metrics": overall,
        "cohort_metrics": cohort_metrics,
        "feature_groups": {
            "R0_valid25_pair_min": ["nested_valid25_pair_min"],
            **feature_names,
        },
        "model_selection_applied": False,
    }
    return summary, oof_rows, fold_metrics, {"folds": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--candidate-records-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    split_sha = _sha256(args.split_manifest)
    if split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    rows = _load_rows(args.feature_ledger_root, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    summary_core, oof_rows, fold_metrics, diagnostics = run_oof(
        rows, candidate_rows, scene_to_fold, cohorts, args.candidate_quality_protocol_name
    )
    summary = {
        "version": f"{args.protocol_name}_candidate_relation_reliability_nested_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "training_contract": {
            "outer_split": "frozen official100 five-fold scene-disjoint 80/20",
            "candidate_quality": "q/valid25/valid50 rebuilt with four inner folds inside every relation outer fold",
            "learned_models": "standardized L2 logistic regression with natural scene-track weights",
            "calibration": "Platt calibration on inner-fold decisions",
            "class_balanced_fit": False,
            "unknown_relations_are_reliability_negatives_only": True,
        },
        "safety_contract": {
            "threshold_selected": False,
            "fit_all_model_exported": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "ap_evaluation_run": False,
        },
        "input_provenance": {
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": split_sha,
            "split_existing_scene_count": split_payload.get("existing_scene_count"),
            "split_new_scene_count": split_payload.get("new_scene_count"),
            "candidate_records_root": str(args.candidate_records_root.resolve()),
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
        (staging / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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

#!/usr/bin/env python3
"""Train a strictly nested continuous relative-IoU margin model.

The model is fitted only on reliable same-target official-train relations and
uses the continuous label ``track IoU - native IoU``.  Inner scene folds train
a Huber center regressor, a lower-quantile regressor, a Platt map for
``margin > 0.05``, and a scene-block conformal correction.  Final outer-fold
models score every validation relation for later safety auditing.  No action
threshold, replacement, candidate mutation, or AP evaluation is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_quality_head_oof import load_rows as load_candidate_rows  # noqa: E402
from tools.train_candidate_relation_reliability_oof import NESTED_FEATURES  # noqa: E402
from tools.train_candidate_relative_quality_nested_q_oof import (  # noqa: E402
    nested_candidate_quality_predictions,
    relation_nested_quality_evidence,
)
from tools.train_candidate_relative_quality_oof import (  # noqa: E402
    MODEL_FEATURES as RELATIVE_FEATURES,
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


IOU_MARGIN = 0.05
LOWER_QUANTILE = 0.10
CONFORMAL_ALPHA = 0.10
RAW_FEATURES = RELATIVE_FEATURES["B_plus_directional_geometry"]
MARGIN_FEATURES = (*RAW_FEATURES, *NESTED_FEATURES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin_training_indexes(rows: list[dict]) -> np.ndarray:
    indexes = np.asarray([
        index for index, row in enumerate(rows)
        if row["labels"]["reliable_pair"]
        and row["labels"]["target_state"] == "same_target"
        and row["labels"]["relative_quality_state"]
        in ("prefer_track", "prefer_native", "equivalent_abstain")
    ], dtype=np.int64)
    if not len(indexes):
        raise ValueError("no reliable same-target margin rows")
    for index in indexes:
        value = rows[index]["labels"].get("iou_margin_track_minus_native")
        if value is None or not math.isfinite(float(value)):
            raise ValueError("margin training row lacks a finite IoU margin")
    return indexes


def margin_values(rows: list[dict]) -> np.ndarray:
    values = np.full(len(rows), np.nan, dtype=np.float64)
    for index in margin_training_indexes(rows):
        values[index] = float(rows[index]["labels"]["iou_margin_track_minus_native"])
    return values


def margin_matrix(rows: list[dict], evidence: list[dict]) -> np.ndarray:
    values = np.asarray([
        [*(float(row["features"][name]) for name in RAW_FEATURES),
         *(float(item[name]) for name in NESTED_FEATURES)]
        for row, item in zip(rows, evidence)
    ], dtype=np.float64)
    if values.shape != (len(rows), len(MARGIN_FEATURES)) or not np.isfinite(values).all():
        raise ValueError("relative-margin feature matrix is malformed")
    return values


def _make_regressor(kind: str, seed: int) -> GradientBoostingRegressor:
    common = dict(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=20,
        subsample=1.0,
        random_state=seed,
    )
    if kind == "center":
        return GradientBoostingRegressor(loss="huber", alpha=0.90, **common)
    if kind == "lower":
        return GradientBoostingRegressor(loss="quantile", alpha=LOWER_QUANTILE, **common)
    raise ValueError(f"unknown margin regressor kind: {kind}")


def scene_block_conformal_correction(
    rows: list[dict], indexes: np.ndarray, margins: np.ndarray,
    lower_predictions: np.ndarray, alpha: float = CONFORMAL_ALPHA,
) -> tuple[float, dict]:
    indexes = np.asarray(indexes, dtype=np.int64)
    if not 0 < alpha < 1 or not len(indexes):
        raise ValueError("scene-block conformal calibration requires rows and alpha in (0, 1)")
    scene_scores = {}
    for index in indexes:
        score = float(lower_predictions[index] - margins[index])
        scene = str(rows[index]["scene_name"])
        scene_scores[scene] = max(scene_scores.get(scene, -float("inf")), score)
    scores = np.asarray(list(scene_scores.values()), dtype=np.float64)
    level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    correction = float(np.quantile(scores, level, method="higher"))
    return correction, {
        "alpha": alpha,
        "calibration_scene_count": len(scores),
        "finite_sample_quantile_level": level,
        "scene_max_residual_min": float(scores.min()),
        "scene_max_residual_median": float(np.median(scores)),
        "scene_max_residual_max": float(scores.max()),
        "correction": correction,
    }


def nested_margin_fit_predict(
    rows: list[dict], matrix: np.ndarray, margins: np.ndarray,
    train_indexes: np.ndarray, validation_indexes: np.ndarray,
    scene_to_fold: dict[str, int], outer_fold_index: int,
) -> tuple[dict[str, np.ndarray], dict]:
    train_indexes = np.asarray(train_indexes, dtype=np.int64)
    validation_indexes = np.asarray(validation_indexes, dtype=np.int64)
    outer_train_scenes = sorted(
        scene for scene, fold in scene_to_fold.items() if fold != outer_fold_index
    )
    folds = seeded_scene_folds(
        outer_train_scenes, INNER_FOLD_COUNT, RANDOM_SEED + outer_fold_index
    )
    center_oof = np.full(len(rows), np.nan, dtype=np.float64)
    lower_oof = np.full(len(rows), np.nan, dtype=np.float64)
    assignments = np.zeros(len(rows), dtype=np.int64)
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
        center = _make_regressor(
            "center", RANDOM_SEED + outer_fold_index * 20 + int(inner["fold_index"])
        )
        lower = _make_regressor(
            "lower", RANDOM_SEED + outer_fold_index * 20 + 10 + int(inner["fold_index"])
        )
        center.fit(matrix[inner_train], margins[inner_train], sample_weight=weights)
        lower.fit(matrix[inner_train], margins[inner_train], sample_weight=weights)
        center_oof[inner_validation] = center.predict(matrix[inner_validation])
        lower_oof[inner_validation] = lower.predict(matrix[inner_validation])
        assignments[inner_validation] += 1
        summaries.append({
            "inner_fold_index": int(inner["fold_index"]),
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "train_margin_relation_count": len(inner_train),
            "validation_margin_relation_count": len(inner_validation),
        })
    if not np.all(assignments[train_indexes] == 1):
        raise AssertionError("every outer-training margin row needs one inner prediction")
    if not np.isfinite(center_oof[train_indexes]).all() or not np.isfinite(lower_oof[train_indexes]).all():
        raise AssertionError("inner margin predictions are incomplete")
    train_weights = scene_track_balanced_weights(rows, train_indexes)[train_indexes]
    positive = (margins[train_indexes] > IOU_MARGIN).astype(np.int64)
    calibrator = _fit_platt(center_oof[train_indexes], positive, train_weights)
    correction, conformal_diagnostics = scene_block_conformal_correction(
        rows, train_indexes, margins, lower_oof
    )
    final_center = _make_regressor("center", RANDOM_SEED + outer_fold_index)
    final_lower = _make_regressor("lower", RANDOM_SEED + 100 + outer_fold_index)
    final_center.fit(matrix[train_indexes], margins[train_indexes], sample_weight=train_weights)
    final_lower.fit(matrix[train_indexes], margins[train_indexes], sample_weight=train_weights)
    center = final_center.predict(matrix[validation_indexes]).astype(np.float64)
    lower_raw = final_lower.predict(matrix[validation_indexes]).astype(np.float64)
    probability = calibrator.predict_proba(center.reshape(-1, 1))[:, 1].astype(np.float64)
    lower_conformal = lower_raw - correction
    for values in (center, lower_raw, probability, lower_conformal):
        if not np.isfinite(values).all():
            raise ValueError("outer relative-margin predictions contain non-finite values")
    return {
        "center_margin": center,
        "probability_margin_gt_005": probability,
        "lower_quantile_raw": lower_raw,
        "lower_conformal": lower_conformal,
    }, {
        "outer_fold_index": outer_fold_index,
        "inner_folds": summaries,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "conformal": conformal_diagnostics,
    }


def _margin_metrics(
    rows: list[dict], indexes: np.ndarray, margins: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict:
    indexes = np.asarray(indexes, dtype=np.int64)
    actual = margins[indexes]
    center = predictions["center_margin"][indexes]
    probability = predictions["probability_margin_gt_005"][indexes]
    lower_raw = predictions["lower_quantile_raw"][indexes]
    lower = predictions["lower_conformal"][indexes]
    labels = (actual > IOU_MARGIN).astype(np.int64)
    correlation = spearmanr(actual, center)
    correlation_value = float(getattr(
        correlation, "statistic", getattr(correlation, "correlation", float("nan"))
    ))
    selected = lower > IOU_MARGIN
    scene_coverage = []
    for scene in sorted({rows[index]["scene_name"] for index in indexes}):
        scene_positions = np.asarray([
            position for position, index in enumerate(indexes)
            if rows[index]["scene_name"] == scene
        ], dtype=np.int64)
        scene_coverage.append(bool(np.all(actual[scene_positions] >= lower[scene_positions])))
    return {
        "count": len(indexes),
        "prefer_track_count": int(labels.sum()),
        "center_mae": float(mean_absolute_error(actual, center)),
        "center_rmse": float(mean_squared_error(actual, center) ** 0.5),
        "center_spearman": correlation_value if np.isfinite(correlation_value) else None,
        "lower_quantile_pinball": float(mean_pinball_loss(actual, lower_raw, alpha=LOWER_QUANTILE)),
        "relation_lower_coverage": float(np.mean(actual >= lower)),
        "scene_simultaneous_lower_coverage": float(np.mean(scene_coverage)),
        "probability_metrics": binary_metrics(labels, probability),
        "conformal_lower_gt_005": {
            "selected_count": int(selected.sum()),
            "correct_count": int(labels[selected].sum()),
            "precision": float(labels[selected].mean()) if np.any(selected) else None,
            "selected_scene_count": len({
                rows[index]["scene_name"] for index in indexes[selected]
            }) if np.any(selected) else 0,
        },
    }


def run_oof(
    rows: list[dict], candidate_rows: list[dict], scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]], candidate_protocol_name: str,
) -> tuple[dict, list[dict], list[dict], dict]:
    margins = margin_values(rows)
    eligible = margin_training_indexes(rows)
    predictions = {
        name: np.full(len(rows), np.nan, dtype=np.float64) for name in (
            "center_margin", "probability_margin_gt_005",
            "lower_quantile_raw", "lower_conformal",
        )
    }
    evidence_rows: list[dict | None] = [None] * len(rows)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics, diagnostics = [], []
    for fold_index in range(OUTER_FOLD_COUNT):
        validation_all = np.asarray([
            index for index, row in enumerate(rows)
            if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        train_margin = np.asarray([
            index for index in eligible if scene_to_fold[rows[index]["scene_name"]] != fold_index
        ], dtype=np.int64)
        validation_margin = np.asarray([
            index for index in eligible if scene_to_fold[rows[index]["scene_name"]] == fold_index
        ], dtype=np.int64)
        quality_lookup, quality_diagnostics = nested_candidate_quality_predictions(
            candidate_rows, fold_index, scene_to_fold, candidate_protocol_name
        )
        evidence = relation_nested_quality_evidence(rows, quality_lookup)
        matrix = margin_matrix(rows, evidence)
        values, model_diagnostics = nested_margin_fit_predict(
            rows, matrix, margins, train_margin, validation_all, scene_to_fold, fold_index
        )
        for name in predictions:
            predictions[name][validation_all] = values[name]
        for index in validation_all:
            evidence_rows[index] = evidence[index]
        assignments[validation_all] += 1
        fold_metrics.append({
            "fold_index": fold_index,
            "train_margin_relation_count": len(train_margin),
            "validation_margin_relation_count": len(validation_margin),
            "metrics": _margin_metrics(rows, validation_margin, margins, predictions),
        })
        diagnostics.append({
            "fold_index": fold_index,
            "candidate_quality": quality_diagnostics,
            "margin_model": model_diagnostics,
        })
    if not np.all(assignments == 1) or any(
        not np.isfinite(values).all() for values in predictions.values()
    ):
        raise AssertionError("every relation must receive one outer margin prediction")
    if any(value is None for value in evidence_rows):
        raise AssertionError("nested margin evidence is incomplete")
    overall = _margin_metrics(rows, eligible, margins, predictions)
    cohort_metrics = {}
    for cohort_name, scenes in cohorts.items():
        indexes = np.asarray([
            index for index in eligible if rows[index]["scene_name"] in scenes
        ], dtype=np.int64)
        cohort_metrics[cohort_name] = {
            "scene_count": len(scenes),
            "metrics": _margin_metrics(rows, indexes, margins, predictions),
        }
    eligible_set = set(eligible.tolist())
    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({
            "scene_name": row["scene_name"],
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "label_margin_available": index in eligible_set,
            "label_iou_margin_track_minus_native": float(margins[index])
            if np.isfinite(margins[index]) else None,
            "label_relative_quality_state": row["labels"]["relative_quality_state"],
            "predictions": {name: float(values[index]) for name, values in predictions.items()},
            "nested_candidate_quality_features": evidence_rows[index],
            "ground_truth_usage": "official_train_offline_label_only",
        })
    summary = {
        "scene_count": len(scene_to_fold),
        "all_relation_count": len(rows),
        "margin_relation_count": len(eligible),
        "prefer_track_count": int(np.sum(margins[eligible] > IOU_MARGIN)),
        "overall_metrics": overall,
        "cohort_metrics": cohort_metrics,
        "feature_names": list(MARGIN_FEATURES),
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
        "version": f"{args.protocol_name}_candidate_relative_margin_nested_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "training_contract": {
            "label": "track_best_gt_iou - native_best_gt_iou on reliable same-target relations",
            "decision_margin": IOU_MARGIN,
            "center_loss": "Huber gradient boosting",
            "lower_loss": f"quantile gradient boosting at tau={LOWER_QUANTILE}",
            "conformal": f"scene-block maximum residual, alpha={CONFORMAL_ALPHA}",
            "candidate_quality": "strictly nested q/valid25/valid50",
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

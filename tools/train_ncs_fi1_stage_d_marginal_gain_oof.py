#!/usr/bin/env python3
"""Train the preregistered stage-D continuous marginal-gain OOF model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import _read_jsonl, _resolve, _sha256  # noqa: E402
from tools.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import FEATURE_NAMES  # noqa: E402
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import regression_metrics  # noqa: E402


VERSION = "ncs_fi1_stage_d_marginal_gain_oof_v1"
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def conservative_append_score(
    corrected_gain: np.ndarray, q90: np.ndarray, geometry_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conservative = np.maximum(0.0, np.asarray(corrected_gain) - np.asarray(q90))
    score = np.asarray(geometry_quality) * conservative
    return conservative, score


def fit_oof(
    matrix: np.ndarray, target: np.ndarray, folds: np.ndarray, model_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    raw_output = np.full(len(target), np.nan, dtype=np.float64)
    corrected_output = np.full(len(target), np.nan, dtype=np.float64)
    q90_output = np.full(len(target), np.nan, dtype=np.float64)
    records = []
    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        fit = np.flatnonzero((folds != outer_fold) & (folds != calibration_fold))
        calibration = np.flatnonzero(folds == calibration_fold)
        test = np.flatnonzero(folds == outer_fold)
        if not len(fit) or not len(calibration) or not len(test):
            raise ValueError(f"empty stage-D OOF split for fold {outer_fold}")
        model = HistGradientBoostingRegressor(
            random_state=20260822 + outer_fold, **MODEL_PARAMS
        )
        model.fit(matrix[fit], target[fit])
        calibration_raw = np.asarray(model.predict(matrix[calibration]), dtype=np.float64)
        test_raw = np.asarray(model.predict(matrix[test]), dtype=np.float64)
        bias = float(np.mean(target[calibration] - calibration_raw))
        calibration_corrected = np.clip(calibration_raw + bias, 0.0, 1.0)
        test_corrected = np.clip(test_raw + bias, 0.0, 1.0)
        q90 = float(np.quantile(np.abs(calibration_corrected - target[calibration]), 0.90))
        raw_output[test] = test_raw
        corrected_output[test] = test_corrected
        q90_output[test] = q90
        model_path = model_root / f"marginal_gain_fold_{outer_fold}.joblib"
        joblib.dump({
            "version": VERSION,
            "model_kind": "continuous_marginal_iou_gain",
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "feature_names": FEATURE_NAMES,
            "model_params": {**MODEL_PARAMS, "random_state": 20260822 + outer_fold},
            "regressor": model,
            "calibration_bias": bias,
            "calibration_absolute_residual_q90": q90,
            "prediction_clip": (0.0, 1.0),
        }, model_path)
        records.append({
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "fit_count": len(fit),
            "calibration_count": len(calibration),
            "test_count": len(test),
            "fit_folds": sorted(set(map(int, folds[fit]))),
            "calibration_bias": bias,
            "calibration_absolute_residual_q90": q90,
            "model_file": str(Path("models") / model_path.name),
            "model_sha256": _sha256(model_path),
        })
    if not np.isfinite(raw_output).all() or not np.isfinite(corrected_output).all() or not np.isfinite(q90_output).all():
        raise ValueError("stage-D OOF outputs are incomplete or non-finite")
    return raw_output, corrected_output, q90_output, records


def run(args: argparse.Namespace) -> dict:
    for name in ("dataset_root", "dataset_audit_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary_path = args.dataset_root / "summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    audit_summary_path = args.dataset_audit_root / "summary.json"
    dataset_audit = json.loads(audit_summary_path.read_text())
    if dataset_audit.get("audit_valid") is not True or dataset_audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("stage-D dataset independent audit did not authorize training")
    dataset_path = args.dataset_root / dataset_summary["files"]["dataset"]
    rows = _read_jsonl(dataset_path)
    if tuple(dataset_summary["feature_names"]) != FEATURE_NAMES:
        raise ValueError("stage-D dataset feature schema differs from training contract")
    matrix = np.asarray([
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in rows
    ], dtype=np.float64)
    target = np.asarray([
        float(row["labels"]["marginal_iou_gain"]) for row in rows
    ], dtype=np.float64)
    folds = np.asarray([int(row["fold_index"]) for row in rows], dtype=np.int8)
    threshold_control = np.asarray([
        float(row["features"]["frozen_threshold_cross_probability"]) for row in rows
    ], dtype=np.float64)
    geometry_quality = np.asarray([
        float(row["features"]["geometry_quality_for_scoring"]) for row in rows
    ], dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("stage-D training matrix or target is non-finite")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    model_root = staging / "models"
    model_root.mkdir()
    try:
        raw, corrected, q90, models = fit_oof(matrix, target, folds, model_root)
        lower = corrected - q90
        conservative, stage_d_score = conservative_append_score(corrected, q90, geometry_quality)
        if np.any(stage_d_score < -1e-12) or np.any(stage_d_score > geometry_quality + 1e-12):
            raise AssertionError("stage-D append score violates its geometry-quality bound")

        prediction_rows = []
        for index, row in enumerate(rows):
            prediction_rows.append({
                "candidate_key": row["candidate_key"],
                "scene_name": row["scene_name"],
                "fold_index": int(row["fold_index"]),
                "union_candidate_id": int(row["union_candidate_id"]),
                "candidate_variant": row["candidate_variant"],
                "geometry_sha256": row["geometry_sha256"],
                "label_marginal_iou_gain": float(target[index]),
                "frozen_threshold_cross_probability_control": float(threshold_control[index]),
                "raw_oof_marginal_iou_gain": float(raw[index]),
                "corrected_oof_marginal_iou_gain": float(corrected[index]),
                "calibration_absolute_residual_q90": float(q90[index]),
                "marginal_gain_lower_confidence_bound": float(lower[index]),
                "conservative_gain": float(conservative[index]),
                "geometry_quality_for_scoring": float(geometry_quality[index]),
                "stage_d_append_score": float(stage_d_score[index]),
                "candidate_retained": True,
                "candidate_deletion": False,
                "candidate_mutation": False,
                "geometry_mutation": False,
                "class_mutation": False,
                "frozen_cache_write": False,
                "ap_computed": False,
            })

        overall = {
            "oof_prediction": regression_metrics(target, corrected),
            "zero_prediction_control": regression_metrics(target, np.zeros(len(target))),
            "frozen_threshold_cross_probability_control": regression_metrics(target, threshold_control),
        }
        fold_metrics = {}
        fold_confident_counts = Counter()
        for fold in range(5):
            selected = folds == fold
            fold_confident_counts[fold] = int(np.sum(selected & (conservative > 0.0)))
            fold_metrics[str(fold)] = {
                "oof_prediction": regression_metrics(target[selected], corrected[selected]),
                "zero_prediction_control": regression_metrics(target[selected], np.zeros(np.sum(selected))),
                "frozen_threshold_cross_probability_control": regression_metrics(target[selected], threshold_control[selected]),
                "positive_target_count": int(np.sum(target[selected] > 0.0)),
                "confident_positive_count": fold_confident_counts[fold],
            }
        confident = conservative > 0.0
        confident_count = int(np.sum(confident))
        confident_true_positive = int(np.sum(confident & (target > 0.0)))
        confident_neutral = confident_count - confident_true_positive
        confident_precision = confident_true_positive / max(1, confident_count)
        score_contract_valid = bool(
            np.isfinite(stage_d_score).all()
            and np.all(stage_d_score >= 0.0)
            and np.all(stage_d_score <= geometry_quality + 1e-12)
        )
        checks = {
            "dataset_audit_valid": True,
            "oof_mae_strictly_better_than_zero_control": (
                overall["oof_prediction"]["mae"] < overall["zero_prediction_control"]["mae"]
            ),
            "oof_mae_strictly_better_than_threshold_cross_probability_control": (
                overall["oof_prediction"]["mae"]
                < overall["frozen_threshold_cross_probability_control"]["mae"]
            ),
            "oof_spearman_strictly_positive": overall["oof_prediction"]["spearman"] > 0.0,
            "confident_positive_count_greater_than_zero": confident_count > 0,
            "confident_true_positive_fraction_at_least_0_70": confident_precision >= 0.70,
            "score_contract_valid": score_contract_valid,
            **{
                f"fold_{fold}_has_confident_positive": fold_confident_counts[fold] > 0
                for fold in range(5)
            },
            "candidate_deletion_count_zero": True,
            "geometry_mutation_count_zero": True,
            "class_mutation_count_zero": True,
            "frozen_cache_write_count_zero": True,
            "ap_computed_false": True,
            "validation60_read_false": True,
            "val312_read_false": True,
        }
        advancement = all(checks.values())
        prediction_path = staging / "oof_marginal_gain_predictions.jsonl"
        plan_path = staging / "stage_d_append_score_plan.jsonl"
        prediction_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prediction_rows
        ))
        plan_path.write_text("".join(
            json.dumps({
                "candidate_key": row["candidate_key"],
                "scene_name": row["scene_name"],
                "fold_index": row["fold_index"],
                "union_candidate_id": row["union_candidate_id"],
                "candidate_variant": row["candidate_variant"],
                "geometry_sha256": row["geometry_sha256"],
                "geometry_quality_for_scoring": row["geometry_quality_for_scoring"],
                "conservative_gain": row["conservative_gain"],
                "stage_d_append_score": row["stage_d_append_score"],
                "frozen_threshold_cross_probability_control": row["frozen_threshold_cross_probability_control"],
                "candidate_retained": True,
                "candidate_deletion": False,
                "candidate_mutation": False,
                "geometry_mutation": False,
                "class_mutation": False,
                "frozen_cache_write": False,
                "ap_computed": False,
            }, ensure_ascii=False, sort_keys=True) + "\n" for row in prediction_rows
        ))
        summary = {
            "version": VERSION,
            "scene_count": int(dataset_summary["scene_count"]),
            "scene_with_candidate_count": int(len(set(row["scene_name"] for row in rows))),
            "candidate_count": len(rows),
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "model_params": MODEL_PARAMS,
            "calibration_contract": "fixed next-fold additive mean-residual correction; absolute residual q90; clip [0,1]",
            "score_contract": "geometry_quality_for_scoring * max(0, corrected_gain-q90)",
            "overall_metrics": overall,
            "fold_metrics": fold_metrics,
            "confident_positive": {
                "count": confident_count,
                "true_positive_count": confident_true_positive,
                "neutral_count": confident_neutral,
                "true_positive_fraction": confident_precision,
                "by_fold": {str(fold): int(fold_confident_counts[fold]) for fold in range(5)},
            },
            "score_distribution": {
                "nonzero_count": int(np.sum(stage_d_score > 0.0)),
                "mean": float(np.mean(stage_d_score)),
                "max": float(np.max(stage_d_score)),
            },
            "advancement_gate": {
                "checks": checks,
                "advancement_authorized_pending_independent_audit": advancement,
            },
            "models": models,
            "files": {
                "predictions": prediction_path.name,
                "plan": plan_path.name,
            },
            "hashes": {
                "predictions": _sha256(prediction_path),
                "plan": _sha256(plan_path),
            },
            "candidate_deletion_count": 0,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "frozen_cache_write": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "dataset_summary_sha256": _sha256(dataset_summary_path),
                "dataset_sha256": _sha256(dataset_path),
                "dataset_audit_summary_sha256": _sha256(audit_summary_path),
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "oof_mae": result["overall_metrics"]["oof_prediction"]["mae"],
        "zero_control_mae": result["overall_metrics"]["zero_prediction_control"]["mae"],
        "confident_positive_count": result["confident_positive"]["count"],
        "advancement_authorized_pending_independent_audit": result["advancement_gate"]["advancement_authorized_pending_independent_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

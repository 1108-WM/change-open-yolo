#!/usr/bin/env python3
"""Independently audit stage-D OOF predictions, q90 bounds, and score plan."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import _read_jsonl, _resolve, _sha256  # noqa: E402
from tools.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import FEATURE_NAMES  # noqa: E402
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import regression_metrics  # noqa: E402


VERSION = "ncs_fi1_stage_d_marginal_gain_oof_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def run(args: argparse.Namespace) -> dict:
    for name in ("dataset_root", "dataset_audit_root", "result_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary = json.loads((args.dataset_root / "summary.json").read_text())
    dataset_audit = json.loads((args.dataset_audit_root / "summary.json").read_text())
    result_summary_path = args.result_root / "summary.json"
    result_summary = json.loads(result_summary_path.read_text())
    dataset_path = args.dataset_root / dataset_summary["files"]["dataset"]
    prediction_path = args.result_root / result_summary["files"]["predictions"]
    plan_path = args.result_root / result_summary["files"]["plan"]
    rows = _read_jsonl(dataset_path)
    predictions = _read_jsonl(prediction_path)
    plan = _read_jsonl(plan_path)
    errors = Counter()
    if dataset_audit.get("audit_valid") is not True or dataset_audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        errors["dataset_audit_not_authorized"] += 1
    if _sha256(prediction_path) != result_summary["hashes"]["predictions"]:
        errors["prediction_sha256_mismatch"] += 1
    if _sha256(plan_path) != result_summary["hashes"]["plan"]:
        errors["plan_sha256_mismatch"] += 1
    if len(rows) != len(predictions) or len(rows) != len(plan):
        errors["row_count_mismatch"] += 1
    if tuple(result_summary["feature_names"]) != FEATURE_NAMES:
        errors["feature_schema_mismatch"] += 1
    if int(result_summary.get("scene_count", -1)) != int(dataset_summary["scene_count"]):
        errors["result_scene_count_mismatch"] += 1
    expected_scene_with_candidates = len({str(row["scene_name"]) for row in rows})
    if int(result_summary.get("scene_with_candidate_count", -1)) != expected_scene_with_candidates:
        errors["result_scene_with_candidate_count_mismatch"] += 1

    matrix = np.asarray([
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in rows
    ], dtype=np.float64)
    target = np.asarray([float(row["labels"]["marginal_iou_gain"]) for row in rows])
    folds = np.asarray([int(row["fold_index"]) for row in rows], dtype=np.int8)
    threshold_control = np.asarray([
        float(row["features"]["frozen_threshold_cross_probability"]) for row in rows
    ])
    geometry_quality = np.asarray([
        float(row["features"]["geometry_quality_for_scoring"]) for row in rows
    ])
    raw = np.full(len(rows), np.nan)
    corrected = np.full(len(rows), np.nan)
    q90 = np.full(len(rows), np.nan)
    models = {}
    for record in result_summary["models"]:
        path = args.result_root / record["model_file"]
        if _sha256(path) != record["model_sha256"]:
            errors["model_sha256_mismatch"] += 1
        payload = joblib.load(path)
        fold = int(record["outer_fold"])
        if fold in models:
            errors["duplicate_model_fold"] += 1
        models[fold] = payload
        if int(payload["calibration_fold"]) != (fold + 1) % 5:
            errors["calibration_fold_contract_mismatch"] += 1
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            errors["model_feature_schema_mismatch"] += 1
        if not _close(payload["calibration_bias"], record["calibration_bias"]):
            errors["model_bias_record_mismatch"] += 1
        if not _close(payload["calibration_absolute_residual_q90"], record["calibration_absolute_residual_q90"]):
            errors["model_q90_record_mismatch"] += 1
    if set(models) != set(range(5)):
        errors["model_fold_coverage_mismatch"] += 1

    for fold in range(5):
        selected = np.flatnonzero(folds == fold)
        payload = models[fold]
        local_raw = np.asarray(payload["regressor"].predict(matrix[selected]), dtype=np.float64)
        local_corrected = np.clip(local_raw + float(payload["calibration_bias"]), 0.0, 1.0)
        raw[selected] = local_raw
        corrected[selected] = local_corrected
        q90[selected] = float(payload["calibration_absolute_residual_q90"])
    if not np.isfinite(raw).all() or not np.isfinite(corrected).all() or not np.isfinite(q90).all():
        errors["nonfinite_recomputed_prediction"] += 1
    lower = corrected - q90
    conservative = np.maximum(0.0, lower)
    scores = geometry_quality * conservative

    plan_by_key = {str(row["candidate_key"]): row for row in plan}
    if len(plan_by_key) != len(plan):
        errors["duplicate_plan_candidate_key"] += 1
    for index, (source, prediction) in enumerate(zip(rows, predictions)):
        key = str(source["candidate_key"])
        if str(prediction["candidate_key"]) != key:
            errors["prediction_identity_mismatch"] += 1
            continue
        expected_values = {
            "label_marginal_iou_gain": target[index],
            "frozen_threshold_cross_probability_control": threshold_control[index],
            "raw_oof_marginal_iou_gain": raw[index],
            "corrected_oof_marginal_iou_gain": corrected[index],
            "calibration_absolute_residual_q90": q90[index],
            "marginal_gain_lower_confidence_bound": lower[index],
            "conservative_gain": conservative[index],
            "geometry_quality_for_scoring": geometry_quality[index],
            "stage_d_append_score": scores[index],
        }
        for name, expected in expected_values.items():
            if not _close(prediction.get(name), expected):
                errors[f"prediction_{name}_mismatch"] += 1
        planned = plan_by_key.get(key)
        if planned is None:
            errors["plan_candidate_missing"] += 1
        else:
            for name in (
                "frozen_threshold_cross_probability_control", "conservative_gain",
                "geometry_quality_for_scoring", "stage_d_append_score",
            ):
                if not _close(planned.get(name), expected_values[name]):
                    errors[f"plan_{name}_mismatch"] += 1
        for contract_row in (prediction, planned or {}):
            for flag in (
                "candidate_deletion", "candidate_mutation", "geometry_mutation",
                "class_mutation", "frozen_cache_write", "ap_computed",
            ):
                if contract_row.get(flag) is not False:
                    errors[f"contract_{flag}_violation"] += 1
            if contract_row.get("candidate_retained") is not True:
                errors["candidate_not_retained"] += 1
    if np.any(scores < 0.0) or np.any(scores > geometry_quality + 1e-12):
        errors["score_quality_bound_violation"] += 1

    overall = {
        "oof_prediction": regression_metrics(target, corrected),
        "zero_prediction_control": regression_metrics(target, np.zeros(len(target))),
        "frozen_threshold_cross_probability_control": regression_metrics(target, threshold_control),
    }
    for metric_group, values in overall.items():
        recorded = result_summary["overall_metrics"][metric_group]
        for name, expected in values.items():
            if not _close(recorded[name], expected):
                errors[f"summary_{metric_group}_{name}_mismatch"] += 1
    confident = conservative > 0.0
    confident_count = int(np.sum(confident))
    true_positive_count = int(np.sum(confident & (target > 0.0)))
    precision = true_positive_count / max(1, confident_count)
    fold_confident = {fold: int(np.sum((folds == fold) & confident)) for fold in range(5)}
    checks = {
        "audit_error_count_zero": sum(errors.values()) == 0,
        "dataset_audit_valid": dataset_audit.get("audit_valid") is True,
        "oof_mae_strictly_better_than_zero_control": (
            overall["oof_prediction"]["mae"] < overall["zero_prediction_control"]["mae"]
        ),
        "oof_mae_strictly_better_than_threshold_cross_probability_control": (
            overall["oof_prediction"]["mae"] < overall["frozen_threshold_cross_probability_control"]["mae"]
        ),
        "oof_spearman_strictly_positive": overall["oof_prediction"]["spearman"] > 0.0,
        "confident_positive_count_greater_than_zero": confident_count > 0,
        "confident_true_positive_fraction_at_least_0_70": precision >= 0.70,
        "score_contract_valid": bool(
            np.isfinite(scores).all() and np.all(scores >= 0.0)
            and np.all(scores <= geometry_quality + 1e-12)
        ),
        **{f"fold_{fold}_has_confident_positive": fold_confident[fold] > 0 for fold in range(5)},
        "candidate_deletion_count_zero": all(row.get("candidate_deletion") is False for row in plan),
        "geometry_mutation_count_zero": all(row.get("geometry_mutation") is False for row in plan),
        "class_mutation_count_zero": all(row.get("class_mutation") is False for row in plan),
        "frozen_cache_write_count_zero": all(row.get("frozen_cache_write") is False for row in plan),
        "ap_computed_false": result_summary.get("ap_computed") is False,
        "validation60_read_false": result_summary.get("validation60_read") is False,
        "val312_read_false": result_summary.get("val312_read") is False,
    }
    advancement = all(checks.values())
    audit = {
        "version": VERSION,
        "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        "overall_metrics": overall,
        "confident_positive": {
            "count": confident_count,
            "true_positive_count": true_positive_count,
            "neutral_count": confident_count - true_positive_count,
            "true_positive_fraction": precision,
            "by_fold": {str(fold): fold_confident[fold] for fold in range(5)},
        },
        "advancement_gate": {
            "checks": checks,
            "advancement_authorized": advancement,
        },
        "candidate_deletion_count": 0,
        "geometry_mutation": False,
        "class_mutation": False,
        "frozen_cache_write": False,
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "dataset_summary_sha256": _sha256(args.dataset_root / "summary.json"),
            "dataset_audit_summary_sha256": _sha256(args.dataset_audit_root / "summary.json"),
            "result_summary_sha256": _sha256(result_summary_path),
            "predictions_sha256": _sha256(prediction_path),
            "plan_sha256": _sha256(plan_path),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "audit_valid": result["audit_valid"],
        "error_count": result["error_count"],
        "advancement_authorized": result["advancement_gate"]["advancement_authorized"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

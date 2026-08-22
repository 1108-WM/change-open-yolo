#!/usr/bin/env python3
"""Train stage-D-v2 rank-conditioned continuous marginal-gain OOF model."""

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
from tools.build_ncs_fi1_stage_d_v2_rank_marginal_dataset_gt import FEATURE_NAMES  # noqa: E402
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import regression_metrics  # noqa: E402
from tools.train_ncs_fi1_stage_d_marginal_gain_oof import MODEL_PARAMS  # noqa: E402


VERSION = "ncs_fi1_stage_d_v2_rank_marginal_oof_v1"
STAGE_LABEL = "stage-D-v2"
SCORE_FIELD = "stage_d_v2_append_score"
PLAN_FILE = "stage_d_v2_append_score_plan.jsonl"


def run(args: argparse.Namespace) -> dict:
    for name in ("dataset_root", "dataset_audit_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary_path = args.dataset_root / "summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    audit_path = args.dataset_audit_root / "summary.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("audit_valid") is not True or audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError(f"{STAGE_LABEL} dataset audit did not authorize training")
    dataset_path = args.dataset_root / dataset_summary["files"]["dataset"]
    rows = _read_jsonl(dataset_path)
    if set(dataset_summary["feature_names"]) != set(FEATURE_NAMES):
        raise ValueError(f"{STAGE_LABEL} feature schema differs")
    matrix = np.asarray([[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows])
    target = np.asarray([float(row["labels"]["rank_conditioned_marginal_iou_gain"]) for row in rows])
    folds = np.asarray([int(row["fold_index"]) for row in rows], dtype=np.int8)
    threshold_control = np.asarray([float(row["features"]["frozen_threshold_cross_probability"]) for row in rows])
    reference_score = np.asarray([float(row["candidate_reference_score"]) for row in rows])

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    model_root = staging / "models"
    model_root.mkdir()
    try:
        raw = np.full(len(rows), np.nan)
        corrected = np.full(len(rows), np.nan)
        q90 = np.full(len(rows), np.nan)
        models = []
        for outer_fold in range(5):
            calibration_fold = (outer_fold + 1) % 5
            fit = np.flatnonzero((folds != outer_fold) & (folds != calibration_fold))
            calibration = np.flatnonzero(folds == calibration_fold)
            test = np.flatnonzero(folds == outer_fold)
            model = HistGradientBoostingRegressor(random_state=20260842 + outer_fold, **MODEL_PARAMS)
            model.fit(matrix[fit], target[fit])
            calibration_raw = np.asarray(model.predict(matrix[calibration]))
            test_raw = np.asarray(model.predict(matrix[test]))
            bias = float(np.mean(target[calibration] - calibration_raw))
            calibration_corrected = np.clip(calibration_raw + bias, 0.0, 1.0)
            test_corrected = np.clip(test_raw + bias, 0.0, 1.0)
            local_q90 = float(np.quantile(np.abs(calibration_corrected - target[calibration]), 0.90))
            raw[test] = test_raw
            corrected[test] = test_corrected
            q90[test] = local_q90
            model_path = model_root / f"rank_marginal_gain_fold_{outer_fold}.joblib"
            joblib.dump({
                "version": VERSION,
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "feature_names": FEATURE_NAMES,
                "model_params": {**MODEL_PARAMS, "random_state": 20260842 + outer_fold},
                "regressor": model,
                "calibration_bias": bias,
                "calibration_absolute_residual_q90": local_q90,
                "prediction_clip": (0.0, 1.0),
            }, model_path)
            models.append({
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "fit_count": len(fit),
                "calibration_count": len(calibration),
                "test_count": len(test),
                "fit_folds": sorted(set(map(int, folds[fit]))),
                "calibration_bias": bias,
                "calibration_absolute_residual_q90": local_q90,
                "model_file": str(Path("models") / model_path.name),
                "model_sha256": _sha256(model_path),
            })
        if not np.isfinite(raw).all() or not np.isfinite(corrected).all() or not np.isfinite(q90).all():
            raise ValueError(f"{STAGE_LABEL} OOF output incomplete")
        lower = corrected - q90
        conservative = np.maximum(0.0, lower)
        score = reference_score * conservative
        predictions = []
        for index, row in enumerate(rows):
            predictions.append({
                "candidate_key": row["candidate_key"],
                "scene_name": row["scene_name"],
                "fold_index": int(row["fold_index"]),
                "union_candidate_id": int(row["union_candidate_id"]),
                "candidate_variant": row["candidate_variant"],
                "geometry_sha256": row["geometry_sha256"],
                "label_rank_conditioned_marginal_iou_gain": float(target[index]),
                "frozen_threshold_cross_probability_control": float(threshold_control[index]),
                "raw_oof_rank_marginal_iou_gain": float(raw[index]),
                "corrected_oof_rank_marginal_iou_gain": float(corrected[index]),
                "calibration_absolute_residual_q90": float(q90[index]),
                "rank_marginal_gain_lower_confidence_bound": float(lower[index]),
                "conservative_gain": float(conservative[index]),
                "candidate_reference_score": float(reference_score[index]),
                SCORE_FIELD: float(score[index]),
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
        fold_confident = Counter()
        for fold in range(5):
            selected = folds == fold
            fold_confident[fold] = int(np.sum(selected & (conservative > 0.0)))
            fold_metrics[str(fold)] = {
                "oof_prediction": regression_metrics(target[selected], corrected[selected]),
                "zero_prediction_control": regression_metrics(target[selected], np.zeros(np.sum(selected))),
                "frozen_threshold_cross_probability_control": regression_metrics(target[selected], threshold_control[selected]),
                "positive_target_count": int(np.sum(target[selected] > 0.0)),
                "confident_positive_count": fold_confident[fold],
            }
        confident = conservative > 0.0
        confident_count = int(np.sum(confident))
        true_positive = int(np.sum(confident & (target > 0.0)))
        precision = true_positive / max(1, confident_count)
        checks = {
            "dataset_audit_valid": True,
            "oof_mae_strictly_better_than_zero_control": overall["oof_prediction"]["mae"] < overall["zero_prediction_control"]["mae"],
            "oof_mae_strictly_better_than_threshold_cross_probability_control": overall["oof_prediction"]["mae"] < overall["frozen_threshold_cross_probability_control"]["mae"],
            "oof_spearman_strictly_positive": overall["oof_prediction"]["spearman"] > 0.0,
            "confident_positive_count_greater_than_zero": confident_count > 0,
            "confident_true_positive_fraction_at_least_0_70": precision >= 0.70,
            "score_contract_valid": bool(np.isfinite(score).all() and np.all(score >= 0.0) and np.all(score <= reference_score + 1e-12)),
            **{f"fold_{fold}_has_confident_positive": fold_confident[fold] > 0 for fold in range(5)},
            "candidate_deletion_count_zero": True,
            "geometry_mutation_count_zero": True,
            "class_mutation_count_zero": True,
            "frozen_cache_write_count_zero": True,
            "ap_computed_false": True,
            "validation60_read_false": True,
            "val312_read_false": True,
        }
        prediction_path = staging / "oof_rank_marginal_gain_predictions.jsonl"
        plan_path = staging / PLAN_FILE
        prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        plan_path.write_text("".join(
            json.dumps({
                "candidate_key": row["candidate_key"],
                "scene_name": row["scene_name"],
                "fold_index": row["fold_index"],
                "union_candidate_id": row["union_candidate_id"],
                "candidate_variant": row["candidate_variant"],
                "geometry_sha256": row["geometry_sha256"],
                "candidate_reference_score": row["candidate_reference_score"],
                "conservative_gain": row["conservative_gain"],
                SCORE_FIELD: row[SCORE_FIELD],
                "frozen_threshold_cross_probability_control": row["frozen_threshold_cross_probability_control"],
                "candidate_retained": True,
                "candidate_deletion": False,
                "candidate_mutation": False,
                "geometry_mutation": False,
                "class_mutation": False,
                "frozen_cache_write": False,
                "ap_computed": False,
            }, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions
        ))
        summary = {
            "version": VERSION,
            "scene_count": int(dataset_summary["scene_count"]),
            "scene_with_candidate_count": len(set(row["scene_name"] for row in rows)),
            "candidate_count": len(rows),
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "model_params": MODEL_PARAMS,
            "calibration_contract": "fixed next-fold additive mean-residual correction; absolute residual q90; clip [0,1]",
            "score_contract": "candidate_reference_score * max(0,corrected_rank_gain-q90)",
            "overall_metrics": overall,
            "fold_metrics": fold_metrics,
            "confident_positive": {
                "count": confident_count,
                "true_positive_count": true_positive,
                "neutral_count": confident_count - true_positive,
                "true_positive_fraction": precision,
                "by_fold": {str(fold): int(fold_confident[fold]) for fold in range(5)},
            },
            "score_distribution": {
                "nonzero_count": int(np.sum(score > 0.0)),
                "mean": float(np.mean(score)),
                "max": float(np.max(score)),
            },
            "advancement_gate": {
                "checks": checks,
                "advancement_authorized_pending_independent_audit": all(checks.values()),
            },
            "models": models,
            "files": {"predictions": prediction_path.name, "plan": plan_path.name},
            "hashes": {"predictions": _sha256(prediction_path), "plan": _sha256(plan_path)},
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
                "dataset_audit_summary_sha256": _sha256(audit_path),
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
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
    result = run(parser.parse_args())
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "oof_mae": result["overall_metrics"]["oof_prediction"]["mae"],
        "zero_control_mae": result["overall_metrics"]["zero_prediction_control"]["mae"],
        "confident_positive_count": result["confident_positive"]["count"],
        "advancement_authorized_pending_independent_audit": result["advancement_gate"]["advancement_authorized_pending_independent_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

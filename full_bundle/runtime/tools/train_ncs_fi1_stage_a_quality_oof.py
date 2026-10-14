#!/usr/bin/env python3
"""Train the single preregistered NCS first-innovation stage-A OOF models.

The script consumes an independently audited train100-only dataset, produces
one calibrated OOF quality prediction per frozen geometry, and writes no AP or
candidate mutation plan.
"""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    FEATURE_NAMES_BY_SOURCE,
    SOURCE_NAMES,
    _read_jsonl,
    _resolve,
    _sha256,
)


VERSION = "ncs_fi1_stage_a_quality_oof_v1"
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 2:
        return None
    left, right = _rank(target), _rank(prediction)
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def reliability(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    bins = []
    weighted_gap = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        select = (prediction >= lower) & (
            prediction <= upper if index == len(edges) - 2 else prediction < upper
        )
        count = int(select.sum())
        if count:
            mean_prediction = float(prediction[select].mean())
            mean_target = float(target[select].mean())
            gap = abs(mean_prediction - mean_target)
            weighted_gap += count * gap
        else:
            mean_prediction = mean_target = gap = None
        bins.append({
            "bin_index": index,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "mean_prediction": mean_prediction,
            "mean_target": mean_target,
            "absolute_gap": gap,
        })
    return {
        "expected_calibration_error": float(weighted_gap / max(1, len(target))),
        "bins": bins,
    }


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    error = prediction - target
    return {
        "count": len(target),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "spearman": _spearman(target, prediction),
        "mean_prediction": float(prediction.mean()),
        "mean_target": float(target.mean()),
        "absolute_mean_bias": float(abs(prediction.mean() - target.mean())),
        "reliability": reliability(target, prediction),
    }


def _model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **MODEL_PARAMS)


def _fit_calibrator(raw: np.ndarray, target: np.ndarray) -> tuple[object, str]:
    raw = np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)
    target = np.asarray(target, dtype=np.float64)
    if len(raw) >= 2 and len(np.unique(raw)) >= 2 and len(np.unique(target)) >= 2:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(raw, target)
        return calibrator, "isotonic"
    return float(target.mean()) if len(target) else 0.0, "constant_calibration_fallback"


def _calibrate(calibrator: object, values: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, float):
        return np.full(len(values), calibrator, dtype=np.float64)
    return np.asarray(calibrator.predict(np.asarray(values, dtype=np.float64)), dtype=np.float64)


def _advancement_gate(overall: dict, by_source: dict, by_fold: dict) -> dict:
    checks = {
        "overall_mae_strictly_better_than_frozen_score": (
            overall["unified_quality"]["mae"] < overall["frozen_score_control"]["mae"]
        ),
    }
    for source in SOURCE_NAMES:
        checks[f"{source}_mae_not_worse_than_frozen_by_more_than_0.02"] = (
            by_source[source]["unified_quality"]["mae"]
            <= by_source[source]["frozen_score_control"]["mae"] + 0.02
        )
        checks[f"{source}_absolute_mean_bias_at_most_0.10"] = (
            by_source[source]["unified_quality"]["absolute_mean_bias"] <= 0.10
        )
    for fold in range(5):
        checks[f"fold_{fold}_mae_at_most_0.30"] = (
            by_fold[str(fold)]["unified_quality"]["mae"] <= 0.30
        )
        for source in SOURCE_NAMES:
            checks[f"fold_{fold}_{source}_has_samples"] = (
                by_fold[str(fold)]["by_source"][source]["count"] > 0
            )
    return {"checks": checks, "advancement_authorized": all(checks.values())}


def run(args: argparse.Namespace) -> dict:
    for name in ("dataset_root", "dataset_audit_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary_path = args.dataset_root / "summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    audit_summary_path = args.dataset_audit_root / "summary.json"
    audit = json.loads(audit_summary_path.read_text())
    if audit.get("audit_valid") is not True or int(audit.get("error_count", -1)) != 0:
        raise ValueError("stage-A dataset audit is not valid")
    rows = _read_jsonl(args.dataset_root / dataset_summary["dataset_file"])
    if len(rows) != int(dataset_summary["geometry_count"]):
        raise ValueError("stage-A dataset row count differs from summary")
    row_ids = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate stage-A dataset row id")

    target = np.asarray([float(row["label_quality_q"]) for row in rows], dtype=np.float64)
    frozen = np.clip(np.asarray(
        [float(row["frozen_score_metadata_only"]) for row in rows], dtype=np.float64
    ), 0.0, 1.0)
    folds = np.asarray([int(row["fold_index"]) for row in rows], dtype=np.int8)
    sources = np.asarray([str(row["candidate_source"]) for row in rows], dtype=object)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    raw_predictions = np.full(len(rows), np.nan, dtype=np.float64)
    model_records = []
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    model_root = staging / "models"
    model_root.mkdir()
    try:
        for outer_fold in range(5):
            calibration_fold = (outer_fold + 1) % 5
            for source in SOURCE_NAMES:
                source_select = sources == source
                fit_indexes = np.flatnonzero(source_select & (folds != outer_fold) & (folds != calibration_fold))
                calibration_indexes = np.flatnonzero(source_select & (folds == calibration_fold))
                test_indexes = np.flatnonzero(source_select & (folds == outer_fold))
                if not len(fit_indexes) or not len(calibration_indexes) or not len(test_indexes):
                    raise ValueError(
                        f"empty fit/calibration/test split for fold {outer_fold}, source {source}"
                    )
                feature_names = FEATURE_NAMES_BY_SOURCE[source]
                matrix = np.asarray([
                    [float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows
                ], dtype=np.float64)
                if not np.isfinite(matrix).all():
                    raise ValueError(f"non-finite feature matrix for {source}")
                model = _model(20260822 + outer_fold)
                model.fit(matrix[fit_indexes], target[fit_indexes])
                calibration_raw = np.clip(model.predict(matrix[calibration_indexes]), 0.0, 1.0)
                test_raw = np.clip(model.predict(matrix[test_indexes]), 0.0, 1.0)
                calibrator, calibration_kind = _fit_calibrator(
                    calibration_raw, target[calibration_indexes]
                )
                test_calibrated = np.clip(_calibrate(calibrator, test_raw), 0.0, 1.0)
                raw_predictions[test_indexes] = test_raw
                predictions[test_indexes] = test_calibrated
                model_path = model_root / f"fold_{outer_fold}_{source}.joblib"
                joblib.dump({
                    "version": VERSION,
                    "outer_fold": outer_fold,
                    "calibration_fold": calibration_fold,
                    "candidate_source": source,
                    "feature_names": feature_names,
                    "model_params": {**MODEL_PARAMS, "random_state": 20260822 + outer_fold},
                    "regressor": model,
                    "calibrator": calibrator,
                    "calibration_kind": calibration_kind,
                }, model_path)
                model_records.append({
                    "outer_fold": outer_fold,
                    "calibration_fold": calibration_fold,
                    "candidate_source": source,
                    "fit_count": len(fit_indexes),
                    "calibration_count": len(calibration_indexes),
                    "test_count": len(test_indexes),
                    "calibration_kind": calibration_kind,
                    "model_file": str(Path("models") / model_path.name),
                    "model_sha256": _sha256(model_path),
                    "fit_folds": sorted(set(int(folds[index]) for index in fit_indexes)),
                })
        if not np.isfinite(predictions).all() or not np.isfinite(raw_predictions).all():
            raise ValueError("OOF prediction coverage is incomplete or non-finite")

        output_rows = []
        for index, row in enumerate(rows):
            output_rows.append({
                "row_id": row["row_id"],
                "scene_name": row["scene_name"],
                "fold_index": int(row["fold_index"]),
                "calibration_fold_index": (int(row["fold_index"]) + 1) % 5,
                "geometry_key": row["geometry_key"],
                "geometry_hash": row["geometry_hash"],
                "candidate_source": row["candidate_source"],
                "label_quality_q": float(target[index]),
                "frozen_score_control": float(frozen[index]),
                "raw_oof_quality": float(raw_predictions[index]),
                "oof_unified_quality": float(predictions[index]),
                "candidate_mutation": False,
                "geometry_mutation": False,
                "score_mutation": False,
                "ap_computed": False,
            })
        prediction_path = staging / "oof_quality_predictions.jsonl"
        prediction_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows
        ))

        overall = {
            "unified_quality": metrics(target, predictions),
            "raw_quality": metrics(target, raw_predictions),
            "frozen_score_control": metrics(target, frozen),
        }
        by_source = {}
        for source in SOURCE_NAMES:
            select = sources == source
            by_source[source] = {
                "unified_quality": metrics(target[select], predictions[select]),
                "raw_quality": metrics(target[select], raw_predictions[select]),
                "frozen_score_control": metrics(target[select], frozen[select]),
            }
        by_fold = {}
        for fold in range(5):
            select = folds == fold
            by_fold[str(fold)] = {
                "unified_quality": metrics(target[select], predictions[select]),
                "raw_quality": metrics(target[select], raw_predictions[select]),
                "frozen_score_control": metrics(target[select], frozen[select]),
                "by_source": {
                    source: metrics(
                        target[select & (sources == source)],
                        predictions[select & (sources == source)],
                    ) for source in SOURCE_NAMES
                },
            }
        gates = _advancement_gate(overall, by_source, by_fold)
        summary = {
            "version": VERSION,
            "dataset_name": dataset_summary.get("dataset_name", "NCS-train100"),
            "geometry_count": len(rows),
            "oof_prediction_count": len(output_rows),
            "source_counts": dict(sorted(Counter(sources.tolist()).items())),
            "model_params": MODEL_PARAMS,
            "calibration_contract": "fixed next fold isotonic; constant fallback only for degenerate calibration",
            "overall_metrics": overall,
            "source_metrics": by_source,
            "fold_metrics": by_fold,
            "advancement_gate": gates,
            "models": model_records,
            "prediction_file": prediction_path.name,
            "prediction_sha256": _sha256(prediction_path),
            "ground_truth_usage": f"{dataset_summary.get('dataset_name', 'NCS-train100')} labels from frozen dataset only",
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation": False,
            "replacement_plan_generated": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "dataset_summary_sha256": _sha256(dataset_summary_path),
                "dataset_audit_sha256": _sha256(audit_summary_path),
                "dataset_sha256": dataset_summary["dataset_sha256"],
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
        "output_root": str(_resolve(args.output_root)),
        "geometry_count": result["geometry_count"],
        "overall_mae": result["overall_metrics"]["unified_quality"]["mae"],
        "advancement_authorized": result["advancement_gate"]["advancement_authorized"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

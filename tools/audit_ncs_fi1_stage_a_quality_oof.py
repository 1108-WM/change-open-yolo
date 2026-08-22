#!/usr/bin/env python3
"""Independently audit stage-A OOF coverage, metrics, models, and invariants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    SOURCE_NAMES,
    _read_jsonl,
    _resolve,
    _sha256,
)


VERSION = "ncs_fi1_stage_a_quality_oof_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _basic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "count": len(error),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_prediction": float(np.mean(prediction)),
        "mean_target": float(np.mean(target)),
        "absolute_mean_bias": float(abs(np.mean(prediction) - np.mean(target))),
    }


def _compare_metrics(expected: dict, actual: dict, errors: Counter, prefix: str) -> None:
    for name, value in actual.items():
        if name == "count":
            if int(expected.get(name, -1)) != int(value):
                errors[f"{prefix}_count_mismatch"] += 1
        elif not _close(expected.get(name), value):
            errors[f"{prefix}_{name}_mismatch"] += 1


def run(args: argparse.Namespace) -> dict:
    for name in ("dataset_root", "dataset_audit_root", "oof_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary_path = args.dataset_root / "summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    dataset_audit_path = args.dataset_audit_root / "summary.json"
    dataset_audit = json.loads(dataset_audit_path.read_text())
    oof_summary_path = args.oof_root / "summary.json"
    oof_summary = json.loads(oof_summary_path.read_text())
    dataset_rows = _read_jsonl(args.dataset_root / dataset_summary["dataset_file"])
    prediction_path = args.oof_root / oof_summary["prediction_file"]
    predictions = _read_jsonl(prediction_path)
    errors = Counter()
    if dataset_audit.get("audit_valid") is not True:
        errors["dataset_audit_invalid"] += 1
    if len(predictions) != len(dataset_rows):
        errors["prediction_count_mismatch"] += 1
    dataset_by_id = {str(row["row_id"]): row for row in dataset_rows}
    if len(dataset_by_id) != len(dataset_rows):
        errors["duplicate_dataset_row_id"] += 1
    prediction_by_id = {}
    source_counts = Counter()
    fold_source_counts = Counter()
    for row in predictions:
        row_id = str(row.get("row_id"))
        if row_id in prediction_by_id:
            errors["duplicate_prediction_row_id"] += 1
        prediction_by_id[row_id] = row
        source = str(row.get("candidate_source"))
        fold = int(row.get("fold_index", -1))
        if source not in SOURCE_NAMES:
            errors["unexpected_source"] += 1
        source_counts[source] += 1
        fold_source_counts[(fold, source)] += 1
        original = dataset_by_id.get(row_id)
        if original is None:
            errors["prediction_without_dataset_row"] += 1
            continue
        for name in ("scene_name", "geometry_key", "geometry_hash", "candidate_source"):
            if row.get(name) != original.get(name):
                errors[f"join_{name}_mismatch"] += 1
        if fold != int(original["fold_index"]):
            errors["fold_mismatch"] += 1
        if int(row.get("calibration_fold_index", -1)) != (fold + 1) % 5:
            errors["calibration_fold_mismatch"] += 1
        if not _close(row.get("label_quality_q"), original["label_quality_q"]):
            errors["label_mismatch"] += 1
        if not _close(row.get("frozen_score_control"), original["frozen_score_metadata_only"]):
            errors["frozen_score_mismatch"] += 1
        for name in ("raw_oof_quality", "oof_unified_quality"):
            try:
                value = float(row[name])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    errors[f"invalid_{name}"] += 1
            except (KeyError, TypeError, ValueError):
                errors[f"invalid_{name}"] += 1
        for name in ("candidate_mutation", "geometry_mutation", "score_mutation"):
            if row.get(name) is not False:
                errors[name] += 1
        if row.get("ap_computed") is not False:
            errors["ap_computed"] += 1
    if set(prediction_by_id) != set(dataset_by_id):
        errors["prediction_coverage_mismatch"] += 1
    if _sha256(prediction_path) != oof_summary.get("prediction_sha256"):
        errors["prediction_sha256_mismatch"] += 1

    model_records = list(oof_summary.get("models", []))
    if len(model_records) != 15:
        errors["model_record_count_mismatch"] += 1
    seen_models = set()
    for record in model_records:
        key = (int(record["outer_fold"]), str(record["candidate_source"]))
        if key in seen_models:
            errors["duplicate_model_record"] += 1
        seen_models.add(key)
        outer_fold, source = key
        if int(record["calibration_fold"]) != (outer_fold + 1) % 5:
            errors["model_calibration_fold_mismatch"] += 1
        if outer_fold in record["fit_folds"] or int(record["calibration_fold"]) in record["fit_folds"]:
            errors["model_scene_fold_leakage"] += 1
        if sorted(record["fit_folds"]) != sorted(
            set(range(5)) - {outer_fold, int(record["calibration_fold"])}
        ):
            errors["model_fit_fold_contract_mismatch"] += 1
        model_path = args.oof_root / record["model_file"]
        if not model_path.is_file() or _sha256(model_path) != record["model_sha256"]:
            errors["model_file_or_sha256_mismatch"] += 1
        if int(record["test_count"]) != fold_source_counts[(outer_fold, source)]:
            errors["model_test_count_mismatch"] += 1

    ordered = [prediction_by_id[row["row_id"]] for row in dataset_rows if row["row_id"] in prediction_by_id]
    if len(ordered) == len(dataset_rows):
        target = np.asarray([float(row["label_quality_q"]) for row in ordered])
        unified = np.asarray([float(row["oof_unified_quality"]) for row in ordered])
        raw = np.asarray([float(row["raw_oof_quality"]) for row in ordered])
        frozen = np.asarray([float(row["frozen_score_control"]) for row in ordered])
        sources = np.asarray([str(row["candidate_source"]) for row in ordered], dtype=object)
        folds = np.asarray([int(row["fold_index"]) for row in ordered], dtype=np.int8)
        for name, values in (
            ("unified_quality", unified), ("raw_quality", raw), ("frozen_score_control", frozen)
        ):
            _compare_metrics(
                oof_summary["overall_metrics"][name], _basic_metrics(target, values),
                errors, f"overall_{name}",
            )
        for source in SOURCE_NAMES:
            select = sources == source
            for name, values in (
                ("unified_quality", unified), ("raw_quality", raw),
                ("frozen_score_control", frozen),
            ):
                _compare_metrics(
                    oof_summary["source_metrics"][source][name],
                    _basic_metrics(target[select], values[select]),
                    errors, f"source_{source}_{name}",
                )
        for fold in range(5):
            select = folds == fold
            for name, values in (
                ("unified_quality", unified), ("raw_quality", raw),
                ("frozen_score_control", frozen),
            ):
                _compare_metrics(
                    oof_summary["fold_metrics"][str(fold)][name],
                    _basic_metrics(target[select], values[select]),
                    errors, f"fold_{fold}_{name}",
                )

    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "dataset_row_count": len(dataset_rows),
        "prediction_row_count": len(predictions),
        "model_record_count": len(model_records),
        "source_counts": dict(sorted(source_counts.items())),
        "fold_source_counts": {
            str(fold): {source: int(fold_source_counts[(fold, source)]) for source in SOURCE_NAMES}
            for fold in range(5)
        },
        "advancement_authorized_as_recorded": bool(
            oof_summary.get("advancement_gate", {}).get("advancement_authorized", False)
        ),
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "replacement_plan_generated": False,
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "dataset_summary_sha256": _sha256(dataset_summary_path),
            "dataset_audit_sha256": _sha256(dataset_audit_path),
            "oof_summary_sha256": _sha256(oof_summary_path),
            "prediction_sha256": _sha256(prediction_path),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

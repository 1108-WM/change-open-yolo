#!/usr/bin/env python3
"""Fit the frozen FI1-D-v3 deployment models on all official100 rows.

The regressors use every official100 training row.  Calibration is deliberately
derived from the already frozen OOF predictions, never from an in-sample
prediction of the final regressor.  The resulting package is therefore suitable
for a single no-GT transfer run on ScanNet200 val312.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    FEATURE_NAMES_BY_SOURCE,
)
from tools.build_ncs_fi1_stage_d_v3_full_rank_marginal_dataset_gt import (  # noqa: E402
    FEATURE_NAMES as STAGE_D_FEATURE_NAMES,
)
from tools.train_ncs_fi1_stage_a_quality_oof import (  # noqa: E402
    MODEL_PARAMS as STAGE_A_MODEL_PARAMS,
    _fit_calibrator,
)
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import (  # noqa: E402
    MODEL_PARAMS as STAGE_C_MODEL_PARAMS,
    QUALITY_FEATURE_NAMES,
)
from tools.train_ncs_fi1_stage_d_marginal_gain_oof import (  # noqa: E402
    MODEL_PARAMS as STAGE_D_MODEL_PARAMS,
)


VERSION = "fi1_d_v3_official100_full_models_v1"
SOURCE_NAMES = ("native", "track", "pair_union")
EXCLUSIVE_ROLES = ("track_only", "native_only")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {path}:{line_number}") from error


def _write_bundle(path: Path, payload: dict) -> dict:
    joblib.dump(payload, path)
    return {
        "model_kind": payload["model_kind"],
        "model_file": str(Path("models") / path.name),
        "model_sha256": _sha256(path),
        "fit_count": int(payload["fit_count"]),
        "calibration_count": int(payload["calibration_count"]),
    }


def _fit_residual_bundle(
    *, matrix: np.ndarray, target: np.ndarray, oof_raw: np.ndarray,
    feature_names: tuple[str, ...], model_kind: str, seed: int,
    model_params: dict, clip: tuple[float, float] | None, output_path: Path,
) -> dict:
    if not (len(matrix) == len(target) == len(oof_raw)) or not len(target):
        raise ValueError(f"{model_kind}: training/calibration row counts differ or are empty")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all() or not np.isfinite(oof_raw).all():
        raise ValueError(f"{model_kind}: non-finite training data")
    regressor = HistGradientBoostingRegressor(random_state=seed, **model_params)
    regressor.fit(matrix, target)
    bias = float(np.mean(target - oof_raw))
    corrected = oof_raw + bias
    if clip is not None:
        corrected = np.clip(corrected, clip[0], clip[1])
    q90 = float(np.quantile(np.abs(corrected - target), 0.90))
    return _write_bundle(output_path, {
        "version": VERSION,
        "model_kind": model_kind,
        "fit_scope": "all official100 rows",
        "calibration_scope": "frozen official100 OOF residuals",
        "feature_names": tuple(feature_names),
        "model_params": {**model_params, "random_state": seed},
        "regressor": regressor,
        "calibration_bias": bias,
        "calibration_absolute_residual_q90": q90,
        "prediction_clip": clip,
        "fit_count": len(target),
        "calibration_count": len(target),
    })


def _stage_a(args: argparse.Namespace, model_root: Path) -> list[dict]:
    dataset_path = args.stage_a_dataset_root / "quality_dataset.jsonl"
    oof_path = args.stage_a_oof_root / "oof_quality_predictions.jsonl"
    oof = {str(row["row_id"]): row for row in _jsonl(oof_path)}
    grouped = {source: {"matrix": [], "target": [], "raw": []} for source in SOURCE_NAMES}
    for row in _jsonl(dataset_path):
        source = str(row["candidate_source"])
        if source not in grouped:
            raise ValueError(f"unexpected stage-A source: {source}")
        prediction = oof.get(str(row["row_id"]))
        if prediction is None:
            raise ValueError(f"missing stage-A OOF prediction: {row['row_id']}")
        names = FEATURE_NAMES_BY_SOURCE[source]
        grouped[source]["matrix"].append([float(row["features"].get(name, 0.0)) for name in names])
        grouped[source]["target"].append(float(row["label_quality_q"]))
        grouped[source]["raw"].append(float(prediction["raw_oof_quality"]))
    records = []
    for source_index, source in enumerate(SOURCE_NAMES):
        values = grouped[source]
        matrix = np.asarray(values["matrix"], dtype=np.float64)
        target = np.asarray(values["target"], dtype=np.float64)
        raw = np.asarray(values["raw"], dtype=np.float64)
        regressor = HistGradientBoostingRegressor(
            random_state=20260862 + source_index, **STAGE_A_MODEL_PARAMS
        )
        regressor.fit(matrix, target)
        calibrator, calibration_kind = _fit_calibrator(raw, target)
        records.append(_write_bundle(model_root / f"stage_a_{source}.joblib", {
            "version": VERSION,
            "model_kind": f"stage_a_{source}_quality",
            "fit_scope": "all official100 rows for this source",
            "calibration_scope": "frozen official100 OOF raw predictions",
            "feature_names": tuple(FEATURE_NAMES_BY_SOURCE[source]),
            "model_params": {**STAGE_A_MODEL_PARAMS, "random_state": 20260862 + source_index},
            "regressor": regressor,
            "calibrator": calibrator,
            "calibration_kind": calibration_kind,
            "prediction_clip": (0.0, 1.0),
            "fit_count": len(target),
            "calibration_count": len(target),
        }))
    if sum(record["fit_count"] for record in records) != len(oof):
        raise ValueError("stage-A dataset and OOF coverage differ")
    return records


def _stage_c_members(args: argparse.Namespace, model_root: Path) -> list[dict]:
    summary = json.loads((args.stage_c_dataset_root / "summary.json").read_text())
    feature_names = tuple(summary["feature_names"])
    oof_path = args.stage_c_oof_root / "oof_member_removal_predictions.jsonl"
    oof = {str(row["atom_id"]): row for row in _jsonl(oof_path)}
    grouped = {role: {"matrix": [], "target": [], "raw": []} for role in EXCLUSIVE_ROLES}
    for row in _jsonl(args.stage_c_dataset_root / summary["files"]["atoms"]):
        role = str(row["role"])
        if role not in grouped:
            continue
        prediction = oof.get(str(row["atom_id"]))
        if prediction is None:
            raise ValueError(f"missing stage-C member OOF prediction: {row['atom_id']}")
        grouped[role]["matrix"].append([float(row["features"][name]) for name in feature_names])
        grouped[role]["target"].append(float(row["label_delta_iou_remove"]))
        grouped[role]["raw"].append(float(prediction["raw_oof_delta_iou_remove"]))
    records = []
    for role_index, role in enumerate(EXCLUSIVE_ROLES):
        values = grouped[role]
        records.append(_fit_residual_bundle(
            matrix=np.asarray(values["matrix"], dtype=np.float64),
            target=np.asarray(values["target"], dtype=np.float64),
            oof_raw=np.asarray(values["raw"], dtype=np.float64),
            feature_names=feature_names,
            model_kind=f"stage_c_member_delta_{role}",
            seed=20260872 + role_index,
            model_params=STAGE_C_MODEL_PARAMS,
            clip=None,
            output_path=model_root / f"stage_c_member_delta_{role}.joblib",
        ))
    return records


def _stage_c_quality(args: argparse.Namespace, model_root: Path) -> list[dict]:
    path = args.stage_c_oof_root / "stage_c_v2_refined_union_plan.jsonl"
    matrix, target, raw = [], [], []
    for row in _jsonl(path):
        matrix.append([float(row["quality_features"][name]) for name in QUALITY_FEATURE_NAMES])
        target.append(float(row["label_temporary_refined_quality_q"]))
        raw.append(float(row["raw_oof_temporary_refined_quality"]))
    return [_fit_residual_bundle(
        matrix=np.asarray(matrix, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        oof_raw=np.asarray(raw, dtype=np.float64),
        feature_names=QUALITY_FEATURE_NAMES,
        model_kind="stage_c_temporary_refined_quality",
        seed=20260882,
        model_params=STAGE_C_MODEL_PARAMS,
        clip=(0.0, 1.0),
        output_path=model_root / "stage_c_temporary_refined_quality.joblib",
    )]


def _stage_d(args: argparse.Namespace, model_root: Path) -> list[dict]:
    summary = json.loads((args.stage_d_dataset_root / "summary.json").read_text())
    dataset_path = args.stage_d_dataset_root / summary["files"]["dataset"]
    oof_path = args.stage_d_oof_root / "oof_rank_marginal_gain_predictions.jsonl"
    oof = {str(row["candidate_key"]): row for row in _jsonl(oof_path)}
    matrix, target, raw = [], [], []
    for row in _jsonl(dataset_path):
        prediction = oof.get(str(row["candidate_key"]))
        if prediction is None:
            raise ValueError(f"missing stage-D OOF prediction: {row['candidate_key']}")
        matrix.append([float(row["features"][name]) for name in STAGE_D_FEATURE_NAMES])
        target.append(float(row["labels"]["rank_conditioned_marginal_iou_gain"]))
        raw.append(float(prediction["raw_oof_rank_marginal_iou_gain"]))
    if len(matrix) != len(oof):
        raise ValueError("stage-D dataset and OOF coverage differ")
    return [_fit_residual_bundle(
        matrix=np.asarray(matrix, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        oof_raw=np.asarray(raw, dtype=np.float64),
        feature_names=STAGE_D_FEATURE_NAMES,
        model_kind="stage_d_v3_rank_marginal_gain",
        seed=20260892,
        model_params=STAGE_D_MODEL_PARAMS,
        clip=(0.0, 1.0),
        output_path=model_root / "stage_d_v3_rank_marginal_gain.joblib",
    )]


def run(args: argparse.Namespace) -> dict:
    for name in (
        "stage_a_dataset_root", "stage_a_oof_root", "stage_c_dataset_root",
        "stage_c_oof_root", "stage_d_dataset_root", "stage_d_oof_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    model_root = staging / "models"
    model_root.mkdir(parents=True)
    try:
        models = []
        models.extend(_stage_a(args, model_root))
        models.extend(_stage_c_members(args, model_root))
        models.extend(_stage_c_quality(args, model_root))
        models.extend(_stage_d(args, model_root))
        if len(models) != 7:
            raise AssertionError("FI1-D-v3 deployment package must contain exactly seven models")
        input_files = {
            "stage_a_dataset": args.stage_a_dataset_root / "quality_dataset.jsonl",
            "stage_a_oof": args.stage_a_oof_root / "oof_quality_predictions.jsonl",
            "stage_c_dataset_summary": args.stage_c_dataset_root / "summary.json",
            "stage_c_oof_summary": args.stage_c_oof_root / "summary.json",
            "stage_d_dataset_summary": args.stage_d_dataset_root / "summary.json",
            "stage_d_oof_summary": args.stage_d_oof_root / "summary.json",
        }
        summary = {
            "version": VERSION,
            "training_dataset": "official100",
            "deployment_target": "single frozen ScanNet200 val312 transfer",
            "model_count": len(models),
            "models": models,
            "stage_a_calibration_contract": "isotonic fitted on frozen OOF raw prediction and label, per source",
            "stage_c_d_calibration_contract": "global additive OOF mean residual and OOF absolute residual q90",
            "ground_truth_usage": "official100 training labels only",
            "validation60_read": False,
            "val312_read": False,
            "ap_computed": False,
            "input_hashes": {name: _sha256(path) for name, path in input_files.items()},
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
    parser.add_argument("--stage-a-dataset-root", type=Path, required=True)
    parser.add_argument("--stage-a-oof-root", type=Path, required=True)
    parser.add_argument("--stage-c-dataset-root", type=Path, required=True)
    parser.add_argument("--stage-c-oof-root", type=Path, required=True)
    parser.add_argument("--stage-d-dataset-root", type=Path, required=True)
    parser.add_argument("--stage-d-oof-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "model_count": result["model_count"],
        "training_dataset": result["training_dataset"],
        "val312_read": result["val312_read"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

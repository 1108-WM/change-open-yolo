#!/usr/bin/env python3
"""Train frozen-split low-capacity semantic reliability models and emit OOF scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


GROUPS = {
    "A_base": list(range(0, 8)),
    "B_plus_yolo": list(range(0, 21)),
    "C_joint_yolo_alpha": list(range(0, 41)),
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metrics(target: np.ndarray, tp50: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict:
    result = {
        "mae": float(mean_absolute_error(target, prediction, sample_weight=weight)),
        "rmse": float(mean_squared_error(target, prediction, sample_weight=weight) ** 0.5),
        "prediction_mean": float(np.average(prediction, weights=weight)),
        "target_mean": float(np.average(target, weights=weight)),
        "spearman": float(spearmanr(target, prediction).statistic),
    }
    result["tp50_roc_auc"] = float(roc_auc_score(tp50, prediction, sample_weight=weight)) if len(np.unique(tp50)) == 2 else None
    return result


def _grouped_metrics(rows: list[dict], indexes: np.ndarray, target, tp50, prediction, weight) -> dict:
    result = {"overall": _metrics(target[indexes], tp50[indexes], prediction[indexes], weight[indexes])}
    for source in ("native", "track", "pair_union"):
        selected = np.asarray([index for index in indexes if rows[index]["candidate_source"] == source], dtype=np.int64)
        result[source] = _metrics(target[selected], tp50[selected], prediction[selected], weight[selected])
    return result


def _class_balanced_fit_weights(rows: list[dict], indexes: np.ndarray, base_weight: np.ndarray) -> np.ndarray:
    """Equalize predicted-class weight within each source without changing source totals."""
    result = np.asarray(base_weight[indexes], dtype=np.float64).copy()
    for source in sorted({str(rows[index]["candidate_source"]) for index in indexes}):
        local_positions = np.asarray([
            position for position, index in enumerate(indexes)
            if str(rows[index]["candidate_source"]) == source
        ], dtype=np.int64)
        classes = sorted({int(rows[indexes[position]]["class_index"]) for position in local_positions})
        source_total = float(result[local_positions].sum())
        per_class_total = source_total / len(classes)
        for class_index in classes:
            class_positions = np.asarray([
                position for position in local_positions
                if int(rows[indexes[position]]["class_index"]) == class_index
            ], dtype=np.int64)
            current_total = float(result[class_positions].sum())
            if current_total <= 0:
                raise ValueError(f"non-positive class weight: {source} {class_index}")
            result[class_positions] *= per_class_total / current_total
    return result


def run(args: argparse.Namespace) -> dict:
    rows = _read_jsonl(args.dataset_root / "rows.jsonl")
    with np.load(args.dataset_root / "dataset.npz") as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        if args.target_field not in payload.files or args.tp50_field not in payload.files:
            raise ValueError("requested target fields are absent from dataset.npz")
        target = np.asarray(payload[args.target_field], dtype=np.float32)
        tp50 = np.asarray(payload[args.tp50_field], dtype=np.int8)
    schema = json.loads((args.dataset_root / "feature_schema.json").read_text())
    feature_names = list(schema["feature_names"])
    if len(rows) != len(features) or features.shape[1] != len(feature_names):
        raise ValueError("dataset rows, feature matrix, and schema disagree")
    scenes = sorted({str(row["scene_name"]) for row in rows})
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)

    weight = np.ones(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        if row["candidate_source"] == "native":
            bound = max(1.0, float(np.expm1(features[index, 5])))
            weight[index] = 1.0 / bound
    original = np.clip(np.asarray([float(row["original_score"]) for row in rows]), 0.0, 1.0)
    predictions = {group: np.full(len(rows), np.nan, dtype=np.float64) for group in GROUPS}
    fit_sources = {item.strip() for item in args.fit_sources.split(",") if item.strip()}
    unknown_fit_sources = fit_sources - {"native", "track", "pair_union"}
    if not fit_sources or unknown_fit_sources:
        raise ValueError(f"invalid --fit-sources: {sorted(fit_sources)}")
    folds = []
    for spec in manifest["folds"]:
        train_scenes, validation_scenes = set(spec["train_scenes"]), set(spec["validation_scenes"])
        if train_scenes & validation_scenes:
            raise ValueError("train/validation scene overlap")
        train = np.asarray([i for i, row in enumerate(rows) if row["scene_name"] in train_scenes], dtype=np.int64)
        fit_train = np.asarray([
            index for index in train if str(rows[index]["candidate_source"]) in fit_sources
        ], dtype=np.int64)
        validation = np.asarray([i for i, row in enumerate(rows) if row["scene_name"] in validation_scenes], dtype=np.int64)
        if not len(train) or not len(fit_train) or not len(validation):
            raise ValueError("empty OOF partition")
        fit_weight = (
            _class_balanced_fit_weights(rows, fit_train, weight)
            if args.class_balance_within_source else weight[fit_train]
        )
        fold_result = {
            "fold_index": int(spec["fold_index"]), "train_row_count": len(train),
            "fit_train_row_count": len(fit_train),
            "validation_row_count": len(validation), "train_scenes": sorted(train_scenes),
            "validation_scenes": sorted(validation_scenes), "metrics": {
                "raw_original_score": _grouped_metrics(rows, validation, target, tp50, original, weight)
            },
        }
        for group, columns in GROUPS.items():
            model = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=120, max_leaf_nodes=7,
                min_samples_leaf=50, l2_regularization=1.0,
                random_state=args.random_seed + int(spec["fold_index"]),
            )
            model.fit(
                features[fit_train][:, columns], target[fit_train], sample_weight=fit_weight
            )
            prediction = np.clip(model.predict(features[validation][:, columns]), 0.0, 1.0)
            predictions[group][validation] = prediction
            fold_result["metrics"][group] = _grouped_metrics(
                rows, validation, target, tp50, predictions[group], weight
            )
        folds.append(fold_result)
        print(f"[Z3 OOF] fold {spec['fold_index']}: train={len(train)} validation={len(validation)}", flush=True)
    if any(np.any(~np.isfinite(values)) for values in predictions.values()):
        raise AssertionError("every row must receive exactly one OOF prediction")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "oof_predictions.jsonl").open("w") as handle:
        for index, row in enumerate(rows):
            output = {
                **row, "sample_weight": float(weight[index]),
                "oof_predictions": {group: float(values[index]) for group, values in predictions.items()},
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "fold_metrics.json").write_text(
        json.dumps({"folds": folds}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    split_sha256 = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    summary = {
        "diagnostic_type": "official100 scene-isolated five-fold OOF semantic reliability",
        "ground_truth_usage": "official_train_supervision_only", "candidate_mutation": False,
        "scene_count": len(scenes), "row_count": len(rows), "fold_count": len(folds),
        "target": args.target_field, "tp50_target": args.tp50_field,
        "model": {
            "type": "HistGradientBoostingRegressor", "learning_rate": 0.05,
            "max_iter": 120, "max_leaf_nodes": 7, "min_samples_leaf": 50,
            "l2_regularization": 1.0, "random_seed": args.random_seed,
        },
        "sample_weight_contract": "native=1/exact_geometry_bound_candidate_count; track=1; pair_union=1",
        "fit_sources": sorted(fit_sources),
        "fit_weight_contract": (
            "within each candidate source, every predicted class present in the training fold has equal total "
            "weight; each source's total base weight is preserved"
            if args.class_balance_within_source else "base sample weights unchanged"
        ),
        "feature_groups": {
            group: {"column_indices": columns, "feature_names": [feature_names[i] for i in columns]}
            for group, columns in GROUPS.items()
        },
        "class_id_is_feature": False,
        "split_manifest": {"path": str(args.split_manifest), "sha256": split_sha256},
        "oof_metrics": {
            "raw_original_score": _grouped_metrics(rows, np.arange(len(rows)), target, tp50, original, weight),
            **{
                group: _grouped_metrics(rows, np.arange(len(rows)), target, tp50, values, weight)
                for group, values in predictions.items()
            },
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260811)
    parser.add_argument("--target-field", default="label_ap_quality")
    parser.add_argument("--tp50-field", default="label_tp50")
    parser.add_argument(
        "--fit-sources", default="native,track,pair_union",
        help="comma-separated candidate sources included in model fitting; all sources still receive OOF predictions",
    )
    parser.add_argument("--class-balance-within-source", action="store_true")
    args = parser.parse_args()
    for name in ("dataset_root", "split_manifest", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    actual = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    if actual != args.expected_split_sha256:
        raise SystemExit("frozen split manifest SHA-256 mismatch")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

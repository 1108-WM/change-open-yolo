#!/usr/bin/env python3
"""Run frozen scene-disjoint candidate-quality OOF experiments.

This diagnostic-only entry point never exports a fit-all model, selects an
action threshold, trains a pair head, modifies candidates, or runs AP.
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (
    GVC_FEATURES,
    NATIVE_SOURCE,
    SOURCES,
    TRACK_FEATURES,
    audit_dataset,
    candidate_ledger_path,
    feature_schema,
    read_jsonl,
    read_scene_list,
)


TARGETS = ("q", "valid25", "valid50")
FROZEN_SCENE_COUNT = 100
FROZEN_FOLD_COUNT = 5
FROZEN_TRAIN_SCENE_COUNT = 80
FROZEN_VALIDATION_SCENE_COUNT = 20


def _as_float(row: dict, field: str) -> float:
    value = row.get(field)
    return float("nan") if value is None else float(value)


def _native_geometry_ids(rows: list[dict], records_root: Path, scene: str) -> None:
    native_rows = [row for row in rows if row["candidate_source"] == NATIVE_SOURCE]
    for row in rows:
        row["_native_geometry_group_id"] = None
    if not native_rows:
        return
    mask_path = records_root / scene / "native_cache" / f"{scene}_pred_masks.npy"
    masks = np.load(mask_path, mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene}: native masks are not point-by-candidate")
    for row in native_rows:
        candidate_id = int(row["candidate_id"])
        if candidate_id < 0 or candidate_id >= masks.shape[1]:
            raise ValueError(f"{scene}: native candidate ID is outside the mask cache")
        packed = np.packbits(np.asarray(masks[:, candidate_id], dtype=np.uint8)).tobytes()
        row["_native_geometry_group_id"] = f"{scene}:{hashlib.sha256(packed).hexdigest()}"


def load_rows(scene_list: Path, records_root: Path) -> list[dict]:
    rows = []
    for scene in read_scene_list(scene_list):
        scene_rows = read_jsonl(candidate_ledger_path(records_root, scene))
        _native_geometry_ids(scene_rows, records_root, scene)
        rows.extend(scene_rows)
    if not rows:
        raise ValueError("no candidate rows")
    return rows


def base_sample_weights(rows: list[dict]) -> np.ndarray:
    """Geometry-folding weights before source balancing."""
    return np.asarray([
        1.0 / float(row["native_exact_geometry_group_size"])
        if row["candidate_source"] == NATIVE_SOURCE else 1.0
        for row in rows
    ], dtype=np.float64)


def source_balanced_weights(
    rows: list[dict], base_weights: np.ndarray, indexes: np.ndarray | None = None,
) -> np.ndarray:
    """Balance native/track mass using only the supplied population.

    For model fitting ``indexes`` is the current outer-fold training set.  For
    evaluation it is omitted, producing one frozen official100 evaluation
    weighting that never depends on a fold's training data.
    """
    if indexes is None:
        indexes = np.arange(len(rows), dtype=np.int64)
    indexes = np.asarray(indexes, dtype=np.int64)
    if not len(indexes):
        raise ValueError("cannot normalize sample weights for an empty population")
    weights = np.zeros(len(rows), dtype=np.float64)
    weights[indexes] = base_weights[indexes]
    # Equal source mass prevents native class expansion from becoming a prior.
    for source in SOURCES:
        source_indexes = indexes[np.asarray([rows[index]["candidate_source"] == source for index in indexes])]
        total = weights[source_indexes].sum()
        if total <= 0:
            raise ValueError(f"no positive sample weight for {source}")
        weights[source_indexes] *= 0.5 * len(indexes) / total
    return weights


def sample_weights(rows: list[dict]) -> np.ndarray:
    """Frozen full-protocol evaluation weights (backward-compatible helper)."""
    return source_balanced_weights(rows, base_sample_weights(rows))


def target_values(rows: list[dict], target: str) -> np.ndarray:
    if target == "q":
        return np.asarray([float(row["label_best_gt_iou"]) for row in rows], dtype=np.float64)
    field = "label_valid_iou25" if target == "valid25" else "label_valid_iou50"
    return np.asarray([int(bool(row[field])) for row in rows], dtype=np.int64)


def build_seeded_scene_folds(scene_names: list[str], fold_count: int, seed: int) -> list[dict]:
    """Legacy helper retained only for manifest construction, never OOF."""
    unique_scenes = sorted(set(scene_names))
    if len(unique_scenes) < fold_count or len(unique_scenes) % fold_count:
        raise ValueError("scene count must be divisible by fold_count")
    shuffled = np.random.default_rng(seed).permutation(unique_scenes).tolist()
    validation_size = len(unique_scenes) // fold_count
    return [
        {
            "fold_index": fold_index,
            "train_scenes": sorted(scene for scene in unique_scenes if scene not in set(shuffled[fold_index * validation_size:(fold_index + 1) * validation_size])),
            "validation_scenes": sorted(shuffled[fold_index * validation_size:(fold_index + 1) * validation_size]),
        }
        for fold_index in range(fold_count)
    ]


def load_frozen_split_manifest(path: Path, scene_names: list[str]) -> dict:
    manifest = json.loads(path.read_text())
    expected = set(scene_names)
    if len(expected) != FROZEN_SCENE_COUNT or manifest.get("scene_count") != FROZEN_SCENE_COUNT:
        raise ValueError("official100 OOF requires exactly 100 scenes")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != FROZEN_FOLD_COUNT:
        raise ValueError("official100 OOF requires exactly five frozen folds")
    normalized = []
    validation_seen: list[str] = []
    for expected_index, fold in enumerate(sorted(folds, key=lambda value: value.get("fold_index", -1))):
        if fold.get("fold_index") != expected_index:
            raise ValueError("frozen fold indices must be exactly 0..4")
        train, validation = list(fold.get("train_scenes", [])), list(fold.get("validation_scenes", []))
        if len(train) != FROZEN_TRAIN_SCENE_COUNT or len(validation) != FROZEN_VALIDATION_SCENE_COUNT:
            raise ValueError("every frozen official100 fold must be 80 train / 20 validation scenes")
        if len(set(train)) != len(train) or len(set(validation)) != len(validation):
            raise ValueError(f"fold {expected_index}: duplicate scene names")
        train_set, validation_set = set(train), set(validation)
        if train_set & validation_set or train_set | validation_set != expected or train_set != expected - validation_set:
            raise ValueError(f"fold {expected_index}: train/validation scenes do not exactly partition official100")
        validation_seen.extend(validation)
        normalized.append({
            "fold_index": expected_index,
            "train_scenes": sorted(train_set),
            "validation_scenes": sorted(validation_set),
        })
    if set(validation_seen) != expected or len(validation_seen) != FROZEN_SCENE_COUNT:
        raise ValueError("every official100 scene must occur in exactly one frozen validation fold")
    return {**manifest, "folds": normalized}


def feature_matrix(rows: list[dict], group: str, protocol_name: str) -> tuple[np.ndarray, list[str]]:
    groups = feature_schema(protocol_name)["groups"]
    if group not in groups:
        raise ValueError(f"unknown feature group {group}")
    names, columns = ["candidate_source=d2b_track"], [np.asarray([float(row["candidate_source"] != NATIVE_SOURCE) for row in rows])]
    for field in groups[group]["numeric"]:
        values = np.asarray([_as_float(row, field) for row in rows], dtype=np.float64)
        columns.append(values)
        names.append(field)
        if field in TRACK_FEATURES:
            columns.append(np.isnan(values).astype(np.float64))
            names.append(f"{field}__missing")
    return np.column_stack(columns), names


class SourcePrior:
    def fit(self, matrix, target, weight):
        self.default = float(np.average(target, weights=weight))
        self.values = {}
        for source in np.unique(matrix[:, 0].astype(int)):
            mask = matrix[:, 0].astype(int) == source
            self.values[source] = float(np.average(target[mask], weights=weight[mask]))
        return self

    def predict(self, matrix):
        return np.asarray([self.values.get(int(value), self.default) for value in matrix[:, 0]], dtype=np.float64)


class ConstantPrediction:
    def __init__(self, value: float): self.value = value
    def predict(self, matrix): return np.full(len(matrix), self.value, dtype=np.float64)


def make_model(target: str, seed: int, train_y: np.ndarray, train_weight: np.ndarray, source_only: bool):
    if source_only:
        return SourcePrior()
    if target != "q" and len(np.unique(train_y)) < 2:
        return ConstantPrediction(float(np.average(train_y, weights=train_weight)))
    if target != "q":
        return HistGradientBoostingClassifier(learning_rate=.05, max_iter=160, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1., random_state=seed)
    return HistGradientBoostingRegressor(learning_rate=.05, max_iter=160, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1., random_state=seed)


def fit_predict(model, x_train, y_train, weight_train, x_validation):
    if isinstance(model, (SourcePrior, ConstantPrediction)):
        if isinstance(model, SourcePrior): model.fit(x_train, y_train, weight_train)
        return model.predict(x_validation)
    model.fit(x_train, y_train, sample_weight=weight_train)
    return model.predict_proba(x_validation)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_validation)


def canonicalize_predictions(target: str, prediction: np.ndarray) -> np.ndarray:
    """Use one [0, 1] prediction contract for metrics and OOF consumers."""
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{target}: model produced non-finite predictions")
    return np.clip(values, 0.0, 1.0)


def _weighted_mean(values, weights) -> float:
    return float(np.average(values, weights=weights)) if len(values) else float("nan")


def _spearman(labels, predictions) -> float | None:
    if len(labels) < 2 or len(np.unique(labels)) < 2 or len(np.unique(predictions)) < 2:
        return None
    result = spearmanr(labels, predictions)
    value = float(getattr(result, "statistic", getattr(result, "correlation", float("nan"))))
    return value if math.isfinite(value) else None


def q_spearman_units(rows, labels, predictions) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    native_groups: dict[str, list[int]] = defaultdict(list)
    track_indices = []
    for index, row in enumerate(rows):
        if row["candidate_source"] == NATIVE_SOURCE:
            group_id = row.get("_native_geometry_group_id") or f"{row['scene_name']}:native:{row['candidate_id']}"
            native_groups[group_id].append(index)
        else:
            track_indices.append(index)
    group_labels = np.asarray([np.median(labels[indexes]) for indexes in native_groups.values()])
    group_predictions = np.asarray([np.median(predictions[indexes]) for indexes in native_groups.values()])
    return {
        "class_expanded_unweighted_spearman": _spearman(labels, predictions),
        "native_geometry_group_unweighted_spearman": _spearman(group_labels, group_predictions),
        "track_raw_unweighted_spearman": _spearman(labels[track_indices], predictions[track_indices]),
        "native_geometry_group_count": len(native_groups),
        "track_raw_count": len(track_indices),
    }


def prediction_distribution(predictions) -> dict:
    values = np.sort(np.asarray(predictions, dtype=np.float64))
    if not len(values): return {"count": 0}
    return {"count": len(values), "min": float(values[0]), "p05": float(np.quantile(values, .05)), "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)), "max": float(values[-1]), "mean": float(values.mean())}


def _ece(labels, scores, weights, bins: int = 10) -> float:
    error, total = 0., float(weights.sum())
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & ((scores < high) if index < bins - 1 else (scores <= high))
        if mask.any(): error += float(weights[mask].sum()) / total * abs(_weighted_mean(labels[mask], weights[mask]) - _weighted_mean(scores[mask], weights[mask]))
    return float(error)


def metrics(target, labels, predictions, weights) -> dict:
    predictions = canonicalize_predictions(target, predictions)
    if target == "q":
        return {"count": len(labels), "mae": float(mean_absolute_error(labels, predictions, sample_weight=weights)), "rmse": float(mean_squared_error(labels, predictions, sample_weight=weights) ** .5), "class_expanded_unweighted_spearman": _spearman(labels, predictions)}
    result = {"count": len(labels), "positive_count": int(labels.sum()), "brier": _weighted_mean((predictions - labels) ** 2, weights), "ece10": _ece(labels, predictions, weights)}
    result.update({"roc_auc": None, "pr_auc": None} if len(np.unique(labels)) < 2 else {"roc_auc": float(roc_auc_score(labels, predictions, sample_weight=weights)), "pr_auc": float(average_precision_score(labels, predictions, sample_weight=weights))})
    return result


def grouped_metrics(rows, target, labels, predictions, weights) -> dict:
    result = {"overall": metrics(target, labels, predictions, weights)}
    if target == "q": result["overall"].update(q_spearman_units(rows, labels, predictions))
    sources = np.asarray([row["candidate_source"] for row in rows])
    for source in SOURCES:
        mask = sources == source
        result[source] = metrics(target, labels[mask], predictions[mask], weights[mask])
    return result


def metric_deltas(current, baseline, target) -> dict:
    names = ("mae", "rmse", "class_expanded_unweighted_spearman") if target == "q" else ("roc_auc", "pr_auc", "brier", "ece10")
    return {**{name: None if current["overall"].get(name) is None or baseline["overall"].get(name) is None else float(current["overall"][name] - baseline["overall"][name]) for name in names}, "interpretation": "positive is better only for Spearman/ROC-AUC/PR-AUC; negative is better for MAE/RMSE/Brier/ECE"}


def run_oof(rows, frozen_manifest: dict, protocol_name: str, seed: int = 20260808):
    scenes = np.asarray([row["scene_name"] for row in rows])
    fold_specs = frozen_manifest["folds"]
    base_weights = base_sample_weights(rows)
    # Evaluation weights are frozen for the whole protocol.  Training weights
    # are recomputed inside every fold from its 80 training scenes only.
    evaluation_weights = source_balanced_weights(rows, base_weights)
    labels = {target: target_values(rows, target) for target in TARGETS}
    groups = list(feature_schema(protocol_name)["groups"])
    matrices = {group: feature_matrix(rows, group, protocol_name)[0] for group in groups}
    predictions = {group: {target: np.full(len(rows), np.nan) for target in TARGETS} for group in groups}
    raw = {target: np.clip(np.asarray([float(row["original_source_score"]) for row in rows]), 0., 1.) for target in TARGETS}
    folds = []
    validation_assignments = np.zeros(len(rows), dtype=np.int64)
    for spec in fold_specs:
        fold_index = int(spec["fold_index"])
        validation = np.flatnonzero(np.isin(scenes, spec["validation_scenes"]))
        train = np.flatnonzero(np.isin(scenes, spec["train_scenes"]))
        if len(set(scenes[validation])) != FROZEN_VALIDATION_SCENE_COUNT or len(set(scenes[train])) != FROZEN_TRAIN_SCENE_COUNT:
            raise AssertionError("row split does not match the frozen 80/20 scene contract")
        validation_assignments[validation] += 1
        fold_rows = [rows[index] for index in validation]
        fold = {"fold_index": fold_index, "train_scenes": spec["train_scenes"], "validation_scenes": spec["validation_scenes"], "metrics": {"raw_original_score": {}}}
        for target in TARGETS:
            fold["metrics"]["raw_original_score"][target] = {"metrics": grouped_metrics(fold_rows, target, labels[target][validation], raw[target][validation], evaluation_weights[validation]), "prediction_distribution": prediction_distribution(raw[target][validation])}
        for group in groups:
            fold["metrics"][group] = {}
            for target in TARGETS:
                train_weights = source_balanced_weights(rows, base_weights, train)
                model = make_model(target, seed + fold_index, labels[target][train], train_weights[train], group == "A_source_only")
                prediction = canonicalize_predictions(target, fit_predict(model, matrices[group][train], labels[target][train], train_weights[train], matrices[group][validation]))
                predictions[group][target][validation] = prediction
                fold["metrics"][group][target] = {"metrics": grouped_metrics(fold_rows, target, labels[target][validation], prediction, evaluation_weights[validation]), "prediction_distribution": prediction_distribution(prediction)}
        folds.append(fold)
    if not np.all(validation_assignments == 1) or any(np.isnan(value).any() for group in predictions.values() for value in group.values()):
        raise AssertionError("every candidate must receive exactly one frozen-manifest OOF prediction")
    oof_rows = [{"scene_name": row["scene_name"], "candidate_source": row["candidate_source"], "candidate_id": row["candidate_id"], "sample_weight": float(evaluation_weights[index]), "label_best_gt_iou": float(labels["q"][index]), "label_valid_iou25": int(labels["valid25"][index]), "label_valid_iou50": int(labels["valid50"][index]), "predictions": {group: {target: float(predictions[group][target][index]) for target in TARGETS} for group in groups}} for index, row in enumerate(rows)]
    overall = {"raw_original_score": {target: grouped_metrics(rows, target, labels[target], raw[target], evaluation_weights) for target in TARGETS}, **{group: {target: grouped_metrics(rows, target, labels[target], predictions[group][target], evaluation_weights) for target in TARGETS} for group in groups}}
    summary = {"version": f"{protocol_name}_candidate_quality_oof_v1", "protocol_name": protocol_name, "fold_count": FROZEN_FOLD_COUNT, "random_seed": seed, "split_strategy": "externally_frozen_manifest", "scene_count": FROZEN_SCENE_COUNT, "candidate_count": len(rows), "sample_weight_contract": {"native_base": "1 / native_exact_geometry_group_size", "track_base": 1.0, "training_source_balance": "per_outer_fold_train_scenes_only", "evaluation_source_balance": "frozen_full_protocol"}, "prediction_contract": {"q": "clipped_to_[0,1]_before_metrics_and_oof_export", "valid25": "clipped_to_[0,1]_before_metrics_and_oof_export", "valid50": "clipped_to_[0,1]_before_metrics_and_oof_export"}, "fit_all_model_exported": False, "pair_actions_enabled": False, "candidate_geometry_modified": False, "ap_evaluation_run": False, "oof_metrics": overall, "overall_metric_deltas": {group: {target: {"vs_source_only": metric_deltas(overall[group][target], overall["A_source_only"][target], target), "vs_raw_original_score": metric_deltas(overall[group][target], overall["raw_original_score"][target], target)} for target in TARGETS} for group in ("B_source_plus_original_score", "C_plus_geometry_track_structure", "D_plus_gvc")}}
    return summary, oof_rows, {"folds": folds, "feature_schema": feature_schema(protocol_name), "frozen_split_manifest": frozen_manifest}


def write_oof(output_root: Path, summary: dict, oof_rows: list[dict], detail: dict) -> None:
    if output_root.exists() and any(output_root.iterdir()): raise ValueError(f"output root is non-empty: {output_root}")
    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "split_manifest.json").write_text(json.dumps(detail["frozen_split_manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "feature_schema.json").write_text(json.dumps(detail["feature_schema"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        for fold in detail["folds"]:
            fold_dir = staging / f"fold_{fold['fold_index']}"; fold_dir.mkdir()
            (fold_dir / "metrics.json").write_text(json.dumps(fold["metrics"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "oof_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in oof_rows))
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--random-seed", type=int, default=20260808)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    scenes = read_scene_list(args.scene_list)
    actual_split_sha256 = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    if actual_split_sha256 != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 does not match --expected-split-sha256")
    frozen_manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    audit_dataset(args.scene_list, args.records_root, [PROJECT_ROOT / "output/scannet200/scene_splits/gvc_holdout_20260803/gvc_safety60.txt", PROJECT_ROOT / "output/scannet200/scene_splits/even48.txt", PROJECT_ROOT / "output/scannet200/scene_splits/odd96.txt", PROJECT_ROOT / "output/scannet200/scene_splits/gvc_holdout_20260803/gvc_test60.txt"], protocol_name=args.protocol_name, prepared_root=args.prepared_root)
    summary, oof_rows, detail = run_oof(load_rows(args.scene_list, args.records_root), frozen_manifest, args.protocol_name, args.random_seed)
    detail["frozen_split_manifest"] = {**frozen_manifest, "source_path": str(args.split_manifest.resolve()), "sha256": actual_split_sha256}
    write_oof(args.output_root, summary, oof_rows, detail)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

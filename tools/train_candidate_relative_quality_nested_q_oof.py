#!/usr/bin/env python3
"""Audit relative quality with candidate-q predictions nested by outer fold.

For each frozen relation outer fold, candidate quality is rebuilt as follows:
the 80 outer-training scenes receive four-fold internal predictions, while the
20 outer-validation scenes are predicted by a candidate-quality model fitted
only on all 80 outer-training scenes.  These predictions are then aggregated
to track--native exact-geometry relation features and consumed by the nested
relative-quality model.

No precomputed candidate OOF score is read.  No fit-all model, threshold,
cascade action, candidate mutation, replacement, or AP result is produced.
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
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    read_scene_list,
)
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    base_sample_weights,
    canonicalize_predictions,
    feature_matrix,
    fit_predict,
    load_rows as load_candidate_rows,
    make_model,
    source_balanced_weights,
    target_values,
)
from tools.train_candidate_relative_quality_oof import (  # noqa: E402
    MODEL_FEATURES as RAW_RELATION_FEATURES,
    _labels,
    _quality_rows,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    OUTER_FOLD_COUNT,
    RANDOM_SEED,
    _prediction_bootstrap,
    grouped_metrics,
    nested_fit_predict,
    scene_track_balanced_weights,
    seeded_scene_folds,
)


NESTED_Q_FEATURES = (
    "nested_track_q",
    "nested_native_q_median",
    "nested_q_delta_track_minus_native",
)
MODEL_FEATURES = {
    "Q_nested_candidate_quality_only": NESTED_Q_FEATURES,
    "E_directional_geometry_plus_nested_q": (
        *RAW_RELATION_FEATURES["B_plus_directional_geometry"],
        *NESTED_Q_FEATURES,
    ),
    "F_full_relation_plus_nested_q": (
        *RAW_RELATION_FEATURES["D_plus_public_component"],
        *NESTED_Q_FEATURES,
    ),
}
BOOTSTRAP_SEED = 20260830
NESTED_CANDIDATE_TARGETS = ("q", "valid25", "valid50")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_key(row: dict) -> tuple[str, str, int]:
    return str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])


def nested_candidate_quality_predictions(
    candidate_rows: list[dict],
    outer_fold_index: int,
    scene_to_fold: dict[str, int],
    protocol_name: str,
    targets: tuple[str, ...] = NESTED_CANDIDATE_TARGETS,
) -> tuple[dict[tuple[str, str, int], dict[str, float]], dict]:
    unknown = set(targets) - set(NESTED_CANDIDATE_TARGETS)
    if not targets or unknown:
        raise ValueError(f"unsupported nested candidate-quality targets: {sorted(unknown)}")
    scenes = np.asarray([row["scene_name"] for row in candidate_rows])
    matrix, feature_names = feature_matrix(candidate_rows, "D_plus_gvc", protocol_name)
    base_weights = base_sample_weights(candidate_rows)
    outer_train_scenes = sorted(scene for scene, fold in scene_to_fold.items() if fold != outer_fold_index)
    outer_validation_scenes = sorted(scene for scene, fold in scene_to_fold.items() if fold == outer_fold_index)
    if len(outer_train_scenes) != 80 or len(outer_validation_scenes) != 20:
        raise AssertionError("nested candidate-q outer split is not 80/20")
    outer_train = np.flatnonzero(np.isin(scenes, outer_train_scenes))
    outer_validation = np.flatnonzero(np.isin(scenes, outer_validation_scenes))
    target_predictions: dict[str, np.ndarray] = {}
    target_diagnostics = {}
    inner_folds = seeded_scene_folds(
        outer_train_scenes, fold_count=4, seed=RANDOM_SEED + outer_fold_index
    )
    for target_index, target in enumerate(targets):
        labels = target_values(candidate_rows, target)
        predictions = np.full(len(candidate_rows), np.nan, dtype=np.float64)
        assignments = np.zeros(len(candidate_rows), dtype=np.int64)
        inner_summaries = []
        for inner in inner_folds:
            train = np.flatnonzero(np.isin(scenes, inner["train_scenes"]))
            validation = np.flatnonzero(np.isin(scenes, inner["validation_scenes"]))
            weights = source_balanced_weights(candidate_rows, base_weights, train)
            seed = RANDOM_SEED + target_index * 1000 + outer_fold_index * 10 + int(inner["fold_index"])
            model = make_model(target, seed, labels[train], weights[train], source_only=False)
            values = canonicalize_predictions(
                target,
                fit_predict(model, matrix[train], labels[train], weights[train], matrix[validation]),
            )
            predictions[validation] = values
            assignments[validation] += 1
            inner_summaries.append({
                "inner_fold_index": int(inner["fold_index"]),
                "train_scene_count": len(inner["train_scenes"]),
                "validation_scene_count": len(inner["validation_scenes"]),
                "train_candidate_count": len(train),
                "validation_candidate_count": len(validation),
            })
        weights = source_balanced_weights(candidate_rows, base_weights, outer_train)
        model = make_model(
            target, RANDOM_SEED + target_index * 1000 + outer_fold_index,
            labels[outer_train], weights[outer_train], source_only=False,
        )
        values = canonicalize_predictions(
            target,
            fit_predict(
                model, matrix[outer_train], labels[outer_train], weights[outer_train],
                matrix[outer_validation],
            ),
        )
        predictions[outer_validation] = values
        assignments[outer_validation] += 1
        if not np.all(assignments == 1) or not np.isfinite(predictions).all():
            raise AssertionError(f"nested candidate {target} must predict every candidate exactly once")
        validation_labels = labels[outer_validation]
        validation_predictions = predictions[outer_validation]
        metrics = {
            "outer_validation_count": len(outer_validation),
        }
        if target == "q":
            correlation = spearmanr(validation_labels, validation_predictions)
            correlation_value = float(getattr(
                correlation, "statistic", getattr(correlation, "correlation", float("nan"))
            ))
            metrics.update({
                "mae": float(mean_absolute_error(validation_labels, validation_predictions)),
                "rmse": float(mean_squared_error(validation_labels, validation_predictions) ** 0.5),
                "spearman": correlation_value if np.isfinite(correlation_value) else None,
            })
        else:
            metrics.update({
                "positive_count": int(validation_labels.sum()),
                "brier": float(brier_score_loss(validation_labels, validation_predictions)),
                "roc_auc": float(roc_auc_score(validation_labels, validation_predictions))
                if len(np.unique(validation_labels)) == 2 else None,
                "pr_auc": float(average_precision_score(validation_labels, validation_predictions))
                if np.any(validation_labels) else None,
            })
        target_predictions[target] = predictions
        target_diagnostics[target] = {"inner_folds": inner_summaries, "metrics": metrics}
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for row_index, row in enumerate(candidate_rows):
        key = _candidate_key(row)
        if key in lookup:
            raise ValueError(f"duplicate candidate identity in nested quality: {key}")
        lookup[key] = {
            target: float(target_predictions[target][row_index]) for target in targets
        }
    diagnostics = {
        "outer_fold_index": outer_fold_index,
        "candidate_quality_feature_group": "D_plus_gvc",
        "candidate_quality_feature_names": feature_names,
        "outer_train_scene_count": len(outer_train_scenes),
        "outer_validation_scene_count": len(outer_validation_scenes),
        "outer_train_candidate_count": len(outer_train),
        "outer_validation_candidate_count": len(outer_validation),
        "targets": target_diagnostics,
    }
    return lookup, diagnostics


def nested_candidate_q_predictions(
    candidate_rows: list[dict],
    outer_fold_index: int,
    scene_to_fold: dict[str, int],
    protocol_name: str,
) -> tuple[dict[tuple[str, str, int], float], dict]:
    lookup, diagnostics = nested_candidate_quality_predictions(
        candidate_rows, outer_fold_index, scene_to_fold, protocol_name, targets=("q",)
    )
    q_lookup = {key: values["q"] for key, values in lookup.items()}
    q_metrics = diagnostics["targets"]["q"]["metrics"]
    diagnostics = {
        **diagnostics,
        "inner_folds": diagnostics["targets"]["q"]["inner_folds"],
        "outer_validation_q_mae": q_metrics["mae"],
        "outer_validation_q_rmse": q_metrics["rmse"],
        "outer_validation_q_spearman": q_metrics["spearman"],
    }
    return q_lookup, diagnostics


def relation_nested_quality_evidence(
    rows: list[dict],
    quality_lookup: dict[tuple[str, str, int], dict[str, float]],
    targets: tuple[str, ...] = NESTED_CANDIDATE_TARGETS,
) -> list[dict]:
    evidence = []
    for row in rows:
        scene = str(row["scene_name"])
        track_key = (scene, TRACK_SOURCE, int(row["track_id"]))
        native_keys = [
            (scene, NATIVE_SOURCE, int(candidate_id))
            for candidate_id in row["native_member_candidate_ids"]
        ]
        if track_key not in quality_lookup or any(key not in quality_lookup for key in native_keys):
            raise ValueError(f"missing nested candidate quality for relation {scene}/{row['track_id']}")
        relation = {}
        for target in targets:
            if target not in quality_lookup[track_key] or any(
                target not in quality_lookup[key] for key in native_keys
            ):
                raise ValueError(f"missing nested candidate {target} for relation {scene}/{row['track_id']}")
            track_value = float(quality_lookup[track_key][target])
            native_values = [float(quality_lookup[key][target]) for key in native_keys]
            native_median = float(np.median(native_values))
            relation.update({
                f"nested_track_{target}": track_value,
                f"nested_native_{target}_median": native_median,
                f"nested_native_{target}_min": float(min(native_values)),
                f"nested_native_{target}_max": float(max(native_values)),
                f"nested_native_{target}_range": float(max(native_values) - min(native_values)),
                f"nested_{target}_pair_min": min(track_value, native_median),
                f"nested_{target}_pair_product": track_value * native_median,
                f"nested_{target}_delta_track_minus_native": track_value - native_median,
            })
        evidence.append(relation)
    return evidence


def relation_matrix_with_nested_q(
    rows: list[dict],
    raw_feature_names: tuple[str, ...],
    q_lookup: dict[tuple[str, str, int], float],
) -> tuple[np.ndarray, list[dict]]:
    structured_lookup = {key: {"q": value} for key, value in q_lookup.items()}
    full_evidence = relation_nested_quality_evidence(rows, structured_lookup, targets=("q",))
    matrix, evidence = [], []
    for row, relation in zip(rows, full_evidence):
        nested = [
            relation["nested_track_q"],
            relation["nested_native_q_median"],
            relation["nested_q_delta_track_minus_native"],
        ]
        raw = [float(row["features"][name]) for name in raw_feature_names]
        matrix.append(raw + nested)
        evidence.append({
            key: relation[key] for key in (
                "nested_track_q", "nested_native_q_median",
                "nested_q_delta_track_minus_native", "nested_native_q_min",
                "nested_native_q_max", "nested_native_q_range",
            )
        })
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(rows), len(raw_feature_names) + len(NESTED_Q_FEATURES)):
        raise AssertionError("nested-q relation matrix shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("nested-q relation matrix contains non-finite values")
    return values, evidence


def run_oof(
    all_relation_rows: list[dict],
    candidate_rows: list[dict],
    scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]],
    candidate_protocol_name: str,
    bootstrap_repetitions: int,
) -> tuple[dict, list[dict], list[dict], dict]:
    rows = _quality_rows(all_relation_rows)
    labels = _labels(rows)
    predictions = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in MODEL_FEATURES}
    nested_q_evidence: list[dict | None] = [None] * len(rows)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics = []
    diagnostics = {
        "candidate_quality_outer_folds": [],
        "relative_quality_models": {name: [] for name in MODEL_FEATURES},
    }
    for fold_index in range(OUTER_FOLD_COUNT):
        q_lookup, q_diagnostics = nested_candidate_q_predictions(
            candidate_rows, fold_index, scene_to_fold, candidate_protocol_name
        )
        diagnostics["candidate_quality_outer_folds"].append(q_diagnostics)
        train = np.asarray([
            index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] != fold_index
        ], dtype=np.int64)
        validation = np.asarray([
            index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        assignments[validation] += 1
        validation_weights = scene_track_balanced_weights(rows, validation)[validation]
        fold = {
            "fold_index": fold_index,
            "train_relation_count": len(train),
            "validation_relation_count": len(validation),
            "validation_prefer_track_count": int(labels[validation].sum()),
            "nested_candidate_quality": q_diagnostics,
            "models": {},
        }
        for model_name, feature_names in MODEL_FEATURES.items():
            raw_names = tuple(
                name for name in feature_names if name not in NESTED_Q_FEATURES
            )
            matrix, evidence = relation_matrix_with_nested_q(rows, raw_names, q_lookup)
            values, model_diagnostics = nested_fit_predict(
                rows, matrix, labels, train, validation, fold_index
            )
            predictions[model_name][validation] = values
            model_diagnostics["feature_names"] = list(feature_names)
            diagnostics["relative_quality_models"][model_name].append(model_diagnostics)
            fold["models"][model_name] = grouped_metrics(
                [rows[index] for index in validation], labels[validation],
                values, validation_weights,
            )
            if model_name == "Q_nested_candidate_quality_only":
                for index in validation:
                    nested_q_evidence[index] = evidence[index]
        fold_metrics.append(fold)
    if not np.all(assignments == 1) or any(not np.isfinite(values).all() for values in predictions.values()):
        raise AssertionError("every relation must receive one nested-q OOF prediction")
    if any(value is None for value in nested_q_evidence):
        raise AssertionError("every relation must retain its nested-q evidence")
    evaluation_weights = scene_track_balanced_weights(rows)
    overall = {
        name: grouped_metrics(rows, labels, values, evaluation_weights)
        for name, values in predictions.items()
    }
    cohort_metrics = {}
    for cohort_name, cohort_scenes in cohorts.items():
        indexes = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in cohort_scenes
        ], dtype=np.int64)
        weights = scene_track_balanced_weights(rows, indexes)[indexes]
        cohort_metrics[cohort_name] = {
            "scene_count": len(cohort_scenes),
            "relation_count": len(indexes),
            "prefer_track_count": int(labels[indexes].sum()),
            "models": {
                name: grouped_metrics(
                    [rows[index] for index in indexes], labels[indexes], values[indexes], weights
                )
                for name, values in predictions.items()
            },
        }
    bootstrap = {
        model_name: {
            kind: _prediction_bootstrap(
                rows, labels, values, kind, bootstrap_repetitions,
                BOOTSTRAP_SEED + model_index * 10 + kind_index,
            )
            for kind_index, kind in enumerate(("scene", "track"))
        }
        for model_index, (model_name, values) in enumerate(predictions.items())
    }
    oof_rows = []
    for index, row in enumerate(rows):
        oof_rows.append({
            "scene_name": row["scene_name"],
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "label_relative_quality_state": row["labels"]["relative_quality_state"],
            "label_prefer_track": int(labels[index]),
            "nested_candidate_quality_features": nested_q_evidence[index],
            "predictions": {name: float(values[index]) for name, values in predictions.items()},
            "ground_truth_usage": "official_train_offline_label_only",
        })
    summary = {
        "scene_count": 100,
        "relation_count": len(rows),
        "prefer_track_count": int(labels.sum()),
        "prefer_native_count": int((labels == 0).sum()),
        "overall_metrics": overall,
        "cohort_metrics": cohort_metrics,
        "model_cluster_bootstrap": bootstrap,
        "model_selection_applied": False,
    }
    return summary, oof_rows, fold_metrics, diagnostics


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
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    actual_split_sha = _sha256(args.split_manifest)
    if actual_split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    relation_rows = _load_rows(args.feature_ledger_root, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    summary_core, oof_rows, fold_metrics, diagnostics = run_oof(
        relation_rows, candidate_rows, scene_to_fold, cohorts,
        args.candidate_quality_protocol_name, args.bootstrap_repetitions,
    )
    summary = {
        "version": f"{args.protocol_name}_candidate_relative_quality_nested_q_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "feature_groups": {name: list(features) for name, features in MODEL_FEATURES.items()},
        "training_contract": {
            "task": "conditional same-target prefer_track versus prefer_native",
            "candidate_quality_target": "best GT IoU q",
            "candidate_quality_model": "frozen D_plus_gvc HistGradientBoosting q regressor",
            "outer_training_candidate_scores": "four-fold scene-disjoint internal predictions inside each relation outer fold",
            "outer_validation_candidate_scores": "candidate-q model fitted only on the 80 relation outer-training scenes",
            "precomputed_candidate_oof_scores_read": False,
            "native_exact_geometry_aggregation": "median of nested candidate-q member predictions",
            "relative_quality_model": "standardized L2 logistic regression with nested Platt calibration",
        },
        "safety_contract": {
            "fit_all_candidate_quality_model_exported": False,
            "fit_all_relative_quality_model_exported": False,
            "target_gate_threshold_selected": False,
            "relative_quality_threshold_selected": False,
            "cascade_evaluated": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "ap_evaluation_run": False,
        },
        "input_provenance": {
            "feature_ledger_summary_path": str((args.feature_ledger_root / "summary.json").resolve()),
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": _sha256(args.scene_list),
            "existing20_scene_list_path": str(args.existing20_scene_list.resolve()),
            "existing20_scene_list_sha256": _sha256(args.existing20_scene_list),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": actual_split_sha,
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
        (staging / "model_diagnostics.json").write_text(
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

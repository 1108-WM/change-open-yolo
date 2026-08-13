#!/usr/bin/env python3
"""Evaluate a fixed binary-plus-official-gap multitask relation head.

The main branch preserves the incumbent class-balanced strict preference loss.
The auxiliary branch uses the official AP50:95 relevance-gap weighted logistic
loss.  Each branch contributes exactly one half of the normalized fit mass;
the mixture weight is fixed and never scanned.  Inner scene folds calibrate the
combined decision score back to strict ``prefer_track`` probability.

This is official100 OOF diagnosis only.  It does not export a fit-all model,
select an action threshold, mutate candidates, or run AP.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_official_relevance_gap_rank_oof import (  # noqa: E402
    FEATURES,
    TIE_EPSILON,
    feature_matrix,
    rank_metrics,
    relevance_targets,
    same_target_rows,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    INNER_FOLD_COUNT,
    OUTER_FOLD_COUNT,
    RANDOM_SEED,
    _fit_platt,
    class_balanced_weights,
    grouped_metrics,
    scene_track_balanced_weights,
    seeded_scene_folds,
)


MAIN_LOSS_FRACTION = 0.50
AUXILIARY_LOSS_FRACTION = 0.50
INCUMBENT_MODEL_NAME = "B_plus_directional_geometry"
STRICT_STATES = ("prefer_track", "prefer_native")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_indexes(rows: list[dict], indexes: np.ndarray | None = None) -> np.ndarray:
    if indexes is None:
        indexes = np.arange(len(rows), dtype=np.int64)
    indexes = np.asarray(indexes, dtype=np.int64)
    selected = np.asarray([
        index for index in indexes
        if rows[index]["labels"]["relative_quality_state"] in STRICT_STATES
    ], dtype=np.int64)
    if not len(selected):
        raise ValueError("strict preference branch has no relations")
    return selected


def strict_labels(rows: list[dict], indexes: np.ndarray) -> np.ndarray:
    labels = np.asarray([
        int(rows[index]["labels"]["relative_quality_state"] == "prefer_track")
        for index in indexes
    ], dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("strict preference branch requires both classes")
    return labels


def combined_training_data(
    rows: list[dict], gaps: np.ndarray, indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    indexes = np.asarray(indexes, dtype=np.int64)
    strict = strict_indexes(rows, indexes)
    official = indexes[np.abs(gaps[indexes]) > TIE_EPSILON]
    if not len(official):
        raise ValueError("official-gap branch has no non-tied relations")

    labels_by_index: dict[int, int] = {}
    strict_y = strict_labels(rows, strict)
    for index, label in zip(strict, strict_y):
        labels_by_index[int(index)] = int(label)
    official_y = (gaps[official] > 0.0).astype(np.int64)
    if set(np.unique(official_y)) != {0, 1}:
        raise ValueError("official-gap branch requires both directions")
    for index, label in zip(official, official_y):
        previous = labels_by_index.get(int(index))
        if previous is not None and previous != int(label):
            raise AssertionError("raw-IoU and official-relevance directions conflict")
        labels_by_index[int(index)] = int(label)

    combined_weights = np.zeros(len(rows), dtype=np.float64)
    main_base = scene_track_balanced_weights(rows, strict)[strict]
    main_weights = class_balanced_weights(strict_y, main_base)
    combined_weights[strict] += MAIN_LOSS_FRACTION * main_weights

    auxiliary_weights = (
        scene_track_balanced_weights(rows, official)[official] * np.abs(gaps[official])
    )
    auxiliary_weights *= len(strict) / auxiliary_weights.sum()
    combined_weights[official] += AUXILIARY_LOSS_FRACTION * auxiliary_weights

    active = np.asarray(sorted(labels_by_index), dtype=np.int64)
    labels = np.asarray([labels_by_index[int(index)] for index in active], dtype=np.int64)
    weights = combined_weights[active]
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("combined loss weights are invalid")
    expected_mass = float(len(strict))
    if not np.isclose(weights.sum(), expected_mass, atol=1e-8):
        raise AssertionError("equal normalized branches do not preserve fit mass")
    return active, labels, weights, {
        "strict_relation_count": len(strict),
        "official_non_tied_relation_count": len(official),
        "combined_unique_relation_count": len(active),
        "main_branch_weight_mass": float(MAIN_LOSS_FRACTION * main_weights.sum()),
        "auxiliary_branch_weight_mass": float(
            AUXILIARY_LOSS_FRACTION * auxiliary_weights.sum()
        ),
        "combined_weight_mass": float(weights.sum()),
    }


def _make_model(seed: int) -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            random_state=seed,
        )),
    ])


def _fit_combined(
    rows: list[dict], matrix: np.ndarray, gaps: np.ndarray,
    indexes: np.ndarray, seed: int,
) -> tuple[Pipeline, dict]:
    active, labels, weights, diagnostics = combined_training_data(rows, gaps, indexes)
    model = _make_model(seed)
    model.fit(matrix[active], labels, classifier__sample_weight=weights)
    classifier = model.named_steps["classifier"]
    diagnostics.update({
        "standardized_coefficients": classifier.coef_[0].astype(float).tolist(),
        "base_intercept": float(classifier.intercept_[0]),
    })
    return model, diagnostics


def nested_combined_fit_predict(
    rows: list[dict], matrix: np.ndarray, gaps: np.ndarray,
    train_indexes: np.ndarray, validation_indexes: np.ndarray, outer_fold_index: int,
) -> tuple[np.ndarray, dict]:
    train_indexes = np.asarray(train_indexes, dtype=np.int64)
    validation_indexes = np.asarray(validation_indexes, dtype=np.int64)
    train_scenes = [rows[index]["scene_name"] for index in train_indexes]
    folds = seeded_scene_folds(
        train_scenes, INNER_FOLD_COUNT, RANDOM_SEED + outer_fold_index
    )
    calibration_indexes = strict_indexes(rows, train_indexes)
    calibration_labels = strict_labels(rows, calibration_indexes)
    inner_decisions = np.full(len(calibration_indexes), np.nan, dtype=np.float64)
    assignments = np.zeros(len(calibration_indexes), dtype=np.int64)
    positions = {int(index): position for position, index in enumerate(calibration_indexes)}
    inner_diagnostics = []
    for inner in folds:
        inner_train_scenes = set(inner["train_scenes"])
        inner_validation_scenes = set(inner["validation_scenes"])
        inner_train = np.asarray([
            index for index in train_indexes
            if rows[index]["scene_name"] in inner_train_scenes
        ], dtype=np.int64)
        inner_validation = np.asarray([
            index for index in calibration_indexes
            if rows[index]["scene_name"] in inner_validation_scenes
        ], dtype=np.int64)
        model, fit_diagnostics = _fit_combined(
            rows, matrix, gaps, inner_train,
            RANDOM_SEED + outer_fold_index * 10 + int(inner["fold_index"]),
        )
        decisions = model.decision_function(matrix[inner_validation])
        for index, value in zip(inner_validation, decisions):
            position = positions[int(index)]
            inner_decisions[position] = float(value)
            assignments[position] += 1
        inner_diagnostics.append({
            "inner_fold_index": int(inner["fold_index"]),
            "train_scene_count": len(inner_train_scenes),
            "validation_scene_count": len(inner_validation_scenes),
            "validation_strict_relation_count": len(inner_validation),
            "fit": fit_diagnostics,
        })
    if not np.all(assignments == 1) or not np.isfinite(inner_decisions).all():
        raise AssertionError("strict calibration rows lack one inner OOF decision")
    calibration_weights = scene_track_balanced_weights(
        rows, calibration_indexes
    )[calibration_indexes]
    calibrator = _fit_platt(inner_decisions, calibration_labels, calibration_weights)
    final_model, final_fit_diagnostics = _fit_combined(
        rows, matrix, gaps, train_indexes, RANDOM_SEED + outer_fold_index
    )
    outer_decisions = np.asarray(
        final_model.decision_function(matrix[validation_indexes]), dtype=np.float64
    )
    probabilities = calibrator.predict_proba(outer_decisions.reshape(-1, 1))[:, 1]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("combined calibrated probabilities are invalid")
    return probabilities.astype(np.float64), {
        "outer_fold_index": outer_fold_index,
        "inner_folds": inner_diagnostics,
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "final_fit": final_fit_diagnostics,
    }


def load_incumbent_predictions(path: Path) -> dict[tuple[str, int, str], float]:
    lookup = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                str(row["scene_name"]), int(row["track_id"]),
                str(row["native_exact_geometry_group_id"]),
            )
            if key in lookup:
                raise ValueError(f"duplicate incumbent OOF key: {key}")
            lookup[key] = float(row["predictions"][INCUMBENT_MODEL_NAME])
    if not lookup:
        raise ValueError("incumbent OOF ledger is empty")
    return lookup


def _row_keys(rows: list[dict]) -> list[tuple[str, int, str]]:
    return [
        (str(row["scene_name"]), int(row["track_id"]),
         str(row["native_exact_geometry_group_id"]))
        for row in rows
    ]


def _logits(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(values / (1.0 - values))


def _comparison_metrics(
    rows: list[dict], indexes: np.ndarray, labels: np.ndarray,
    challenger: np.ndarray, incumbent: np.ndarray,
) -> dict:
    indexes = np.asarray(indexes, dtype=np.int64)
    weights = scene_track_balanced_weights(rows, indexes)[indexes]
    return {
        "challenger": grouped_metrics(
            [rows[index] for index in indexes], labels, challenger[indexes], weights
        ),
        "incumbent": grouped_metrics(
            [rows[index] for index in indexes], labels, incumbent[indexes], weights
        ),
    }


def run_oof(
    all_rows: list[dict], scene_to_fold: dict[str, int], cohorts: dict[str, set[str]],
    incumbent_lookup: dict[tuple[str, int, str], float],
) -> tuple[dict, list[dict], list[dict], dict]:
    rows = same_target_rows(all_rows)
    matrix = feature_matrix(rows)
    _, _, gaps = relevance_targets(rows)
    keys = _row_keys(rows)
    incumbent = np.full(len(rows), np.nan, dtype=np.float64)
    strict = strict_indexes(rows)
    for index in strict:
        if keys[index] not in incumbent_lookup:
            raise ValueError(f"incumbent OOF lacks strict relation: {keys[index]}")
        incumbent[index] = incumbent_lookup[keys[index]]

    challenger = np.full(len(rows), np.nan, dtype=np.float64)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics, diagnostics = [], []
    for fold_index in range(OUTER_FOLD_COUNT):
        train = np.asarray([
            index for index, row in enumerate(rows)
            if scene_to_fold[row["scene_name"]] != fold_index
        ], dtype=np.int64)
        validation = np.asarray([
            index for index, row in enumerate(rows)
            if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        if len({rows[index]["scene_name"] for index in train}) != 80:
            raise AssertionError("combined outer training split must contain 80 scenes")
        if len({rows[index]["scene_name"] for index in validation}) != 20:
            raise AssertionError("combined outer validation split must contain 20 scenes")
        values, model_diagnostics = nested_combined_fit_predict(
            rows, matrix, gaps, train, validation, fold_index
        )
        challenger[validation] = values
        assignments[validation] += 1
        validation_strict = strict_indexes(rows, validation)
        raw_labels = strict_labels(rows, validation_strict)
        official_strict = validation_strict[
            np.abs(gaps[validation_strict]) > TIE_EPSILON
        ]
        official_labels = (gaps[official_strict] > 0.0).astype(np.int64)
        main = _comparison_metrics(
            rows, validation_strict, raw_labels, challenger, incumbent
        )
        official_challenger = rank_metrics(
            rows, official_strict, gaps, _logits(challenger)
        )
        official_incumbent = rank_metrics(
            rows, official_strict, gaps, _logits(incumbent)
        )
        fold_metrics.append({
            "fold_index": fold_index,
            "validation_strict_relation_count": len(validation_strict),
            "validation_official_non_tied_strict_relation_count": len(official_strict),
            "main_raw_preference": main,
            "official_relevance_strict_subset": {
                "challenger": official_challenger,
                "incumbent": official_incumbent,
            },
        })
        diagnostics.append(model_diagnostics)
    if not np.all(assignments == 1) or not np.isfinite(challenger).all():
        raise AssertionError("every same-target relation requires one combined OOF score")

    raw_labels = strict_labels(rows, strict)
    official_non_tied = np.flatnonzero(np.abs(gaps) > TIE_EPSILON)
    official_strict = strict[np.abs(gaps[strict]) > TIE_EPSILON]
    main_overall = _comparison_metrics(rows, strict, raw_labels, challenger, incumbent)
    official_challenger = rank_metrics(
        rows, official_strict, gaps, _logits(challenger)
    )
    official_incumbent = rank_metrics(
        rows, official_strict, gaps, _logits(incumbent)
    )
    all_non_tied_challenger = rank_metrics(
        rows, official_non_tied, gaps, _logits(challenger)
    )

    fold_main_pr_wins = sum(
        fold["main_raw_preference"]["challenger"]["scene_track_balanced"]["pr_auc"]
        >= fold["main_raw_preference"]["incumbent"]["scene_track_balanced"]["pr_auc"]
        for fold in fold_metrics
    )
    fold_official_pr_wins = sum(
        fold["official_relevance_strict_subset"]["challenger"][
            "scene_track_balanced_pr_auc"
        ] >= fold["official_relevance_strict_subset"]["incumbent"][
            "scene_track_balanced_pr_auc"
        ] for fold in fold_metrics
    )
    main_challenger = main_overall["challenger"]["scene_track_balanced"]
    main_incumbent = main_overall["incumbent"]["scene_track_balanced"]
    gate_checks = {
        "overall_main_pr_auc_not_lower": (
            main_challenger["pr_auc"] >= main_incumbent["pr_auc"]
        ),
        "overall_official_pr_auc_not_lower": (
            official_challenger["scene_track_balanced_pr_auc"]
            >= official_incumbent["scene_track_balanced_pr_auc"]
        ),
        "overall_official_roc_auc_not_lower": (
            official_challenger["scene_track_balanced_roc_auc"]
            >= official_incumbent["scene_track_balanced_roc_auc"]
        ),
        "overall_official_gap_spearman_higher": (
            official_challenger["official_relevance_gap_spearman"]
            > official_incumbent["official_relevance_gap_spearman"]
        ),
        "main_pr_auc_non_lower_in_at_least_three_folds": fold_main_pr_wins >= 3,
        "official_pr_auc_non_lower_in_at_least_three_folds": fold_official_pr_wins >= 3,
    }
    gate_passed = all(gate_checks.values())

    cohort_metrics = {}
    for cohort_name, scenes in cohorts.items():
        cohort_strict = np.asarray([
            index for index in strict if rows[index]["scene_name"] in scenes
        ], dtype=np.int64)
        cohort_official = cohort_strict[
            np.abs(gaps[cohort_strict]) > TIE_EPSILON
        ]
        cohort_metrics[cohort_name] = {
            "scene_count": len(scenes),
            "main_raw_preference": _comparison_metrics(
                rows, cohort_strict, strict_labels(rows, cohort_strict),
                challenger, incumbent,
            ),
            "official_relevance_strict_subset": {
                "challenger": rank_metrics(
                    rows, cohort_official, gaps, _logits(challenger)
                ),
                "incumbent": rank_metrics(
                    rows, cohort_official, gaps, _logits(incumbent)
                ),
            },
        }

    oof_rows = [{
        "scene_name": row["scene_name"],
        "fold_index": int(scene_to_fold[row["scene_name"]]),
        "track_id": int(row["track_id"]),
        "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
        "label_relative_quality_state": row["labels"]["relative_quality_state"],
        "label_official_relevance_gap": float(gaps[index]),
        "prediction_multitask_prefer_track_probability": float(challenger[index]),
        "prediction_incumbent_prefer_track_probability": (
            float(incumbent[index]) if np.isfinite(incumbent[index]) else None
        ),
        "ground_truth_usage": "official_train_offline_label_only",
    } for index, row in enumerate(rows)]
    summary = {
        "scene_count": len(scene_to_fold),
        "same_target_relation_count": len(rows),
        "strict_preference_relation_count": len(strict),
        "official_non_tied_relation_count": len(official_non_tied),
        "official_non_tied_strict_relation_count": len(official_strict),
        "overall_metrics": {
            "main_raw_preference": main_overall,
            "official_relevance_strict_subset": {
                "challenger": official_challenger,
                "incumbent": official_incumbent,
            },
            "official_relevance_all_non_tied_challenger_only": all_non_tied_challenger,
        },
        "cohort_metrics": cohort_metrics,
        "selection_gate": {
            "checks": gate_checks,
            "main_pr_auc_non_lower_fold_count": fold_main_pr_wins,
            "official_pr_auc_non_lower_fold_count": fold_official_pr_wins,
            "outer_fold_count": OUTER_FOLD_COUNT,
            "passed": gate_passed,
            "ap_eligible": gate_passed,
        },
        "feature_names": list(FEATURES),
        "model_selection_applied": False,
    }
    return summary, oof_rows, fold_metrics, {"folds": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--incumbent-oof-path", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
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
    incumbent_lookup = load_incumbent_predictions(args.incumbent_oof_path)
    summary_core, oof_rows, fold_metrics, diagnostics = run_oof(
        rows, scene_to_fold, cohorts, incumbent_lookup
    )
    summary = {
        "version": f"{args.protocol_name}_candidate_relative_quality_multitask_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "training_contract": {
            "main_loss": "class-balanced strict prefer_track/prefer_native logistic loss",
            "main_loss_fraction": MAIN_LOSS_FRACTION,
            "auxiliary_loss": "abs(official_relevance_gap) weighted direction logistic loss",
            "auxiliary_loss_fraction": AUXILIARY_LOSS_FRACTION,
            "branch_normalization": "each branch normalized to strict-relation fit mass before 0.5 mixing",
            "calibration": "four-fold scene-disjoint inner OOF Platt map to strict prefer_track probability",
            "model": "standardized L2 logistic regression, C=1.0",
            "features": "frozen B_plus_directional_geometry only",
            "hyperparameter_scan": False,
        },
        "safety_contract": {
            "fit_all_model_exported": False,
            "threshold_selected": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "candidate_deleted": False,
            "ap_evaluation_run": False,
            "holdout_dataset_read": False,
        },
        "input_provenance": {
            "feature_ledger_summary_path": str((args.feature_ledger_root / "summary.json").resolve()),
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "incumbent_oof_path": str(args.incumbent_oof_path.resolve()),
            "incumbent_oof_sha256": _sha256(args.incumbent_oof_path),
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": _sha256(args.scene_list),
            "existing20_scene_list_path": str(args.existing20_scene_list.resolve()),
            "existing20_scene_list_sha256": _sha256(args.existing20_scene_list),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": split_sha,
            "split_existing_scene_count": split_payload.get("existing_scene_count"),
            "split_new_scene_count": split_payload.get("new_scene_count"),
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
        if not summary["selection_gate"]["passed"]:
            (staging / "NOT_SELECTED_DO_NOT_DEPLOY.json").write_text(json.dumps({
                "decision": "NOT_SELECTED_DO_NOT_DEPLOY",
                "reason": "The fixed 50/50 multitask head failed at least one pre-registered OOF gate.",
                "selection_gate": summary["selection_gate"],
                "ap_evaluation_run": False,
                "fit_all_model_exported": False,
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

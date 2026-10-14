#!/usr/bin/env python3
"""Diagnose a fixed official-relevance-gap ranking loss on official100.

For every reliable same-target track--native relation, candidate quality is
the fraction of official AP50:95 IoU thresholds passed.  The signed target is
``R(track) - R(native)``.  A scene-disjoint logistic ranker uses the frozen
directional-geometry features and the fixed loss

    abs(gap) * log(1 + exp(-sign(gap) * relation_score)).

Tied relevance pairs are scored but do not supervise the ranking loss.  This
entry point is diagnostic only: it does not select a threshold, export an
all-scene model, mutate candidates, generate actions, or run AP.
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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
from tools.train_candidate_relative_quality_oof import (  # noqa: E402
    MODEL_FEATURES as RELATIVE_FEATURES,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    LEARNED_QUALITY_MARKERS,
    OUTER_FOLD_COUNT,
    RANDOM_SEED,
    scene_track_balanced_weights,
)


OFFICIAL_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
FEATURES = RELATIVE_FEATURES["B_plus_directional_geometry"]
TIE_EPSILON = 1e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_relevance(iou: float) -> float:
    value = float(iou)
    if not math.isfinite(value):
        raise ValueError("official relevance requires a finite IoU")
    return float(np.mean([value > threshold for threshold in OFFICIAL_THRESHOLDS]))


def same_target_rows(rows: list[dict]) -> list[dict]:
    selected = [
        row for row in rows
        if row["labels"]["reliable_pair"]
        and row["labels"]["target_state"] == "same_target"
        and row["labels"]["relative_quality_state"]
        in ("prefer_track", "prefer_native", "equivalent_abstain")
    ]
    if not selected:
        raise ValueError("no reliable same-target relations")
    return selected


def relevance_targets(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track = np.asarray([
        official_relevance(row["labels"]["track_best_gt_iou"]) for row in rows
    ], dtype=np.float64)
    native = np.asarray([
        official_relevance(row["labels"]["native_best_gt_iou"]) for row in rows
    ], dtype=np.float64)
    gap = track - native
    if not np.isfinite(gap).all():
        raise ValueError("official relevance gaps are non-finite")
    return track, native, gap


def feature_matrix(rows: list[dict]) -> np.ndarray:
    forbidden = [
        name for name in FEATURES
        if any(marker in name for marker in LEARNED_QUALITY_MARKERS)
        or name.startswith("label_") or "gt_" in name
    ]
    if forbidden:
        raise ValueError(f"ranking features contain learned/GT fields: {forbidden}")
    matrix = np.asarray([
        [float(row["features"][name]) for name in FEATURES] for row in rows
    ], dtype=np.float64)
    if matrix.shape != (len(rows), len(FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError("official relevance rank matrix is malformed or non-finite")
    return matrix


def fit_gap_ranker(
    rows: list[dict], matrix: np.ndarray, gaps: np.ndarray,
    train_indexes: np.ndarray, validation_indexes: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict]:
    train_indexes = np.asarray(train_indexes, dtype=np.int64)
    validation_indexes = np.asarray(validation_indexes, dtype=np.int64)
    active = train_indexes[np.abs(gaps[train_indexes]) > TIE_EPSILON]
    labels = (gaps[active] > 0.0).astype(np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("official relevance rank training partition lacks a direction")
    base = scene_track_balanced_weights(rows, active)[active]
    weights = base * np.abs(gaps[active])
    weights *= len(weights) / weights.sum()
    model = Pipeline([
        ("standardize", StandardScaler()),
        ("rank", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            random_state=seed,
        )),
    ])
    model.fit(matrix[active], labels, rank__sample_weight=weights)
    scores = np.asarray(model.decision_function(matrix[validation_indexes]), dtype=np.float64)
    if scores.shape != (len(validation_indexes),) or not np.isfinite(scores).all():
        raise ValueError("official relevance ranker emitted malformed scores")
    classifier = model.named_steps["rank"]
    return scores, {
        "train_non_tied_relation_count": len(active),
        "train_track_better_count": int(labels.sum()),
        "train_native_better_count": int((labels == 0).sum()),
        "loss_weight": "scene-track balanced mass multiplied by abs(official_relevance_gap)",
        "standardized_coefficients": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def rank_metrics(
    rows: list[dict], indexes: np.ndarray, gaps: np.ndarray, scores: np.ndarray,
) -> dict:
    indexes = np.asarray(indexes, dtype=np.int64)
    indexes = indexes[np.abs(gaps[indexes]) > TIE_EPSILON]
    if not len(indexes):
        raise ValueError("rank metrics require non-tied relevance gaps")
    labels = (gaps[indexes] > 0.0).astype(np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("rank metrics require both relevance directions")
    scene_weights = scene_track_balanced_weights(rows, indexes)[indexes]
    loss_weights = scene_weights * np.abs(gaps[indexes])
    values = scores[indexes]
    directions = np.where(labels == 1, 1.0, -1.0)
    correlation = spearmanr(gaps[indexes], values)
    correlation_value = float(getattr(
        correlation, "statistic", getattr(correlation, "correlation", float("nan"))
    ))
    return {
        "count": len(indexes),
        "track_better_count": int(labels.sum()),
        "native_better_count": int((labels == 0).sum()),
        "mean_absolute_official_relevance_gap": float(np.mean(np.abs(gaps[indexes]))),
        "gap_weighted_pairwise_logistic_loss": float(np.average(
            np.logaddexp(0.0, -directions * values), weights=loss_weights
        )),
        "gap_weighted_direction_accuracy_at_zero": float(np.average(
            (values > 0.0) == labels, weights=loss_weights
        )),
        "relation_unweighted_pr_auc": float(average_precision_score(labels, values)),
        "relation_unweighted_roc_auc": float(roc_auc_score(labels, values)),
        "scene_track_balanced_pr_auc": float(average_precision_score(
            labels, values, sample_weight=scene_weights
        )),
        "scene_track_balanced_roc_auc": float(roc_auc_score(
            labels, values, sample_weight=scene_weights
        )),
        "official_relevance_gap_spearman": (
            correlation_value if np.isfinite(correlation_value) else None
        ),
    }


def run_oof(
    all_rows: list[dict], scene_to_fold: dict[str, int], cohorts: dict[str, set[str]],
) -> tuple[dict, list[dict], list[dict], dict]:
    rows = same_target_rows(all_rows)
    matrix = feature_matrix(rows)
    track_relevance, native_relevance, gaps = relevance_targets(rows)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
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
        train_scenes = {rows[index]["scene_name"] for index in train}
        validation_scenes = {rows[index]["scene_name"] for index in validation}
        if len(train_scenes) != 80 or len(validation_scenes) != 20 or train_scenes & validation_scenes:
            raise AssertionError("official relevance rank rows violate frozen outer split")
        scores, model_diagnostics = fit_gap_ranker(
            rows, matrix, gaps, train, validation, RANDOM_SEED + fold_index
        )
        predictions[validation] = scores
        assignments[validation] += 1
        active_validation = validation[np.abs(gaps[validation]) > TIE_EPSILON]
        baseline = np.asarray([
            float(row["features"]["original_score_delta_track_minus_native"])
            for row in rows
        ], dtype=np.float64)
        fold_metrics.append({
            "fold_index": fold_index,
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "validation_same_target_relation_count": len(validation),
            "validation_non_tied_relation_count": len(active_validation),
            "gap_weighted_ranker": rank_metrics(rows, active_validation, gaps, predictions),
            "original_score_delta_baseline": rank_metrics(rows, active_validation, gaps, baseline),
        })
        diagnostics.append({"fold_index": fold_index, **model_diagnostics})
    if not np.all(assignments == 1) or not np.isfinite(predictions).all():
        raise AssertionError("every same-target relation needs exactly one OOF rank score")
    baseline = np.asarray([
        float(row["features"]["original_score_delta_track_minus_native"]) for row in rows
    ], dtype=np.float64)
    non_tied = np.flatnonzero(np.abs(gaps) > TIE_EPSILON)
    strict = np.asarray([
        index for index in non_tied
        if rows[index]["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native")
    ], dtype=np.int64)
    cohort_metrics = {}
    for cohort_name, scenes in cohorts.items():
        indexes = np.asarray([
            index for index in non_tied if rows[index]["scene_name"] in scenes
        ], dtype=np.int64)
        cohort_metrics[cohort_name] = {
            "scene_count": len(scenes),
            "gap_weighted_ranker": rank_metrics(rows, indexes, gaps, predictions),
            "original_score_delta_baseline": rank_metrics(rows, indexes, gaps, baseline),
        }
    oof_rows = [{
        "scene_name": row["scene_name"],
        "fold_index": int(scene_to_fold[row["scene_name"]]),
        "track_id": int(row["track_id"]),
        "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
        "label_track_official_relevance": float(track_relevance[index]),
        "label_native_official_relevance": float(native_relevance[index]),
        "label_official_relevance_gap": float(gaps[index]),
        "label_relative_quality_state": row["labels"]["relative_quality_state"],
        "prediction_gap_rank_score": float(predictions[index]),
        "original_score_delta_baseline": float(baseline[index]),
        "ground_truth_usage": "official_train_offline_label_only",
    } for index, row in enumerate(rows)]
    summary = {
        "scene_count": len(scene_to_fold),
        "same_target_relation_count": len(rows),
        "non_tied_official_relevance_relation_count": len(non_tied),
        "tied_official_relevance_relation_count": int(len(rows) - len(non_tied)),
        "track_better_count": int(np.sum(gaps[non_tied] > 0.0)),
        "native_better_count": int(np.sum(gaps[non_tied] < 0.0)),
        "overall_metrics": {
            "all_non_tied": {
                "gap_weighted_ranker": rank_metrics(rows, non_tied, gaps, predictions),
                "original_score_delta_baseline": rank_metrics(rows, non_tied, gaps, baseline),
            },
            "strict_raw_iou_preference_subset": {
                "gap_weighted_ranker": rank_metrics(rows, strict, gaps, predictions),
                "original_score_delta_baseline": rank_metrics(rows, strict, gaps, baseline),
            },
        },
        "cohort_metrics": cohort_metrics,
        "feature_names": list(FEATURES),
        "model_selection_applied": False,
    }
    return summary, oof_rows, fold_metrics, {"folds": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
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
    summary_core, oof_rows, fold_metrics, diagnostics = run_oof(rows, scene_to_fold, cohorts)
    summary = {
        "version": f"{args.protocol_name}_candidate_official_relevance_gap_rank_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "training_contract": {
            "population": "reliable same-target relations including raw-IoU equivalent/abstain",
            "label": "R(track)-R(native), where R is mean[IoU > AP threshold] for 0.50:0.05:0.90",
            "loss": "abs(gap) * softplus(-sign(gap) * relation_score)",
            "tie_handling": "zero official-relevance gaps do not supervise; they still receive OOF scores",
            "model": "standardized L2 logistic regression, C=1.0",
            "features": "frozen B_plus_directional_geometry only",
            "class_balancing": False,
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
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

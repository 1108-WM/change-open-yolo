#!/usr/bin/env python3
"""Audit a threshold-free two-stage OOF cascade over every relation.

Stage one is the frozen B overlap-geometry target-consistency model.  Stage two
is the frozen E directional-geometry plus nested candidate-q relative-quality
model.  Both models are refit inside each frozen outer fold and applied to all
relations in that fold, including unknown and different-target coexist rows.
The sole cascade score is the preregistered product
``P(same target) * P(track better | same target)``.

This diagnostic reports ranking and fixed top-fraction safety only.  It does
not select a threshold, generate an action, mutate candidates, or run AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_quality_head_oof import load_rows as load_candidate_rows  # noqa: E402
from tools.train_candidate_relative_quality_nested_q_oof import (  # noqa: E402
    nested_candidate_q_predictions,
    relation_matrix_with_nested_q,
)
from tools.train_candidate_relative_quality_oof import (  # noqa: E402
    MODEL_FEATURES as RELATIVE_FEATURES,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    MODEL_FEATURES as TARGET_FEATURES,
    OUTER_FOLD_COUNT,
    nested_fit_predict,
)


TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05)
TARGET_MODEL_NAME = "B_overlap_geometry"
RELATIVE_MODEL_NAME = "E_directional_geometry_plus_nested_q"
CASCADE_SCORE_NAME = "two_stage_product"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(rows: list[dict], feature_names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([
        [float(row["features"][name]) for name in feature_names] for row in rows
    ], dtype=np.float64)
    if values.shape != (len(rows), len(feature_names)) or not np.isfinite(values).all():
        raise ValueError("cascade relation feature matrix is malformed")
    return values


def _target_labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([
        int(row["labels"]["target_state"] == "same_target") for row in rows
    ], dtype=np.int64)


def _relative_labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([
        int(row["labels"]["relative_quality_state"] == "prefer_track") for row in rows
    ], dtype=np.int64)


def _cascade_product(
    target_same_probability: np.ndarray,
    relative_track_better_probability: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_same_probability, dtype=np.float64)
    relative = np.asarray(relative_track_better_probability, dtype=np.float64)
    if target.shape != relative.shape:
        raise ValueError("two-stage cascade probability arrays must have the same shape")
    if (
        not np.isfinite(target).all()
        or not np.isfinite(relative).all()
        or np.any((target < 0) | (target > 1))
        or np.any((relative < 0) | (relative > 1))
    ):
        raise ValueError("two-stage cascade inputs must be finite probabilities in [0, 1]")
    return target * relative


def _target_training_indexes(rows: list[dict], scene_to_fold: dict[str, int], fold_index: int) -> np.ndarray:
    return np.asarray([
        index for index, row in enumerate(rows)
        if scene_to_fold[row["scene_name"]] != fold_index
        and row["labels"]["target_state"] in ("same_target", "different_target_coexist")
    ], dtype=np.int64)


def _relative_training_indexes(rows: list[dict], scene_to_fold: dict[str, int], fold_index: int) -> np.ndarray:
    return np.asarray([
        index for index, row in enumerate(rows)
        if scene_to_fold[row["scene_name"]] != fold_index
        and row["labels"]["target_state"] == "same_target"
        and row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native")
    ], dtype=np.int64)


def _reference_lookup(path: Path, model_name: str) -> dict[tuple[str, int, str], float]:
    lookup = {}
    for row in read_jsonl(path):
        key = (
            str(row["scene_name"]), int(row["track_id"]),
            str(row["native_exact_geometry_group_id"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate reference OOF relation: {key}")
        lookup[key] = float(row["predictions"][model_name])
    return lookup


def _validate_reference_parity(
    rows: list[dict],
    predictions: np.ndarray,
    reference: dict[tuple[str, int, str], float],
    state: str,
) -> dict:
    differences = []
    matched = 0
    for index, row in enumerate(rows):
        if state == "target":
            eligible = row["labels"]["target_state"] in ("same_target", "different_target_coexist")
        elif state == "relative":
            eligible = (
                row["labels"]["target_state"] == "same_target"
                and row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native")
            )
        else:
            raise ValueError(f"unknown parity state: {state}")
        if not eligible:
            continue
        key = (
            str(row["scene_name"]), int(row["track_id"]),
            str(row["native_exact_geometry_group_id"]),
        )
        if key not in reference:
            raise ValueError(f"missing {state} reference prediction: {key}")
        differences.append(abs(float(predictions[index]) - reference[key]))
        matched += 1
    maximum = max(differences, default=0.0)
    if maximum > 1e-12 or matched != len(reference):
        raise AssertionError(
            f"{state} cascade predictions differ from frozen task OOF: matched={matched}, max={maximum}"
        )
    return {"matched_relation_count": matched, "max_absolute_difference": maximum}


def _wilson(successes: int, total: int) -> dict | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    value = successes / total
    denominator = 1 + z * z / total
    centre = (value + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        value * (1 - value) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "level": 0.95,
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
    }


def _top_metrics(rows: list[dict], labels: np.ndarray, scores: np.ndarray) -> dict:
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(scores[index]), rows[index]["scene_name"], int(rows[index]["track_id"]),
            rows[index]["native_exact_geometry_group_id"],
        ),
    )
    positives = int(labels.sum())
    result = {}
    for fraction in TOP_FRACTIONS:
        count = min(len(rows), max(1, math.ceil(len(rows) * fraction)))
        selected = order[:count]
        correct = int(labels[selected].sum())
        error_relative_states = Counter(
            rows[index]["labels"]["relative_quality_state"]
            for index in selected if not labels[index]
        )
        error_target_states = Counter(
            rows[index]["labels"]["target_state"]
            for index in selected if not labels[index]
        )
        error_joint_states = Counter(
            f'{rows[index]["labels"]["target_state"]}:'
            f'{rows[index]["labels"]["relative_quality_state"]}'
            for index in selected if not labels[index]
        )
        key = f"top_{fraction * 100:g}pct".replace(".", "_")
        result[key] = {
            "selected_count": count,
            "correct_count": correct,
            "precision": correct / count,
            "recall": correct / positives if positives else None,
            "precision_wilson_95": _wilson(correct, count),
            "positive_scene_count": len({
                rows[index]["scene_name"] for index in selected if labels[index]
            }),
            "error_relative_quality_state_counts": dict(sorted(error_relative_states.items())),
            "error_target_state_counts": dict(sorted(error_target_states.items())),
            "error_joint_state_counts": dict(sorted(error_joint_states.items())),
            "score_min": float(min(scores[index] for index in selected)),
            "score_max": float(max(scores[index] for index in selected)),
        }
    return result


def _score_metrics(rows: list[dict], labels: np.ndarray, scores: np.ndarray) -> dict:
    result = {
        "count": len(rows),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if np.any(labels) else None,
        "top_selection": _top_metrics(rows, labels, scores),
    }
    return result


def run_cascade(
    rows: list[dict],
    candidate_rows: list[dict],
    scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]],
    candidate_protocol_name: str,
    target_reference: dict,
    relative_reference: dict,
) -> tuple[dict, list[dict], list[dict], dict]:
    target_features = TARGET_FEATURES[TARGET_MODEL_NAME]
    relative_raw_features = RELATIVE_FEATURES["B_plus_directional_geometry"]
    target_matrix = _matrix(rows, target_features)
    target_labels = _target_labels(rows)
    relative_labels = _relative_labels(rows)
    target_predictions = np.full(len(rows), np.nan, dtype=np.float64)
    relative_predictions = np.full(len(rows), np.nan, dtype=np.float64)
    nested_q_evidence: list[dict | None] = [None] * len(rows)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics, diagnostics = [], []
    for fold_index in range(OUTER_FOLD_COUNT):
        validation = np.asarray([
            index for index, row in enumerate(rows)
            if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        target_train = _target_training_indexes(rows, scene_to_fold, fold_index)
        relative_train = _relative_training_indexes(rows, scene_to_fold, fold_index)
        target_values, target_diagnostics = nested_fit_predict(
            rows, target_matrix, target_labels, target_train, validation, fold_index
        )
        q_lookup, q_diagnostics = nested_candidate_q_predictions(
            candidate_rows, fold_index, scene_to_fold, candidate_protocol_name
        )
        relative_matrix, evidence = relation_matrix_with_nested_q(
            rows, relative_raw_features, q_lookup
        )
        relative_values, relative_diagnostics = nested_fit_predict(
            rows, relative_matrix, relative_labels, relative_train, validation, fold_index
        )
        target_predictions[validation] = target_values
        relative_predictions[validation] = relative_values
        assignments[validation] += 1
        for index in validation:
            nested_q_evidence[index] = evidence[index]
        fold_rows = [rows[index] for index in validation]
        fold_labels = relative_labels[validation]
        product = _cascade_product(target_values, relative_values)
        fold_metrics.append({
            "fold_index": fold_index,
            "validation_relation_count": len(validation),
            "prefer_track_count": int(fold_labels.sum()),
            "scores": {
                "target_same_probability": _score_metrics(fold_rows, fold_labels, target_values),
                "relative_track_better_probability": _score_metrics(fold_rows, fold_labels, relative_values),
                CASCADE_SCORE_NAME: _score_metrics(fold_rows, fold_labels, product),
            },
        })
        diagnostics.append({
            "fold_index": fold_index,
            "target_training_relation_count": len(target_train),
            "relative_training_relation_count": len(relative_train),
            "validation_relation_count": len(validation),
            "target_model": target_diagnostics,
            "nested_candidate_quality": q_diagnostics,
            "relative_model": relative_diagnostics,
        })
    if not np.all(assignments == 1):
        raise AssertionError("every relation must be assigned to one cascade validation fold")
    if not np.isfinite(target_predictions).all() or not np.isfinite(relative_predictions).all():
        raise AssertionError("cascade contains missing predictions")
    if any(value is None for value in nested_q_evidence):
        raise AssertionError("cascade nested-q evidence is incomplete")
    parity = {
        "target_consistency": _validate_reference_parity(
            rows, target_predictions, target_reference, "target"
        ),
        "relative_quality": _validate_reference_parity(
            rows, relative_predictions, relative_reference, "relative"
        ),
    }
    product = _cascade_product(target_predictions, relative_predictions)
    overall = {
        "target_same_probability": _score_metrics(rows, relative_labels, target_predictions),
        "relative_track_better_probability": _score_metrics(rows, relative_labels, relative_predictions),
        CASCADE_SCORE_NAME: _score_metrics(rows, relative_labels, product),
    }
    cohort_metrics = {}
    for cohort_name, cohort_scenes in cohorts.items():
        indexes = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in cohort_scenes
        ], dtype=np.int64)
        cohort_rows = [rows[index] for index in indexes]
        labels = relative_labels[indexes]
        cohort_metrics[cohort_name] = {
            "scene_count": len(cohort_scenes),
            "relation_count": len(indexes),
            "prefer_track_count": int(labels.sum()),
            "scores": {
                "target_same_probability": _score_metrics(
                    cohort_rows, labels, target_predictions[indexes]
                ),
                "relative_track_better_probability": _score_metrics(
                    cohort_rows, labels, relative_predictions[indexes]
                ),
                CASCADE_SCORE_NAME: _score_metrics(
                    cohort_rows, labels, product[indexes]
                ),
            },
        }
    scored_rows = []
    for index, row in enumerate(rows):
        scored_rows.append({
            "scene_name": row["scene_name"],
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "label_target_state": row["labels"]["target_state"],
            "label_relative_quality_state": row["labels"]["relative_quality_state"],
            "label_safe_replacement": int(relative_labels[index]),
            "target_same_probability": float(target_predictions[index]),
            "relative_track_better_probability": float(relative_predictions[index]),
            CASCADE_SCORE_NAME: float(product[index]),
            "nested_candidate_quality_features": nested_q_evidence[index],
            "ground_truth_usage": "official_train_offline_label_only",
        })
    summary = {
        "scene_count": len(scene_to_fold),
        "relation_count": len(rows),
        "prefer_track_count": int(relative_labels.sum()),
        "overall_scores": overall,
        "cohort_metrics": cohort_metrics,
        "reference_prediction_parity": parity,
    }
    return summary, scored_rows, fold_metrics, {"folds": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--candidate-records-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--target-oof-predictions", type=Path, required=True)
    parser.add_argument("--relative-oof-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    actual_split_sha = _sha256(args.split_manifest)
    if actual_split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    rows = _load_rows(args.feature_ledger_root, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    target_reference = _reference_lookup(args.target_oof_predictions, TARGET_MODEL_NAME)
    relative_reference = _reference_lookup(args.relative_oof_predictions, RELATIVE_MODEL_NAME)
    summary_core, scored_rows, fold_metrics, diagnostics = run_cascade(
        rows, candidate_rows, scene_to_fold, cohorts, args.candidate_quality_protocol_name,
        target_reference, relative_reference,
    )
    summary = {
        "version": f"{args.protocol_name}_candidate_two_stage_cascade_oof_audit_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "cascade_contract": {
            "target_model": TARGET_MODEL_NAME,
            "relative_quality_model": RELATIVE_MODEL_NAME,
            "score_formula": "P(same target) * P(track better | same target)",
            "all_relations_scored": True,
            "unknown_relations_scored_for_safety_audit": True,
            "top_fractions": list(TOP_FRACTIONS),
        },
        "safety_contract": {
            "threshold_selected": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "fit_all_model_exported": False,
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
            "target_oof_predictions_path": str(args.target_oof_predictions.resolve()),
            "target_oof_predictions_sha256": _sha256(args.target_oof_predictions),
            "relative_oof_predictions_path": str(args.relative_oof_predictions.resolve()),
            "relative_oof_predictions_sha256": _sha256(args.relative_oof_predictions),
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
        (staging / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "scored_relations.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored_rows
        ))
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scene-isolated OOF diagnosis for useful novel pair proposals.

Only proposals whose geometry is not already in the frozen candidate pool are
scored.  The binary target is whether appending the union crosses at least one
official IoU threshold for a GT not already covered at that threshold.  No
candidate is materialized and AP is never computed.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.build_train_candidate_pair_union_utility_ledger import VERSIONS as LEDGER_VERSIONS  # noqa: E402
from tools.train_candidate_component_action_head_oof import EXPECTED_SPLIT_SHA256  # noqa: E402
from tools.train_candidate_component_list_calibration_head_oof import MODEL_PARAMS  # noqa: E402
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


VERSIONS = {
    "pair_union": "official100_pair_union_threshold_cross_oof_v1",
    "pair_intersection": "official100_pair_intersection_threshold_cross_oof_v1",
}
VERSION = VERSIONS["pair_union"]
RANDOM_SEED = 20260815
MIN_VALIDATION_POSITIVE_COUNT = 2


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0]["model_features"])
    if any(sorted(row["model_features"]) != names for row in rows):
        raise ValueError("pair-union feature schemas differ")
    forbidden = [name for name in names if "label" in name or "best_gt" in name]
    if forbidden:
        raise AssertionError(f"GT fields leaked into proposal features: {forbidden}")
    matrix = np.asarray([
        [float(row["model_features"][name]) for name in names] for row in rows
    ], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("pair-union feature matrix contains non-finite values")
    return matrix, names


def _component_weights(rows: list[dict], indexes: np.ndarray) -> np.ndarray:
    counts = Counter(
        (rows[int(index)]["scene_name"], rows[int(index)]["relation_component_id"])
        for index in indexes
    )
    weights = np.zeros(len(rows), dtype=np.float64)
    for index in indexes:
        row = rows[int(index)]
        weights[index] = 1.0 / counts[(row["scene_name"], row["relation_component_id"])]
    weights[indexes] /= weights[indexes].mean()
    return weights


def _balanced_weights(
    rows: list[dict], indexes: np.ndarray, labels: np.ndarray,
) -> np.ndarray:
    weights = _component_weights(rows, indexes)
    for value in (0, 1):
        selected = indexes[labels[indexes] == value]
        if not len(selected):
            raise ValueError("proposal fold lacks one target class")
        weights[selected] *= 0.5 * len(indexes) / weights[selected].sum()
    weights[indexes] /= weights[indexes].mean()
    return weights


def prior_correct_balanced_probability(
    balanced_probability: np.ndarray, natural_positive_rate: float,
) -> np.ndarray:
    """Undo the artificial 50/50 class prior used for rare-state fitting."""
    probability = np.clip(np.asarray(balanced_probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    prior = min(1 - 1e-6, max(1e-6, float(natural_positive_rate)))
    odds = probability / (1.0 - probability)
    corrected_odds = odds * prior / (1.0 - prior)
    return corrected_odds / (1.0 + corrected_odds)


def heuristic_union_probability(row: dict) -> float:
    """Fixed no-GT baseline: both sources must be credible and complementary."""
    features = row["model_features"]
    track = min(1.0, max(0.0, float(features["relation__track_original_score"])))
    native = min(1.0, max(0.0, float(features["relation__native_original_score_median"])))
    overlap = min(1.0, max(0.0, float(features["relation__point_iou"])))
    return min(track, native) * (1.0 - overlap)


def heuristic_intersection_probability(row: dict) -> float:
    """Fixed no-GT baseline: both sources must be credible and overlapping."""
    features = row["model_features"]
    track = min(1.0, max(0.0, float(features["relation__track_original_score"])))
    native = min(1.0, max(0.0, float(features["relation__native_original_score_median"])))
    overlap = min(1.0, max(0.0, float(features["relation__point_iou"])))
    return min(track, native) * overlap


def heuristic_proposal_probability(kind: str, row: dict) -> float:
    if kind == "pair_union":
        return heuristic_union_probability(row)
    if kind == "pair_intersection":
        return heuristic_intersection_probability(row)
    raise ValueError(f"unsupported pair proposal kind: {kind}")


def metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray,
) -> dict:
    return {
        "count": len(labels),
        "positive_count": int(labels.sum()),
        "component_balanced_positive_rate": float(np.average(labels, weights=weights)),
        "component_balanced_pr_auc": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "component_balanced_roc_auc": float(
            roc_auc_score(labels, scores, sample_weight=weights)
        ),
        "component_balanced_brier": float(
            brier_score_loss(labels, scores, sample_weight=weights)
        ),
        "component_balanced_log_loss": float(
            log_loss(labels, scores, sample_weight=weights, labels=[0, 1])
        ),
        "unweighted_pr_auc": float(average_precision_score(labels, scores)),
        "unweighted_roc_auc": float(roc_auc_score(labels, scores)),
    }


def _delta(challenger: dict, baseline: dict) -> dict:
    return {
        name: challenger[name] - baseline[name]
        for name in (
            "component_balanced_pr_auc", "component_balanced_roc_auc",
            "component_balanced_brier", "component_balanced_log_loss",
            "unweighted_pr_auc", "unweighted_roc_auc",
        )
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("must use frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    ledger_summary_path = args.utility_ledger_root / "summary.json"
    ledger_summary = json.loads(ledger_summary_path.read_text())
    if ledger_summary.get("version") != LEDGER_VERSIONS[args.proposal_kind]:
        raise ValueError(f"unexpected {args.proposal_kind} utility ledger version")
    if ledger_summary.get("proposal_kind", "pair_union") != args.proposal_kind:
        raise ValueError("pair proposal kind differs from requested diagnosis")
    if not ledger_summary.get("learned_candidate_quality_scores_excluded_from_features"):
        raise ValueError("proposal features must exclude learned candidate-quality scores")
    all_rows = read_jsonl(
        args.utility_ledger_root / f"{args.proposal_kind}_utilities.jsonl"
    )
    positive_outside_novel = [
        row for row in all_rows
        if row["labels"]["crosses_any_official_threshold"]
        and not row["model_features"]["proposal__geometry_novel_vs_existing"]
    ]
    if positive_outside_novel:
        raise ValueError("an exact existing geometry cannot add threshold coverage")
    rows = [
        row for row in all_rows
        if row["model_features"]["proposal__geometry_novel_vs_existing"]
        and row["model_features"]["proposal__accepted_min_region"]
    ]
    matrix, feature_names = _matrix(rows)
    labels = np.asarray([
        int(row["labels"]["crosses_any_official_threshold"]) for row in rows
    ], dtype=np.int64)
    heuristic = np.asarray([
        heuristic_proposal_probability(args.proposal_kind, row) for row in rows
    ])
    scene_array = np.asarray([row["scene_name"] for row in rows], dtype=object)
    oof_scores = np.full(len(rows), np.nan, dtype=np.float64)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_details = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        train = np.flatnonzero(np.isin(scene_array, fold["train_scenes"]))
        validation = np.flatnonzero(np.isin(scene_array, fold["validation_scenes"]))
        if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[validation])) != {0, 1}:
            raise ValueError(f"fold {fold_index}: proposal labels lack one class")
        natural_weights = _component_weights(rows, train)
        natural_rate = float(np.average(labels[train], weights=natural_weights[train]))
        balanced = _balanced_weights(rows, train, labels)
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            **{**MODEL_PARAMS, "random_state": RANDOM_SEED + fold_index * 100},
        )
        model.fit(matrix[train], labels[train], sample_weight=balanced[train])
        raw = model.predict_proba(matrix[validation])[:, 1]
        scores = prior_correct_balanced_probability(raw, natural_rate)
        oof_scores[validation] = scores
        assignments[validation] += 1
        validation_weights = _component_weights(rows, validation)[validation]
        baseline_metrics = metrics(labels[validation], heuristic[validation], validation_weights)
        challenger_metrics = metrics(labels[validation], scores, validation_weights)
        fold_details.append({
            "fold_index": fold_index,
            "train_count": len(train),
            "train_positive_count": int(labels[train].sum()),
            "validation_count": len(validation),
            "validation_positive_count": int(labels[validation].sum()),
            "training_component_balanced_natural_positive_rate": natural_rate,
            "fixed_complementarity_heuristic": baseline_metrics,
            f"{args.proposal_kind}_threshold_cross_head": challenger_metrics,
            "delta_head_minus_heuristic": _delta(challenger_metrics, baseline_metrics),
        })
        print(
            f"[{args.proposal_kind} threshold-cross OOF] fold {fold_index} complete",
            flush=True,
        )
    if not np.all(assignments == 1) or not np.isfinite(oof_scores).all():
        raise AssertionError("OOF proposal predictions are incomplete")

    indexes = np.arange(len(rows), dtype=np.int64)
    overall_weights = _component_weights(rows, indexes)[indexes]
    baseline_overall = metrics(labels, heuristic, overall_weights)
    challenger_overall = metrics(labels, oof_scores, overall_weights)
    delta_overall = _delta(challenger_overall, baseline_overall)
    pr_wins = sum(
        fold["delta_head_minus_heuristic"]["component_balanced_pr_auc"] >= 0.0
        for fold in fold_details
    )
    roc_wins = sum(
        fold["delta_head_minus_heuristic"]["component_balanced_roc_auc"] >= 0.0
        for fold in fold_details
    )
    brier_wins = sum(
        fold["delta_head_minus_heuristic"]["component_balanced_brier"] <= 0.0
        for fold in fold_details
    )
    log_wins = sum(
        fold["delta_head_minus_heuristic"]["component_balanced_log_loss"] <= 0.0
        for fold in fold_details
    )
    minimum_validation_positive_count = min(
        fold["validation_positive_count"] for fold in fold_details
    )
    gates = {
        "at_least_two_positive_proposals_in_every_validation_fold": (
            minimum_validation_positive_count >= MIN_VALIDATION_POSITIVE_COUNT
        ),
        "overall_pr_not_lower": delta_overall["component_balanced_pr_auc"] >= 0.0,
        "overall_roc_not_lower": delta_overall["component_balanced_roc_auc"] >= 0.0,
        "overall_brier_not_higher": delta_overall["component_balanced_brier"] <= 0.0,
        "overall_log_loss_not_higher": delta_overall["component_balanced_log_loss"] <= 0.0,
        "pr_non_lower_in_at_least_three_folds": pr_wins >= 3,
        "roc_non_lower_in_at_least_three_folds": roc_wins >= 3,
        "brier_non_higher_in_at_least_three_folds": brier_wins >= 3,
        "log_loss_non_higher_in_at_least_three_folds": log_wins >= 3,
    }
    gates["proposal_materialization_allowed"] = all(gates.values())

    prediction_rows = []
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    for row, baseline, score in zip(rows, heuristic, oof_scores):
        prediction_rows.append({
            "scene_name": row["scene_name"],
            "relation_component_id": int(row["relation_component_id"]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "fold_index": scene_to_fold[row["scene_name"]],
            "fixed_complementarity_heuristic": float(baseline),
            "threshold_cross_probability": float(score),
            "label_crosses_any_official_threshold": int(
                row["labels"]["crosses_any_official_threshold"]
            ),
            "label_official_threshold_cross_count": int(
                row["labels"]["official_threshold_cross_count"]
            ),
            "label_iou_gain": float(row["labels"]["iou_gain"]),
        })
    summary = {
        "version": VERSIONS[args.proposal_kind],
        "proposal_kind": args.proposal_kind,
        "scene_count": len(scenes),
        "all_pair_proposal_count": len(all_rows),
        "novel_accepted_proposal_count": len(rows),
        "positive_proposal_count": int(labels.sum()),
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "learned_candidate_quality_scores_used": False,
            "ground_truth_fields_in_features": False,
            "GVC_usage": "continuous no-GT relation feature only",
        },
        "target_contract": {
            "binary_target": "proposal crosses at least one official IoU50:0.05:0.90 threshold absent from frozen scene candidate pool",
            "class_balance": "component-balanced 50/50 training weights with deterministic natural-prior correction",
            "threshold_selected": False,
            "hyperparameter_scan": False,
        },
        "overall_metrics": {
            "fixed_complementarity_heuristic": baseline_overall,
            f"{args.proposal_kind}_threshold_cross_head": challenger_overall,
            "delta_head_minus_heuristic": delta_overall,
        },
        "fold_details": fold_details,
        "gate_counts": {
            "pr_win_folds": pr_wins,
            "roc_win_folds": roc_wins,
            "brier_win_folds": brier_wins,
            "log_loss_win_folds": log_wins,
            "minimum_validation_positive_count": minimum_validation_positive_count,
        },
        "gates": gates,
        "proposal_materialized": False,
        "ap_computed": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "utility_ledger_summary_sha256": _sha256(ledger_summary_path),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / f"oof_{args.proposal_kind}_predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in prediction_rows
        ))
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if not gates["proposal_materialization_allowed"]:
            (staging / "NOT_SELECTED_DO_NOT_RUN_AP.json").write_text(
                json.dumps({
                    "decision": "NOT_SELECTED_DO_NOT_RUN_AP",
                    "proposal_kind": args.proposal_kind,
                    "reason": (
                        "The pair-proposal OOF diagnosis failed at least one "
                        "preregistered data-coverage or probabilistic-quality gate."
                    ),
                    "failed_gates": sorted(
                        key for key, passed in gates.items()
                        if key != "proposal_materialization_allowed" and not passed
                    ),
                    "ap_evaluation_run": False,
                    "full_model_exported": False,
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--utility-ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--proposal-kind", choices=tuple(VERSIONS), default="pair_union"
    )
    args = parser.parse_args()
    for name in ("scene_list", "split_manifest", "utility_ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "overall_metrics": summary["overall_metrics"],
        "gate_counts": summary["gate_counts"],
        "gates": summary["gates"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

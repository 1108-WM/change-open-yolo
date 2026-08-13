#!/usr/bin/env python3
"""Diagnose a track-only keep/harm head on the retained union features.

The retained component-union schema, unique-winner label, HGB parameters, and
official100 outer folds are unchanged.  The only training change is restricting
the keep classifier to track candidates, which are the only candidates modified
by the deployed cubic suppression policy.  Native candidates are neither fitted
nor scored by the challenger.  This tool exports OOF diagnostics only and never
runs AP or fits an all-scene model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    TRACK_SOURCE,
    read_jsonl,
    read_scene_list,
)
from tools.train_candidate_component_action_head_oof import EXPECTED_SPLIT_SHA256  # noqa: E402
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    _relation_evidence_for_outer_fold,
)
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    MODEL_PARAMS,
    RANDOM_SEED,
    _balanced_winner_weights,
    _component_weights,
    _feature_matrix,
    _load_component_candidates,
    build_candidate_feature_rows,
)
from tools.train_candidate_component_union_list_head_oof import augment_union_features  # noqa: E402
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_union_ap25_protected_head_oof import _load_union_features  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_champion(path: Path) -> dict[tuple[str, str, int], float]:
    lookup = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                str(row["scene_name"]), str(row["candidate_source"]),
                int(row["candidate_id"]),
            )
            if key in lookup:
                raise ValueError(f"duplicate champion candidate: {key}")
            lookup[key] = float(row["keep_probability"])
    if not lookup:
        raise ValueError("champion OOF predictions are empty")
    return lookup


def fit_track_only(
    rows: list[dict], matrix: np.ndarray,
    train_scenes: list[str], validation_scenes: list[str], seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenes = np.asarray([str(row["scene_name"]) for row in rows], dtype=object)
    tracks = np.asarray([
        row["candidate_source"] == TRACK_SOURCE for row in rows
    ], dtype=bool)
    train = np.flatnonzero(tracks & np.isin(scenes, train_scenes))
    validation = np.flatnonzero(tracks & np.isin(scenes, validation_scenes))
    labels = np.asarray([
        int(row["label_component_unique_winner"]) for row in rows
    ], dtype=np.int64)
    if set(np.unique(labels[train])) != {0, 1}:
        raise ValueError("track-only training fold lacks a keep/harm class")
    weights = _balanced_winner_weights(rows, train, labels)
    model = HistGradientBoostingClassifier(
        loss="log_loss", **{**MODEL_PARAMS, "random_state": seed}
    )
    model.fit(matrix[train], labels[train], sample_weight=weights[train])
    scores = model.predict_proba(matrix[validation])[:, 1].astype(np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("track-only keep head emitted invalid probabilities")
    return validation, labels[validation], scores


def track_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("track metrics require both classes")
    return {
        "count": len(labels),
        "keep_count": int(labels.sum()),
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


def _metric_delta(challenger: dict, incumbent: dict) -> dict:
    return {
        name: challenger[name] - incumbent[name]
        for name in (
            "component_balanced_pr_auc", "component_balanced_roc_auc",
            "component_balanced_brier", "component_balanced_log_loss",
            "unweighted_pr_auc", "unweighted_roc_auc",
        )
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("must use the frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    candidates, _ = _load_component_candidates(
        scenes, args.action_ledger_root, candidate_rows
    )
    relation_rows = []
    relation_by_component = defaultdict(list)
    for scene in scenes:
        for row in read_jsonl(
            args.relation_feature_ledger_root / scene / "relation_features.jsonl"
        ):
            relation_rows.append(row)
            relation_by_component[(scene, int(row["relation_component_id"]))].append(row)
    union_lookup, union_names, union_summary = _load_union_features(
        args.component_union_feature_ledger_root, scenes
    )
    champion_lookup = load_champion(args.champion_oof_predictions)
    champion_summary = json.loads(args.champion_summary.read_text())
    champion_feature_names = champion_summary["feature_contract"]["feature_names"]

    track_indexes = np.asarray([
        index for index, row in enumerate(candidates)
        if row["candidate_source"] == TRACK_SOURCE
    ], dtype=np.int64)
    labels = np.asarray([
        int(row["label_component_unique_winner"]) for row in candidates
    ], dtype=np.int64)
    incumbent_scores = np.full(len(candidates), np.nan, dtype=np.float64)
    challenger_scores = np.full(len(candidates), np.nan, dtype=np.float64)
    assignments = np.zeros(len(candidates), dtype=np.int64)
    fold_metrics = []
    feature_names = None
    for index in track_indexes:
        row = candidates[int(index)]
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        if key not in champion_lookup:
            raise ValueError(f"champion prediction missing track: {key}")
        incumbent_scores[index] = champion_lookup[key]

    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked_all, evidence_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index,
            args.candidate_quality_protocol_name,
            relative_target="incumbent_binary",
        )
        stacked_by_component = defaultdict(list)
        for raw, stacked in zip(relation_rows, stacked_all):
            stacked_by_component[
                (str(raw["scene_name"]), int(raw["relation_component_id"]))
            ].append(stacked)
        model_rows = build_candidate_feature_rows(
            candidates, relation_by_component, stacked_by_component
        )
        model_rows = augment_union_features(model_rows, union_lookup, union_names)
        matrix, current_names = _feature_matrix(model_rows)
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise AssertionError("track-only feature schema changed across folds")
        validation, validation_labels, values = fit_track_only(
            model_rows, matrix, fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        challenger_scores[validation] = values
        assignments[validation] += 1
        weights = _component_weights(model_rows, validation)[validation]
        incumbent_metric = track_metrics(
            validation_labels, incumbent_scores[validation], weights
        )
        challenger_metric = track_metrics(validation_labels, values, weights)
        fold_metrics.append({
            "fold_index": fold_index,
            "incumbent_joint_head_on_tracks": incumbent_metric,
            "track_only_head": challenger_metric,
            "delta_track_only_minus_incumbent": _metric_delta(
                challenger_metric, incumbent_metric
            ),
            "relation_evidence": evidence_diagnostics,
        })
    if not np.all(assignments[track_indexes] == 1):
        raise AssertionError("every track candidate must receive one OOF prediction")
    if np.any(assignments[np.setdiff1d(np.arange(len(candidates)), track_indexes)] != 0):
        raise AssertionError("track-only challenger scored native candidates")
    if not np.isfinite(challenger_scores[track_indexes]).all():
        raise AssertionError("track-only OOF scores are incomplete")
    if feature_names != champion_feature_names:
        raise AssertionError("track-only diagnostic does not reproduce champion feature schema")

    weights = _component_weights(candidates, track_indexes)[track_indexes]
    incumbent_overall = track_metrics(
        labels[track_indexes], incumbent_scores[track_indexes], weights
    )
    challenger_overall = track_metrics(
        labels[track_indexes], challenger_scores[track_indexes], weights
    )
    pr_wins = sum(
        fold["delta_track_only_minus_incumbent"]["component_balanced_pr_auc"] >= 0.0
        for fold in fold_metrics
    )
    roc_wins = sum(
        fold["delta_track_only_minus_incumbent"]["component_balanced_roc_auc"] >= 0.0
        for fold in fold_metrics
    )
    brier_wins = sum(
        fold["delta_track_only_minus_incumbent"]["component_balanced_brier"] <= 0.0
        for fold in fold_metrics
    )
    log_loss_wins = sum(
        fold["delta_track_only_minus_incumbent"]["component_balanced_log_loss"] <= 0.0
        for fold in fold_metrics
    )
    gate_checks = {
        "champion_feature_schema_exact": feature_names == champion_feature_names,
        "overall_pr_auc_not_lower": (
            challenger_overall["component_balanced_pr_auc"]
            >= incumbent_overall["component_balanced_pr_auc"]
        ),
        "overall_roc_auc_not_lower": (
            challenger_overall["component_balanced_roc_auc"]
            >= incumbent_overall["component_balanced_roc_auc"]
        ),
        "overall_brier_not_higher": (
            challenger_overall["component_balanced_brier"]
            <= incumbent_overall["component_balanced_brier"]
        ),
        "overall_log_loss_not_higher": (
            challenger_overall["component_balanced_log_loss"]
            <= incumbent_overall["component_balanced_log_loss"]
        ),
        "pr_auc_non_lower_in_at_least_three_folds": pr_wins >= 3,
        "roc_auc_non_lower_in_at_least_three_folds": roc_wins >= 3,
        "brier_non_higher_in_at_least_three_folds": brier_wins >= 3,
        "log_loss_non_higher_in_at_least_three_folds": log_loss_wins >= 3,
    }
    gate_passed = all(gate_checks.values())
    summary = {
        "version": "official100_component_union_track_only_keep_oof_diagnostic_v1",
        "scene_count": 100,
        "track_candidate_count": len(track_indexes),
        "track_keep_count": int(labels[track_indexes].sum()),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "overall_metrics": {
            "incumbent_joint_head_on_tracks": incumbent_overall,
            "track_only_head": challenger_overall,
            "delta_track_only_minus_incumbent": _metric_delta(
                challenger_overall, incumbent_overall
            ),
        },
        "fold_metrics": fold_metrics,
        "selection_gate": {
            "checks": gate_checks,
            "pr_auc_non_lower_fold_count": pr_wins,
            "roc_auc_non_lower_fold_count": roc_wins,
            "brier_non_higher_fold_count": brier_wins,
            "log_loss_non_higher_fold_count": log_loss_wins,
            "passed": gate_passed,
            "ap_eligible": gate_passed,
        },
        "training_contract": {
            "only_change": "fit the unchanged keep classifier on track candidates only",
            "label": "component unique representative binary keep/harm",
            "features_unchanged": True,
            "model_params_unchanged": True,
            "hyperparameter_scan": False,
            "native_candidates_fitted": False,
            "native_candidates_scored": False,
        },
        "safety_contract": {
            "ap_evaluation_run": False,
            "fit_all_model_exported": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
            "candidate_count_modified": False,
            "holdout_dataset_read": False,
        },
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "relation_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
            "action_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "component_union_summary_sha256": _sha256(
                args.component_union_feature_ledger_root / "summary.json"
            ),
            "component_union_version": union_summary["version"],
            "champion_summary_sha256": _sha256(args.champion_summary),
            "champion_oof_predictions_sha256": _sha256(args.champion_oof_predictions),
        },
    }
    oof_rows = []
    for index in track_indexes:
        row = candidates[int(index)]
        oof_rows.append({
            "scene_name": row["scene_name"],
            "relation_component_id": int(row["relation_component_id"]),
            "candidate_source": row["candidate_source"],
            "candidate_id": int(row["candidate_id"]),
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "label_component_unique_winner": int(labels[index]),
            "incumbent_joint_keep_probability": float(incumbent_scores[index]),
            "track_only_keep_probability": float(challenger_scores[index]),
            "ground_truth_usage": "official_train_offline_label_only",
        })
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "oof_predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in oof_rows
        ))
        marker = "AP_ELIGIBLE.json" if gate_passed else "NOT_SELECTED_DO_NOT_RUN_AP.json"
        (staging / marker).write_text(json.dumps({
            "decision": "AP_ELIGIBLE" if gate_passed else "NOT_SELECTED_DO_NOT_RUN_AP",
            "selection_gate": summary["selection_gate"],
            "ap_evaluation_run": False,
            "fit_all_model_exported": False,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--candidate-records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--component-union-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--champion-summary", type=Path, required=True)
    parser.add_argument("--champion-oof-predictions", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_feature_ledger_root", "champion_summary",
        "champion_oof_predictions", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "overall_metrics": summary["overall_metrics"],
        "selection_gate": summary["selection_gate"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

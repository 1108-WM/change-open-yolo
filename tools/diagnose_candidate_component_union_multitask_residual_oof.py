#!/usr/bin/env python3
"""Diagnose one minimal multitask disagreement residual without running AP.

The retained component-union feature schema is left intact.  The challenger
adds exactly one inference feature per candidate:

    mean(multitask candidate-win score) - mean(incumbent candidate-win score)

Both relation channels are strictly rebuilt inside each official100 outer
fold.  The diagnostic compares only the unique-representative winner head and
verifies that its reconstructed incumbent predictions match the retained OOF
ledger.  It does not evaluate AP or export a fit-all model.
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
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
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
from tools.train_candidate_component_union_list_head_oof import (  # noqa: E402
    augment_union_features,
    collapse_multitask_auxiliary_to_candidate_win_residual,
    strip_multitask_auxiliary_features,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_union_ap25_protected_head_oof import _load_union_features  # noqa: E402


RESIDUAL_FEATURE = "relation__multitask_candidate_win_residual__mean"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_champion(path: Path) -> dict[tuple[str, str, int], float]:
    lookup = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                str(row["scene_name"]), str(row["candidate_source"]),
                int(row["candidate_id"]),
            )
            lookup[key] = float(row["keep_probability"])
    if not lookup:
        raise ValueError("retained champion OOF predictions are empty")
    return lookup


def _fit_winner(
    rows: list[dict], matrix: np.ndarray,
    train_scenes: list[str], validation_scenes: list[str], seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    train = np.flatnonzero(np.isin(scenes, train_scenes))
    validation = np.flatnonzero(np.isin(scenes, validation_scenes))
    labels = np.asarray([
        int(row["label_component_unique_winner"]) for row in rows
    ], dtype=np.int64)
    weights = _balanced_winner_weights(rows, train, labels)
    model = HistGradientBoostingClassifier(
        loss="log_loss", **{**MODEL_PARAMS, "random_state": seed}
    )
    model.fit(matrix[train], labels[train], sample_weight=weights[train])
    scores = model.predict_proba(matrix[validation])[:, 1].astype(np.float64)
    return validation, labels[validation], scores


def _metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict:
    return {
        "count": len(labels),
        "winner_count": int(labels.sum()),
        "unweighted_pr_auc": float(average_precision_score(labels, scores)),
        "unweighted_roc_auc": float(roc_auc_score(labels, scores)),
        "component_balanced_pr_auc": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
        "component_balanced_roc_auc": float(
            roc_auc_score(labels, scores, sample_weight=weights)
        ),
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
    champion_lookup = _load_champion(args.champion_oof_predictions)

    incumbent_scores = np.full(len(candidates), np.nan, dtype=np.float64)
    residual_scores = np.full(len(candidates), np.nan, dtype=np.float64)
    labels = np.asarray([
        int(row["label_component_unique_winner"]) for row in candidates
    ], dtype=np.int64)
    assignments = np.zeros(len(candidates), dtype=np.int64)
    candidate_keys = [
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        for row in candidates
    ]
    fold_metrics = []
    incumbent_names = None
    residual_names = None
    residual_values = np.full(len(candidates), np.nan, dtype=np.float64)
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked_all, evidence_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index,
            args.candidate_quality_protocol_name,
            relative_target="incumbent_plus_multitask_auxiliary",
        )
        stacked_by_component = defaultdict(list)
        for raw, stacked in zip(relation_rows, stacked_all):
            stacked_by_component[
                (str(raw["scene_name"]), int(raw["relation_component_id"]))
            ].append(stacked)
        dual_rows = build_candidate_feature_rows(
            candidates, relation_by_component, stacked_by_component
        )
        dual_rows = augment_union_features(dual_rows, union_lookup, union_names)
        incumbent_rows = strip_multitask_auxiliary_features(dual_rows)
        challenger_rows = collapse_multitask_auxiliary_to_candidate_win_residual(dual_rows)
        incumbent_matrix, current_incumbent_names = _feature_matrix(incumbent_rows)
        challenger_matrix, current_residual_names = _feature_matrix(challenger_rows)
        if incumbent_names is None:
            incumbent_names = current_incumbent_names
            residual_names = current_residual_names
        elif (
            incumbent_names != current_incumbent_names
            or residual_names != current_residual_names
        ):
            raise AssertionError("minimal residual feature schema changed across folds")
        incumbent_result = _fit_winner(
            incumbent_rows, incumbent_matrix,
            fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        challenger_result = _fit_winner(
            challenger_rows, challenger_matrix,
            fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        validation, validation_labels, incumbent_values = incumbent_result
        challenger_validation, challenger_labels, challenger_values = challenger_result
        if not np.array_equal(validation, challenger_validation) or not np.array_equal(
            validation_labels, challenger_labels
        ):
            raise AssertionError("incumbent and residual winner populations differ")
        incumbent_scores[validation] = incumbent_values
        residual_scores[validation] = challenger_values
        assignments[validation] += 1
        for index in validation:
            residual_values[index] = float(
                challenger_rows[int(index)]["model_features"][RESIDUAL_FEATURE]
            )
        weights = _component_weights(incumbent_rows, validation)[validation]
        incumbent_metric = _metrics(validation_labels, incumbent_values, weights)
        challenger_metric = _metrics(validation_labels, challenger_values, weights)
        fold_metrics.append({
            "fold_index": fold_index,
            "incumbent": incumbent_metric,
            "minimal_residual": challenger_metric,
            "delta_minimal_residual_minus_incumbent": {
                name: challenger_metric[name] - incumbent_metric[name]
                for name in (
                    "unweighted_pr_auc", "unweighted_roc_auc",
                    "component_balanced_pr_auc", "component_balanced_roc_auc",
                )
            },
            "relation_evidence": evidence_diagnostics,
        })
    if not np.all(assignments == 1) or not np.isfinite(incumbent_scores).all() \
            or not np.isfinite(residual_scores).all() or not np.isfinite(residual_values).all():
        raise AssertionError("minimal residual OOF coverage is incomplete")
    champion_scores = np.asarray([
        champion_lookup[key] for key in candidate_keys
    ], dtype=np.float64)
    champion_max_abs_error = float(np.max(np.abs(champion_scores - incumbent_scores)))
    all_indexes = np.arange(len(candidates), dtype=np.int64)
    weights = _component_weights(candidates, all_indexes)[all_indexes]
    incumbent_overall = _metrics(labels, incumbent_scores, weights)
    challenger_overall = _metrics(labels, residual_scores, weights)
    pr_wins = sum(
        fold["delta_minimal_residual_minus_incumbent"]["component_balanced_pr_auc"] >= 0.0
        for fold in fold_metrics
    )
    roc_wins = sum(
        fold["delta_minimal_residual_minus_incumbent"]["component_balanced_roc_auc"] >= 0.0
        for fold in fold_metrics
    )
    gate_checks = {
        "reconstructed_incumbent_matches_champion": champion_max_abs_error <= 1e-12,
        "overall_component_balanced_pr_auc_not_lower": (
            challenger_overall["component_balanced_pr_auc"]
            >= incumbent_overall["component_balanced_pr_auc"]
        ),
        "overall_component_balanced_roc_auc_not_lower": (
            challenger_overall["component_balanced_roc_auc"]
            >= incumbent_overall["component_balanced_roc_auc"]
        ),
        "component_balanced_pr_auc_non_lower_in_at_least_three_folds": pr_wins >= 3,
        "component_balanced_roc_auc_non_lower_in_at_least_three_folds": roc_wins >= 3,
    }
    gate_passed = all(gate_checks.values())
    summary = {
        "version": "official100_component_union_minimal_multitask_residual_oof_diagnostic_v1",
        "scene_count": 100,
        "candidate_count": len(candidates),
        "incumbent_feature_count": len(incumbent_names),
        "challenger_feature_count": len(residual_names),
        "added_features": [RESIDUAL_FEATURE],
        "residual_distribution": {
            "min": float(residual_values.min()),
            "mean": float(residual_values.mean()),
            "median": float(np.median(residual_values)),
            "max": float(residual_values.max()),
            "nonzero_count": int(np.sum(np.abs(residual_values) > 1e-12)),
        },
        "overall_metrics": {
            "incumbent": incumbent_overall,
            "minimal_residual": challenger_overall,
            "delta_minimal_residual_minus_incumbent": {
                name: challenger_overall[name] - incumbent_overall[name]
                for name in (
                    "unweighted_pr_auc", "unweighted_roc_auc",
                    "component_balanced_pr_auc", "component_balanced_roc_auc",
                )
            },
        },
        "fold_metrics": fold_metrics,
        "champion_reconstruction_max_abs_keep_probability_error": champion_max_abs_error,
        "selection_gate": {
            "checks": gate_checks,
            "component_balanced_pr_auc_non_lower_fold_count": pr_wins,
            "component_balanced_roc_auc_non_lower_fold_count": roc_wins,
            "passed": gate_passed,
            "ap_eligible": gate_passed,
        },
        "training_contract": {
            "incumbent_features_preserved": True,
            "new_feature_count": 1,
            "new_feature": RESIDUAL_FEATURE,
            "hyperparameter_scan": False,
            "winner_model_and_params_unchanged": True,
            "outer_split": "frozen official100 80/20 x5",
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
            "champion_oof_predictions_sha256": _sha256(args.champion_oof_predictions),
        },
    }
    oof_rows = [{
        "scene_name": candidate["scene_name"],
        "relation_component_id": int(candidate["relation_component_id"]),
        "candidate_source": candidate["candidate_source"],
        "candidate_id": int(candidate["candidate_id"]),
        "fold_index": int(scene_to_fold[candidate["scene_name"]]),
        "label_component_unique_winner": int(labels[index]),
        "multitask_candidate_win_residual_mean": float(residual_values[index]),
        "incumbent_keep_probability": float(incumbent_scores[index]),
        "minimal_residual_keep_probability": float(residual_scores[index]),
        "ground_truth_usage": "official_train_offline_label_only",
    } for index, candidate in enumerate(candidates)]
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
        if not gate_passed:
            (staging / "NOT_SELECTED_DO_NOT_RUN_AP.json").write_text(json.dumps({
                "decision": "NOT_SELECTED_DO_NOT_RUN_AP",
                "selection_gate": summary["selection_gate"],
                "reason": "The minimal residual failed at least one preregistered winner-head OOF gate.",
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
    parser.add_argument("--champion-oof-predictions", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_feature_ledger_root", "champion_oof_predictions",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "overall_metrics": summary["overall_metrics"],
        "selection_gate": summary["selection_gate"],
        "champion_reconstruction_max_abs_keep_probability_error": summary[
            "champion_reconstruction_max_abs_keep_probability_error"
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

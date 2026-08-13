#!/usr/bin/env python3
"""OOF diagnosis for a direct candidate-level marginal AP-harm objective.

The fixed component-union inference features are retained.  Only related track
candidates are trained and scored.  The preregistered objective has three
equally weighted channels: whether demoting a track harms AP, the positive AP
harm magnitude, and component-local marginal-utility ranking.  This tool never
runs AP, fits an all-scene model, or reads a holdout split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    TRACK_SOURCE,
    read_jsonl,
    read_scene_list,
)
from tools.build_train_candidate_track_marginal_harm_ledger import EPS  # noqa: E402
from tools.diagnose_candidate_component_union_track_only_keep_oof import (  # noqa: E402
    _metric_delta,
    load_champion,
    track_metrics,
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
from tools.train_candidate_component_union_list_head_oof import (  # noqa: E402
    augment_union_features,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_union_ap25_protected_head_oof import (  # noqa: E402
    _load_union_features,
)


VERSION = "official100_track_marginal_harm_oof_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def joint_harm_probability(
    state_probability: float, log_magnitude: float, centered_rank_score: float,
) -> float:
    """Fixed equal-coefficient combination of the three loss channels."""
    return _sigmoid(
        _logit(state_probability) + max(0.0, float(log_magnitude))
        + float(centered_rank_score)
    )


def load_marginal_labels(
    root: Path, scenes: list[str],
) -> dict[tuple[str, int, int], dict]:
    lookup = {}
    for scene in scenes:
        path = root / scene / "track_marginal_harm_utilities.jsonl"
        for row in read_jsonl(path):
            key = (
                str(row["scene_name"]), int(row["relation_component_id"]),
                int(row["track_id"]),
            )
            if key in lookup:
                raise ValueError(f"duplicate marginal label: {key}")
            delta = float(row["labels"]["delta_official_ap"])
            decision = str(row["labels"]["decision"])
            expected = (
                "suppress_beneficial" if delta > EPS else (
                    "suppress_harmful" if delta < -EPS else "neutral"
                )
            )
            if decision != expected:
                raise ValueError(f"marginal decision and delta differ: {key}")
            lookup[key] = {
                "label_demote_harms_ap": int(delta < -EPS),
                "label_keep_utility": float(-delta),
                "label_positive_harm_magnitude": float(max(0.0, -delta)),
                "label_delta_official_ap": delta,
                "label_decision": decision,
                "evaluator_candidate_present": bool(
                    row["confidence_audit"]["evaluator_candidate_present"]
                ),
            }
    if not lookup:
        raise ValueError("marginal AP-harm ledger is empty")
    return lookup


def attach_marginal_labels(rows: list[dict], lookup: dict) -> list[dict]:
    output = []
    seen = set()
    for row in rows:
        if row["candidate_source"] != TRACK_SOURCE:
            continue
        key = (
            str(row["scene_name"]), int(row["relation_component_id"]),
            int(row["candidate_id"]),
        )
        if key not in lookup:
            raise ValueError(f"track feature row lacks marginal label: {key}")
        seen.add(key)
        output.append({**row, **lookup[key]})
    if seen != set(lookup):
        raise ValueError("marginal labels and track feature rows differ")
    return output


def _fit_pairwise_ranker(
    rows: list[dict], matrix: np.ndarray, train: np.ndarray,
    validation: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict]:
    utility = np.asarray([float(row["label_keep_utility"]) for row in rows])
    by_component: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in train:
        row = rows[int(index)]
        by_component[(row["scene_name"], row["relation_component_id"])].append(int(index))
    pair_specs = []
    for indexes in by_component.values():
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1:]:
                gap = float(utility[left] - utility[right])
                if abs(gap) <= EPS:
                    continue
                pair_specs.append((left, right, gap))
    if not pair_specs:
        raise ValueError("marginal rank loss has no non-tied component pairs")
    gap_scale = float(np.median([abs(gap) for _, _, gap in pair_specs]))
    if gap_scale <= 0.0:
        raise ValueError("marginal rank gap scale is not positive")
    differences, labels, weights = [], [], []
    for left, right, gap in pair_specs:
        direction = 1.0 if gap > 0.0 else -1.0
        difference = direction * (matrix[left] - matrix[right])
        weight = max(0.25, min(5.0, abs(gap) / gap_scale))
        differences.extend((difference, -difference))
        labels.extend((1, 0))
        weights.extend((weight, weight))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("rank", LogisticRegression(
            C=0.5, penalty="l2", solver="lbfgs", max_iter=5000,
            fit_intercept=False, random_state=seed,
        )),
    ])
    model.fit(
        np.asarray(differences), np.asarray(labels),
        rank__sample_weight=np.asarray(weights),
    )
    raw = np.asarray(model.decision_function(matrix[validation]), dtype=np.float64)
    return raw, {
        "non_tied_pair_count": len(pair_specs),
        "pair_count_symmetric": len(labels),
        "training_gap_median": gap_scale,
    }


def fit_fold(
    rows: list[dict], matrix: np.ndarray, train_scenes: list[str],
    validation_scenes: list[str], seed: int,
) -> tuple[np.ndarray, dict, dict]:
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    train = np.flatnonzero(np.isin(scenes, train_scenes))
    validation = np.flatnonzero(np.isin(scenes, validation_scenes))
    labels = np.asarray([int(row["label_demote_harms_ap"]) for row in rows], dtype=np.int64)
    harm = np.asarray([
        float(row["label_positive_harm_magnitude"]) for row in rows
    ], dtype=np.float64)
    if set(np.unique(labels[train])) != {0, 1}:
        raise ValueError("marginal state training fold lacks one class")
    balanced = _balanced_winner_weights(rows, train, labels)
    params = {**MODEL_PARAMS, "random_state": seed}
    state_model = HistGradientBoostingClassifier(loss="log_loss", **params)
    state_model.fit(matrix[train], labels[train], sample_weight=balanced[train])
    state_probability = np.asarray(
        state_model.predict_proba(matrix[validation])[:, 1], dtype=np.float64
    )

    positive_scale = float(np.median(harm[train][labels[train] == 1]))
    if positive_scale <= 0.0:
        raise ValueError("positive marginal harm scale is not positive")
    magnitude_target = np.log1p(harm / positive_scale)
    magnitude_model = HistGradientBoostingRegressor(loss="squared_error", **params)
    magnitude_model.fit(
        matrix[train], magnitude_target[train], sample_weight=balanced[train]
    )
    log_magnitude = np.maximum(
        0.0, np.asarray(magnitude_model.predict(matrix[validation]), dtype=np.float64)
    )

    rank_raw, rank_details = _fit_pairwise_ranker(
        rows, matrix, train, validation, seed + 1000
    )
    centered_rank = np.zeros(len(validation), dtype=np.float64)
    by_component: dict[tuple[str, int], list[int]] = defaultdict(list)
    for local, index in enumerate(validation):
        row = rows[int(index)]
        by_component[(row["scene_name"], row["relation_component_id"])].append(local)
    for locals_ in by_component.values():
        values = rank_raw[locals_]
        centered_rank[locals_] = values - values.mean()
    joint = np.asarray([
        joint_harm_probability(state, magnitude, rank)
        for state, magnitude, rank in zip(
            state_probability, log_magnitude, centered_rank
        )
    ], dtype=np.float64)
    predictions = {
        int(index): {
            "state_probability": float(state_probability[local]),
            "log_magnitude_prediction": float(log_magnitude[local]),
            "pairwise_raw_score": float(rank_raw[local]),
            "pairwise_centered_score": float(centered_rank[local]),
            "joint_harm_probability": float(joint[local]),
        }
        for local, index in enumerate(validation)
    }
    details = {
        "train_count": len(train),
        "validation_count": len(validation),
        "train_positive_count": int(labels[train].sum()),
        "validation_positive_count": int(labels[validation].sum()),
        "positive_harm_median_for_log_scale": positive_scale,
        "pairwise": rank_details,
    }
    return validation, predictions, details


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("must use the frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    marginal_lookup = load_marginal_labels(args.marginal_harm_ledger_root, scenes)
    marginal_summary = json.loads(
        (args.marginal_harm_ledger_root / "summary.json").read_text()
    )
    if marginal_summary["version"] != "official100_track_marginal_harm_ledger_v1":
        raise ValueError("unexpected marginal AP-harm ledger version")

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

    oof_rows = {}
    fold_details = []
    feature_names = None
    canonical_track_keys = None
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
        track_rows = attach_marginal_labels(model_rows, marginal_lookup)
        keys = [
            (row["scene_name"], row["relation_component_id"], row["candidate_id"])
            for row in track_rows
        ]
        if canonical_track_keys is None:
            canonical_track_keys = keys
        elif canonical_track_keys != keys:
            raise AssertionError("track row order changed across outer folds")
        matrix, current_names = _feature_matrix(track_rows)
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise AssertionError("marginal harm feature schema changed across folds")
        validation, predictions, details = fit_fold(
            track_rows, matrix, fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        labels = np.asarray([
            int(row["label_demote_harms_ap"]) for row in track_rows
        ], dtype=np.int64)
        incumbent = np.asarray([
            champion_lookup[(
                row["scene_name"], row["candidate_source"], row["candidate_id"]
            )]
            for row in track_rows
        ], dtype=np.float64)
        state = np.asarray([
            predictions[int(index)]["state_probability"] for index in validation
        ])
        joint = np.asarray([
            predictions[int(index)]["joint_harm_probability"] for index in validation
        ])
        weights = _component_weights(track_rows, validation)[validation]
        incumbent_metrics = track_metrics(labels[validation], incumbent[validation], weights)
        state_metrics = track_metrics(labels[validation], state, weights)
        joint_metrics = track_metrics(labels[validation], joint, weights)
        fold_details.append({
            "fold_index": fold_index,
            "fit": details,
            "relation_evidence": evidence_diagnostics,
            "incumbent_keep_head": incumbent_metrics,
            "marginal_state_head": state_metrics,
            "marginal_joint_score": joint_metrics,
            "delta_state_minus_incumbent": _metric_delta(
                state_metrics, incumbent_metrics
            ),
            "delta_joint_minus_state": _metric_delta(joint_metrics, state_metrics),
        })
        for index in validation:
            row = track_rows[int(index)]
            key = (
                row["scene_name"], row["relation_component_id"], row["candidate_id"]
            )
            if key in oof_rows:
                raise AssertionError(f"duplicate OOF marginal prediction: {key}")
            oof_rows[key] = {
                "scene_name": row["scene_name"],
                "relation_component_id": int(row["relation_component_id"]),
                "track_id": int(row["candidate_id"]),
                "fold_index": fold_index,
                "incumbent_keep_probability": float(incumbent[int(index)]),
                **predictions[int(index)],
                "label_demote_harms_ap": int(row["label_demote_harms_ap"]),
                "label_keep_utility": float(row["label_keep_utility"]),
                "label_delta_official_ap": float(row["label_delta_official_ap"]),
                "label_decision": row["label_decision"],
            }
        print(f"[marginal AP-harm OOF] fold {fold_index} complete", flush=True)

    if len(oof_rows) != len(marginal_lookup):
        raise AssertionError("OOF marginal predictions do not cover all labels")
    ordered = [oof_rows[key] for key in sorted(oof_rows)]
    labels = np.asarray([row["label_demote_harms_ap"] for row in ordered], dtype=np.int64)
    incumbent = np.asarray([row["incumbent_keep_probability"] for row in ordered])
    state = np.asarray([row["state_probability"] for row in ordered])
    joint = np.asarray([row["joint_harm_probability"] for row in ordered])
    counts = Counter((row["scene_name"], row["relation_component_id"]) for row in ordered)
    weights = np.asarray([
        1.0 / counts[(row["scene_name"], row["relation_component_id"])]
        for row in ordered
    ])
    incumbent_overall = track_metrics(labels, incumbent, weights)
    state_overall = track_metrics(labels, state, weights)
    joint_overall = track_metrics(labels, joint, weights)

    state_pr_wins = sum(
        fold["delta_state_minus_incumbent"]["component_balanced_pr_auc"] >= 0.0
        for fold in fold_details
    )
    state_roc_wins = sum(
        fold["delta_state_minus_incumbent"]["component_balanced_roc_auc"] >= 0.0
        for fold in fold_details
    )
    state_brier_wins = sum(
        fold["delta_state_minus_incumbent"]["component_balanced_brier"] <= 0.0
        for fold in fold_details
    )
    state_log_wins = sum(
        fold["delta_state_minus_incumbent"]["component_balanced_log_loss"] <= 0.0
        for fold in fold_details
    )
    joint_pr_wins = sum(
        fold["delta_joint_minus_state"]["component_balanced_pr_auc"] >= 0.0
        for fold in fold_details
    )
    joint_roc_wins = sum(
        fold["delta_joint_minus_state"]["component_balanced_roc_auc"] >= 0.0
        for fold in fold_details
    )
    gates = {
        "state_overall_pr_not_lower": (
            state_overall["component_balanced_pr_auc"]
            >= incumbent_overall["component_balanced_pr_auc"]
        ),
        "state_overall_roc_not_lower": (
            state_overall["component_balanced_roc_auc"]
            >= incumbent_overall["component_balanced_roc_auc"]
        ),
        "state_overall_brier_not_higher": (
            state_overall["component_balanced_brier"]
            <= incumbent_overall["component_balanced_brier"]
        ),
        "state_overall_log_loss_not_higher": (
            state_overall["component_balanced_log_loss"]
            <= incumbent_overall["component_balanced_log_loss"]
        ),
        "state_pr_non_lower_in_at_least_three_folds": state_pr_wins >= 3,
        "state_roc_non_lower_in_at_least_three_folds": state_roc_wins >= 3,
        "state_brier_non_higher_in_at_least_three_folds": state_brier_wins >= 3,
        "state_log_loss_non_higher_in_at_least_three_folds": state_log_wins >= 3,
        "joint_overall_pr_not_lower_than_state": (
            joint_overall["component_balanced_pr_auc"]
            >= state_overall["component_balanced_pr_auc"]
        ),
        "joint_overall_roc_not_lower_than_state": (
            joint_overall["component_balanced_roc_auc"]
            >= state_overall["component_balanced_roc_auc"]
        ),
        "joint_pr_non_lower_in_at_least_three_folds": joint_pr_wins >= 3,
        "joint_roc_non_lower_in_at_least_three_folds": joint_roc_wins >= 3,
    }
    gates["ap_evaluation_allowed"] = all(gates.values())
    summary = {
        "version": VERSION,
        "scene_count": len(scenes),
        "track_count": len(ordered),
        "positive_track_count": int(labels.sum()),
        "loss_contract": {
            "state": "component-balanced class-balanced binary log loss",
            "magnitude": "same fixed weights; squared loss on log1p(positive AP harm / training-fold positive median)",
            "rank": "component-local utility-gap-weighted symmetric pairwise logistic loss",
            "joint_score": "sigmoid(logit(state) + nonnegative log-magnitude + component-centered rank)",
            "channel_coefficients": {"state": 1.0, "magnitude": 1.0, "rank": 1.0},
            "loss_or_score_weight_scan": False,
        },
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "component_union_feature_count": len(union_names),
            "ground_truth_fields_in_features": False,
            "GVC_usage": "continuous feature only; never a hard gate",
        },
        "overall_metrics": {
            "incumbent_keep_head": incumbent_overall,
            "marginal_state_head": state_overall,
            "marginal_joint_score": joint_overall,
            "delta_state_minus_incumbent": _metric_delta(
                state_overall, incumbent_overall
            ),
            "delta_joint_minus_state": _metric_delta(joint_overall, state_overall),
        },
        "fold_details": fold_details,
        "gates": gates,
        "gate_counts": {
            "state_pr_win_folds": state_pr_wins,
            "state_roc_win_folds": state_roc_wins,
            "state_brier_win_folds": state_brier_wins,
            "state_log_loss_win_folds": state_log_wins,
            "joint_pr_win_folds": joint_pr_wins,
            "joint_roc_win_folds": joint_roc_wins,
        },
        "candidate_count_modified": False,
        "candidate_geometry_modified": False,
        "candidate_class_modified": False,
        "candidate_files_modified": False,
        "full_model_fitted": False,
        "ap_evaluated": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official100 scene-isolated OOF labels only",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "marginal_harm_summary_sha256": _sha256(
                args.marginal_harm_ledger_root / "summary.json"
            ),
            "relation_summary_sha256": _sha256(
                args.relation_feature_ledger_root / "summary.json"
            ),
            "action_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "component_union_summary_sha256": _sha256(
                args.component_union_feature_ledger_root / "summary.json"
            ),
            "component_union_version": union_summary["version"],
            "champion_summary_sha256": _sha256(args.champion_summary),
        },
    }
    if feature_names != champion_feature_names:
        raise AssertionError("marginal diagnostic does not reproduce champion feature schema")
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "oof_track_predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered
        ))
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
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
    parser.add_argument("--marginal-harm-ledger-root", type=Path, required=True)
    parser.add_argument("--champion-oof-predictions", type=Path, required=True)
    parser.add_argument("--champion-summary", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_feature_ledger_root", "marginal_harm_ledger_root",
        "champion_oof_predictions", "champion_summary", "output_root",
    ):
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

#!/usr/bin/env python3
"""Train a source-aware component list calibration head on official100 OOF.

Every outer fold rebuilds candidate-quality and relation evidence without the
20 validation scenes.  The head scores native exact-geometry representatives
and filtered D2b tracks jointly inside each relation component.  It never
adds, removes, or changes a candidate mask/class; only in-memory scores are
evaluated on official train GT.

The implemented score contract is

    new_score = sigmoid(logit(original_score) + delta_score)

where ``delta_score`` is recorded as the logit difference between the frozen
coexist score and the OOF calibrated list score.  Three fixed ablations are
reported: harm-only suppression, marginal quality calibration, and the full
listwise correction.  No safety60/even48/test60 input is read.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    read_jsonl,
    read_scene_list,
)
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    _scene_inputs,
    _scene_records,
    _sha256,
    _write_jsonl,
    configure_track_score_context,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import _set_match_scores  # noqa: E402
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_SPLIT_SHA256,
    _metrics_from_records,
)
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    _relation_evidence_for_outer_fold,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)


VERSION = "official100_component_list_calibration_head_oof_v1"
RANDOM_SEED = 20260812
OFFICIAL_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 220,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 12,
    "l2_regularization": 10.0,
    "early_stopping": False,
}
RAW_RELATION_FEATURES = (
    "point_iou",
    "native_inside_track_ratio",
    "track_inside_native_ratio",
    "aabb_iou",
    "centroid_distance_normalized",
    "native_shared_superpoint_fraction",
    "track_shared_superpoint_fraction",
    "mean_rgb_distance",
    "mean_normal_difference",
    "exclusive_boundary_contact_ratio_mean",
    "public_projected_box_iou_mean",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "public_track_minus_native_gvc",
)
CANDIDATE_NUMERIC_FEATURES = (
    "original_source_score",
    "point_count",
    "point_fraction_of_scene",
    "native_exact_geometry_group_size",
    "gvc_excluded_mean",
    "gvc_excluded_max",
    "gvc_excluded_variance",
    "gvc_excluded_selected_view_count",
    "gvc_excluded_matched_view_count",
    "gvc_excluded_zero_support_fraction",
    "support_view_count",
    "superpoint_count",
    "source_frame_count",
    "merge_action_count",
    "mean_consensus_rate",
    "mean_edge_score",
    "mean_node_quality",
)
POLICIES = (
    "frozen_coexist",
    "harm_only",
    "marginal_quality",
    "listwise_joint",
    "anchored_listwise_joint",
    "focal_harm",
    "focal_listwise_joint",
    "focal_quality_joint",
    "track_harm_focal_suppression",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite feature {name}: {value}")
    return result


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


def calibrated_score(original_score: float, target_score: float) -> tuple[float, float]:
    """Return ``(new_score, delta)`` under the required logit residual form."""
    target = _clip_probability(target_score)
    delta = _logit(target) - _logit(original_score)
    reconstructed = _sigmoid(_logit(original_score) + delta)
    if not math.isclose(reconstructed, target, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("logit residual reconstruction failed")
    return reconstructed, delta


def focal_risk_target(original_score: float, keep_probability: float) -> float:
    """Preserve confident candidates, suppress risk with a fixed cubic tail.

    The cubic is a fixed focal-risk shape, not a fitted threshold or scanned
    weight.  It also permits a confident track to calibrate upward toward one.
    """
    base = min(1.0, max(0.0, float(original_score)))
    keep = _clip_probability(keep_probability)
    risk = (1.0 - keep) ** 3
    target = base + (1.0 - base) * keep - base * risk
    return min(1.0 - 1e-6, max(1e-6, target))


def assign_component_labels(rows: list[dict]) -> None:
    """Attach the GT-only marginal representative targets used for training."""
    by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_component[(row["scene_name"], int(row["relation_component_id"]))].append(row)
    for component_rows in by_component.values():
        by_target: dict[int, list[dict]] = defaultdict(list)
        for row in component_rows:
            target_id = row["label_best_gt_instance_id"]
            if target_id is not None and float(row["label_best_gt_iou"]) > 0.25:
                by_target[int(target_id)].append(row)
        winners = set()
        for candidates in by_target.values():
            winner = max(candidates, key=lambda row: (
                float(row["label_best_gt_iou"]),
                float(row["original_source_score"]),
                int(row["candidate_source"] == NATIVE_SOURCE),
                -int(row["candidate_id"]),
            ))
            winners.add((winner["candidate_source"], int(winner["candidate_id"])))
        for row in component_rows:
            key = (row["candidate_source"], int(row["candidate_id"]))
            quality = float(row["label_best_gt_iou"])
            winner = key in winners
            row["label_component_unique_winner"] = int(winner)
            row["label_calibrated_quality"] = quality if winner else 0.0
            row["label_official_relevance"] = (
                float(np.mean([quality > threshold for threshold in OFFICIAL_THRESHOLDS]))
                if winner else 0.0
            )
            row["label_harm_kind"] = (
                "winner" if winner else (
                    "duplicate_valid" if quality > 0.25 and row["label_best_gt_instance_id"] is not None
                    else "invalid_or_low_quality"
                )
            )


def _load_component_candidates(
    scenes: list[str], action_ledger_root: Path, candidate_rows: list[dict],
) -> tuple[list[dict], dict[tuple[str, int], list[tuple[str, int, int]]]]:
    candidate_lookup = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in candidate_rows
    }
    result = []
    component_keys = {}
    seen = set()
    for scene in scenes:
        coexist_rows = [
            row for row in read_jsonl(
                action_ledger_root / scene / "component_action_utilities.jsonl"
            ) if row["action_kind"] == "coexist"
        ]
        for action in sorted(coexist_rows, key=lambda row: int(row["relation_component_id"])):
            component_id = int(action["relation_component_id"])
            keys = [
                (scene, NATIVE_SOURCE, int(candidate_id))
                for candidate_id in action["component_kept_native_representative_candidate_ids"]
            ]
            keys.extend(
                (scene, TRACK_SOURCE, int(candidate_id))
                for candidate_id in action["component_kept_track_ids"]
            )
            if len(keys) < 2:
                raise ValueError(f"{scene}/{component_id}: component list is too small")
            component_keys[(scene, component_id)] = keys
            for key in keys:
                if key in seen:
                    raise ValueError(f"candidate repeated across components: {key}")
                seen.add(key)
                if key not in candidate_lookup:
                    raise ValueError(f"component candidate absent from quality ledger: {key}")
                result.append({
                    **candidate_lookup[key],
                    "relation_component_id": component_id,
                })
    assign_component_labels(result)
    return result, component_keys


def _aggregate(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"invalid aggregation for {prefix}")
    return {
        f"{prefix}__min": float(array.min()),
        f"{prefix}__mean": float(array.mean()),
        f"{prefix}__max": float(array.max()),
    }


def _candidate_relation_pairs(
    candidate: dict, raw_rows: list[dict], stacked_rows: list[dict],
) -> list[tuple[dict, dict]]:
    if len(raw_rows) != len(stacked_rows):
        raise ValueError("raw and stacked relation evidence differ")
    if candidate["candidate_source"] == TRACK_SOURCE:
        selected = [
            (raw, stacked) for raw, stacked in zip(raw_rows, stacked_rows)
            if int(raw["track_id"]) == int(candidate["candidate_id"])
        ]
    else:
        selected = [
            (raw, stacked) for raw, stacked in zip(raw_rows, stacked_rows)
            if int(candidate["candidate_id"]) in {
                int(value) for value in raw["native_member_candidate_ids"]
            }
        ]
    if not selected:
        raise ValueError("component candidate has no relation evidence")
    return selected


def build_candidate_feature_rows(
    candidates: list[dict], relation_by_component: dict,
    stacked_by_component: dict,
) -> list[dict]:
    rows = []
    for candidate in candidates:
        component_key = (str(candidate["scene_name"]), int(candidate["relation_component_id"]))
        raw_rows = relation_by_component[component_key]
        stacked_rows = stacked_by_component[component_key]
        pairs = _candidate_relation_pairs(candidate, raw_rows, stacked_rows)
        source_track = candidate["candidate_source"] == TRACK_SOURCE
        features = {
            "candidate_source_is_track": float(source_track),
            "candidate_relation_degree": float(len(pairs)),
            "component_relation_count": float(len(raw_rows)),
            "component_native_count": float(len({
                raw["native_exact_geometry_group_id"] for raw in raw_rows
            })),
            "component_track_count": float(len({int(raw["track_id"]) for raw in raw_rows})),
        }
        for field in CANDIDATE_NUMERIC_FEATURES:
            value = candidate.get(field)
            missing = value is None
            features[f"candidate__{field}"] = 0.0 if missing else _finite(value, field)
            features[f"candidate__{field}__missing"] = float(missing)
        candidate_q_field = "nested_track_q" if source_track else "nested_native_q_median"
        candidate_v25_field = "nested_track_valid25" if source_track else "nested_native_valid25_median"
        candidate_v50_field = "nested_track_valid50" if source_track else "nested_native_valid50_median"
        other_q_field = "nested_native_q_median" if source_track else "nested_track_q"
        win_field = "track_win_relation_score" if source_track else "baseline_win_relation_score"
        other_win_field = "baseline_win_relation_score" if source_track else "track_win_relation_score"
        perspective = {
            "candidate_nested_q": [stacked[candidate_q_field] for _, stacked in pairs],
            "candidate_nested_valid25": [stacked[candidate_v25_field] for _, stacked in pairs],
            "candidate_nested_valid50": [stacked[candidate_v50_field] for _, stacked in pairs],
            "other_nested_q": [stacked[other_q_field] for _, stacked in pairs],
            "candidate_q_advantage": [
                stacked[candidate_q_field] - stacked[other_q_field] for _, stacked in pairs
            ],
            "same_target_score": [stacked["same_target_score"] for _, stacked in pairs],
            "different_target_score": [stacked["different_target_score"] for _, stacked in pairs],
            "candidate_win_relation_score": [stacked[win_field] for _, stacked in pairs],
            "other_win_relation_score": [stacked[other_win_field] for _, stacked in pairs],
            "coexist_relation_score": [stacked["coexist_relation_score"] for _, stacked in pairs],
        }
        auxiliary_fields = (
            "multitask_track_better_score",
            "multitask_baseline_better_score",
            "multitask_track_win_relation_score",
            "multitask_baseline_win_relation_score",
        )
        auxiliary_available = [
            all(field in stacked for field in auxiliary_fields) for _, stacked in pairs
        ]
        if any(auxiliary_available) and not all(auxiliary_available):
            raise ValueError("multitask auxiliary relation evidence is only partially available")
        if all(auxiliary_available):
            auxiliary_win = (
                "multitask_track_win_relation_score" if source_track
                else "multitask_baseline_win_relation_score"
            )
            auxiliary_other_win = (
                "multitask_baseline_win_relation_score" if source_track
                else "multitask_track_win_relation_score"
            )
            perspective.update({
                "multitask_candidate_win_relation_score": [
                    stacked[auxiliary_win] for _, stacked in pairs
                ],
                "multitask_other_win_relation_score": [
                    stacked[auxiliary_other_win] for _, stacked in pairs
                ],
                "multitask_track_better_score": [
                    stacked["multitask_track_better_score"] for _, stacked in pairs
                ],
                "multitask_baseline_better_score": [
                    stacked["multitask_baseline_better_score"] for _, stacked in pairs
                ],
            })
        for name, values in perspective.items():
            features.update(_aggregate([float(value) for value in values], f"relation__{name}"))
        for field in RAW_RELATION_FEATURES:
            features.update(_aggregate(
                [_finite(raw["features"][field], field) for raw, _ in pairs],
                f"raw_relation__{field}",
            ))
        output_row = {
            "scene_name": str(candidate["scene_name"]),
            "relation_component_id": int(candidate["relation_component_id"]),
            "candidate_source": str(candidate["candidate_source"]),
            "candidate_id": int(candidate["candidate_id"]),
            "model_features": features,
        }
        if "label_component_unique_winner" in candidate:
            output_row.update({
                "label_component_unique_winner": int(candidate["label_component_unique_winner"]),
                "label_calibrated_quality": float(candidate["label_calibrated_quality"]),
                "label_official_relevance": float(candidate["label_official_relevance"]),
                "label_best_gt_iou": float(candidate["label_best_gt_iou"]),
                "label_harm_kind": str(candidate["label_harm_kind"]),
            })
        rows.append(output_row)

    by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_component[(row["scene_name"], row["relation_component_id"])].append(row)
    for component_rows in by_component.values():
        q_values = np.asarray([
            row["model_features"]["relation__candidate_nested_q__mean"]
            for row in component_rows
        ], dtype=np.float64)
        for row, q_value in zip(component_rows, q_values):
            same_source = [
                other["model_features"]["relation__candidate_nested_q__mean"]
                for other in component_rows
                if other["candidate_source"] == row["candidate_source"]
            ]
            order = sorted(
                range(len(component_rows)),
                key=lambda index: (-q_values[index], component_rows[index]["candidate_source"], component_rows[index]["candidate_id"]),
            )
            index = component_rows.index(row)
            features = row["model_features"]
            features["context__candidate_q_minus_component_mean"] = float(q_value - q_values.mean())
            features["context__candidate_q_minus_component_max"] = float(q_value - q_values.max())
            features["context__candidate_q_rank_fraction"] = float(order.index(index) / max(1, len(order) - 1))
            features["context__candidate_q_minus_source_mean"] = float(q_value - np.mean(same_source))
            features["context__candidate_q_minus_source_max"] = float(q_value - np.max(same_source))
            features["context__source_candidate_fraction"] = float(len(same_source) / len(component_rows))
    return rows


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0]["model_features"])
    if any(sorted(row["model_features"]) != names for row in rows):
        raise ValueError("candidate list feature schemas differ")
    forbidden = [name for name in names if "label" in name or "best_gt" in name]
    if forbidden:
        raise AssertionError(f"GT field leaked into model features: {forbidden}")
    matrix = np.asarray([
        [_finite(row["model_features"][name], name) for name in names] for row in rows
    ], dtype=np.float64)
    return matrix, names


def _component_weights(rows: list[dict], indexes: np.ndarray) -> np.ndarray:
    counts = Counter(
        (rows[index]["scene_name"], rows[index]["relation_component_id"])
        for index in indexes
    )
    weights = np.zeros(len(rows), dtype=np.float64)
    for index in indexes:
        key = (rows[index]["scene_name"], rows[index]["relation_component_id"])
        weights[index] = 1.0 / counts[key]
    weights[indexes] /= weights[indexes].mean()
    return weights


def _balanced_winner_weights(rows: list[dict], indexes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = _component_weights(rows, indexes)
    for value in (0, 1):
        selected = indexes[labels[indexes] == value]
        if not len(selected):
            raise ValueError("winner classifier fold lacks one class")
        weights[selected] *= 0.5 * len(indexes) / weights[selected].sum()
    weights[indexes] /= weights[indexes].mean()
    return weights


def _fit_pairwise_ranker(
    rows: list[dict], matrix: np.ndarray, train: np.ndarray, validation: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict]:
    target = np.asarray([float(row["label_calibrated_quality"]) for row in rows])
    by_component: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in train:
        row = rows[int(index)]
        by_component[(row["scene_name"], row["relation_component_id"])].append(int(index))
    differences, labels, weights = [], [], []
    for indexes in by_component.values():
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1:]:
                gap = float(target[left] - target[right])
                if abs(gap) <= 1e-12:
                    continue
                direction = 1.0 if gap > 0.0 else -1.0
                difference = direction * (matrix[left] - matrix[right])
                weight = max(0.25, min(5.0, abs(gap) * 5.0))
                differences.extend((difference, -difference))
                labels.extend((1, 0))
                weights.extend((weight, weight))
    if not differences:
        raise ValueError("pairwise list loss has no non-tied pairs")
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
        "pair_count_symmetric": len(labels),
        "non_tied_pair_count": len(labels) // 2,
    }


def fit_fold(
    rows: list[dict], matrix: np.ndarray,
    train_scenes: list[str], validation_scenes: list[str], seed: int,
) -> tuple[dict[tuple[str, int, int], dict], dict]:
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    train = np.flatnonzero(np.isin(scenes, train_scenes))
    validation = np.flatnonzero(np.isin(scenes, validation_scenes))
    winner = np.asarray([int(row["label_component_unique_winner"]) for row in rows])
    quality = np.asarray([float(row["label_calibrated_quality"]) for row in rows])
    weights = _balanced_winner_weights(rows, train, winner)
    params = {**MODEL_PARAMS, "random_state": seed}
    winner_model = HistGradientBoostingClassifier(loss="log_loss", **params)
    quality_model = HistGradientBoostingRegressor(loss="squared_error", **params)
    winner_model.fit(matrix[train], winner[train], sample_weight=weights[train])
    quality_model.fit(matrix[train], quality[train], sample_weight=_component_weights(rows, train)[train])
    keep_probability = winner_model.predict_proba(matrix[validation])[:, 1]
    quality_prediction = np.clip(quality_model.predict(matrix[validation]), 0.0, 1.0)
    pairwise_raw, pairwise_details = _fit_pairwise_ranker(
        rows, matrix, train, validation, seed + 1000
    )
    centered_pairwise = np.zeros(len(validation), dtype=np.float64)
    validation_by_component: dict[tuple[str, int], list[int]] = defaultdict(list)
    for local, index in enumerate(validation):
        row = rows[int(index)]
        validation_by_component[(row["scene_name"], row["relation_component_id"])].append(local)
    for locals_ in validation_by_component.values():
        values = pairwise_raw[locals_]
        centered_pairwise[locals_] = values - values.mean()
    predictions = {}
    for local, index in enumerate(validation):
        row = rows[int(index)]
        marginal = _clip_probability(keep_probability[local] * quality_prediction[local])
        listwise = _sigmoid(_logit(marginal) + centered_pairwise[local])
        predictions[(row["scene_name"], row["relation_component_id"], row["candidate_id"] if row["candidate_source"] == NATIVE_SOURCE else -row["candidate_id"] - 1)] = {
            "candidate_source": row["candidate_source"],
            "candidate_id": row["candidate_id"],
            "keep_probability": float(keep_probability[local]),
            "quality_prediction": float(quality_prediction[local]),
            "pairwise_raw_score": float(pairwise_raw[local]),
            "pairwise_centered_score": float(centered_pairwise[local]),
            "marginal_quality_score": float(marginal),
            "listwise_joint_score": float(listwise),
            "label_component_unique_winner": int(winner[index]),
            "label_calibrated_quality": float(quality[index]),
            "label_best_gt_iou": float(rows[index]["label_best_gt_iou"]),
            "label_harm_kind": rows[index]["label_harm_kind"],
        }
    validation_labels = winner[validation]
    return predictions, {
        "train_candidate_count": len(train),
        "validation_candidate_count": len(validation),
        "validation_winner_count": int(validation_labels.sum()),
        "winner_roc_auc": float(roc_auc_score(validation_labels, keep_probability)),
        "winner_pr_auc": float(average_precision_score(validation_labels, keep_probability)),
        "pairwise": pairwise_details,
    }


def _prediction_lookup(predictions: dict) -> dict[tuple[str, str, int], dict]:
    result = {}
    for prediction in predictions.values():
        key = (
            str(prediction["scene_name"]),
            str(prediction["candidate_source"]),
            int(prediction["candidate_id"]),
        )
        result[key] = prediction
    return result


def _evaluate_oof_scores(args, scenes: list[str], manifest: dict, predictions: dict) -> tuple[dict, dict, list[dict]]:
    by_candidate = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in predictions.values()
    }
    policy_records = {name: {} for name in POLICIES}
    score_rows = []
    original_load_ids = instance_eval.util_3d.load_ids
    _configure_scannet200_instance_eval()
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            uuid_to_key = {uuid: key for key, uuid in cache["uuid_by_candidate"].items()}
            base_scores = {
                uuid_to_key[row["uuid"]]: float(row["confidence"])
                for row in cache["pred"]["chair"]
            }
            policy_scores = {name: dict(base_scores) for name in POLICIES}
            for candidate_key, original in base_scores.items():
                source = NATIVE_SOURCE if candidate_key[0] == "native" else TRACK_SOURCE
                prediction = by_candidate.get((scene, source, int(candidate_key[1])))
                if prediction is None:
                    continue
                pair_keep = _sigmoid(float(prediction["pairwise_centered_score"]) + _logit(prediction["keep_probability"]))
                targets = {
                    "harm_only": original * float(prediction["keep_probability"]),
                    "marginal_quality": float(prediction["marginal_quality_score"]),
                    "listwise_joint": float(prediction["listwise_joint_score"]),
                    "anchored_listwise_joint": math.sqrt(
                        max(0.0, original) * float(prediction["listwise_joint_score"])
                    ),
                    "focal_harm": focal_risk_target(original, prediction["keep_probability"]),
                    "focal_listwise_joint": focal_risk_target(original, pair_keep),
                    "focal_quality_joint": focal_risk_target(
                        original,
                        math.sqrt(max(1e-6, pair_keep * prediction["quality_prediction"])),
                    ),
                    "track_harm_focal_suppression": (
                        original * (1.0 - (1.0 - float(prediction["keep_probability"])) ** 3)
                        if source == TRACK_SOURCE else original
                    ),
                }
                for policy, target in targets.items():
                    if policy == "track_harm_focal_suppression" and source != TRACK_SOURCE:
                        # Frozen native scores, including exact 1.0 ties, must
                        # bypass logit clipping and remain bit-for-bit intact.
                        continue
                    new_score, delta = calibrated_score(original, target)
                    policy_scores[policy][candidate_key] = new_score
                    score_rows.append({
                        "scene_name": scene,
                        "candidate_source": source,
                        "candidate_id": int(candidate_key[1]),
                        "policy": policy,
                        "original_score": original,
                        "new_score": new_score,
                        "delta_score_logit": delta,
                        "pair_keep_probability": pair_keep,
                        "candidate_retained": True,
                    })
            for policy, scores in policy_scores.items():
                uuid_scores = {
                    cache["uuid_by_candidate"][key]: score for key, score in scores.items()
                }
                _set_match_scores({"scene": {"gt": cache["gt"], "pred": cache["pred"]}}, uuid_scores)
                policy_records[policy][scene] = _scene_records(
                    cache, cache["all_native_representatives"], cache["all_track_ids"]
                )
            print(f"[component list OOF AP] {scene_index}/100 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    overall = {name: _metrics_from_records(records) for name, records in policy_records.items()}
    fold_metrics = []
    for fold in manifest["folds"]:
        validation = set(fold["validation_scenes"])
        fold_metrics.append({
            "fold_index": int(fold["fold_index"]),
            "policy_metrics": {
                name: _metrics_from_records({scene: records[scene] for scene in validation})
                for name, records in policy_records.items()
            },
        })
    return overall, {"folds": fold_metrics}, score_rows


def _triplet(metrics: dict) -> dict[str, float]:
    return {
        "ap": float(metrics["official_ap"]),
        "ap50": float(metrics["threshold_metrics"]["50"]["ap"]),
        "ap25": float(metrics["threshold_metrics"]["25"]["ap"]),
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
    score_context = configure_track_score_context(args, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    candidates, component_keys = _load_component_candidates(
        scenes, args.action_ledger_root, candidate_rows
    )
    relation_rows = []
    relation_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for scene in scenes:
        for row in read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl"):
            relation_rows.append(row)
            relation_by_component[(scene, int(row["relation_component_id"]))].append(row)

    final_predictions = {}
    fold_details = []
    feature_names = None
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked_all, stacked_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows,
            candidate_rows,
            scene_to_fold,
            fold_index,
            args.candidate_quality_protocol_name,
        )
        stacked_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for raw, stacked in zip(relation_rows, stacked_all):
            stacked_by_component[(str(raw["scene_name"]), int(raw["relation_component_id"]))].append(stacked)
        model_rows = build_candidate_feature_rows(
            candidates, relation_by_component, stacked_by_component
        )
        matrix, current_names = _feature_matrix(model_rows)
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise AssertionError("feature schema changed across outer folds")
        predictions, metrics = fit_fold(
            model_rows,
            matrix,
            fold["train_scenes"],
            fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        validation = set(fold["validation_scenes"])
        for key, prediction in predictions.items():
            row_key = (prediction["candidate_source"], int(prediction["candidate_id"]))
            full_key = (key[0], row_key[0], row_key[1])
            if full_key in final_predictions:
                raise AssertionError("OOF candidate prediction repeated")
            final_predictions[full_key] = {
                "scene_name": key[0],
                "relation_component_id": key[1],
                **prediction,
            }
        expected_validation = sum(
            row["scene_name"] in validation for row in candidates
        )
        if len(predictions) != expected_validation:
            raise AssertionError("outer validation candidate coverage differs")
        fold_details.append({
            "fold_index": fold_index,
            "stacked_evidence": stacked_diagnostics,
            "head_metrics": metrics,
        })
        print(f"[component list head] outer fold {fold_index} complete", flush=True)
    if len(final_predictions) != len(candidates):
        raise AssertionError("OOF list predictions do not cover every component candidate")

    overall, fold_ap, score_rows = _evaluate_oof_scores(
        args, scenes, manifest, final_predictions
    )
    baseline = _triplet(overall["frozen_coexist"])
    triplets = {name: _triplet(metrics) for name, metrics in overall.items()}
    deltas = {
        name: {key: value - baseline[key] for key, value in triplet.items()}
        for name, triplet in triplets.items()
    }
    soft_summary = json.loads(args.soft_suppression_summary.read_text())
    summary = {
        "version": VERSION,
        "scene_count": 100,
        "component_count": len(component_keys),
        "controlled_candidate_count": len(candidates),
        "controlled_source_counts": dict(sorted(Counter(
            row["candidate_source"] for row in candidates
        ).items())),
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "candidate_quality_rebuilt_inside_each_outer_fold": True,
            "target_and_relative_relation_scores_rebuilt_inside_each_outer_fold": True,
            "ground_truth_fields_in_features": False,
            "GVC_usage": "continuous candidate and relation evidence only; never a hard gate",
        },
        "loss_contract": {
            "listwise_AP_surrogate": "component-local utility-gap-weighted symmetric pairwise logistic ranking",
            "marginal_utility": "component unique representative binary log loss with component/class balanced weights",
            "harm_safety": "all non-representative candidates are explicit negative states; high-IoU duplicates are not positive quality targets",
            "calibration": "regress unique representative best-GT IoU, zero for duplicate/invalid component candidates",
            "GVC_consistency": "GVC enters as continuous features; no GT-derived or thresholded GVC action",
            "new_score_formula": "sigmoid(logit(original_score) + delta_score)",
            "continuous_weight_scan": False,
        },
        "model_contract": {
            "winner_and_quality_models": "HistGradientBoosting",
            "params": MODEL_PARAMS,
            "pairwise_model": "standardized L2 logistic regression without intercept",
            "outer_split": "frozen official100 80/20 x5",
            "fit_all_model_exported": False,
        },
        "fold_details": fold_details,
        "policy_metrics": triplets,
        "policy_delta_vs_frozen_coexist": deltas,
        "fold_policy_metrics": fold_ap,
        "frozen_soft_suppression_reference": {
            "metrics": {
                "ap": float(soft_summary["soft_global_record_metrics"]["official_ap"]),
                "ap50": float(soft_summary["soft_global_record_metrics"]["threshold_metrics"]["50"]["ap"]),
                "ap25": float(soft_summary["soft_global_record_metrics"]["threshold_metrics"]["25"]["ap"]),
            },
            "delta_vs_coexist": soft_summary["delta_vs_coexist"],
        },
        "score_context": score_context,
        "candidate_count_modified": False,
        "candidate_geometry_modified": False,
        "candidate_class_modified": False,
        "candidate_files_modified": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official train nested OOF supervision and offline AP only",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "action_ledger_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "relation_feature_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
            "soft_suppression_summary_sha256": _sha256(args.soft_suppression_summary),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "oof_candidate_predictions.jsonl", [
            final_predictions[key] for key in sorted(final_predictions)
        ])
        _write_jsonl(staging / "oof_candidate_scores.jsonl", score_rows)
        (staging / "feature_schema.json").write_text(
            json.dumps(summary["feature_contract"], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
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
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--candidate-records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--soft-suppression-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--track-score-mode", choices=("oof_quality",), default="oof_quality")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root", "gt_dir",
        "oof_predictions", "soft_suppression_summary", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "policy_metrics": summary["policy_metrics"],
        "policy_delta_vs_frozen_coexist": summary["policy_delta_vs_frozen_coexist"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

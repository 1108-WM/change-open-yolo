#!/usr/bin/env python3
"""严格折内重建候选质量和关系证据，训练 official100 结构化组件选择头。

每个外折中，80 个训练场景的候选质量证据由内部四折预测产生，
20 个验证场景只由这 80 个场景拟合的模型预测。“是否同一目标”和
“同一目标时轨迹候选是否更好”两个关系分数也使用同样的折内场景隔离。

动作头不再用单一的极小 AP 变化回归，而是分解为正收益、无变化和有害
三种状态，再分别学习正收益幅度与伤害幅度。输出仅用于 official train 折外
诊断和内存 AP；不导出全数据模型、不生成推理动作、不修改候选。
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    ACTION_TIE_PRIORITY,
    EPS,
    _sha256,
    _write_jsonl,
    configure_track_score_context,
)
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_ACTION_LEDGER_SCENE_SHA256,
    EXPECTED_SPLIT_SHA256,
    RELATION_FEATURES,
    _action_utility,
    action_feature_row,
    evaluate_policies,
    load_training_rows,
    selection_metrics,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_relative_quality_nested_q_oof import (  # noqa: E402
    nested_candidate_quality_predictions,
    relation_nested_quality_evidence,
)
from tools.train_candidate_relative_quality_oof import (  # noqa: E402
    MODEL_FEATURES as RELATIVE_MODEL_FEATURES,
)
from tools.train_candidate_official_relevance_gap_rank_oof import (  # noqa: E402
    FEATURES as MULTITASK_RELATIVE_FEATURES,
    official_relevance,
)
from tools.train_candidate_relative_quality_multitask_oof import (  # noqa: E402
    _fit_combined,
    strict_indexes,
    strict_labels,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    MODEL_FEATURES as TARGET_MODEL_FEATURES,
    _fit_base,
    _fit_platt,
    scene_track_balanced_weights,
    seeded_scene_folds,
)


RANDOM_SEED = 20260811
UTILITY_SCALE = 1_000_000.0
INNER_FOLD_COUNT = 4
TARGET_FEATURES = TARGET_MODEL_FEATURES["B_overlap_geometry"]
RELATIVE_FEATURES = RELATIVE_MODEL_FEATURES["B_plus_directional_geometry"]
STACKED_RELATION_FIELDS = (
    "nested_track_q",
    "nested_native_q_median",
    "nested_q_delta_track_minus_native",
    "nested_track_valid25",
    "nested_native_valid25_median",
    "nested_valid25_delta_track_minus_native",
    "nested_track_valid50",
    "nested_native_valid50_median",
    "nested_valid50_delta_track_minus_native",
    "same_target_score",
    "different_target_score",
    "track_better_score",
    "baseline_better_score",
    "track_win_relation_score",
    "baseline_win_relation_score",
    "coexist_relation_score",
)
ACTION_MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 220,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 10,
    "l2_regularization": 10.0,
    "early_stopping": False,
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"特征 {name} 不是有限数")
    return result


def _relation_matrix(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([
        [_finite(row["features"][name], name) for name in names] for row in rows
    ], dtype=np.float64)
    if values.shape != (len(rows), len(names)):
        raise AssertionError("关系特征矩阵形状不一致")
    return values


def _crossfit_relation_score(
    rows: list[dict], outer_train_scenes: list[str], outer_validation_scenes: list[str],
    feature_names: tuple[str, ...], task_selector, label_getter, seed: int,
) -> tuple[np.ndarray, dict]:
    """只用可靠任务行拟合，但为每条关系产生无标签可用的分数。"""
    matrix = _relation_matrix(rows, feature_names)
    scenes = np.asarray([str(row["scene_name"]) for row in rows], dtype=object)
    task = np.asarray([bool(task_selector(row)) for row in rows], dtype=bool)
    labels = np.asarray([int(label_getter(row)) if task[index] else 0 for index, row in enumerate(rows)], dtype=np.int64)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    assignments = np.zeros(len(rows), dtype=np.int64)
    inner_details = []
    inner_folds = seeded_scene_folds(
        outer_train_scenes, INNER_FOLD_COUNT, seed
    )

    def fit_and_predict(train_scenes: list[str], prediction_scenes: list[str], model_seed: int) -> tuple[int, int]:
        train = np.flatnonzero(task & np.isin(scenes, train_scenes))
        prediction = np.flatnonzero(np.isin(scenes, prediction_scenes))
        if len(train) == 0 or set(np.unique(labels[train])) != {0, 1}:
            raise ValueError("关系子任务训练折缺少两类样本")
        base_weights = scene_track_balanced_weights(rows, train)[train]
        model = _fit_base(matrix[train], labels[train], base_weights, model_seed)
        values = model.predict_proba(matrix[prediction])[:, 1]
        predictions[prediction] = values
        assignments[prediction] += 1
        return len(train), len(prediction)

    for inner in inner_folds:
        train_count, prediction_count = fit_and_predict(
            inner["train_scenes"], inner["validation_scenes"],
            seed + 10 + int(inner["fold_index"]),
        )
        inner_details.append({
            "inner_fold_index": int(inner["fold_index"]),
            "train_task_relation_count": train_count,
            "prediction_relation_count": prediction_count,
            "train_scene_count": len(inner["train_scenes"]),
            "validation_scene_count": len(inner["validation_scenes"]),
        })
    outer_train_count, outer_validation_count = fit_and_predict(
        outer_train_scenes, outer_validation_scenes, seed + 100,
    )
    if not np.all(assignments == 1) or not np.isfinite(predictions).all():
        raise AssertionError("折内关系分数未完整覆盖 official100")
    validation_task = np.flatnonzero(task & np.isin(scenes, outer_validation_scenes))
    validation_labels = labels[validation_task]
    validation_scores = predictions[validation_task]
    metrics = {
        "validation_task_relation_count": len(validation_task),
        "validation_positive_count": int(validation_labels.sum()),
        "roc_auc": float(roc_auc_score(validation_labels, validation_scores)),
        "pr_auc": float(average_precision_score(validation_labels, validation_scores)),
    }
    return predictions, {
        "feature_names": list(feature_names),
        "inner_folds": inner_details,
        "outer_train_task_relation_count": outer_train_count,
        "outer_validation_relation_count": outer_validation_count,
        "outer_validation_metrics": metrics,
    }


def _crossfit_multitask_relation_score(
    rows: list[dict], outer_train_scenes: list[str], outer_validation_scenes: list[str],
    feature_names: tuple[str, ...], seed: int,
) -> tuple[np.ndarray, dict]:
    """Cross-fit the fixed 50/50 strict-preference plus official-gap head.

    Every fitted relation model receives its own scene-disjoint calibration
    folds.  It is trained only on reliable same-target labels but predicts all
    relations, so downstream component features remain available at inference.
    """
    if tuple(feature_names) != tuple(MULTITASK_RELATIVE_FEATURES):
        raise ValueError("multitask relative head requires frozen B directional geometry")
    matrix = _relation_matrix(rows, feature_names)
    scenes = np.asarray([str(row["scene_name"]) for row in rows], dtype=object)
    same_target = np.asarray([
        bool(row["labels"]["reliable_pair"])
        and row["labels"]["target_state"] == "same_target"
        for row in rows
    ], dtype=bool)
    gaps = np.zeros(len(rows), dtype=np.float64)
    for index in np.flatnonzero(same_target):
        labels = rows[int(index)]["labels"]
        gaps[index] = (
            official_relevance(labels["track_best_gt_iou"])
            - official_relevance(labels["native_best_gt_iou"])
        )
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fit_details = []

    def fit_and_predict(
        train_scenes: list[str], prediction_scenes: list[str], model_seed: int,
    ) -> tuple[int, int, dict]:
        train_task = np.flatnonzero(same_target & np.isin(scenes, train_scenes))
        calibration = strict_indexes(rows, train_task)
        calibration_labels = strict_labels(rows, calibration)
        calibration_decisions = np.full(len(calibration), np.nan, dtype=np.float64)
        calibration_assignments = np.zeros(len(calibration), dtype=np.int64)
        positions = {int(index): position for position, index in enumerate(calibration)}
        calibration_folds = seeded_scene_folds(
            train_scenes, INNER_FOLD_COUNT, model_seed + 700
        )
        calibration_details = []
        for fold in calibration_folds:
            fold_train_scenes = set(fold["train_scenes"])
            fold_validation_scenes = set(fold["validation_scenes"])
            fold_train = np.asarray([
                index for index in train_task
                if rows[index]["scene_name"] in fold_train_scenes
            ], dtype=np.int64)
            fold_validation = np.asarray([
                index for index in calibration
                if rows[index]["scene_name"] in fold_validation_scenes
            ], dtype=np.int64)
            model, diagnostics = _fit_combined(
                rows, matrix, gaps, fold_train,
                model_seed + 10 + int(fold["fold_index"]),
            )
            values = model.decision_function(matrix[fold_validation])
            for index, value in zip(fold_validation, values):
                position = positions[int(index)]
                calibration_decisions[position] = float(value)
                calibration_assignments[position] += 1
            calibration_details.append({
                "fold_index": int(fold["fold_index"]),
                "train_same_target_relation_count": len(fold_train),
                "validation_strict_relation_count": len(fold_validation),
                "fit": diagnostics,
            })
        if not np.all(calibration_assignments == 1) or not np.isfinite(calibration_decisions).all():
            raise AssertionError("multitask relation calibration coverage is incomplete")
        calibration_weights = scene_track_balanced_weights(rows, calibration)[calibration]
        calibrator = _fit_platt(
            calibration_decisions, calibration_labels, calibration_weights
        )
        final_model, final_diagnostics = _fit_combined(
            rows, matrix, gaps, train_task, model_seed + 100
        )
        prediction = np.flatnonzero(np.isin(scenes, prediction_scenes))
        decisions = final_model.decision_function(matrix[prediction])
        values = calibrator.predict_proba(decisions.reshape(-1, 1))[:, 1]
        predictions[prediction] = values
        assignments[prediction] += 1
        return len(train_task), len(prediction), {
            "calibration_folds": calibration_details,
            "calibration_strict_relation_count": len(calibration),
            "platt_intercept": float(calibrator.intercept_[0]),
            "platt_slope": float(calibrator.coef_[0, 0]),
            "final_fit": final_diagnostics,
        }

    inner_folds = seeded_scene_folds(outer_train_scenes, INNER_FOLD_COUNT, seed)
    for inner in inner_folds:
        train_count, prediction_count, diagnostics = fit_and_predict(
            inner["train_scenes"], inner["validation_scenes"],
            seed + 1000 + int(inner["fold_index"]) * 100,
        )
        fit_details.append({
            "role": "outer_train_crossfit",
            "inner_fold_index": int(inner["fold_index"]),
            "train_same_target_relation_count": train_count,
            "prediction_relation_count": prediction_count,
            **diagnostics,
        })
    outer_train_count, outer_validation_count, outer_diagnostics = fit_and_predict(
        outer_train_scenes, outer_validation_scenes, seed + 2000,
    )
    fit_details.append({
        "role": "outer_validation_fit",
        "train_same_target_relation_count": outer_train_count,
        "prediction_relation_count": outer_validation_count,
        **outer_diagnostics,
    })
    if not np.all(assignments == 1) or not np.isfinite(predictions).all():
        raise AssertionError("multitask relation scores do not cover official100 exactly once")
    validation_task = np.flatnonzero(
        same_target & np.isin(scenes, outer_validation_scenes)
        & np.asarray([
            row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native")
            for row in rows
        ], dtype=bool)
    )
    validation_labels = np.asarray([
        int(rows[index]["labels"]["relative_quality_state"] == "prefer_track")
        for index in validation_task
    ], dtype=np.int64)
    validation_scores = predictions[validation_task]
    return predictions, {
        "target": "fixed_50_50_binary_plus_official_relevance_gap",
        "feature_names": list(feature_names),
        "fits": fit_details,
        "outer_validation_metrics": {
            "validation_task_relation_count": len(validation_task),
            "validation_positive_count": int(validation_labels.sum()),
            "roc_auc": float(roc_auc_score(validation_labels, validation_scores)),
            "pr_auc": float(average_precision_score(validation_labels, validation_scores)),
        },
    }


def _aggregate(rows: list[dict], fields: tuple[str, ...], prefix: str) -> dict[str, float]:
    if not rows:
        raise ValueError("结构化关系聚合不能为空")
    result = {}
    for field in fields:
        values = np.asarray([_finite(row[field], field) for row in rows], dtype=np.float64)
        result[f"{prefix}{field}__min"] = float(values.min())
        result[f"{prefix}{field}__mean"] = float(values.mean())
        result[f"{prefix}{field}__max"] = float(values.max())
    return result


def structured_action_feature_row(
    action: dict, relation_rows: list[dict], stacked_rows: list[dict],
) -> dict[str, float]:
    if len(relation_rows) != len(stacked_rows) or not relation_rows:
        raise ValueError("原始关系与折内证据未一一对齐")
    result = action_feature_row(action, relation_rows)
    selected_track_id = action.get("selected_track_id")
    selected = stacked_rows if selected_track_id is None else [
        row for raw, row in zip(relation_rows, stacked_rows)
        if int(raw["track_id"]) == int(selected_track_id)
    ]
    if not selected:
        raise ValueError("只保留一条轨迹候选动作缺少折内证据")
    result.update(_aggregate(stacked_rows, STACKED_RELATION_FIELDS, "stacked_component__"))
    result.update(_aggregate(selected, STACKED_RELATION_FIELDS, "stacked_selected__"))

    by_track: dict[int, list[dict]] = defaultdict(list)
    for raw, stacked in zip(relation_rows, stacked_rows):
        by_track[int(raw["track_id"])].append(stacked)
    track_summaries = {
        track_id: {
            "track_win": float(np.mean([row["track_win_relation_score"] for row in rows])),
            "same_target": float(np.mean([row["same_target_score"] for row in rows])),
            "quality_delta": float(np.mean([
                row["nested_q_delta_track_minus_native"] for row in rows
            ])),
        }
        for track_id, rows in by_track.items()
    }
    if selected_track_id is None:
        result.update({
            "selected_track_win_rank_fraction": 0.0,
            "selected_track_win_minus_best_other": 0.0,
            "selected_same_target_minus_best_other": 0.0,
            "selected_quality_delta_minus_best_other": 0.0,
        })
    else:
        selected_track_id = int(selected_track_id)
        current = track_summaries[selected_track_id]
        others = [value for key, value in track_summaries.items() if key != selected_track_id]
        order = sorted(
            track_summaries, key=lambda key: (-track_summaries[key]["track_win"], key)
        )
        result["selected_track_win_rank_fraction"] = float(
            order.index(selected_track_id) / max(1, len(order) - 1)
        )
        for field, output in (
            ("track_win", "selected_track_win_minus_best_other"),
            ("same_target", "selected_same_target_minus_best_other"),
            ("quality_delta", "selected_quality_delta_minus_best_other"),
        ):
            result[output] = float(
                current[field] - max((value[field] for value in others), default=current[field])
            )
    return result


def _relation_evidence_for_outer_fold(
    relation_rows: list[dict], candidate_rows: list[dict], scene_to_fold: dict[str, int],
    outer_fold: int, protocol_name: str,
    target_features: tuple[str, ...] = TARGET_FEATURES,
    relative_features: tuple[str, ...] = RELATIVE_FEATURES,
    relative_target: str = "incumbent_binary",
) -> tuple[list[dict], dict]:
    outer_train_scenes = sorted(scene for scene, fold in scene_to_fold.items() if fold != outer_fold)
    outer_validation_scenes = sorted(scene for scene, fold in scene_to_fold.items() if fold == outer_fold)
    quality_lookup, quality_diagnostics = nested_candidate_quality_predictions(
        candidate_rows, outer_fold, scene_to_fold, protocol_name,
        targets=("q", "valid25", "valid50"),
    )
    quality_rows = relation_nested_quality_evidence(relation_rows, quality_lookup)
    same_target, target_diagnostics = _crossfit_relation_score(
        relation_rows, outer_train_scenes, outer_validation_scenes,
        target_features,
        lambda row: row["labels"]["target_state"] in ("same_target", "different_target_coexist"),
        lambda row: row["labels"]["target_state"] == "same_target",
        RANDOM_SEED + outer_fold * 1000,
    )
    multitask_auxiliary = None
    multitask_auxiliary_diagnostics = None
    if relative_target in (
        "incumbent_binary", "incumbent_plus_multitask_auxiliary",
    ):
        track_better, relative_diagnostics = _crossfit_relation_score(
            relation_rows, outer_train_scenes, outer_validation_scenes,
            relative_features,
            lambda row: row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native"),
            lambda row: row["labels"]["relative_quality_state"] == "prefer_track",
            RANDOM_SEED + 500 + outer_fold * 1000,
        )
        if relative_target == "incumbent_plus_multitask_auxiliary":
            multitask_auxiliary, multitask_auxiliary_diagnostics = (
                _crossfit_multitask_relation_score(
                    relation_rows, outer_train_scenes, outer_validation_scenes,
                    relative_features, RANDOM_SEED + 750 + outer_fold * 1000,
                )
            )
    elif relative_target == "multitask_binary_plus_official_gap":
        track_better, relative_diagnostics = _crossfit_multitask_relation_score(
            relation_rows, outer_train_scenes, outer_validation_scenes,
            relative_features, RANDOM_SEED + 500 + outer_fold * 1000,
        )
    else:
        raise ValueError(f"unknown relative relation target: {relative_target}")
    stacked = []
    for index, quality in enumerate(quality_rows):
        same = float(same_target[index])
        better = float(track_better[index])
        stacked_row = {
            **quality,
            "same_target_score": same,
            "different_target_score": 1.0 - same,
            "track_better_score": better,
            "baseline_better_score": 1.0 - better,
            "track_win_relation_score": same * better,
            "baseline_win_relation_score": same * (1.0 - better),
            "coexist_relation_score": 1.0 - same,
        }
        if multitask_auxiliary is not None:
            auxiliary = float(multitask_auxiliary[index])
            stacked_row.update({
                "multitask_track_better_score": auxiliary,
                "multitask_baseline_better_score": 1.0 - auxiliary,
                "multitask_track_win_relation_score": same * auxiliary,
                "multitask_baseline_win_relation_score": same * (1.0 - auxiliary),
            })
        stacked.append(stacked_row)
    diagnostics = {
        "candidate_quality": quality_diagnostics,
        "target_consistency": target_diagnostics,
        "relative_quality": relative_diagnostics,
    }
    if multitask_auxiliary_diagnostics is not None:
        diagnostics["multitask_relative_quality_auxiliary"] = (
            multitask_auxiliary_diagnostics
        )
    return stacked, diagnostics


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0]["model_features"])
    if any(sorted(row["model_features"]) != names for row in rows):
        raise ValueError("结构化动作特征字段不一致")
    forbidden = [name for name in names if "label" in name or "best_gt" in name or "iou_margin" in name]
    if forbidden:
        raise AssertionError(f"真实标签字段泄漏进结构化动作头：{forbidden}")
    matrix = np.asarray([
        [_finite(row["model_features"][name], name) for name in names] for row in rows
    ], dtype=np.float64)
    return matrix, names


def _component_weights(rows: list[dict], indexes: np.ndarray) -> np.ndarray:
    counts = Counter(
        (rows[index]["scene_name"], int(rows[index]["relation_component_id"]))
        for index in indexes
    )
    weights = np.zeros(len(rows), dtype=np.float64)
    for index in indexes:
        key = (rows[index]["scene_name"], int(rows[index]["relation_component_id"]))
        weights[index] = 1.0 / counts[key]
    weights[indexes] /= weights[indexes].mean()
    return weights


def class_balanced_state_weights(
    rows: list[dict], indexes: np.ndarray, states: np.ndarray,
    harmful_state_weight: float = 1.0,
) -> np.ndarray:
    if harmful_state_weight < 1.0 or not math.isfinite(harmful_state_weight):
        raise ValueError("harmful state weight must be finite and at least one")
    weights = _component_weights(rows, indexes)
    total = float(weights[indexes].sum())
    classes = sorted(set(int(value) for value in states[indexes]))
    risk = {state: (harmful_state_weight if state == 0 else 1.0) for state in classes}
    risk_total = sum(risk.values())
    for state in classes:
        selected = indexes[states[indexes] == state]
        mass = float(weights[selected].sum())
        weights[selected] *= total * risk[state] / (risk_total * mass)
    weights[indexes] /= weights[indexes].mean()
    return weights


def _safe_auc(labels: np.ndarray, scores: np.ndarray, metric) -> float | None:
    return None if len(np.unique(labels)) < 2 else float(metric(labels, scores))


def _fit_action_kind(
    rows: list[dict], matrix: np.ndarray, train: np.ndarray, validation: np.ndarray,
    seed: int, harmful_state_weight: float = 1.0,
) -> dict[int, dict]:
    utility = np.asarray([float(row["label_utility"]) for row in rows], dtype=np.float64)
    states = np.asarray([
        2 if value > EPS else (0 if value < -EPS else 1) for value in utility
    ], dtype=np.int64)
    weights = class_balanced_state_weights(
        rows, train, states, harmful_state_weight=harmful_state_weight,
    )
    common = {**ACTION_MODEL_PARAMS, "random_state": seed}
    state_model = HistGradientBoostingClassifier(loss="log_loss", **common)
    state_model.fit(matrix[train], states[train], sample_weight=weights[train])
    probabilities = state_model.predict_proba(matrix[validation])
    class_to_column = {int(value): index for index, value in enumerate(state_model.classes_)}

    positive_train = train[states[train] == 2]
    harmful_train = train[states[train] == 0]
    if len(positive_train) < 20 or len(harmful_train) < 20:
        raise ValueError("动作类型缺少足够的正收益或有害样本")
    magnitude_weights = _component_weights(rows, train)
    positive_target = utility[positive_train] * UTILITY_SCALE
    harmful_target = -utility[harmful_train] * UTILITY_SCALE
    gain_mean = HistGradientBoostingRegressor(loss="squared_error", **common)
    gain_lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.25, **common)
    cost_mean = HistGradientBoostingRegressor(loss="squared_error", **common)
    cost_upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.75, **common)
    gain_mean.fit(matrix[positive_train], positive_target, sample_weight=magnitude_weights[positive_train])
    gain_lower.fit(matrix[positive_train], positive_target, sample_weight=magnitude_weights[positive_train])
    cost_mean.fit(matrix[harmful_train], harmful_target, sample_weight=magnitude_weights[harmful_train])
    cost_upper.fit(matrix[harmful_train], harmful_target, sample_weight=magnitude_weights[harmful_train])
    gain_mean_values = np.maximum(0.0, gain_mean.predict(matrix[validation])) / UTILITY_SCALE
    gain_lower_values = np.maximum(0.0, gain_lower.predict(matrix[validation])) / UTILITY_SCALE
    cost_mean_values = np.maximum(0.0, cost_mean.predict(matrix[validation])) / UTILITY_SCALE
    cost_upper_values = np.maximum(0.0, cost_upper.predict(matrix[validation])) / UTILITY_SCALE
    result = {}
    for local, index in enumerate(validation):
        probability = {
            "harmful": float(probabilities[local, class_to_column[0]]),
            "neutral": float(probabilities[local, class_to_column[1]]),
            "positive": float(probabilities[local, class_to_column[2]]),
        }
        mean_utility = probability["positive"] * gain_mean_values[local] - probability["harmful"] * cost_mean_values[local]
        lower_utility = probability["positive"] * gain_lower_values[local] - probability["harmful"] * cost_upper_values[local]
        result[int(index)] = {
            "state_probability": probability,
            "predicted_positive_gain_mean": float(gain_mean_values[local]),
            "predicted_positive_gain_lower25": float(gain_lower_values[local]),
            "predicted_harm_cost_mean": float(cost_mean_values[local]),
            "predicted_harm_cost_upper75": float(cost_upper_values[local]),
            "predicted_mean_utility": float(mean_utility),
            "predicted_lower_utility": float(lower_utility),
            "label_utility": float(utility[index]),
            "label_state": int(states[index]),
        }
    return result


def fit_outer_action_head(
    rows: list[dict], matrix: np.ndarray, train_scenes: list[str], validation_scenes: list[str],
    fold_index: int, harmful_state_weight: float = 1.0,
) -> tuple[dict[tuple[str, int, str], dict], dict]:
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    output = {}
    metrics = {}
    for kind_index, kind in enumerate(("baseline_only", "track_only_one")):
        kind_mask = np.asarray([row["action_kind"] == kind for row in rows], dtype=bool)
        train = np.flatnonzero(kind_mask & np.isin(scenes, train_scenes))
        validation = np.flatnonzero(kind_mask & np.isin(scenes, validation_scenes))
        predictions = _fit_action_kind(
            rows, matrix, train, validation,
            RANDOM_SEED + fold_index * 100 + kind_index,
            harmful_state_weight=harmful_state_weight,
        )
        labels = np.asarray([int(rows[index]["label_positive"]) for index in validation])
        scores = np.asarray([predictions[int(index)]["state_probability"]["positive"] for index in validation])
        utilities = np.asarray([float(rows[index]["label_utility"]) for index in validation])
        predicted_utilities = np.asarray([predictions[int(index)]["predicted_mean_utility"] for index in validation])
        correlation = spearmanr(utilities, predicted_utilities)
        correlation_value = float(getattr(correlation, "statistic", correlation.correlation))
        metrics[kind] = {
            "train_action_count": len(train),
            "validation_action_count": len(validation),
            "validation_positive_count": int(labels.sum()),
            "positive_roc_auc": _safe_auc(labels, scores, roc_auc_score),
            "positive_pr_auc": _safe_auc(labels, scores, average_precision_score),
            "utility_spearman": correlation_value if math.isfinite(correlation_value) else None,
        }
        for index in validation:
            row = rows[int(index)]
            output[(
                str(row["scene_name"]), int(row["relation_component_id"]), str(row["action_name"])
            )] = predictions[int(index)]
    return output, metrics


def choose_structured_action(
    actions: list[dict], predictions: dict, model_rows_by_key: dict, policy: str,
) -> dict:
    coexist = next(row for row in actions if row["action_kind"] == "coexist")
    eligible = []
    for action in actions:
        if action["action_kind"] == "coexist":
            continue
        key = (str(action["scene_name"]), int(action["relation_component_id"]), str(action["action_name"]))
        prediction = predictions[key]
        probability = prediction.get(
            "state_probability", {"positive": 1.0, "harmful": 0.0}
        )
        if policy == "pairwise_action_rank":
            coexist_key = (
                str(action["scene_name"]), int(action["relation_component_id"]), "coexist"
            )
            score = float(predictions[key]["pairwise_score"])
            coexist_score = float(predictions[coexist_key]["pairwise_score"])
            accepted = score > coexist_score
        elif policy == "structured_mean":
            score = float(prediction["predicted_mean_utility"])
        elif policy in (
            "structured_lower", "structured_lower_relation_veto",
            "structured_lower_bidirectional_relation_veto",
        ):
            score = float(prediction["predicted_lower_utility"])
        else:
            raise ValueError(f"未知结构化策略：{policy}")
        if policy != "pairwise_action_rank":
            accepted = score > 0.0 and probability["positive"] > probability["harmful"]
        if accepted and policy in (
            "structured_lower_relation_veto",
            "structured_lower_bidirectional_relation_veto",
        ) and action["action_kind"] == "track_only_one":
            features = model_rows_by_key[key]["model_features"]
            accepted = (
                features["stacked_selected__same_target_score__min"] >= 0.5
                and features["stacked_selected__track_better_score__mean"] >= 0.5
            )
        if (
            accepted
            and policy == "structured_lower_bidirectional_relation_veto"
            and action["action_kind"] == "baseline_only"
        ):
            features = model_rows_by_key[key]["model_features"]
            accepted = (
                features["stacked_component__same_target_score__min"] >= 0.5
                and features["stacked_component__baseline_better_score__mean"] >= 0.5
            )
        if accepted:
            eligible.append((score, probability["positive"], action))
    if not eligible:
        return coexist
    return max(eligible, key=lambda item: (
        item[0], item[1], -ACTION_TIE_PRIORITY[str(item[2]["action_kind"])],
        str(item[2]["action_name"]),
    ))[2]


def _pairwise_action_rank_predictions(
    rows: list[dict], matrix: np.ndarray,
    train_scenes: list[str], validation_scenes: list[str], seed: int,
) -> dict[int, dict]:
    """Fit a component-list pairwise ranker with utility-weighted margins.

    Each component contributes both directions of every non-tied action pair.
    The pair weight is proportional to the observed utility gap, so near-ties
    do not dominate the ranking objective while clearly harmful actions do.
    """
    scenes = np.asarray([str(row["scene_name"]) for row in rows], dtype=object)
    utilities = np.asarray([float(row["label_utility"]) for row in rows], dtype=np.float64)
    train = np.flatnonzero(np.isin(scenes, train_scenes))
    validation = np.flatnonzero(np.isin(scenes, validation_scenes))
    by_component: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in train:
        row = rows[int(index)]
        by_component[(str(row["scene_name"]), int(row["relation_component_id"]))].append(int(index))
    differences, labels, weights = [], [], []
    positive_gaps = []
    for indexes in by_component.values():
        for left_offset, left in enumerate(indexes):
            for right in indexes[left_offset + 1:]:
                gap = float(utilities[left] - utilities[right])
                if abs(gap) <= EPS:
                    continue
                positive_gaps.append(abs(gap))
                direction = 1 if gap > 0.0 else -1
                difference = matrix[left] - matrix[right]
                differences.extend((direction * difference, -direction * difference))
                labels.extend((1, 0))
    if not differences or set(labels) != {0, 1}:
        raise ValueError("pairwise action ranker has insufficient non-tied utility pairs")
    scale = max(float(np.median(positive_gaps)), EPS)
    weights = np.asarray([
        min(5.0, max(0.25, abs(utilities[left] - utilities[right]) / scale))
        for indexes in by_component.values()
        for left_offset, left in enumerate(indexes)
        for right in indexes[left_offset + 1:]
        if abs(float(utilities[left] - utilities[right])) > EPS
        for _direction in (1, 0)
    ], dtype=np.float64)
    pair_matrix = np.asarray(differences, dtype=np.float64)
    pair_labels = np.asarray(labels, dtype=np.int64)
    model = Pipeline([
        ("standardize", StandardScaler()),
        ("rank", LogisticRegression(
            C=0.5, penalty="l2", solver="lbfgs", max_iter=5000,
            random_state=seed,
        )),
    ])
    model.fit(pair_matrix, pair_labels, rank__sample_weight=weights)
    scores = model.decision_function(matrix[validation])
    return {
        int(index): {
            "pairwise_score": float(score),
            "label_utility": float(utilities[index]),
        }
        for index, score in zip(validation, scores)
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("必须使用冻结 official100 场景与五折清单")
    action_summary = json.loads((args.action_ledger_root / "summary.json").read_text())
    if action_summary.get("scene_count") != 100 or action_summary.get("is_smoke_subset"):
        raise ValueError("动作效用账本不是完整 official100")
    if action_summary["input_provenance"]["scene_list_sha256"] != EXPECTED_ACTION_LEDGER_SCENE_SHA256:
        raise ValueError("动作效用账本场景身份不一致")
    score_context = configure_track_score_context(args, scenes)
    if action_summary["score_context"]["track_score_mode"] != args.track_score_mode:
        raise ValueError("动作标签与轨迹分数上下文不一致")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    base_rows, all_actions, base_contract = load_training_rows(
        scenes, args.action_ledger_root, args.relation_feature_ledger_root
    )
    base_rows_by_key = {
        (row["scene_name"], int(row["relation_component_id"]), row["action_name"]): row
        for row in base_rows
    }
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    relation_rows = []
    relation_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for scene in scenes:
        rows = read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl")
        for row in rows:
            relation_rows.append(row)
            relation_by_component[(scene, int(row["relation_component_id"]))].append(row)
    target_feature_set = getattr(args, "target_relation_feature_set", "B_overlap_geometry")
    relative_feature_set = getattr(args, "relative_relation_feature_set", "B_plus_directional_geometry")
    target_features = TARGET_MODEL_FEATURES[target_feature_set]
    relative_features = RELATIVE_MODEL_FEATURES[relative_feature_set]

    final_predictions = {}
    final_model_rows = {}
    final_pairwise_predictions = {}
    final_pairwise_model_rows = {}
    fold_details = []
    feature_names = None
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked_all, stacked_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index,
            args.candidate_quality_protocol_name,
            target_features, relative_features,
        )
        stacked_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for raw, stacked in zip(relation_rows, stacked_all):
            stacked_by_component[(str(raw["scene_name"]), int(raw["relation_component_id"]))].append(stacked)
        model_rows = []
        all_model_rows = []
        for key, base in base_rows_by_key.items():
            action = next(row for row in all_actions[(key[0], key[1])] if row["action_name"] == key[2])
            model_row = {
                **{field: base[field] for field in (
                    "scene_name", "relation_component_id", "action_name", "action_kind",
                    "selected_track_id", "label_utility", "label_positive",
                )},
                "model_features": structured_action_feature_row(
                    action, relation_by_component[(key[0], key[1])],
                    stacked_by_component[(key[0], key[1])],
                ),
            }
            model_rows.append(model_row)
            all_model_rows.append(model_row)
        for (scene_name, component_id), actions in all_actions.items():
            for action in actions:
                if action["action_kind"] == "coexist":
                    key = (scene_name, int(component_id), str(action["action_name"]))
                    all_model_rows.append({
                        "scene_name": scene_name,
                        "relation_component_id": int(component_id),
                        "action_name": str(action["action_name"]),
                        "action_kind": str(action["action_kind"]),
                        "selected_track_id": action.get("selected_track_id"),
                        "label_utility": _action_utility(action),
                        "label_positive": bool(_action_utility(action) > EPS),
                        "model_features": structured_action_feature_row(
                            action, relation_by_component[(scene_name, int(component_id))],
                            stacked_by_component[(scene_name, int(component_id))],
                        ),
                    })
        matrix, current_feature_names = _feature_matrix(model_rows)
        if feature_names is None:
            feature_names = current_feature_names
        elif feature_names != current_feature_names:
            raise AssertionError("外折之间结构化特征模式发生变化")
        fold_predictions, action_metrics = fit_outer_action_head(
            model_rows, matrix, fold["train_scenes"], fold["validation_scenes"], fold_index,
            harmful_state_weight=getattr(args, "harmful_state_weight", 1.0),
        )
        overlap = set(final_predictions) & set(fold_predictions)
        if overlap:
            raise AssertionError("外折动作预测身份重复")
        final_predictions.update(fold_predictions)
        for row in model_rows:
            if row["scene_name"] in set(fold["validation_scenes"]):
                key = (row["scene_name"], int(row["relation_component_id"]), row["action_name"])
                final_model_rows[key] = row
        all_matrix, all_feature_names = _feature_matrix(all_model_rows)
        if all_feature_names != current_feature_names:
            raise AssertionError("coexist and non-coexist action feature schemas differ")
        pairwise_fold_predictions = _pairwise_action_rank_predictions(
            all_model_rows, all_matrix, fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + 9000 + fold_index,
        )
        for index, prediction in pairwise_fold_predictions.items():
            row = all_model_rows[index]
            key = (str(row["scene_name"]), int(row["relation_component_id"]), str(row["action_name"]))
            if key in final_pairwise_predictions:
                raise AssertionError("pairwise OOF action identity repeated")
            final_pairwise_predictions[key] = prediction
            final_pairwise_model_rows[key] = row
        fold_details.append({
            "fold_index": fold_index,
            "stacked_evidence": stacked_diagnostics,
            "action_head_metrics": action_metrics,
        })
        print(f"[结构化选择头] 外折 {fold_index} 完成", flush=True)
    if len(final_predictions) != len(base_rows) or len(final_model_rows) != len(base_rows):
        raise AssertionError("结构化外折预测未完整覆盖非共存动作")
    expected_pairwise_count = sum(len(actions) for actions in all_actions.values())
    if len(final_pairwise_predictions) != expected_pairwise_count:
        raise AssertionError("pairwise OOF predictions do not cover all component actions")

    policies = {
        "always_coexist": {
            key: next(row for row in actions if row["action_kind"] == "coexist")
            for key, actions in all_actions.items()
        }
    }
    for policy in (
        "structured_mean", "structured_lower", "structured_lower_relation_veto",
        "structured_lower_bidirectional_relation_veto",
    ):
        policies[policy] = {
            key: choose_structured_action(actions, final_predictions, final_model_rows, policy)
            for key, actions in all_actions.items()
        }
    policies["pairwise_action_rank"] = {
        key: choose_structured_action(
            actions, final_pairwise_predictions, final_pairwise_model_rows,
            "pairwise_action_rank",
        )
        for key, actions in all_actions.items()
    }
    policy_selection = {
        name: selection_metrics(selected, all_actions) for name, selected in policies.items()
    }
    policy_ap, fold_ap = evaluate_policies(scenes, policies, args, manifest)
    baseline = policy_ap["always_coexist"]["global_metrics"]
    delta = {
        name: {
            "official_ap": result["global_metrics"]["official_ap"] - baseline["official_ap"],
            "ap50": result["global_metrics"]["threshold_metrics"]["50"]["ap"] - baseline["threshold_metrics"]["50"]["ap"],
            "ap25": result["global_metrics"]["threshold_metrics"]["25"]["ap"] - baseline["threshold_metrics"]["25"]["ap"],
        }
        for name, result in policy_ap.items()
    }
    prediction_rows = []
    for key, prediction in sorted(final_predictions.items()):
        prediction_rows.append({
            "scene_name": key[0],
            "relation_component_id": key[1],
            "action_name": key[2],
            "action_kind": final_model_rows[key]["action_kind"],
            "selected_track_id": final_model_rows[key]["selected_track_id"],
            **prediction,
        })
    selected_rows = []
    for policy_name, selected in policies.items():
        for (scene, component_id), action in sorted(selected.items()):
            selected_rows.append({
                "policy": policy_name,
                "scene_name": scene,
                "relation_component_id": component_id,
                "selected_action_name": action["action_name"],
                "selected_action_kind": action["action_kind"],
                "selected_track_id": action.get("selected_track_id"),
                "label_utility_fixed_coexist_single_component": _action_utility(action),
                "inference_action_generated": False,
            })
    summary = {
        "version": "official100_component_action_head_structured_oof_v3",
        "scene_count": 100,
        "component_count": len(all_actions),
        "modeled_noncoexist_action_count": len(base_rows),
        "feature_contract": {
            **base_contract,
            "raw_relation_features": list(RELATION_FEATURES),
            "stacked_relation_fields": list(STACKED_RELATION_FIELDS),
            "feature_names": feature_names,
            "candidate_quality_rebuilt_inside_each_outer_fold": True,
            "target_and_relative_relation_scores_rebuilt_inside_each_outer_fold": True,
            "target_relation_feature_set": target_feature_set,
            "target_relation_feature_names": list(target_features),
            "relative_relation_feature_set": relative_feature_set,
            "relative_relation_feature_names": list(relative_features),
            "ground_truth_fields_in_action_features": False,
            "pairwise_action_feature_names": feature_names,
        },
        "loss_contract": {
            "action_types_fitted_separately": ["只保留基线候选", "只保留一条轨迹候选"],
            "state_head": "正收益/无变化/有害三分类对数损失，按动作类型和状态平衡",
            "harmful_state_weight": getattr(args, "harmful_state_weight", 1.0),
            "asymmetric_risk_loss": (
                "有害状态的目标权重相对中性/正收益状态提高，随后归一化总样本质量"
            ),
            "positive_gain_heads": "正收益条件下的均值与25%分位回归",
            "harm_cost_heads": "有害条件下的均值与75%分位回归",
            "mean_utility": "P(正收益)×收益均值-P(有害)×伤害均值",
            "lower_utility": "P(正收益)×收益25%分位-P(有害)×伤害75%分位",
            "global_ap_non_additive_warning": True,
            "pairwise_action_rank": {
                "loss": "utility-gap-weighted symmetric pairwise logistic ranking",
                "near_tie_excluded": True,
                "coexist_included_as_abstain_action": True,
                "action_selection": "max pairwise score only when above coexist score",
            },
        },
        "model_contract": {
            "model": "HistGradientBoosting",
            "params": ACTION_MODEL_PARAMS,
            "outer_fold": "冻结 official100 80/20 场景划分",
            "fit_all_model_exported": False,
            "threshold_scanning": False,
        },
        "score_context": score_context,
        "fold_details": fold_details,
        "policy_selection_metrics": policy_selection,
        "policy_ap_results": policy_ap,
        "policy_ap_delta_vs_coexist": delta,
        "fold_policy_ap": fold_ap,
        "ground_truth_usage": "official train 折内监督和离线 AP 诊断仅使用",
        "fit_all_model_exported": False,
        "inference_action_generated": False,
        "candidate_files_modified": False,
        "safety60_evaluated": False,
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "action_ledger_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "relation_feature_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
            "candidate_records_root": str(args.candidate_records_root.resolve()),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "oof_action_predictions.jsonl", prediction_rows)
        _write_jsonl(staging / "policy_selected_actions_diagnostic_only.jsonl", selected_rows)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--harmful-state-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-relation-feature-set", choices=tuple(TARGET_MODEL_FEATURES),
        default="B_overlap_geometry",
    )
    parser.add_argument(
        "--relative-relation-feature-set", choices=tuple(RELATIVE_MODEL_FEATURES),
        default="B_plus_directional_geometry",
    )
    parser.add_argument("--track-score-mode", choices=("original", "oof_quality"), default="oof_quality")
    parser.add_argument("--oof-predictions", type=Path)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root", "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.oof_predictions is not None:
        args.oof_predictions = _resolve(args.oof_predictions)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_root}")
    summary = run(args)
    print(json.dumps({
        "policy_selection_metrics": summary["policy_selection_metrics"],
        "policy_ap_delta_vs_coexist": summary["policy_ap_delta_vs_coexist"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

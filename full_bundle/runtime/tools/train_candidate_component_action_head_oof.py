#!/usr/bin/env python3
"""训练并折外评估 official100 组件级候选选择头。

模型只读取关系账本中的无真实标签几何、多视角、外观和原始分数特征。
候选质量学习分数被显式排除，避免其外折训练依赖形成二次泄漏。
每个外折在 80 个训练场景拟合，在冻结的 20 个场景预测；最终以内存方式
同时应用全部折外动作并计算全局类别无关 AP，不写回候选或推理动作。
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
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    ACTION_TIE_PRIORITY,
    EPS,
    _action_candidate_sets,
    _scene_inputs,
    _scene_records,
    _sha256,
    _write_jsonl,
    build_component_actions,
    configure_track_score_context,
    global_metrics,
)
from tools.diagnose_d2b_native_track_ranking_oracle_gt import (  # noqa: E402
    _append,
    _empty_record,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


RANDOM_SEED = 20260810
UTILITY_SCALE = 1_000_000.0
QUANTILE = 0.25
EXPECTED_SPLIT_SHA256 = "710157cb78b66048a181f5e38701a274f24add6416a2f25627d9e499b0668c72"
EXPECTED_ACTION_LEDGER_SCENE_SHA256 = "dfa9017e206190eb2973b247c78e4bf1b2d9c01bb8468a15775c30335e44fb68"

# 只保留无需训练标签即可获得的关系特征。任何候选质量折外预测字段均不得进入。
RELATION_FEATURES = (
    "point_iou",
    "track_inside_native_ratio",
    "native_inside_track_ratio",
    "aabb_iou",
    "track_aabb_coverage",
    "native_aabb_coverage",
    "centroid_distance_normalized",
    "mean_rgb_distance",
    "mean_normal_difference",
    "track_shared_superpoint_fraction",
    "native_shared_superpoint_fraction",
    "exclusive_boundary_contact_ratio_mean",
    "exclusive_boundary_distance_weighted_mean",
    "exclusive_boundary_normal_difference_weighted_mean",
    "exclusive_boundary_color_difference_weighted_mean",
    "public_common_selected_view_count",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "public_projected_box_iou_mean",
    "public_track_minus_native_gvc",
    "track_original_score",
    "native_original_score_median",
    "original_score_delta_track_minus_native",
    "track_point_count",
    "native_point_count",
    "log_track_over_native_point_count",
    "track_superpoint_component_count",
    "native_superpoint_component_count",
    "track_overlapping_native_group_count",
    "native_group_overlapping_track_count",
)
AGGREGATIONS = ("min", "mean", "max")
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 240,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 10.0,
    "early_stopping": False,
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _finite(value, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"特征 {field} 不是有限数")
    return result


def _aggregate_feature_rows(rows: list[dict], prefix: str) -> dict[str, float]:
    if not rows:
        raise ValueError("关系特征聚合集合不能为空")
    result = {}
    for field in RELATION_FEATURES:
        values = np.asarray([
            _finite(row["features"][field], field) for row in rows
        ], dtype=np.float64)
        result[f"{prefix}{field}__min"] = float(values.min())
        result[f"{prefix}{field}__mean"] = float(values.mean())
        result[f"{prefix}{field}__max"] = float(values.max())
    return result


def action_feature_row(action: dict, relation_rows: list[dict]) -> dict:
    """把可变大小关系组件转换为动作级固定长度特征。"""
    if not relation_rows:
        raise ValueError("组件缺少关系特征")
    component_id = int(action["relation_component_id"])
    if any(int(row["relation_component_id"]) != component_id for row in relation_rows):
        raise ValueError("动作与关系特征的组件编号不一致")
    selected_track_id = action.get("selected_track_id")
    if selected_track_id is None:
        selected_rows = relation_rows
    else:
        selected_rows = [
            row for row in relation_rows if int(row["track_id"]) == int(selected_track_id)
        ]
        if not selected_rows:
            raise ValueError("仅一条轨迹候选动作缺少对应关系特征")
    component_track_ids = {int(value) for value in action["component_track_ids"]}
    component_group_ids = {
        str(value) for value in action["component_native_exact_geometry_group_ids"]
    }
    kept_track_ids = {int(value) for value in action["kept_track_ids"]}
    kept_group_ids = {
        str(value) for value in action["kept_native_exact_geometry_group_ids"]
    }
    result = {
        "action_is_baseline_only": float(action["action_kind"] == "baseline_only"),
        "action_is_track_only_one": float(action["action_kind"] == "track_only_one"),
        "component_relation_count": float(len(relation_rows)),
        "component_track_count": float(len(component_track_ids)),
        "component_native_geometry_group_count": float(len(component_group_ids)),
        "action_kept_track_fraction": float(len(kept_track_ids) / max(1, len(component_track_ids))),
        "action_kept_native_group_fraction": float(len(kept_group_ids) / max(1, len(component_group_ids))),
        "action_removed_track_count": float(len(component_track_ids - kept_track_ids)),
        "action_removed_native_group_count": float(len(component_group_ids - kept_group_ids)),
        "selected_track_relation_fraction": float(len(selected_rows) / len(relation_rows)),
    }
    result.update(_aggregate_feature_rows(relation_rows, "component__"))
    result.update(_aggregate_feature_rows(selected_rows, "selected__"))
    return result


def feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise ValueError("动作特征行为空")
    names = sorted(rows[0]["model_features"])
    if any(sorted(row["model_features"]) != names for row in rows):
        raise ValueError("动作特征字段不一致")
    forbidden = [
        name for name in names
        if "_q" in name or "_valid25" in name or "_valid50" in name or "label" in name
    ]
    if forbidden:
        raise AssertionError(f"学习候选质量或真实标签字段泄漏进选择头：{forbidden}")
    matrix = np.asarray([
        [_finite(row["model_features"][name], name) for name in names]
        for row in rows
    ], dtype=np.float64)
    return matrix, names


def component_balanced_weights(rows: list[dict], train_indexes: np.ndarray) -> np.ndarray:
    """每个训练组件总权重相同，再按真实 AP 影响大小温和加权。"""
    counts = Counter(
        (rows[index]["scene_name"], int(rows[index]["relation_component_id"]))
        for index in train_indexes
    )
    absolute = np.asarray([
        abs(float(rows[index]["label_utility"])) for index in train_indexes
    ], dtype=np.float64)
    nonzero = absolute[absolute > EPS]
    reference = float(np.median(nonzero)) if len(nonzero) else 1.0
    result = np.ones(len(rows), dtype=np.float64)
    for index in train_indexes:
        key = (rows[index]["scene_name"], int(rows[index]["relation_component_id"]))
        importance = 1.0 + min(10.0, abs(float(rows[index]["label_utility"])) / max(EPS, reference))
        result[index] = importance / counts[key]
    mean = float(result[train_indexes].mean())
    result[train_indexes] /= max(mean, EPS)
    return result


def _action_utility(row: dict) -> float:
    return float(row["labels"]["delta_vs_fixed_coexist"]["official_ap"])


def load_training_rows(
    scenes: list[str], action_ledger_root: Path, relation_feature_root: Path,
) -> tuple[list[dict], dict[tuple[str, int], list[dict]], dict]:
    model_rows = []
    all_actions: dict[tuple[str, int], list[dict]] = {}
    identities = set()
    for scene in scenes:
        relation_rows = read_jsonl(relation_feature_root / scene / "relation_features.jsonl")
        relations_by_component: dict[int, list[dict]] = defaultdict(list)
        for row in relation_rows:
            relations_by_component[int(row["relation_component_id"])].append(row)
        action_rows = read_jsonl(action_ledger_root / scene / "component_action_utilities.jsonl")
        by_component: dict[int, list[dict]] = defaultdict(list)
        for row in action_rows:
            component_id = int(row["relation_component_id"])
            by_component[component_id].append(row)
        if set(by_component) != set(relations_by_component):
            raise ValueError(f"{scene}: 动作组件与关系特征组件不一致")
        for component_id, actions in sorted(by_component.items()):
            names = {row["action_name"] for row in actions}
            expected = {
                row["action_name"] for row in build_component_actions({
                    "relation_component_id": component_id,
                    "native_exact_geometry_group_ids": actions[0]["component_native_exact_geometry_group_ids"],
                    "track_ids": actions[0]["component_track_ids"],
                })
            }
            if names != expected:
                raise ValueError(f"{scene}/{component_id}: 动作空间不完整")
            all_actions[(scene, component_id)] = sorted(actions, key=lambda row: row["action_name"])
            for action in actions:
                identity = (scene, component_id, str(action["action_name"]))
                if identity in identities:
                    raise ValueError(f"动作身份重复：{identity}")
                identities.add(identity)
                if action["action_kind"] == "coexist":
                    continue
                utility = _action_utility(action)
                model_rows.append({
                    "scene_name": scene,
                    "relation_component_id": component_id,
                    "action_name": str(action["action_name"]),
                    "action_kind": str(action["action_kind"]),
                    "selected_track_id": action.get("selected_track_id"),
                    "label_utility": utility,
                    "label_positive": bool(utility > EPS),
                    "model_features": action_feature_row(
                        action, relations_by_component[component_id]
                    ),
                })
    feature_contract = {
        "relation_features": list(RELATION_FEATURES),
        "aggregations": list(AGGREGATIONS),
        "candidate_quality_learned_fields_excluded": True,
        "ground_truth_fields_in_features": False,
        "feature_count": len(model_rows[0]["model_features"]) if model_rows else 0,
    }
    return model_rows, all_actions, feature_contract


def _safe_binary_metric(function, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, scores))


def _spearman(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if len(labels) < 2 or np.all(labels == labels[0]) or np.all(predictions == predictions[0]):
        return None
    return float(spearmanr(labels, predictions).statistic)


def fit_oof(
    rows: list[dict], matrix: np.ndarray, manifest: dict,
) -> tuple[dict[tuple[str, int, str], dict], list[dict]]:
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    labels_utility = np.asarray([row["label_utility"] for row in rows], dtype=np.float64)
    labels_scaled = labels_utility * UTILITY_SCALE
    labels_positive = np.asarray([row["label_positive"] for row in rows], dtype=np.int64)
    predictions_mean = np.full(len(rows), np.nan, dtype=np.float64)
    predictions_quantile = np.full(len(rows), np.nan, dtype=np.float64)
    predictions_positive = np.full(len(rows), np.nan, dtype=np.float64)
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_rows = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        train = np.flatnonzero(np.isin(scenes, fold["train_scenes"]))
        validation = np.flatnonzero(np.isin(scenes, fold["validation_scenes"]))
        if len(set(scenes[train])) != 80 or len(set(scenes[validation])) != 20:
            raise AssertionError("动作行未遵守冻结 80/20 场景划分")
        assignments[validation] += 1
        weights = component_balanced_weights(rows, train)
        common = {**MODEL_PARAMS, "random_state": RANDOM_SEED + fold_index}
        mean_model = HistGradientBoostingRegressor(loss="squared_error", **common)
        quantile_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=QUANTILE, **common
        )
        positive_model = HistGradientBoostingClassifier(loss="log_loss", **common)
        mean_model.fit(matrix[train], labels_scaled[train], sample_weight=weights[train])
        quantile_model.fit(matrix[train], labels_scaled[train], sample_weight=weights[train])
        positive_model.fit(matrix[train], labels_positive[train], sample_weight=weights[train])
        predictions_mean[validation] = mean_model.predict(matrix[validation]) / UTILITY_SCALE
        predictions_quantile[validation] = quantile_model.predict(matrix[validation]) / UTILITY_SCALE
        predictions_positive[validation] = positive_model.predict_proba(matrix[validation])[:, 1]
        fold_rows.append({
            "fold_index": fold_index,
            "train_scene_count": 80,
            "validation_scene_count": 20,
            "train_action_count": len(train),
            "validation_action_count": len(validation),
            "validation_positive_count": int(labels_positive[validation].sum()),
            "metrics": {
                "positive_roc_auc": _safe_binary_metric(
                    roc_auc_score, labels_positive[validation], predictions_positive[validation]
                ),
                "positive_pr_auc": _safe_binary_metric(
                    average_precision_score, labels_positive[validation], predictions_positive[validation]
                ),
                "utility_spearman_mean": _spearman(
                    labels_utility[validation], predictions_mean[validation]
                ),
                "utility_spearman_quantile25": _spearman(
                    labels_utility[validation], predictions_quantile[validation]
                ),
                "utility_mae_mean": float(np.mean(np.abs(
                    labels_utility[validation] - predictions_mean[validation]
                ))),
            },
        })
    if not np.all(assignments == 1):
        raise AssertionError("每条非共存动作必须恰好得到一次折外预测")
    if any(np.isnan(values).any() for values in (
        predictions_mean, predictions_quantile, predictions_positive,
    )):
        raise AssertionError("折外预测存在缺失")
    result = {}
    for index, row in enumerate(rows):
        result[(
            row["scene_name"], int(row["relation_component_id"]), row["action_name"]
        )] = {
            "predicted_mean_utility": float(predictions_mean[index]),
            "predicted_quantile25_utility": float(predictions_quantile[index]),
            "predicted_positive_probability": float(predictions_positive[index]),
            "label_utility": float(labels_utility[index]),
            "label_positive": bool(labels_positive[index]),
        }
    return result, fold_rows


def choose_action(
    actions: list[dict], predictions: dict[tuple[str, int, str], dict], policy: str,
) -> dict:
    coexist = next(row for row in actions if row["action_kind"] == "coexist")
    candidates = []
    for action in actions:
        if action["action_kind"] == "coexist":
            continue
        key = (
            str(action["scene_name"]), int(action["relation_component_id"]),
            str(action["action_name"]),
        )
        prediction = predictions[key]
        if policy == "mean_positive":
            eligible = prediction["predicted_mean_utility"] > 0.0
            score = prediction["predicted_mean_utility"]
        elif policy == "probability_mean":
            eligible = (
                prediction["predicted_positive_probability"] >= 0.5
                and prediction["predicted_mean_utility"] > 0.0
            )
            score = prediction["predicted_mean_utility"]
        elif policy == "quantile25":
            eligible = (
                prediction["predicted_positive_probability"] >= 0.5
                and prediction["predicted_quantile25_utility"] > 0.0
            )
            score = prediction["predicted_quantile25_utility"]
        else:
            raise ValueError(f"未知策略：{policy}")
        if eligible:
            candidates.append((score, prediction["predicted_positive_probability"], action))
    if not candidates:
        return coexist
    return max(candidates, key=lambda item: (
        item[0], item[1], -ACTION_TIE_PRIORITY[str(item[2]["action_kind"])],
        str(item[2]["action_name"]),
    ))[2]


def build_policy_actions(
    all_actions: dict[tuple[str, int], list[dict]],
    predictions: dict[tuple[str, int, str], dict],
) -> dict[str, dict[tuple[str, int], dict]]:
    policies = {
        "always_coexist": {},
        "always_baseline_only": {},
        "oracle_independent_best": {},
        "mean_positive": {},
        "probability_mean": {},
        "quantile25": {},
    }
    for key, actions in all_actions.items():
        policies["always_coexist"][key] = next(
            row for row in actions if row["action_kind"] == "coexist"
        )
        policies["always_baseline_only"][key] = next(
            row for row in actions if row["action_kind"] == "baseline_only"
        )
        maximum = max(_action_utility(row) for row in actions)
        tied = [row for row in actions if maximum - _action_utility(row) <= EPS]
        policies["oracle_independent_best"][key] = min(
            tied, key=lambda row: (
                ACTION_TIE_PRIORITY[str(row["action_kind"])], str(row["action_name"])
            )
        )
        for policy in ("mean_positive", "probability_mean", "quantile25"):
            policies[policy][key] = choose_action(actions, predictions, policy)
    return policies


def selection_metrics(
    selected: dict[tuple[str, int], dict], all_actions: dict[tuple[str, int], list[dict]],
) -> dict:
    noncoexist = [row for row in selected.values() if row["action_kind"] != "coexist"]
    utilities = np.asarray([_action_utility(row) for row in noncoexist], dtype=np.float64)
    harmed = int(np.count_nonzero(utilities < -EPS))
    positive = int(np.count_nonzero(utilities > EPS))
    regrets = []
    exact_best = 0
    for key, row in selected.items():
        best = max(_action_utility(candidate) for candidate in all_actions[key])
        current = _action_utility(row)
        regrets.append(best - current)
        if best - current <= EPS:
            exact_best += 1
    return {
        "selected_noncoexist_component_count": len(noncoexist),
        "selected_action_kind_counts": dict(sorted(Counter(
            row["action_kind"] for row in selected.values()
        ).items())),
        "selected_positive_count": positive,
        "selected_harmful_count": harmed,
        "selected_positive_precision": float(positive / max(1, len(noncoexist))),
        "selected_true_utility_mean": float(utilities.mean()) if len(utilities) else 0.0,
        "selected_true_utility_sum_nonadditive": float(utilities.sum()) if len(utilities) else 0.0,
        "component_exact_best_or_tied_rate": float(exact_best / max(1, len(selected))),
        "component_regret_mean": float(np.mean(regrets)),
        "component_regret_p90": float(np.quantile(regrets, 0.9)),
    }


def _metrics_from_records(records: dict[str, dict]) -> dict:
    fixed = {tag: _empty_record() for tag in next(iter(records.values()))}
    for scene_records in records.values():
        for tag, values in scene_records.items():
            _append(fixed[tag], values)
    empty = {tag: (np.empty(0), np.empty(0), 0, False, False) for tag in fixed}
    return global_metrics(fixed, empty)


def evaluate_policies(
    scenes: list[str], policies: dict[str, dict[tuple[str, int], dict]], args,
    manifest: dict,
) -> tuple[dict, dict]:
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(
        original_load_ids(path)
    )
    try:
        caches = {scene: _scene_inputs(scene, args) for scene in scenes}
        policy_records = {name: {} for name in policies}
        policy_counts = {name: {} for name in policies}
        for scene_index, scene in enumerate(scenes, start=1):
            cache = caches[scene]
            for policy_name, selections in policies.items():
                kept_native = set(cache["all_native_representatives"])
                kept_tracks = set(cache["all_track_ids"])
                for component in cache["components"]:
                    component_id = int(component["relation_component_id"])
                    action = selections[(scene, component_id)]
                    action_native, action_tracks = _action_candidate_sets(
                        cache, component, action
                    )
                    # 组件互不相交，因此每次结果可直接覆盖当前全局集合。
                    component_native = {
                        int(cache["groups"][str(group_id)]["representative_candidate_id"])
                        for group_id in component["native_exact_geometry_group_ids"]
                    }
                    component_tracks = {int(value) for value in component["track_ids"]}
                    kept_native.difference_update(component_native)
                    kept_native.update(action_native & component_native)
                    kept_tracks.difference_update(component_tracks)
                    kept_tracks.update(action_tracks & component_tracks)
                policy_records[policy_name][scene] = _scene_records(
                    cache, kept_native, kept_tracks
                )
                policy_counts[policy_name][scene] = {
                    "native_representative_count": len(kept_native),
                    "track_count": len(kept_tracks),
                }
            print(f"[折外动作 AP] {scene_index}/{len(scenes)} {scene}", flush=True)
        overall = {
            name: {
                "global_metrics": _metrics_from_records(records),
                "scene_candidate_counts": policy_counts[name],
            }
            for name, records in policy_records.items()
        }
        fold_metrics = []
        for fold in manifest["folds"]:
            validation = set(fold["validation_scenes"])
            fold_metrics.append({
                "fold_index": int(fold["fold_index"]),
                "validation_scene_count": len(validation),
                "policy_global_metrics": {
                    name: _metrics_from_records({
                        scene: records[scene] for scene in validation
                    })
                    for name, records in policy_records.items()
                },
            })
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    return overall, {"folds": fold_metrics}


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("组件选择头固定要求 official100 的 100 个场景")
    if _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("冻结五折清单 SHA-256 不一致")
    action_summary_path = args.action_ledger_root / "summary.json"
    action_summary = json.loads(action_summary_path.read_text())
    if action_summary.get("scene_count") != 100 or action_summary.get("is_smoke_subset"):
        raise ValueError("动作效用账本不是完整 official100")
    if action_summary["input_provenance"]["scene_list_sha256"] != EXPECTED_ACTION_LEDGER_SCENE_SHA256:
        raise ValueError("动作效用账本场景清单不一致")
    score_context = configure_track_score_context(args, scenes)
    if action_summary.get("score_context", {}).get("track_score_mode", "original") != args.track_score_mode:
        raise ValueError("动作效用标签与折外 AP 的轨迹分数上下文不一致")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    model_rows, all_actions, feature_contract = load_training_rows(
        scenes, args.action_ledger_root, args.relation_feature_ledger_root
    )
    matrix, feature_names = feature_matrix(model_rows)
    predictions, fold_model_metrics = fit_oof(model_rows, matrix, manifest)
    policies = build_policy_actions(all_actions, predictions)
    policy_selection_metrics = {
        name: selection_metrics(selected, all_actions)
        for name, selected in policies.items()
    }
    policy_ap, fold_ap = evaluate_policies(scenes, policies, args, manifest)

    oof_rows = []
    for row in model_rows:
        key = (row["scene_name"], int(row["relation_component_id"]), row["action_name"])
        oof_rows.append({
            "scene_name": row["scene_name"],
            "relation_component_id": int(row["relation_component_id"]),
            "action_name": row["action_name"],
            "action_kind": row["action_kind"],
            "selected_track_id": row["selected_track_id"],
            **predictions[key],
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

    baseline = policy_ap["always_coexist"]["global_metrics"]
    summary = {
        "version": "official100_component_action_head_oof_v1",
        "experiment_type": "scene-disjoint five-fold OOF component action head and in-memory class-agnostic AP",
        "scene_count": 100,
        "component_count": len(all_actions),
        "modeled_noncoexist_action_count": len(model_rows),
        "feature_contract": {**feature_contract, "feature_names": feature_names},
        "loss_contract": {
            "positive_head": "binary log loss",
            "mean_utility_head": "squared error on global AP delta scaled by 1e6",
            "lower_utility_head": f"quantile loss at q={QUANTILE}",
            "sample_weight": "equal total mass per training component, multiplied by clipped AP-impact importance computed inside each outer train fold",
            "coexist_score": "fixed zero; coexist rows are not fitted",
        },
        "model_contract": {
            "model": "HistGradientBoosting",
            "params": MODEL_PARAMS,
            "random_seed": RANDOM_SEED,
            "candidate_quality_oof_features_used": False,
            "outer_fold": "frozen official100 80/20 scene split",
            "inner_threshold_selection": False,
        },
        "score_context": score_context,
        "fixed_policy_contracts": {
            "mean_positive": "choose maximum predicted mean utility only when >0",
            "probability_mean": "require positive probability >=0.5 and predicted mean utility >0",
            "quantile25": "require positive probability >=0.5 and predicted 25% quantile utility >0",
        },
        "fold_model_metrics": fold_model_metrics,
        "policy_selection_metrics": policy_selection_metrics,
        "policy_ap_results": policy_ap,
        "policy_ap_delta_vs_coexist": {
            name: {
                "official_ap": result["global_metrics"]["official_ap"] - baseline["official_ap"],
                "ap50": result["global_metrics"]["threshold_metrics"]["50"]["ap"] - baseline["threshold_metrics"]["50"]["ap"],
                "ap25": result["global_metrics"]["threshold_metrics"]["25"]["ap"] - baseline["threshold_metrics"]["25"]["ap"],
            }
            for name, result in policy_ap.items()
        },
        "fold_policy_ap": fold_ap,
        "ground_truth_usage": "official train labels for OOF supervision and offline AP only",
        "selection_head_fit_all_exported": False,
        "inference_action_generated": False,
        "candidate_files_modified": False,
        "safety60_evaluated": False,
        "global_ap_non_additive_warning": True,
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "action_ledger_summary_sha256": _sha256(action_summary_path),
            "relation_feature_summary_sha256": _sha256(
                args.relation_feature_ledger_root / "summary.json"
            ),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "oof_action_predictions.jsonl", oof_rows)
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
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--track-score-mode", choices=("original", "oof_quality"), default="original"
    )
    parser.add_argument("--oof-predictions", type=Path)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "relation_feature_ledger_root",
        "action_ledger_root", "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.oof_predictions is not None:
        args.oof_predictions = _resolve(args.oof_predictions)
    for path in (args.scene_list, args.split_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.records_root, args.relation_feature_ledger_root,
        args.action_ledger_root, args.gt_dir,
    ):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_root}")
    summary = run(args)
    print(json.dumps({
        "fold_model_metrics": summary["fold_model_metrics"],
        "policy_selection_metrics": summary["policy_selection_metrics"],
        "policy_ap_delta_vs_coexist": summary["policy_ap_delta_vs_coexist"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

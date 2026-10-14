"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

from collections import Counter, defaultdict

from pathlib import Path

import numpy as np

from tools.audit_train_candidate_quality_dataset import read_jsonl

from tools.fi1_shared.build_train_candidate_component_action_utility_ledger import EPS, _action_candidate_sets, _scene_inputs, _scene_records, build_component_actions, global_metrics

from tools.fi1_instance_eval_shared import _append, _empty_record

from tools.fi1_instance_eval_shared import _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval

EXPECTED_SPLIT_SHA256 = "710157cb78b66048a181f5e38701a274f24add6416a2f25627d9e499b0668c72"

EXPECTED_ACTION_LEDGER_SCENE_SHA256 = "dfa9017e206190eb2973b247c78e4bf1b2d9c01bb8468a15775c30335e44fb68"

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

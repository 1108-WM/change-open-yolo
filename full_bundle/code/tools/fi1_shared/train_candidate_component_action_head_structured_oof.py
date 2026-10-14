"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

from collections import defaultdict

import numpy as np

from tools.fi1_shared.build_train_candidate_component_action_utility_ledger import ACTION_TIE_PRIORITY

from tools.fi1_shared.train_candidate_component_action_head_oof import action_feature_row

from tools.fi1_shared.train_candidate_relative_quality_oof import MODEL_FEATURES as RELATIVE_MODEL_FEATURES

from tools.fi1_shared.train_candidate_target_consistency_oof import MODEL_FEATURES as TARGET_MODEL_FEATURES

UTILITY_SCALE = 1_000_000.0

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

def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"特征 {name} 不是有限数")
    return result

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

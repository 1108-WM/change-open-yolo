import numpy as np
import pytest

from tools.train_candidate_component_action_head_structured_oof import (
    STACKED_RELATION_FIELDS,
    _crossfit_multitask_relation_score,
    class_balanced_state_weights,
    choose_structured_action,
    structured_action_feature_row,
)
from tools.train_candidate_official_relevance_gap_rank_oof import FEATURES


def _raw(track_id: int):
    from tests.test_train_candidate_component_action_head_oof import _relation

    return _relation(track_id)


def _stacked(same: float, better: float, q_delta: float):
    row = {
        "nested_track_q": 0.8,
        "nested_native_q_median": 0.8 - q_delta,
        "nested_q_delta_track_minus_native": q_delta,
        "nested_track_valid25": 0.9,
        "nested_native_valid25_median": 0.7,
        "nested_valid25_delta_track_minus_native": 0.2,
        "nested_track_valid50": 0.8,
        "nested_native_valid50_median": 0.6,
        "nested_valid50_delta_track_minus_native": 0.2,
        "same_target_score": same,
        "different_target_score": 1.0 - same,
        "track_better_score": better,
        "baseline_better_score": 1.0 - better,
        "track_win_relation_score": same * better,
        "baseline_win_relation_score": same * (1.0 - better),
        "coexist_relation_score": 1.0 - same,
    }
    assert set(row) == set(STACKED_RELATION_FIELDS)
    return row


def _action(kind: str, name: str, selected=None):
    return {
        "scene_name": "scene0001_01",
        "relation_component_id": 0,
        "action_kind": kind,
        "action_name": name,
        "selected_track_id": selected,
        "component_track_ids": [3, 4],
        "component_native_exact_geometry_group_ids": ["g0"],
        "kept_track_ids": [3, 4] if kind == "coexist" else ([selected] if selected is not None else []),
        "kept_native_exact_geometry_group_ids": ["g0"] if kind != "track_only_one" else [],
    }


def test_structured_features_rank_selected_track_and_keep_gt_out():
    action = _action("track_only_one", "track_only_one:3", 3)
    features = structured_action_feature_row(
        action,
        [_raw(3), _raw(4)],
        [_stacked(0.9, 0.8, 0.3), _stacked(0.7, 0.6, 0.1)],
    )
    assert features["selected_track_win_rank_fraction"] == pytest.approx(0.0)
    assert features["selected_track_win_minus_best_other"] > 0
    assert features["stacked_selected__same_target_score__min"] == pytest.approx(0.9)
    assert not any("label" in name or "best_gt" in name for name in features)


def test_state_weights_balance_each_present_class():
    rows = [
        {"scene_name": "a", "relation_component_id": 0},
        {"scene_name": "a", "relation_component_id": 0},
        {"scene_name": "b", "relation_component_id": 0},
        {"scene_name": "c", "relation_component_id": 0},
    ]
    states = np.asarray([0, 0, 1, 2], dtype=np.int64)
    indexes = np.arange(4)
    weights = class_balanced_state_weights(rows, indexes, states)
    masses = [weights[states == state].sum() for state in (0, 1, 2)]
    assert masses[0] == pytest.approx(masses[1])
    assert masses[1] == pytest.approx(masses[2])


def test_state_weights_can_penalize_harmful_actions_asymmetrically():
    rows = [
        {"scene_name": "a", "relation_component_id": 0},
        {"scene_name": "b", "relation_component_id": 0},
        {"scene_name": "c", "relation_component_id": 0},
    ]
    states = np.asarray([0, 1, 2], dtype=np.int64)
    indexes = np.arange(3)
    weights = class_balanced_state_weights(
        rows, indexes, states, harmful_state_weight=2.0,
    )
    assert weights[0] == pytest.approx(2.0 * weights[1])
    assert weights[1] == pytest.approx(weights[2])


def test_relation_veto_rejects_low_same_target_track_action():
    coexist = _action("coexist", "coexist")
    track = _action("track_only_one", "track_only_one:3", 3)
    key = ("scene0001_01", 0, "track_only_one:3")
    predictions = {
        key: {
            "state_probability": {"positive": 0.8, "neutral": 0.1, "harmful": 0.1},
            "predicted_mean_utility": 0.01,
            "predicted_lower_utility": 0.005,
        }
    }
    model_rows = {
        key: {"model_features": {
            "stacked_selected__same_target_score__min": 0.4,
            "stacked_selected__track_better_score__mean": 0.8,
        }}
    }
    assert choose_structured_action(
        [coexist, track], predictions, model_rows, "structured_lower_relation_veto"
    )["action_kind"] == "coexist"
    assert choose_structured_action(
        [coexist, track], predictions, model_rows, "structured_lower"
    )["action_kind"] == "track_only_one"


def test_bidirectional_veto_rejects_unsafe_baseline_only_action():
    coexist = _action("coexist", "coexist")
    baseline = _action("baseline_only", "baseline_only")
    key = ("scene0001_01", 0, "baseline_only")
    predictions = {
        key: {
            "state_probability": {"positive": 0.8, "neutral": 0.1, "harmful": 0.1},
            "predicted_mean_utility": 0.01,
            "predicted_lower_utility": 0.005,
        }
    }
    model_rows = {
        key: {"model_features": {
            "stacked_component__same_target_score__min": 0.4,
            "stacked_component__baseline_better_score__mean": 0.8,
        }}
    }
    assert choose_structured_action(
        [coexist, baseline], predictions, model_rows,
        "structured_lower_bidirectional_relation_veto",
    )["action_kind"] == "coexist"
    assert choose_structured_action(
        [coexist, baseline], predictions, model_rows, "structured_lower",
    )["action_kind"] == "baseline_only"


def test_bidirectional_veto_accepts_supported_baseline_only_action():
    coexist = _action("coexist", "coexist")
    baseline = _action("baseline_only", "baseline_only")
    key = ("scene0001_01", 0, "baseline_only")
    predictions = {
        key: {
            "state_probability": {"positive": 0.8, "neutral": 0.1, "harmful": 0.1},
            "predicted_mean_utility": 0.01,
            "predicted_lower_utility": 0.005,
        }
    }
    model_rows = {
        key: {"model_features": {
            "stacked_component__same_target_score__min": 0.7,
            "stacked_component__baseline_better_score__mean": 0.6,
        }}
    }
    assert choose_structured_action(
        [coexist, baseline], predictions, model_rows,
        "structured_lower_bidirectional_relation_veto",
    )["action_kind"] == "baseline_only"


def test_multitask_relation_crossfit_scores_every_relation_without_gt_features():
    rows = []
    for scene_index in range(20):
        scene = f"scene{scene_index:02d}"
        specifications = (
            ("same_target", "prefer_track", 0.82, 0.61, 1.0),
            ("same_target", "prefer_native", 0.61, 0.82, -1.0),
            ("same_target", "equivalent_abstain", 0.70, 0.69, 0.0),
            ("different_target_coexist", "coexist", 0.75, 0.72, 0.3),
        )
        for relation_index, (target, state, track_iou, native_iou, direction) in enumerate(specifications):
            rows.append({
                "scene_name": scene,
                "track_id": relation_index,
                "features": {
                    name: float(
                        0.01 * scene_index + 0.02 * feature_index
                        + direction * (1.0 if feature_index < 6 else 0.1)
                    )
                    for feature_index, name in enumerate(FEATURES)
                },
                "labels": {
                    "reliable_pair": True,
                    "target_state": target,
                    "relative_quality_state": state,
                    "track_best_gt_iou": track_iou,
                    "native_best_gt_iou": native_iou,
                },
            })
    predictions, diagnostics = _crossfit_multitask_relation_score(
        rows,
        [f"scene{index:02d}" for index in range(16)],
        [f"scene{index:02d}" for index in range(16, 20)],
        FEATURES,
        seed=123,
    )
    assert predictions.shape == (len(rows),)
    assert np.isfinite(predictions).all()
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    assert diagnostics["target"] == "fixed_50_50_binary_plus_official_relevance_gap"

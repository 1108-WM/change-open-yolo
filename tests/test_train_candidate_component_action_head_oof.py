import numpy as np
import pytest

from tools.train_candidate_component_action_head_oof import (
    EPS,
    action_feature_row,
    choose_action,
    component_balanced_weights,
    feature_matrix,
)


def _relation(track_id=3):
    fields = {
        "point_iou": 0.7,
        "track_inside_native_ratio": 0.8,
        "native_inside_track_ratio": 0.75,
        "aabb_iou": 0.6,
        "track_aabb_coverage": 0.8,
        "native_aabb_coverage": 0.7,
        "centroid_distance_normalized": 0.1,
        "mean_rgb_distance": 0.2,
        "mean_normal_difference": 0.3,
        "track_shared_superpoint_fraction": 0.9,
        "native_shared_superpoint_fraction": 0.8,
        "exclusive_boundary_contact_ratio_mean": 0.1,
        "exclusive_boundary_distance_weighted_mean": 0.02,
        "exclusive_boundary_normal_difference_weighted_mean": 0.1,
        "exclusive_boundary_color_difference_weighted_mean": 0.1,
        "public_common_selected_view_count": 3,
        "public_same_matched_observation_fraction": 0.5,
        "public_different_matched_observation_fraction": 0.5,
        "public_projected_box_iou_mean": 0.4,
        "public_track_minus_native_gvc": 0.1,
        "track_original_score": 0.8,
        "native_original_score_median": 0.2,
        "original_score_delta_track_minus_native": 0.6,
        "track_point_count": 1000,
        "native_point_count": 1100,
        "log_track_over_native_point_count": -0.1,
        "track_superpoint_component_count": 1,
        "native_superpoint_component_count": 1,
        "track_overlapping_native_group_count": 1,
        "native_group_overlapping_track_count": 2,
        # 这些字段存在于源账本，但必须不进入模型。
        "delta_D_plus_gvc_q": 0.9,
        "track_D_plus_gvc_valid50": 0.9,
    }
    return {
        "relation_component_id": 0,
        "track_id": track_id,
        "features": fields,
        "labels": {"relative_quality_state": "prefer_track"},
    }


def _action(kind, name, selected=None):
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


def test_action_features_exclude_learned_quality_and_labels():
    action = _action("track_only_one", "track_only_one:3", 3)
    features = action_feature_row(action, [_relation(3), _relation(4)])
    rows = [{"model_features": features}]
    matrix, names = feature_matrix(rows)
    assert matrix.shape == (1, len(names))
    assert not any("_q" in name or "valid50" in name or "label" in name for name in names)
    assert features["selected_track_relation_fraction"] == pytest.approx(0.5)


def test_policy_falls_back_to_coexist_until_fixed_gate_passes():
    coexist = _action("coexist", "coexist")
    baseline = _action("baseline_only", "baseline_only")
    track = _action("track_only_one", "track_only_one:3", 3)
    actions = [coexist, baseline, track]
    predictions = {
        ("scene0001_01", 0, "baseline_only"): {
            "predicted_mean_utility": 0.01,
            "predicted_quantile25_utility": -0.01,
            "predicted_positive_probability": 0.49,
        },
        ("scene0001_01", 0, "track_only_one:3"): {
            "predicted_mean_utility": 0.02,
            "predicted_quantile25_utility": 0.005,
            "predicted_positive_probability": 0.8,
        },
    }
    assert choose_action(actions, predictions, "probability_mean")["action_name"] == "track_only_one:3"
    assert choose_action(actions, predictions, "quantile25")["action_name"] == "track_only_one:3"
    predictions[("scene0001_01", 0, "track_only_one:3")]["predicted_quantile25_utility"] = -EPS
    assert choose_action(actions, predictions, "quantile25")["action_name"] == "coexist"


def test_component_weights_equalize_action_count_before_impact_multiplier():
    rows = [
        {"scene_name": "a", "relation_component_id": 0, "label_utility": 0.0},
        {"scene_name": "a", "relation_component_id": 0, "label_utility": 0.0},
        {"scene_name": "b", "relation_component_id": 0, "label_utility": 0.0},
    ]
    weights = component_balanced_weights(rows, np.arange(3))
    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] + weights[1] == pytest.approx(weights[2])

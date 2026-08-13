import pytest

from tools.build_frozen_structured_soft_suppression_plan import soft_plan_row


def _action(kind="baseline_only"):
    return {
        "policy": "structured_lower_relation_veto",
        "scene_name": "scene0000_00",
        "relation_component_id": 0,
        "component_native_exact_geometry_group_ids": ["g0", "g1"],
        "component_track_ids": [3, 4],
        "selected_action_name": kind,
        "selected_action_kind": kind,
        "selected_track_id": None,
        "kept_native_exact_geometry_group_ids": ["g0", "g1"],
        "kept_track_ids": [],
        "ground_truth_usage": "none",
        "candidate_materialized": False,
        "ap_evaluation_run": False,
    }


def test_soft_plan_retains_candidates_and_uses_frozen_probability_difference():
    row = soft_plan_row(_action(), {
        "state_probability": {"positive": 0.7, "harmful": 0.2, "neutral": 0.1},
    })
    assert row["suppression_strength"] == pytest.approx(0.5)
    assert row["suppression_factor"] == pytest.approx(0.5)
    assert row["suppressed_native_exact_geometry_group_ids"] == []
    assert row["suppressed_track_ids"] == [3, 4]
    assert row["candidate_retained"] is True
    assert row["candidate_count_modified"] is False


def test_coexist_has_no_suppression():
    action = _action("coexist")
    action["kept_track_ids"] = [3, 4]
    row = soft_plan_row(action, None)
    assert row["suppression_strength"] == 0.0
    assert row["suppression_factor"] == 1.0
    assert row["suppressed_native_exact_geometry_group_ids"] == []
    assert row["suppressed_track_ids"] == []

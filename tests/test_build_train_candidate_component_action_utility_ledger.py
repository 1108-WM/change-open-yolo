import numpy as np
import pytest

from tools.build_train_candidate_component_action_utility_ledger import (
    ACTION_TIE_PRIORITY,
    _decision,
    build_component_actions,
    global_metrics,
    relation_label_context,
    select_candidate_sets,
)
from tools.diagnose_d2b_native_track_ranking_oracle_gt import _empty_record


def test_component_actions_follow_frozen_three_action_contract():
    component = {
        "relation_component_id": 4,
        "native_exact_geometry_group_ids": ["g1", "g0"],
        "track_ids": [9, 3],
    }
    rows = build_component_actions(component)
    assert [row["action_name"] for row in rows] == [
        "coexist", "baseline_only", "track_only_one:3", "track_only_one:9",
    ]
    assert rows[0]["kept_native_exact_geometry_group_ids"] == ["g0", "g1"]
    assert rows[0]["kept_track_ids"] == [3, 9]
    assert rows[1]["kept_track_ids"] == []
    assert rows[2]["kept_native_exact_geometry_group_ids"] == []


def test_candidate_selection_changes_only_current_component():
    kept_native, kept_tracks = select_candidate_sets(
        {1, 2, 3, 4}, {10, 11, 12},
        {2, 3}, {10, 11},
        set(), {11},
    )
    assert kept_native == {1, 4}
    assert kept_tracks == {11, 12}
    with pytest.raises(ValueError, match="组件外"):
        select_candidate_sets({1}, {10}, {1}, {10}, {2}, set())


def test_relation_context_separates_full_partial_and_missing_reliability():
    rows = [
        {"labels": {"reliable_pair": True, "target_state": "same_target", "relative_quality_state": "prefer_track"}},
        {"labels": {"reliable_pair": False, "target_state": "unknown", "relative_quality_state": "unknown"}},
    ]
    context = relation_label_context(rows)
    assert context["relation_label_reliability_state"] == "partially_reliable"
    assert context["reliable_relation_fraction"] == pytest.approx(0.5)
    assert context["target_state_counts"] == {"same_target": 1, "unknown": 1}


def test_global_metrics_reports_ap_and_matched_gt_coverage():
    fixed = {str(int(round(value * 100))): _empty_record() for value in (.25, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95)}
    trial = {}
    for tag in fixed:
        trial[tag] = (
            np.asarray([1.0, 0.0]),
            np.asarray([0.9, 0.1]),
            1,
            True,
            True,
        )
    metrics = global_metrics(fixed, trial)
    assert metrics["official_ap"] == pytest.approx(0.5)
    assert metrics["threshold_metrics"]["50"]["true_positive_count"] == 1
    assert metrics["threshold_metrics"]["50"]["matched_gt_coverage"] == pytest.approx(0.5)
    assert _decision(1e-5) == "strict_positive"
    assert _decision(-1e-5) == "harmful"
    assert _decision(0.0) == "neutral"
    assert ACTION_TIE_PRIORITY["coexist"] < ACTION_TIE_PRIORITY["baseline_only"]

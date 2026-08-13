import pytest

from tools.evaluate_official100_structured_soft_suppression_oof_ap import (
    _suppressed_candidate_keys,
    suppression_strength,
)


def test_suppression_strength_uses_positive_minus_harmful_probability():
    assert suppression_strength({
        "state_probability": {"positive": 0.8, "neutral": 0.1, "harmful": 0.1}
    }) == pytest.approx(0.7)
    assert suppression_strength({
        "state_probability": {"positive": 0.1, "neutral": 0.2, "harmful": 0.7}
    }) == 0.0


def test_soft_suppression_keeps_selected_track_and_targets_competitors():
    action = {
        "action_kind": "track_only_one",
        "selected_track_id": 2,
        "component_track_ids": [1, 2, 3],
        "component_native_exact_geometry_group_ids": ["g0", "g1"],
    }
    assert _suppressed_candidate_keys(action) == {
        ("native_group", "g0"), ("native_group", "g1"),
        ("track", 1), ("track", 3),
    }

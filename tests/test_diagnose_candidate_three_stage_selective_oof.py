import numpy as np

from tools.diagnose_candidate_three_stage_selective_oof import (
    clopper_pearson_upper,
    join_predictions,
    learn_then_test_threshold,
)


def _relation():
    return {
        "scene_name": "scene0001_00",
        "track_id": 3,
        "native_exact_geometry_group_id": "group0",
        "labels": {
            "relative_quality_state": "prefer_track",
            "target_state": "same_target",
        },
    }


def test_three_stage_join_uses_exact_probability_product_and_conformal_gate():
    row = _relation()
    key = ("scene0001_00", 3, "group0")
    reliability = {key: {
        "scene_name": key[0], "track_id": key[1],
        "native_exact_geometry_group_id": key[2], "fold_index": 0,
        "predictions": {"R2_plus_relation": 0.9},
    }}
    target = {key: {
        "scene_name": key[0], "track_id": key[1],
        "native_exact_geometry_group_id": key[2], "fold_index": 0,
        "target_same_probability": 0.8,
    }}
    margin = {key: {
        "scene_name": key[0], "track_id": key[1],
        "native_exact_geometry_group_id": key[2], "fold_index": 0,
        "predictions": {
            "probability_margin_gt_005": 0.7,
            "lower_conformal": 0.06,
        },
    }}
    joined = join_predictions(
        [row], reliability, target, margin, {"scene0001_00": 0}
    )[0]
    assert np.isclose(joined["three_stage_product"], 0.9 * 0.8 * 0.7)
    assert joined["passes_conformal_margin"] is True


def test_risk_control_abstains_when_conformal_gate_selects_nothing():
    result = learn_then_test_threshold(
        np.asarray([1, 0]), np.asarray([0.9, 0.8]), np.asarray([False, False])
    )
    assert result["status"] == "not_run_no_conformal_candidates"
    assert result["selected_threshold"] is None
    assert result["selected_count"] == 0


def test_risk_control_can_pass_many_error_free_calibration_examples():
    labels = np.ones(200, dtype=np.int64)
    scores = np.linspace(0.8, 1.0, 200)
    result = learn_then_test_threshold(
        labels, scores, np.ones(200, dtype=bool), thresholds=(0.8, 0.9)
    )
    assert result["status"] == "passed"
    assert result["selected_threshold"] == 0.8
    assert result["selected_count"] == 200
    assert clopper_pearson_upper(0, 200, 0.025) < 0.1

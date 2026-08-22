import numpy as np

from tools.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import (
    label_candidate,
    soft_quality,
)


def test_continuous_marginal_gain_uses_best_improvement_not_best_raw_iou():
    labels = label_candidate(
        {1001: 0.80, 1002: 0.60},
        {1001: 0.79, 1002: 0.20},
    )
    assert labels["selected_gain_gt_encoded_id"] == 1002
    assert np.isclose(labels["marginal_iou_gain"], 0.40)
    assert np.isclose(labels["candidate_best_iou"], 0.80)


def test_continuous_marginal_gain_is_nonnegative_and_keeps_threshold_control():
    labels = label_candidate({1001: 0.62}, {1001: 0.70})
    assert labels["marginal_iou_gain"] == 0.0
    assert labels["official_threshold_cross_count"] == 0
    assert not labels["crosses_any_official_threshold"]


def test_soft_quality_has_fixed_linear_support():
    assert soft_quality(0.49) == 0.0
    assert np.isclose(soft_quality(0.725), 0.5)
    assert soft_quality(0.96) == 1.0

import numpy as np
import pytest

from tools.diagnose_candidate_pair_union_threshold_cross_oof import (
    heuristic_intersection_probability,
    heuristic_union_probability,
    prior_correct_balanced_probability,
)


def test_prior_correction_restores_rare_natural_prior_at_balanced_half_score():
    corrected = prior_correct_balanced_probability(np.asarray([0.5]), 0.05)
    assert corrected[0] == pytest.approx(0.05)


def test_fixed_union_heuristic_requires_quality_and_complementarity():
    row = {"model_features": {
        "relation__track_original_score": 0.8,
        "relation__native_original_score_median": 0.9,
        "relation__point_iou": 0.25,
    }}
    assert heuristic_union_probability(row) == pytest.approx(0.6)


def test_fixed_intersection_heuristic_requires_quality_and_overlap():
    row = {"model_features": {
        "relation__track_original_score": 0.8,
        "relation__native_original_score_median": 0.9,
        "relation__point_iou": 0.25,
    }}
    assert heuristic_intersection_probability(row) == pytest.approx(0.2)

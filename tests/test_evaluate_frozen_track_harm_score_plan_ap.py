import pytest

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE
from tools.evaluate_frozen_track_harm_score_plan_ap import validate_score_row


def _row(source, coexist, planned, multiplier, keep):
    return {
        "candidate_source": source,
        "frozen_coexist_score": coexist,
        "planned_score": planned,
        "suppression_multiplier": multiplier,
        "keep_probability": keep,
        "candidate_removed": False,
        "geometry_modified": False,
        "class_modified": False,
        "ground_truth_usage": "none",
        "ap_evaluation_run": False,
    }


def test_validate_score_row_accepts_exact_native_freeze():
    validate_score_row(_row(NATIVE_SOURCE, 1.0, 1.0, 1.0, None))


def test_validate_score_row_accepts_frozen_track_formula():
    validate_score_row(_row(TRACK_SOURCE, 0.8, 0.8 * 0.875, 0.875, 0.5))


def test_validate_score_row_rejects_native_clipping():
    with pytest.raises(ValueError, match="native"):
        validate_score_row(_row(NATIVE_SOURCE, 1.0, 0.999999, 1.0, None))

import numpy as np
import pytest

from tools.diagnose_candidate_two_stage_cascade_oof import (
    _cascade_product,
    _top_metrics,
    _validate_reference_parity,
)


def _row(index, target_state="unknown", relative_state="unknown"):
    return {
        "scene_name": f"scene{index:04d}_00",
        "track_id": index,
        "native_exact_geometry_group_id": f"group{index}",
        "labels": {
            "target_state": target_state,
            "relative_quality_state": relative_state,
        },
    }


def test_cascade_score_is_preregistered_probability_product():
    target = np.asarray([0.9, 0.2, 0.0, 1.0])
    relative = np.asarray([0.8, 0.7, 1.0, 0.25])
    assert np.allclose(_cascade_product(target, relative), [0.72, 0.14, 0.0, 0.25])

    with pytest.raises(ValueError, match="same shape"):
        _cascade_product(np.asarray([0.5]), np.asarray([0.5, 0.6]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _cascade_product(np.asarray([1.1]), np.asarray([0.5]))


def test_fixed_top_fractions_use_ceiling_and_count_unknown_and_coexist_errors():
    rows = [_row(index) for index in range(200)]
    rows[0] = _row(0, "same_target", "prefer_track")
    rows[1] = _row(1, "different_target_coexist", "coexist")
    labels = np.asarray([1] + [0] * 199, dtype=np.int64)
    scores = np.linspace(1.0, 0.0, 200)

    metrics = _top_metrics(rows, labels, scores)
    top_half = metrics["top_0_5pct"]
    top_one = metrics["top_1pct"]
    assert top_half["selected_count"] == 1
    assert top_half["precision"] == 1.0
    assert top_one["selected_count"] == 2
    assert top_one["precision"] == 0.5
    assert top_one["error_target_state_counts"] == {"different_target_coexist": 1}
    assert top_one["error_relative_quality_state_counts"] == {"coexist": 1}
    assert top_one["error_joint_state_counts"] == {
        "different_target_coexist:coexist": 1,
    }


def test_reference_parity_checks_every_eligible_relation_exactly():
    rows = [
        _row(0, "same_target", "prefer_track"),
        _row(1, "different_target_coexist", "coexist"),
        _row(2, "unknown", "unknown"),
    ]
    predictions = np.asarray([0.8, 0.2, 0.4])
    target_reference = {
        ("scene0000_00", 0, "group0"): 0.8,
        ("scene0001_00", 1, "group1"): 0.2,
    }
    result = _validate_reference_parity(
        rows, predictions, target_reference, state="target"
    )
    assert result == {"matched_relation_count": 2, "max_absolute_difference": 0.0}

    bad_reference = dict(target_reference)
    bad_reference[("scene0001_00", 1, "group1")] = 0.2001
    with pytest.raises(AssertionError, match="differ"):
        _validate_reference_parity(rows, predictions, bad_reference, state="target")


def test_relative_reference_parity_only_uses_strict_same_target_preferences():
    rows = [
        _row(0, "same_target", "prefer_track"),
        _row(1, "same_target", "prefer_native"),
        _row(2, "same_target", "equivalent_abstain"),
        _row(3, "different_target_coexist", "coexist"),
    ]
    predictions = np.asarray([0.7, 0.1, 0.4, 0.3])
    reference = {
        ("scene0000_00", 0, "group0"): 0.7,
        ("scene0001_00", 1, "group1"): 0.1,
    }
    result = _validate_reference_parity(rows, predictions, reference, state="relative")
    assert result["matched_relation_count"] == 2
    assert result["max_absolute_difference"] == 0.0

import numpy as np

from tools.train_candidate_relative_quality_multitask_oof import (
    AUXILIARY_LOSS_FRACTION,
    MAIN_LOSS_FRACTION,
    combined_training_data,
    strict_indexes,
)


def _row(scene, track_id, state):
    return {
        "scene_name": scene,
        "track_id": track_id,
        "labels": {"relative_quality_state": state},
    }


def test_fixed_multitask_weights_have_equal_normalized_branch_mass():
    rows = [
        _row("a", 0, "prefer_track"),
        _row("a", 1, "prefer_native"),
        _row("b", 0, "prefer_native"),
        _row("b", 1, "equivalent_abstain"),
    ]
    gaps = np.asarray([0.4, -0.2, 0.0, 0.1], dtype=np.float64)
    active, labels, weights, diagnostics = combined_training_data(
        rows, gaps, np.arange(len(rows))
    )
    assert active.tolist() == [0, 1, 2, 3]
    assert labels.tolist() == [1, 0, 0, 1]
    assert np.isclose(
        diagnostics["main_branch_weight_mass"],
        MAIN_LOSS_FRACTION * diagnostics["strict_relation_count"],
    )
    assert np.isclose(
        diagnostics["auxiliary_branch_weight_mass"],
        AUXILIARY_LOSS_FRACTION * diagnostics["strict_relation_count"],
    )
    assert np.isclose(weights.sum(), diagnostics["strict_relation_count"])


def test_official_tie_keeps_strict_main_supervision():
    rows = [
        _row("a", 0, "prefer_track"),
        _row("a", 1, "prefer_native"),
        _row("b", 0, "prefer_native"),
        _row("b", 1, "equivalent_abstain"),
    ]
    gaps = np.asarray([0.0, -0.3, -0.2, 0.1], dtype=np.float64)
    active, labels, weights, _ = combined_training_data(
        rows, gaps, np.arange(len(rows))
    )
    assert active.tolist() == [0, 1, 2, 3]
    assert labels.tolist() == [1, 0, 0, 1]
    assert weights[0] > 0
    assert strict_indexes(rows).tolist() == [0, 1, 2]

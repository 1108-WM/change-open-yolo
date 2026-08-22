import numpy as np

from tools.train_ncs_fi1_stage_c_v2_refinement_oof import (
    conservative_kept_indexes,
    regression_metrics,
)


def _atom(raw_id, role):
    return {"raw_superpoint_id": raw_id, "role": role, "points": np.asarray([raw_id])}


def test_conservative_removal_uses_positive_lower_bound_and_shared_connectivity():
    atoms = [_atom(1, "shared"), _atom(2, "track_only"), _atom(3, "native_only"), _atom(9, "native_only")]
    neighbors = {
        1: [{"neighbor_superpoint_id": 2}],
        2: [{"neighbor_superpoint_id": 1}, {"neighbor_superpoint_id": 3}],
        3: [{"neighbor_superpoint_id": 2}],
    }
    kept, details = conservative_kept_indexes(atoms, [None, 0.1, -0.1, -0.1], neighbors)
    assert kept == {0}
    assert details["confident_removed_indexes"] == {1}
    assert details["connectivity_removed_indexes"] == {2, 3}


def test_no_shared_union_falls_back_without_removal():
    atoms = [_atom(1, "track_only"), _atom(2, "native_only")]
    kept, details = conservative_kept_indexes(atoms, [0.2, 0.3], {})
    assert kept == {0, 1}
    assert details["fallback_no_shared"]


def test_signed_regression_metrics_do_not_clip_predictions():
    result = regression_metrics(np.asarray([-0.1, 0.2]), np.asarray([-0.2, 0.1]))
    assert result["mae"] == 0.1
    assert result["mean_prediction"] == -0.05

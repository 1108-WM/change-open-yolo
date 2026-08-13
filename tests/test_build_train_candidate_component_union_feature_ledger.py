import numpy as np

from tools.build_train_candidate_component_union_feature_ledger import (
    component_track_features,
)


def test_component_union_features_capture_union_residual_and_multiple_native_groups():
    native = {
        "n0": np.asarray([0, 1, 2], dtype=np.int64),
        "n1": np.asarray([3, 4], dtype=np.int64),
    }
    tracks = {
        7: np.asarray([0, 1, 3, 5], dtype=np.int64),
        8: np.asarray([0, 2, 6], dtype=np.int64),
    }
    rows = component_track_features(native, tracks)
    row = rows[7]
    assert row["track_inside_native_union_ratio"] == 0.75
    assert row["track_residual_outside_native_union_fraction"] == 0.25
    assert row["overlapped_native_group_count"] == 2.0
    assert row["positive_overlap_peer_track_count"] == 1.0


def test_component_union_features_are_zero_safe_without_peer_overlap():
    rows = component_track_features(
        {"n": np.asarray([0, 1], dtype=np.int64)},
        {1: np.asarray([0, 2], dtype=np.int64)},
    )
    assert rows[1]["peer_track_point_iou__max"] == 0.0
    assert rows[1]["positive_overlap_peer_track_fraction"] == 0.0

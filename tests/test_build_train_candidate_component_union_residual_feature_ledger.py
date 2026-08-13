import numpy as np

from tools.build_train_candidate_component_union_residual_feature_ledger import (
    residual_region_features,
)


def _context():
    return {
        "raw_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "sizes": np.asarray([3, 2, 1], dtype=np.int64),
        "neighbors": {
            10: [{"neighbor_superpoint_id": 11, "boundary_contact_count": 2}],
            11: [
                {"neighbor_superpoint_id": 10, "boundary_contact_count": 2},
                {"neighbor_superpoint_id": 12, "boundary_contact_count": 1},
            ],
            12: [{"neighbor_superpoint_id": 11, "boundary_contact_count": 1}],
        },
    }


def test_residual_features_capture_concentration_connectivity_boundary_and_views():
    processed = np.zeros((6, 10), dtype=np.float32)
    processed[:, 3:6] = np.asarray([
        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [255, 0, 0], [255, 0, 0], [255, 0, 0],
    ])
    processed[:, 6] = 1.0
    processed[:, 9] = np.asarray([10, 10, 10, 11, 11, 12])
    features = residual_region_features(
        processed, _context(),
        np.asarray([0, 1, 3, 4, 5]),
        np.asarray([0, 1, 2]),
        observations=[
            (np.asarray([3, 4]), 0.8, 0.9),
            (np.asarray([4, 5]), 1.0, 0.7),
        ],
    )
    assert features["residual_point_count"] == 3.0
    assert features["residual_superpoint_count"] == 2.0
    assert features["residual_dominant_superpoint_fraction"] == 2.0 / 3.0
    assert features["residual_connected_component_count"] == 1.0
    assert features["residual_largest_connected_component_fraction"] == 1.0
    assert features["residual_boundary_native_superpoint_fraction"] == 0.5
    assert features["residual_point_multiview_fraction"] == 1.0 / 3.0
    assert features["residual_point_unobserved_fraction"] == 0.0


def test_residual_features_are_zero_safe_when_track_is_inside_native_union():
    processed = np.zeros((2, 10), dtype=np.float32)
    processed[:, 6] = 1.0
    processed[:, 9] = 10
    context = {
        "raw_ids": np.asarray([10]), "sizes": np.asarray([2]), "neighbors": {},
    }
    features = residual_region_features(
        processed, context, np.asarray([0, 1]), np.asarray([0, 1]), observations=[]
    )
    assert features["residual_empty"] == 1.0
    assert features["residual_superpoint_count"] == 0.0
    assert features["residual_observation_missing"] == 1.0


def test_residual_features_reject_label_bearing_array_contract():
    processed = np.zeros((2, 12), dtype=np.float32)
    try:
        residual_region_features(
            processed, {"raw_ids": np.asarray([]), "sizes": np.asarray([]), "neighbors": {}},
            np.asarray([0]), np.asarray([1]),
        )
    except ValueError as error:
        assert "exactly inference columns" in str(error)
    else:
        raise AssertionError("label-bearing input must be rejected")

import pytest

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (
    GVC_FEATURE_NAMES,
    THRESHOLDS,
    UNION_FEATURE_NAMES,
    _gvc_features,
    _quality_target,
    _union_features,
)


def test_quality_target_is_shared_multithreshold_geometry_quality():
    assert len(THRESHOLDS) == 10
    assert _quality_target(0.49) == 0.0
    assert _quality_target(0.50) == 0.1
    assert _quality_target(0.74) == 0.5
    assert _quality_target(0.95) == 1.0


def test_gvc_features_use_source_frame_excluded_evidence_only():
    row = {
        "original_source_score": 0.7,
        "gvc_including_source_frames": {"gvc": {"mean": 0.99}},
        "gvc_source_frame_excluded": {
            "gvc": {"mean": 0.2, "max": 0.3, "variance": 0.01},
            "projected_box_iou": {"mean": 0.4, "max": 0.5, "variance": 0.02},
            "visible_point_mask_support": {"mean": 0.6, "max": 0.7, "variance": 0.03},
            "selected_view_count": 2,
            "matched_selected_view_count": 1,
            "zero_support_selected_view_fraction": 0.5,
            "selected_frames": [
                {"visible_point_fraction": 0.25},
                {"visible_point_fraction": 0.75},
            ],
        },
    }
    node = {"point_count": 10, "member_count": 2, "canonical_frozen_score": 0.8}
    features = _gvc_features(row, node, 100)
    assert tuple(features) == GVC_FEATURE_NAMES
    assert features["gvc_mean"] == pytest.approx(0.2)
    assert features["matched_view_fraction"] == pytest.approx(0.5)
    assert features["visible_point_fraction_mean"] == pytest.approx(0.5)


def test_union_features_join_frozen_parent_and_relation_evidence():
    union = {
        "base_quality": 0.4,
        "track_quality_q": 0.5,
        "native_group_median_quality_q": 0.6,
        "threshold_cross_probability": 0.1,
        "balanced_fit_raw_probability": 0.2,
        "support_relation_count": 2,
    }
    relation = {"features": {
        "point_iou": 0.3,
        "native_inside_track_ratio": 0.4,
        "track_inside_native_ratio": 0.5,
    }}
    node = {"point_count": 20, "member_count": 1, "canonical_frozen_score": 0.05}
    features = _union_features(union, relation, node, 200)
    assert tuple(features) == UNION_FEATURE_NAMES
    assert features["base_quality"] == pytest.approx(0.4)
    assert features["point_iou"] == pytest.approx(0.3)
    assert features["public_track_minus_native_gvc"] == 0.0

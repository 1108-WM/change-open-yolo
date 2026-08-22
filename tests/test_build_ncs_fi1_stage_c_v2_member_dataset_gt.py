import numpy as np
import pytest

from tools.build_ncs_fi1_stage_c_v2_member_dataset_gt import (
    fixed_target_removal_labels,
    member_evidence_features,
)


def test_fixed_target_removal_label_is_signed_continuous_utility():
    helpful = fixed_target_removal_labels(
        union_point_count=100, atom_point_count=20, target_point_count=80,
        union_target_intersection=70, atom_target_intersection=0,
    )
    harmful = fixed_target_removal_labels(
        union_point_count=100, atom_point_count=20, target_point_count=80,
        union_target_intersection=70, atom_target_intersection=20,
    )
    assert helpful["delta_iou_remove"] > 0.0
    assert harmful["delta_iou_remove"] < 0.0
    assert helpful["delta_iou_remove"] != round(helpful["delta_iou_remove"])


def test_member_evidence_is_atom_local_and_uses_relative_depth_weights():
    source = [np.asarray([True, False, True]), np.asarray([True, True, False])]
    visible = [np.asarray([True, True, False]), np.asarray([True, True, True])]
    inside = [np.asarray([True, False, False]), np.asarray([True, True, False])]
    weights = [np.asarray([0.8, 0.0, 0.0]), np.asarray([0.5, 0.9, 0.0])]
    result = member_evidence_features(
        np.asarray([0, 1]), source, visible, inside, weights, [0.9, 0.7], [0.8, 0.6]
    )
    assert result["member_source_coverage_mean"] == pytest.approx(0.75)
    assert result["member_relative_mask_coverage_mean"] == pytest.approx(0.75)
    assert result["member_relative_depth_weighted_coverage_mean"] == pytest.approx(0.55)
    assert result["member_relative_point_inside_given_visible_mean"] == pytest.approx(0.75)

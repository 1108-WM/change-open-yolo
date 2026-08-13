import numpy as np
import pytest

from tools.build_train_candidate_pair_union_utility_ledger import (
    proposal_points,
    select_union_target,
)


def test_union_target_prefers_official_threshold_crossing_before_raw_iou():
    selected = select_union_target(
        {1: 0.91, 2: 0.56},
        {1: 0.90, 2: 0.49},
    )
    assert selected["target_gt_instance_id"] == 2
    assert selected["official_threshold_cross_count"] == 2
    assert selected["iou_gain"] == pytest.approx(0.07)


def test_union_target_handles_void_only_proposal():
    selected = select_union_target({}, {1: 0.5})
    assert selected["target_gt_instance_id"] is None
    assert selected["official_threshold_cross_count"] == 0


def test_pair_intersection_geometry_is_exact_shared_point_set():
    points = proposal_points(
        "pair_intersection",
        np.asarray([1, 2, 4, 8], dtype=np.int64),
        np.asarray([2, 3, 4, 9], dtype=np.int64),
    )
    assert points.tolist() == [2, 4]

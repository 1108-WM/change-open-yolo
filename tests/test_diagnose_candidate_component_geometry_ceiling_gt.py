import numpy as np
import pytest

from tools.diagnose_candidate_component_geometry_ceiling_gt import (
    iou_by_gt,
    superpoint_agreement_fusion,
    superpoint_vote_fusion,
)


def test_iou_by_gt_uses_candidate_and_gt_union_sizes():
    gt = np.asarray([1, 1, 1, 2, 2, 0])
    values = iou_by_gt(np.asarray([0, 1, 3]), gt, {1: 3, 2: 2})
    assert values[1] == pytest.approx(2 / 4)
    assert values[2] == pytest.approx(1 / 4)


def test_fixed_superpoint_fusions_expand_selected_raw_superpoints():
    superpoints = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    points_by_sp = {0: np.arange(4), 1: np.arange(4, 8)}
    sizes = {0: 4, 1: 4}
    left = np.asarray([0, 1, 4])
    right = np.asarray([2, 3, 5, 6, 7])
    vote = superpoint_vote_fusion(
        left, right, superpoints, points_by_sp, sizes
    )
    agreement = superpoint_agreement_fusion(
        left, right, superpoints, points_by_sp
    )
    assert np.array_equal(vote, np.arange(8))
    assert np.array_equal(agreement, np.arange(8))

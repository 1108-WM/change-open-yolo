import numpy as np
import pytest

from tools.evaluate_official100_geometry_group_ranking_oof_ap import (
    build_group_aware_scores,
    geometry_groups_and_audit,
)


def test_geometry_groups_are_exact_and_validate_training_group_sizes():
    masks = np.asarray([
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ], dtype=bool)
    groups, audit = geometry_groups_and_audit(masks, [2, 2, 1, 2], [2, 2, 1, 1])
    assert groups == [[0, 1], [2], [3]]
    assert audit["geometry_group_count"] == 3
    with pytest.raises(ValueError, match="组大小"):
        geometry_groups_and_audit(masks, [2, 2, 1, 2], [1, 1, 1, 1])


def test_group_representative_uses_highest_original_score_then_smallest_id():
    original, quality, rows = build_group_aware_scores(
        np.asarray([0.4, 0.9, 0.9, 0.2]),
        np.asarray([0.7, 0.8, 0.6, 0.3]),
        [[0, 1, 2], [3]],
    )
    assert rows[0]["representative_candidate_id"] == 1
    assert rows[0]["group_median_oof_quality"] == pytest.approx(0.7)
    assert original.tolist() == pytest.approx([-1.6, 0.9, -1.1, 0.2])
    assert quality.tolist() == pytest.approx([-1.6, 0.7, -1.1, 0.3])


def test_group_scoring_rejects_missing_or_overlapping_members():
    with pytest.raises(ValueError, match="覆盖"):
        build_group_aware_scores(
            np.asarray([0.4, 0.9]), np.asarray([0.7, 0.8]), [[0], [0]]
        )

import numpy as np

from tools.dm_sms_core import (
    canonical_member,
    compute_sms,
    exact_geometry_groups,
    sms_keep_mask,
    visible_ratio_multiscale_feature,
)


def test_sms_uses_all_proposals_for_selected_class_population():
    similarities = np.asarray([
        [0.9, 0.1],
        [0.8, 0.7],
        [0.0, 0.95],
    ], dtype=np.float32)
    result = compute_sms(similarities)
    expected_class0 = (0.9 - np.mean([0.9, 0.8, 0.0])) / np.std([0.9, 0.8, 0.0])
    assert np.isclose(result.scores[0], expected_class0)
    assert result.top_classes.tolist() == [0, 0, 1]


def test_sms_degenerate_class_is_validated_and_retained():
    result = compute_sms(np.ones((3, 2), dtype=np.float32))
    assert not result.valid.any()
    assert sms_keep_mask(result, threshold=0.0).all()


def test_sms_threshold_deletes_strictly_negative_valid_values():
    result = compute_sms(np.asarray([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32))
    assert sms_keep_mask(result, threshold=0.0).tolist() == [False, True]


def test_visible_ratio_multiscale_feature_matches_weighted_sum():
    features = np.asarray([
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    ])
    output = visible_ratio_multiscale_feature(features, np.asarray([1.0, 0.5]))
    expected = np.asarray([3.0, 1.5])
    expected /= np.linalg.norm(expected)
    assert np.allclose(output, expected)


def test_exact_geometry_groups_and_canonical_member():
    masks = np.asarray([
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=bool)
    groups = exact_geometry_groups(masks)
    assert sorted(sorted(group) for group in groups) == [[0, 1], [2]]
    assert canonical_member([0, 1], np.asarray([0.2, 0.8, 0.1])) == 1
    assert canonical_member([0, 1], np.asarray([0.8, 0.8, 0.1])) == 0


def test_canonical_member_uses_source_then_source_local_id_for_exact_ties():
    scores = np.asarray([0.9, 0.9, 0.9, 0.9])
    source_ranks = np.asarray([2, 1, 0, 0])
    candidate_ids = np.asarray([0, 0, 9, 3])
    assert canonical_member(
        [0, 1, 2, 3], scores, source_ranks, candidate_ids
    ) == 3

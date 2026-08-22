import numpy as np

from tools.build_ncs_fi1_stage_d_v2_rank_marginal_dataset_gt import rank_labels, rank_prefix


def test_rank_prefix_includes_only_baseline_at_or_above_candidate_score():
    baseline = [
        ({"geometry_key": "a"}, np.asarray([1]), 0.8),
        ({"geometry_key": "b"}, np.asarray([2]), 0.5),
        ({"geometry_key": "c"}, np.asarray([3]), 0.2),
    ]
    selected = rank_prefix(baseline, 0.5)
    assert [row[0]["geometry_key"] for row in selected] == ["a", "b"]


def test_rank_conditioned_label_preserves_continuous_gain():
    labels = rank_labels({1001: 0.7}, {1001: 0.4})
    assert np.isclose(labels["rank_conditioned_marginal_iou_gain"], 0.3)
    assert np.isclose(labels["rank_prefix_best_iou_for_selected_gain_target"], 0.4)

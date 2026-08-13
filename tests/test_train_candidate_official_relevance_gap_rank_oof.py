import numpy as np

from tools.train_candidate_official_relevance_gap_rank_oof import (
    FEATURES,
    official_relevance,
    rank_metrics,
    relevance_targets,
    same_target_rows,
)


def _row(scene, track_id, state, track_iou, native_iou):
    target = "same_target" if state != "coexist" else "different_target_coexist"
    return {
        "scene_name": scene,
        "track_id": track_id,
        "labels": {
            "reliable_pair": state != "unknown",
            "target_state": target if state != "unknown" else "unknown",
            "relative_quality_state": state,
            "track_best_gt_iou": track_iou,
            "native_best_gt_iou": native_iou,
        },
    }


def test_official_relevance_uses_the_nine_strict_ap_thresholds():
    assert official_relevance(0.50) == 0.0
    assert official_relevance(0.500001) == 1.0 / 9.0
    assert official_relevance(0.90) == 8.0 / 9.0
    assert official_relevance(0.900001) == 1.0


def test_same_target_population_keeps_equivalent_but_excludes_other_targets():
    rows = [
        _row("a", 0, "prefer_track", 0.8, 0.6),
        _row("a", 1, "prefer_native", 0.6, 0.8),
        _row("b", 2, "equivalent_abstain", 0.7, 0.7),
        _row("b", 3, "coexist", 0.8, 0.6),
        _row("c", 4, "unknown", 0.0, 0.0),
    ]
    selected = same_target_rows(rows)
    assert [row["track_id"] for row in selected] == [0, 1, 2]
    track, native, gap = relevance_targets(selected)
    assert np.allclose(gap, track - native)
    assert gap[0] > 0 and gap[1] < 0 and gap[2] == 0


def test_gap_weighted_rank_metrics_reward_correct_ordering():
    rows = [
        _row("a", 0, "prefer_track", 0.8, 0.6),
        _row("a", 1, "prefer_native", 0.6, 0.8),
        _row("b", 0, "prefer_track", 0.9, 0.5),
        _row("b", 1, "prefer_native", 0.5, 0.9),
    ]
    _, _, gaps = relevance_targets(rows)
    indexes = np.arange(len(rows))
    aligned = rank_metrics(rows, indexes, gaps, gaps * 5.0)
    reversed_ = rank_metrics(rows, indexes, gaps, -gaps * 5.0)
    assert aligned["gap_weighted_pairwise_logistic_loss"] < reversed_[
        "gap_weighted_pairwise_logistic_loss"
    ]
    assert aligned["relation_unweighted_roc_auc"] == 1.0


def test_rank_features_are_pure_inference_fields():
    assert not any(name.startswith("label_") or "gt_" in name for name in FEATURES)

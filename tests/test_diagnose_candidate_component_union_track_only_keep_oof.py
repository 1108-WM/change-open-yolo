import numpy as np

from tools.train_candidate_component_list_calibration_head_oof import _feature_matrix
from tools.diagnose_candidate_component_union_track_only_keep_oof import (
    fit_track_only,
    track_metrics,
)


def test_track_metrics_reward_better_ranking_and_calibration():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    good = np.asarray([0.05, 0.15, 0.8, 0.9])
    bad = 1.0 - good
    weights = np.ones(4)
    good_metrics = track_metrics(labels, good, weights)
    bad_metrics = track_metrics(labels, bad, weights)
    assert good_metrics["component_balanced_pr_auc"] > bad_metrics[
        "component_balanced_pr_auc"
    ]
    assert good_metrics["component_balanced_brier"] < bad_metrics[
        "component_balanced_brier"
    ]


def test_track_only_fit_never_scores_native_candidates():
    rows = []
    for scene_index in range(8):
        scene = f"s{scene_index}"
        for source, label, value in (
            ("d2b_track", 0, -1.0),
            ("d2b_track", 1, 1.0),
            ("native_mask3d_yoloworld", 0, -0.5),
        ):
            rows.append({
                "scene_name": scene,
                "relation_component_id": scene_index,
                "candidate_source": source,
                "label_component_unique_winner": label,
                "model_features": {"signal": value + 0.01 * scene_index},
            })
    matrix, _ = _feature_matrix(rows)
    validation, labels, scores = fit_track_only(
        rows, matrix,
        [f"s{index}" for index in range(6)],
        ["s6", "s7"],
        seed=7,
    )
    assert len(validation) == 4
    assert all(rows[index]["candidate_source"] == "d2b_track" for index in validation)
    assert labels.tolist() == [0, 1, 0, 1]
    assert np.isfinite(scores).all()

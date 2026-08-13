import numpy as np

from tools.train_candidate_target_consistency_oof import (
    LEARNED_QUALITY_MARKERS,
    MODEL_FEATURES,
    binary_metrics,
    class_balanced_weights,
    nested_fit_predict,
    scene_track_balanced_weights,
    seeded_scene_folds,
)


def test_frozen_feature_groups_exclude_learned_candidate_quality_scores():
    for feature_names in MODEL_FEATURES.values():
        assert not any(
            marker in name for name in feature_names for marker in LEARNED_QUALITY_MARKERS
        )
        assert not any(name.startswith("label_") or "gt_" in name for name in feature_names)


def test_scene_track_weights_equalize_scenes_and_tracks():
    rows = [
        {"scene_name": "a", "track_id": 0},
        {"scene_name": "a", "track_id": 0},
        {"scene_name": "a", "track_id": 1},
        {"scene_name": "b", "track_id": 2},
    ]
    weights = scene_track_balanced_weights(rows)
    assert np.isclose(weights[:3].sum(), weights[3:].sum())
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights.sum(), len(rows))


def test_class_weights_equalize_binary_mass():
    labels = np.asarray([0, 0, 0, 1])
    weights = class_balanced_weights(labels, np.ones(4))
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())


def test_seeded_inner_folds_assign_each_scene_once():
    scenes = [f"scene{index}" for index in range(8)]
    folds = seeded_scene_folds(scenes, fold_count=4, seed=7)
    validation = [scene for fold in folds for scene in fold["validation_scenes"]]
    assert len(folds) == 4
    assert sorted(validation) == sorted(scenes)
    assert all(len(fold["train_scenes"]) == 6 for fold in folds)
    assert all(len(fold["validation_scenes"]) == 2 for fold in folds)


def test_nested_fit_predict_uses_inner_oof_calibration():
    rows, labels, matrix = [], [], []
    for scene_index in range(10):
        for relation_index in range(4):
            label = relation_index % 2
            rows.append({"scene_name": f"scene{scene_index}", "track_id": relation_index // 2})
            labels.append(label)
            matrix.append([label + 0.05 * scene_index, relation_index])
    labels = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(matrix, dtype=np.float64)
    train = np.arange(0, 8 * 4, dtype=np.int64)
    validation = np.arange(8 * 4, 10 * 4, dtype=np.int64)
    predictions, diagnostics = nested_fit_predict(
        rows, matrix, labels, train, validation, outer_fold_index=0
    )
    assert len(predictions) == len(validation)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    assert diagnostics["inner_fold_count"] == 4
    assert len(diagnostics["inner_folds"]) == 4


def test_binary_metrics_report_calibration_and_ranking():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(labels, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["brier"] < 0.05
    assert metrics["ece10"] >= 0.0

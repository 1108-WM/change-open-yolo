import json

import pytest

from tools.train_candidate_quality_head_oof import (
    build_seeded_scene_folds,
    canonicalize_predictions,
    load_frozen_split_manifest,
    q_spearman_units,
    run_oof,
    source_balanced_weights,
    base_sample_weights,
)


def _row(scene, source, candidate_id, iou):
    row = {
        "scene_name": scene, "candidate_source": source, "candidate_id": candidate_id,
        "point_count": 10 + candidate_id, "point_fraction_of_scene": .1,
        "original_source_score": iou, "gvc_excluded_mean": iou, "gvc_excluded_max": iou,
        "gvc_excluded_variance": .01, "gvc_excluded_selected_view_count": 2,
        "gvc_excluded_matched_view_count": 2, "gvc_excluded_zero_support_fraction": 0.,
        "label_best_gt_iou": iou, "label_valid_iou25": iou >= .25, "label_valid_iou50": iou >= .5,
    }
    if source == "native_mask3d_yoloworld":
        row["native_exact_geometry_group_size"] = 2
    else:
        row.update({
            "support_view_count": 2, "superpoint_count": 3, "source_frame_count": 2,
            "merge_action_count": 0, "mean_consensus_rate": iou, "mean_edge_score": iou,
            "mean_node_quality": iou,
        })
    return row


def test_oof_is_scene_disjoint_and_returns_one_prediction_per_candidate():
    rows = []
    scenes = []
    for index in range(100):
        scene = f"scene{index:04d}_00"
        scenes.append(scene)
        rows.extend([_row(scene, "native_mask3d_yoloworld", 0, .8), _row(scene, "d2b_track", 1, .2)])
    manifest = {"scene_count": 100, "folds": build_seeded_scene_folds(scenes, fold_count=5, seed=20260808)}
    summary, predictions, detail = run_oof(rows, manifest, "official100_v1", seed=20260808)
    assert summary["candidate_count"] == len(rows)
    assert len(predictions) == len(rows)
    assert len(detail["folds"]) == 5
    assert all(len(fold["validation_scenes"]) == 20 for fold in detail["folds"])
    assert all("D_plus_gvc" in row["predictions"] for row in predictions)
    q = summary["oof_metrics"]["D_plus_gvc"]["q"]["overall"]
    assert set(("class_expanded_unweighted_spearman", "native_geometry_group_unweighted_spearman", "track_raw_unweighted_spearman")) <= set(q)


def test_seeded_scene_folds_apply_seed_deterministically():
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    first = build_seeded_scene_folds(scenes, fold_count=5, seed=7)
    assert first == build_seeded_scene_folds(scenes, fold_count=5, seed=7)
    assert first != build_seeded_scene_folds(scenes, fold_count=5, seed=8)
    assert all((len(fold["train_scenes"]), len(fold["validation_scenes"])) == (80, 20) for fold in first)


def test_frozen_manifest_rejects_non_partitioned_validation(tmp_path):
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    manifest = {"scene_count": 100, "folds": build_seeded_scene_folds(scenes, fold_count=5, seed=7)}
    manifest["folds"][1]["validation_scenes"][0] = manifest["folds"][0]["validation_scenes"][0]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partition|exactly one"):
        load_frozen_split_manifest(path, scenes)


def test_q_predictions_are_clipped_before_oof_consumers_see_them():
    result = canonicalize_predictions("q", [-0.2, 0.25, 1.4])
    assert result.tolist() == [0.0, 0.25, 1.0]


def test_native_geometry_spearman_folds_duplicate_class_expansions():
    rows = [
        {"candidate_source": "native_mask3d_yoloworld", "scene_name": "a", "candidate_id": 0, "_native_geometry_group_id": "a:one"},
        {"candidate_source": "native_mask3d_yoloworld", "scene_name": "a", "candidate_id": 1, "_native_geometry_group_id": "a:one"},
        {"candidate_source": "native_mask3d_yoloworld", "scene_name": "a", "candidate_id": 2, "_native_geometry_group_id": "a:two"},
        {"candidate_source": "native_mask3d_yoloworld", "scene_name": "a", "candidate_id": 3, "_native_geometry_group_id": "a:two"},
    ]
    result = q_spearman_units(rows, [0.2, 0.2, 0.8, 0.8], [0.3, 0.4, 0.6, 0.7])
    assert result["native_geometry_group_count"] == 2
    assert result["native_geometry_group_unweighted_spearman"] == pytest.approx(1.0)


def test_training_source_balance_uses_only_current_fold_rows():
    rows = [
        _row("a", "native_mask3d_yoloworld", 0, .8),
        _row("a", "native_mask3d_yoloworld", 1, .8),
        _row("b", "native_mask3d_yoloworld", 2, .8),
        _row("b", "d2b_track", 3, .2),
        _row("c", "d2b_track", 4, .2),
    ]
    train = [0, 1, 3]
    weights = source_balanced_weights(rows, base_sample_weights(rows), train)
    assert weights[[0, 1]].sum() == pytest.approx(1.5)
    assert weights[[3]].sum() == pytest.approx(1.5)
    assert weights[[2, 4]].sum() == 0.0

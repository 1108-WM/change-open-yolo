import json

import pytest
import numpy as np

from tools.diagnose_train_candidate_pair_oof_scores import _load_split_folds, _score_metrics, build_scored_pairs
from tools.train_candidate_quality_head_oof import build_seeded_scene_folds


def _score(scene, source, candidate_id, q):
    return {
        "scene_name": scene, "candidate_source": source, "candidate_id": candidate_id,
        "predictions": {
            "C_plus_geometry_track_structure": {"q": q, "valid25": q, "valid50": q},
            "D_plus_gvc": {"q": q, "valid25": q, "valid50": q},
        },
    }


def test_pair_score_audit_uses_native_median_and_preserves_group_range():
    scene = "scene0001_00"
    lookup = {
        (scene, "d2b_track", 3): _score(scene, "d2b_track", 3, .9),
        (scene, "native_mask3d_yoloworld", 0): _score(scene, "native_mask3d_yoloworld", 0, .2),
        (scene, "native_mask3d_yoloworld", 1): _score(scene, "native_mask3d_yoloworld", 1, .6),
    }
    pairs = [{
        "scene_name": scene, "track_id": 3, "native_exact_geometry_group_id": "group0",
        "native_member_candidate_ids": [0, 1], "label_pair_preference": "prefer_track",
        "same_best_gt": True, "reliable_pair": True, "reliability_state": "both_iou25_reliable",
        "track_original_score": .7, "native_original_score_median": .4,
    }]
    scored, groups = build_scored_pairs(lookup, {scene: 0}, pairs)
    assert abs(scored[0]["delta_C_plus_geometry_track_structure_q"] - .5) < 1e-12
    assert abs(scored[0]["delta_raw_original_score"] - .3) < 1e-12
    assert abs(groups[0]["scores"]["D_plus_gvc:q"]["range"] - .4) < 1e-12


def test_pair_score_audit_rejects_unreliable_preference_label():
    scene = "scene0001_00"
    lookup = {
        (scene, "d2b_track", 3): _score(scene, "d2b_track", 3, .9),
        (scene, "native_mask3d_yoloworld", 0): _score(scene, "native_mask3d_yoloworld", 0, .2),
    }
    pairs = [{
        "scene_name": scene, "track_id": 3, "native_exact_geometry_group_id": "group0",
        "native_member_candidate_ids": [0], "label_pair_preference": "prefer_track",
        "same_best_gt": True, "reliable_pair": False, "reliability_state": "native_only_iou25_reliable",
        "track_original_score": .7, "native_original_score_median": .4,
    }]
    with pytest.raises(ValueError, match="unreliable"):
        build_scored_pairs(lookup, {scene: 0}, pairs)


def test_top_half_percent_has_wilson_interval():
    rows = [
        {"scene_name": "scene0001_00", "track_id": index, "native_exact_geometry_group_id": str(index),
         "label_pair_preference": "prefer_track" if index == 0 else "unknown", "score": 1.0 - index / 100}
        for index in range(100)
    ]
    labels = np.asarray([1] + [0] * 99, dtype=np.int64)
    result = _score_metrics(rows, labels, "score")
    top = result["top_selection"]["top_0_5pct"]
    assert top["selected_count"] == 1
    assert top["precision"] == 1.0
    assert top["precision_wilson_95"]["lower"] < 1.0


def test_pair_audit_requires_the_frozen_official100_partition(tmp_path):
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    payload = {
        "scene_count": 100,
        "existing_scene_count": 20,
        "new_scene_count": 80,
        "folds": build_seeded_scene_folds(scenes, fold_count=5, seed=20260808),
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload))
    mapping, cohorts, _ = _load_split_folds(path, scenes)
    assert len(mapping) == 100
    assert (len(cohorts["existing20"]), len(cohorts["new80"])) == (20, 80)
    payload["folds"][0]["train_scenes"] = payload["folds"][0]["train_scenes"][1:]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="80 train"):
        _load_split_folds(path, scenes)

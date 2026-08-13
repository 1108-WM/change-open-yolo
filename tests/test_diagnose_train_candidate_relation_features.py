import json

from tools.diagnose_train_candidate_relation_features import (
    _load_partition,
    _metric,
    _numeric_feature_names,
    _task_rows,
)
from tools.train_candidate_quality_head_oof import build_seeded_scene_folds


def _row(scene, track, target, quality, value):
    return {
        "scene_name": scene,
        "track_id": track,
        "features": {"signal": value, "flag": value > 0.5, "text": "audit"},
        "labels": {"target_state": target, "relative_quality_state": quality},
    }


def test_same_target_task_excludes_unknown_instead_of_using_it_as_negative():
    rows = [
        _row("a", 0, "same_target", "prefer_track", 0.9),
        _row("b", 1, "different_target_coexist", "coexist", 0.1),
        _row("c", 2, "unknown", "unknown", 1.0),
    ]
    selected, labels = _task_rows(rows, "same_target_vs_coexist")
    assert len(selected) == 2
    assert labels.tolist() == [1, 0]
    assert _metric(selected, labels, "signal")["natural_roc_auc"] == 1.0


def test_numeric_features_exclude_non_numeric_evidence():
    rows = [
        _row("a", 0, "same_target", "prefer_track", 0.9),
        _row("b", 1, "different_target_coexist", "coexist", 0.1),
    ]
    assert _numeric_feature_names(rows) == ["flag", "signal"]


def test_partition_uses_explicit_existing20_not_scene_order(tmp_path):
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    payload = {
        "scene_count": 100,
        "existing_scene_count": 20,
        "new_scene_count": 80,
        "folds": build_seeded_scene_folds(scenes, fold_count=5, seed=20260808),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    existing_path = tmp_path / "existing20.txt"
    existing_path.write_text("\n".join(scenes[-20:]) + "\n")
    _, cohorts, _ = _load_partition(manifest, scenes, existing_path)
    assert cohorts["existing20"] == set(scenes[-20:])
    assert cohorts["new80"] == set(scenes[:-20])

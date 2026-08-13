import json

import numpy as np

from tools.build_train_candidate_pair_ledger_v2 import build_scene


def test_pair_v2_collapses_native_geometry_and_marks_coexist(tmp_path):
    scene = "scene0001_00"
    records = tmp_path / "records"
    ledger = records / scene / "candidate_quality_training_ledger" / scene
    ledger.mkdir(parents=True)
    native_cache = records / scene / "native_cache"
    native_cache.mkdir()
    masks = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
    np.save(native_cache / f"{scene}_pred_masks.npy", masks)
    candidates = [
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 0, "original_source_score": .8, "label_best_gt_iou": .7, "label_best_gt_instance_id": 1, "label_valid_iou25": True, "label_valid_iou50": True},
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 1, "original_source_score": .6, "label_best_gt_iou": .7, "label_best_gt_instance_id": 1, "label_valid_iou25": True, "label_valid_iou50": True},
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 2, "original_source_score": .4, "label_best_gt_iou": .8, "label_best_gt_instance_id": 2, "label_valid_iou25": True, "label_valid_iou50": True},
        {"scene_name": scene, "candidate_source": "d2b_track", "candidate_id": 9, "original_source_score": .5, "label_best_gt_iou": .9, "label_best_gt_instance_id": 2, "label_valid_iou25": True, "label_valid_iou50": True},
    ]
    (ledger / "candidate_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    pairs = []
    for native_id in (0, 1, 2):
        pairs.append({"track_id": 9, "native_candidate_id": native_id, "point_iou": .5 if native_id < 2 else .4, "track_inside_native_ratio": .5, "native_inside_track_ratio": .5, "mutual_duplicate_strict_099": False})
    (ledger / "pair_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pairs))
    rows, summary = build_scene(scene, records)
    assert summary["v1_relation_count"] == 3
    assert summary["v2_geometry_relation_count"] == 2
    assert summary["folded_source_relation_count"] == 3
    duplicate = next(row for row in rows if row["native_exact_geometry_group_size"] == 2)
    assert duplicate["native_member_candidate_ids"] == [0, 1]
    assert duplicate["label_pair_preference"] == "coexist"
    matched = next(row for row in rows if row["native_exact_geometry_group_size"] == 1)
    assert matched["same_best_gt"] is True
    assert matched["label_pair_preference"] == "prefer_track"


def test_pair_v2_marks_low_iou_gt_id_as_unknown(tmp_path):
    scene = "scene0002_00"
    records = tmp_path / "records"
    ledger = records / scene / "candidate_quality_training_ledger" / scene
    ledger.mkdir(parents=True)
    native_cache = records / scene / "native_cache"
    native_cache.mkdir()
    np.save(native_cache / f"{scene}_pred_masks.npy", np.array([[1], [1], [0]], dtype=bool))
    candidates = [
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 0, "original_source_score": .8, "label_best_gt_iou": .01, "label_best_gt_instance_id": 1, "label_valid_iou25": False, "label_valid_iou50": False},
        {"scene_name": scene, "candidate_source": "d2b_track", "candidate_id": 3, "original_source_score": .6, "label_best_gt_iou": .4, "label_best_gt_instance_id": 2, "label_valid_iou25": True, "label_valid_iou50": False},
    ]
    (ledger / "candidate_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    (ledger / "pair_labels.jsonl").write_text(json.dumps({"track_id": 3, "native_candidate_id": 0, "point_iou": .2, "track_inside_native_ratio": .3, "native_inside_track_ratio": .3, "mutual_duplicate_strict_099": False}) + "\n")
    rows, _ = build_scene(scene, records)
    assert rows[0]["label_pair_preference"] == "unknown"
    assert rows[0]["reliable_pair"] is False
    assert rows[0]["same_best_gt"] is None
    assert rows[0]["iou_margin"] is None

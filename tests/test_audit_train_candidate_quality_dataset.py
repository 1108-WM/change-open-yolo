import json

import pytest

from tools.audit_train_candidate_quality_dataset import audit_dataset, feature_schema


def _row(scene, source, candidate_id):
    row = {
        "scene_name": scene, "candidate_source": source, "candidate_id": candidate_id,
        "ground_truth_usage": "label_only", "point_count": 10, "point_fraction_of_scene": .1,
        "original_source_score": .6, "gvc_excluded_mean": .4, "gvc_excluded_max": .5,
        "gvc_excluded_variance": .01, "gvc_excluded_selected_view_count": 2,
        "gvc_excluded_matched_view_count": 2, "gvc_excluded_zero_support_fraction": 0.,
        "label_best_gt_iou": .6, "label_best_gt_instance_id": 1,
        "label_valid_iou25": True, "label_valid_iou50": True,
    }
    if source == "native_mask3d_yoloworld":
        row["native_exact_geometry_group_size"] = 2
    return row


def _write_scene(root, scene):
    record = root / scene
    record.mkdir(parents=True)
    (record / "native_export_manifest.json").write_text("{}")
    (record / "track_pipeline_manifest.json").write_text("{}")
    ledger = record / "candidate_quality_training_ledger" / scene
    ledger.mkdir(parents=True)
    rows = [_row(scene, "native_mask3d_yoloworld", 0), _row(scene, "d2b_track", 0)]
    (ledger / "candidate_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_audit_rejects_evaluation_overlap(tmp_path):
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene0001_01\n")
    _write_scene(tmp_path / "records", "scene0001_01")
    evaluation = tmp_path / "evaluation.txt"
    evaluation.write_text("scene0001_01\n")
    with pytest.raises(ValueError, match="overlap"):
        audit_dataset(scene_list, tmp_path / "records", [evaluation])


def test_feature_schema_excludes_labels_and_identifiers():
    schema = feature_schema("official100_v1")
    assert schema["protocol_name"] == "official100_v1"
    assert schema["version"] == "official100_v1_candidate_quality_feature_schema_v1"
    fields = [field for group in schema["groups"].values() for field in group["categorical"] + group["numeric"]]
    assert not any(field.startswith("label_") for field in fields)
    assert "candidate_id" not in fields
    assert "native_exact_geometry_group_size" not in fields


def test_audit_rejects_missing_evaluation_scene_list(tmp_path):
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene0001_01\n")
    with pytest.raises(ValueError, match="missing required evaluation"):
        audit_dataset(scene_list, tmp_path / "records", [tmp_path / "missing.txt"])


def test_audit_rejects_iou_label_contract_mismatch(tmp_path):
    scene = "scene0001_01"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(f"{scene}\n")
    _write_scene(tmp_path / "records", scene)
    ledger = tmp_path / "records" / scene / "candidate_quality_training_ledger" / scene / "candidate_labels.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0]["label_best_gt_iou"] = 1.1
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    evaluation = tmp_path / "evaluation.txt"
    evaluation.write_text("scene9999_00\n")
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        audit_dataset(scene_list, tmp_path / "records", [evaluation])

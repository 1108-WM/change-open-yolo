import json
from argparse import Namespace

import pytest
import yaml

from tools.build_dm_sms1_candidate_evidence_manifest import build_candidate_row, run


def test_candidate_evidence_builds_two_order_tasks_without_deciding_class():
    attribute = {
        "task_id": "task", "scene_name": "scene", "plan_index": 0,
        "plan_key": "plan", "geometry_key": "plan", "visual_geometry_key": "scene:g",
        "geometry_hash": "g",
    }
    semantic = {
        "scene_name": "scene", "plan_key": "plan", "geometry_key": "plan",
        "visual_geometry_key": "scene:g", "geometry_hash": "g",
        "geometry_locator_read_only": {}, "candidate_source": "native",
        "frozen_class_index": 0, "challenger_score": 0.5, "append_only": False,
        "canonical_frozen_class_index": 0, "canonical_frozen_score": 0.5,
        "finite_class_hypotheses": [
            {"class_index": 0, "sources": ["frozen_control"]},
            {"class_index": 1, "sources": ["alpha_main"]},
        ],
    }
    row = build_candidate_row(attribute, semantic, ["chair", "table"])
    assert row["candidate_order_ab"] == ["chair", "table"]
    assert row["candidate_order_ba"] == ["table", "chair"]
    assert row["canonical_frozen_class_index"] == 0
    assert row["class_decision_made"] is False
    assert row["selected_class_index"] is None


def test_candidate_evidence_rejects_cross_scene_geometry_join():
    attribute = {
        "task_id": "task", "scene_name": "scene_a", "plan_key": "plan",
        "geometry_key": "plan", "visual_geometry_key": "scene_a:g",
        "geometry_hash": "same",
    }
    semantic = {
        "scene_name": "scene_b", "plan_key": "plan", "geometry_key": "plan",
        "visual_geometry_key": "scene_b:g", "geometry_hash": "same",
        "geometry_locator_read_only": {}, "candidate_source": "native",
        "frozen_class_index": 0, "challenger_score": 0.5, "append_only": False,
        "canonical_frozen_class_index": 0, "canonical_frozen_score": 0.5,
        "finite_class_hypotheses": [{"class_index": 0, "sources": ["frozen_control"]}],
    }
    with pytest.raises(ValueError, match="scene and geometry join mismatch"):
        build_candidate_row(attribute, semantic, ["chair"])


def _write_root(root, filename, rows, summary=None):
    root.mkdir()
    with (root / filename).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if summary is not None:
        (root / "summary.json").write_text(json.dumps(summary))


def test_run_joins_by_scene_and_geometry_hash(tmp_path):
    attribute_root = tmp_path / "attributes"
    semantic_root = tmp_path / "semantics"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"network2d": {"text_prompts": ["chair", "table"]}}))
    attributes = [
        {"task_id": "a", "scene_name": "scene_a", "plan_index": 0, "plan_key": "a", "geometry_key": "a", "visual_geometry_key": "a:g", "geometry_hash": "same", "attribute_extraction_completed": False},
        {"task_id": "b", "scene_name": "scene_b", "plan_index": 1, "plan_key": "b", "geometry_key": "b", "visual_geometry_key": "b:g", "geometry_hash": "same", "attribute_extraction_completed": False},
    ]
    semantics = [
        {"scene_name": "scene_a", "plan_key": "a", "geometry_key": "a", "visual_geometry_key": "a:g", "geometry_hash": "same", "geometry_locator_read_only": {}, "candidate_source": "native", "frozen_class_index": 0, "challenger_score": 0.1, "append_only": False, "canonical_frozen_class_index": 0, "canonical_frozen_score": 0.1, "finite_class_hypotheses": [{"class_index": 0}]},
        {"scene_name": "scene_b", "plan_key": "b", "geometry_key": "b", "visual_geometry_key": "b:g", "geometry_hash": "same", "geometry_locator_read_only": {}, "candidate_source": "native", "frozen_class_index": 1, "challenger_score": 0.2, "append_only": False, "canonical_frozen_class_index": 1, "canonical_frozen_score": 0.2, "finite_class_hypotheses": [{"class_index": 1}]},
    ]
    _write_root(attribute_root, "attribute_extraction_manifest.jsonl", attributes)
    _write_root(semantic_root, "semantic_arbitration_manifest.jsonl", semantics)
    result = run(Namespace(attribute_root=attribute_root, semantic_root=semantic_root, output_root=output_root, config_path=config_path))
    rows = [json.loads(line) for line in (output_root / "candidate_evidence_manifest.jsonl").read_text().splitlines()]
    assert result["geometry_count"] == 2
    assert [row["canonical_frozen_class_index"] for row in rows] == [0, 1]


def test_run_rejects_duplicate_scene_geometry_identity(tmp_path):
    attribute_root = tmp_path / "attributes"
    semantic_root = tmp_path / "semantics"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"network2d": {"text_prompts": ["chair"]}}))
    attribute = {"task_id": "a", "scene_name": "scene", "plan_index": 0, "plan_key": "p", "geometry_key": "p", "visual_geometry_key": "g", "geometry_hash": "same", "attribute_extraction_completed": False}
    semantic = {"scene_name": "scene", "plan_key": "p", "geometry_key": "p", "visual_geometry_key": "g", "geometry_hash": "same", "geometry_locator_read_only": {}, "candidate_source": "native", "frozen_class_index": 0, "challenger_score": 0.1, "append_only": False, "canonical_frozen_class_index": 0, "canonical_frozen_score": 0.1, "finite_class_hypotheses": [{"class_index": 0}]}
    _write_root(attribute_root, "attribute_extraction_manifest.jsonl", [attribute, dict(attribute, task_id="b")])
    _write_root(semantic_root, "semantic_arbitration_manifest.jsonl", [semantic])
    with pytest.raises(ValueError, match="duplicate scene/plan identities"):
        run(Namespace(attribute_root=attribute_root, semantic_root=semantic_root, output_root=output_root, config_path=config_path))

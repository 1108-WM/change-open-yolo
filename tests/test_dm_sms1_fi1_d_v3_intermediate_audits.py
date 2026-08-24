from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audit_dm_sms1_attribute_extraction_manifest import audit as audit_attribute
from tools.audit_dm_sms1_candidate_evidence_manifest import audit as audit_candidate
from tools.audit_dm_sms1_semantic_arbitration_manifest import audit as audit_semantic
from tools.build_dm_sms1_attribute_extraction_manifest import build_attribute_row
from tools.build_dm_sms1_candidate_evidence_manifest import build_candidate_row
from tools.run_dm_sms1_fi1_d_v3_val312_pipeline import _commands, _outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assets(root: Path) -> dict[str, str]:
    root.mkdir(parents=True)
    values = {}
    for name in ("rgb.jpg", "depth.png", "pose.txt", "intrinsics.txt"):
        path = root / name
        path.write_bytes(name.encode())
        values[name] = str(path)
    return values


def _fixture(tmp_path: Path) -> dict[str, Path]:
    scene = "scene_smoke"
    digest = "geometry-hash"
    locator = {"kind": "synthetic", "points": [0, 1]}
    members = [
        {
            "plan_index": 0, "plan_key": "plan-a", "fi1_d_v3_plan_key": "plan-a",
            "candidate_source": "native", "frozen_class_index": 1,
            "challenger_score": 0.9, "frozen_score": 0.9,
            "append_only": False, "fi1_d_v3_append_only": False,
            "geometry_locator_read_only": locator, "geometry_locator": locator,
        },
        {
            "plan_index": 1, "plan_key": "plan-b", "fi1_d_v3_plan_key": "plan-b",
            "candidate_source": "refined_union", "frozen_class_index": 2,
            "challenger_score": 0.8, "frozen_score": 0.8,
            "append_only": True, "fi1_d_v3_append_only": True,
            "geometry_locator_read_only": locator, "geometry_locator": locator,
        },
    ]
    joint_row = {
        "scene_name": scene, "geometry_key": f"{scene}:visual:{digest}",
        "geometry_hash": digest, "point_count": 2,
        "member_count": 2, "members": members,
    }
    joint_root = tmp_path / "joint"
    _write_jsonl(joint_root / "unique_geometry_ledger.jsonl", [joint_row])

    assets = _assets(tmp_path / "assets")
    alpha_row = {
        **joint_row,
        "alpha_class_index": 3, "alpha_top_similarity": 0.75, "sms_keep": True,
    }
    alpha_root = tmp_path / "alpha"
    _write_jsonl(alpha_root / "scenes" / scene / "records.jsonl", [alpha_row])
    (alpha_root / "summary.json").write_text(json.dumps({"geometry_count": 1, "member_count": 2}))

    semantic_rows = []
    for member in members:
        frozen = int(member["frozen_class_index"])
        semantic_rows.append({
            "scene_name": scene,
            "plan_index": member["plan_index"],
            "plan_key": member["plan_key"],
            "fi1_d_v3_plan_key": member["plan_key"],
            "geometry_key": member["plan_key"],
            "visual_geometry_key": joint_row["geometry_key"],
            "geometry_hash": digest,
            "point_count": 2,
            "geometry_locator_read_only": locator,
            "candidate_source": member["candidate_source"],
            "canonical_candidate_source": member["candidate_source"],
            "frozen_class_index": frozen,
            "canonical_frozen_class_index": frozen,
            "challenger_score": member["challenger_score"],
            "canonical_frozen_score": member["challenger_score"],
            "append_only": member["append_only"],
            "fi1_d_v3_append_only": member["append_only"],
            "alpha_class_index": 3,
            "alpha_top_similarity": 0.75,
            "sms_keep": True,
            "finite_class_hypotheses": [
                {"class_index": frozen, "sources": ["frozen_control"]},
                {"class_index": 3, "sources": ["alpha_main"]},
            ],
            "selected_views": [{
                "selection_rank": 0, "frame_id": "0", "frame_index": 0,
                "visible_ratio": 1.0, "visible_point_count": 2,
                "rgb_path": assets["rgb.jpg"], "depth_path": assets["depth.png"],
                "pose_path": assets["pose.txt"], "intrinsics_path": assets["intrinsics.txt"],
                "sam_box_prompt_xyxy": [0, 0, 1, 1], "sam_mask_sha256": "a" * 64,
                "sam_mask_valid": True, "sam_mask_area": 2,
            }],
            "candidate_retained": True, "candidate_deletion": False,
            "candidate_mutation": False, "geometry_mutation": False,
            "class_mutation": False, "score_mutation": False,
            "class_decision_made": False, "ground_truth_read": False, "ap_computed": False,
        })
    semantic_root = tmp_path / "semantic"
    _write_jsonl(semantic_root / "semantic_arbitration_manifest.jsonl", semantic_rows)
    (semantic_root / "summary.json").write_text(json.dumps({
        "candidate_count": 2, "unique_geometry_count": 1, "candidate_deletion_count": 0,
        "selected_view_count": 2, "candidate_hypothesis_count": 4,
        "ground_truth_read": False, "ap_computed": False,
    }))

    attribute_rows = [build_attribute_row(row) for row in semantic_rows]
    attribute_root = tmp_path / "attribute"
    _write_jsonl(attribute_root / "attribute_extraction_manifest.jsonl", attribute_rows)
    (attribute_root / "summary.json").write_text(json.dumps({
        "task_count": 2, "candidate_count": 2, "unique_geometry_count": 1,
        "candidate_deletion_count": 0, "view_input_count": 2,
        "candidate_labels_hidden": True, "ground_truth_read": False, "ap_computed": False,
    }))

    class_names = [f"class-{index}" for index in range(198)]
    candidate_rows = [
        build_candidate_row(attribute_rows[index], semantic_rows[index], class_names)
        for index in range(2)
    ]
    candidate_root = tmp_path / "candidate"
    _write_jsonl(candidate_root / "candidate_evidence_manifest.jsonl", candidate_rows)
    (candidate_root / "summary.json").write_text(json.dumps({
        "candidate_count": 2, "unique_geometry_count": 1, "candidate_deletion_count": 0,
        "candidate_pair_count": 2, "single_candidate_count": 0,
        "class_decision_made": False, "selected_class_count": 0,
    }))
    return {
        "joint": joint_root, "alpha": alpha_root, "semantic": semantic_root,
        "attribute": attribute_root, "candidate": candidate_root,
    }


def test_intermediate_audits_accept_complete_duplicate_safe_chain(tmp_path: Path):
    roots = _fixture(tmp_path)
    assert audit_semantic(roots["semantic"], roots["alpha"], roots["joint"], 2, 1)["audit_valid"]
    assert audit_attribute(roots["attribute"], roots["semantic"], 2, 1)["audit_valid"]
    assert audit_candidate(roots["candidate"], roots["attribute"], roots["semantic"], 2, 1)["audit_valid"]


@pytest.mark.parametrize(
    "mutation",
    ["tamper_score", "tamper_locator", "tamper_source", "omit", "reorder"],
)
def test_semantic_audit_rejects_tamper_omission_and_reorder(tmp_path: Path, mutation: str):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["semantic"] / "semantic_arbitration_manifest.jsonl")
    if mutation == "tamper_score":
        rows[0]["challenger_score"] = 0.1
    elif mutation == "tamper_locator":
        rows[0]["geometry_locator_read_only"] = {"kind": "tampered"}
    elif mutation == "tamper_source":
        rows[0]["candidate_source"] = "tampered"
    elif mutation == "omit":
        rows.pop()
    else:
        rows.reverse()
    _write_jsonl(roots["semantic"] / "semantic_arbitration_manifest.jsonl", rows)
    result = audit_semantic(roots["semantic"], roots["alpha"], roots["joint"], 2, 1)
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


@pytest.mark.parametrize("mutation", ["tamper", "tamper_view", "omit", "reorder"])
def test_attribute_audit_rejects_tamper_omission_and_reorder(tmp_path: Path, mutation: str):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["attribute"] / "attribute_extraction_manifest.jsonl")
    if mutation == "tamper":
        rows[0]["geometry_hash"] = "tampered"
    elif mutation == "tamper_view":
        rows[0]["view_inputs"][0]["frame_id"] = "tampered"
    elif mutation == "omit":
        rows.pop()
    else:
        rows.reverse()
    _write_jsonl(roots["attribute"] / "attribute_extraction_manifest.jsonl", rows)
    result = audit_attribute(roots["attribute"], roots["semantic"], 2, 1)
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


@pytest.mark.parametrize("mutation", ["tamper", "tamper_index", "omit", "reorder"])
def test_candidate_audit_rejects_tamper_omission_and_reorder(tmp_path: Path, mutation: str):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["candidate"] / "candidate_evidence_manifest.jsonl")
    if mutation == "tamper":
        rows[0]["frozen_class_index"] = 99
    elif mutation == "tamper_index":
        rows[0]["plan_index"] = 99
    elif mutation == "omit":
        rows.pop()
    else:
        rows.reverse()
    _write_jsonl(roots["candidate"] / "candidate_evidence_manifest.jsonl", rows)
    result = audit_candidate(roots["candidate"], roots["attribute"], roots["semantic"], 2, 1)
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


def test_pipeline_passes_independent_upstream_audit_inputs(tmp_path: Path):
    outputs = _outputs(tmp_path / "fresh_duplicate_safe_run")
    config_keys = (
        "scene_list", "prepared_root", "legacy_unique_geometry_root",
        "fi1_d_v3_inference_root", "fi1_d_v3_inference_audit_root",
        "fi1_d_v3_ap_result_root", "fi1_d_v3_ap_audit_root", "config_path",
        "asset_provenance", "alpha_clip_source", "alpha_clip_base",
        "alpha_clip_checkpoint", "sam_source", "sam_checkpoint", "qwen_model_dir",
        "ground_truth_root", "run_root", "preregistration_path",
    )
    config = {key: tmp_path / key for key in config_keys}
    commands = _commands(
        "manifests",
        config,
        outputs,
        authorize_ap=False,
    )
    semantic_audit, attribute_audit, candidate_audit = commands[1], commands[3], commands[5]
    assert semantic_audit[semantic_audit.index("--alpha-ledger-root") + 1] == str(
        outputs["alpha_embeddings"]
    )
    assert semantic_audit[semantic_audit.index("--joint-geometry-root") + 1] == str(
        outputs["geometry"]
    )
    assert attribute_audit[attribute_audit.index("--semantic-root") + 1] == str(
        outputs["semantic_manifest"]
    )
    assert candidate_audit[candidate_audit.index("--attribute-root") + 1] == str(
        outputs["attribute_manifest"]
    )
    assert candidate_audit[candidate_audit.index("--semantic-root") + 1] == str(
        outputs["semantic_manifest"]
    )
    for command in (semantic_audit, attribute_audit, candidate_audit):
        assert command[command.index("--expected-candidate-count") + 1] == "39304"
        assert command[command.index("--expected-unique-geometry-count") + 1] == "39250"


def test_example_paths_use_fresh_duplicate_safe_run_root():
    path = Path(__file__).resolve().parents[1] / "docs/DM_SMS1_FI1_D_V3_VAL312_PATHS.example.json"
    config = json.loads(path.read_text())
    assert config["run_root"].endswith(
        "/dm_sms1_fi1_d_v3_val312_duplicate_safe_20260824"
    )
    assert not config["run_root"].endswith("/dm_sms1_fi1_d_v3_val312_joint_20260824")

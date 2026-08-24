from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools.audit_dm_sms1_attribute_extraction_manifest import audit as audit_attribute
from tools.audit_dm_sms1_candidate_evidence_manifest import audit as audit_candidate
from tools.audit_dm_sms1_semantic_arbitration_manifest import audit as audit_semantic
from tools.build_dm_sms1_attribute_extraction_manifest import build_attribute_row
from tools.build_dm_sms1_candidate_evidence_manifest import build_candidate_row
from tools.build_dm_sms1_semantic_arbitration_manifest import build_rows
from tools.run_dm_sms1_fi1_d_v3_val312_pipeline import _commands, _outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _alpha_views(root: Path) -> list[dict]:
    scene_root = root / "scene_smoke"
    for name in ("rgb", "depth", "poses"):
        (scene_root / name).mkdir(parents=True, exist_ok=True)
    (scene_root / "intrinsics.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    views = []
    for index, (center, visible_ratio) in enumerate(((0, 0.9), (1, 0.8), (2, 0.7), (4, 0.6))):
        frame_id = str(index)
        rgb = scene_root / "rgb" / f"{frame_id}.jpg"
        depth = scene_root / "depth" / f"{frame_id}.png"
        pose = scene_root / "poses" / f"{frame_id}.txt"
        rgb.write_bytes(f"rgb-{index}".encode())
        depth.write_bytes(f"depth-{index}".encode())
        pose.write_text(
            f"1 0 0 {center}\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        )
        views.append({
            "view_rank": index,
            "frame_index": index,
            "frame_id": frame_id,
            "rgb_path": str(rgb),
            "visible_point_count": 20 - index,
            "visible_ratio": visible_ratio,
            "sam_box_prompt_xyxy": [index, index + 1, index + 2, index + 3],
            "sam_mask_sha256": str(index) * 64,
            "sam_mask_valid": True,
            "sam_mask_area": 100 + index,
        })
    return views


def _fixture(tmp_path: Path) -> dict[str, Path]:
    scene = "scene_smoke"
    digest = "geometry-hash"
    locator = {"kind": "synthetic", "points": [0, 1]}
    members = [
        {
            "plan_index": 0, "plan_key": "plan-a", "fi1_d_v3_plan_key": "plan-a",
            "candidate_id": 10, "candidate_source": "native", "frozen_class_index": 3,
            "challenger_score": 0.9, "frozen_score": 0.9,
            "point_count": 2,
            "append_only": False, "fi1_d_v3_append_only": False,
            "geometry_locator_read_only": locator, "geometry_locator": locator,
        },
        {
            "plan_index": 1, "plan_key": "plan-b", "fi1_d_v3_plan_key": "plan-b",
            "candidate_id": 11, "candidate_source": "refined_union", "frozen_class_index": 2,
            "challenger_score": 0.8, "frozen_score": 0.8,
            "point_count": 2,
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

    alpha_row = {
        **joint_row,
        "alpha_class_index": 3, "alpha_top_similarity": 0.75, "sms_keep": True,
        "views": _alpha_views(tmp_path / "assets"),
    }
    alpha_root = tmp_path / "alpha"
    _write_jsonl(alpha_root / "scenes" / scene / "records.jsonl", [alpha_row])
    (alpha_root / "summary.json").write_text(json.dumps({"geometry_count": 1, "member_count": 2}))

    semantic_rows = build_rows(alpha_row, target_count=3, max_input_views=20)
    semantic_root = tmp_path / "semantic"
    _write_jsonl(semantic_root / "semantic_arbitration_manifest.jsonl", semantic_rows)
    (semantic_root / "summary.json").write_text(json.dumps({
        "candidate_count": 2, "unique_geometry_count": 1, "candidate_deletion_count": 0,
        "selected_view_count": 6,
        "candidate_hypothesis_count": sum(
            len(row["finite_class_hypotheses"]) for row in semantic_rows
        ),
        "target_views": 3, "max_input_views": 20,
        "ground_truth_read": False, "ap_computed": False,
    }))

    attribute_rows = [build_attribute_row(row) for row in semantic_rows]
    attribute_root = tmp_path / "attribute"
    _write_jsonl(attribute_root / "attribute_extraction_manifest.jsonl", attribute_rows)
    (attribute_root / "summary.json").write_text(json.dumps({
        "task_count": 2, "candidate_count": 2, "unique_geometry_count": 1,
        "candidate_deletion_count": 0, "view_input_count": 6,
        "candidate_labels_hidden": True, "ground_truth_read": False, "ap_computed": False,
    }))

    class_names = [f"class-{index}" for index in range(198)]
    config_path = tmp_path / "config_scannet200.yaml"
    config_path.write_text(yaml.safe_dump({"network2d": {"text_prompts": class_names}}))
    candidate_rows = [
        build_candidate_row(attribute_rows[index], semantic_rows[index], class_names)
        for index in range(2)
    ]
    candidate_root = tmp_path / "candidate"
    _write_jsonl(candidate_root / "candidate_evidence_manifest.jsonl", candidate_rows)
    (candidate_root / "summary.json").write_text(json.dumps({
        "candidate_count": 2, "unique_geometry_count": 1, "candidate_deletion_count": 0,
        "candidate_pair_count": sum(len(row["candidate_hypotheses"]) == 2 for row in candidate_rows),
        "single_candidate_count": sum(len(row["candidate_hypotheses"]) == 1 for row in candidate_rows),
        "class_decision_made": False, "selected_class_count": 0,
        "input_provenance": {
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    }))
    return {
        "joint": joint_root, "alpha": alpha_root, "semantic": semantic_root,
        "attribute": attribute_root, "candidate": candidate_root, "config": config_path,
    }


def test_intermediate_audits_accept_complete_duplicate_safe_chain(tmp_path: Path):
    roots = _fixture(tmp_path)
    assert audit_semantic(roots["semantic"], roots["alpha"], roots["joint"], 2, 1)["audit_valid"]
    assert audit_attribute(roots["attribute"], roots["semantic"], 2, 1)["audit_valid"]
    assert audit_candidate(
        roots["candidate"], roots["attribute"], roots["semantic"], 2, 1, roots["config"]
    )["audit_valid"]


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


@pytest.mark.parametrize(
    "field",
    [
        "selection_rank", "source_view_index", "frame_id", "frame_index",
        "visible_ratio", "visible_point_count", "rgb_path", "depth_path", "pose_path",
        "intrinsics_path", "sam_box_prompt_xyxy", "sam_mask_sha256", "sam_mask_valid",
        "sam_mask_area", "view_selection_reason",
    ],
)
def test_semantic_audit_rejects_each_selected_view_field_tamper(tmp_path: Path, field: str):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["semantic"] / "semantic_arbitration_manifest.jsonl")
    view = rows[0]["selected_views"][0]
    value = view[field]
    if isinstance(value, bool):
        view[field] = not value
    elif isinstance(value, int):
        view[field] = value + 1
    elif isinstance(value, float):
        view[field] = value + 0.01
    elif isinstance(value, list):
        view[field] = value + [999]
    else:
        view[field] = str(value) + "-tampered"
    _write_jsonl(roots["semantic"] / "semantic_arbitration_manifest.jsonl", rows)
    result = audit_semantic(roots["semantic"], roots["alpha"], roots["joint"], 2, 1)
    assert result["audit_valid"] is False


@pytest.mark.parametrize(
    "mutation",
    ["finite_class", "finite_sources", "point_count", "candidate_id", "shared_member_count"],
)
def test_semantic_audit_rejects_finite_candidate_and_member_tamper(
    tmp_path: Path, mutation: str,
):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["semantic"] / "semantic_arbitration_manifest.jsonl")
    if mutation == "finite_class":
        rows[1]["finite_class_hypotheses"][0]["class_index"] = 99
    elif mutation == "finite_sources":
        rows[0]["finite_class_hypotheses"][0]["sources"] = ["frozen_control"]
    elif mutation == "point_count":
        rows[0]["point_count"] += 1
    elif mutation == "candidate_id":
        rows[0]["canonical_candidate_id"] += 1
    else:
        rows[0]["visual_evidence_shared_member_count"] -= 1
    _write_jsonl(roots["semantic"] / "semantic_arbitration_manifest.jsonl", rows)
    result = audit_semantic(roots["semantic"], roots["alpha"], roots["joint"], 2, 1)
    assert result["audit_valid"] is False


def test_semantic_audit_rejects_alpha_member_count_tamper(tmp_path: Path):
    roots = _fixture(tmp_path)
    path = roots["alpha"] / "scenes/scene_smoke/records.jsonl"
    rows = _read_jsonl(path)
    rows[0]["member_count"] = 1
    _write_jsonl(path, rows)
    assert not audit_semantic(roots["semantic"], roots["alpha"], roots["joint"], 2, 1)[
        "audit_valid"
    ]


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


@pytest.mark.parametrize(
    "mutation",
    ["task_id", "prompt", "response_schema"],
)
def test_attribute_audit_rejects_exact_contract_tamper(tmp_path: Path, mutation: str):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["attribute"] / "attribute_extraction_manifest.jsonl")
    if mutation == "task_id":
        rows[0]["task_id"] += "-tampered"
    elif mutation == "prompt":
        rows[0]["attribute_prompt"] += " "
    else:
        rows[0]["response_schema"]["appearance"]["observation"] = "tampered"
    _write_jsonl(roots["attribute"] / "attribute_extraction_manifest.jsonl", rows)
    assert not audit_attribute(roots["attribute"], roots["semantic"], 2, 1)["audit_valid"]


@pytest.mark.parametrize(
    "field",
    [
        "view_rank", "frame_id", "frame_index", "rgb_path", "depth_path", "pose_path",
        "intrinsics_path", "sam_box_prompt_xyxy", "sam_mask_sha256", "visible_ratio",
        "visible_point_count",
    ],
)
def test_attribute_audit_rejects_each_view_input_field_tamper(tmp_path: Path, field: str):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["attribute"] / "attribute_extraction_manifest.jsonl")
    view = rows[0]["view_inputs"][0]
    value = view[field]
    if isinstance(value, int):
        view[field] = value + 1
    elif isinstance(value, float):
        view[field] = value + 0.01
    elif isinstance(value, list):
        view[field] = value + [999]
    else:
        view[field] = str(value) + "-tampered"
    _write_jsonl(roots["attribute"] / "attribute_extraction_manifest.jsonl", rows)
    assert not audit_attribute(roots["attribute"], roots["semantic"], 2, 1)["audit_valid"]


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
    result = audit_candidate(
        roots["candidate"], roots["attribute"], roots["semantic"], 2, 1, roots["config"]
    )
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


@pytest.mark.parametrize(
    "mutation",
    [
        "class_index", "class_name", "order_ab", "order_ba", "prompt_ab", "prompt_ba",
        "attribute_switch", "swap_switch", "decision_rule",
    ],
)
def test_candidate_audit_rejects_mapping_prompt_and_switch_tamper(
    tmp_path: Path, mutation: str,
):
    roots = _fixture(tmp_path)
    rows = _read_jsonl(roots["candidate"] / "candidate_evidence_manifest.jsonl")
    row = rows[1]
    if mutation == "class_index":
        row["candidate_hypotheses"][0]["class_index"] = 99
    elif mutation == "class_name":
        row["candidate_hypotheses"][0]["class_name"] = "tampered"
    elif mutation == "order_ab":
        row["candidate_order_ab"].reverse()
    elif mutation == "order_ba":
        row["candidate_order_ba"].reverse()
    elif mutation == "prompt_ab":
        row["evidence_prompt_ab"] += " "
    elif mutation == "prompt_ba":
        row["evidence_prompt_ba"] += " "
    elif mutation == "attribute_switch":
        row["attribute_evidence_required"] = False
    elif mutation == "swap_switch":
        row["swap_order_required"] = False
    else:
        row["decision_rule"]["no_score_change"] = False
    _write_jsonl(roots["candidate"] / "candidate_evidence_manifest.jsonl", rows)
    result = audit_candidate(
        roots["candidate"], roots["attribute"], roots["semantic"], 2, 1, roots["config"]
    )
    assert result["audit_valid"] is False


def test_candidate_audit_rejects_frozen_config_mapping_tamper(tmp_path: Path):
    roots = _fixture(tmp_path)
    config = yaml.safe_load(roots["config"].read_text())
    config["network2d"]["text_prompts"][197] = "tampered-unused-class-name"
    roots["config"].write_text(yaml.safe_dump(config))
    result = audit_candidate(
        roots["candidate"], roots["attribute"], roots["semantic"], 2, 1, roots["config"]
    )
    assert result["audit_valid"] is False


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
    assert candidate_audit[candidate_audit.index("--config-path") + 1] == str(
        config["config_path"]
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

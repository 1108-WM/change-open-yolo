from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.audit_dm_sms1_attribute_extraction_manifest import audit as audit_attribute
from tools.audit_dm_sms1_candidate_evidence_manifest import audit as audit_candidate
from tools.audit_dm_sms1_fi1_d_v3_prediction_cache import run as audit_cache
from tools.audit_dm_sms1_full_safe_decision_ledger import audit as audit_full
from tools.audit_dm_sms1_semantic_arbitration_manifest import audit as audit_semantic
from tools.audit_dm_sms1_vlm_batch_outputs import audit as audit_qwen
from tools.build_dm_sms1_attribute_extraction_manifest import build_attribute_row
from tools.build_dm_sms1_candidate_evidence_manifest import build_candidate_row
from tools.build_dm_sms1_fi1_d_v3_prediction_cache import run as build_cache
from tools.build_dm_sms1_full_safe_decision_ledger import run as build_full
from tools.build_dm_sms1_semantic_arbitration_manifest import build_rows
from tools.dm_sms1_terminal_safe_keep import (
    TERMINAL_KEEP_REASON,
    terminal_expected_identities,
    terminal_identity,
)
from tools.dm_sms_core import geometry_hash
from tools.run_dm_sms1_vlm_batch_smoke import _validate_qwen_selection, select_batch
from tools.run_dm_sms1_fi1_d_v3_val312_pipeline import (
    TERMINAL_SAFE_KEEP_PREREGISTRATION,
    _commands,
    _outputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _terminal_fixture(tmp_path: Path) -> dict[str, Path]:
    identities = sorted(terminal_expected_identities())
    joint_rows = []
    alpha_rows_by_scene: dict[str, list[dict]] = {}
    scene_names = []
    prepared_root = tmp_path / "prepared"
    for offset, (plan_index, plan_key) in enumerate(identities):
        scene = plan_key.split(":", 1)[0]
        scene_names.append(scene)
        points = np.asarray([offset + 1, offset + 3], dtype=np.int64)
        locator_path = tmp_path / f"points-{offset}.npz"
        np.savez_compressed(locator_path, point_indices=points)
        locator = {
            "kind": "point_indices_npz", "points_path": str(locator_path),
            "array_key": "point_indices",
        }
        digest = geometry_hash(points)
        member = {
            "plan_index": plan_index, "plan_key": plan_key,
            "fi1_d_v3_plan_key": plan_key, "candidate_id": offset,
            "candidate_source": "native", "frozen_class_index": 198,
            "frozen_class_valid": False, "challenger_score": 0.1 + offset / 10,
            "frozen_score": 0.1 + offset / 10, "point_count": len(points),
            "append_only": False, "fi1_d_v3_append_only": False,
            "candidate_retained": True, "candidate_deletion": False,
            "geometry_locator_read_only": locator, "geometry_locator": locator,
            "geometry_hash": digest, "geometry_mutation": False,
            "class_mutation": False, "score_mutation": False,
        }
        geometry = {
            "scene_name": scene, "geometry_key": f"{scene}:visual_geometry:{digest}",
            "geometry_hash": digest, "point_count": len(points),
            "member_count": 1, "members": [member],
        }
        joint_rows.append(geometry)
        alpha_rows_by_scene[scene] = [{
            **geometry, "selected_view_count": 0, "views": [],
            "alpha_feature_valid": False, "alpha_class_index": None,
            "alpha_top_similarity": None, "sms_keep": True,
        }]
        prepared_scene = prepared_root / scene
        prepared_scene.mkdir(parents=True)
        np.save(prepared_scene / f"{scene.removeprefix('scene')}.npy", np.zeros((8, 4)))

    joint_root = tmp_path / "joint"
    _write_jsonl(joint_root / "unique_geometry_ledger.jsonl", joint_rows)
    _write_summary(joint_root / "summary.json", {
        "scene_count": 4, "contract_valid": True,
        "ground_truth_read": False, "ap_computed": False,
    })
    geometry_audit = tmp_path / "geometry_audit"
    _write_summary(geometry_audit / "summary.json", {"audit_valid": True, "error_count": 0})

    alpha_root = tmp_path / "alpha"
    for scene, rows in alpha_rows_by_scene.items():
        _write_jsonl(alpha_root / "scenes" / scene / "records.jsonl", rows)
    _write_summary(alpha_root / "summary.json", {
        "geometry_count": 4, "member_count": 4,
        "ground_truth_read": False, "ap_computed": False,
    })

    semantic_rows = []
    for rows in alpha_rows_by_scene.values():
        semantic_rows.extend(build_rows(rows[0], target_count=3, max_input_views=20))
    semantic_rows.sort(key=lambda row: row["plan_index"])
    semantic_root = tmp_path / "semantic"
    _write_jsonl(semantic_root / "semantic_arbitration_manifest.jsonl", semantic_rows)
    _write_summary(semantic_root / "summary.json", {
        "candidate_count": 4, "unique_geometry_count": 4,
        "candidate_deletion_count": 0, "selected_view_count": 0,
        "candidate_hypothesis_count": 0, "terminal_safe_keep_count": 4,
        "target_views": 3, "max_input_views": 20,
        "ground_truth_read": False, "ap_computed": False,
    })

    attribute_rows = [build_attribute_row(row) for row in semantic_rows]
    attribute_root = tmp_path / "attribute"
    _write_jsonl(attribute_root / "attribute_extraction_manifest.jsonl", attribute_rows)
    _write_summary(attribute_root / "summary.json", {
        "task_count": 4, "candidate_count": 4, "unique_geometry_count": 4,
        "candidate_deletion_count": 0, "view_input_count": 0,
        "terminal_safe_keep_count": 4, "candidate_labels_hidden": True,
        "ground_truth_read": False, "ap_computed": False,
    })

    class_names = [f"class-{index}" for index in range(198)]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"network2d": {"text_prompts": class_names}}))
    candidate_rows = [
        build_candidate_row(attribute, semantic, class_names)
        for attribute, semantic in zip(attribute_rows, semantic_rows)
    ]
    candidate_root = tmp_path / "candidate"
    candidate_path = candidate_root / "candidate_evidence_manifest.jsonl"
    _write_jsonl(candidate_path, candidate_rows)
    _write_summary(candidate_root / "summary.json", {
        "candidate_count": 4, "unique_geometry_count": 4,
        "candidate_deletion_count": 0, "candidate_pair_count": 0,
        "single_candidate_count": 0, "terminal_safe_keep_count": 4,
        "class_decision_made": False, "selected_class_count": 0,
        "input_provenance": {"config_sha256": _sha256(config)},
    })

    pair_path = tmp_path / "pair_decisions.jsonl"
    pair_path.write_text("")
    full_root = tmp_path / "full"
    build_full(pair_path, candidate_path, full_root)

    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(sorted(scene_names)) + "\n")
    cache_root = tmp_path / "cache"
    build_cache(argparse.Namespace(
        scene_list=scene_list, ledger_root=joint_root,
        ledger_audit_root=geometry_audit, prepared_root=prepared_root,
        output_root=cache_root, expected_scene_count=4, dataset_name="synthetic",
    ))
    return {
        "joint": joint_root, "geometry_audit": geometry_audit, "alpha": alpha_root,
        "semantic": semantic_root, "attribute": attribute_root,
        "candidate": candidate_root, "candidate_path": candidate_path,
        "config": config, "pair": pair_path, "full": full_root,
        "scene_list": scene_list, "prepared": prepared_root, "cache": cache_root,
    }


def _audit_chain(roots: dict[str, Path]) -> None:
    expected = terminal_expected_identities()
    assert audit_semantic(
        roots["semantic"], roots["alpha"], roots["joint"], 4, 4, expected
    )["audit_valid"]
    assert audit_attribute(roots["attribute"], roots["semantic"], 4, 4, expected)["audit_valid"]
    assert audit_candidate(
        roots["candidate"], roots["attribute"], roots["semantic"], 4, 4,
        roots["config"], expected,
    )["audit_valid"]
    assert audit_full(roots["full"], roots["candidate_path"], expected)["audit_valid"]
    result = audit_cache(argparse.Namespace(
        cache_root=roots["cache"], ledger_root=roots["joint"],
        decision_root=roots["full"], output_root=roots["cache"].parent / "cache_audit",
        expected_candidate_count=4, expected_unique_geometry_count=4,
        expected_terminal_identities=expected,
    ))
    assert result["audit_valid"]
    assert result["terminal_safe_keep_count"] == 4


def test_terminal_safe_keep_complete_synthetic_chain(tmp_path: Path):
    roots = _terminal_fixture(tmp_path)
    _audit_chain(roots)
    semantic = _rows(roots["semantic"] / "semantic_arbitration_manifest.jsonl")
    assert all(row["finite_class_hypotheses"] == [] for row in semantic)
    assert all(row["selected_views"] == [] for row in semantic)
    candidates = _rows(roots["candidate_path"])
    assert select_batch(candidates, scene_count=4, per_scene=1) == []
    decisions = _rows(roots["full"] / "safe_decisions.jsonl")
    assert all(row["arbitrated_class_index"] == 198 for row in decisions)
    assert all(row["decision_source"] == "terminal_safe_keep" for row in decisions)


def test_terminal_identity_rejects_cross_paired_plan_index_and_key():
    identities = sorted(terminal_expected_identities())
    assert terminal_identity(*identities[0])
    assert not terminal_identity(identities[0][0], identities[1][1])


def test_pipeline_preflight_freezes_terminal_preregistration_path(tmp_path: Path):
    config_keys = (
        "scene_list", "prepared_root", "legacy_unique_geometry_root",
        "fi1_d_v3_inference_root", "fi1_d_v3_inference_audit_root",
        "fi1_d_v3_ap_result_root", "fi1_d_v3_ap_audit_root", "config_path",
        "asset_provenance", "alpha_clip_source", "alpha_clip_base",
        "alpha_clip_checkpoint", "sam_source", "sam_checkpoint", "qwen_model_dir",
        "ground_truth_root", "run_root", "preregistration_path",
    )
    config = {key: tmp_path / key for key in config_keys}
    command = _commands("preflight", config, _outputs(config["run_root"]), False)[0]
    index = command.index("--terminal-safe-keep-preregistration-path")
    assert command[index + 1] == str(TERMINAL_SAFE_KEEP_PREREGISTRATION)


@pytest.mark.parametrize(
    "mutation",
    ["ordinary_marked_terminal", "omit", "view", "alpha_class", "finite_candidate", "count"],
)
def test_semantic_audit_rejects_terminal_contract_tamper(tmp_path: Path, mutation: str):
    roots = _terminal_fixture(tmp_path)
    path = roots["semantic"] / "semantic_arbitration_manifest.jsonl"
    rows = _rows(path)
    if mutation == "ordinary_marked_terminal":
        rows[0]["plan_index"] = 100
        rows[0]["plan_key"] = "scene0019_01:ordinary"
        rows[0]["fi1_d_v3_plan_key"] = rows[0]["plan_key"]
        rows[0]["geometry_key"] = rows[0]["plan_key"]
    elif mutation == "omit":
        rows.pop()
    elif mutation == "view":
        rows[0]["selected_views"] = [{"frame_id": "forged"}]
    elif mutation == "alpha_class":
        rows[0]["alpha_class_index"] = 1
    elif mutation == "finite_candidate":
        rows[0]["finite_class_hypotheses"] = [{"class_index": 1, "sources": ["alpha_main"]}]
    else:
        summary = json.loads((roots["semantic"] / "summary.json").read_text())
        summary["terminal_safe_keep_count"] = 3
        _write_summary(roots["semantic"] / "summary.json", summary)
    _write_jsonl(path, rows)
    result = audit_semantic(
        roots["semantic"], roots["alpha"], roots["joint"], 4, 4,
        terminal_expected_identities(),
    )
    assert result["audit_valid"] is False


@pytest.mark.parametrize("mutation", ["view", "prompt", "completed", "execution"])
def test_attribute_audit_rejects_terminal_execution_or_fabrication(tmp_path: Path, mutation: str):
    roots = _terminal_fixture(tmp_path)
    path = roots["attribute"] / "attribute_extraction_manifest.jsonl"
    rows = _rows(path)
    if mutation == "view":
        rows[0]["view_inputs"] = [{"frame_id": "forged"}]
    elif mutation == "prompt":
        rows[0]["attribute_prompt"] = "forged"
    elif mutation == "completed":
        rows[0]["attribute_extraction_completed"] = True
        rows[0]["attribute_output"] = {"forged": True}
    else:
        rows[0]["attribute_execution_required"] = True
    _write_jsonl(path, rows)
    assert not audit_attribute(
        roots["attribute"], roots["semantic"], 4, 4, terminal_expected_identities()
    )["audit_valid"]


@pytest.mark.parametrize("mutation", ["class_name", "prompt", "execution", "finite_candidate"])
def test_candidate_audit_rejects_terminal_qwen_or_foreground_fields(tmp_path: Path, mutation: str):
    roots = _terminal_fixture(tmp_path)
    path = roots["candidate_path"]
    rows = _rows(path)
    if mutation in {"class_name", "finite_candidate"}:
        rows[0]["candidate_hypotheses"] = [{
            "class_index": 1, "class_name": "class-1", "sources": ["alpha_main"],
        }]
    elif mutation == "prompt":
        rows[0]["evidence_prompt_ab"] = "forged"
    else:
        rows[0]["qwen_execution_required"] = True
    _write_jsonl(path, rows)
    assert not audit_candidate(
        roots["candidate"], roots["attribute"], roots["semantic"], 4, 4,
        roots["config"], terminal_expected_identities(),
    )["audit_valid"]


def test_qwen_selection_and_audit_reject_terminal_task(tmp_path: Path):
    roots = _terminal_fixture(tmp_path)
    candidates = _rows(roots["candidate_path"])
    with pytest.raises(ValueError, match="terminal-safe-keep"):
        _validate_qwen_selection([candidates[0]])
    batch = tmp_path / "batch"
    selection = [{
        "task_id": candidates[0]["task_id"], "scene_name": candidates[0]["scene_name"],
        "plan_key": candidates[0]["plan_key"], "geometry_hash": candidates[0]["geometry_hash"],
        "candidate_indices": [], "candidate_names": [],
    }]
    _write_jsonl(batch / "selection.jsonl", selection)
    _write_jsonl(batch / "batch_outputs.jsonl", [])
    result = audit_qwen(
        batch, roots["candidate_path"],
        roots["attribute"] / "attribute_extraction_manifest.jsonl",
        tmp_path / "qwen_audit", roots["config"],
    )
    assert result["audit_valid"] is False


def test_full_decision_audit_rejects_terminal_class_change(tmp_path: Path):
    roots = _terminal_fixture(tmp_path)
    path = roots["full"] / "safe_decisions.jsonl"
    rows = _rows(path)
    rows[0]["arbitrated_class_index"] = 1
    rows[0]["class_changed"] = True
    _write_jsonl(path, rows)
    assert not audit_full(
        roots["full"], roots["candidate_path"], terminal_expected_identities()
    )["audit_valid"]


def test_full_decision_audit_rejects_terminal_reorder(tmp_path: Path):
    roots = _terminal_fixture(tmp_path)
    path = roots["full"] / "safe_decisions.jsonl"
    rows = _rows(path)
    rows.reverse()
    _write_jsonl(path, rows)
    assert not audit_full(
        roots["full"], roots["candidate_path"], terminal_expected_identities()
    )["audit_valid"]


@pytest.mark.parametrize("mutation", ["delete", "mask", "score", "source", "order"])
def test_cache_audit_rejects_terminal_column_tamper(tmp_path: Path, mutation: str):
    roots = _terminal_fixture(tmp_path)
    mutated = tmp_path / "mutated_cache"
    shutil.copytree(roots["cache"], mutated)
    scene = sorted(path.name for path in (mutated / "prediction_cache").iterdir())[0]
    scene_root = mutated / "prediction_cache" / scene
    if mutation == "mask":
        values = np.load(scene_root / "masks.npy")
        values[:, 0] = False
        values[0, 0] = True
        np.save(scene_root / "masks.npy", values, allow_pickle=False)
    elif mutation == "score":
        values = np.load(scene_root / "frozen_scores.npy")
        values[0] += 0.25
        np.save(scene_root / "frozen_scores.npy", values, allow_pickle=False)
    elif mutation == "source":
        values = json.loads((scene_root / "sources.json").read_text())
        values[0] = "track"
        (scene_root / "sources.json").write_text(json.dumps(values) + "\n")
    elif mutation == "order":
        values = json.loads((scene_root / "plan_indices.json").read_text())
        values[0] += 1
        (scene_root / "plan_indices.json").write_text(json.dumps(values) + "\n")
    else:
        for name in ("masks.npy", "frozen_classes.npy", "frozen_scores.npy"):
            values = np.load(scene_root / name)
            values = values[:, :0] if values.ndim == 2 else values[:0]
            np.save(scene_root / name, values, allow_pickle=False)
        for name in ("geometry_hashes.json", "plan_keys.json", "sources.json", "plan_indices.json"):
            (scene_root / name).write_text("[]\n")
    summary_path = scene_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    for name in summary["file_sha256"]:
        summary["file_sha256"][name] = _sha256(scene_root / name)
    _write_summary(summary_path, summary)
    result = audit_cache(argparse.Namespace(
        cache_root=mutated, ledger_root=roots["joint"], decision_root=roots["full"],
        output_root=tmp_path / f"cache_audit_{mutation}", expected_candidate_count=4,
        expected_unique_geometry_count=4,
        expected_terminal_identities=terminal_expected_identities(),
    ))
    assert result["audit_valid"] is False

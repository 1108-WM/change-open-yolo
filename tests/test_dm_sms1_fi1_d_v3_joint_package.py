from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluate.scannet200.scannet_constants import CLASS_LABELS_200, VALID_CLASS_IDS_200
from tools.audit_dm_sms1_fi1_d_v3_open_vocab_ap import _csv_metrics
from tools.audit_dm_sms1_fi1_d_v3_open_vocab_ap import run as audit_ap
from tools.audit_dm_sms1_fi1_d_v3_prediction_cache import run as audit_cache
from tools.audit_dm_sms1_fi1_d_v3_unique_geometry_ledger import run as audit_geometry
from tools.build_dm_sms1_fi1_d_v3_prediction_cache import run as build_cache
from tools.build_dm_sms1_fi1_d_v3_unique_geometry_ledger import run as build_geometry
from tools.dm_sms_core import geometry_hash
from tools.evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt import (
    AUTHORIZATION_ID,
    FrozenPredictionMapping,
    run as evaluate_ap,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npz(path: Path, points: list[int]) -> dict:
    np.savez_compressed(path, point_indices=np.asarray(points, dtype=np.int64))
    return {"kind": "point_indices_npz", "points_path": str(path), "array_key": "point_indices"}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _joint_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    scene = "scene_smoke"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")
    original_points = [0, 1]
    refined_points = [0, 2]
    original_locator = _npz(tmp_path / "original.npz", original_points)
    refined_locator = _npz(tmp_path / "refined.npz", refined_points)
    original_digest = geometry_hash(np.asarray(original_points, dtype=np.int64))
    refined_sha1 = geometry_hash(np.asarray(refined_points, dtype=np.int64))
    refined_sha256 = hashlib.sha256(np.asarray(refined_points, dtype=np.int64).tobytes()).hexdigest()

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_row = {
        "scene_name": scene,
        "geometry_key": "pair-union-key",
        "geometry_hash": original_digest,
        "point_count": len(original_points),
        "canonical_candidate_source": "pair_union",
        "canonical_candidate_id": 7,
        "canonical_frozen_class_index": 4,
        "canonical_frozen_score": 0.6,
        "canonical_geometry_locator": original_locator,
    }
    _write_jsonl(legacy_root / "unique_geometry_ledger.jsonl", [legacy_row])

    inference_root = tmp_path / "inference"
    plan_root = inference_root / "complete_plan"
    plan_root.mkdir(parents=True)
    plan = [
        {
            "plan_key": "pair-union-key", "scene_name": scene,
            "candidate_source": "pair_union", "geometry_key": "pair-union-key",
            "geometry_locator_read_only": original_locator,
            "geometry_digest": original_digest, "geometry_digest_algorithm": "sha1_point_indices",
            "point_count": len(original_points), "frozen_class_index": 4,
            "control_score": 0.6, "challenger_score": 0.55,
            "candidate_retained": True, "candidate_deletion": False,
            "geometry_mutation": False, "class_mutation": False, "append_only": False,
        },
        {
            "plan_key": "refined-key", "scene_name": scene,
            "candidate_source": "refined_union", "geometry_key": None,
            "original_union_geometry_key": "pair-union-key",
            "geometry_locator_read_only": refined_locator,
            "geometry_digest": refined_sha256, "geometry_digest_algorithm": "sha256_point_indices",
            "point_count": len(refined_points), "frozen_class_index": 4,
            "control_score": None, "challenger_score": 0.45,
            "candidate_retained": True, "candidate_deletion": False,
            "geometry_mutation": False, "class_mutation": False, "append_only": True,
        },
    ]
    plan_path = plan_root / "complete_inference_ap_plan.jsonl"
    _write_jsonl(plan_path, plan)
    (plan_root / "summary.json").write_text(json.dumps({
        "files": {"plan": plan_path.name}, "hashes": {"plan": _sha256(plan_path)},
    }))
    (inference_root / "summary.json").write_text(json.dumps({
        "scene_count": 1, "ground_truth_usage": "none", "ap_computed": False,
        "candidate_deletion_count": 0, "geometry_mutation": False, "class_mutation": False,
    }))
    inference_audit = tmp_path / "inference_audit"
    inference_audit.mkdir()
    (inference_audit / "summary.json").write_text(json.dumps({
        "audit_valid": True, "error_count": 0,
        "advancement_gate": {"advancement_authorized": True},
    }))
    return scene_list, legacy_root, inference_root, inference_audit, Path(refined_sha1)


def test_joint_adapter_accepts_append_only_refined_union_and_cache_is_exact(tmp_path: Path):
    scene_list, legacy_root, inference_root, inference_audit, _ = _joint_inputs(tmp_path)
    geometry_root = tmp_path / "geometry"
    summary = build_geometry(argparse.Namespace(
        scene_list=scene_list, inference_root=inference_root,
        inference_audit_root=inference_audit, legacy_unique_geometry_root=legacy_root,
        output_root=geometry_root, expected_scene_count=1, class_count=198,
        dataset_name="smoke",
    ))
    assert summary["unique_geometry_count"] == 2
    assert summary["fi1_d_v3_refined_union_candidate_count"] == 1
    geometry_audit = tmp_path / "geometry_audit"
    audited = audit_geometry(argparse.Namespace(
        ledger_root=geometry_root, inference_root=inference_root, output_root=geometry_audit,
    ))
    assert audited["audit_valid"] is True

    prepared = tmp_path / "prepared" / "scene_smoke"
    prepared.mkdir(parents=True)
    np.save(prepared / "_smoke.npy", np.zeros((4, 10), dtype=np.float32))
    cache_root = tmp_path / "cache"
    cache_summary = build_cache(argparse.Namespace(
        scene_list=scene_list, ledger_root=geometry_root, ledger_audit_root=geometry_audit,
        prepared_root=tmp_path / "prepared", output_root=cache_root,
        expected_scene_count=1, dataset_name="smoke",
    ))
    assert cache_summary["geometry_count"] == 2
    cache_audit = audit_cache(argparse.Namespace(
        cache_root=cache_root, ledger_root=geometry_root, output_root=tmp_path / "cache_audit",
    ))
    assert cache_audit["audit_valid"] is True

    hashes = json.loads((cache_root / "prediction_cache/scene_smoke/geometry_hashes.json").read_text())
    plan_keys = json.loads((cache_root / "prediction_cache/scene_smoke/plan_keys.json").read_text())
    decisions = {
        ("scene_smoke", plan_key): {
            "geometry_hash": digest,
            "canonical_frozen_class_index": 4,
            "arbitrated_class_index": 5 if index == 0 else 4,
        }
        for index, (digest, plan_key) in enumerate(zip(hashes, plan_keys))
    }
    control = FrozenPredictionMapping(["scene_smoke"], cache_root, decisions, challenge=False)
    challenge = FrozenPredictionMapping(["scene_smoke"], cache_root, decisions, challenge=True)
    control_prediction = dict(control.items())["scene_smoke"]
    challenge_prediction = dict(challenge.items())["scene_smoke"]
    assert np.array_equal(control_prediction["pred_masks"], challenge_prediction["pred_masks"])
    assert np.array_equal(control_prediction["pred_scores"], challenge_prediction["pred_scores"])
    assert np.count_nonzero(control_prediction["pred_classes"] != challenge_prediction["pred_classes"]) == 1


def test_geometry_audit_rejects_empty_members_without_crashing(tmp_path: Path):
    scene_list, legacy_root, inference_root, inference_audit, _ = _joint_inputs(tmp_path)
    geometry_root = tmp_path / "geometry"
    build_geometry(argparse.Namespace(
        scene_list=scene_list, inference_root=inference_root,
        inference_audit_root=inference_audit, legacy_unique_geometry_root=legacy_root,
        output_root=geometry_root, expected_scene_count=1, class_count=198,
        dataset_name="smoke",
    ))
    rows = [json.loads(line) for line in (geometry_root / "unique_geometry_ledger.jsonl").read_text().splitlines()]
    rows[0]["members"] = []
    _write_jsonl(geometry_root / "unique_geometry_ledger.jsonl", rows)
    result = audit_geometry(argparse.Namespace(
        ledger_root=geometry_root, inference_root=inference_root, output_root=tmp_path / "audit",
    ))
    assert result["audit_valid"] is False
    assert result["errors"]["member_contract"] >= 1


def test_ap_entry_requires_both_permission_and_frozen_authorization(tmp_path: Path):
    with pytest.raises(PermissionError):
        evaluate_ap(argparse.Namespace(
            allow_gt_evaluation=False, authorization_id="wrong",
            scene_list=tmp_path / "missing", ground_truth_root=tmp_path,
            cache_root=tmp_path, cache_audit_root=tmp_path, decision_root=tmp_path,
            preregistration_path=tmp_path / "missing", output_root=tmp_path / "out",
            expected_scene_count=312, dataset_name="smoke",
        ))
    with pytest.raises(PermissionError):
        evaluate_ap(argparse.Namespace(
            allow_gt_evaluation=True, authorization_id="wrong",
            scene_list=tmp_path / "missing", ground_truth_root=tmp_path,
            cache_root=tmp_path, cache_audit_root=tmp_path, decision_root=tmp_path,
            preregistration_path=tmp_path / "missing", output_root=tmp_path / "out",
            expected_scene_count=312, dataset_name="smoke",
        ))


def test_csv_audit_recomputes_official_aggregate_and_frequency_groups(tmp_path: Path):
    csv_path = tmp_path / "ap.csv"
    rows = [
        (name, class_id, index / 1000.0, index / 900.0, index / 800.0)
        for index, (name, class_id) in enumerate(zip(CLASS_LABELS_200, VALID_CLASS_IDS_200))
        if name not in {"wall", "floor"}
    ]
    csv_path.write_text(
        "class,class id,ap,ap50,ap25\n"
        + "".join(f"{name},{class_id},{ap},{ap50},{ap25}\n" for name, class_id, ap, ap50, ap25 in rows)
    )
    metrics, class_count = _csv_metrics(csv_path)
    assert class_count == 198
    assert metrics["ap"] == pytest.approx(np.mean([row[2] for row in rows]))
    assert all(np.isfinite(metrics[name]) for name in metrics)


def test_one_shot_ap_markers_and_independent_csv_audit(tmp_path: Path, monkeypatch):
    scene_list, legacy_root, inference_root, inference_audit, _ = _joint_inputs(tmp_path)
    geometry_root = tmp_path / "geometry"
    build_geometry(argparse.Namespace(
        scene_list=scene_list, inference_root=inference_root,
        inference_audit_root=inference_audit, legacy_unique_geometry_root=legacy_root,
        output_root=geometry_root, expected_scene_count=1, class_count=198,
        dataset_name="smoke",
    ))
    geometry_audit = tmp_path / "geometry_audit"
    audit_geometry(argparse.Namespace(
        ledger_root=geometry_root, inference_root=inference_root, output_root=geometry_audit,
    ))
    prepared = tmp_path / "prepared" / "scene_smoke"
    prepared.mkdir(parents=True)
    np.save(prepared / "_smoke.npy", np.zeros((4, 10), dtype=np.float32))
    cache_root = tmp_path / "cache"
    build_cache(argparse.Namespace(
        scene_list=scene_list, ledger_root=geometry_root, ledger_audit_root=geometry_audit,
        prepared_root=tmp_path / "prepared", output_root=cache_root,
        expected_scene_count=1, dataset_name="smoke",
    ))
    cache_audit_root = tmp_path / "cache_audit"
    audit_cache(argparse.Namespace(
        cache_root=cache_root, ledger_root=geometry_root, output_root=cache_audit_root,
    ))
    hashes = json.loads((cache_root / "prediction_cache/scene_smoke/geometry_hashes.json").read_text())
    plan_keys = json.loads((cache_root / "prediction_cache/scene_smoke/plan_keys.json").read_text())
    decision_root = tmp_path / "decisions"
    decision_root.mkdir()
    decision_rows = [{
        "scene_name": "scene_smoke", "plan_key": plan_key, "geometry_hash": digest,
        "canonical_frozen_class_index": 4,
        "arbitrated_class_index": 5 if index == 0 else 4,
        "class_changed": index == 0,
    } for index, (digest, plan_key) in enumerate(zip(hashes, plan_keys))]
    _write_jsonl(decision_root / "safe_decisions.jsonl", decision_rows)
    (decision_root / "summary.json").write_text(json.dumps({
        "candidate_count": 2, "geometry_count": 2,
        "two_candidate_count": 1, "single_candidate_count": 1,
        "model_evidence_valid_count": 1, "invalid_evidence_fallback_count": 0,
        "class_change_count": 1, "ground_truth_read": False, "ap_computed": False,
    }))
    (decision_root / "audit_summary.json").write_text(json.dumps({
        "audit_valid": True, "error_count": 0,
    }))
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("frozen preregistration\n")
    gt_root = tmp_path / "gt"
    gt_root.mkdir()

    def fake_evaluate(mapping, _gt_root, csv_path):
        list(mapping.items())
        value = 0.2 if mapping.challenge else 0.1
        rows = [
            (name, class_id) for name, class_id in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200)
            if name not in {"wall", "floor"}
        ]
        csv_path.write_text(
            "class,class id,ap,ap50,ap25\n"
            + "".join(f"{name},{class_id},{value},{value},{value}\n" for name, class_id in rows)
        )
        return {name: value for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")}

    monkeypatch.setattr(
        "tools.evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt._evaluate", fake_evaluate
    )
    result_root = tmp_path / "ap"
    summary = evaluate_ap(argparse.Namespace(
        allow_gt_evaluation=True, authorization_id=AUTHORIZATION_ID,
        scene_list=scene_list, ground_truth_root=gt_root, cache_root=cache_root,
        cache_audit_root=cache_audit_root, decision_root=decision_root,
        preregistration_path=preregistration, output_root=result_root,
        expected_scene_count=1, dataset_name="smoke",
    ))
    assert summary["delta"]["ap"] == pytest.approx(0.1)
    assert (result_root / "ap_invocation_started.json").is_file()
    assert (result_root / "ap_invocation_completed.json").is_file()
    with pytest.raises(FileExistsError):
        evaluate_ap(argparse.Namespace(
            allow_gt_evaluation=True, authorization_id=AUTHORIZATION_ID,
            scene_list=scene_list, ground_truth_root=gt_root, cache_root=cache_root,
            cache_audit_root=cache_audit_root, decision_root=decision_root,
            preregistration_path=preregistration, output_root=result_root,
            expected_scene_count=1, dataset_name="smoke",
        ))
    audit = audit_ap(argparse.Namespace(
        result_root=result_root, scene_list=scene_list, cache_root=cache_root,
        cache_audit_root=cache_audit_root, decision_root=decision_root,
        preregistration_path=preregistration, output_root=tmp_path / "ap_audit",
        expected_scene_count=1,
    ))
    assert audit["audit_valid"] is True

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.apply_dm_sms1_semantic_arbitration import decide_row
from tools.audit_dm_sms1_fi1_d_v3_prediction_cache import run as audit_cache
from tools.audit_dm_sms1_fi1_d_v3_unique_geometry_ledger import run as audit_geometry
from tools.audit_dm_sms1_full_safe_decision_ledger import audit as audit_full_decisions
from tools.build_dm_sms1_attribute_extraction_manifest import build_attribute_row
from tools.build_dm_sms1_candidate_evidence_manifest import build_candidate_row
from tools.build_dm_sms1_fi1_d_v3_prediction_cache import run as build_cache
from tools.build_dm_sms1_fi1_d_v3_unique_geometry_ledger import run as build_geometry
from tools.build_dm_sms1_full_safe_decision_ledger import run as build_full_decisions
from tools.build_dm_sms1_semantic_arbitration_manifest import build_rows
from tools.dm_sms_core import geometry_hash
from tools.evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt import FrozenPredictionMapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _locator(path: Path, points: list[int]) -> dict:
    np.savez_compressed(path, point_indices=np.asarray(points, dtype=np.int64))
    return {"kind": "point_indices_npz", "points_path": str(path), "array_key": "point_indices"}


def _digest(locator: dict, algorithm: str) -> str:
    points = np.load(locator["points_path"])["point_indices"].astype(np.int64)
    if algorithm == "sha1_point_indices":
        return geometry_hash(points)
    return hashlib.sha256(np.unique(points).tobytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scene = "scene_smoke"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")
    locators = {
        "native-a": _locator(tmp_path / "a.npz", [0, 1]),
        "pair-b": _locator(tmp_path / "b.npz", [2, 3]),
        "track-c": _locator(tmp_path / "c.npz", [4, 5]),
        "pair-d": _locator(tmp_path / "d.npz", [6, 7]),
    }
    legacy_specs = [
        ("native-a", "native", 1, 10),
        ("pair-b", "pair_union", 2, 20),
        ("track-c", "track", 1, 30),
        ("pair-d", "pair_union", 1, 40),
    ]
    legacy_rows = []
    for key, source, class_index, candidate_id in legacy_specs:
        points = np.load(locators[key]["points_path"])["point_indices"]
        legacy_rows.append({
            "scene_name": scene,
            "geometry_key": key,
            "geometry_hash": geometry_hash(points),
            "point_count": len(points),
            "canonical_candidate_source": source,
            "canonical_candidate_id": candidate_id,
            "canonical_frozen_class_index": class_index,
            "canonical_frozen_score": 0.1,
            "canonical_geometry_locator": locators[key],
        })
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    _write_jsonl(legacy_root / "unique_geometry_ledger.jsonl", legacy_rows)

    plan_specs = [
        ("native-a", "native", "native-a", None, 1, 0.91, False),
        ("pair-b", "pair_union", "pair-b", None, 2, 0.82, False),
        ("track-c", "track", "track-c", None, 1, 0.73, False),
        ("pair-d", "pair_union", "pair-d", None, 1, 0.64, False),
        # Different class and score, but exactly the native-a geometry.
        ("refined-b-a", "refined_union", "native-a", "pair-b", 2, 0.55, True),
        # Same class and different score, but exactly the track-c geometry.
        ("refined-d-c", "refined_union", "track-c", "pair-d", 1, 0.46, True),
    ]
    plan = []
    for plan_key, source, locator_key, parent, class_index, score, append_only in plan_specs:
        locator = locators[locator_key]
        algorithm = "sha256_point_indices" if append_only else "sha1_point_indices"
        points = np.load(locator["points_path"])["point_indices"]
        plan.append({
            "plan_key": plan_key,
            "scene_name": scene,
            "candidate_source": source,
            "geometry_key": None if append_only else plan_key,
            "original_union_geometry_key": parent,
            "geometry_locator_read_only": locator,
            "geometry_digest": _digest(locator, algorithm),
            "geometry_digest_algorithm": algorithm,
            "point_count": len(points),
            "frozen_class_index": class_index,
            "challenger_score": score,
            "candidate_retained": True,
            "candidate_deletion": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "append_only": append_only,
        })
    inference_root = tmp_path / "inference"
    complete_plan = inference_root / "complete_plan"
    complete_plan.mkdir(parents=True)
    plan_path = complete_plan / "complete_inference_ap_plan.jsonl"
    _write_jsonl(plan_path, plan)
    (complete_plan / "summary.json").write_text(json.dumps({
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
    return scene_list, legacy_root, inference_root, inference_audit


def _build_duplicate_geometry(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scene_list, legacy_root, inference_root, inference_audit = _inputs(tmp_path)
    geometry_root = tmp_path / "geometry"
    summary = build_geometry(argparse.Namespace(
        scene_list=scene_list, inference_root=inference_root,
        inference_audit_root=inference_audit, legacy_unique_geometry_root=legacy_root,
        output_root=geometry_root, expected_scene_count=1, class_count=198,
        dataset_name="synthetic",
    ))
    assert summary["member_count"] == 6
    assert summary["unique_geometry_count"] == 4
    assert summary["duplicate_geometry_group_count"] == 2
    assert summary["duplicate_geometry_scene_count"] == 1
    assert summary["duplicate_group_different_class_count"] == 1
    assert summary["duplicate_group_different_score_count"] == 2
    geometry_audit = tmp_path / "geometry_audit"
    result = audit_geometry(argparse.Namespace(
        ledger_root=geometry_root, inference_root=inference_root,
        output_root=geometry_audit, expected_candidate_count=6,
        expected_unique_geometry_count=4, expected_duplicate_group_count=2,
        expected_duplicate_scene_count=1, expected_different_class_group_count=1,
        expected_different_score_group_count=2,
    ))
    assert result["audit_valid"] is True
    return scene_list, inference_root, geometry_root, geometry_audit


def _visual_assets(tmp_path: Path) -> dict:
    root = tmp_path / "assets"
    root.mkdir(parents=True)
    paths = {name: root / name for name in ("rgb.jpg", "depth.png", "pose.txt", "intrinsics.txt")}
    paths["rgb.jpg"].write_bytes(b"rgb")
    paths["depth.png"].write_bytes(b"depth")
    np.savetxt(paths["pose.txt"], np.eye(4))
    np.savetxt(paths["intrinsics.txt"], np.eye(4))
    return {name: str(path) for name, path in paths.items()}


def _semantic_candidates(geometry_root: Path, tmp_path: Path) -> list[dict]:
    assets = _visual_assets(tmp_path)
    geometry_rows = [
        json.loads(line)
        for line in (geometry_root / "unique_geometry_ledger.jsonl").read_text().splitlines()
    ]
    candidates = []
    for row in geometry_rows:
        alpha_row = dict(row)
        alpha_row.update({
            "alpha_class_index": 3,
            "alpha_top_similarity": 0.8,
            "sms_keep": True,
            "views": [{
                "frame_id": "0", "frame_index": 0,
                "visible_ratio": 1.0, "visible_point_count": row["point_count"],
                "rgb_path": assets["rgb.jpg"], "depth_path": assets["depth.png"],
                "pose_path": assets["pose.txt"], "intrinsics_path": assets["intrinsics.txt"],
                "sam_box_prompt_xyxy": [0.0, 0.0, 1.0, 1.0],
                "sam_mask_sha256": "a" * 64, "sam_mask_valid": True, "sam_mask_area": 4,
            }],
        })
        candidates.extend(build_rows(alpha_row, target_count=1, max_input_views=1))
    return sorted(candidates, key=lambda row: row["plan_index"])


def _evidence(candidate: dict) -> dict:
    incumbent = int(candidate["canonical_frozen_class_index"])
    alternative = next(
        int(item["class_index"])
        for item in candidate["candidate_hypotheses"]
        if int(item["class_index"]) != incumbent
    )

    def item(class_index: int, supported: bool) -> dict:
        return {
            "class_index": class_index, "supported": supported,
            "strong_counterevidence": False, "support_evidence": "visible",
            "counterevidence": "none", "confidence": 0.9,
        }

    return {
        "task_id": candidate["task_id"],
        "order_ab": {"candidate_results": [item(incumbent, False), item(alternative, True)]},
        "order_ba": {"candidate_results": [item(alternative, True), item(incumbent, False)]},
    }


def test_duplicate_geometry_evidence_expands_by_plan_key_and_cache_preserves_columns(tmp_path: Path):
    scene_list, _inference_root, geometry_root, geometry_audit = _build_duplicate_geometry(tmp_path)
    semantic_rows = _semantic_candidates(geometry_root, tmp_path)
    assert len(semantic_rows) == 6
    assert len({(row["scene_name"], row["geometry_hash"]) for row in semantic_rows}) == 4
    by_plan = {row["plan_key"]: row for row in semantic_rows}
    assert by_plan["native-a"]["selected_views"] == by_plan["refined-b-a"]["selected_views"]
    assert by_plan["native-a"]["canonical_frozen_class_index"] == 1
    assert by_plan["refined-b-a"]["canonical_frozen_class_index"] == 2
    assert by_plan["native-a"]["canonical_frozen_score"] == 0.91
    assert by_plan["refined-b-a"]["canonical_frozen_score"] == 0.55
    assert by_plan["track-c"]["canonical_frozen_class_index"] == 1
    assert by_plan["refined-d-c"]["canonical_frozen_class_index"] == 1
    assert by_plan["track-c"]["canonical_frozen_score"] != by_plan["refined-d-c"]["canonical_frozen_score"]

    attributes = [build_attribute_row(row) for row in semantic_rows]
    assert len({row["task_id"] for row in attributes}) == 6
    assert by_plan["native-a"]["geometry_hash"] == by_plan["refined-b-a"]["geometry_hash"]
    attribute_by_plan = {row["plan_key"]: row for row in attributes}
    assert attribute_by_plan["native-a"]["task_id"] != attribute_by_plan["refined-b-a"]["task_id"]
    assert "native-a" in attribute_by_plan["native-a"]["task_id"]
    assert "refined-b-a" in attribute_by_plan["refined-b-a"]["task_id"]

    class_names = [f"class-{index}" for index in range(198)]
    candidate_rows = [
        build_candidate_row(attribute, by_plan[attribute["plan_key"]], class_names)
        for attribute in attributes
    ]
    pair_decisions = [decide_row(row, _evidence(row)) for row in candidate_rows]
    for row in pair_decisions:
        row["model_evidence_valid"] = True
        row["fallback_reason"] = None
    assert {(row["scene_name"], row["plan_key"]) for row in pair_decisions} == {
        ("scene_smoke", row["plan_key"]) for row in candidate_rows
    }

    candidate_path = tmp_path / "candidate_manifest.jsonl"
    pair_path = tmp_path / "pair_decisions.jsonl"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(pair_path, pair_decisions)
    full_root = tmp_path / "full_decisions"
    full_summary = build_full_decisions(pair_path, candidate_path, full_root)
    assert full_summary["candidate_count"] == 6
    assert audit_full_decisions(full_root, candidate_path)["audit_valid"] is True

    prepared = tmp_path / "prepared" / "scene_smoke"
    prepared.mkdir(parents=True)
    np.save(prepared / "_smoke.npy", np.zeros((10, 4), dtype=np.float32))
    cache_root = tmp_path / "cache"
    cache_summary = build_cache(argparse.Namespace(
        scene_list=scene_list, ledger_root=geometry_root, ledger_audit_root=geometry_audit,
        prepared_root=tmp_path / "prepared", output_root=cache_root,
        expected_scene_count=1, dataset_name="synthetic",
    ))
    assert cache_summary["candidate_count"] == 6
    assert cache_summary["unique_geometry_count"] == 4
    scene_root = cache_root / "prediction_cache" / "scene_smoke"
    masks = np.load(scene_root / "masks.npy")
    plan_keys = json.loads((scene_root / "plan_keys.json").read_text())
    hashes = json.loads((scene_root / "geometry_hashes.json").read_text())
    assert plan_keys == ["native-a", "pair-b", "track-c", "pair-d", "refined-b-a", "refined-d-c"]
    assert len(plan_keys) == 6 and len(set(plan_keys)) == 6
    assert len(hashes) == 6 and len(set(hashes)) == 4
    assert np.array_equal(masks[:, 0], masks[:, 4])
    assert np.array_equal(masks[:, 2], masks[:, 5])

    cache_audit = audit_cache(argparse.Namespace(
        cache_root=cache_root, ledger_root=geometry_root, decision_root=full_root,
        output_root=tmp_path / "cache_audit",
    ))
    assert cache_audit["audit_valid"] is True
    assert cache_audit["candidate_deletion_count"] == 0
    assert cache_audit["control_challenge_masks_identical"] is True
    assert cache_audit["control_challenge_scores_identical"] is True
    assert cache_audit["control_challenge_order_identical"] is True

    decision_rows = [
        json.loads(line) for line in (full_root / "safe_decisions.jsonl").read_text().splitlines()
    ]
    decisions = {(row["scene_name"], row["plan_key"]): row for row in decision_rows}
    control = FrozenPredictionMapping(["scene_smoke"], cache_root, decisions, challenge=False)
    challenge = FrozenPredictionMapping(["scene_smoke"], cache_root, decisions, challenge=True)
    control_prediction = dict(control.items())["scene_smoke"]
    challenge_prediction = dict(challenge.items())["scene_smoke"]
    assert np.array_equal(control_prediction["pred_masks"], challenge_prediction["pred_masks"])
    assert np.array_equal(control_prediction["pred_scores"], challenge_prediction["pred_scores"])
    assert challenge_prediction["pred_classes"].shape == (6,)


def test_duplicate_safe_audits_reject_member_mutation_and_candidate_collapse(tmp_path: Path):
    scene_list, inference_root, geometry_root, geometry_audit = _build_duplicate_geometry(tmp_path)
    mutated_geometry = tmp_path / "mutated_geometry"
    shutil.copytree(geometry_root, mutated_geometry)
    rows = [
        json.loads(line)
        for line in (mutated_geometry / "unique_geometry_ledger.jsonl").read_text().splitlines()
    ]
    rows[0]["members"][0]["frozen_score"] += 0.01
    _write_jsonl(mutated_geometry / "unique_geometry_ledger.jsonl", rows)
    result = audit_geometry(argparse.Namespace(
        ledger_root=mutated_geometry, inference_root=inference_root,
        output_root=tmp_path / "mutated_geometry_audit",
    ))
    assert result["audit_valid"] is False
    assert result["errors"]["class_or_score_mismatch"] >= 1

    semantic_rows = _semantic_candidates(geometry_root, tmp_path / "second")
    class_names = [f"class-{index}" for index in range(198)]
    attributes = [build_attribute_row(row) for row in semantic_rows]
    candidates = [
        build_candidate_row(attribute, semantic_rows[index], class_names)
        for index, attribute in enumerate(attributes)
    ]
    pair_rows = [decide_row(row, _evidence(row)) for row in candidates]
    for row in pair_rows:
        row["model_evidence_valid"] = True
        row["fallback_reason"] = None
    candidate_path = tmp_path / "candidates.jsonl"
    pair_path = tmp_path / "pairs.jsonl"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(pair_path, pair_rows)
    full_root = tmp_path / "full"
    build_full_decisions(pair_path, candidate_path, full_root)

    prepared = tmp_path / "prepared" / "scene_smoke"
    prepared.mkdir(parents=True)
    np.save(prepared / "_smoke.npy", np.zeros((10, 4), dtype=np.float32))
    cache_root = tmp_path / "cache"
    build_cache(argparse.Namespace(
        scene_list=scene_list, ledger_root=geometry_root, ledger_audit_root=geometry_audit,
        prepared_root=tmp_path / "prepared", output_root=cache_root,
        expected_scene_count=1, dataset_name="synthetic",
    ))
    collapsed = tmp_path / "collapsed_cache"
    shutil.copytree(cache_root, collapsed)
    root = collapsed / "prediction_cache" / "scene_smoke"
    for name in ("masks.npy", "frozen_classes.npy", "frozen_scores.npy"):
        values = np.load(root / name)
        values = values[:, :-1] if values.ndim == 2 else values[:-1]
        np.save(root / name, values, allow_pickle=False)
    for name in ("geometry_hashes.json", "plan_keys.json", "sources.json", "plan_indices.json"):
        values = json.loads((root / name).read_text())[:-1]
        (root / name).write_text(json.dumps(values) + "\n")
    collapsed_audit = audit_cache(argparse.Namespace(
        cache_root=collapsed, ledger_root=geometry_root, decision_root=full_root,
        output_root=tmp_path / "collapsed_audit",
    ))
    assert collapsed_audit["audit_valid"] is False
    assert collapsed_audit["error_count"] > 0

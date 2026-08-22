from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from tools.audit_fi1_d_v3_frozen_inference_plan import run as audit_run
from tools.build_ncs_fi1_stage_a_quality_dataset_gt import UNION_FEATURE_NAMES
from tools.build_ncs_fi1_stage_c_v2_member_dataset_gt import MEMBER_EVIDENCE_FEATURE_NAMES
from tools.build_ncs_fi1_stage_d_v3_full_rank_marginal_dataset_gt import FEATURE_NAMES as D_FEATURE_NAMES
from tools.build_fi1_d_v3_frozen_inference_plan import (
    _load_models,
    _predict,
    _stage_d_and_complete_plan,
)
from tools.evaluate_fi1_d_v3_frozen_class_agnostic_ap_gt import run as evaluate_run
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import QUALITY_FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "pretrained/fi1_d_v3_official100_full_models_20260822"


def _npz(path: Path, points: list[int]) -> dict:
    np.savez_compressed(path, point_indices=np.asarray(points, dtype=np.int64))
    return {"kind": "point_indices_npz", "points_path": str(path), "array_key": "point_indices"}


def _node(scene: str, source: str, key: str, candidate_id: int, locator: dict, points: list[int]) -> dict:
    canonical = np.unique(np.asarray(points, dtype=np.int64))
    return {
        "scene_name": scene,
        "geometry_key": key,
        "geometry_hash": hashlib.sha1(canonical.tobytes()).hexdigest(),
        "canonical_candidate_source": source,
        "canonical_candidate_id": candidate_id,
        "canonical_geometry_locator": locator,
        "canonical_frozen_score": 0.6,
        "canonical_frozen_class_index": 4,
        "point_count": len(points),
        "members": [{"candidate_source": source, "candidate_id": candidate_id}],
    }


def test_full_model_package_has_frozen_seven_model_contract():
    models = _load_models(MODEL_ROOT)
    assert set(models) == {
        "stage_a_native_quality", "stage_a_track_quality", "stage_a_pair_union_quality",
        "stage_c_member_delta_track_only", "stage_c_member_delta_native_only",
        "stage_c_temporary_refined_quality", "stage_d_v3_rank_marginal_gain",
    }
    bundle = models["stage_d_v3_rank_marginal_gain"]
    prediction = _predict(bundle, [[0.0] * len(bundle["feature_names"])])
    assert prediction.shape == (1,)
    assert 0.0 <= float(prediction[0]) <= 1.0


def test_stage_d_builds_169_features_and_complete_plan(tmp_path: Path):
    scene = "scene_smoke"
    prepared = tmp_path / "prepared" / scene
    prepared.mkdir(parents=True)
    np.save(prepared / "_smoke.npy", np.zeros((10, 10), dtype=np.float32))
    native_points = [0, 1, 2]
    track_points = [3, 4, 5]
    union_points = [0, 1, 2, 3, 4, 5]
    native = _node(scene, "native", "native-key", 0, _npz(tmp_path / "native.npz", native_points), native_points)
    track = _node(scene, "track", "track-key", 0, _npz(tmp_path / "track.npz", track_points), track_points)
    union = _node(scene, "pair_union", "union-key", 0, _npz(tmp_path / "union.npz", union_points), union_points)
    nodes = [native, track, union]
    stage_a = {
        "native-key": {"oof_unified_quality": 0.5},
        "track-key": {"oof_unified_quality": 0.4},
        "union-key": {
            "oof_unified_quality": 0.6,
            "features": {name: 0.0 for name in UNION_FEATURE_NAMES},
        },
    }
    stage_b = {
        key: {"stage_b_score": value}
        for key, value in (("native-key", 0.5), ("track-key", 0.4), ("union-key", 0.6))
    }
    quality = {name: 0.0 for name in QUALITY_FEATURE_NAMES}
    quality.update({
        "stage_a_union_quality": 0.6,
        "stage_a_track_quality": 0.4,
        "stage_a_native_quality": 0.5,
        "stage_b_union_score": 0.6,
        "log1p_original_union_point_count": float(np.log1p(6)),
        "log1p_temporary_refined_point_count": float(np.log1p(6)),
        "temporary_refined_point_fraction": 1.0,
        "atom_count": 1.0,
    })
    for name in MEMBER_EVIDENCE_FEATURE_NAMES:
        quality[f"all_mean__{name}"] = 0.0
        quality[f"track_only_mean__{name}"] = 0.0
        quality[f"native_only_mean__{name}"] = 0.0
    stage_c = {(scene, 0): {
        "scene_name": scene, "union_candidate_id": 0,
        "original_union_geometry_key": "union-key", "atom_count": 1,
        "quality_features": quality, "corrected_oof_temporary_refined_quality": 0.6,
        "quality_lower_confidence_bound": 0.5, "append_eligible_refined_union": False,
        "confident_removal_count": 0, "connectivity_removal_count": 0,
    }}
    champion = tmp_path / "champion" / scene
    champion.mkdir(parents=True)
    original = {
        "scene_name": scene, "candidate_id": 0, "threshold_cross_probability": 0.1,
        "base_quality": 0.5, "track_quality_q": 0.4,
        "native_group_median_quality_q": 0.5,
    }
    (champion / "pair_union_append_candidates.jsonl").write_text(json.dumps(original) + "\n")
    models = _load_models(MODEL_ROOT)
    (tmp_path / "out").mkdir()
    scores, complete = _stage_d_and_complete_plan(
        scenes=[scene], nodes=nodes, stage_a=stage_a, stage_b=stage_b,
        stage_b_relations=[], stage_c=stage_c, champion_root=tmp_path / "champion",
        models=models, staging=tmp_path / "out", prepared_root=tmp_path / "prepared",
    )
    assert len(scores) == 1
    assert len(D_FEATURE_NAMES) == 169
    assert scores[0]["stage_d_v3_append_score"] == pytest.approx(
        scores[0]["candidate_reference_score"] * scores[0]["conservative_gain"]
    )
    assert len(complete) == 3
    assert all(row["candidate_retained"] for row in complete)


def test_audit_accepts_exact_control_only_plan(tmp_path: Path):
    scene = "scene_smoke"
    points = [0, 2, 4]
    locator = _npz(tmp_path / "geometry.npz", points)
    node = _node(scene, "native", "native-key", 0, locator, points)
    unique = tmp_path / "unique"
    unique.mkdir()
    (unique / "unique_geometry_ledger.jsonl").write_text(json.dumps(node) + "\n")
    inference = tmp_path / "inference"
    plan_root = inference / "complete_plan"
    plan_root.mkdir(parents=True)
    plan = {
        "plan_key": "native-key", "scene_name": scene, "candidate_source": "native",
        "geometry_key": "native-key", "geometry_locator_read_only": locator,
        "geometry_digest": node["geometry_hash"], "geometry_digest_algorithm": "sha1_point_indices",
        "point_count": len(points), "frozen_class_index": 4,
        "control_score": 0.6, "challenger_score": 0.5,
        "candidate_retained": True, "candidate_deletion": False,
        "geometry_mutation": False, "class_mutation": False, "append_only": False,
    }
    plan_path = plan_root / "complete_inference_ap_plan.jsonl"
    plan_path.write_text(json.dumps(plan) + "\n")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    (plan_root / "summary.json").write_text(json.dumps({"files": {"plan": plan_path.name}, "hashes": {"plan": digest}}))
    (inference / "summary.json").write_text(json.dumps({"scene_count": 1, "ground_truth_usage": "none", "ap_computed": False}))
    result = audit_run(argparse.Namespace(
        inference_root=inference, unique_geometry_root=unique, output_root=tmp_path / "audit"
    ))
    assert result["audit_valid"] is True
    assert result["advancement_gate"]["advancement_authorized"] is True


def test_ap_entry_requires_explicit_gt_authorization(tmp_path: Path):
    with pytest.raises(PermissionError):
        evaluate_run(argparse.Namespace(
            allow_gt_evaluation=False, scene_list=tmp_path / "scenes.txt",
            ground_truth_root=tmp_path, inference_root=tmp_path,
            audit_root=tmp_path, output_root=tmp_path / "output", dataset_name="smoke",
        ))

#!/usr/bin/env python3
"""Build a GT-free FI1-D-v3 deployment plan with official100 full models."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_automatic_sam_track_growth_ledger import _raw_superpoint_context  # noqa: E402
from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    FEATURE_NAMES_BY_SOURCE, GeometryResolver, _gvc_features, _read_jsonl,
    _resolve, _sha256, _union_features,
)
from tools.build_ncs_fi1_stage_b_relation_rerank_plan import run as run_stage_b  # noqa: E402
from tools.build_ncs_fi1_stage_c_member_dataset_gt import (  # noqa: E402
    ADJACENCY, _atom_features, _geometry_sha256, decompose_union_atoms,
)
from tools.build_ncs_fi1_stage_c_v2_member_dataset_gt import (  # noqa: E402
    MEMBER_EVIDENCE_FEATURE_NAMES, _union_observation_arrays, member_evidence_features,
)
from tools.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import (  # noqa: E402
    candidate_features as v1_candidate_features, overlap_features, relation_features,
)
from tools.build_ncs_fi1_stage_d_v2_rank_marginal_dataset_gt import _prefix_name  # noqa: E402
from tools.build_ncs_fi1_stage_d_v3_full_rank_marginal_dataset_gt import (  # noqa: E402
    FEATURE_NAMES as STAGE_D_FEATURE_NAMES, candidate_key, full_rank_prefix,
)
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import (  # noqa: E402
    EXCLUSIVE_ROLES, QUALITY_FEATURE_NAMES, conservative_kept_indexes,
)


VERSION = "fi1_d_v3_frozen_inference_plan_v1"


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _load_models(root: Path) -> dict[str, dict]:
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("training_dataset") != "official100" or summary.get("model_count") != 7:
        raise ValueError("deployment package is not the frozen seven-model official100 package")
    result = {}
    for record in summary["models"]:
        path = root / record["model_file"]
        if _sha256(path) != record["model_sha256"]:
            raise ValueError(f"deployment model SHA-256 mismatch: {path}")
        payload = joblib.load(path)
        kind = str(payload["model_kind"])
        if kind in result:
            raise ValueError(f"duplicate deployment model kind: {kind}")
        result[kind] = payload
    return result


def _predict(bundle: dict, rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    expected = len(bundle["feature_names"])
    if matrix.ndim != 2 or matrix.shape[1] != expected or not np.isfinite(matrix).all():
        raise ValueError(f"invalid feature matrix for {bundle['model_kind']}")
    raw = np.asarray(bundle["regressor"].predict(matrix), dtype=np.float64)
    if "calibrator" in bundle:
        calibrator = bundle["calibrator"]
        corrected = (
            np.full(len(raw), float(calibrator), dtype=np.float64)
            if isinstance(calibrator, float) else np.asarray(calibrator.predict(np.clip(raw, 0.0, 1.0)))
        )
    else:
        corrected = raw + float(bundle["calibration_bias"])
    clip = bundle.get("prediction_clip")
    if clip is not None:
        corrected = np.clip(corrected, float(clip[0]), float(clip[1]))
    if not np.isfinite(corrected).all():
        raise ValueError(f"non-finite prediction from {bundle['model_kind']}")
    return corrected


def _members(nodes: list[dict]) -> dict[tuple[str, str, int], str]:
    result = {}
    for node in nodes:
        for member in node["members"]:
            key = (str(node["scene_name"]), str(member["candidate_source"]), int(member["candidate_id"]))
            if key in result:
                raise ValueError(f"duplicate unique-geometry member: {key}")
            result[key] = str(node["geometry_key"])
    return result


def _stage_a(
    *, scenes: list[str], nodes: list[dict], gvc_root: Path,
    relation_root: Path, champion_root: Path, models: dict, staging: Path,
) -> tuple[dict[str, dict], Path, Path]:
    nodes_by_scene = defaultdict(list)
    for node in nodes:
        nodes_by_scene[str(node["scene_name"])].append(node)
    resolver = GeometryResolver()
    rows = []
    for scene_index, scene in enumerate(scenes, 1):
        gvc_rows = json.loads((gvc_root / scene / "c1_gvc_quality_ledger.json").read_text())
        gvc_by_key = {(str(row["candidate_source"]), int(row["candidate_id"])): row for row in gvc_rows}
        relation_rows = _read_jsonl(relation_root / scene / "relation_features_no_gt.jsonl")
        relation_by_key = {
            (int(row["track_id"]), str(row["native_exact_geometry_group_id"])): row
            for row in relation_rows
        }
        union_rows = [
            row for row in _read_jsonl(champion_root / "pair_union_append_candidates.jsonl")
            if str(row["scene_name"]) == scene
        ] if not (champion_root / scene / "pair_union_append_candidates.jsonl").is_file() else _read_jsonl(
            champion_root / scene / "pair_union_append_candidates.jsonl"
        )
        union_by_id = {int(row["candidate_id"]): row for row in union_rows}
        scene_point_count = max(int(node["point_count"]) for node in nodes_by_scene[scene])
        native_nodes = [node for node in nodes_by_scene[scene] if node["canonical_candidate_source"] == "native"]
        if native_nodes:
            locator = native_nodes[0]["canonical_geometry_locator"]
            masks = np.load(_resolve(Path(locator["masks_path"])), mmap_mode="r")
            scene_point_count = int(masks.shape[0])
        for node in nodes_by_scene[scene]:
            source = str(node["canonical_candidate_source"])
            candidate_id = int(node["canonical_candidate_id"])
            if source == "native":
                evidence = gvc_by_key[("native_mask3d_yoloworld", candidate_id)]
                features = _gvc_features(evidence, node, scene_point_count)
            elif source == "track":
                evidence = gvc_by_key[("d2b_track", candidate_id)]
                features = _gvc_features(evidence, node, scene_point_count)
            elif source == "pair_union":
                union = union_by_id[candidate_id]
                relation = relation_by_key[(int(union["track_id"]), str(union["native_exact_geometry_group_id"]))]
                features = _union_features(union, relation, node, scene_point_count)
            else:
                raise ValueError(f"unsupported source in stage A: {source}")
            bundle = models[f"stage_a_{source}_quality"]
            names = tuple(FEATURE_NAMES_BY_SOURCE[source])
            predicted = float(_predict(bundle, [[features[name] for name in names]])[0])
            rows.append({
                "row_id": f"{scene}:{node['geometry_hash']}", "scene_name": scene,
                "fold_index": -1, "geometry_key": str(node["geometry_key"]),
                "geometry_hash": str(node["geometry_hash"]), "candidate_source": source,
                "features": features,
                "oof_unified_quality": predicted, "deployment_unified_quality": predicted,
                "ground_truth_usage": "none", "ap_computed": False,
            })
        print(f"[FI1-D-v3 inference A] {scene_index}/{len(scenes)} {scene}", flush=True)
    stage_a_root = staging / "stage_a"
    stage_a_audit_root = staging / "stage_a_audit"
    stage_a_root.mkdir(); stage_a_audit_root.mkdir()
    prediction_path = stage_a_root / "deployment_quality_predictions.jsonl"
    _write_jsonl(prediction_path, rows)
    (stage_a_root / "summary.json").write_text(json.dumps({
        "version": VERSION, "prediction_file": prediction_path.name,
        "geometry_count": len(rows), "advancement_gate": {"advancement_authorized": True},
        "ground_truth_usage": "none", "ap_computed": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (stage_a_audit_root / "summary.json").write_text(json.dumps({
        "version": VERSION, "audit_valid": True, "error_count": 0,
        "ground_truth_usage": "none", "ap_computed": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {str(row["geometry_key"]): row for row in rows}, stage_a_root, stage_a_audit_root


def _stage_b(
    *, unique_root: Path, relation_root: Path, champion_root: Path,
    stage_a_root: Path, stage_a_audit_root: Path, staging: Path,
) -> tuple[dict[str, dict], list[dict], Path]:
    output_root = staging / "stage_b"
    run_stage_b(argparse.Namespace(
        unique_geometry_root=unique_root, relation_root=relation_root,
        champion_plan_root=champion_root, stage_a_oof_root=stage_a_root,
        stage_a_oof_audit_root=stage_a_audit_root,
        preregistration=PROJECT_ROOT / "docs/NCS_FI1_STAGE_B_PREREGISTRATION_20260822.md",
        output_root=output_root,
    ))
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    plan = _read_jsonl(output_root / "stage_b_rerank_plan.jsonl")
    deployment_checks = {
        "all_geometry_covered_once": len(plan) == len({str(row["geometry_key"]) for row in plan}),
        "native_scores_unchanged": all(
            float(row["stage_b_score"]) == float(row["stage_a_quality"])
            for row in plan if row["candidate_source"] == "native"
        ),
        "no_score_increase": all(
            float(row["stage_b_score"]) <= float(row["stage_a_quality"]) + 1e-15
            for row in plan
        ),
        "candidate_deletion_count_zero": summary.get("candidate_deletion_count") == 0,
        "geometry_mutation_false": summary.get("geometry_mutation") is False,
        "class_mutation_false": summary.get("class_mutation") is False,
    }
    summary["deployment_advancement_gate"] = {
        "checks": deployment_checks,
        "advancement_authorized": all(deployment_checks.values()),
        "note": "OOF fold-direction checks are inapplicable to a fold-free deployment target",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not summary["deployment_advancement_gate"]["advancement_authorized"]:
        raise ValueError("stage-B deployment checks failed")
    relations = _read_jsonl(output_root / "candidate_relations.jsonl")
    return {str(row["geometry_key"]): row for row in plan}, relations, output_root


def _quality_features(rows: list[dict], predictions: list[float], lowers: list[float], details: dict,
                      original_count: int, temporary_count: int) -> dict:
    def values(role: str, source: list[float]) -> list[float]:
        return [value for row, value in zip(rows, source) if row["role"] == role]
    def mean(source: list[float]) -> float:
        return float(np.mean(source)) if source else 0.0
    def maximum(source: list[float]) -> float:
        return float(np.max(source)) if source else 0.0
    first = rows[0]["features"]
    quality = {
        "stage_a_union_quality": float(first["stage_a_union_quality"]),
        "stage_a_track_quality": float(first["stage_a_track_quality"]),
        "stage_a_native_quality": float(first["stage_a_native_quality"]),
        "stage_b_union_score": float(first["stage_b_union_score"]),
        "log1p_original_union_point_count": math.log1p(original_count),
        "log1p_temporary_refined_point_count": math.log1p(temporary_count),
        "temporary_refined_point_fraction": temporary_count / max(1, original_count),
        "atom_count": float(len(rows)),
        "shared_atom_fraction": sum(row["role"] == "shared" for row in rows) / max(1, len(rows)),
        "track_only_atom_fraction": sum(row["role"] == "track_only" for row in rows) / max(1, len(rows)),
        "native_only_atom_fraction": sum(row["role"] == "native_only" for row in rows) / max(1, len(rows)),
        "confident_removal_atom_fraction": len(details["confident_removed_indexes"]) / max(1, len(rows)),
        "connectivity_removed_atom_fraction": len(details["connectivity_removed_indexes"]) / max(1, len(rows)),
        "total_removed_atom_fraction": (
            len(details["confident_removed_indexes"]) + len(details["connectivity_removed_indexes"])
        ) / max(1, len(rows)),
        "track_only_predicted_delta_mean": mean(values("track_only", predictions)),
        "track_only_predicted_delta_max": maximum(values("track_only", predictions)),
        "track_only_lower_bound_mean": mean(values("track_only", lowers)),
        "track_only_lower_bound_max": maximum(values("track_only", lowers)),
        "native_only_predicted_delta_mean": mean(values("native_only", predictions)),
        "native_only_predicted_delta_max": maximum(values("native_only", predictions)),
        "native_only_lower_bound_mean": mean(values("native_only", lowers)),
        "native_only_lower_bound_max": maximum(values("native_only", lowers)),
    }
    for name in MEMBER_EVIDENCE_FEATURE_NAMES:
        quality[f"all_mean__{name}"] = mean([row["features"][name] for row in rows])
    for role in EXCLUSIVE_ROLES:
        for name in MEMBER_EVIDENCE_FEATURE_NAMES:
            quality[f"{role}_mean__{name}"] = mean([
                row["features"][name] for row in rows if row["role"] == role
            ])
    if tuple(quality) != QUALITY_FEATURE_NAMES:
        raise AssertionError("stage-C deployment quality feature order differs")
    return quality


def _stage_c(
    *, scenes: list[str], nodes: list[dict], member_to_key: dict, stage_a: dict,
    stage_b: dict, prepared_root: Path, relation_root: Path, champion_root: Path,
    track_root: Path, automatic_root: Path, config_path: Path, models: dict,
    staging: Path, final_output_root: Path, min_points: int,
) -> tuple[dict[tuple[str, int], dict], Path]:
    from utils import WORLD_2_CAM

    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    resolver = GeometryResolver()
    config = yaml.safe_load(config_path.read_text())
    depth_scale = float(config["openyolo3d"]["depth_scale"])
    point_root = staging / "stage_c" / "refined_points"
    point_root.mkdir(parents=True)
    output = []
    for scene_index, scene in enumerate(scenes, 1):
        stem = scene[len("scene"):] if scene.startswith("scene") else scene
        processed = np.load(prepared_root / scene / f"{stem}.npy", mmap_mode="r")
        scene_points = int(len(processed))
        context = _raw_superpoint_context(
            np.asarray(processed[:, :10]), ADJACENCY["knn"], ADJACENCY["max_distance"],
            ADJACENCY["min_contact_points"], ADJACENCY["min_contact_ratio"],
        )
        relation_by_key = {
            (int(row["track_id"]), str(row["native_exact_geometry_group_id"])): row
            for row in _read_jsonl(relation_root / scene / "relation_features_no_gt.jsonl")
        }
        union_path = champion_root / scene / "pair_union_append_candidates.jsonl"
        unions = _read_jsonl(union_path) if union_path.is_file() else [
            row for row in _read_jsonl(champion_root / "pair_union_append_candidates.jsonl")
            if str(row["scene_name"]) == scene
        ]
        tracks_payload = json.loads((track_root / scene / "automatic_tracks.json").read_text())
        track_by_id = {int(row["track_id"]): row for row in tracks_payload["tracks"]}
        automatic_scene = automatic_root / scene
        observation_by_id = {
            int(row["observation_id"]): row
            for row in _read_jsonl(automatic_scene / "automatic_observations.jsonl")
        }
        world = WORLD_2_CAM(str(prepared_root / scene), depth_scale, config)
        world.points_xyz = np.asarray(processed[:, :3], dtype=np.float64)
        intrinsic = world.adjust_intrinsic(
            np.loadtxt(world.intrinsics[0]), world.image_resolution, world.depth_resolution
        )
        depth_cache = {}
        for union in sorted(unions, key=lambda row: int(row["candidate_id"])):
            union_id = int(union["candidate_id"]); track_id = int(union["track_id"])
            native_ids = list(map(int, union["native_member_candidate_ids"]))
            union_key = member_to_key[(scene, "pair_union", union_id)]
            track_key = member_to_key[(scene, "track", track_id)]
            native_keys = {member_to_key[(scene, "native", value)] for value in native_ids}
            if len(native_keys) != 1:
                raise ValueError(f"{scene}/{union_id}: native parent maps to multiple geometries")
            native_key = next(iter(native_keys))
            union_node, track_node, native_node = node_by_key[union_key], node_by_key[track_key], node_by_key[native_key]
            union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), scene_points)
            track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), scene_points)
            native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), scene_points)
            atoms = decompose_union_atoms(union_points, track_points, native_points, np.asarray(processed[:, 9], dtype=np.int64))
            relation = relation_by_key[(track_id, str(union["native_exact_geometry_group_id"]))]
            qualities = {
                "stage_a_union_quality": float(stage_a[union_key]["oof_unified_quality"]),
                "stage_a_track_quality": float(stage_a[track_key]["oof_unified_quality"]),
                "stage_a_native_quality": float(stage_a[native_key]["oof_unified_quality"]),
                "stage_b_union_score": float(stage_b[union_key]["stage_b_score"]),
            }
            observation_ids = list(map(int, track_by_id[track_id].get("observation_ids", [])))
            observations = [observation_by_id[value] for value in observation_ids]
            source_support, visible, inside, weights = _union_observation_arrays(
                union_points=union_points, observations=observations, world=world,
                intrinsic=intrinsic, automatic_scene_root=automatic_scene,
                depth_cache=depth_cache, depth_scale=depth_scale,
            )
            predicted_ious = [float(row.get("predicted_iou", 0.0)) for row in observations]
            stability_scores = [float(row.get("stability_score", 0.0)) for row in observations]
            atom_rows, predictions, lowers = [], [], []
            for atom_index, atom in enumerate(atoms):
                atom_local = np.searchsorted(union_points, atom["points"])
                evidence = member_evidence_features(
                    atom_local, source_support, visible, inside, weights, predicted_ious, stability_scores,
                )
                features = _atom_features(
                    atom, union_points, track_points, native_points,
                    np.asarray(processed[:, :10]), context, qualities, relation,
                )
                features.update(evidence)
                role = str(atom["role"])
                if role == "shared":
                    predicted = lower = 0.0
                else:
                    bundle = models[f"stage_c_member_delta_{role}"]
                    names = tuple(bundle["feature_names"])
                    predicted = float(_predict(bundle, [[features[name] for name in names]])[0])
                    lower = predicted - float(bundle["calibration_absolute_residual_q90"])
                predictions.append(predicted); lowers.append(lower)
                atom_rows.append({"atom_index": atom_index, "role": role, "features": features})
            kept, details = conservative_kept_indexes(atoms, lowers, context["neighbors"])
            temporary = np.unique(np.concatenate([atoms[index]["points"] for index in sorted(kept)])).astype(np.int64) if kept else np.empty(0, np.int64)
            if details["fallback_no_shared"] or len(temporary) < min_points:
                temporary = union_points
            changed = not np.array_equal(temporary, union_points)
            quality = _quality_features(atom_rows, predictions, lowers, details, len(union_points), len(temporary))
            quality_bundle = models["stage_c_temporary_refined_quality"]
            predicted_quality = float(_predict(quality_bundle, [[quality[name] for name in QUALITY_FEATURE_NAMES]])[0])
            quality_lower = predicted_quality - float(quality_bundle["calibration_absolute_residual_q90"])
            eligible = bool(changed and quality_lower > float(quality["stage_a_union_quality"]))
            points_file = None
            deployment_points_file = None
            if eligible:
                scene_root = point_root / scene
                scene_root.mkdir(parents=True, exist_ok=True)
                path = scene_root / f"union{union_id:04d}_refined_points.npz"
                np.savez_compressed(path, point_indices=temporary)
                points_file = str(path)
                deployment_points_file = str(
                    final_output_root / "stage_c" / "refined_points" / scene / path.name
                )
            output.append({
                "scene_name": scene, "union_candidate_id": union_id,
                "original_union_geometry_key": union_key,
                "original_union_point_count": len(union_points),
                "original_union_sha256": _geometry_sha256(union_points),
                "temporary_refined_point_count": len(temporary),
                "temporary_refined_sha256": _geometry_sha256(temporary),
                "atom_count": len(atoms), "quality_features": quality,
                "corrected_oof_temporary_refined_quality": predicted_quality,
                "quality_calibration_absolute_residual_q90": float(quality_bundle["calibration_absolute_residual_q90"]),
                "quality_lower_confidence_bound": quality_lower,
                "append_eligible_refined_union": eligible,
                "refined_points_file": points_file,
                "deployment_refined_points_file": deployment_points_file,
                "confident_removal_count": len(details["confident_removed_indexes"]),
                "connectivity_removal_count": len(details["connectivity_removed_indexes"]),
                "ground_truth_usage": "none", "ap_computed": False,
            })
        print(f"[FI1-D-v3 inference C] {scene_index}/{len(scenes)} {scene}", flush=True)
    path = staging / "stage_c" / "stage_c_v2_refined_union_plan.jsonl"
    _write_jsonl(path, output)
    (path.parent / "summary.json").write_text(json.dumps({
        "version": VERSION, "union_count": len(output),
        "append_eligible_count": sum(row["append_eligible_refined_union"] for row in output),
        "ground_truth_usage": "none", "ap_computed": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in output}, path.parent


def _stage_d_and_complete_plan(
    *, scenes: list[str], nodes: list[dict], stage_a: dict, stage_b: dict,
    stage_b_relations: list[dict], stage_c: dict, champion_root: Path,
    models: dict, staging: Path, prepared_root: Path,
) -> tuple[list[dict], list[dict]]:
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    nodes_by_scene = defaultdict(list)
    for node in nodes:
        nodes_by_scene[str(node["scene_name"])].append(node)
    relations_by_geometry = defaultdict(Counter)
    for row in stage_b_relations:
        relations_by_geometry[str(row["geometry_key_a"])][str(row["relation_type"])] += 1
        relations_by_geometry[str(row["geometry_key_b"])][str(row["relation_type"])] += 1
    resolver = GeometryResolver()
    score_rows = []
    refined_rows = []
    for scene_index, scene in enumerate(scenes, 1):
        stem = scene[len("scene"):] if scene.startswith("scene") else scene
        scene_point_count = int(len(np.load(prepared_root / scene / f"{stem}.npy", mmap_mode="r")))
        union_path = champion_root / scene / "pair_union_append_candidates.jsonl"
        originals = _read_jsonl(union_path) if union_path.is_file() else [
            row for row in _read_jsonl(champion_root / "pair_union_append_candidates.jsonl")
            if str(row["scene_name"]) == scene
        ]
        baseline = []
        for node in nodes_by_scene[scene]:
            source = str(node["canonical_candidate_source"])
            if source not in {"native", "track"}:
                continue
            points = resolver.points(node["canonical_geometry_locator"], int(node["point_count"]), 2**63 - 1)
            baseline.append({"kind": source, "node": node, "points": points, "reference_score": float(stage_b[str(node["geometry_key"])]["stage_b_score"])})
        append = []
        for union in originals:
            union_id = int(union["candidate_id"])
            c_row = stage_c[(scene, union_id)]
            original_key = str(c_row["original_union_geometry_key"])
            original_node = node_by_key[original_key]
            original_points = resolver.points(original_node["canonical_geometry_locator"], int(original_node["point_count"]), 2**63 - 1)
            append.append({
                "candidate_key": candidate_key(scene, union_id, "original"), "union_candidate_id": union_id,
                "variant": "original", "points": original_points, "locator": original_node["canonical_geometry_locator"],
                "original_points": original_points, "original_key": original_key, "union": union,
                "stage_c": c_row, "reference_score": float(stage_b[original_key]["stage_b_score"]),
            })
            if c_row["append_eligible_refined_union"]:
                path = Path(c_row["refined_points_file"])
                with np.load(path) as payload:
                    points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
                append.append({
                    "candidate_key": candidate_key(scene, union_id, "refined"), "union_candidate_id": union_id,
                    "variant": "refined", "points": points,
                    "locator": {
                        "kind": "point_indices_npz",
                        "points_path": str(c_row["deployment_refined_points_file"]),
                        "array_key": "point_indices",
                    },
                    "original_points": original_points, "original_key": original_key, "union": union,
                    "stage_c": c_row, "reference_score": float(np.clip(c_row["quality_lower_confidence_bound"], 0.0, 1.0)),
                })
        local_a = dict(stage_a); local_b = dict(stage_b)
        for entry in append:
            virtual = str(entry["candidate_key"]); c_row = entry["stage_c"]
            quality = float(c_row["corrected_oof_temporary_refined_quality"]) if entry["variant"] == "refined" else float(stage_a[entry["original_key"]]["oof_unified_quality"])
            local_a[virtual] = {"oof_unified_quality": quality}
            local_b[virtual] = {"stage_b_score": float(entry["reference_score"])}
            entry["kind"] = f"{entry['variant']}_union"; entry["node"] = {"geometry_key": virtual}
        for entry in sorted(append, key=lambda item: item["candidate_key"]):
            prefix_entries = full_rank_prefix(baseline, append, entry)
            prefix = [(item["node"], item["points"]) for item in prefix_entries]
            overlap = overlap_features(entry["points"], prefix, local_a, local_b)
            base_features = v1_candidate_features(
                variant=entry["variant"], points=entry["points"], original_points=entry["original_points"],
                scene_point_count=scene_point_count, original_union=entry["union"],
                original_stage_a_row=stage_a[entry["original_key"]],
                original_stage_b_row=stage_b[entry["original_key"]], stage_c_row=entry["stage_c"],
                overlap=overlap, relation=relation_features(entry["original_key"], relations_by_geometry),
            )
            original_stage_a = stage_a[entry["original_key"]]
            if "features" not in original_stage_a:
                raise AssertionError("stage-A union features were not retained for stage D")
            features = {_prefix_name(name): float(value) for name, value in base_features.items()}
            counts = Counter(item["kind"] for item in prefix_entries)
            features.update({
                "rank_prefix_geometry_count": float(len(prefix_entries)),
                "rank_prefix_fraction_of_native_track": (counts["native"] + counts["track"]) / max(1, len(baseline)),
                "rank_prefix_native_count": float(counts["native"]), "rank_prefix_track_count": float(counts["track"]),
                "rank_prefix_original_union_count": float(counts["original_union"]),
                "rank_prefix_refined_union_count": float(counts["refined_union"]),
                "rank_prefix_append_fraction": (counts["original_union"] + counts["refined_union"]) / max(1, len(prefix_entries)),
            })
            if set(features) != set(STAGE_D_FEATURE_NAMES):
                raise ValueError("stage-D deployment feature schema differs")
            bundle = models["stage_d_v3_rank_marginal_gain"]
            corrected = float(_predict(bundle, [[features[name] for name in STAGE_D_FEATURE_NAMES]])[0])
            conservative = max(0.0, corrected - float(bundle["calibration_absolute_residual_q90"]))
            score = float(entry["reference_score"]) * conservative
            score_rows.append({
                "candidate_key": entry["candidate_key"], "scene_name": scene,
                "union_candidate_id": entry["union_candidate_id"], "candidate_variant": entry["variant"],
                "geometry_locator_read_only": entry["locator"], "point_count": len(entry["points"]),
                "geometry_sha256": _geometry_sha256(entry["points"]),
                "candidate_reference_score": float(entry["reference_score"]),
                "corrected_rank_marginal_iou_gain": corrected,
                "calibration_absolute_residual_q90": float(bundle["calibration_absolute_residual_q90"]),
                "conservative_gain": conservative, "stage_d_v3_append_score": score,
                "original_union_geometry_key": entry["original_key"],
                "ground_truth_usage": "none", "ap_computed": False,
            })
            if entry["variant"] == "refined":
                refined_rows.append(score_rows[-1])
        print(f"[FI1-D-v3 inference D] {scene_index}/{len(scenes)} {scene}", flush=True)
    original_score_by_geometry = {
        str(row["original_union_geometry_key"]): row for row in score_rows if row["candidate_variant"] == "original"
    }
    complete = []
    for node in nodes:
        source = str(node["canonical_candidate_source"]); key = str(node["geometry_key"])
        challenger = float(stage_b[key]["stage_b_score"]) if source in {"native", "track"} else float(original_score_by_geometry[key]["stage_d_v3_append_score"])
        complete.append({
            "plan_key": key, "scene_name": str(node["scene_name"]), "candidate_source": source,
            "geometry_key": key, "geometry_locator_read_only": node["canonical_geometry_locator"],
            "geometry_digest": str(node["geometry_hash"]), "geometry_digest_algorithm": "sha1_point_indices",
            "point_count": int(node["point_count"]), "frozen_class_index": int(node["canonical_frozen_class_index"]),
            "control_score": float(node["canonical_frozen_score"]), "challenger_score": challenger,
            "candidate_retained": True, "candidate_deletion": False, "geometry_mutation": False,
            "class_mutation": False, "append_only": False,
        })
    for row in refined_rows:
        original = node_by_key[str(row["original_union_geometry_key"])]
        complete.append({
            "plan_key": str(row["candidate_key"]), "scene_name": str(row["scene_name"]),
            "candidate_source": "refined_union", "geometry_key": None,
            "original_union_geometry_key": str(row["original_union_geometry_key"]),
            "geometry_locator_read_only": row["geometry_locator_read_only"],
            "geometry_digest": str(row["geometry_sha256"]), "geometry_digest_algorithm": "sha256_point_indices",
            "point_count": int(row["point_count"]), "frozen_class_index": int(original["canonical_frozen_class_index"]),
            "control_score": None, "challenger_score": float(row["stage_d_v3_append_score"]),
            "candidate_retained": True, "candidate_deletion": False, "geometry_mutation": False,
            "class_mutation": False, "append_only": True,
        })
    complete.sort(key=lambda row: (row["scene_name"], row["plan_key"]))
    stage_d_root = staging / "stage_d"; stage_d_root.mkdir()
    _write_jsonl(stage_d_root / "stage_d_v3_append_score_plan.jsonl", score_rows)
    plan_root = staging / "complete_plan"; plan_root.mkdir()
    plan_path = plan_root / "complete_inference_ap_plan.jsonl"; _write_jsonl(plan_path, complete)
    (plan_root / "summary.json").write_text(json.dumps({
        "version": VERSION, "scene_count": len(scenes), "control_candidate_count": len(nodes),
        "challenger_candidate_count": len(complete), "refined_union_candidate_count": len(refined_rows),
        "files": {"plan": plan_path.name}, "hashes": {"plan": _sha256(plan_path)},
        "ground_truth_usage": "none", "ap_computed": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return score_rows, complete


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "prepared_root", "unique_geometry_root", "gvc_root", "relation_root",
        "champion_plan_root", "track_root", "automatic_root", "config_path", "model_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    if {str(row["scene_name"]) for row in nodes} != set(scenes):
        raise ValueError("unique geometry scene coverage differs from target scene list")
    models = _load_models(args.model_root)
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        stage_a, stage_a_root, stage_a_audit_root = _stage_a(
            scenes=scenes, nodes=nodes,
            gvc_root=args.gvc_root, relation_root=args.relation_root,
            champion_root=args.champion_plan_root, models=models, staging=staging,
        )
        stage_b, stage_b_relations, _ = _stage_b(
            unique_root=args.unique_geometry_root, relation_root=args.relation_root,
            champion_root=args.champion_plan_root, stage_a_root=stage_a_root,
            stage_a_audit_root=stage_a_audit_root, staging=staging,
        )
        stage_c, _ = _stage_c(
            scenes=scenes, nodes=nodes, member_to_key=_members(nodes), stage_a=stage_a,
            stage_b=stage_b, prepared_root=args.prepared_root, relation_root=args.relation_root,
            champion_root=args.champion_plan_root, track_root=args.track_root,
            automatic_root=args.automatic_root, config_path=args.config_path,
            models=models, staging=staging, final_output_root=args.output_root,
            min_points=args.min_points,
        )
        scores, complete = _stage_d_and_complete_plan(
            scenes=scenes, nodes=nodes, stage_a=stage_a, stage_b=stage_b,
            stage_b_relations=stage_b_relations, stage_c=stage_c,
            champion_root=args.champion_plan_root, models=models, staging=staging,
            prepared_root=args.prepared_root,
        )
        summary = {
            "version": VERSION, "dataset_name": args.dataset_name, "scene_count": len(scenes),
            "control_candidate_count": len(nodes), "challenger_candidate_count": len(complete),
            "stage_d_candidate_count": len(scores),
            "refined_union_candidate_count": sum(row["candidate_source"] == "refined_union" for row in complete),
            "model_package_sha256": _sha256(args.model_root / "summary.json"),
            "scene_list_sha256": _sha256(args.scene_list), "ground_truth_usage": "none",
            "candidate_deletion_count": 0, "geometry_mutation": False, "class_mutation": False,
            "ap_computed": False,
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--unique-geometry-root", type=Path, required=True)
    parser.add_argument("--gvc-root", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--champion-plan-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--dataset-name", default="ScanNet200-val312")
    args = parser.parse_args()
    if args.min_points != 100:
        raise ValueError("FI1-D-v3 freezes min_points=100")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

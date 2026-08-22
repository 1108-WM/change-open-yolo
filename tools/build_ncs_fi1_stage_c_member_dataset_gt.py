#!/usr/bin/env python3
"""Build the preregistered train100 pair-union member-refinement dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_automatic_sam_track_growth_ledger import (  # noqa: E402
    _normalize_normals,
    _normalize_rgb,
    _raw_superpoint_context,
)
from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver,
    _fold_by_scene,
    _quality_target,
    _read_jsonl,
    _read_scenes,
    _resolve,
    _sha256,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402


VERSION = "ncs_fi1_stage_c_member_dataset_v1"
ROLES = ("shared", "track_only", "native_only")
ADJACENCY = {
    "knn": 12,
    "max_distance": 0.05,
    "min_contact_points": 3,
    "min_contact_ratio": 0.02,
}
RELATION_FEATURES = (
    "point_iou",
    "native_inside_track_ratio",
    "track_inside_native_ratio",
    "aabb_iou",
    "centroid_distance_normalized",
    "native_shared_superpoint_fraction",
    "track_shared_superpoint_fraction",
    "public_common_selected_view_count",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "public_projected_box_iou_mean",
    "public_native_gvc_mean",
    "public_track_gvc_mean",
    "public_track_minus_native_gvc",
    "public_same_observation_both_depth_consistent_count",
    "exclusive_boundary_contact_ratio_mean",
    "exclusive_boundary_distance_weighted_mean",
    "exclusive_boundary_normal_difference_weighted_mean",
    "exclusive_boundary_color_difference_weighted_mean",
    "mean_rgb_distance",
    "mean_normal_difference",
)
FEATURE_NAMES = (
    "role_shared",
    "role_track_only",
    "role_native_only",
    "log1p_atom_point_count",
    "atom_fraction_of_union",
    "atom_fraction_of_track",
    "atom_fraction_of_native",
    "atom_fraction_of_raw_superpoint",
    "atom_centroid_distance_to_union_normalized",
    "atom_rgb_distance_to_union",
    "atom_normal_difference_to_union",
    "raw_superpoint_degree",
    "raw_superpoint_boundary_contact_count",
    "raw_superpoint_boundary_contact_ratio_mean",
    "raw_superpoint_boundary_distance_mean",
    "raw_superpoint_boundary_color_difference_mean",
    "raw_superpoint_boundary_normal_difference_mean",
    "stage_a_union_quality",
    "stage_a_track_quality",
    "stage_a_native_quality",
    "stage_b_union_score",
    *tuple(f"relation__{name}" for name in RELATION_FEATURES),
)


def _geometry_sha256(points: np.ndarray) -> str:
    values = np.unique(np.asarray(points, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def decompose_union_atoms(
    union_points: np.ndarray,
    track_points: np.ndarray,
    native_points: np.ndarray,
    raw_superpoint_ids: np.ndarray,
) -> list[dict]:
    """Partition one union into disjoint raw-superpoint-by-role atoms."""
    union = np.unique(np.asarray(union_points, dtype=np.int64))
    track = np.unique(np.asarray(track_points, dtype=np.int64))
    native = np.unique(np.asarray(native_points, dtype=np.int64))
    raw = np.asarray(raw_superpoint_ids, dtype=np.int64)
    if len(raw) <= int(union[-1]):
        raise ValueError("raw superpoint array does not cover union points")
    if not np.array_equal(union, np.union1d(track, native)):
        raise ValueError("frozen union differs from the union of its parents")
    in_track = np.zeros(len(raw), dtype=bool)
    in_native = np.zeros(len(raw), dtype=bool)
    in_track[track] = True
    in_native[native] = True
    roles = np.full(len(raw), -1, dtype=np.int8)
    roles[in_track & in_native] = 0
    roles[in_track & ~in_native] = 1
    roles[~in_track & in_native] = 2
    atoms = []
    for raw_id in np.unique(raw[union]):
        local = union[raw[union] == raw_id]
        for role_index, role in enumerate(ROLES):
            points = local[roles[local] == role_index]
            if len(points):
                atoms.append({
                    "raw_superpoint_id": int(raw_id),
                    "role": role,
                    "points": points,
                })
    concatenated = np.concatenate([row["points"] for row in atoms]) if atoms else np.empty(0, np.int64)
    if len(concatenated) != len(union) or not np.array_equal(np.sort(concatenated), union):
        raise AssertionError("atom partition does not conserve the frozen union")
    return atoms


def _mean_edge(edges: list[dict], name: str) -> float:
    return float(np.mean([float(row[name]) for row in edges])) if edges else 0.0


def _atom_features(
    atom: dict,
    union: np.ndarray,
    track: np.ndarray,
    native: np.ndarray,
    processed: np.ndarray,
    context: dict,
    qualities: dict,
    relation: dict,
) -> dict[str, float]:
    points = atom["points"]
    raw_id = int(atom["raw_superpoint_id"])
    raw_ids = np.asarray(processed[:, 9], dtype=np.int64)
    raw_size = int(np.count_nonzero(raw_ids == raw_id))
    xyz = np.asarray(processed[:, :3], dtype=np.float64)
    rgb = _normalize_rgb(processed[:, 3:6])
    normals = _normalize_normals(processed[:, 6:9])
    union_extent = xyz[union].max(axis=0) - xyz[union].min(axis=0)
    scale = max(1e-8, float(np.linalg.norm(union_extent)))
    atom_normal = _normalize_normals(normals[points].mean(axis=0, keepdims=True))[0]
    union_normal = _normalize_normals(normals[union].mean(axis=0, keepdims=True))[0]
    edges = list(context["neighbors"].get(raw_id, []))
    relation_raw = relation["features"]
    values = {
        "role_shared": float(atom["role"] == "shared"),
        "role_track_only": float(atom["role"] == "track_only"),
        "role_native_only": float(atom["role"] == "native_only"),
        "log1p_atom_point_count": math.log1p(len(points)),
        "atom_fraction_of_union": len(points) / max(1, len(union)),
        "atom_fraction_of_track": len(points) / max(1, len(track)),
        "atom_fraction_of_native": len(points) / max(1, len(native)),
        "atom_fraction_of_raw_superpoint": len(points) / max(1, raw_size),
        "atom_centroid_distance_to_union_normalized": float(
            np.linalg.norm(xyz[points].mean(axis=0) - xyz[union].mean(axis=0)) / scale
        ),
        "atom_rgb_distance_to_union": float(
            np.linalg.norm(rgb[points].mean(axis=0) - rgb[union].mean(axis=0)) / math.sqrt(3.0)
        ),
        "atom_normal_difference_to_union": float(
            1.0 - abs(float(np.dot(atom_normal, union_normal)))
        ),
        "raw_superpoint_degree": float(len(edges)),
        "raw_superpoint_boundary_contact_count": float(sum(
            int(row["boundary_contact_count"]) for row in edges
        )),
        "raw_superpoint_boundary_contact_ratio_mean": _mean_edge(edges, "boundary_contact_ratio"),
        "raw_superpoint_boundary_distance_mean": _mean_edge(edges, "mean_boundary_distance"),
        "raw_superpoint_boundary_color_difference_mean": _mean_edge(edges, "mean_color_difference"),
        "raw_superpoint_boundary_normal_difference_mean": _mean_edge(edges, "mean_normal_difference"),
        **qualities,
        **{
            f"relation__{name}": float(relation_raw.get(name, 0.0))
            for name in RELATION_FEATURES
        },
    }
    output = {name: float(values[name]) for name in FEATURE_NAMES}
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("non-finite stage-C atom feature")
    return output


def run(args: argparse.Namespace) -> dict:
    dataset_name = str(args.dataset_name)
    for name in (
        "scene_list", "fold_manifest", "prepared_root", "ground_truth_root",
        "unique_geometry_root", "champion_plan_root", "relation_root",
        "stage_a_oof_root", "stage_b_root", "stage_b_audit_root",
        "preregistration", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    folds = _fold_by_scene(args.fold_manifest, scenes)
    stage_b_audit = json.loads((args.stage_b_audit_root / "summary.json").read_text())
    if stage_b_audit.get("audit_valid") is not True or stage_b_audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("stage-B independent audit did not authorize stage C")

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    member_to_key = {}
    for node in nodes:
        for member in node["members"]:
            key = (str(node["scene_name"]), str(member["candidate_source"]), int(member["candidate_id"]))
            if key in member_to_key:
                raise ValueError(f"duplicate unique-geometry member: {key}")
            member_to_key[key] = str(node["geometry_key"])
    oof_summary = json.loads((args.stage_a_oof_root / "summary.json").read_text())
    predictions = _read_jsonl(args.stage_a_oof_root / oof_summary["prediction_file"])
    quality_by_key = {str(row["geometry_key"]): float(row["oof_unified_quality"]) for row in predictions}
    stage_b_summary = json.loads((args.stage_b_root / "summary.json").read_text())
    stage_b_plan = _read_jsonl(args.stage_b_root / stage_b_summary["files"]["plan"])
    stage_b_by_key = {str(row["geometry_key"]): float(row["stage_b_score"]) for row in stage_b_plan}
    unions = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    unions_by_scene = {scene: [] for scene in scenes}
    for row in unions:
        unions_by_scene[str(row["scene_name"])].append(row)

    resolver = GeometryResolver()
    atom_rows = []
    union_rows = []
    role_counts = Counter()
    fold_atom_counts = Counter()
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        for scene_index, scene in enumerate(scenes, 1):
            scene_stem = scene[len("scene"):] if scene.startswith("scene") else scene
            prepared_path = args.prepared_root / scene / f"{scene_stem}.npy"
            processed_raw = np.load(prepared_path, mmap_mode="r")
            if processed_raw.ndim != 2 or processed_raw.shape[1] < 10:
                raise ValueError(f"{scene}: prepared scene lacks inference columns 0:10")
            processed = np.asarray(processed_raw[:, :10])
            gt_path = args.ground_truth_root / f"{scene}.txt"
            gt, gt_meta = _load_gt(gt_path, args.min_gt_points)
            if len(gt) != len(processed):
                raise ValueError(f"{scene}: GT and prepared point counts differ")
            context = _raw_superpoint_context(
                processed, ADJACENCY["knn"], ADJACENCY["max_distance"],
                ADJACENCY["min_contact_points"], ADJACENCY["min_contact_ratio"],
            )
            relation_by_key = {}
            for relation in _read_jsonl(args.relation_root / scene / "relation_features_no_gt.jsonl"):
                key = (int(relation["track_id"]), str(relation["native_exact_geometry_group_id"]))
                if key in relation_by_key:
                    raise ValueError(f"{scene}: duplicate direct relation {key}")
                relation_by_key[key] = relation
            for union in sorted(unions_by_scene[scene], key=lambda row: int(row["candidate_id"])):
                union_id = int(union["candidate_id"])
                track_id = int(union["track_id"])
                native_ids = list(map(int, union["native_member_candidate_ids"]))
                union_key = member_to_key[(scene, "pair_union", union_id)]
                track_key = member_to_key[(scene, "track", track_id)]
                native_keys = {member_to_key[(scene, "native", member_id)] for member_id in native_ids}
                if len(native_keys) != 1:
                    raise ValueError(f"{scene}/{union_id}: native parent maps to multiple geometries")
                native_key = next(iter(native_keys))
                union_node, track_node, native_node = (
                    node_by_key[union_key], node_by_key[track_key], node_by_key[native_key]
                )
                union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), len(gt))
                track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), len(gt))
                native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), len(gt))
                atoms = decompose_union_atoms(
                    union_points, track_points, native_points,
                    np.asarray(processed[:, 9], dtype=np.int64),
                )
                best = _best_gt(union_points, gt, gt_meta)
                target_gt = best["best_gt"]
                target_encoded = int(target_gt["encoded_id"]) if target_gt else None
                relation_key = (track_id, str(union["native_exact_geometry_group_id"]))
                relation = relation_by_key[relation_key]
                qualities = {
                    "stage_a_union_quality": quality_by_key[union_key],
                    "stage_a_track_quality": quality_by_key[track_key],
                    "stage_a_native_quality": quality_by_key[native_key],
                    "stage_b_union_score": stage_b_by_key[union_key],
                }
                local_roles = Counter()
                for atom_index, atom in enumerate(atoms):
                    points = atom["points"]
                    purity = float(np.mean(gt[points] == target_encoded)) if target_encoded is not None else 0.0
                    atom_id = f"{scene}:union:{union_id}:atom:{atom_index}"
                    atom_rows.append({
                        "atom_id": atom_id,
                        "scene_name": scene,
                        "fold_index": int(folds[scene]),
                        "union_candidate_id": union_id,
                        "union_geometry_key": union_key,
                        "track_geometry_key": track_key,
                        "native_geometry_key": native_key,
                        "raw_superpoint_id": int(atom["raw_superpoint_id"]),
                        "role": atom["role"],
                        "point_count": len(points),
                        "point_sha256": _geometry_sha256(points),
                        "features": _atom_features(
                            atom, union_points, track_points, native_points,
                            processed, context, qualities, relation,
                        ),
                        "label_target_gt_encoded_id": target_encoded,
                        "label_target_gt_iou_of_original_union": float(best["best_iou"]),
                        "label_retention_probability": purity,
                        "ground_truth_usage": f"{dataset_name} atom label fields only",
                        "candidate_mutation": False,
                        "frozen_geometry_mutation": False,
                        "ap_computed": False,
                    })
                    role_counts[atom["role"]] += 1
                    local_roles[atom["role"]] += 1
                    fold_atom_counts[int(folds[scene])] += 1
                union_rows.append({
                    "scene_name": scene,
                    "fold_index": int(folds[scene]),
                    "union_candidate_id": union_id,
                    "union_geometry_key": union_key,
                    "track_geometry_key": track_key,
                    "native_geometry_key": native_key,
                    "original_union_point_count": len(union_points),
                    "original_union_sha256": _geometry_sha256(union_points),
                    "atom_count": len(atoms),
                    "role_atom_counts": {role: int(local_roles[role]) for role in ROLES},
                    "atom_point_count_sum": int(sum(len(atom["points"]) for atom in atoms)),
                    "label_original_union_best_iou": float(best["best_iou"]),
                    "label_original_union_quality_q": _quality_target(float(best["best_iou"])),
                    "original_union_retained": True,
                    "append_only": True,
                    "ap_computed": False,
                })
            print(f"[stage C member dataset] {scene_index}/{len(scenes)} {scene}: {len(unions_by_scene[scene])} unions", flush=True)

        atom_rows.sort(key=lambda row: row["atom_id"])
        union_rows.sort(key=lambda row: (row["scene_name"], row["union_candidate_id"]))
        atom_path = staging / "member_atoms.jsonl"
        union_path = staging / "pair_unions.jsonl"
        atom_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in atom_rows))
        union_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in union_rows))
        summary = {
            "version": VERSION,
            "dataset_name": dataset_name,
            "preregistration_status": "frozen_before_first_run",
            "scene_count": len(scenes),
            "union_count": len(union_rows),
            "atom_count": len(atom_rows),
            "role_atom_counts": {role: int(role_counts[role]) for role in ROLES},
            "fold_atom_counts": {str(fold): int(fold_atom_counts[fold]) for fold in range(5)},
            "feature_names": list(FEATURE_NAMES),
            "adjacency_parameters": ADJACENCY,
            "atom_target": "fraction of atom points belonging to original union best class-agnostic GT",
            "files": {"atoms": atom_path.name, "unions": union_path.name},
            "hashes": {"atoms": _sha256(atom_path), "unions": _sha256(union_path)},
            "ground_truth_usage": f"{dataset_name} label fields only",
            "feature_ground_truth_usage": "none",
            "original_union_retained_count": len(union_rows),
            "candidate_mutation": False,
            "frozen_geometry_mutation": False,
            "frozen_cache_write": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration),
                "scene_list_sha256": _sha256(args.scene_list),
                "fold_manifest_sha256": _sha256(args.fold_manifest),
                "unique_geometry_summary_sha256": _sha256(args.unique_geometry_root / "summary.json"),
                "champion_plan_summary_sha256": _sha256(args.champion_plan_root / "summary.json"),
                "relation_summary_sha256": _sha256(args.relation_root / "summary.json"),
                "stage_a_oof_summary_sha256": _sha256(args.stage_a_oof_root / "summary.json"),
                "stage_b_summary_sha256": _sha256(args.stage_b_root / "summary.json"),
                "stage_b_audit_sha256": _sha256(args.stage_b_audit_root / "summary.json"),
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"))
    parser.add_argument("--fold-manifest", type=Path, default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--champion-plan-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"))
    parser.add_argument("--relation-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/relation_ledger"))
    parser.add_argument("--stage-a-oof-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_20260822"))
    parser.add_argument("--stage-b-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_20260822"))
    parser.add_argument("--stage-b-audit-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_audit_20260822"))
    parser.add_argument("--preregistration", type=Path, default=Path("docs/NCS_FI1_STAGE_C_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    parser.add_argument("--dataset-name", default="NCS-train100")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "union_count": result["union_count"],
        "atom_count": result["atom_count"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

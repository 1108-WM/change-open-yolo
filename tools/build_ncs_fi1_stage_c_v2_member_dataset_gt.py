#!/usr/bin/env python3
"""Build C-v2 member-level multiview/depth evidence and removal-utility labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_details_same_frame_hierarchy_preprocessor import (  # noqa: E402
    decode_binary_mask_rle,
)
from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver, _quality_target, _read_jsonl, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import (  # noqa: E402
    ROLES, _geometry_sha256, decompose_union_atoms,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402
from tools.export_mv3dis_relative_depth_observations import (  # noqa: E402
    PROJECTION_CONTRACT, mask_points_from_projection, project_relative_depth_frame,
)


VERSION = "ncs_fi1_stage_c_v2_member_dataset_v1"
MEMBER_EVIDENCE_FEATURE_NAMES = (
    "member_log1p_track_observation_count",
    "member_source_supported_observation_fraction",
    "member_source_coverage_mean",
    "member_source_coverage_min",
    "member_source_coverage_max",
    "member_source_predicted_iou_weighted_coverage",
    "member_source_stability_weighted_coverage",
    "member_source_point_support_mean",
    "member_source_point_multiview_fraction",
    "member_source_point_unobserved_fraction",
    "member_relative_visible_observation_fraction",
    "member_relative_mask_coverage_mean",
    "member_relative_mask_coverage_min",
    "member_relative_mask_coverage_max",
    "member_relative_depth_weighted_coverage_mean",
    "member_relative_depth_weighted_coverage_min",
    "member_relative_depth_weighted_coverage_max",
    "member_relative_inside_depth_weight_mean",
    "member_relative_point_visible_view_mean",
    "member_relative_point_inside_view_mean",
    "member_relative_point_inside_given_visible_mean",
    "member_relative_point_multiview_inside_fraction",
    "member_relative_point_visible_but_outside_fraction",
)


def fixed_target_removal_labels(
    *, union_point_count: int, atom_point_count: int, target_point_count: int,
    union_target_intersection: int, atom_target_intersection: int,
) -> dict[str, float]:
    """Return signed fixed-target IoU/Q changes caused by deleting one atom."""
    before_union = union_point_count + target_point_count - union_target_intersection
    before_iou = union_target_intersection / max(1, before_union)
    after_count = union_point_count - atom_point_count
    after_intersection = union_target_intersection - atom_target_intersection
    after_union = after_count + target_point_count - after_intersection
    after_iou = after_intersection / max(1, after_union)
    return {
        "fixed_target_iou_before": float(before_iou),
        "fixed_target_iou_after_removal": float(after_iou),
        "delta_iou_remove": float(after_iou - before_iou),
        "fixed_target_q_before": float(_quality_target(before_iou)),
        "fixed_target_q_after_removal": float(_quality_target(after_iou)),
        "delta_q_remove": float(_quality_target(after_iou) - _quality_target(before_iou)),
    }


def _summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.min()), float(array.max())


def member_evidence_features(
    atom_local_indexes: np.ndarray,
    source_support: list[np.ndarray],
    relative_visible: list[np.ndarray],
    relative_inside: list[np.ndarray],
    relative_weights: list[np.ndarray],
    predicted_ious: list[float],
    stability_scores: list[float],
) -> dict[str, float]:
    """Aggregate genuinely atom-local evidence across the parent track views."""
    indexes = np.asarray(atom_local_indexes, dtype=np.int64)
    observation_count = len(source_support)
    if not (
        len(relative_visible) == len(relative_inside) == len(relative_weights)
        == len(predicted_ious) == len(stability_scores) == observation_count
    ):
        raise ValueError("member evidence observation arrays differ in length")
    point_count = len(indexes)
    source_coverages = []
    visible_coverages = []
    mask_coverages = []
    weighted_coverages = []
    inside_weight_means = []
    point_source_support = np.zeros(point_count, dtype=np.int32)
    point_visible_support = np.zeros(point_count, dtype=np.int32)
    point_inside_support = np.zeros(point_count, dtype=np.int32)
    for source, visible, inside, weights in zip(
        source_support, relative_visible, relative_inside, relative_weights
    ):
        local_source = np.asarray(source, dtype=bool)[indexes]
        local_visible = np.asarray(visible, dtype=bool)[indexes]
        local_inside = np.asarray(inside, dtype=bool)[indexes]
        local_weights = np.asarray(weights, dtype=np.float64)[indexes]
        if np.any(local_inside & ~local_visible):
            raise ValueError("relative mask support occurs outside relative visibility")
        source_count = int(local_source.sum())
        visible_count = int(local_visible.sum())
        inside_count = int(local_inside.sum())
        source_coverages.append(source_count / max(1, point_count))
        point_source_support += local_source
        point_visible_support += local_visible
        point_inside_support += local_inside
        if visible_count:
            visible_coverages.append(visible_count / max(1, point_count))
            mask_coverages.append(inside_count / visible_count)
            weighted_coverages.append(float(local_weights[local_inside].sum() / visible_count))
            inside_weight_means.append(float(
                local_weights[local_inside].mean() if inside_count else 0.0
            ))
    source_mean, source_min, source_max = _summary(source_coverages)
    mask_mean, mask_min, mask_max = _summary(mask_coverages)
    weighted_mean, weighted_min, weighted_max = _summary(weighted_coverages)
    inside_weight_mean = float(np.mean(inside_weight_means)) if inside_weight_means else 0.0
    predicted_weight = float(sum(predicted_ious))
    stability_weight = float(sum(stability_scores))
    conditional = np.divide(
        point_inside_support,
        np.maximum(point_visible_support, 1),
        dtype=np.float64,
    )
    output = {
        "member_log1p_track_observation_count": math.log1p(observation_count),
        "member_source_supported_observation_fraction": float(
            np.mean(np.asarray(source_coverages) > 0.0) if observation_count else 0.0
        ),
        "member_source_coverage_mean": source_mean,
        "member_source_coverage_min": source_min,
        "member_source_coverage_max": source_max,
        "member_source_predicted_iou_weighted_coverage": float(
            np.dot(source_coverages, predicted_ious) / max(predicted_weight, 1e-12)
            if observation_count else 0.0
        ),
        "member_source_stability_weighted_coverage": float(
            np.dot(source_coverages, stability_scores) / max(stability_weight, 1e-12)
            if observation_count else 0.0
        ),
        "member_source_point_support_mean": float(
            point_source_support.mean() if point_count else 0.0
        ),
        "member_source_point_multiview_fraction": float(
            np.mean(point_source_support >= 2) if point_count else 0.0
        ),
        "member_source_point_unobserved_fraction": float(
            np.mean(point_source_support == 0) if point_count else 0.0
        ),
        "member_relative_visible_observation_fraction": float(
            len(visible_coverages) / max(1, observation_count)
        ),
        "member_relative_mask_coverage_mean": mask_mean,
        "member_relative_mask_coverage_min": mask_min,
        "member_relative_mask_coverage_max": mask_max,
        "member_relative_depth_weighted_coverage_mean": weighted_mean,
        "member_relative_depth_weighted_coverage_min": weighted_min,
        "member_relative_depth_weighted_coverage_max": weighted_max,
        "member_relative_inside_depth_weight_mean": inside_weight_mean,
        "member_relative_point_visible_view_mean": float(
            point_visible_support.mean() if point_count else 0.0
        ),
        "member_relative_point_inside_view_mean": float(
            point_inside_support.mean() if point_count else 0.0
        ),
        "member_relative_point_inside_given_visible_mean": float(
            conditional.mean() if point_count else 0.0
        ),
        "member_relative_point_multiview_inside_fraction": float(
            np.mean(point_inside_support >= 2) if point_count else 0.0
        ),
        "member_relative_point_visible_but_outside_fraction": float(
            np.mean((point_visible_support > 0) & (point_inside_support == 0))
            if point_count else 0.0
        ),
    }
    if tuple(output) != MEMBER_EVIDENCE_FEATURE_NAMES:
        raise AssertionError("member evidence feature order differs")
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("non-finite member evidence feature")
    return output


def _union_observation_arrays(
    *, union_points: np.ndarray, observations: list[dict], world: object,
    intrinsic: np.ndarray, automatic_scene_root: Path, depth_cache: dict[int, np.ndarray],
    depth_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    union = np.asarray(union_points, dtype=np.int64)
    points_h = np.column_stack((
        np.asarray(world.points_xyz[union], dtype=np.float64),
        np.ones(len(union), dtype=np.float64),
    ))
    source_arrays = []
    visible_arrays = []
    inside_arrays = []
    weight_arrays = []
    for observation in observations:
        frame_index = int(observation["frame_index"])
        depth_map = depth_cache.get(frame_index)
        if depth_map is None:
            depth_map = np.asarray(
                imageio.imread(world.depth_maps_paths[frame_index]), dtype=np.float64
            ) / depth_scale
            depth_cache[frame_index] = depth_map
        pixels, visible, weights = project_relative_depth_frame(
            points_h, np.loadtxt(world.poses[frame_index]), intrinsic, depth_map,
        )
        mask = decode_binary_mask_rle(observation["mask_rle"])
        inside_local, inside_weights = mask_points_from_projection(
            mask, pixels, visible, weights, world.depth_resolution,
        )
        inside = np.zeros(len(union), dtype=bool)
        inside[inside_local] = True
        local_weights = np.zeros(len(union), dtype=np.float32)
        local_weights[inside_local] = inside_weights
        points_path = Path(observation["point_indices_path"])
        if not points_path.is_file():
            points_path = automatic_scene_root / "points" / f"obs{int(observation['observation_id']):06d}_points.npz"
        with np.load(points_path) as payload:
            source_points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        positions = np.searchsorted(union, source_points)
        valid = (positions < len(union)) & (union[np.minimum(positions, len(union) - 1)] == source_points)
        source = np.zeros(len(union), dtype=bool)
        source[positions[valid]] = True
        source_arrays.append(source)
        visible_arrays.append(np.asarray(visible, dtype=bool))
        inside_arrays.append(inside)
        weight_arrays.append(local_weights)
    return source_arrays, visible_arrays, inside_arrays, weight_arrays


def run(args: argparse.Namespace) -> dict:
    dataset_name = str(args.dataset_name)
    for name in (
        "base_dataset_root", "base_dataset_audit_root", "prepared_root",
        "ground_truth_root", "unique_geometry_root", "track_root", "automatic_root",
        "config_path", "preregistration", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    base_summary_path = args.base_dataset_root / "summary.json"
    base_summary = json.loads(base_summary_path.read_text())
    base_audit_path = args.base_dataset_audit_root / "summary.json"
    base_audit = json.loads(base_audit_path.read_text())
    if base_audit.get("audit_valid") is not True or base_audit.get("error_count") != 0:
        raise ValueError("C-v1 member dataset audit is not valid")
    base_atoms_path = args.base_dataset_root / base_summary["files"]["atoms"]
    base_unions_path = args.base_dataset_root / base_summary["files"]["unions"]
    base_atoms = _read_jsonl(base_atoms_path)
    base_unions = _read_jsonl(base_unions_path)
    atom_rows_by_union = {}
    for row in base_atoms:
        key = (str(row["scene_name"]), int(row["union_candidate_id"]))
        atom_rows_by_union.setdefault(key, []).append(row)
    for rows in atom_rows_by_union.values():
        rows.sort(key=lambda row: int(str(row["atom_id"]).rsplit(":", 1)[1]))
    union_rows_by_scene = {}
    for row in base_unions:
        union_rows_by_scene.setdefault(str(row["scene_name"]), []).append(row)

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    config = yaml.safe_load(args.config_path.read_text())
    depth_scale = float(config["openyolo3d"]["depth_scale"])
    resolver = GeometryResolver()
    output_rows = []
    union_output_rows = []
    role_counts = Counter()
    fold_counts = Counter()
    distinct_delta_values = set()
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        from utils import WORLD_2_CAM

        scenes = sorted(union_rows_by_scene)
        for scene_index, scene in enumerate(scenes, 1):
            stem = scene[len("scene"):] if scene.startswith("scene") else scene
            processed = np.load(args.prepared_root / scene / f"{stem}.npy", mmap_mode="r")
            gt, gt_meta = _load_gt(args.ground_truth_root / f"{scene}.txt", args.min_gt_points)
            if len(processed) != len(gt):
                raise ValueError(f"{scene}: prepared/GT point counts differ")
            world = WORLD_2_CAM(str(args.prepared_root / scene), depth_scale, config)
            world.points_xyz = np.asarray(processed[:, :3], dtype=np.float64)
            intrinsic = world.adjust_intrinsic(
                np.loadtxt(world.intrinsics[0]), world.image_resolution, world.depth_resolution
            )
            tracks_payload = json.loads((args.track_root / scene / "automatic_tracks.json").read_text())
            track_by_id = {int(row["track_id"]): row for row in tracks_payload["tracks"]}
            automatic_scene = args.automatic_root / scene
            observation_by_id = {
                int(row["observation_id"]): row
                for row in _read_jsonl(automatic_scene / "automatic_observations.jsonl")
            }
            depth_cache = {}
            for union in sorted(union_rows_by_scene[scene], key=lambda row: int(row["union_candidate_id"])):
                union_id = int(union["union_candidate_id"])
                base_rows = atom_rows_by_union[(scene, union_id)]
                union_node = node_by_key[str(union["union_geometry_key"])]
                track_node = node_by_key[str(union["track_geometry_key"])]
                native_node = node_by_key[str(union["native_geometry_key"])]
                union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), len(gt))
                track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), len(gt))
                native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), len(gt))
                atoms = decompose_union_atoms(
                    union_points, track_points, native_points,
                    np.asarray(processed[:, 9], dtype=np.int64),
                )
                if len(atoms) != len(base_rows):
                    raise ValueError(f"{scene}/{union_id}: C-v1 atom count differs")
                track_id = int(track_node["canonical_candidate_id"])
                observation_ids = list(map(int, track_by_id[track_id].get("observation_ids", [])))
                observations = [observation_by_id[observation_id] for observation_id in observation_ids]
                source_support, visible, inside, weights = _union_observation_arrays(
                    union_points=union_points,
                    observations=observations,
                    world=world,
                    intrinsic=intrinsic,
                    automatic_scene_root=automatic_scene,
                    depth_cache=depth_cache,
                    depth_scale=depth_scale,
                )
                predicted_ious = [float(row.get("predicted_iou", 0.0)) for row in observations]
                stability_scores = [float(row.get("stability_score", 0.0)) for row in observations]
                best = _best_gt(union_points, gt, gt_meta)
                target = best["best_gt"]
                target_encoded = int(target["encoded_id"]) if target else None
                target_point_count = int(target["point_count"]) if target else 0
                union_target_intersection = int(np.count_nonzero(gt[union_points] == target_encoded)) if target else 0
                local_role_counts = Counter()
                for atom_index, (atom, base_row) in enumerate(zip(atoms, base_rows)):
                    if _geometry_sha256(atom["points"]) != str(base_row["point_sha256"]):
                        raise ValueError(f"{base_row['atom_id']}: atom geometry differs from C-v1")
                    atom_local = np.searchsorted(union_points, atom["points"])
                    evidence = member_evidence_features(
                        atom_local, source_support, visible, inside, weights,
                        predicted_ious, stability_scores,
                    )
                    atom_target_intersection = int(
                        np.count_nonzero(gt[atom["points"]] == target_encoded)
                    ) if target else 0
                    labels = fixed_target_removal_labels(
                        union_point_count=len(union_points),
                        atom_point_count=len(atom["points"]),
                        target_point_count=target_point_count,
                        union_target_intersection=union_target_intersection,
                        atom_target_intersection=atom_target_intersection,
                    )
                    features = {**base_row["features"], **evidence}
                    output_rows.append({
                        "atom_id": base_row["atom_id"],
                        "scene_name": scene,
                        "fold_index": int(base_row["fold_index"]),
                        "union_candidate_id": union_id,
                        "union_geometry_key": union["union_geometry_key"],
                        "track_geometry_key": union["track_geometry_key"],
                        "native_geometry_key": union["native_geometry_key"],
                        "raw_superpoint_id": int(atom["raw_superpoint_id"]),
                        "role": atom["role"],
                        "point_count": len(atom["points"]),
                        "point_sha256": _geometry_sha256(atom["points"]),
                        "track_observation_count": len(observations),
                        "features": features,
                        "label_fixed_target_gt_encoded_id": target_encoded,
                        "label_atom_target_intersection": atom_target_intersection,
                        **{f"label_{name}": value for name, value in labels.items()},
                        "ground_truth_usage": f"{dataset_name} fixed-target removal labels only",
                        "feature_ground_truth_usage": "none",
                        "relative_depth_contract": PROJECTION_CONTRACT,
                        "candidate_mutation": False,
                        "frozen_geometry_mutation": False,
                        "ap_computed": False,
                    })
                    role_counts[atom["role"]] += 1
                    local_role_counts[atom["role"]] += 1
                    fold_counts[int(base_row["fold_index"])] += 1
                    if atom["role"] != "shared":
                        distinct_delta_values.add(round(labels["delta_iou_remove"], 6))
                union_output_rows.append({
                    **union,
                    "fixed_target_gt_encoded_id": target_encoded,
                    "fixed_target_gt_point_count": target_point_count,
                    "fixed_target_union_intersection": union_target_intersection,
                    "fixed_target_union_iou": float(best["best_iou"]),
                    "fixed_target_union_q": float(_quality_target(float(best["best_iou"]))),
                    "track_observation_count": len(observations),
                    "role_atom_counts": {role: int(local_role_counts[role]) for role in ROLES},
                    "relative_depth_contract": PROJECTION_CONTRACT,
                })
            print(f"[stage C-v2 dataset] {scene_index}/{len(scenes)} {scene}: {len(union_rows_by_scene[scene])} unions", flush=True)
            del world, processed, depth_cache

        output_rows.sort(key=lambda row: row["atom_id"])
        union_output_rows.sort(key=lambda row: (row["scene_name"], row["union_candidate_id"]))
        atoms_path = staging / "member_removal_dataset.jsonl"
        unions_path = staging / "pair_unions.jsonl"
        atoms_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows))
        unions_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in union_output_rows))
        feature_names = list(base_summary["feature_names"]) + list(MEMBER_EVIDENCE_FEATURE_NAMES)
        summary = {
            "version": VERSION,
            "preregistration_status": "frozen_before_first_run",
            "dataset_scene_count": int(base_summary["scene_count"]),
            "involved_union_scene_count": len(scenes),
            "union_count": len(union_output_rows),
            "atom_count": len(output_rows),
            "role_atom_counts": {role: int(role_counts[role]) for role in ROLES},
            "fold_atom_counts": {str(fold): int(fold_counts[fold]) for fold in range(5)},
            "exclusive_distinct_delta_iou_remove_rounded_6_count": len(distinct_delta_values),
            "feature_names": feature_names,
            "member_evidence_feature_names": list(MEMBER_EVIDENCE_FEATURE_NAMES),
            "primary_target": "fixed-target delta_iou_remove",
            "secondary_target": "fixed-target delta_q_remove",
            "relative_depth_contract": PROJECTION_CONTRACT,
            "files": {"atoms": atoms_path.name, "unions": unions_path.name},
            "hashes": {"atoms": _sha256(atoms_path), "unions": _sha256(unions_path)},
            "dataset_name": dataset_name,
            "ground_truth_usage": f"{dataset_name} fixed-target removal labels only",
            "feature_ground_truth_usage": "none",
            "candidate_mutation": False,
            "frozen_geometry_mutation": False,
            "frozen_cache_write": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration),
                "base_dataset_summary_sha256": _sha256(base_summary_path),
                "base_dataset_audit_sha256": _sha256(base_audit_path),
                "base_atoms_sha256": _sha256(base_atoms_path),
                "base_unions_sha256": _sha256(base_unions_path),
                "unique_geometry_summary_sha256": _sha256(args.unique_geometry_root / "summary.json"),
                "config_sha256": _sha256(args.config_path),
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
    parser.add_argument("--base-dataset-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_c_member_dataset_train100_20260822"))
    parser.add_argument("--base-dataset-audit-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_c_member_dataset_train100_audit_20260822"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--track-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/d2b_tracks_filtered"))
    parser.add_argument("--automatic-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/sam_automatic_uniform30"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--preregistration", type=Path, default=Path("docs/NCS_FI1_STAGE_C_V2_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    parser.add_argument("--dataset-name", default="NCS-train100")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "union_count": result["union_count"],
        "atom_count": result["atom_count"],
        "distinct_delta_count": result["exclusive_distinct_delta_iou_remove_rounded_6_count"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

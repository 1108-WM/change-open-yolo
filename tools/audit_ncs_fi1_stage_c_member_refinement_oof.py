#!/usr/bin/env python3
"""Independently audit stage-C OOF predictions and refined-union materialization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_automatic_sam_track_growth_ledger import _raw_superpoint_context  # noqa: E402
from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver, _quality_target, _read_jsonl, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import (  # noqa: E402
    ADJACENCY, FEATURE_NAMES, _geometry_sha256, decompose_union_atoms,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402
from tools.train_ncs_fi1_stage_a_quality_oof import _calibrate, metrics  # noqa: E402
from tools.train_ncs_fi1_stage_c_member_refinement_oof import (  # noqa: E402
    QUALITY_FEATURE_NAMES, connected_refinement_indexes,
)


VERSION = "ncs_fi1_stage_c_member_refinement_oof_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _predict_model(payload: dict, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.clip(payload["regressor"].predict(matrix), 0.0, 1.0)
    calibrated = np.clip(_calibrate(payload["calibrator"], raw), 0.0, 1.0)
    return raw, calibrated


def run(args: argparse.Namespace) -> dict:
    for name in (
        "dataset_root", "result_root", "prepared_root", "ground_truth_root",
        "unique_geometry_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary = json.loads((args.dataset_root / "summary.json").read_text())
    result_summary_path = args.result_root / "summary.json"
    result_summary = json.loads(result_summary_path.read_text())
    atoms_path = args.dataset_root / dataset_summary["files"]["atoms"]
    unions_path = args.dataset_root / dataset_summary["files"]["unions"]
    member_path = args.result_root / result_summary["files"]["member_predictions"]
    plan_path = args.result_root / result_summary["files"]["refined_plan"]
    atoms = _read_jsonl(atoms_path)
    unions = _read_jsonl(unions_path)
    members = _read_jsonl(member_path)
    plan = _read_jsonl(plan_path)
    errors = Counter()
    if _sha256(member_path) != result_summary["hashes"]["member_predictions"]:
        errors["member_predictions_sha256_mismatch"] += 1
    if _sha256(plan_path) != result_summary["hashes"]["refined_plan"]:
        errors["refined_plan_sha256_mismatch"] += 1
    if len(atoms) != len(members):
        errors["member_prediction_count_mismatch"] += 1
    if len(unions) != len(plan):
        errors["refined_plan_count_mismatch"] += 1

    model_by_kind_fold = {}
    for record in result_summary["models"]:
        path = args.result_root / record["model_file"]
        if _sha256(path) != record["model_sha256"]:
            errors["model_sha256_mismatch"] += 1
        payload = joblib.load(path)
        key = (str(record["model_kind"]), int(record["outer_fold"]))
        if key in model_by_kind_fold:
            errors["duplicate_model_kind_fold"] += 1
        model_by_kind_fold[key] = payload
        if int(payload["outer_fold"]) != int(record["outer_fold"]):
            errors["model_outer_fold_mismatch"] += 1
        if int(payload["calibration_fold"]) != (int(record["outer_fold"]) + 1) % 5:
            errors["model_calibration_fold_mismatch"] += 1

    atom_matrix = np.asarray([
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in atoms
    ], dtype=np.float64)
    atom_target = np.asarray([float(row["label_retention_probability"]) for row in atoms])
    atom_folds = np.asarray([int(row["fold_index"]) for row in atoms], dtype=np.int8)
    audited_atom_prediction = np.full(len(atoms), np.nan)
    for fold in range(5):
        select = np.flatnonzero(atom_folds == fold)
        payload = model_by_kind_fold[("member_retention", fold)]
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            errors["member_model_feature_schema_mismatch"] += 1
        raw, calibrated = _predict_model(payload, atom_matrix[select])
        audited_atom_prediction[select] = calibrated
        for local, index in enumerate(select):
            row, prediction = atoms[index], members[index]
            if str(row["atom_id"]) != str(prediction["atom_id"]):
                errors["member_prediction_identity_mismatch"] += 1
            if not _close(prediction["raw_oof_retention_probability"], raw[local]):
                errors["member_raw_prediction_mismatch"] += 1
            if not _close(prediction["oof_retention_probability"], calibrated[local]):
                errors["member_calibrated_prediction_mismatch"] += 1
            if bool(prediction["shared_forced_keep"]) != (row["role"] == "shared"):
                errors["shared_forced_keep_flag_mismatch"] += 1
    if not np.isfinite(audited_atom_prediction).all():
        errors["member_prediction_coverage"] += 1

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    union_by_key = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in unions}
    plan_by_key = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in plan}
    if len(plan_by_key) != len(plan):
        errors["duplicate_refined_plan_key"] += 1
    if set(plan_by_key) != set(union_by_key):
        errors["refined_plan_coverage"] += 1
    atom_indexes_by_union = defaultdict(list)
    for index, row in enumerate(atoms):
        atom_indexes_by_union[(str(row["scene_name"]), int(row["union_candidate_id"]))].append(index)
    for indexes in atom_indexes_by_union.values():
        indexes.sort(key=lambda index: int(str(atoms[index]["atom_id"]).rsplit(":", 1)[1]))

    resolver = GeometryResolver()
    scene_cache = {}
    deltas = []
    fold_deltas = defaultdict(list)
    status_counts = Counter()
    appended_count = 0
    for key, ledger in sorted(union_by_key.items()):
        scene, union_id = key
        row = plan_by_key[key]
        cached = scene_cache.get(scene)
        if cached is None:
            stem = scene[len("scene"):] if scene.startswith("scene") else scene
            processed = np.load(args.prepared_root / scene / f"{stem}.npy", mmap_mode="r")
            gt, gt_meta = _load_gt(args.ground_truth_root / f"{scene}.txt", args.min_gt_points)
            context = _raw_superpoint_context(
                np.asarray(processed[:, :10]), ADJACENCY["knn"], ADJACENCY["max_distance"],
                ADJACENCY["min_contact_points"], ADJACENCY["min_contact_ratio"],
            )
            cached = (processed, gt, gt_meta, context)
            scene_cache[scene] = cached
        processed, gt, gt_meta, context = cached
        union_node = node_by_key[str(ledger["union_geometry_key"])]
        track_node = node_by_key[str(ledger["track_geometry_key"])]
        native_node = node_by_key[str(ledger["native_geometry_key"])]
        union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), len(gt))
        track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), len(gt))
        native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), len(gt))
        decomposed = decompose_union_atoms(
            union_points, track_points, native_points,
            np.asarray(processed[:, 9], dtype=np.int64),
        )
        indexes = atom_indexes_by_union[key]
        probabilities = [float(audited_atom_prediction[index]) for index in indexes]
        if len(indexes) != len(decomposed):
            errors["materialization_atom_count_mismatch"] += 1
            continue
        for atom, index in zip(decomposed, indexes):
            if _geometry_sha256(atom["points"]) != atoms[index]["point_sha256"]:
                errors["materialization_atom_hash_mismatch"] += 1
        kept = connected_refinement_indexes(
            decomposed, probabilities, context["neighbors"],
            float(result_summary["retention_threshold"]),
        )
        shared = {index for index, atom in enumerate(decomposed) if atom["role"] == "shared"}
        if not shared <= kept:
            errors["shared_atom_deleted"] += 1
        temporary = np.unique(np.concatenate([
            decomposed[index]["points"] for index in sorted(kept)
        ])).astype(np.int64) if kept else np.empty(0, np.int64)
        if len(temporary) < int(result_summary["minimum_refined_point_count"]):
            refined = union_points
            status = "fallback_original_union_min_points"
            appended = False
        elif np.array_equal(temporary, union_points):
            refined = union_points
            status = "no_op_exact_original_union"
            appended = False
        else:
            refined = temporary
            status = "append_refined_union"
            appended = True
        if str(row["materialization_status"]) != status:
            errors["materialization_status_mismatch"] += 1
        if bool(row["append_refined_candidate"]) != appended:
            errors["append_flag_mismatch"] += 1
        if int(row["kept_atom_count"]) != len(kept):
            errors["kept_atom_count_mismatch"] += 1
        if int(row["removed_atom_count"]) != len(decomposed) - len(kept):
            errors["removed_atom_count_mismatch"] += 1
        if int(row["kept_point_count_before_fallback"]) != len(temporary):
            errors["temporary_point_count_mismatch"] += 1
        if int(row["refined_point_count"]) != len(refined):
            errors["refined_point_count_mismatch"] += 1
        if str(row["refined_geometry_sha256"]) != _geometry_sha256(refined):
            errors["refined_geometry_hash_mismatch"] += 1
        if appended:
            appended_count += 1
            path = args.result_root / row["refined_points_file"]
            with np.load(path) as payload:
                materialized = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
            if not np.array_equal(materialized, refined):
                errors["refined_points_file_mismatch"] += 1
        elif row.get("refined_points_file") is not None:
            errors["unexpected_refined_points_file"] += 1
        if not np.all(np.isin(refined, union_points)):
            errors["outside_union_point_added"] += 1
        original_best = _best_gt(union_points, gt, gt_meta)
        refined_best = _best_gt(refined, gt, gt_meta)
        original_q = _quality_target(float(original_best["best_iou"]))
        refined_q = _quality_target(float(refined_best["best_iou"]))
        delta = refined_q - original_q
        for name, expected in (
            ("label_original_union_best_iou", original_best["best_iou"]),
            ("label_original_union_quality_q", original_q),
            ("label_refined_best_iou", refined_best["best_iou"]),
            ("label_refined_quality_q", refined_q),
            ("label_quality_delta", delta),
        ):
            if not _close(row[name], expected):
                errors[f"{name}_mismatch"] += 1
        deltas.append(delta)
        fold_deltas[int(row["fold_index"])].append(delta)
        status_counts[status] += 1
        for name in ("candidate_deletion", "parent_geometry_mutation", "frozen_cache_write", "ap_computed"):
            if row.get(name) is not False:
                errors[name] += 1
        if row.get("original_union_retained") is not True:
            errors["original_union_not_retained"] += 1

    quality_matrix = np.asarray([
        [float(row["quality_features"][name]) for name in QUALITY_FEATURE_NAMES]
        for row in plan
    ], dtype=np.float64)
    quality_target = np.asarray([float(row["label_refined_quality_q"]) for row in plan])
    quality_folds = np.asarray([int(row["fold_index"]) for row in plan], dtype=np.int8)
    quality_prediction = np.full(len(plan), np.nan)
    for fold in range(5):
        select = np.flatnonzero(quality_folds == fold)
        payload = model_by_kind_fold[("refined_quality", fold)]
        if tuple(payload["feature_names"]) != QUALITY_FEATURE_NAMES:
            errors["quality_model_feature_schema_mismatch"] += 1
        raw, calibrated = _predict_model(payload, quality_matrix[select])
        quality_prediction[select] = calibrated
        for local, index in enumerate(select):
            if not _close(plan[index]["raw_oof_refined_quality"], raw[local]):
                errors["quality_raw_prediction_mismatch"] += 1
            if not _close(plan[index]["stage_c_oof_quality"], calibrated[local]):
                errors["quality_calibrated_prediction_mismatch"] += 1

    atom_metric = metrics(atom_target, audited_atom_prediction)
    quality_metric = metrics(quality_target, quality_prediction)
    delta_array = np.asarray(deltas, dtype=np.float64)
    fold_mean_delta = {str(fold): float(np.mean(fold_deltas[fold])) for fold in range(5)}
    change_counts = {
        "improved": int(np.sum(delta_array > 1e-12)),
        "neutral": int(np.sum(np.abs(delta_array) <= 1e-12)),
        "harmed": int(np.sum(delta_array < -1e-12)),
    }
    checks = {
        "member_oof_mae_at_most_0.25": atom_metric["mae"] <= 0.25,
        "member_oof_absolute_mean_bias_at_most_0.10": atom_metric["absolute_mean_bias"] <= 0.10,
        "all_folds_have_member_samples": all(np.sum(atom_folds == fold) > 0 for fold in range(5)),
        "shared_point_deletion_zero": errors.get("shared_atom_deleted", 0) == 0,
        "outside_union_addition_zero": errors.get("outside_union_point_added", 0) == 0,
        "original_unions_retained": errors.get("original_union_not_retained", 0) == 0,
        "overall_refined_quality_strictly_improves": float(delta_array.mean()) > 0.0,
        **{f"fold_{fold}_quality_delta_at_least_minus_0.01": fold_mean_delta[str(fold)] >= -0.01 for fold in range(5)},
        "refined_score_absolute_mean_bias_at_most_0.10": quality_metric["absolute_mean_bias"] <= 0.10,
    }
    if result_summary.get("materialization_status_counts") != dict(sorted(status_counts.items())):
        errors["summary_status_counts_mismatch"] += 1
    if result_summary.get("quality_change_counts") != change_counts:
        errors["summary_quality_change_counts_mismatch"] += 1
    if not _close(result_summary.get("mean_quality_delta"), delta_array.mean()):
        errors["summary_mean_quality_delta_mismatch"] += 1
    if result_summary.get("fold_mean_quality_delta") != fold_mean_delta:
        errors["summary_fold_mean_quality_delta_mismatch"] += 1
    expected_authorized = not errors and all(checks.values())
    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "union_count": len(plan),
        "atom_count": len(atoms),
        "appended_refined_union_count": appended_count,
        "materialization_status_counts": dict(sorted(status_counts.items())),
        "quality_change_counts": change_counts,
        "mean_quality_delta": float(delta_array.mean()),
        "fold_mean_quality_delta": fold_mean_delta,
        "member_metrics": atom_metric,
        "refined_quality_metrics": quality_metric,
        "advancement_gate": {
            "checks": checks,
            "advancement_authorized": expected_authorized,
        },
        "candidate_deletion": False,
        "parent_geometry_mutation": False,
        "frozen_cache_write": False,
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "result_summary_sha256": _sha256(result_summary_path),
            "member_predictions_sha256": _sha256(member_path),
            "refined_plan_sha256": _sha256(plan_path),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

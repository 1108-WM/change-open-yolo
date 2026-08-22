#!/usr/bin/env python3
"""Independently audit C-v2 role models, conservative materialization, and gates."""

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
    ADJACENCY, _geometry_sha256, decompose_union_atoms,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import (  # noqa: E402
    EXCLUSIVE_ROLES, QUALITY_FEATURE_NAMES, conservative_kept_indexes,
    regression_metrics,
)


VERSION = "ncs_fi1_stage_c_v2_refinement_oof_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _predict(payload: dict, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    raw = np.asarray(payload["regressor"].predict(matrix), dtype=np.float64)
    corrected = raw + float(payload["calibration_bias"])
    clip = payload.get("prediction_clip")
    if clip is not None:
        corrected = np.clip(corrected, float(clip[0]), float(clip[1]))
    return raw, corrected, float(payload["calibration_absolute_residual_q90"])


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
    member_rows = _read_jsonl(member_path)
    plan_rows = _read_jsonl(plan_path)
    errors = Counter()
    if _sha256(member_path) != result_summary["hashes"]["member_predictions"]:
        errors["member_prediction_sha256_mismatch"] += 1
    if _sha256(plan_path) != result_summary["hashes"]["refined_plan"]:
        errors["refined_plan_sha256_mismatch"] += 1
    if len(member_rows) != len(atoms):
        errors["member_prediction_count_mismatch"] += 1
    if len(plan_rows) != len(unions):
        errors["plan_count_mismatch"] += 1

    models = {}
    for record in result_summary["models"]:
        path = args.result_root / record["model_file"]
        if _sha256(path) != record["model_sha256"]:
            errors["model_sha256_mismatch"] += 1
        payload = joblib.load(path)
        key = (str(record["model_kind"]), int(record["outer_fold"]))
        if key in models:
            errors["duplicate_model_kind_fold"] += 1
        models[key] = payload
        if int(payload["calibration_fold"]) != (int(payload["outer_fold"]) + 1) % 5:
            errors["calibration_fold_contract_mismatch"] += 1
        if not _close(payload["calibration_bias"], record["calibration_bias"]):
            errors["calibration_bias_record_mismatch"] += 1
        if not _close(payload["calibration_absolute_residual_q90"], record["calibration_absolute_residual_q90"]):
            errors["calibration_q90_record_mismatch"] += 1

    feature_names = tuple(dataset_summary["feature_names"])
    member_corrected = np.full(len(atoms), np.nan)
    member_q90 = np.full(len(atoms), np.nan)
    role_metrics = {}
    for role in EXCLUSIVE_ROLES:
        indexes = np.asarray([index for index, row in enumerate(atoms) if row["role"] == role], dtype=np.int64)
        matrix = np.asarray([
            [float(atoms[index]["features"][name]) for name in feature_names]
            for index in indexes
        ], dtype=np.float64)
        target = np.asarray([float(atoms[index]["label_delta_iou_remove"]) for index in indexes])
        folds = np.asarray([int(atoms[index]["fold_index"]) for index in indexes], dtype=np.int8)
        local_raw = np.full(len(indexes), np.nan)
        local_corrected = np.full(len(indexes), np.nan)
        local_q90 = np.full(len(indexes), np.nan)
        for fold in range(5):
            selected = np.flatnonzero(folds == fold)
            payload = models[(f"member_delta_{role}", fold)]
            if tuple(payload["feature_names"]) != feature_names:
                errors["member_model_feature_schema_mismatch"] += 1
            raw, corrected, q90 = _predict(payload, matrix[selected])
            local_raw[selected] = raw
            local_corrected[selected] = corrected
            local_q90[selected] = q90
        member_corrected[indexes] = local_corrected
        member_q90[indexes] = local_q90
        role_metrics[role] = {
            "oof_prediction": regression_metrics(target, local_corrected),
            "zero_prediction_control": regression_metrics(target, np.zeros(len(target))),
            "confident_positive_lower_bound_count": int(np.sum(local_corrected - local_q90 > 0.0)),
        }
        for local, global_index in enumerate(indexes):
            source = atoms[global_index]
            prediction = member_rows[global_index]
            if source["atom_id"] != prediction["atom_id"]:
                errors["member_identity_mismatch"] += 1
            if not _close(prediction["raw_oof_delta_iou_remove"], local_raw[local]):
                errors["member_raw_prediction_mismatch"] += 1
            if not _close(prediction["corrected_oof_delta_iou_remove"], local_corrected[local]):
                errors["member_corrected_prediction_mismatch"] += 1
            if not _close(prediction["calibration_absolute_residual_q90"], local_q90[local]):
                errors["member_q90_mismatch"] += 1
            lower = local_corrected[local] - local_q90[local]
            if not _close(prediction["lower_confidence_bound"], lower):
                errors["member_lower_bound_mismatch"] += 1
            expected_action = "confident_remove" if lower > 0.0 else "uncertain_keep"
            if prediction["member_action"] != expected_action:
                errors["member_action_mismatch"] += 1
    for index, row in enumerate(atoms):
        if row["role"] == "shared":
            prediction = member_rows[index]
            if prediction["atom_id"] != row["atom_id"] or prediction["member_action"] != "forced_keep_shared":
                errors["shared_prediction_contract_mismatch"] += 1
            if any(prediction[name] is not None for name in (
                "raw_oof_delta_iou_remove", "corrected_oof_delta_iou_remove",
                "calibration_absolute_residual_q90", "lower_confidence_bound",
            )):
                errors["shared_prediction_nonnull"] += 1

    union_by_key = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in unions}
    plan_by_key = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in plan_rows}
    if len(plan_by_key) != len(plan_rows) or set(plan_by_key) != set(union_by_key):
        errors["plan_coverage_mismatch"] += 1
    atom_indexes_by_union = defaultdict(list)
    for index, row in enumerate(atoms):
        atom_indexes_by_union[(str(row["scene_name"]), int(row["union_candidate_id"]))].append(index)
    for indexes in atom_indexes_by_union.values():
        indexes.sort(key=lambda index: int(str(atoms[index]["atom_id"]).rsplit(":", 1)[1]))

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    resolver = GeometryResolver()
    scene_cache = {}
    temporary_points_by_key = {}
    quality_targets = []
    quality_folds = []
    quality_matrix = []
    for key, union in sorted(union_by_key.items()):
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
        union_node = node_by_key[str(union["union_geometry_key"])]
        track_node = node_by_key[str(union["track_geometry_key"])]
        native_node = node_by_key[str(union["native_geometry_key"])]
        union_points = resolver.points(union_node["canonical_geometry_locator"], int(union_node["point_count"]), len(gt))
        track_points = resolver.points(track_node["canonical_geometry_locator"], int(track_node["point_count"]), len(gt))
        native_points = resolver.points(native_node["canonical_geometry_locator"], int(native_node["point_count"]), len(gt))
        decomposed = decompose_union_atoms(
            union_points, track_points, native_points,
            np.asarray(processed[:, 9], dtype=np.int64),
        )
        indexes = atom_indexes_by_union[key]
        lower_bounds = [
            None if atoms[index]["role"] == "shared"
            else float(member_corrected[index] - member_q90[index])
            for index in indexes
        ]
        kept, details = conservative_kept_indexes(decomposed, lower_bounds, context["neighbors"])
        temporary = np.unique(np.concatenate([
            decomposed[index]["points"] for index in sorted(kept)
        ])).astype(np.int64) if kept else np.empty(0, np.int64)
        if details["fallback_no_shared"]:
            temporary = union_points
            status = "fallback_original_union_no_shared"
        elif len(temporary) < 100:
            temporary = union_points
            status = "fallback_original_union_min_points"
        elif np.array_equal(temporary, union_points):
            status = "no_op_exact_original_union"
        else:
            status = "temporary_refined_union"
        if status != row["temporary_status"]:
            errors["temporary_status_mismatch"] += 1
        if len(details["confident_removed_indexes"]) != int(row["confident_removal_count"]):
            errors["confident_removal_count_mismatch"] += 1
        if len(details["connectivity_removed_indexes"]) != int(row["connectivity_removal_count"]):
            errors["connectivity_removal_count_mismatch"] += 1
        if str(row["temporary_refined_sha256"]) != _geometry_sha256(temporary):
            errors["temporary_geometry_hash_mismatch"] += 1
        if int(row["temporary_refined_point_count"]) != len(temporary):
            errors["temporary_point_count_mismatch"] += 1
        original_best = _best_gt(union_points, gt, gt_meta)
        temporary_best = _best_gt(temporary, gt, gt_meta)
        original_q = _quality_target(float(original_best["best_iou"]))
        temporary_q = _quality_target(float(temporary_best["best_iou"]))
        expected_labels = {
            "label_original_union_best_iou": float(original_best["best_iou"]),
            "label_original_union_quality_q": original_q,
            "label_temporary_refined_best_iou": float(temporary_best["best_iou"]),
            "label_temporary_refined_quality_q": temporary_q,
            "label_temporary_quality_delta": temporary_q - original_q,
        }
        for name, expected in expected_labels.items():
            if not _close(row[name], expected):
                errors[f"{name}_mismatch"] += 1
        if sorted(row["quality_features"]) != sorted(QUALITY_FEATURE_NAMES):
            errors["quality_feature_schema_mismatch"] += 1
        quality_matrix.append([float(row["quality_features"][name]) for name in QUALITY_FEATURE_NAMES])
        quality_targets.append(temporary_q)
        quality_folds.append(int(row["fold_index"]))
        temporary_points_by_key[key] = (union_points, temporary)

    quality_matrix_array = np.asarray(quality_matrix, dtype=np.float64)
    quality_target_array = np.asarray(quality_targets, dtype=np.float64)
    quality_fold_array = np.asarray(quality_folds, dtype=np.int8)
    quality_raw = np.full(len(plan_rows), np.nan)
    quality_corrected = np.full(len(plan_rows), np.nan)
    quality_q90 = np.full(len(plan_rows), np.nan)
    for fold in range(5):
        selected = np.flatnonzero(quality_fold_array == fold)
        payload = models[("temporary_refined_quality", fold)]
        if tuple(payload["feature_names"]) != QUALITY_FEATURE_NAMES:
            errors["quality_model_feature_schema_mismatch"] += 1
        raw, corrected, q90 = _predict(payload, quality_matrix_array[selected])
        quality_raw[selected] = raw
        quality_corrected[selected] = corrected
        quality_q90[selected] = q90

    status_counts = Counter()
    append_deltas = []
    append_fold_deltas = defaultdict(list)
    all_final_deltas = []
    for index, row in enumerate(plan_rows):
        key = (str(row["scene_name"]), int(row["union_candidate_id"]))
        union_points, temporary = temporary_points_by_key[key]
        for name, expected in (
            ("raw_oof_temporary_refined_quality", quality_raw[index]),
            ("corrected_oof_temporary_refined_quality", quality_corrected[index]),
            ("quality_calibration_absolute_residual_q90", quality_q90[index]),
            ("quality_lower_confidence_bound", quality_corrected[index] - quality_q90[index]),
        ):
            if not _close(row[name], expected):
                errors[f"{name}_mismatch"] += 1
        reference = float(row["quality_features"]["stage_a_union_quality"])
        changed = not np.array_equal(temporary, union_points)
        eligible = changed and quality_corrected[index] - quality_q90[index] > reference
        if bool(row["append_eligible_refined_union"]) != eligible:
            errors["append_eligibility_mismatch"] += 1
        if eligible:
            status = "append_eligible_refined_union"
            path = args.result_root / row["refined_points_file"]
            with np.load(path) as payload:
                materialized = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
            if not np.array_equal(materialized, temporary):
                errors["refined_points_file_mismatch"] += 1
            final_q = float(row["label_temporary_refined_quality_q"])
            delta = float(row["label_temporary_quality_delta"])
            append_deltas.append(delta)
            append_fold_deltas[int(row["fold_index"])].append(delta)
        else:
            status = "quality_guard_fallback_original_union"
            if row.get("refined_points_file") is not None:
                errors["unexpected_refined_points_file"] += 1
            final_q = float(row["label_original_union_quality_q"])
        final_delta = final_q - float(row["label_original_union_quality_q"])
        all_final_deltas.append(final_delta)
        status_counts[status] += 1
        if row["final_status"] != status or not _close(row["label_final_quality_q"], final_q) or not _close(row["label_final_quality_delta"], final_delta):
            errors["final_plan_mismatch"] += 1
        for name in ("candidate_deletion", "parent_geometry_mutation", "frozen_cache_write", "ap_computed"):
            if row.get(name) is not False:
                errors[name] += 1
        if row.get("original_union_retained") is not True:
            errors["original_union_not_retained"] += 1
        if int(row.get("shared_point_deleted_count", -1)) != 0:
            errors["shared_point_deleted"] += 1
        if int(row.get("outside_union_added_point_count", -1)) != 0:
            errors["outside_union_point_added"] += 1

    append_array = np.asarray(append_deltas, dtype=np.float64)
    append_count = len(append_deltas)
    change_counts = {
        "improved": int(np.sum(append_array > 1e-12)) if append_count else 0,
        "neutral": int(np.sum(np.abs(append_array) <= 1e-12)) if append_count else 0,
        "harmed": int(np.sum(append_array < -1e-12)) if append_count else 0,
    }
    fold_delta = {
        str(fold): float(np.mean(append_fold_deltas[fold])) if append_fold_deltas[fold] else None
        for fold in range(5)
    }
    all_final_delta = float(np.mean(all_final_deltas))
    checks = {
        "track_only_oof_mae_strictly_better_than_zero": role_metrics["track_only"]["oof_prediction"]["mae"] < role_metrics["track_only"]["zero_prediction_control"]["mae"],
        "native_only_oof_mae_strictly_better_than_zero": role_metrics["native_only"]["oof_prediction"]["mae"] < role_metrics["native_only"]["zero_prediction_control"]["mae"],
        "shared_point_deletion_zero": errors.get("shared_point_deleted", 0) == 0,
        "outside_union_addition_zero": errors.get("outside_union_point_added", 0) == 0,
        "original_union_deletion_zero": errors.get("original_union_not_retained", 0) == 0,
        "append_eligible_count_positive": append_count > 0,
        "append_improvement_fraction_at_least_0.70": change_counts["improved"] / max(1, append_count) >= 0.70,
        "append_harm_fraction_at_most_0.10": change_counts["harmed"] / max(1, append_count) <= 0.10,
        "append_mean_quality_delta_positive": float(append_array.mean()) > 0.0 if append_count else False,
        **{f"fold_{fold}_append_mean_quality_delta_nonnegative": fold_delta[str(fold)] is not None and fold_delta[str(fold)] >= 0.0 for fold in range(5)},
        "all_union_conservative_mean_quality_delta_nonnegative": all_final_delta >= 0.0,
    }
    if result_summary.get("final_status_counts") != dict(sorted(status_counts.items())):
        errors["summary_status_counts_mismatch"] += 1
    if result_summary.get("append_quality_change_counts") != change_counts:
        errors["summary_change_counts_mismatch"] += 1
    if not _close(result_summary.get("append_mean_quality_delta"), append_array.mean() if append_count else None):
        errors["summary_append_mean_delta_mismatch"] += 1
    if result_summary.get("append_fold_mean_quality_delta") != fold_delta:
        errors["summary_fold_delta_mismatch"] += 1
    if not _close(result_summary.get("all_union_conservative_mean_quality_delta"), all_final_delta):
        errors["summary_all_union_delta_mismatch"] += 1
    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "dataset_scene_count": int(dataset_summary["dataset_scene_count"]),
        "union_count": len(plan_rows),
        "atom_count": len(atoms),
        "exclusive_role_metrics": role_metrics,
        "final_status_counts": dict(sorted(status_counts.items())),
        "append_eligible_count": append_count,
        "append_quality_change_counts": change_counts,
        "append_mean_quality_delta": float(append_array.mean()) if append_count else None,
        "append_fold_mean_quality_delta": fold_delta,
        "all_union_conservative_mean_quality_delta": all_final_delta,
        "advancement_gate": {
            "checks": checks,
            "advancement_authorized": not errors and all(checks.values()),
        },
        "shared_point_deleted_count": 0,
        "outside_union_added_point_count": 0,
        "original_union_deleted_count": 0,
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

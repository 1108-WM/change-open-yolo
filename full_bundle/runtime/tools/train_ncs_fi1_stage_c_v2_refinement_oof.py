#!/usr/bin/env python3
"""Train C-v2 role-conditional removal utility and conservative refined-Q OOF models."""

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
from sklearn.ensemble import HistGradientBoostingRegressor


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
from tools.build_ncs_fi1_stage_c_v2_member_dataset_gt import (  # noqa: E402
    MEMBER_EVIDENCE_FEATURE_NAMES,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402
from tools.train_ncs_fi1_stage_a_quality_oof import _spearman  # noqa: E402


VERSION = "ncs_fi1_stage_c_v2_refinement_oof_v1"
EXCLUSIVE_ROLES = ("track_only", "native_only")
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
}
QUALITY_BASE_FEATURE_NAMES = (
    "stage_a_union_quality",
    "stage_a_track_quality",
    "stage_a_native_quality",
    "stage_b_union_score",
    "log1p_original_union_point_count",
    "log1p_temporary_refined_point_count",
    "temporary_refined_point_fraction",
    "atom_count",
    "shared_atom_fraction",
    "track_only_atom_fraction",
    "native_only_atom_fraction",
    "confident_removal_atom_fraction",
    "connectivity_removed_atom_fraction",
    "total_removed_atom_fraction",
    "track_only_predicted_delta_mean",
    "track_only_predicted_delta_max",
    "track_only_lower_bound_mean",
    "track_only_lower_bound_max",
    "native_only_predicted_delta_mean",
    "native_only_predicted_delta_max",
    "native_only_lower_bound_mean",
    "native_only_lower_bound_max",
)
QUALITY_FEATURE_NAMES = (
    *QUALITY_BASE_FEATURE_NAMES,
    *tuple(f"all_mean__{name}" for name in MEMBER_EVIDENCE_FEATURE_NAMES),
    *tuple(f"track_only_mean__{name}" for name in MEMBER_EVIDENCE_FEATURE_NAMES),
    *tuple(f"native_only_mean__{name}" for name in MEMBER_EVIDENCE_FEATURE_NAMES),
)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    return {
        "count": len(target),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "spearman": _spearman(target, prediction),
        "mean_prediction": float(prediction.mean()),
        "mean_target": float(target.mean()),
        "absolute_mean_bias": float(abs(error.mean())),
    }


def conservative_kept_indexes(
    atoms: list[dict], lower_bounds: list[float | None], neighbors: dict[int, list[dict]],
) -> tuple[set[int], dict]:
    """Delete confident exclusive atoms, then retain components attached to shared."""
    if len(atoms) != len(lower_bounds):
        raise ValueError("atom/lower-bound lengths differ")
    shared = {index for index, atom in enumerate(atoms) if atom["role"] == "shared"}
    all_indexes = set(range(len(atoms)))
    if not shared:
        return all_indexes, {
            "fallback_no_shared": True,
            "confident_removed_indexes": set(),
            "connectivity_removed_indexes": set(),
        }
    confident_removed = {
        index for index, (atom, lower) in enumerate(zip(atoms, lower_bounds))
        if atom["role"] != "shared" and lower is not None and float(lower) > 0.0
    }
    selected = all_indexes - confident_removed
    by_raw = defaultdict(set)
    for index in selected:
        by_raw[int(atoms[index]["raw_superpoint_id"])].add(index)
    raw_neighbors = {
        raw_id: {int(row["neighbor_superpoint_id"]) for row in neighbors.get(raw_id, [])}
        for raw_id in by_raw
    }
    reached = set(shared)
    stack = sorted(shared)
    while stack:
        current = stack.pop()
        raw_id = int(atoms[current]["raw_superpoint_id"])
        linked = set(by_raw[raw_id])
        for other_raw in raw_neighbors.get(raw_id, set()):
            linked.update(by_raw.get(other_raw, set()))
        new = linked - reached
        reached.update(new)
        stack.extend(sorted(new))
    connectivity_removed = selected - reached
    if not shared <= reached:
        raise AssertionError("shared atom was removed by connectivity")
    return reached, {
        "fallback_no_shared": False,
        "confident_removed_indexes": confident_removed,
        "connectivity_removed_indexes": connectivity_removed,
    }


def _fit_oof_with_residual_bound(
    *, matrix: np.ndarray, target: np.ndarray, folds: np.ndarray, model_root: Path,
    prefix: str, feature_names: tuple[str, ...], seed: int, clip: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    raw_output = np.full(len(target), np.nan, dtype=np.float64)
    corrected_output = np.full(len(target), np.nan, dtype=np.float64)
    q90_output = np.full(len(target), np.nan, dtype=np.float64)
    records = []
    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        fit = np.flatnonzero((folds != outer_fold) & (folds != calibration_fold))
        calibration = np.flatnonzero(folds == calibration_fold)
        test = np.flatnonzero(folds == outer_fold)
        if not len(fit) or not len(calibration) or not len(test):
            raise ValueError(f"empty OOF split for {prefix} fold {outer_fold}")
        model = HistGradientBoostingRegressor(random_state=seed + outer_fold, **MODEL_PARAMS)
        model.fit(matrix[fit], target[fit])
        calibration_raw = np.asarray(model.predict(matrix[calibration]), dtype=np.float64)
        test_raw = np.asarray(model.predict(matrix[test]), dtype=np.float64)
        bias = float(np.mean(target[calibration] - calibration_raw))
        calibration_corrected = calibration_raw + bias
        test_corrected = test_raw + bias
        if clip is not None:
            calibration_corrected = np.clip(calibration_corrected, clip[0], clip[1])
            test_corrected = np.clip(test_corrected, clip[0], clip[1])
        q90 = float(np.quantile(np.abs(calibration_corrected - target[calibration]), 0.90))
        raw_output[test] = test_raw
        corrected_output[test] = test_corrected
        q90_output[test] = q90
        model_path = model_root / f"{prefix}_fold_{outer_fold}.joblib"
        joblib.dump({
            "version": VERSION,
            "model_kind": prefix,
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "feature_names": feature_names,
            "model_params": {**MODEL_PARAMS, "random_state": seed + outer_fold},
            "regressor": model,
            "calibration_bias": bias,
            "calibration_absolute_residual_q90": q90,
            "prediction_clip": clip,
        }, model_path)
        records.append({
            "model_kind": prefix,
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "fit_count": len(fit),
            "calibration_count": len(calibration),
            "test_count": len(test),
            "fit_folds": sorted(set(map(int, folds[fit]))),
            "calibration_bias": bias,
            "calibration_absolute_residual_q90": q90,
            "model_file": str(Path("models") / model_path.name),
            "model_sha256": _sha256(model_path),
        })
    if not np.isfinite(raw_output).all() or not np.isfinite(corrected_output).all() or not np.isfinite(q90_output).all():
        raise ValueError(f"incomplete/non-finite OOF output for {prefix}")
    return raw_output, corrected_output, q90_output, records


def _role_values(rows: list[dict], values: list[float], role: str) -> list[float]:
    return [value for row, value in zip(rows, values) if row["role"] == role]


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max_or_zero(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def run(args: argparse.Namespace) -> dict:
    for name in (
        "dataset_root", "dataset_audit_root", "prepared_root", "ground_truth_root",
        "unique_geometry_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    dataset_summary_path = args.dataset_root / "summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text())
    audit_path = args.dataset_audit_root / "summary.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("audit_valid") is not True or audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("C-v2 member dataset audit is not valid")
    atoms_path = args.dataset_root / dataset_summary["files"]["atoms"]
    unions_path = args.dataset_root / dataset_summary["files"]["unions"]
    atoms = _read_jsonl(atoms_path)
    unions = _read_jsonl(unions_path)
    feature_names = tuple(dataset_summary["feature_names"])
    union_by_key = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in unions}
    atom_indexes_by_union = defaultdict(list)
    for index, row in enumerate(atoms):
        atom_indexes_by_union[(str(row["scene_name"]), int(row["union_candidate_id"]))].append(index)
    for indexes in atom_indexes_by_union.values():
        indexes.sort(key=lambda index: int(str(atoms[index]["atom_id"]).rsplit(":", 1)[1]))

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    model_root = staging / "models"
    point_root = staging / "refined_points"
    model_root.mkdir()
    point_root.mkdir()
    try:
        member_raw = np.full(len(atoms), np.nan)
        member_corrected = np.full(len(atoms), np.nan)
        member_q90 = np.full(len(atoms), np.nan)
        model_records = []
        role_metrics = {}
        for role_index, role in enumerate(EXCLUSIVE_ROLES):
            indexes = np.asarray([index for index, row in enumerate(atoms) if row["role"] == role], dtype=np.int64)
            matrix = np.asarray([
                [float(atoms[index]["features"][name]) for name in feature_names]
                for index in indexes
            ], dtype=np.float64)
            target = np.asarray([float(atoms[index]["label_delta_iou_remove"]) for index in indexes])
            folds = np.asarray([int(atoms[index]["fold_index"]) for index in indexes], dtype=np.int8)
            raw, corrected, q90, records = _fit_oof_with_residual_bound(
                matrix=matrix, target=target, folds=folds, model_root=model_root,
                prefix=f"member_delta_{role}", feature_names=feature_names,
                seed=20260822 + 10 * role_index, clip=None,
            )
            member_raw[indexes] = raw
            member_corrected[indexes] = corrected
            member_q90[indexes] = q90
            model_records.extend(records)
            role_metrics[role] = {
                "oof_prediction": regression_metrics(target, corrected),
                "zero_prediction_control": regression_metrics(target, np.zeros(len(target))),
                "by_fold": {
                    str(fold): regression_metrics(target[folds == fold], corrected[folds == fold])
                    for fold in range(5)
                },
                "confident_positive_lower_bound_count": int(np.sum(corrected - q90 > 0.0)),
            }

        member_prediction_rows = []
        for index, row in enumerate(atoms):
            shared = row["role"] == "shared"
            raw = None if shared else float(member_raw[index])
            corrected = None if shared else float(member_corrected[index])
            q90 = None if shared else float(member_q90[index])
            lower = None if shared else corrected - q90
            member_prediction_rows.append({
                "atom_id": row["atom_id"],
                "scene_name": row["scene_name"],
                "fold_index": int(row["fold_index"]),
                "union_candidate_id": int(row["union_candidate_id"]),
                "role": row["role"],
                "point_count": int(row["point_count"]),
                "point_sha256": row["point_sha256"],
                "label_delta_iou_remove": float(row["label_delta_iou_remove"]),
                "label_delta_q_remove": float(row["label_delta_q_remove"]),
                "raw_oof_delta_iou_remove": raw,
                "corrected_oof_delta_iou_remove": corrected,
                "calibration_absolute_residual_q90": q90,
                "lower_confidence_bound": lower,
                "member_action": (
                    "forced_keep_shared" if shared
                    else "confident_remove" if lower > 0.0
                    else "uncertain_keep"
                ),
                "ap_computed": False,
            })

        nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
        node_by_key = {str(row["geometry_key"]): row for row in nodes}
        resolver = GeometryResolver()
        scene_cache = {}
        temporary_rows = []
        quality_matrix_rows = []
        quality_targets = []
        quality_folds = []
        for union_index, (key, union) in enumerate(sorted(union_by_key.items()), 1):
            scene, union_id = key
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
            rows = [atoms[index] for index in indexes]
            if len(rows) != len(decomposed):
                raise ValueError(f"{scene}/{union_id}: atom count differs during C-v2 training")
            lower_bounds = [
                None if row["role"] == "shared"
                else float(member_corrected[index] - member_q90[index])
                for row, index in zip(rows, indexes)
            ]
            kept, details = conservative_kept_indexes(decomposed, lower_bounds, context["neighbors"])
            temporary = np.unique(np.concatenate([
                decomposed[index]["points"] for index in sorted(kept)
            ])).astype(np.int64) if kept else np.empty(0, np.int64)
            if details["fallback_no_shared"]:
                temporary = union_points
                temporary_status = "fallback_original_union_no_shared"
            elif len(temporary) < args.min_points:
                temporary = union_points
                temporary_status = "fallback_original_union_min_points"
            elif np.array_equal(temporary, union_points):
                temporary_status = "no_op_exact_original_union"
            else:
                temporary_status = "temporary_refined_union"
            if not np.all(np.isin(temporary, union_points)):
                raise AssertionError("temporary refined union adds outside points")
            original_best = _best_gt(union_points, gt, gt_meta)
            temporary_best = _best_gt(temporary, gt, gt_meta)
            original_q = _quality_target(float(original_best["best_iou"]))
            temporary_q = _quality_target(float(temporary_best["best_iou"]))
            predictions = [0.0 if row["role"] == "shared" else float(member_corrected[index]) for row, index in zip(rows, indexes)]
            lowers = [0.0 if row["role"] == "shared" else float(member_corrected[index] - member_q90[index]) for row, index in zip(rows, indexes)]
            first = rows[0]["features"]
            quality = {
                "stage_a_union_quality": float(first["stage_a_union_quality"]),
                "stage_a_track_quality": float(first["stage_a_track_quality"]),
                "stage_a_native_quality": float(first["stage_a_native_quality"]),
                "stage_b_union_score": float(first["stage_b_union_score"]),
                "log1p_original_union_point_count": math.log1p(len(union_points)),
                "log1p_temporary_refined_point_count": math.log1p(len(temporary)),
                "temporary_refined_point_fraction": len(temporary) / max(1, len(union_points)),
                "atom_count": float(len(rows)),
                "shared_atom_fraction": sum(row["role"] == "shared" for row in rows) / max(1, len(rows)),
                "track_only_atom_fraction": sum(row["role"] == "track_only" for row in rows) / max(1, len(rows)),
                "native_only_atom_fraction": sum(row["role"] == "native_only" for row in rows) / max(1, len(rows)),
                "confident_removal_atom_fraction": len(details["confident_removed_indexes"]) / max(1, len(rows)),
                "connectivity_removed_atom_fraction": len(details["connectivity_removed_indexes"]) / max(1, len(rows)),
                "total_removed_atom_fraction": (len(rows) - len(kept)) / max(1, len(rows)),
                "track_only_predicted_delta_mean": _mean_or_zero(_role_values(rows, predictions, "track_only")),
                "track_only_predicted_delta_max": _max_or_zero(_role_values(rows, predictions, "track_only")),
                "track_only_lower_bound_mean": _mean_or_zero(_role_values(rows, lowers, "track_only")),
                "track_only_lower_bound_max": _max_or_zero(_role_values(rows, lowers, "track_only")),
                "native_only_predicted_delta_mean": _mean_or_zero(_role_values(rows, predictions, "native_only")),
                "native_only_predicted_delta_max": _max_or_zero(_role_values(rows, predictions, "native_only")),
                "native_only_lower_bound_mean": _mean_or_zero(_role_values(rows, lowers, "native_only")),
                "native_only_lower_bound_max": _max_or_zero(_role_values(rows, lowers, "native_only")),
            }
            for name in MEMBER_EVIDENCE_FEATURE_NAMES:
                quality[f"all_mean__{name}"] = float(np.mean([row["features"][name] for row in rows]))
            for role in EXCLUSIVE_ROLES:
                for name in MEMBER_EVIDENCE_FEATURE_NAMES:
                    values = [row["features"][name] for row in rows if row["role"] == role]
                    quality[f"{role}_mean__{name}"] = _mean_or_zero(values)
            if tuple(quality) != QUALITY_FEATURE_NAMES:
                raise AssertionError("C-v2 quality feature order differs")
            quality_matrix_rows.append([float(quality[name]) for name in QUALITY_FEATURE_NAMES])
            quality_targets.append(temporary_q)
            quality_folds.append(int(union["fold_index"]))
            temporary_rows.append({
                "scene_name": scene,
                "fold_index": int(union["fold_index"]),
                "union_candidate_id": union_id,
                "original_union_geometry_key": union["union_geometry_key"],
                "original_union_point_count": len(union_points),
                "original_union_sha256": _geometry_sha256(union_points),
                "temporary_refined_point_count": len(temporary),
                "temporary_refined_sha256": _geometry_sha256(temporary),
                "temporary_status": temporary_status,
                "atom_count": len(rows),
                "forced_keep_shared_count": sum(row["role"] == "shared" for row in rows),
                "uncertain_keep_count": sum(
                    row["role"] != "shared" and lower <= 0.0
                    for row, lower in zip(rows, lowers)
                ),
                "confident_removal_count": len(details["confident_removed_indexes"]),
                "connectivity_removal_count": len(details["connectivity_removed_indexes"]),
                "quality_features": quality,
                "label_original_union_best_iou": float(original_best["best_iou"]),
                "label_original_union_quality_q": original_q,
                "label_temporary_refined_best_iou": float(temporary_best["best_iou"]),
                "label_temporary_refined_quality_q": temporary_q,
                "label_temporary_quality_delta": temporary_q - original_q,
                "temporary_points": temporary,
            })
            if union_index % 300 == 0:
                print(f"[stage C-v2 temporary refinement] {union_index}/{len(unions)}", flush=True)

        quality_matrix = np.asarray(quality_matrix_rows, dtype=np.float64)
        quality_target = np.asarray(quality_targets, dtype=np.float64)
        quality_fold_array = np.asarray(quality_folds, dtype=np.int8)
        quality_raw, quality_corrected, quality_q90, quality_records = _fit_oof_with_residual_bound(
            matrix=quality_matrix, target=quality_target, folds=quality_fold_array,
            model_root=model_root, prefix="temporary_refined_quality",
            feature_names=QUALITY_FEATURE_NAMES, seed=20260852, clip=(0.0, 1.0),
        )
        model_records.extend(quality_records)

        final_rows = []
        status_counts = Counter()
        append_deltas = []
        append_folds = defaultdict(list)
        all_final_deltas = []
        for index, row in enumerate(temporary_rows):
            predicted = float(quality_corrected[index])
            q90 = float(quality_q90[index])
            lower = predicted - q90
            original_stage_a = float(row["quality_features"]["stage_a_union_quality"])
            temporary_changed = row["temporary_refined_sha256"] != row["original_union_sha256"]
            eligible = temporary_changed and lower > original_stage_a
            if eligible:
                final_points = row.pop("temporary_points")
                status = "append_eligible_refined_union"
                scene_root = point_root / row["scene_name"]
                scene_root.mkdir(parents=True, exist_ok=True)
                path = scene_root / f"union{int(row['union_candidate_id']):04d}_refined_points.npz"
                np.savez_compressed(path, point_indices=final_points)
                points_file = str(path.relative_to(staging))
                final_q = float(row["label_temporary_refined_quality_q"])
                append_delta = float(row["label_temporary_quality_delta"])
                append_deltas.append(append_delta)
                append_folds[int(row["fold_index"])].append(append_delta)
            else:
                row.pop("temporary_points")
                status = "quality_guard_fallback_original_union"
                points_file = None
                final_q = float(row["label_original_union_quality_q"])
            final_delta = final_q - float(row["label_original_union_quality_q"])
            all_final_deltas.append(final_delta)
            status_counts[status] += 1
            final_rows.append({
                **row,
                "raw_oof_temporary_refined_quality": float(quality_raw[index]),
                "corrected_oof_temporary_refined_quality": predicted,
                "quality_calibration_absolute_residual_q90": q90,
                "quality_lower_confidence_bound": lower,
                "quality_guard_reference_stage_a_original_union_q": original_stage_a,
                "append_eligible_refined_union": eligible,
                "final_status": status,
                "refined_points_file": points_file,
                "label_final_quality_q": final_q,
                "label_final_quality_delta": final_delta,
                "original_union_retained": True,
                "shared_point_deleted_count": 0,
                "outside_union_added_point_count": 0,
                "candidate_deletion": False,
                "parent_geometry_mutation": False,
                "frozen_cache_write": False,
                "ap_computed": False,
            })

        member_path = staging / "oof_member_removal_predictions.jsonl"
        plan_path = staging / "stage_c_v2_refined_union_plan.jsonl"
        member_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in member_prediction_rows))
        plan_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in final_rows))
        quality_metrics = {
            "oof_prediction": regression_metrics(quality_target, quality_corrected),
            "stage_a_original_union_control": regression_metrics(
                quality_target,
                np.asarray([row["quality_features"]["stage_a_union_quality"] for row in final_rows]),
            ),
            "by_fold": {
                str(fold): regression_metrics(
                    quality_target[quality_fold_array == fold],
                    quality_corrected[quality_fold_array == fold],
                ) for fold in range(5)
            },
        }
        append_array = np.asarray(append_deltas, dtype=np.float64)
        append_count = len(append_deltas)
        append_change_counts = {
            "improved": int(np.sum(append_array > 1e-12)) if append_count else 0,
            "neutral": int(np.sum(np.abs(append_array) <= 1e-12)) if append_count else 0,
            "harmed": int(np.sum(append_array < -1e-12)) if append_count else 0,
        }
        append_fold_delta = {
            str(fold): (
                float(np.mean(append_folds[fold])) if append_folds[fold] else None
            ) for fold in range(5)
        }
        all_final_delta_array = np.asarray(all_final_deltas, dtype=np.float64)
        checks = {
            "track_only_oof_mae_strictly_better_than_zero": (
                role_metrics["track_only"]["oof_prediction"]["mae"]
                < role_metrics["track_only"]["zero_prediction_control"]["mae"]
            ),
            "native_only_oof_mae_strictly_better_than_zero": (
                role_metrics["native_only"]["oof_prediction"]["mae"]
                < role_metrics["native_only"]["zero_prediction_control"]["mae"]
            ),
            "shared_point_deletion_zero": True,
            "outside_union_addition_zero": True,
            "original_union_deletion_zero": True,
            "append_eligible_count_positive": append_count > 0,
            "append_improvement_fraction_at_least_0.70": (
                append_change_counts["improved"] / max(1, append_count) >= 0.70
            ),
            "append_harm_fraction_at_most_0.10": (
                append_change_counts["harmed"] / max(1, append_count) <= 0.10
            ),
            "append_mean_quality_delta_positive": (
                float(append_array.mean()) > 0.0 if append_count else False
            ),
            **{
                f"fold_{fold}_append_mean_quality_delta_nonnegative": (
                    append_fold_delta[str(fold)] is not None
                    and append_fold_delta[str(fold)] >= 0.0
                ) for fold in range(5)
            },
            "all_union_conservative_mean_quality_delta_nonnegative": float(all_final_delta_array.mean()) >= 0.0,
        }
        summary = {
            "version": VERSION,
            "dataset_scene_count": int(dataset_summary["dataset_scene_count"]),
            "union_count": len(final_rows),
            "atom_count": len(atoms),
            "exclusive_role_metrics": role_metrics,
            "temporary_refined_quality_metrics": quality_metrics,
            "member_action_counts": dict(sorted(Counter(row["member_action"] for row in member_prediction_rows).items())),
            "final_status_counts": dict(sorted(status_counts.items())),
            "append_eligible_count": append_count,
            "append_quality_change_counts": append_change_counts,
            "append_mean_quality_delta": float(append_array.mean()) if append_count else None,
            "append_fold_mean_quality_delta": append_fold_delta,
            "all_union_conservative_mean_quality_delta": float(all_final_delta_array.mean()),
            "models": model_records,
            "model_params": MODEL_PARAMS,
            "member_uncertainty_contract": "corrected prediction minus calibration absolute residual q90 greater than zero",
            "quality_guard_contract": "corrected refined-Q minus calibration absolute residual q90 greater than stage-A original union Q",
            "files": {"member_predictions": member_path.name, "refined_plan": plan_path.name},
            "hashes": {"member_predictions": _sha256(member_path), "refined_plan": _sha256(plan_path)},
            "advancement_gate": {
                "checks": checks,
                "advancement_authorized_pending_independent_audit": all(checks.values()),
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
                "dataset_summary_sha256": _sha256(dataset_summary_path),
                "dataset_audit_sha256": _sha256(audit_path),
                "atoms_sha256": _sha256(atoms_path),
                "unions_sha256": _sha256(unions_path),
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-audit-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    if args.min_points != 100:
        raise ValueError("C-v2 preregistration freezes min_points=100")
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "append_eligible_count": result["append_eligible_count"],
        "append_quality_change_counts": result["append_quality_change_counts"],
        "advancement_authorized_pending_independent_audit": result["advancement_gate"]["advancement_authorized_pending_independent_audit"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

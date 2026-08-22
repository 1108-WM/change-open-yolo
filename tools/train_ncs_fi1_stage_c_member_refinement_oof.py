#!/usr/bin/env python3
"""Train OOF member retention, materialize refined unions, and predict refined Q."""

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
    ADJACENCY, FEATURE_NAMES, RELATION_FEATURES, _geometry_sha256,
    decompose_union_atoms,
)
from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402
from tools.train_ncs_fi1_stage_a_quality_oof import (  # noqa: E402
    _calibrate, _fit_calibrator, metrics,
)


VERSION = "ncs_fi1_stage_c_member_refinement_oof_v1"
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
}
QUALITY_FEATURE_NAMES = (
    "stage_a_union_quality",
    "stage_a_track_quality",
    "stage_a_native_quality",
    "stage_b_union_score",
    "log1p_original_union_point_count",
    "log1p_refined_point_count",
    "refined_point_fraction",
    "atom_count",
    "kept_atom_fraction",
    "removed_atom_fraction",
    "shared_atom_fraction",
    "track_only_atom_fraction",
    "native_only_atom_fraction",
    "predicted_purity_point_weighted_mean",
    "predicted_purity_atom_mean",
    "predicted_purity_atom_min",
    "predicted_purity_atom_max",
    "kept_predicted_purity_point_weighted_mean",
    "removed_predicted_purity_point_weighted_mean",
    *tuple(f"relation__{name}" for name in RELATION_FEATURES),
)


def connected_refinement_indexes(
    atoms: list[dict], probabilities: list[float], neighbors: dict[int, list[dict]],
    threshold: float = 0.50,
) -> set[int]:
    """Apply the frozen shared-safe threshold and raw-superpoint connectivity rule."""
    if len(atoms) != len(probabilities):
        raise ValueError("atom/probability lengths differ")
    selected = {
        index for index, (atom, probability) in enumerate(zip(atoms, probabilities))
        if atom["role"] == "shared" or float(probability) >= threshold
    }
    if not selected:
        return set()
    by_raw = defaultdict(set)
    for index in selected:
        by_raw[int(atoms[index]["raw_superpoint_id"])].add(index)
    raw_neighbors = {
        raw_id: {int(row["neighbor_superpoint_id"]) for row in neighbors.get(raw_id, [])}
        for raw_id in by_raw
    }
    unseen = set(selected)
    components = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = {seed}
        while stack:
            current = stack.pop()
            raw_id = int(atoms[current]["raw_superpoint_id"])
            linked = set(by_raw[raw_id])
            for other_raw in raw_neighbors.get(raw_id, set()):
                linked.update(by_raw.get(other_raw, set()))
            reached = unseen.intersection(linked)
            unseen.difference_update(reached)
            component.update(reached)
            stack.extend(sorted(reached))
        components.append(component)
    shared = {index for index in selected if atoms[index]["role"] == "shared"}
    if shared:
        return set().union(*(component for component in components if component & shared))
    return max(
        components,
        key=lambda component: (
            sum(len(atoms[index]["points"]) for index in component),
            -min(component),
        ),
    )


def _fit_oof(
    matrix: np.ndarray, target: np.ndarray, folds: np.ndarray, model_root: Path,
    prefix: str, feature_names: tuple[str, ...], seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    calibrated = np.full(len(target), np.nan, dtype=np.float64)
    raw_output = np.full(len(target), np.nan, dtype=np.float64)
    records = []
    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        fit = np.flatnonzero((folds != outer_fold) & (folds != calibration_fold))
        calibration = np.flatnonzero(folds == calibration_fold)
        test = np.flatnonzero(folds == outer_fold)
        if not len(fit) or not len(calibration) or not len(test):
            raise ValueError(f"empty OOF split for {prefix} fold {outer_fold}")
        model = HistGradientBoostingRegressor(
            random_state=seed + outer_fold, **MODEL_PARAMS
        )
        model.fit(matrix[fit], target[fit])
        calibration_raw = np.clip(model.predict(matrix[calibration]), 0.0, 1.0)
        test_raw = np.clip(model.predict(matrix[test]), 0.0, 1.0)
        calibrator, calibration_kind = _fit_calibrator(
            calibration_raw, target[calibration]
        )
        prediction = np.clip(_calibrate(calibrator, test_raw), 0.0, 1.0)
        raw_output[test] = test_raw
        calibrated[test] = prediction
        model_path = model_root / f"{prefix}_fold_{outer_fold}.joblib"
        joblib.dump({
            "version": VERSION,
            "model_kind": prefix,
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "feature_names": feature_names,
            "model_params": {**MODEL_PARAMS, "random_state": seed + outer_fold},
            "regressor": model,
            "calibrator": calibrator,
            "calibration_kind": calibration_kind,
        }, model_path)
        records.append({
            "model_kind": prefix,
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "fit_count": len(fit),
            "calibration_count": len(calibration),
            "test_count": len(test),
            "fit_folds": sorted(set(map(int, folds[fit]))),
            "calibration_kind": calibration_kind,
            "model_file": str(Path("models") / model_path.name),
            "model_sha256": _sha256(model_path),
        })
    if not np.isfinite(calibrated).all() or not np.isfinite(raw_output).all():
        raise ValueError(f"incomplete/non-finite OOF predictions for {prefix}")
    return raw_output, calibrated, records


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    return float(np.average(values, weights=weights)) if values else 0.0


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
        raise ValueError("stage-C member dataset audit is not valid")
    atoms_path = args.dataset_root / dataset_summary["files"]["atoms"]
    unions_path = args.dataset_root / dataset_summary["files"]["unions"]
    atom_rows = _read_jsonl(atoms_path)
    union_rows = _read_jsonl(unions_path)
    atoms_by_union = defaultdict(list)
    for index, row in enumerate(atom_rows):
        atoms_by_union[(str(row["scene_name"]), int(row["union_candidate_id"]))].append((index, row))
    for rows in atoms_by_union.values():
        rows.sort(key=lambda item: int(str(item[1]["atom_id"]).rsplit(":", 1)[1]))

    atom_matrix = np.asarray([
        [float(row["features"][name]) for name in FEATURE_NAMES] for row in atom_rows
    ], dtype=np.float64)
    atom_target = np.asarray([
        float(row["label_retention_probability"]) for row in atom_rows
    ], dtype=np.float64)
    atom_folds = np.asarray([int(row["fold_index"]) for row in atom_rows], dtype=np.int8)
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    model_root = staging / "models"
    point_root = staging / "refined_points"
    model_root.mkdir()
    point_root.mkdir()
    try:
        atom_raw, atom_prediction, atom_models = _fit_oof(
            atom_matrix, atom_target, atom_folds, model_root,
            "member_retention", FEATURE_NAMES, 20260822,
        )
        atom_prediction_rows = []
        for index, row in enumerate(atom_rows):
            atom_prediction_rows.append({
                "atom_id": row["atom_id"],
                "scene_name": row["scene_name"],
                "fold_index": int(row["fold_index"]),
                "union_candidate_id": int(row["union_candidate_id"]),
                "raw_superpoint_id": int(row["raw_superpoint_id"]),
                "role": row["role"],
                "point_count": int(row["point_count"]),
                "point_sha256": row["point_sha256"],
                "label_retention_probability": float(atom_target[index]),
                "raw_oof_retention_probability": float(atom_raw[index]),
                "oof_retention_probability": float(atom_prediction[index]),
                "shared_forced_keep": row["role"] == "shared",
                "ap_computed": False,
            })

        nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
        node_by_key = {str(row["geometry_key"]): row for row in nodes}
        union_ledger = {
            (str(row["scene_name"]), int(row["union_candidate_id"])): row
            for row in union_rows
        }
        resolver = GeometryResolver()
        scene_cache = {}
        refined_rows = []
        quality_features = []
        quality_targets = []
        quality_folds = []
        quality_controls = []
        status_counts = Counter()
        fold_delta = defaultdict(list)
        for union_index, ((scene, union_id), ledger) in enumerate(sorted(union_ledger.items()), 1):
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
            indexed_rows = atoms_by_union[(scene, union_id)]
            if len(decomposed) != len(indexed_rows):
                raise ValueError(f"{scene}/{union_id}: atom count differs during refinement")
            probabilities = []
            for atom, (global_index, row) in zip(decomposed, indexed_rows):
                if _geometry_sha256(atom["points"]) != row["point_sha256"]:
                    raise ValueError(f"{row['atom_id']}: atom geometry differs during refinement")
                probabilities.append(float(atom_prediction[global_index]))
            kept_indexes = connected_refinement_indexes(
                decomposed, probabilities, context["neighbors"], args.retention_threshold,
            )
            shared_indexes = {index for index, atom in enumerate(decomposed) if atom["role"] == "shared"}
            if not shared_indexes <= kept_indexes:
                raise AssertionError("shared atom was removed")
            temporary = np.unique(np.concatenate([
                decomposed[index]["points"] for index in sorted(kept_indexes)
            ])).astype(np.int64) if kept_indexes else np.empty(0, np.int64)
            if len(temporary) < args.min_points:
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
            if not np.all(np.isin(refined, union_points)):
                raise AssertionError("refined union added points outside original union")
            original_best = _best_gt(union_points, gt, gt_meta)
            refined_best = _best_gt(refined, gt, gt_meta)
            original_q = _quality_target(float(original_best["best_iou"]))
            refined_q = _quality_target(float(refined_best["best_iou"]))
            delta_q = refined_q - original_q
            fold = int(ledger["fold_index"])
            fold_delta[fold].append(delta_q)
            status_counts[status] += 1
            local = staging / "refined_points" / scene
            points_file = None
            if appended:
                local.mkdir(parents=True, exist_ok=True)
                path = local / f"union{union_id:04d}_refined_points.npz"
                np.savez_compressed(path, point_indices=refined)
                points_file = str(path.relative_to(staging))
            global_indexes = [item[0] for item in indexed_rows]
            weights = [int(atom_rows[index]["point_count"]) for index in global_indexes]
            probs = [float(atom_prediction[index]) for index in global_indexes]
            kept_local = sorted(kept_indexes)
            removed_local = sorted(set(range(len(decomposed))) - kept_indexes)
            first_features = atom_rows[global_indexes[0]]["features"]
            aggregate = {
                "stage_a_union_quality": float(first_features["stage_a_union_quality"]),
                "stage_a_track_quality": float(first_features["stage_a_track_quality"]),
                "stage_a_native_quality": float(first_features["stage_a_native_quality"]),
                "stage_b_union_score": float(first_features["stage_b_union_score"]),
                "log1p_original_union_point_count": math.log1p(len(union_points)),
                "log1p_refined_point_count": math.log1p(len(refined)),
                "refined_point_fraction": len(refined) / max(1, len(union_points)),
                "atom_count": float(len(decomposed)),
                "kept_atom_fraction": len(kept_indexes) / max(1, len(decomposed)),
                "removed_atom_fraction": len(removed_local) / max(1, len(decomposed)),
                "shared_atom_fraction": sum(atom["role"] == "shared" for atom in decomposed) / max(1, len(decomposed)),
                "track_only_atom_fraction": sum(atom["role"] == "track_only" for atom in decomposed) / max(1, len(decomposed)),
                "native_only_atom_fraction": sum(atom["role"] == "native_only" for atom in decomposed) / max(1, len(decomposed)),
                "predicted_purity_point_weighted_mean": _weighted_mean(probs, weights),
                "predicted_purity_atom_mean": float(np.mean(probs)),
                "predicted_purity_atom_min": float(np.min(probs)),
                "predicted_purity_atom_max": float(np.max(probs)),
                "kept_predicted_purity_point_weighted_mean": _weighted_mean(
                    [probs[index] for index in kept_local], [weights[index] for index in kept_local]
                ),
                "removed_predicted_purity_point_weighted_mean": _weighted_mean(
                    [probs[index] for index in removed_local], [weights[index] for index in removed_local]
                ),
                **{
                    f"relation__{name}": float(first_features[f"relation__{name}"])
                    for name in RELATION_FEATURES
                },
            }
            quality_features.append([float(aggregate[name]) for name in QUALITY_FEATURE_NAMES])
            quality_targets.append(refined_q)
            quality_folds.append(fold)
            quality_controls.append(float(aggregate["stage_b_union_score"]))
            refined_rows.append({
                "scene_name": scene,
                "fold_index": fold,
                "union_candidate_id": union_id,
                "original_union_geometry_key": ledger["union_geometry_key"],
                "original_union_point_count": len(union_points),
                "original_union_sha256": _geometry_sha256(union_points),
                "refined_point_count": len(refined),
                "refined_geometry_sha256": _geometry_sha256(refined),
                "refined_points_file": points_file,
                "materialization_status": status,
                "append_refined_candidate": appended,
                "original_union_retained": True,
                "atom_count": len(decomposed),
                "kept_atom_count": len(kept_indexes),
                "removed_atom_count": len(removed_local),
                "kept_point_count_before_fallback": len(temporary),
                "shared_atom_count": len(shared_indexes),
                "shared_point_count": int(sum(len(decomposed[index]["points"]) for index in shared_indexes)),
                "shared_point_deleted_count": 0,
                "outside_union_added_point_count": 0,
                "quality_features": aggregate,
                "label_original_union_best_iou": float(original_best["best_iou"]),
                "label_original_union_quality_q": original_q,
                "label_refined_best_iou": float(refined_best["best_iou"]),
                "label_refined_quality_q": refined_q,
                "label_quality_delta": delta_q,
                "candidate_deletion": False,
                "parent_geometry_mutation": False,
                "frozen_cache_write": False,
                "ap_computed": False,
            })
            if union_index % 200 == 0:
                print(f"[stage C refinement] {union_index}/{len(union_ledger)}", flush=True)

        quality_matrix = np.asarray(quality_features, dtype=np.float64)
        quality_target = np.asarray(quality_targets, dtype=np.float64)
        quality_folds_array = np.asarray(quality_folds, dtype=np.int8)
        quality_raw, quality_prediction, quality_models = _fit_oof(
            quality_matrix, quality_target, quality_folds_array, model_root,
            "refined_quality", QUALITY_FEATURE_NAMES, 20260827,
        )
        for index, row in enumerate(refined_rows):
            row["raw_oof_refined_quality"] = float(quality_raw[index])
            row["stage_c_oof_quality"] = float(quality_prediction[index])

        atom_prediction_path = staging / "oof_member_predictions.jsonl"
        refined_path = staging / "refined_union_plan.jsonl"
        atom_prediction_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in atom_prediction_rows
        ))
        refined_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in refined_rows
        ))
        atom_overall = metrics(atom_target, atom_prediction)
        atom_by_fold = {
            str(fold): metrics(atom_target[atom_folds == fold], atom_prediction[atom_folds == fold])
            for fold in range(5)
        }
        quality_control = np.clip(np.asarray(quality_controls, dtype=np.float64), 0.0, 1.0)
        quality_overall = {
            "stage_c_oof_quality": metrics(quality_target, quality_prediction),
            "stage_b_union_score_control": metrics(quality_target, quality_control),
        }
        quality_by_fold = {
            str(fold): metrics(
                quality_target[quality_folds_array == fold],
                quality_prediction[quality_folds_array == fold],
            ) for fold in range(5)
        }
        deltas = np.asarray([float(row["label_quality_delta"]) for row in refined_rows])
        change_counts = {
            "improved": int(np.sum(deltas > 1e-12)),
            "neutral": int(np.sum(np.abs(deltas) <= 1e-12)),
            "harmed": int(np.sum(deltas < -1e-12)),
        }
        fold_mean_delta = {
            str(fold): float(np.mean(fold_delta[fold])) for fold in range(5)
        }
        checks = {
            "member_oof_mae_at_most_0.25": atom_overall["mae"] <= 0.25,
            "member_oof_absolute_mean_bias_at_most_0.10": atom_overall["absolute_mean_bias"] <= 0.10,
            "all_folds_have_member_samples": all(atom_by_fold[str(fold)]["count"] > 0 for fold in range(5)),
            "shared_point_deletion_zero": all(row["shared_point_deleted_count"] == 0 for row in refined_rows),
            "outside_union_addition_zero": all(row["outside_union_added_point_count"] == 0 for row in refined_rows),
            "original_unions_retained": all(row["original_union_retained"] for row in refined_rows),
            "overall_refined_quality_strictly_improves": float(deltas.mean()) > 0.0,
            **{f"fold_{fold}_quality_delta_at_least_minus_0.01": fold_mean_delta[str(fold)] >= -0.01 for fold in range(5)},
            "refined_score_absolute_mean_bias_at_most_0.10": quality_overall["stage_c_oof_quality"]["absolute_mean_bias"] <= 0.10,
        }
        summary = {
            "version": VERSION,
            "union_count": len(refined_rows),
            "atom_count": len(atom_rows),
            "retention_threshold": args.retention_threshold,
            "minimum_refined_point_count": args.min_points,
            "model_params": MODEL_PARAMS,
            "member_metrics": {"overall": atom_overall, "by_fold": atom_by_fold},
            "refined_quality_metrics": {"overall": quality_overall, "by_fold": quality_by_fold},
            "materialization_status_counts": dict(sorted(status_counts.items())),
            "quality_change_counts": change_counts,
            "mean_quality_delta": float(deltas.mean()),
            "fold_mean_quality_delta": fold_mean_delta,
            "shared_point_deleted_count": 0,
            "outside_union_added_point_count": 0,
            "original_union_deleted_count": 0,
            "models": atom_models + quality_models,
            "files": {
                "member_predictions": atom_prediction_path.name,
                "refined_plan": refined_path.name,
            },
            "hashes": {
                "member_predictions": _sha256(atom_prediction_path),
                "refined_plan": _sha256(refined_path),
            },
            "advancement_gate": {
                "checks": checks,
                "advancement_authorized_pending_independent_audit": all(checks.values()),
            },
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
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
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
    parser.add_argument("--retention-threshold", type=float, default=0.50)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    if args.retention_threshold != 0.50 or args.min_points != 100:
        raise ValueError("stage-C preregistration freezes threshold=0.50 and min_points=100")
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "union_count": result["union_count"],
        "mean_quality_delta": result["mean_quality_delta"],
        "advancement_authorized_pending_independent_audit": result["advancement_gate"]["advancement_authorized_pending_independent_audit"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

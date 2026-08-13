#!/usr/bin/env python3
"""Train nested OOF relative-quality models on reliable same-target pairs.

Only ``prefer_track`` and ``prefer_native`` official-train labels are used.
Different-target coexist, equivalent/abstain, and unknown relations are
excluded.  The model estimates whether the frozen track geometry is better
than the frozen native exact-geometry group, conditional on the relation
already being known to represent the same target.

This first version deliberately excludes learned candidate-quality scores.
It exports conditional OOF diagnostics only and never chooses a target gate,
replacement threshold, action plan, candidate mutation, or AP evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_train_candidate_relation_feature_ledger import (  # noqa: E402
    EXCLUSIVE_PAIR_MODEL_FEATURES,
)
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    LEARNED_QUALITY_MARKERS,
    OUTER_FOLD_COUNT,
    _prediction_bootstrap,
    grouped_metrics,
    nested_fit_predict,
    scene_track_balanced_weights,
)


BOOTSTRAP_SEED = 20260820


MODEL_FEATURES = {
    "A_original_scores": (
        "track_original_score",
        "native_original_score_median",
        "original_score_delta_track_minus_native",
    ),
    "B_plus_directional_geometry": (
        "track_original_score",
        "native_original_score_median",
        "original_score_delta_track_minus_native",
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance_normalized",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
    ),
    "C_plus_topology_appearance": (
        "track_original_score",
        "native_original_score_median",
        "original_score_delta_track_minus_native",
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance_normalized",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
        "shared_superpoint_count",
        "track_shared_superpoint_fraction",
        "native_shared_superpoint_fraction",
        "track_superpoint_count",
        "native_superpoint_count",
        "track_superpoint_component_count",
        "native_superpoint_component_count",
        "native_superpoint_full_occupancy_fraction",
        "native_superpoint_mean_occupancy",
        "mean_rgb_distance",
        "mean_normal_absolute_dot",
        "mean_normal_difference",
        "track_normal_coherence",
        "native_normal_coherence",
        "exclusive_boundary_superpoint_edge_count",
        "exclusive_boundary_contact_point_count",
        "exclusive_boundary_contact_ratio_mean",
        "exclusive_boundary_contact_ratio_max",
        "exclusive_boundary_distance_weighted_mean",
        "exclusive_boundary_normal_difference_weighted_mean",
        "exclusive_boundary_color_difference_weighted_mean",
    ),
    "D_plus_public_component": (
        "track_original_score",
        "native_original_score_median",
        "original_score_delta_track_minus_native",
        "point_iou",
        "track_inside_native_ratio",
        "native_inside_track_ratio",
        "log_track_over_native_point_count",
        "track_point_count",
        "native_point_count",
        "aabb_iou",
        "track_aabb_coverage",
        "native_aabb_coverage",
        "centroid_distance_normalized",
        "track_bbox_diagonal",
        "native_bbox_diagonal",
        "shared_superpoint_count",
        "track_shared_superpoint_fraction",
        "native_shared_superpoint_fraction",
        "track_superpoint_count",
        "native_superpoint_count",
        "track_superpoint_component_count",
        "native_superpoint_component_count",
        "native_superpoint_full_occupancy_fraction",
        "native_superpoint_mean_occupancy",
        "mean_rgb_distance",
        "mean_normal_absolute_dot",
        "mean_normal_difference",
        "track_normal_coherence",
        "native_normal_coherence",
        "exclusive_boundary_superpoint_edge_count",
        "exclusive_boundary_contact_point_count",
        "exclusive_boundary_contact_ratio_mean",
        "exclusive_boundary_contact_ratio_max",
        "exclusive_boundary_distance_weighted_mean",
        "exclusive_boundary_normal_difference_weighted_mean",
        "exclusive_boundary_color_difference_weighted_mean",
        "public_common_eligible_view_count",
        "public_common_selected_view_count",
        "public_common_view_available",
        "public_projected_box_iou_mean",
        "public_projected_box_iou_min",
        "public_projected_box_iou_max",
        "public_same_matched_observation_fraction",
        "public_different_matched_observation_fraction",
        "public_track_gvc_mean",
        "public_native_gvc_mean",
        "public_track_minus_native_gvc",
        "public_track_visible_fraction_mean",
        "public_native_visible_fraction_mean",
        "public_centroid_camera_range_absolute_delta_mean",
        "public_centroid_camera_range_absolute_delta_max",
        "relation_component_track_count",
        "relation_component_native_group_count",
        "relation_component_native_member_candidate_count",
        "relation_component_relation_count",
        "track_overlapping_native_group_count",
        "track_overlapping_native_member_candidate_count",
        "native_group_overlapping_track_count",
        "hypothetical_pair_native_member_deletion_count",
        "hypothetical_track_wide_native_member_deletion_count",
    ),
}
MODEL_FEATURES["E_plus_exclusive_pair"] = (
    MODEL_FEATURES["D_plus_public_component"] + EXCLUSIVE_PAIR_MODEL_FEATURES
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_feature_contract(rows: list[dict]) -> None:
    available = set(rows[0]["features"]) if rows else set()
    for model_name, feature_names in MODEL_FEATURES.items():
        missing = set(feature_names) - available
        if missing:
            raise ValueError(f"{model_name}: missing features: {sorted(missing)}")
        forbidden = [
            name for name in feature_names
            if any(marker in name for marker in LEARNED_QUALITY_MARKERS)
            or name.startswith("label_")
            or "gt_" in name
        ]
        if forbidden:
            raise ValueError(f"{model_name}: forbidden learned/GT features: {forbidden}")


def _quality_rows(rows: list[dict]) -> list[dict]:
    selected = [
        row for row in rows
        if row["labels"]["target_state"] == "same_target"
        and row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native")
    ]
    if not selected:
        raise ValueError("no reliable same-target relative-quality rows")
    if any(not row["labels"]["reliable_pair"] for row in selected):
        raise ValueError("relative-quality rows must all be reliable")
    return selected


def _labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([
        int(row["labels"]["relative_quality_state"] == "prefer_track") for row in rows
    ], dtype=np.int64)


def _matrix(rows: list[dict], feature_names: tuple[str, ...]) -> np.ndarray:
    matrix = np.asarray([
        [float(row["features"][name]) for name in feature_names] for row in rows
    ], dtype=np.float64)
    if matrix.shape != (len(rows), len(feature_names)) or not np.isfinite(matrix).all():
        raise ValueError("relative-quality matrix is malformed or non-finite")
    return matrix


def run_oof(
    all_rows: list[dict],
    scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]],
    bootstrap_repetitions: int,
) -> tuple[dict, list[dict], list[dict], dict]:
    _validate_feature_contract(all_rows)
    rows = _quality_rows(all_rows)
    labels = _labels(rows)
    matrices = {name: _matrix(rows, features) for name, features in MODEL_FEATURES.items()}
    predictions = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in MODEL_FEATURES}
    assignments = np.zeros(len(rows), dtype=np.int64)
    fold_metrics, diagnostics = [], {name: [] for name in MODEL_FEATURES}
    for fold_index in range(OUTER_FOLD_COUNT):
        train = np.asarray([
            index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] != fold_index
        ], dtype=np.int64)
        validation = np.asarray([
            index for index, row in enumerate(rows) if scene_to_fold[row["scene_name"]] == fold_index
        ], dtype=np.int64)
        train_scenes = {rows[index]["scene_name"] for index in train}
        validation_scenes = {rows[index]["scene_name"] for index in validation}
        if len(train_scenes) != 80 or len(validation_scenes) != 20 or train_scenes & validation_scenes:
            raise AssertionError("relative-quality rows do not follow frozen outer split")
        if set(np.unique(labels[train])) != {0, 1}:
            raise ValueError(f"outer fold {fold_index} training partition lacks a class")
        assignments[validation] += 1
        validation_weights = scene_track_balanced_weights(rows, validation)[validation]
        fold = {
            "fold_index": fold_index,
            "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "train_relation_count": len(train),
            "validation_relation_count": len(validation),
            "validation_prefer_track_count": int(labels[validation].sum()),
            "models": {},
        }
        for model_name, matrix in matrices.items():
            values, model_diagnostics = nested_fit_predict(
                rows, matrix, labels, train, validation, fold_index
            )
            predictions[model_name][validation] = values
            model_diagnostics["feature_names"] = list(MODEL_FEATURES[model_name])
            diagnostics[model_name].append(model_diagnostics)
            fold["models"][model_name] = grouped_metrics(
                [rows[index] for index in validation],
                labels[validation], values, validation_weights,
            )
        fold_metrics.append(fold)
    if not np.all(assignments == 1) or any(not np.isfinite(values).all() for values in predictions.values()):
        raise AssertionError("every relative-quality relation must receive one OOF prediction")
    evaluation_weights = scene_track_balanced_weights(rows)
    overall = {
        name: grouped_metrics(rows, labels, values, evaluation_weights)
        for name, values in predictions.items()
    }
    cohort_metrics = {}
    for cohort_name, cohort_scenes in cohorts.items():
        indexes = np.asarray([
            index for index, row in enumerate(rows) if row["scene_name"] in cohort_scenes
        ], dtype=np.int64)
        weights = scene_track_balanced_weights(rows, indexes)[indexes]
        cohort_metrics[cohort_name] = {
            "scene_count": len(cohort_scenes),
            "relation_count": len(indexes),
            "prefer_track_count": int(labels[indexes].sum()),
            "models": {
                name: grouped_metrics(
                    [rows[index] for index in indexes], labels[indexes], values[indexes], weights
                )
                for name, values in predictions.items()
            },
        }
    bootstrap = {
        model_name: {
            kind: _prediction_bootstrap(
                rows, labels, values, kind, bootstrap_repetitions,
                BOOTSTRAP_SEED + model_index * 10 + kind_index,
            )
            for kind_index, kind in enumerate(("scene", "track"))
        }
        for model_index, (model_name, values) in enumerate(predictions.items())
    }
    oof_rows = [
        {
            "scene_name": row["scene_name"],
            "fold_index": int(scene_to_fold[row["scene_name"]]),
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "label_relative_quality_state": row["labels"]["relative_quality_state"],
            "label_prefer_track": int(labels[index]),
            "evaluation_scene_track_weight": float(evaluation_weights[index]),
            "predictions": {name: float(values[index]) for name, values in predictions.items()},
            "ground_truth_usage": "official_train_offline_label_only",
        }
        for index, row in enumerate(rows)
    ]
    summary = {
        "scene_count": 100,
        "relation_count": len(rows),
        "prefer_track_count": int(labels.sum()),
        "prefer_native_count": int((labels == 0).sum()),
        "outer_fold_count": OUTER_FOLD_COUNT,
        "inner_fold_count": 4,
        "overall_metrics": overall,
        "cohort_metrics": cohort_metrics,
        "model_cluster_bootstrap": bootstrap,
        "model_selection_applied": False,
    }
    return summary, oof_rows, fold_metrics, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    actual_split_sha = _sha256(args.split_manifest)
    if actual_split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    all_rows = _load_rows(args.feature_ledger_root, scenes)
    summary_core, oof_rows, fold_metrics, diagnostics = run_oof(
        all_rows, scene_to_fold, cohorts, args.bootstrap_repetitions
    )
    summary = {
        "version": f"{args.protocol_name}_candidate_relative_quality_nested_oof_v1",
        "protocol_name": args.protocol_name,
        **summary_core,
        "feature_groups": {name: list(features) for name, features in MODEL_FEATURES.items()},
        "training_contract": {
            "task": "conditional on reliable same-target relation, predict prefer_track versus prefer_native",
            "excluded_relation_states": ["different_target_coexist", "equivalent_abstain", "unknown"],
            "outer_split": "frozen official100 five-fold scene-disjoint 80/20",
            "inner_split": "four-fold scene-disjoint within each outer training partition",
            "model": "standardized L2 logistic regression with nested Platt calibration",
            "fit_weighting": "scene-track balanced, then class balanced inside each fit partition",
            "learned_candidate_quality_features_used": False,
            "learned_candidate_quality_exclusion_reason": "first relative-quality ablation measures raw relation evidence before implementing nested candidate-quality predictions",
        },
        "safety_contract": {
            "fit_all_model_exported": False,
            "target_gate_threshold_selected": False,
            "relative_quality_threshold_selected": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "ap_evaluation_run": False,
            "cascade_evaluated": False,
        },
        "input_provenance": {
            "feature_ledger_summary_path": str((args.feature_ledger_root / "summary.json").resolve()),
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": _sha256(args.scene_list),
            "existing20_scene_list_path": str(args.existing20_scene_list.resolve()),
            "existing20_scene_list_sha256": _sha256(args.existing20_scene_list),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": actual_split_sha,
            "split_existing_scene_count": split_payload.get("existing_scene_count"),
            "split_new_scene_count": split_payload.get("new_scene_count"),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "fold_metrics.json").write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "model_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "oof_predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in oof_rows
        ))
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

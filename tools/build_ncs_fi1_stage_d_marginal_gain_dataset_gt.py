#!/usr/bin/env python3
"""Build the preregistered stage-D continuous marginal-gain dataset.

GT is used only for labels.  Features come from frozen stage A/B/C-v2 OOF
artifacts and read-only geometry.  The tool never computes AP, mutates a
candidate, writes a frozen cache, or reads validation60/val312.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver, UNION_FEATURE_NAMES, _fold_by_scene, _quality_target,
    _read_jsonl, _read_scenes, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import _geometry_sha256  # noqa: E402
from tools.build_train_scene_candidate_quality_ledger import _load_gt  # noqa: E402
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import QUALITY_FEATURE_NAMES  # noqa: E402


VERSION = "ncs_fi1_stage_d_marginal_gain_dataset_v1"
EXPECTED_BASELINE_COUNT = 8591
EXPECTED_ORIGINAL_COUNT = 1578
EXPECTED_REFINED_COUNT = 85
OFFICIAL_THRESHOLDS = tuple(value / 100.0 for value in range(50, 95, 5))
RELATION_TYPES = (
    "exact_duplicate", "near_duplicate", "containment",
    "complementary", "conflict", "uncertain",
)

BASE_FEATURE_NAMES = (
    "variant_refined",
    "log1p_candidate_point_count",
    "candidate_point_fraction_of_scene",
    "candidate_point_fraction_of_original_union",
    "candidate_removed_point_fraction_from_original_union",
    "stage_a_original_union_quality",
    "stage_b_original_union_score",
    "frozen_threshold_cross_probability",
    "frozen_union_base_quality",
    "frozen_track_quality_q",
    "frozen_native_group_median_quality_q",
    "stage_c_candidate_quality_prediction",
    "stage_c_candidate_quality_lower_bound",
    "geometry_quality_for_scoring",
    "stage_c_confident_removal_fraction",
    "stage_c_connectivity_removal_fraction",
    "stage_c_total_removal_fraction",
    "baseline_max_point_iou",
    "baseline_max_candidate_inside_ratio",
    "baseline_max_baseline_inside_ratio",
    "baseline_max_min_point_count_ratio",
    "baseline_candidate_new_point_fraction",
    "baseline_intersection_count",
    "baseline_iou_ge_005_count",
    "baseline_near_duplicate_count",
    "baseline_containment_count",
    "baseline_overlap_stage_a_quality_max",
    "baseline_overlap_stage_a_quality_mean",
    "baseline_overlap_stage_b_score_max",
    "baseline_overlap_stage_b_score_mean",
    "stage_b_relation_count",
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *tuple(f"stage_a_union__{name}" for name in UNION_FEATURE_NAMES),
    *tuple(f"stage_c__{name}" for name in QUALITY_FEATURE_NAMES),
    *tuple(f"stage_b_relation_count__{name}" for name in RELATION_TYPES),
    *tuple(f"stage_b_relation_fraction__{name}" for name in RELATION_TYPES),
)


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def soft_quality(iou: float) -> float:
    return float(np.clip((float(iou) - 0.50) / 0.45, 0.0, 1.0))


def iou_by_gt(points: np.ndarray, gt: np.ndarray, gt_meta: dict[int, dict]) -> dict[int, float]:
    if not len(points):
        return {}
    labels, intersections = np.unique(gt[points], return_counts=True)
    result = {}
    for encoded, intersection in zip(labels, intersections):
        encoded = int(encoded)
        meta = gt_meta.get(encoded)
        if meta is None:
            continue
        intersection = int(intersection)
        union = len(points) + int(meta["point_count"]) - intersection
        result[encoded] = float(intersection / max(1, union))
    return result


def update_existing_best(
    existing_best: dict[int, float], points: np.ndarray, gt: np.ndarray,
    gt_meta: dict[int, dict],
) -> None:
    for encoded, iou in iou_by_gt(points, gt, gt_meta).items():
        existing_best[encoded] = max(existing_best[encoded], iou)


def label_candidate(candidate_iou: dict[int, float], existing_best: dict[int, float]) -> dict:
    gain_candidates = []
    q_gains = []
    soft_gains = []
    threshold_candidates = []
    for encoded, iou in candidate_iou.items():
        base = float(existing_best[encoded])
        gain = max(0.0, float(iou) - base)
        gain_candidates.append((gain, float(iou), -encoded, encoded, base))
        q_gains.append(max(0.0, _quality_target(iou) - _quality_target(base)))
        soft_gains.append(max(0.0, soft_quality(iou) - soft_quality(base)))
        crossings = sum(iou >= threshold and base < threshold for threshold in OFFICIAL_THRESHOLDS)
        threshold_candidates.append((crossings, float(iou) - base, float(iou), -encoded, encoded, base))
    if gain_candidates:
        gain, selected_iou, _, selected_gt, selected_base = max(gain_candidates)
        candidate_best_iou = max(candidate_iou.values())
        threshold = max(threshold_candidates)
    else:
        gain = selected_iou = selected_base = candidate_best_iou = 0.0
        selected_gt = None
        threshold = (0, 0.0, 0.0, 0, None, 0.0)
    return {
        "selected_gain_gt_encoded_id": selected_gt,
        "selected_gain_candidate_iou": float(selected_iou),
        "existing_best_iou_for_selected_gain_target": float(selected_base),
        "candidate_best_iou": float(candidate_best_iou),
        "candidate_quality_q": _quality_target(candidate_best_iou),
        "marginal_iou_gain": float(gain),
        "marginal_q_gain": float(max(q_gains, default=0.0)),
        "marginal_soft_quality_gain": float(max(soft_gains, default=0.0)),
        "threshold_cross_target_gt_encoded_id": threshold[4],
        "official_threshold_cross_count": int(threshold[0]),
        "official_threshold_cross_fraction": float(threshold[0] / len(OFFICIAL_THRESHOLDS)),
        "crosses_any_official_threshold": bool(threshold[0] > 0),
    }


def overlap_features(
    candidate: np.ndarray, baseline: list[tuple[dict, np.ndarray]],
    stage_a_by_key: dict[str, dict], stage_b_by_key: dict[str, dict],
) -> dict[str, float]:
    point_ious = []
    candidate_inside = []
    baseline_inside = []
    min_ratios = []
    stage_a_scores = []
    stage_b_scores = []
    near_count = containment_count = iou_005_count = 0
    for node, points in baseline:
        intersection = int(len(np.intersect1d(candidate, points, assume_unique=True)))
        if not intersection:
            continue
        union = len(candidate) + len(points) - intersection
        iou = intersection / max(1, union)
        inside_candidate = intersection / max(1, len(candidate))
        inside_baseline = intersection / max(1, len(points))
        point_ious.append(iou)
        candidate_inside.append(inside_candidate)
        baseline_inside.append(inside_baseline)
        min_ratios.append(min(len(candidate), len(points)) / max(len(candidate), len(points)))
        key = str(node["geometry_key"])
        stage_a_scores.append(float(stage_a_by_key[key]["oof_unified_quality"]))
        stage_b_scores.append(float(stage_b_by_key[key]["stage_b_score"]))
        iou_005_count += int(iou >= 0.05)
        near_count += int(min(inside_candidate, inside_baseline) >= 0.90)
        containment_count += int(
            max(inside_candidate, inside_baseline) >= 0.90
            and min(inside_candidate, inside_baseline) < 0.90
        )
    return {
        "baseline_max_point_iou": max(point_ious, default=0.0),
        "baseline_max_candidate_inside_ratio": max(candidate_inside, default=0.0),
        "baseline_max_baseline_inside_ratio": max(baseline_inside, default=0.0),
        "baseline_max_min_point_count_ratio": max(min_ratios, default=0.0),
        "baseline_candidate_new_point_fraction": 1.0 - max(candidate_inside, default=0.0),
        "baseline_intersection_count": float(len(point_ious)),
        "baseline_iou_ge_005_count": float(iou_005_count),
        "baseline_near_duplicate_count": float(near_count),
        "baseline_containment_count": float(containment_count),
        "baseline_overlap_stage_a_quality_max": max(stage_a_scores, default=0.0),
        "baseline_overlap_stage_a_quality_mean": float(np.mean(stage_a_scores)) if stage_a_scores else 0.0,
        "baseline_overlap_stage_b_score_max": max(stage_b_scores, default=0.0),
        "baseline_overlap_stage_b_score_mean": float(np.mean(stage_b_scores)) if stage_b_scores else 0.0,
    }


def relation_features(original_key: str, relations_by_geometry: dict[str, Counter]) -> dict[str, float]:
    counts = relations_by_geometry.get(original_key, Counter())
    total = int(sum(counts.values()))
    output = {"stage_b_relation_count": float(total)}
    for relation_type in RELATION_TYPES:
        count = int(counts[relation_type])
        output[f"stage_b_relation_count__{relation_type}"] = float(count)
        output[f"stage_b_relation_fraction__{relation_type}"] = count / max(1, total)
    return output


def candidate_features(
    *, variant: str, points: np.ndarray, original_points: np.ndarray, scene_point_count: int,
    original_union: dict, original_stage_a_row: dict, original_stage_b_row: dict,
    stage_c_row: dict, overlap: dict[str, float], relation: dict[str, float],
) -> dict[str, float]:
    refined = variant == "refined"
    atom_count = max(1, int(stage_c_row["atom_count"]))
    if refined:
        quality_prediction = float(stage_c_row["corrected_oof_temporary_refined_quality"])
        quality_lower = float(stage_c_row["quality_lower_confidence_bound"])
        geometry_quality = float(np.clip(quality_lower, 0.0, 1.0))
    else:
        quality_prediction = float(original_stage_a_row["oof_unified_quality"])
        quality_lower = quality_prediction
        geometry_quality = float(np.clip(original_stage_b_row["stage_b_score"], 0.0, 1.0))
    values = {
        "variant_refined": float(refined),
        "log1p_candidate_point_count": math.log1p(len(points)),
        "candidate_point_fraction_of_scene": len(points) / max(1, scene_point_count),
        "candidate_point_fraction_of_original_union": len(points) / max(1, len(original_points)),
        "candidate_removed_point_fraction_from_original_union": 1.0 - len(points) / max(1, len(original_points)),
        "stage_a_original_union_quality": float(original_stage_a_row["oof_unified_quality"]),
        "stage_b_original_union_score": float(original_stage_b_row["stage_b_score"]),
        "frozen_threshold_cross_probability": float(original_union["threshold_cross_probability"]),
        "frozen_union_base_quality": float(original_union["base_quality"]),
        "frozen_track_quality_q": float(original_union["track_quality_q"]),
        "frozen_native_group_median_quality_q": float(original_union["native_group_median_quality_q"]),
        "stage_c_candidate_quality_prediction": quality_prediction,
        "stage_c_candidate_quality_lower_bound": quality_lower,
        "geometry_quality_for_scoring": geometry_quality,
        "stage_c_confident_removal_fraction": int(stage_c_row["confident_removal_count"]) / atom_count if refined else 0.0,
        "stage_c_connectivity_removal_fraction": int(stage_c_row["connectivity_removal_count"]) / atom_count if refined else 0.0,
        "stage_c_total_removal_fraction": (
            int(stage_c_row["confident_removal_count"]) + int(stage_c_row["connectivity_removal_count"])
        ) / atom_count if refined else 0.0,
        **overlap,
        **relation,
    }
    values.update({
        f"stage_a_union__{name}": float(original_stage_a_row["features"][name])
        for name in UNION_FEATURE_NAMES
    })
    values.update({
        f"stage_c__{name}": float(stage_c_row["quality_features"][name])
        for name in QUALITY_FEATURE_NAMES
    })
    if set(values) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(values))
        extra = sorted(set(values) - set(FEATURE_NAMES))
        raise ValueError(f"feature schema differs; missing={missing}, extra={extra}")
    return {name: _finite(values[name], name) for name in FEATURE_NAMES}


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "fold_manifest", "ground_truth_root", "unique_geometry_root",
        "stage_a_dataset_root", "stage_a_oof_root", "stage_b_root", "stage_b_audit_root",
        "champion_plan_root", "stage_c_v2_root", "stage_c_v2_audit_root",
        "preregistration", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    folds = _fold_by_scene(args.fold_manifest, scenes)
    if len(scenes) != 100:
        raise ValueError("stage D requires exactly NCS-train100")
    stage_b_audit = json.loads((args.stage_b_audit_root / "summary.json").read_text())
    stage_c_audit = json.loads((args.stage_c_v2_audit_root / "summary.json").read_text())
    if stage_b_audit.get("audit_valid") is not True:
        raise ValueError("stage-B independent audit is not valid")
    if stage_c_audit.get("audit_valid") is not True or not stage_c_audit.get("advancement_gate", {}).get("advancement_authorized"):
        raise ValueError("stage-C-v2 independent audit did not authorize stage D")

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    nodes_by_scene = defaultdict(list)
    node_by_key = {}
    for node in nodes:
        nodes_by_scene[str(node["scene_name"])].append(node)
        node_by_key[str(node["geometry_key"])] = node
    baseline_count = sum(
        str(node["canonical_candidate_source"]) in {"native", "track"} for node in nodes
    )
    if baseline_count != EXPECTED_BASELINE_COUNT:
        raise ValueError(f"baseline geometry count differs: {baseline_count}")

    stage_a_dataset = _read_jsonl(args.stage_a_dataset_root / "quality_dataset.jsonl")
    stage_a_features_by_key = {str(row["geometry_key"]): row for row in stage_a_dataset}
    stage_a_oof = _read_jsonl(args.stage_a_oof_root / "oof_quality_predictions.jsonl")
    stage_a_by_key = {}
    for row in stage_a_oof:
        key = str(row["geometry_key"])
        if key not in stage_a_features_by_key:
            raise ValueError(f"stage-A OOF row lacks dataset features: {key}")
        stage_a_by_key[key] = {**row, "features": stage_a_features_by_key[key]["features"]}
    stage_b_plan = _read_jsonl(args.stage_b_root / "stage_b_rerank_plan.jsonl")
    stage_b_by_key = {str(row["geometry_key"]): row for row in stage_b_plan}
    stage_b_relations = _read_jsonl(args.stage_b_root / "candidate_relations.jsonl")
    relations_by_geometry = defaultdict(Counter)
    for row in stage_b_relations:
        relation_type = str(row["relation_type"])
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"unexpected stage-B relation type: {relation_type}")
        relations_by_geometry[str(row["geometry_key_a"])][relation_type] += 1
        relations_by_geometry[str(row["geometry_key_b"])][relation_type] += 1

    original_unions = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    original_union_by_key = {
        (str(row["scene_name"]), int(row["candidate_id"])): row for row in original_unions
    }
    stage_c_rows = _read_jsonl(args.stage_c_v2_root / "stage_c_v2_refined_union_plan.jsonl")
    stage_c_by_key = {
        (str(row["scene_name"]), int(row["union_candidate_id"])): row for row in stage_c_rows
    }
    refined_count = sum(bool(row["append_eligible_refined_union"]) for row in stage_c_rows)
    if len(original_unions) != EXPECTED_ORIGINAL_COUNT or len(stage_c_rows) != EXPECTED_ORIGINAL_COUNT:
        raise ValueError("original/stage-C union count differs from preregistration")
    if refined_count != EXPECTED_REFINED_COUNT:
        raise ValueError(f"refined candidate count differs: {refined_count}")

    resolver = GeometryResolver()
    output_rows = []
    scene_summaries = []
    positive_by_fold = Counter()
    for scene_index, scene in enumerate(scenes, 1):
        gt_path = args.ground_truth_root / f"{scene}.txt"
        gt, gt_meta = _load_gt(gt_path, args.min_gt_points)
        baseline = []
        existing_best = {encoded: 0.0 for encoded in gt_meta}
        for node in nodes_by_scene[scene]:
            if str(node["canonical_candidate_source"]) not in {"native", "track"}:
                continue
            points = resolver.points(
                node["canonical_geometry_locator"], int(node["point_count"]), len(gt)
            )
            baseline.append((node, points))
            update_existing_best(existing_best, points, gt, gt_meta)
        local_original = local_refined = local_positive = 0
        local_keys = sorted(key for key in original_union_by_key if key[0] == scene)
        for key in local_keys:
            original_union = original_union_by_key[key]
            stage_c_row = stage_c_by_key[key]
            original_key = str(stage_c_row["original_union_geometry_key"])
            node = node_by_key[original_key]
            original_points = resolver.points(
                node["canonical_geometry_locator"], int(node["point_count"]), len(gt)
            )
            variants = [("original", original_points, node["canonical_geometry_locator"])]
            if stage_c_row["append_eligible_refined_union"]:
                refined_path = args.stage_c_v2_root / str(stage_c_row["refined_points_file"])
                with np.load(refined_path) as payload:
                    refined_points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
                variants.append((
                    "refined", refined_points,
                    {"kind": "point_indices_npz", "points_path": str(refined_path), "array_key": "point_indices"},
                ))
            relation = relation_features(original_key, relations_by_geometry)
            for variant, points, locator in variants:
                overlap = overlap_features(points, baseline, stage_a_by_key, stage_b_by_key)
                features = candidate_features(
                    variant=variant, points=points, original_points=original_points,
                    scene_point_count=len(gt), original_union=original_union,
                    original_stage_a_row=stage_a_by_key[original_key],
                    original_stage_b_row=stage_b_by_key[original_key],
                    stage_c_row=stage_c_row, overlap=overlap, relation=relation,
                )
                labels = label_candidate(iou_by_gt(points, gt, gt_meta), existing_best)
                candidate_key = f"{scene}:union:{key[1]:04d}:{variant}"
                row = {
                    "candidate_key": candidate_key,
                    "scene_name": scene,
                    "fold_index": int(folds[scene]),
                    "union_candidate_id": int(key[1]),
                    "candidate_variant": variant,
                    "original_union_geometry_key": original_key,
                    "geometry_locator_read_only": locator,
                    "geometry_sha256": _geometry_sha256(points),
                    "point_count": int(len(points)),
                    "features": features,
                    "labels": labels,
                    "ground_truth_usage": "NCS-train100 label fields only",
                    "feature_ground_truth_usage": "none",
                    "candidate_retained": True,
                    "candidate_deletion": False,
                    "candidate_mutation": False,
                    "geometry_mutation": False,
                    "class_mutation": False,
                    "frozen_cache_write": False,
                    "ap_computed": False,
                }
                output_rows.append(row)
                local_original += int(variant == "original")
                local_refined += int(variant == "refined")
                if labels["marginal_iou_gain"] > 0.0:
                    local_positive += 1
                    positive_by_fold[int(folds[scene])] += 1
        scene_summaries.append({
            "scene_name": scene,
            "fold_index": int(folds[scene]),
            "baseline_geometry_count": len(baseline),
            "gt_instance_count": len(gt_meta),
            "original_candidate_count": local_original,
            "refined_candidate_count": local_refined,
            "positive_marginal_gain_count": local_positive,
            "ground_truth_sha256": _sha256(gt_path),
        })
        print(
            f"[stage D dataset] {scene_index}/{len(scenes)} {scene}: "
            f"baseline={len(baseline)} original={local_original} refined={local_refined} positive={local_positive}",
            flush=True,
        )

    output_rows.sort(key=lambda row: row["candidate_key"])
    candidate_keys = [row["candidate_key"] for row in output_rows]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("duplicate stage-D candidate key")
    original_count = sum(row["candidate_variant"] == "original" for row in output_rows)
    refined_count = sum(row["candidate_variant"] == "refined" for row in output_rows)
    if original_count != EXPECTED_ORIGINAL_COUNT or refined_count != EXPECTED_REFINED_COUNT:
        raise ValueError("materialized action-set counts differ from preregistration")
    positive_values = [
        float(row["labels"]["marginal_iou_gain"]) for row in output_rows
        if float(row["labels"]["marginal_iou_gain"]) > 0.0
    ]
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        dataset_path = staging / "marginal_gain_dataset.jsonl"
        dataset_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows
        ))
        summary = {
            "version": VERSION,
            "preregistration_status": "frozen_before_first_run",
            "scene_count": len(scenes),
            "baseline_geometry_count": baseline_count,
            "candidate_count": len(output_rows),
            "original_candidate_count": original_count,
            "refined_candidate_count": refined_count,
            "positive_marginal_gain_count": len(positive_values),
            "positive_marginal_gain_distinct_rounded_6_count": len(set(round(value, 6) for value in positive_values)),
            "fold_positive_marginal_gain_counts": {
                str(fold): int(positive_by_fold[fold]) for fold in range(5)
            },
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "target_contract": {
                "primary": "max_gt max(0, candidate_iou - native_track_existing_best_iou)",
                "quality_q": "mean indicator at IoU 0.50:0.05:0.95",
                "soft_quality": "clip((iou-0.50)/0.45,0,1)",
                "threshold_cross_control": "IoU 0.50:0.05:0.90",
            },
            "files": {"dataset": dataset_path.name},
            "hashes": {"dataset": _sha256(dataset_path)},
            "ground_truth_usage": "NCS-train100 label fields only",
            "feature_ground_truth_usage": "none",
            "candidate_deletion_count": 0,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "frozen_cache_write": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration),
                "scene_list_sha256": _sha256(args.scene_list),
                "fold_manifest_sha256": _sha256(args.fold_manifest),
                "unique_geometry_summary_sha256": _sha256(args.unique_geometry_root / "summary.json"),
                "stage_a_dataset_summary_sha256": _sha256(args.stage_a_dataset_root / "summary.json"),
                "stage_a_oof_summary_sha256": _sha256(args.stage_a_oof_root / "summary.json"),
                "stage_b_summary_sha256": _sha256(args.stage_b_root / "summary.json"),
                "stage_b_audit_summary_sha256": _sha256(args.stage_b_audit_root / "summary.json"),
                "champion_plan_summary_sha256": _sha256(args.champion_plan_root / "summary.json"),
                "stage_c_v2_summary_sha256": _sha256(args.stage_c_v2_root / "summary.json"),
                "stage_c_v2_audit_summary_sha256": _sha256(args.stage_c_v2_audit_root / "summary.json"),
            },
            "scene_summaries": scene_summaries,
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
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"
    ))
    parser.add_argument("--fold-manifest", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json"
    ))
    parser.add_argument("--ground-truth-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"
    ))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"
    ))
    parser.add_argument("--stage-a-dataset-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_a_quality_dataset_train100_20260822"
    ))
    parser.add_argument("--stage-a-oof-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_20260822"
    ))
    parser.add_argument("--stage-b-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_20260822"
    ))
    parser.add_argument("--stage-b-audit-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_audit_20260822"
    ))
    parser.add_argument("--champion-plan-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"
    ))
    parser.add_argument("--stage-c-v2-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_c_v2_refinement_oof_train100_20260822"
    ))
    parser.add_argument("--stage-c-v2-audit-root", type=Path, default=Path(
        "docs/diagnostics/ncs_fi1_stage_c_v2_refinement_oof_train100_audit_20260822"
    ))
    parser.add_argument("--preregistration", type=Path, default=Path(
        "docs/NCS_FI1_STAGE_D_PREREGISTRATION_20260822.md"
    ))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "candidate_count": result["candidate_count"],
        "positive_marginal_gain_count": result["positive_marginal_gain_count"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

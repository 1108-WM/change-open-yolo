#!/usr/bin/env python3
"""Independently audit stage-D-v2 rank-prefix geometry, labels, and features."""

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

from tools.audit_ncs_fi1_stage_d_marginal_gain_dataset import _labels, _overlap  # noqa: E402
from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    GeometryResolver, UNION_FEATURE_NAMES, _read_jsonl, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import _geometry_sha256  # noqa: E402
from tools.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import (  # noqa: E402
    EXPECTED_BASELINE_COUNT, EXPECTED_ORIGINAL_COUNT, EXPECTED_REFINED_COUNT, RELATION_TYPES,
)
from tools.build_ncs_fi1_stage_d_v2_rank_marginal_dataset_gt import FEATURE_NAMES  # noqa: E402
from tools.build_train_scene_candidate_quality_ledger import _load_gt  # noqa: E402
from tools.train_ncs_fi1_stage_c_v2_refinement_oof import QUALITY_FEATURE_NAMES  # noqa: E402


VERSION = "ncs_fi1_stage_d_v2_rank_marginal_dataset_audit_v1"


def _close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _ious(points: np.ndarray, gt: np.ndarray, gt_meta: dict[int, dict]) -> dict[int, float]:
    result = {}
    for encoded, meta in gt_meta.items():
        intersection = int(np.count_nonzero(gt[points] == encoded))
        if intersection:
            result[encoded] = intersection / max(1, len(points) + int(meta["point_count"]) - intersection)
    return result


def run(args: argparse.Namespace) -> dict:
    for name in (
        "dataset_root", "ground_truth_root", "unique_geometry_root", "stage_a_dataset_root",
        "stage_a_oof_root", "stage_b_root", "champion_plan_root", "stage_c_v2_root",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    summary = json.loads((args.dataset_root / "summary.json").read_text())
    dataset_path = args.dataset_root / summary["files"]["dataset"]
    rows = _read_jsonl(dataset_path)
    errors = Counter()
    if _sha256(dataset_path) != summary["hashes"]["dataset"]:
        errors["dataset_sha256_mismatch"] += 1
    if set(summary["feature_names"]) != set(FEATURE_NAMES):
        errors["summary_feature_schema_mismatch"] += 1

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    nodes_by_scene = defaultdict(list)
    for node in nodes:
        nodes_by_scene[str(node["scene_name"])].append(node)
    baseline_count = sum(str(node["canonical_candidate_source"]) in {"native", "track"} for node in nodes)
    if baseline_count != EXPECTED_BASELINE_COUNT:
        errors["baseline_geometry_count_mismatch"] += 1
    stage_a_dataset = {
        str(row["geometry_key"]): row for row in _read_jsonl(args.stage_a_dataset_root / "quality_dataset.jsonl")
    }
    stage_a = {
        str(row["geometry_key"]): row for row in _read_jsonl(args.stage_a_oof_root / "oof_quality_predictions.jsonl")
    }
    stage_b = {
        str(row["geometry_key"]): row for row in _read_jsonl(args.stage_b_root / "stage_b_rerank_plan.jsonl")
    }
    relation_counts = defaultdict(Counter)
    for relation in _read_jsonl(args.stage_b_root / "candidate_relations.jsonl"):
        relation_counts[str(relation["geometry_key_a"])][str(relation["relation_type"])] += 1
        relation_counts[str(relation["geometry_key_b"])][str(relation["relation_type"])] += 1
    originals = {
        (str(row["scene_name"]), int(row["candidate_id"])): row
        for row in _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    }
    stage_c = {
        (str(row["scene_name"]), int(row["union_candidate_id"])): row
        for row in _read_jsonl(args.stage_c_v2_root / "stage_c_v2_refined_union_plan.jsonl")
    }
    rows_by_scene = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["scene_name"])].append(row)

    resolver = GeometryResolver()
    positive_values = []
    positive_by_fold = Counter()
    original_count = refined_count = 0
    for scene, scene_rows in sorted(rows_by_scene.items()):
        gt, gt_meta = _load_gt(args.ground_truth_root / f"{scene}.txt", args.min_gt_points)
        baseline = []
        for node in nodes_by_scene[scene]:
            if str(node["canonical_candidate_source"]) in {"native", "track"}:
                points = resolver.points(node["canonical_geometry_locator"], int(node["point_count"]), len(gt))
                baseline.append((node, points, float(stage_b[str(node["geometry_key"])]["stage_b_score"])))
        for row in scene_rows:
            key = (scene, int(row["union_candidate_id"]))
            variant = str(row["candidate_variant"])
            original_count += int(variant == "original")
            refined_count += int(variant == "refined")
            c_row = stage_c[key]
            union = originals[key]
            original_key = str(c_row["original_union_geometry_key"])
            original_node = node_by_key[original_key]
            original_points = resolver.points(original_node["canonical_geometry_locator"], int(original_node["point_count"]), len(gt))
            points = resolver.points(row["geometry_locator_read_only"], int(row["point_count"]), len(gt))
            if _geometry_sha256(points) != str(row["geometry_sha256"]):
                errors["geometry_sha256_mismatch"] += 1
            expected_reference = (
                float(np.clip(c_row["quality_lower_confidence_bound"], 0.0, 1.0))
                if variant == "refined" else float(stage_b[original_key]["stage_b_score"])
            )
            if not _close(row["candidate_reference_score"], expected_reference):
                errors["candidate_reference_score_mismatch"] += 1
            prefix = [(node, base_points) for node, base_points, score in baseline if score >= expected_reference]
            if int(row["rank_prefix_geometry_count"]) != len(prefix):
                errors["rank_prefix_geometry_count_mismatch"] += 1
            if any(float(stage_b[str(node["geometry_key"])]["stage_b_score"]) < expected_reference for node, _ in prefix):
                errors["rank_prefix_score_order_violation"] += 1
            prefix_best = {encoded: 0.0 for encoded in gt_meta}
            for _, base_points in prefix:
                for encoded, iou in _ious(base_points, gt, gt_meta).items():
                    prefix_best[encoded] = max(prefix_best[encoded], iou)
            expected_labels = _labels(_ious(points, gt, gt_meta), prefix_best)
            expected_labels["rank_conditioned_marginal_iou_gain"] = expected_labels.pop("marginal_iou_gain")
            expected_labels["rank_conditioned_marginal_q_gain"] = expected_labels.pop("marginal_q_gain")
            expected_labels["rank_conditioned_marginal_soft_quality_gain"] = expected_labels.pop("marginal_soft_quality_gain")
            expected_labels["rank_prefix_best_iou_for_selected_gain_target"] = expected_labels.pop("existing_best_iou_for_selected_gain_target")
            for name, expected in expected_labels.items():
                observed = row["labels"].get(name)
                if isinstance(expected, bool):
                    if observed is not expected:
                        errors[f"label_{name}_mismatch"] += 1
                elif not _close(observed, expected):
                    errors[f"label_{name}_mismatch"] += 1
            gain = float(expected_labels["rank_conditioned_marginal_iou_gain"])
            if gain > 0.0:
                positive_values.append(gain)
                positive_by_fold[int(row["fold_index"])] += 1

            expected_overlap = _overlap(points, prefix, stage_a, stage_b)
            for name, expected in expected_overlap.items():
                renamed = "rank_prefix_" + name[len("baseline_"):]
                if not _close(row["features"].get(renamed), expected):
                    errors[f"feature_{renamed}_mismatch"] += 1
            atom_count = max(1, int(c_row["atom_count"]))
            quality_prediction = float(c_row["corrected_oof_temporary_refined_quality"]) if variant == "refined" else float(stage_a[original_key]["oof_unified_quality"])
            quality_lower = float(c_row["quality_lower_confidence_bound"]) if variant == "refined" else float(stage_a[original_key]["oof_unified_quality"])
            expected_base = {
                "variant_refined": float(variant == "refined"),
                "log1p_candidate_point_count": math.log1p(len(points)),
                "candidate_point_fraction_of_scene": len(points) / len(gt),
                "candidate_point_fraction_of_original_union": len(points) / len(original_points),
                "candidate_removed_point_fraction_from_original_union": 1.0 - len(points) / len(original_points),
                "stage_a_original_union_quality": float(stage_a[original_key]["oof_unified_quality"]),
                "stage_b_original_union_score": float(stage_b[original_key]["stage_b_score"]),
                "frozen_threshold_cross_probability": float(union["threshold_cross_probability"]),
                "frozen_union_base_quality": float(union["base_quality"]),
                "frozen_track_quality_q": float(union["track_quality_q"]),
                "frozen_native_group_median_quality_q": float(union["native_group_median_quality_q"]),
                "stage_c_candidate_quality_prediction": quality_prediction,
                "stage_c_candidate_quality_lower_bound": quality_lower,
                "geometry_quality_for_scoring": expected_reference,
                "stage_c_confident_removal_fraction": int(c_row["confident_removal_count"]) / atom_count if variant == "refined" else 0.0,
                "stage_c_connectivity_removal_fraction": int(c_row["connectivity_removal_count"]) / atom_count if variant == "refined" else 0.0,
                "stage_c_total_removal_fraction": (int(c_row["confident_removal_count"]) + int(c_row["connectivity_removal_count"])) / atom_count if variant == "refined" else 0.0,
                "rank_prefix_geometry_count": float(len(prefix)),
                "rank_prefix_fraction_of_native_track": len(prefix) / max(1, len(baseline)),
            }
            for name, expected in expected_base.items():
                if not _close(row["features"].get(name), expected):
                    errors[f"feature_{name}_mismatch"] += 1
            for name in UNION_FEATURE_NAMES:
                if not _close(row["features"].get(f"stage_a_union__{name}"), stage_a_dataset[original_key]["features"][name]):
                    errors["stage_a_union_feature_mismatch"] += 1
            for name in QUALITY_FEATURE_NAMES:
                if not _close(row["features"].get(f"stage_c__{name}"), c_row["quality_features"][name]):
                    errors["stage_c_feature_mismatch"] += 1
            counts = relation_counts.get(original_key, Counter())
            total = sum(counts.values())
            if not _close(row["features"].get("stage_b_relation_count"), total):
                errors["stage_b_relation_count_mismatch"] += 1
            for relation_type in RELATION_TYPES:
                if not _close(row["features"].get(f"stage_b_relation_count__{relation_type}"), counts[relation_type]):
                    errors["stage_b_relation_type_count_mismatch"] += 1
                if not _close(row["features"].get(f"stage_b_relation_fraction__{relation_type}"), counts[relation_type] / max(1, total)):
                    errors["stage_b_relation_type_fraction_mismatch"] += 1
            if set(row["features"]) != set(FEATURE_NAMES):
                errors["row_feature_schema_mismatch"] += 1
            for flag in ("candidate_deletion", "candidate_mutation", "geometry_mutation", "class_mutation", "frozen_cache_write", "ap_computed"):
                if row.get(flag) is not False:
                    errors[f"contract_{flag}_violation"] += 1
            if row.get("candidate_retained") is not True:
                errors["candidate_not_retained"] += 1

    checks = {
        "audit_error_count_zero": sum(errors.values()) == 0,
        "baseline_geometry_count_exact": baseline_count == EXPECTED_BASELINE_COUNT,
        "original_candidate_count_exact": original_count == EXPECTED_ORIGINAL_COUNT,
        "refined_candidate_count_exact": refined_count == EXPECTED_REFINED_COUNT,
        "positive_rank_marginal_gain_count_at_least_250": len(positive_values) >= 250,
        "positive_distinct_rounded_6_count_at_least_100": len(set(round(value, 6) for value in positive_values)) >= 100,
        **{f"fold_{fold}_has_positive_rank_marginal_gain": positive_by_fold[fold] > 0 for fold in range(5)},
        "candidate_deletion_count_zero": all(row.get("candidate_deletion") is False for row in rows),
        "geometry_mutation_count_zero": all(row.get("geometry_mutation") is False for row in rows),
        "ap_computed_false": summary.get("ap_computed") is False,
        "validation60_read_false": summary.get("validation60_read") is False,
        "val312_read_false": summary.get("val312_read") is False,
    }
    advancement = all(checks.values())
    audit = {
        "version": VERSION,
        "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        "counts": {
            "baseline_geometry": baseline_count,
            "candidate": len(rows),
            "original": original_count,
            "refined": refined_count,
            "positive_rank_marginal_gain": len(positive_values),
            "positive_distinct_rounded_6": len(set(round(value, 6) for value in positive_values)),
            "fold_positive": {str(fold): int(positive_by_fold[fold]) for fold in range(5)},
        },
        "advancement_gate": {"checks": checks, "advancement_authorized": advancement},
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "dataset_summary_sha256": _sha256(args.dataset_root / "summary.json"),
            "dataset_sha256": _sha256(dataset_path),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--stage-a-dataset-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_a_quality_dataset_train100_20260822"))
    parser.add_argument("--stage-a-oof-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_20260822"))
    parser.add_argument("--stage-b-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_20260822"))
    parser.add_argument("--champion-plan-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"))
    parser.add_argument("--stage-c-v2-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_c_v2_refinement_oof_train100_20260822"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    result = run(parser.parse_args())
    print(json.dumps({
        "audit_valid": result["audit_valid"],
        "error_count": result["error_count"],
        "advancement_authorized": result["advancement_gate"]["advancement_authorized"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

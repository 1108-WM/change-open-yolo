#!/usr/bin/env python3
"""Build stage-D-v2 rank-prefix-conditioned continuous marginal-gain labels."""

from __future__ import annotations

import argparse
import json
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
    GeometryResolver, _fold_by_scene, _read_jsonl, _read_scenes, _resolve, _sha256,
)
from tools.build_ncs_fi1_stage_c_member_dataset_gt import _geometry_sha256  # noqa: E402
from tools.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import (  # noqa: E402
    EXPECTED_BASELINE_COUNT, EXPECTED_ORIGINAL_COUNT, EXPECTED_REFINED_COUNT,
    FEATURE_NAMES as V1_FEATURE_NAMES, candidate_features as v1_candidate_features,
    iou_by_gt, label_candidate, overlap_features, relation_features,
)
from tools.build_train_scene_candidate_quality_ledger import _load_gt  # noqa: E402


VERSION = "ncs_fi1_stage_d_v2_rank_marginal_dataset_v1"


def _prefix_name(name: str) -> str:
    return "rank_prefix_" + name[len("baseline_"):] if name.startswith("baseline_") else name


FEATURE_NAMES = (
    *tuple(_prefix_name(name) for name in V1_FEATURE_NAMES),
    "rank_prefix_geometry_count",
    "rank_prefix_fraction_of_native_track",
)


def rank_prefix(
    baseline: list[tuple[dict, np.ndarray, float]], candidate_reference_score: float,
) -> list[tuple[dict, np.ndarray]]:
    return [
        (node, points) for node, points, score in baseline
        if float(score) >= float(candidate_reference_score)
    ]


def rank_labels(candidate_iou: dict[int, float], prefix_best: dict[int, float]) -> dict:
    labels = label_candidate(candidate_iou, prefix_best)
    labels["rank_conditioned_marginal_iou_gain"] = labels.pop("marginal_iou_gain")
    labels["rank_conditioned_marginal_q_gain"] = labels.pop("marginal_q_gain")
    labels["rank_conditioned_marginal_soft_quality_gain"] = labels.pop("marginal_soft_quality_gain")
    labels["rank_prefix_best_iou_for_selected_gain_target"] = labels.pop(
        "existing_best_iou_for_selected_gain_target"
    )
    return labels


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
        raise ValueError("stage D-v2 requires exactly NCS-train100")
    stage_b_audit = json.loads((args.stage_b_audit_root / "summary.json").read_text())
    stage_c_audit = json.loads((args.stage_c_v2_audit_root / "summary.json").read_text())
    if stage_b_audit.get("audit_valid") is not True:
        raise ValueError("stage-B audit is not valid")
    if stage_c_audit.get("audit_valid") is not True or not stage_c_audit.get("advancement_gate", {}).get("advancement_authorized"):
        raise ValueError("stage-C-v2 audit did not authorize downstream work")

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    nodes_by_scene = defaultdict(list)
    node_by_key = {}
    for node in nodes:
        nodes_by_scene[str(node["scene_name"])].append(node)
        node_by_key[str(node["geometry_key"])] = node
    baseline_count = sum(str(node["canonical_candidate_source"]) in {"native", "track"} for node in nodes)
    if baseline_count != EXPECTED_BASELINE_COUNT:
        raise ValueError("native+track baseline count differs from preregistration")

    stage_a_dataset_rows = _read_jsonl(args.stage_a_dataset_root / "quality_dataset.jsonl")
    stage_a_dataset = {str(row["geometry_key"]): row for row in stage_a_dataset_rows}
    stage_a_oof = {}
    for row in _read_jsonl(args.stage_a_oof_root / "oof_quality_predictions.jsonl"):
        key = str(row["geometry_key"])
        stage_a_oof[key] = {**row, "features": stage_a_dataset[key]["features"]}
    stage_b_rows = _read_jsonl(args.stage_b_root / "stage_b_rerank_plan.jsonl")
    stage_b = {str(row["geometry_key"]): row for row in stage_b_rows}
    relations_by_geometry = defaultdict(Counter)
    for row in _read_jsonl(args.stage_b_root / "candidate_relations.jsonl"):
        relations_by_geometry[str(row["geometry_key_a"])][str(row["relation_type"])] += 1
        relations_by_geometry[str(row["geometry_key_b"])][str(row["relation_type"])] += 1
    originals = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    original_by_key = {(str(row["scene_name"]), int(row["candidate_id"])): row for row in originals}
    stage_c_rows = _read_jsonl(args.stage_c_v2_root / "stage_c_v2_refined_union_plan.jsonl")
    stage_c = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in stage_c_rows}
    if len(originals) != EXPECTED_ORIGINAL_COUNT or len(stage_c_rows) != EXPECTED_ORIGINAL_COUNT:
        raise ValueError("original union coverage differs")
    if sum(bool(row["append_eligible_refined_union"]) for row in stage_c_rows) != EXPECTED_REFINED_COUNT:
        raise ValueError("refined union coverage differs")

    resolver = GeometryResolver()
    output = []
    positive_by_fold = Counter()
    scene_summaries = []
    for scene_index, scene in enumerate(scenes, 1):
        gt_path = args.ground_truth_root / f"{scene}.txt"
        gt, gt_meta = _load_gt(gt_path, args.min_gt_points)
        baseline = []
        for node in nodes_by_scene[scene]:
            if str(node["canonical_candidate_source"]) not in {"native", "track"}:
                continue
            points = resolver.points(node["canonical_geometry_locator"], int(node["point_count"]), len(gt))
            baseline.append((node, points, float(stage_b[str(node["geometry_key"])]["stage_b_score"])))
        local_original = local_refined = local_positive = 0
        for key in sorted(key for key in original_by_key if key[0] == scene):
            union = original_by_key[key]
            c_row = stage_c[key]
            original_key = str(c_row["original_union_geometry_key"])
            original_node = node_by_key[original_key]
            original_points = resolver.points(
                original_node["canonical_geometry_locator"], int(original_node["point_count"]), len(gt)
            )
            variants = [("original", original_points, original_node["canonical_geometry_locator"])]
            if c_row["append_eligible_refined_union"]:
                refined_path = args.stage_c_v2_root / str(c_row["refined_points_file"])
                with np.load(refined_path) as payload:
                    refined_points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
                variants.append((
                    "refined", refined_points,
                    {"kind": "point_indices_npz", "points_path": str(refined_path), "array_key": "point_indices"},
                ))
            relation = relation_features(original_key, relations_by_geometry)
            for variant, points, locator in variants:
                reference_score = (
                    float(np.clip(c_row["quality_lower_confidence_bound"], 0.0, 1.0))
                    if variant == "refined" else float(stage_b[original_key]["stage_b_score"])
                )
                prefix = rank_prefix(baseline, reference_score)
                prefix_best = {encoded: 0.0 for encoded in gt_meta}
                for _, prefix_points in prefix:
                    for encoded, iou in iou_by_gt(prefix_points, gt, gt_meta).items():
                        prefix_best[encoded] = max(prefix_best[encoded], iou)
                labels = rank_labels(iou_by_gt(points, gt, gt_meta), prefix_best)
                v1_overlap = overlap_features(points, prefix, stage_a_oof, stage_b)
                v1_features = v1_candidate_features(
                    variant=variant, points=points, original_points=original_points,
                    scene_point_count=len(gt), original_union=union,
                    original_stage_a_row=stage_a_oof[original_key],
                    original_stage_b_row=stage_b[original_key], stage_c_row=c_row,
                    overlap=v1_overlap, relation=relation,
                )
                features = {_prefix_name(name): float(value) for name, value in v1_features.items()}
                features["rank_prefix_geometry_count"] = float(len(prefix))
                features["rank_prefix_fraction_of_native_track"] = len(prefix) / max(1, len(baseline))
                if set(features) != set(FEATURE_NAMES):
                    raise ValueError("stage-D-v2 feature schema differs")
                candidate_key = f"{scene}:union:{key[1]:04d}:{variant}"
                output.append({
                    "candidate_key": candidate_key,
                    "scene_name": scene,
                    "fold_index": int(folds[scene]),
                    "union_candidate_id": int(key[1]),
                    "candidate_variant": variant,
                    "original_union_geometry_key": original_key,
                    "geometry_locator_read_only": locator,
                    "geometry_sha256": _geometry_sha256(points),
                    "point_count": int(len(points)),
                    "candidate_reference_score": reference_score,
                    "rank_prefix_geometry_count": len(prefix),
                    "features": features,
                    "labels": labels,
                    "ground_truth_usage": "NCS-train100 rank-conditioned label fields only",
                    "feature_ground_truth_usage": "none",
                    "candidate_retained": True,
                    "candidate_deletion": False,
                    "candidate_mutation": False,
                    "geometry_mutation": False,
                    "class_mutation": False,
                    "frozen_cache_write": False,
                    "ap_computed": False,
                })
                local_original += int(variant == "original")
                local_refined += int(variant == "refined")
                if labels["rank_conditioned_marginal_iou_gain"] > 0.0:
                    local_positive += 1
                    positive_by_fold[int(folds[scene])] += 1
        scene_summaries.append({
            "scene_name": scene,
            "fold_index": int(folds[scene]),
            "native_track_geometry_count": len(baseline),
            "original_candidate_count": local_original,
            "refined_candidate_count": local_refined,
            "positive_rank_marginal_gain_count": local_positive,
            "ground_truth_sha256": _sha256(gt_path),
        })
        print(
            f"[stage D-v2 dataset] {scene_index}/{len(scenes)} {scene}: "
            f"baseline={len(baseline)} original={local_original} refined={local_refined} positive={local_positive}",
            flush=True,
        )

    output.sort(key=lambda row: row["candidate_key"])
    if len({row["candidate_key"] for row in output}) != len(output):
        raise ValueError("duplicate stage-D-v2 candidate key")
    original_count = sum(row["candidate_variant"] == "original" for row in output)
    refined_count = sum(row["candidate_variant"] == "refined" for row in output)
    positive_values = [
        float(row["labels"]["rank_conditioned_marginal_iou_gain"]) for row in output
        if float(row["labels"]["rank_conditioned_marginal_iou_gain"]) > 0.0
    ]
    if original_count != EXPECTED_ORIGINAL_COUNT or refined_count != EXPECTED_REFINED_COUNT:
        raise ValueError("stage-D-v2 action-set count differs")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        dataset_path = staging / "rank_marginal_gain_dataset.jsonl"
        dataset_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output
        ))
        summary = {
            "version": VERSION,
            "preregistration_status": "frozen_before_first_run",
            "scene_count": len(scenes),
            "baseline_geometry_count": baseline_count,
            "candidate_count": len(output),
            "original_candidate_count": original_count,
            "refined_candidate_count": refined_count,
            "positive_rank_marginal_gain_count": len(positive_values),
            "positive_rank_marginal_gain_distinct_rounded_6_count": len(set(round(value, 6) for value in positive_values)),
            "fold_positive_rank_marginal_gain_counts": {str(fold): int(positive_by_fold[fold]) for fold in range(5)},
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "target_contract": "max_gt max(0,candidate_iou-rank_prefix_best_iou), prefix stage_b_score >= candidate_reference_score",
            "files": {"dataset": dataset_path.name},
            "hashes": {"dataset": _sha256(dataset_path)},
            "ground_truth_usage": "NCS-train100 rank-conditioned label fields only",
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
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--stage-a-dataset-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_a_quality_dataset_train100_20260822"))
    parser.add_argument("--stage-a-oof-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_a_unified_quality_oof_train100_20260822"))
    parser.add_argument("--stage-b-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_20260822"))
    parser.add_argument("--stage-b-audit-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_audit_20260822"))
    parser.add_argument("--champion-plan-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"))
    parser.add_argument("--stage-c-v2-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_c_v2_refinement_oof_train100_20260822"))
    parser.add_argument("--stage-c-v2-audit-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_c_v2_refinement_oof_train100_audit_20260822"))
    parser.add_argument("--preregistration", type=Path, default=Path("docs/NCS_FI1_STAGE_D_V2_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    result = run(parser.parse_args())
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "positive_rank_marginal_gain_count": result["positive_rank_marginal_gain_count"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

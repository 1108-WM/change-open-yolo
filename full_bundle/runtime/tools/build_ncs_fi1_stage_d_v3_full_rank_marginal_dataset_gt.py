#!/usr/bin/env python3
"""Build D-v3 full-rank-prefix continuous marginal-gain labels on a fixed 100-scene split."""

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
    candidate_features as v1_candidate_features, iou_by_gt, overlap_features, relation_features,
)
from tools.build_ncs_fi1_stage_d_v2_rank_marginal_dataset_gt import (  # noqa: E402
    FEATURE_NAMES as V2_FEATURE_NAMES, _prefix_name, rank_labels,
)
from tools.build_train_scene_candidate_quality_ledger import _load_gt  # noqa: E402

VERSION = "ncs_fi1_stage_d_v3_full_rank_marginal_dataset_v1"
EXTRA_FEATURE_NAMES = (
    "rank_prefix_native_count", "rank_prefix_track_count",
    "rank_prefix_original_union_count", "rank_prefix_refined_union_count",
    "rank_prefix_append_fraction",
)
FEATURE_NAMES = (*V2_FEATURE_NAMES, *EXTRA_FEATURE_NAMES)


def candidate_key(scene: str, union_candidate_id: int, variant: str) -> str:
    return f"{scene}:union:{union_candidate_id:04d}:{variant}"


def append_precedes(left: dict, right: dict) -> bool:
    """Frozen descending-score, ascending-key order among append candidates."""
    left_score = float(left["reference_score"])
    right_score = float(right["reference_score"])
    return left_score > right_score or (
        left_score == right_score and str(left["candidate_key"]) < str(right["candidate_key"])
    )


def full_rank_prefix(
    baseline: list[dict], append_candidates: list[dict], current: dict,
) -> list[dict]:
    score = float(current["reference_score"])
    return [entry for entry in baseline if float(entry["reference_score"]) >= score] + [
        entry for entry in append_candidates
        if entry["candidate_key"] != current["candidate_key"] and append_precedes(entry, current)
    ]


def _append_inventory(
    scene: str, scene_point_count: int, resolver: GeometryResolver, node_by_key: dict,
    originals: dict, stage_c: dict, stage_b: dict, stage_c_root: Path,
) -> list[dict]:
    result = []
    for key in sorted(key for key in originals if key[0] == scene):
        union = originals[key]
        c_row = stage_c[key]
        original_key = str(c_row["original_union_geometry_key"])
        original_node = node_by_key[original_key]
        original_points = resolver.points(
            original_node["canonical_geometry_locator"], int(original_node["point_count"]), scene_point_count
        )
        result.append({
            "candidate_key": candidate_key(scene, key[1], "original"),
            "union_candidate_id": key[1], "variant": "original", "points": original_points,
            "locator": original_node["canonical_geometry_locator"], "original_points": original_points,
            "original_key": original_key, "union": union, "stage_c": c_row,
            "reference_score": float(stage_b[original_key]["stage_b_score"]),
        })
        if c_row["append_eligible_refined_union"]:
            refined_path = stage_c_root / str(c_row["refined_points_file"])
            with np.load(refined_path) as payload:
                refined_points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
            result.append({
                "candidate_key": candidate_key(scene, key[1], "refined"),
                "union_candidate_id": key[1], "variant": "refined", "points": refined_points,
                "locator": {"kind": "point_indices_npz", "points_path": str(refined_path), "array_key": "point_indices"},
                "original_points": original_points, "original_key": original_key,
                "union": union, "stage_c": c_row,
                "reference_score": float(np.clip(c_row["quality_lower_confidence_bound"], 0.0, 1.0)),
            })
    return result


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "fold_manifest", "ground_truth_root", "unique_geometry_root",
        "stage_a_dataset_root", "stage_a_oof_root", "stage_b_root", "stage_b_audit_root",
        "champion_plan_root", "stage_c_v2_root", "stage_c_v2_audit_root",
        "preregistration", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    dataset_name = str(args.dataset_name)
    folds = _fold_by_scene(args.fold_manifest, scenes)
    if len(scenes) != 100:
        raise ValueError(f"stage D-v3 requires exactly 100 scenes from {dataset_name}")
    stage_b_audit = json.loads((args.stage_b_audit_root / "summary.json").read_text())
    stage_c_audit = json.loads((args.stage_c_v2_audit_root / "summary.json").read_text())
    if stage_b_audit.get("audit_valid") is not True:
        raise ValueError("stage-B audit is not valid")
    if stage_c_audit.get("audit_valid") is not True or not stage_c_audit.get("advancement_gate", {}).get("advancement_authorized"):
        raise ValueError("stage-C-v2 audit did not authorize downstream work")

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    unique_summary = json.loads((args.unique_geometry_root / "summary.json").read_text())
    nodes_by_scene = defaultdict(list)
    node_by_key = {}
    for node in nodes:
        nodes_by_scene[str(node["scene_name"])].append(node)
        node_by_key[str(node["geometry_key"])] = node
    baseline_count = sum(str(node["canonical_candidate_source"]) in {"native", "track"} for node in nodes)
    expected_baseline_count = sum(
        int(unique_summary.get("canonical_source_counts", {}).get(source, 0))
        for source in ("native", "track")
    )
    if baseline_count != expected_baseline_count:
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
    originals_list = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    originals = {(str(row["scene_name"]), int(row["candidate_id"])): row for row in originals_list}
    stage_c_rows = _read_jsonl(args.stage_c_v2_root / "stage_c_v2_refined_union_plan.jsonl")
    stage_c = {(str(row["scene_name"]), int(row["union_candidate_id"])): row for row in stage_c_rows}
    expected_original_count = len(originals)
    if len(stage_c_rows) != expected_original_count or set(stage_c) != set(originals):
        raise ValueError("original union coverage differs")
    expected_refined_count = sum(bool(row["append_eligible_refined_union"]) for row in stage_c_rows)

    resolver = GeometryResolver()
    output = []
    positive_by_fold = Counter()
    scene_summaries = []
    for scene_index, scene in enumerate(scenes, 1):
        gt_path = args.ground_truth_root / f"{scene}.txt"
        gt, gt_meta = _load_gt(gt_path, args.min_gt_points)
        baseline = []
        for node in nodes_by_scene[scene]:
            source = str(node["canonical_candidate_source"])
            if source not in {"native", "track"}:
                continue
            points = resolver.points(node["canonical_geometry_locator"], int(node["point_count"]), len(gt))
            baseline.append({
                "kind": source, "node": node, "points": points,
                "reference_score": float(stage_b[str(node["geometry_key"])]["stage_b_score"]),
            })
        append_candidates = _append_inventory(
            scene, len(gt), resolver, node_by_key, originals, stage_c, stage_b, args.stage_c_v2_root
        )
        # Virtual no-GT scores let the unchanged overlap feature implementation consume append prefix items.
        local_stage_a = dict(stage_a_oof)
        local_stage_b = dict(stage_b)
        for entry in append_candidates:
            virtual_key = str(entry["candidate_key"])
            c_row = entry["stage_c"]
            quality = (
                float(c_row["corrected_oof_temporary_refined_quality"])
                if entry["variant"] == "refined"
                else float(stage_a_oof[entry["original_key"]]["oof_unified_quality"])
            )
            local_stage_a[virtual_key] = {"oof_unified_quality": quality}
            local_stage_b[virtual_key] = {"stage_b_score": float(entry["reference_score"])}
            entry["kind"] = f"{entry['variant']}_union"
            entry["node"] = {"geometry_key": virtual_key}
        local_positive = 0
        for entry in sorted(append_candidates, key=lambda item: item["candidate_key"]):
            prefix_entries = full_rank_prefix(baseline, append_candidates, entry)
            prefix = [(item["node"], item["points"]) for item in prefix_entries]
            prefix_best = {encoded: 0.0 for encoded in gt_meta}
            for item in prefix_entries:
                for encoded, iou in iou_by_gt(item["points"], gt, gt_meta).items():
                    prefix_best[encoded] = max(prefix_best[encoded], iou)
            labels = rank_labels(iou_by_gt(entry["points"], gt, gt_meta), prefix_best)
            overlap = overlap_features(entry["points"], prefix, local_stage_a, local_stage_b)
            v1_features = v1_candidate_features(
                variant=entry["variant"], points=entry["points"], original_points=entry["original_points"],
                scene_point_count=len(gt), original_union=entry["union"],
                original_stage_a_row=stage_a_oof[entry["original_key"]],
                original_stage_b_row=stage_b[entry["original_key"]], stage_c_row=entry["stage_c"],
                overlap=overlap, relation=relation_features(entry["original_key"], relations_by_geometry),
            )
            features = {_prefix_name(name): float(value) for name, value in v1_features.items()}
            counts = Counter(item["kind"] for item in prefix_entries)
            features.update({
                "rank_prefix_geometry_count": float(len(prefix_entries)),
                "rank_prefix_fraction_of_native_track": (counts["native"] + counts["track"]) / max(1, len(baseline)),
                "rank_prefix_native_count": float(counts["native"]),
                "rank_prefix_track_count": float(counts["track"]),
                "rank_prefix_original_union_count": float(counts["original_union"]),
                "rank_prefix_refined_union_count": float(counts["refined_union"]),
                "rank_prefix_append_fraction": (counts["original_union"] + counts["refined_union"]) / max(1, len(prefix_entries)),
            })
            if set(features) != set(FEATURE_NAMES):
                raise ValueError("stage-D-v3 feature schema differs")
            output.append({
                "candidate_key": entry["candidate_key"], "scene_name": scene,
                "fold_index": int(folds[scene]), "union_candidate_id": int(entry["union_candidate_id"]),
                "candidate_variant": entry["variant"], "original_union_geometry_key": entry["original_key"],
                "geometry_locator_read_only": entry["locator"],
                "geometry_sha256": _geometry_sha256(entry["points"]), "point_count": int(len(entry["points"])),
                "candidate_reference_score": float(entry["reference_score"]),
                "rank_prefix_geometry_count": len(prefix_entries),
                "rank_prefix_source_counts": {
                    "native": counts["native"], "track": counts["track"],
                    "original_union": counts["original_union"], "refined_union": counts["refined_union"],
                },
                "features": features, "labels": labels,
                "ground_truth_usage": f"{dataset_name} full-rank-prefix label fields only",
                "feature_ground_truth_usage": "none", "candidate_retained": True,
                "candidate_deletion": False, "candidate_mutation": False, "geometry_mutation": False,
                "class_mutation": False, "frozen_cache_write": False, "ap_computed": False,
            })
            if labels["rank_conditioned_marginal_iou_gain"] > 0.0:
                local_positive += 1
                positive_by_fold[int(folds[scene])] += 1
        scene_summaries.append({
            "scene_name": scene, "fold_index": int(folds[scene]),
            "native_track_geometry_count": len(baseline),
            "original_candidate_count": sum(item["variant"] == "original" for item in append_candidates),
            "refined_candidate_count": sum(item["variant"] == "refined" for item in append_candidates),
            "positive_full_rank_marginal_gain_count": local_positive,
            "ground_truth_sha256": _sha256(gt_path),
        })
        print(f"[stage D-v3 dataset] {scene_index}/{len(scenes)} {scene}: baseline={len(baseline)} append={len(append_candidates)} positive={local_positive}", flush=True)

    output.sort(key=lambda row: row["candidate_key"])
    if len({row["candidate_key"] for row in output}) != len(output):
        raise ValueError("duplicate stage-D-v3 candidate key")
    original_count = sum(row["candidate_variant"] == "original" for row in output)
    refined_count = sum(row["candidate_variant"] == "refined" for row in output)
    positive_values = [float(row["labels"]["rank_conditioned_marginal_iou_gain"]) for row in output if float(row["labels"]["rank_conditioned_marginal_iou_gain"]) > 0.0]
    if original_count != expected_original_count or refined_count != expected_refined_count:
        raise ValueError("stage-D-v3 action-set count differs")
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        dataset_path = staging / "full_rank_marginal_gain_dataset.jsonl"
        dataset_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output))
        summary = {
            "version": VERSION, "preregistration_status": "frozen_before_first_run",
            "dataset_name": dataset_name,
            "scene_count": len(scenes), "baseline_geometry_count": baseline_count,
            "expected_baseline_geometry_count": expected_baseline_count,
            "candidate_count": len(output), "original_candidate_count": original_count,
            "refined_candidate_count": refined_count,
            "positive_rank_marginal_gain_count": len(positive_values),
            "positive_rank_marginal_gain_distinct_rounded_6_count": len(set(round(value, 6) for value in positive_values)),
            "fold_positive_rank_marginal_gain_counts": {str(fold): int(positive_by_fold[fold]) for fold in range(5)},
            "feature_names": list(FEATURE_NAMES), "feature_count": len(FEATURE_NAMES),
            "target_contract": "max_gt max(0,candidate_iou-full_rank_prefix_best_iou); baseline ties first; append ties by candidate_key",
            "files": {"dataset": dataset_path.name}, "hashes": {"dataset": _sha256(dataset_path)},
            "ground_truth_usage": f"{dataset_name} full-rank-prefix label fields only", "feature_ground_truth_usage": "none",
            "candidate_deletion_count": 0, "candidate_mutation": False, "geometry_mutation": False,
            "class_mutation": False, "frozen_cache_write": False, "ap_computed": False,
            "validation60_read": False, "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration), "scene_list_sha256": _sha256(args.scene_list),
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
    parser.add_argument("--preregistration", type=Path, default=Path("docs/NCS_FI1_STAGE_D_V3_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-gt-points", type=int, default=100)
    parser.add_argument("--dataset-name", default="NCS-train100")
    result = run(parser.parse_args())
    print(json.dumps({"candidate_count": result["candidate_count"], "positive_rank_marginal_gain_count": result["positive_rank_marginal_gain_count"], "ap_computed": result["ap_computed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

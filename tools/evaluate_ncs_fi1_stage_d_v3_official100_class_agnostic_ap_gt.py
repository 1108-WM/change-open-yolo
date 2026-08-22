#!/usr/bin/env python3
"""Run the single frozen D-v3 control/challenger class-agnostic AP evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import GeometryResolver, _read_jsonl, _read_scenes, _resolve, _sha256  # noqa: E402
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS, _class_agnostic_gt_ids, _configure_scannet200_instance_eval,
    instance_eval,
)

VERSION = "ncs_fi1_stage_d_v3_official100_class_agnostic_ap_v1"


def _metrics(ap_scores: np.ndarray) -> dict[str, float]:
    averages = instance_eval.compute_averages(ap_scores)
    chair = averages["classes"]["chair"]
    return {"ap": float(chair["ap"]), "ap50": float(chair["ap50%"]), "ap25": float(chair["ap25%"]) }


def _prediction(masks: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "pred_masks": masks,
        "pred_scores": np.asarray(scores, dtype=np.float64),
        "pred_classes": np.full(len(scores), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }


def run(args: argparse.Namespace) -> dict:
    if not args.allow_gt_evaluation:
        raise PermissionError("this frozen official100 AP evaluation requires --allow-gt-evaluation")
    for name in ("scene_list", "ground_truth_root", "plan_root", "plan_audit_root", "preregistration", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    dataset_name = str(args.dataset_name)
    if len(scenes) != 100:
        raise ValueError(f"D-v3 AP evaluation requires exactly 100 scenes from {dataset_name}")
    plan_summary = json.loads((args.plan_root / "summary.json").read_text())
    plan_audit = json.loads((args.plan_audit_root / "summary.json").read_text())
    if plan_audit.get("audit_valid") is not True or plan_audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("complete OOF plan audit did not authorize AP evaluation")
    plan_path = args.plan_root / plan_summary["files"]["plan"]
    rows = _read_jsonl(plan_path)
    rows_by_scene = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["scene_name"])].append(row)
    if set(rows_by_scene) != set(scenes):
        raise ValueError("complete plan scene coverage differs from official100")

    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    resolver = GeometryResolver()
    control_matches = {}
    challenger_matches = {}
    fold_by_scene = {}

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for scene_index, scene in enumerate(scenes, 1):
            gt_file = args.ground_truth_root / f"{scene}.txt"
            point_count = len(original_load_ids(str(gt_file)))
            scene_rows = sorted(rows_by_scene[scene], key=lambda row: str(row["plan_key"]))
            folds = {int(row["fold_index"]) for row in scene_rows}
            if len(folds) != 1:
                raise ValueError(f"{scene}: inconsistent fold assignment")
            fold_by_scene[scene] = folds.pop()
            masks = np.zeros((point_count, len(scene_rows)), dtype=bool)
            control_indices = []
            control_scores = []
            challenger_scores = []
            for column, row in enumerate(scene_rows):
                points = resolver.points(row["geometry_locator_read_only"], int(row["point_count"]), point_count)
                masks[points, column] = True
                challenger_scores.append(float(row["challenger_score"]))
                if row["control_score"] is not None:
                    control_indices.append(column)
                    control_scores.append(float(row["control_score"]))
            control_prediction = _prediction(masks[:, control_indices], np.asarray(control_scores))
            challenger_prediction = _prediction(masks, np.asarray(challenger_scores))
            control_gt, control_pred = instance_eval.assign_instances_for_scan(control_prediction, str(gt_file))
            challenger_gt, challenger_pred = instance_eval.assign_instances_for_scan(challenger_prediction, str(gt_file))
            match_key = os.path.abspath(str(gt_file))
            control_matches[match_key] = {"gt": control_gt, "pred": control_pred}
            challenger_matches[match_key] = {"gt": challenger_gt, "pred": challenger_pred}
            print(f"[D-v3 class-agnostic AP] {scene_index}/{len(scenes)} {scene}: control={len(control_scores)} challenger={len(challenger_scores)}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    control_ap, _, _, _, _, _ = instance_eval.evaluate_matches(control_matches)
    challenger_ap, _, _, _, _, _ = instance_eval.evaluate_matches(challenger_matches)
    control_metrics = _metrics(control_ap)
    challenger_metrics = _metrics(challenger_ap)
    fold_records = {}
    fold_arrays = {}
    for fold in range(5):
        selected_keys = {
            os.path.abspath(str(args.ground_truth_root / f"{scene}.txt"))
            for scene in scenes if fold_by_scene[scene] == fold
        }
        local_control, _, _, _, _, _ = instance_eval.evaluate_matches({key: value for key, value in control_matches.items() if key in selected_keys})
        local_challenger, _, _, _, _, _ = instance_eval.evaluate_matches({key: value for key, value in challenger_matches.items() if key in selected_keys})
        control_fold_metrics = _metrics(local_control)
        challenger_fold_metrics = _metrics(local_challenger)
        fold_records[str(fold)] = {
            "scene_count": len(selected_keys), "control": control_fold_metrics,
            "challenger": challenger_fold_metrics,
            "delta": {name: challenger_fold_metrics[name] - control_fold_metrics[name] for name in ("ap", "ap50", "ap25")},
        }
        fold_arrays[f"control_fold_{fold}"] = local_control
        fold_arrays[f"challenger_fold_{fold}"] = local_challenger
    delta = {name: challenger_metrics[name] - control_metrics[name] for name in ("ap", "ap50", "ap25")}

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        arrays_path = staging / "raw_ap_arrays.npz"
        np.savez_compressed(arrays_path, control_overall=control_ap, challenger_overall=challenger_ap, **fold_arrays)
        summary = {
            "version": VERSION, "dataset_name": dataset_name,
            "evaluation_scope": f"{dataset_name} class-agnostic instance AP",
            "single_fixed_challenger": True, "threshold_or_weight_scan_count": 0,
            "scene_count": len(scenes), "control_candidate_count": int(plan_summary["control_candidate_count"]),
            "challenger_candidate_count": int(plan_summary["challenger_candidate_count"]),
            "control": control_metrics, "challenger": challenger_metrics, "delta": delta,
            "folds": fold_records,
            "primary_class_agnostic_ap_improved": delta["ap"] > 0.0,
            "all_reported_metrics_improved": all(value > 0.0 for value in delta.values()),
            "files": {"raw_ap_arrays": arrays_path.name}, "hashes": {"raw_ap_arrays": _sha256(arrays_path)},
            "candidate_deletion_count": 0, "geometry_mutation": False, "class_mutation": False,
            "validation60_read": False, "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration), "scene_list_sha256": _sha256(args.scene_list),
                "plan_summary_sha256": _sha256(args.plan_root / "summary.json"), "plan_sha256": _sha256(plan_path),
                "plan_audit_summary_sha256": _sha256(args.plan_audit_root / "summary.json"),
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
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"))
    parser.add_argument("--ground-truth-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"))
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--plan-audit-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=Path("docs/NCS_FI1_STAGE_D_V3_PREREGISTRATION_20260822.md"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--dataset-name", default="NCS-train100")
    result = run(parser.parse_args())
    print(json.dumps({"control": result["control"], "challenger": result["challenger"], "delta": result["delta"], "primary_class_agnostic_ap_improved": result["primary_class_agnostic_ap_improved"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

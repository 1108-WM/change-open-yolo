#!/usr/bin/env python3
"""Run the one-shot Legacy-control/FI1-D-v3 class-agnostic AP comparison."""

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

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import GeometryResolver, _read_jsonl, _resolve, _sha256  # noqa: E402
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS, _class_agnostic_gt_ids, _configure_scannet200_instance_eval,
    instance_eval,
)


VERSION = "fi1_d_v3_frozen_class_agnostic_ap_v1"


def _scenes(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    return rows


def _metrics(scores: np.ndarray) -> dict[str, float]:
    chair = instance_eval.compute_averages(scores)["classes"]["chair"]
    return {"ap": float(chair["ap"]), "ap50": float(chair["ap50%"]), "ap25": float(chair["ap25%"])}


def _prediction(masks: np.ndarray, scores: list[float]) -> dict:
    return {
        "pred_masks": masks, "pred_scores": np.asarray(scores, dtype=np.float64),
        "pred_classes": np.full(len(scores), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }


def run(args: argparse.Namespace) -> dict:
    if not args.allow_gt_evaluation:
        raise PermissionError("the frozen final AP run requires --allow-gt-evaluation")
    for name in ("scene_list", "ground_truth_root", "inference_root", "audit_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    audit = json.loads((args.audit_root / "summary.json").read_text())
    if audit.get("audit_valid") is not True or audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        raise ValueError("inference-plan audit did not authorize AP evaluation")
    plan_root = args.inference_root / "complete_plan"
    plan_summary = json.loads((plan_root / "summary.json").read_text())
    plan_path = plan_root / plan_summary["files"]["plan"]
    rows_by_scene = defaultdict(list)
    for row in _read_jsonl(plan_path):
        rows_by_scene[str(row["scene_name"])].append(row)
    if set(rows_by_scene) != set(scenes):
        raise ValueError("complete plan scene coverage differs from the frozen scene list")

    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    resolver = GeometryResolver()
    control_matches, challenger_matches = {}, {}

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for scene_index, scene in enumerate(scenes, 1):
            gt_file = args.ground_truth_root / f"{scene}.txt"
            point_count = len(original_load_ids(str(gt_file)))
            rows = sorted(rows_by_scene[scene], key=lambda row: str(row["plan_key"]))
            masks = np.zeros((point_count, len(rows)), dtype=bool)
            control_columns, control_scores, challenger_scores = [], [], []
            for column, row in enumerate(rows):
                points = resolver.points(row["geometry_locator_read_only"], int(row["point_count"]), point_count)
                masks[points, column] = True
                challenger_scores.append(float(row["challenger_score"]))
                if row["control_score"] is not None:
                    control_columns.append(column); control_scores.append(float(row["control_score"]))
            control_gt, control_pred = instance_eval.assign_instances_for_scan(
                _prediction(masks[:, control_columns], control_scores), str(gt_file)
            )
            challenger_gt, challenger_pred = instance_eval.assign_instances_for_scan(
                _prediction(masks, challenger_scores), str(gt_file)
            )
            key = os.path.abspath(str(gt_file))
            control_matches[key] = {"gt": control_gt, "pred": control_pred}
            challenger_matches[key] = {"gt": challenger_gt, "pred": challenger_pred}
            print(f"[FI1-D-v3 final AP] {scene_index}/{len(scenes)} {scene}: control={len(control_scores)} challenger={len(challenger_scores)}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    control_array, _, _, _, _, _ = instance_eval.evaluate_matches(control_matches)
    challenger_array, _, _, _, _, _ = instance_eval.evaluate_matches(challenger_matches)
    control = _metrics(control_array); challenger = _metrics(challenger_array)
    delta = {name: challenger[name] - control[name] for name in ("ap", "ap50", "ap25")}
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        arrays_path = staging / "raw_ap_arrays.npz"
        np.savez_compressed(arrays_path, control=control_array, challenger=challenger_array)
        summary = {
            "version": VERSION, "dataset_name": args.dataset_name,
            "evaluation_scope": f"{args.dataset_name} class-agnostic instance AP",
            "single_fixed_challenger": True, "threshold_or_weight_scan_count": 0,
            "scene_count": len(scenes), "control": control, "challenger": challenger, "delta": delta,
            "primary_class_agnostic_ap_improved": delta["ap"] > 0.0,
            "files": {"raw_ap_arrays": arrays_path.name}, "hashes": {"raw_ap_arrays": _sha256(arrays_path)},
            "input_provenance": {"scene_list_sha256": _sha256(args.scene_list), "plan_sha256": _sha256(plan_path), "audit_sha256": _sha256(args.audit_root / "summary.json")},
        }
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ScanNet200-val312")
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({"control": result["control"], "challenger": result["challenger"], "delta": result["delta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

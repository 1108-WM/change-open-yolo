#!/usr/bin/env python3
"""Evaluate the single frozen append-only pair-union OOF policy on official100."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import TRACK_SOURCE, read_scene_list  # noqa: E402
from tools.build_candidate_pair_union_oof_plan import (  # noqa: E402
    POLICY,
    VERSION as PLAN_VERSION,
)
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    _resolve,
    _scene_inputs,
    _scene_records,
    _sha256,
    configure_track_score_context,
)
from tools.construct_c1c_global_feasible_ap_oracle_gt import (  # noqa: E402
    DIAGNOSTIC,
    _record_from_matches,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    _load_track_points,
    _prediction,
)
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_SPLIT_SHA256,
    _metrics_from_records,
)
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


VERSION = "official100_pair_union_oof_ap_v1"


def _triplet(metrics: dict) -> dict:
    return {
        "ap": float(metrics["official_ap"]),
        "ap50": float(metrics["threshold_metrics"]["50"]["ap"]),
        "ap25": float(metrics["threshold_metrics"]["25"]["ap"]),
    }


def _appended_scene_records(scene: str, cache: dict, plan_rows: list[dict], args) -> dict:
    cache_root = args.records_root / scene / "native_cache"
    native_masks = np.load(cache_root / f"{scene}_pred_masks.npy", mmap_mode="r")
    native_scores = np.asarray(
        np.load(cache_root / f"{scene}_pred_scores.npy", mmap_mode="r"), dtype=np.float64
    )
    native_ids = sorted(cache["all_native_representatives"])
    track_ids = sorted(cache["all_track_ids"])
    track_masks = np.zeros((native_masks.shape[0], len(track_ids)), dtype=bool)
    track_scores = np.zeros(len(track_ids), dtype=np.float64)
    for column, track_id in enumerate(track_ids):
        points, _ = _load_track_points(cache["track_by_id"][track_id], native_masks.shape[0])
        track_masks[points, column] = True
        track_scores[column] = float(
            args.track_oof_scores[(scene, TRACK_SOURCE, track_id)]["q"]
        )

    append_masks = np.zeros((native_masks.shape[0], len(plan_rows)), dtype=bool)
    append_scores = np.zeros(len(plan_rows), dtype=np.float64)
    for column, row in enumerate(sorted(plan_rows, key=lambda item: int(item["candidate_id"]))):
        path = Path(row["points_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if len(points) != int(row["point_count"]):
            raise ValueError(f"{scene}: append candidate point count mismatch")
        append_masks[points, column] = True
        append_scores[column] = float(row["new_score"])

    combined_masks = np.concatenate([
        np.asarray(native_masks[:, native_ids], dtype=bool), track_masks, append_masks,
    ], axis=1)
    combined_scores = np.concatenate([
        native_scores[native_ids], track_scores, append_scores,
    ])
    gt, pred = instance_eval.assign_instances_for_scan(
        _prediction(combined_masks, combined_scores, len(combined_scores)),
        str(args.gt_dir / f"{scene}.txt"),
    )
    return {
        str(int(round(threshold * 100))): _record_from_matches(gt, pred, threshold)
        for threshold in DIAGNOSTIC
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("AP evaluation requires frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    plan_summary_path = args.plan_root / "summary.json"
    plan_summary = json.loads(plan_summary_path.read_text())
    if plan_summary.get("version") != PLAN_VERSION or plan_summary.get("policy") != POLICY:
        raise ValueError("unexpected frozen pair-union plan")
    if plan_summary.get("ap_computed") is not False:
        raise ValueError("plan must precede AP evaluation")
    all_plan_rows = [
        json.loads(line)
        for line in (args.plan_root / "pair_union_append_plan.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_scene = {scene: [] for scene in scenes}
    for row in all_plan_rows:
        by_scene[str(row["scene_name"])].append(row)
    if len(all_plan_rows) != int(plan_summary["unique_materialized_candidate_count"]):
        raise ValueError("plan candidate count differs from summary")

    configure_track_score_context(args, scenes)
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(
        original_load_ids(path)
    )
    baseline_records, appended_records = {}, {}
    try:
        for index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            appended_records[scene] = _appended_scene_records(
                scene, cache, by_scene[scene], args
            )
            print(
                f"[single frozen pair-union AP] {index}/100 {scene}: "
                f"append={len(by_scene[scene])}",
                flush=True,
            )
            del cache
            gc.collect()
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    baseline_metrics = _metrics_from_records(baseline_records)
    selected_metrics = _metrics_from_records(appended_records)
    baseline = _triplet(baseline_metrics)
    selected = _triplet(selected_metrics)
    delta = {key: selected[key] - baseline[key] for key in baseline}
    folds = []
    for fold in manifest["folds"]:
        validation = set(fold["validation_scenes"])
        fold_base = _triplet(_metrics_from_records({
            scene: baseline_records[scene] for scene in validation
        }))
        fold_selected = _triplet(_metrics_from_records({
            scene: appended_records[scene] for scene in validation
        }))
        folds.append({
            "fold_index": int(fold["fold_index"]),
            "frozen_coexist": fold_base,
            POLICY: fold_selected,
            "delta": {
                key: fold_selected[key] - fold_base[key] for key in fold_base
            },
        })
    summary = {
        "version": VERSION,
        "policy": POLICY,
        "scene_count": len(scenes),
        "appended_candidate_count": len(all_plan_rows),
        "metrics": {
            "frozen_coexist": baseline,
            POLICY: selected,
        },
        "delta_vs_frozen_coexist": delta,
        "folds": folds,
        "main_ap_positive_fold_count": sum(
            fold["delta"]["ap"] > 0.0 for fold in folds
        ),
        "evaluation_contract": {
            "evaluated_challenger_count": 1,
            "append_only": True,
            "existing_candidate_count_modified": False,
            "existing_candidate_geometry_modified": False,
            "existing_candidate_score_modified": False,
            "existing_candidate_class_modified": False,
            "weight_threshold_exponent_scan": False,
        },
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official100 AP evaluation only after frozen no-GT append plan",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "plan_summary_sha256": _sha256(plan_summary_path),
            "plan_rows_sha256": _sha256(args.plan_root / "pair_union_append_plan.jsonl"),
            "quality_oof_predictions_sha256": _sha256(args.oof_predictions),
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
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--track-score-mode", choices=("oof_quality",), default="oof_quality")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("must pass --allow-gt-evaluation")
    for name in (
        "scene_list", "split_manifest", "records_root", "relation_feature_ledger_root",
        "gt_dir", "oof_predictions", "plan_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "metrics": summary["metrics"],
        "delta_vs_frozen_coexist": summary["delta_vs_frozen_coexist"],
        "main_ap_positive_fold_count": summary["main_ap_positive_fold_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

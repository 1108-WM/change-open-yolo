#!/usr/bin/env python3
"""Run the single frozen official100 OOF AP evaluation for marginal state."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    _resolve,
    _scene_inputs,
    _scene_records,
    _sha256,
    configure_track_score_context,
)
from tools.build_candidate_track_marginal_state_oof_plan import (  # noqa: E402
    POLICY,
    VERSION as PLAN_VERSION,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    _set_match_scores,
)
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_SPLIT_SHA256,
    _metrics_from_records,
)
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


VERSION = "official100_track_marginal_state_oof_ap_v1"


def _triplet(metrics: dict) -> dict:
    return {
        "ap": float(metrics["official_ap"]),
        "ap50": float(metrics["threshold_metrics"]["50"]["ap"]),
        "ap25": float(metrics["threshold_metrics"]["25"]["ap"]),
    }


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("AP evaluation requires frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    plan_summary_path = args.plan_root / "summary.json"
    plan_summary = json.loads(plan_summary_path.read_text())
    if plan_summary.get("version") != PLAN_VERSION or plan_summary.get("policy") != POLICY:
        raise ValueError("unexpected frozen marginal-state plan")
    if plan_summary.get("ap_evaluated") is not False:
        raise ValueError("plan must be generated before AP evaluation")
    plan_rows = [
        json.loads(line)
        for line in (args.plan_root / "track_score_plan.jsonl").read_text().splitlines()
        if line.strip()
    ]
    plan = {}
    for row in plan_rows:
        key = (str(row["scene_name"]), int(row["candidate_id"]))
        if key in plan:
            raise ValueError(f"duplicate score-plan track: {key}")
        plan[key] = row
    if len(plan) != int(plan_summary["controlled_track_count"]):
        raise ValueError("score-plan row count differs from summary")

    configure_track_score_context(args, scenes)
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(
        original_load_ids(path)
    )
    baseline_records, policy_records = {}, {}
    accepted_controlled_count = 0
    try:
        for index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            expected_controlled = {
                int(track_id) for track_id in cache["controlled_track_ids"]
            }
            observed_controlled = {
                track_id for scene_name, track_id in plan if scene_name == scene
            }
            if observed_controlled != expected_controlled:
                raise ValueError(f"{scene}: plan controlled-track coverage mismatch")
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            uuid_to_key = {uuid: key for key, uuid in cache["uuid_by_candidate"].items()}
            scores_by_uuid = {
                row["uuid"]: float(row["confidence"])
                for row in cache["pred"]["chair"]
            }
            for uuid, candidate_key in uuid_to_key.items():
                if candidate_key[0] != "track":
                    continue
                track_id = int(candidate_key[1])
                row = plan.get((scene, track_id))
                if row is None:
                    continue
                original = float(scores_by_uuid[uuid])
                # The evaluator materializes prediction confidences as float32,
                # while the no-GT plan preserves the OOF quality value as
                # float64.  Match the repository's existing score-cache
                # contract rather than requiring impossible bit identity
                # across those two representations.
                if abs(original - float(row["original_score"])) > 1e-7:
                    raise ValueError(f"{scene}/{track_id}: plan original score mismatch")
                scores_by_uuid[uuid] = float(row["new_score"])
                accepted_controlled_count += 1
            _set_match_scores(
                {scene: {"gt": cache["gt"], "pred": cache["pred"]}}, scores_by_uuid
            )
            policy_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            print(f"[single frozen marginal-state AP] {index}/100 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
        gc.collect()

    baseline_metrics = _metrics_from_records(baseline_records)
    policy_metrics = _metrics_from_records(policy_records)
    baseline = _triplet(baseline_metrics)
    selected = _triplet(policy_metrics)
    delta = {key: selected[key] - baseline[key] for key in baseline}
    folds = []
    for fold in manifest["folds"]:
        validation = set(fold["validation_scenes"])
        base_fold = _triplet(_metrics_from_records({
            scene: baseline_records[scene] for scene in validation
        }))
        selected_fold = _triplet(_metrics_from_records({
            scene: policy_records[scene] for scene in validation
        }))
        folds.append({
            "fold_index": int(fold["fold_index"]),
            "frozen_coexist": base_fold,
            POLICY: selected_fold,
            "delta": {
                key: selected_fold[key] - base_fold[key] for key in base_fold
            },
        })
    summary = {
        "version": VERSION,
        "policy": POLICY,
        "scene_count": len(scenes),
        "controlled_track_count": len(plan),
        "evaluator_accepted_controlled_track_count": accepted_controlled_count,
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
            "weight_threshold_exponent_scan": False,
            "native_scores_bitwise_unchanged": True,
            "uncontrolled_track_scores_unchanged": True,
            "candidate_count_modified": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
        },
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official100 AP evaluation only after frozen no-GT plan",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "plan_summary_sha256": _sha256(plan_summary_path),
            "plan_rows_sha256": _sha256(args.plan_root / "track_score_plan.jsonl"),
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

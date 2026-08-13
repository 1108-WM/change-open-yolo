#!/usr/bin/env python3
"""Evaluate the single frozen champion plus pair-union combined OOF policy."""
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
from tools.build_candidate_champion_pair_union_combined_oof_plan import (  # noqa: E402
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
from tools.train_candidate_component_union_list_head_oof import (  # noqa: E402
    PRIMARY_POLICY as CHAMPION_POLICY,
)
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


VERSION = "official100_champion_pair_union_combined_oof_ap_v1"


def _triplet(metrics: dict) -> dict:
    return {
        "ap": float(metrics["official_ap"]),
        "ap50": float(metrics["threshold_metrics"]["50"]["ap"]),
        "ap25": float(metrics["threshold_metrics"]["25"]["ap"]),
    }


def _combined_scene_records(
    scene: str, cache: dict, overrides: dict[int, float],
    union_rows: list[dict], args,
) -> dict:
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
        track_scores[column] = float(overrides.get(
            track_id, args.track_oof_scores[(scene, TRACK_SOURCE, track_id)]["q"]
        ))

    append_masks = np.zeros((native_masks.shape[0], len(union_rows)), dtype=bool)
    append_scores = np.zeros(len(union_rows), dtype=np.float64)
    for column, row in enumerate(sorted(union_rows, key=lambda item: int(item["candidate_id"]))):
        path = Path(row["points_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if len(points) != int(row["point_count"]):
            raise ValueError(f"{scene}: union plan point count mismatch")
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
        raise ValueError("combined AP requires frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    plan_summary_path = args.combined_plan_root / "summary.json"
    plan_summary = json.loads(plan_summary_path.read_text())
    if plan_summary.get("version") != PLAN_VERSION or plan_summary.get("policy") != POLICY:
        raise ValueError("unexpected combined frozen plan")
    if plan_summary.get("ap_computed") is not False:
        raise ValueError("combined plan must precede AP evaluation")
    champion_summary = json.loads(args.champion_summary.read_text())
    champion_metrics = {
        key: float(value) for key, value in champion_summary["policy_metrics"][
            CHAMPION_POLICY
        ].items()
    }
    champion_baseline = {
        key: float(value) for key, value in champion_summary["policy_metrics"][
            "frozen_coexist"
        ].items()
    }
    champion_fold_delta = {
        int(row["fold_index"]): {
            key: float(value) for key, value in row["delta"].items()
        }
        for row in champion_summary["primary_fold_deltas"]
    }

    overrides_by_scene = {scene: {} for scene in scenes}
    for line in (
        args.combined_plan_root / "champion_track_score_overrides.jsonl"
    ).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        overrides_by_scene[str(row["scene_name"])][int(row["candidate_id"])] = float(
            row["new_score"]
        )
    unions_by_scene = {scene: [] for scene in scenes}
    for line in (
        args.combined_plan_root / "pair_union_append_candidates.jsonl"
    ).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            unions_by_scene[str(row["scene_name"])].append(row)

    configure_track_score_context(args, scenes)
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(
        original_load_ids(path)
    )
    baseline_records, combined_records = {}, {}
    try:
        for index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            combined_records[scene] = _combined_scene_records(
                scene, cache, overrides_by_scene[scene], unions_by_scene[scene], args
            )
            print(
                f"[single frozen combined AP] {index}/100 {scene}: "
                f"track_overrides={len(overrides_by_scene[scene])}, "
                f"union_append={len(unions_by_scene[scene])}",
                flush=True,
            )
            del cache
            gc.collect()
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    baseline = _triplet(_metrics_from_records(baseline_records))
    combined = _triplet(_metrics_from_records(combined_records))
    for key in baseline:
        if abs(baseline[key] - champion_baseline[key]) > 1e-12:
            raise ValueError(f"frozen baseline differs from champion protocol: {key}")
    delta_vs_frozen = {key: combined[key] - baseline[key] for key in baseline}
    delta_vs_champion = {key: combined[key] - champion_metrics[key] for key in combined}
    folds = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        validation = set(fold["validation_scenes"])
        fold_base = _triplet(_metrics_from_records({
            scene: baseline_records[scene] for scene in validation
        }))
        fold_combined = _triplet(_metrics_from_records({
            scene: combined_records[scene] for scene in validation
        }))
        combined_delta = {
            key: fold_combined[key] - fold_base[key] for key in fold_base
        }
        folds.append({
            "fold_index": fold_index,
            "frozen_coexist": fold_base,
            POLICY: fold_combined,
            "combined_delta_vs_frozen": combined_delta,
            "delta_combined_minus_champion": {
                key: combined_delta[key] - champion_fold_delta[fold_index][key]
                for key in combined_delta
            },
        })
    summary = {
        "version": VERSION,
        "policy": POLICY,
        "scene_count": len(scenes),
        "metrics": {
            "frozen_coexist": baseline,
            CHAMPION_POLICY: champion_metrics,
            POLICY: combined,
        },
        "delta_vs_frozen_coexist": delta_vs_frozen,
        "delta_vs_champion": delta_vs_champion,
        "folds": folds,
        "main_ap_positive_fold_count_vs_frozen": sum(
            fold["combined_delta_vs_frozen"]["ap"] > 0.0 for fold in folds
        ),
        "main_ap_positive_fold_count_vs_champion": sum(
            fold["delta_combined_minus_champion"]["ap"] > 0.0 for fold in folds
        ),
        "evaluation_contract": {
            "evaluated_challenger_count": 1,
            "champion_formula_unchanged": True,
            "pair_union_formula_unchanged": True,
            "threshold_weight_exponent_scan": False,
            "native_modified": False,
            "uncontrolled_track_modified": False,
            "pair_union_append_only": True,
        },
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official100 AP evaluation only after frozen combined no-GT plan",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "combined_plan_summary_sha256": _sha256(plan_summary_path),
            "champion_summary_sha256": _sha256(args.champion_summary),
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
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--champion-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("must pass --allow-gt-evaluation")
    for name in (
        "scene_list", "split_manifest", "records_root", "relation_feature_ledger_root",
        "gt_dir", "oof_predictions", "combined_plan_root", "champion_summary",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "metrics": summary["metrics"],
        "delta_vs_champion": summary["delta_vs_champion"],
        "main_ap_positive_fold_count_vs_champion": summary[
            "main_ap_positive_fold_count_vs_champion"
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build candidate-level AP utility labels for demoting one related track.

For every track that belongs to a native--track relation component, this
offline-only diagnostic keeps every candidate and every mask fixed, changes
only that track's confidence to ``0.0``, and measures the resulting global
class-agnostic AP delta against frozen coexistence.  Other scenes, components,
tracks, native scores, classes, geometry, and GT associations remain fixed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list
from tools.build_train_candidate_component_action_utility_ledger import (
    EXPECTED_SCENE_LIST_SHA256,
    EPS,
    _fixed_records_without_scene,
    _metric_delta,
    _resolve,
    _scene_inputs,
    _scene_records,
    _sha256,
    _write_jsonl,
    configure_track_score_context,
    global_metrics,
    relation_label_context,
)
from tools.construct_c1c_global_feasible_ap_oracle_gt import DIAGNOSTIC, OFFICIAL
from tools.diagnose_d2b_native_track_ranking_oracle_gt import _append, _empty_record
from tools.diagnose_gvc_class_agnostic_ap import (
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)


DEMOTED_TRACK_SCORE = 0.0


def marginal_decision(delta_official_ap: float) -> str:
    """Name the effect of suppressing the track, not the quality of the track."""
    if delta_official_ap > EPS:
        return "suppress_beneficial"
    if delta_official_ap < -EPS:
        return "suppress_harmful"
    return "neutral"


def _records_with_one_track_score(
    cache: dict, track_id: int, demoted_score: float = DEMOTED_TRACK_SCORE,
) -> tuple[dict, dict]:
    """Reuse fixed GT association and replace one accepted track confidence."""
    key = ("track", int(track_id))
    target_uuid = cache["uuid_by_candidate"].get(key)
    if target_uuid is None:
        # The evaluator ignores candidates below its fixed minimum region size.
        records = _scene_records(
            cache, cache["all_native_representatives"], cache["all_track_ids"]
        )
        return records, {
            "evaluator_candidate_present": False,
            "top_level_prediction_confidence_change_count": 0,
            "matched_prediction_confidence_change_count": 0,
            "original_track_score": None,
            "demoted_track_score": None,
        }

    original_rows = [
        row for row in cache["pred"]["chair"] if row["uuid"] == target_uuid
    ]
    if len(original_rows) != 1:
        raise ValueError(f"track {track_id}: evaluator prediction UUID is not unique")
    original_score = float(original_rows[0]["confidence"])
    if demoted_score > original_score + EPS:
        raise ValueError(
            f"track {track_id}: fixed demotion score {demoted_score} exceeds "
            f"original score {original_score}"
        )

    top_level_changes = 0
    pred_rows = []
    for row in cache["pred"]["chair"]:
        copied = dict(row)
        if copied["uuid"] == target_uuid:
            copied["confidence"] = float(demoted_score)
            top_level_changes += 1
        pred_rows.append(copied)

    matched_changes = 0
    gt_rows = []
    for gt_row in cache["gt"]["chair"]:
        matched = []
        for match in gt_row["matched_pred"]:
            copied = dict(match)
            if copied["uuid"] == target_uuid:
                copied["confidence"] = float(demoted_score)
                matched_changes += 1
            matched.append(copied)
        gt_rows.append(dict(gt_row, matched_pred=matched))
    if top_level_changes != 1:
        raise ValueError(f"track {track_id}: expected exactly one confidence change")

    from tools.construct_c1c_global_feasible_ap_oracle_gt import _record_from_matches

    records = {
        str(int(round(threshold * 100))): _record_from_matches(
            {"chair": gt_rows}, {"chair": pred_rows}, threshold
        )
        for threshold in DIAGNOSTIC
    }
    return records, {
        "evaluator_candidate_present": True,
        "top_level_prediction_confidence_change_count": top_level_changes,
        "matched_prediction_confidence_change_count": matched_changes,
        "original_track_score": original_score,
        "demoted_track_score": float(demoted_score),
    }


def _controlled_tracks(cache: dict) -> list[tuple[int, int]]:
    result = []
    seen = set()
    for component in cache["components"]:
        component_id = int(component["relation_component_id"])
        for track_id in sorted(int(value) for value in component["track_ids"]):
            if track_id in seen:
                raise ValueError(f"track {track_id}: appears in multiple components")
            seen.add(track_id)
            result.append((component_id, track_id))
    if seen != cache["controlled_track_ids"]:
        raise ValueError("controlled track set differs from relation components")
    return result


def run(args: argparse.Namespace) -> dict:
    all_scenes = read_scene_list(args.scene_list)
    if len(all_scenes) != args.expected_scene_count:
        raise ValueError(
            f"fixed protocol requires {args.expected_scene_count} scenes, got {len(all_scenes)}"
        )
    scene_list_sha = _sha256(args.scene_list)
    if args.expected_scene_count == 100 and scene_list_sha != EXPECTED_SCENE_LIST_SHA256:
        raise ValueError(f"official100 scene-list SHA-256 mismatch: {scene_list_sha}")
    scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    if not scenes:
        raise ValueError("smoke scene count must be positive")

    relation_summary_path = args.relation_feature_ledger_root / "summary.json"
    relation_summary = json.loads(relation_summary_path.read_text())
    if int(relation_summary["scene_count"]) != args.expected_scene_count:
        raise ValueError("relation ledger scene count differs from fixed protocol")
    if relation_summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("relation feature ledger must not use GT features")
    score_context = configure_track_score_context(args, all_scenes)

    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(
        original_load_ids(path)
    )
    caches, baseline_records = {}, {}
    try:
        for index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            caches[scene] = cache
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            print(f"[fixed coexist] {index}/{len(scenes)} {scene}", flush=True)

        all_baseline = {
            str(int(round(value * 100))): _empty_record() for value in DIAGNOSTIC
        }
        for records in baseline_records.values():
            for tag, values in records.items():
                _append(all_baseline[tag], values)
        empty_trial = {
            tag: ([], [], 0, False, False) for tag in all_baseline
        }
        baseline_metrics = global_metrics(all_baseline, empty_trial)

        staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        decision_counts: Counter[str] = Counter()
        scene_summaries = []
        track_count = 0
        accepted_track_count = 0
        try:
            for scene_index, scene in enumerate(scenes, start=1):
                cache = caches[scene]
                fixed = _fixed_records_without_scene(baseline_records, scene)
                component_by_id = {
                    int(row["relation_component_id"]): row for row in cache["components"]
                }
                rows = []
                for component_id, track_id in _controlled_tracks(cache):
                    trial_records, confidence_audit = _records_with_one_track_score(
                        cache, track_id, args.demoted_track_score
                    )
                    metrics = global_metrics(fixed, trial_records)
                    delta = _metric_delta(metrics, baseline_metrics)
                    decision = marginal_decision(delta["official_ap"])
                    decision_counts[decision] += 1
                    accepted_track_count += int(
                        confidence_audit["evaluator_candidate_present"]
                    )
                    context = relation_label_context(
                        cache["relations_by_component"][component_id]
                    )
                    row = {
                        "scene_name": scene,
                        "relation_component_id": component_id,
                        "track_id": track_id,
                        "component_track_count": int(
                            component_by_id[component_id]["track_count"]
                        ),
                        "component_relation_label_context": context,
                        "action_name": f"demote_one_track:{track_id}",
                        "action_kind": "demote_one_track",
                        "confidence_audit": confidence_audit,
                        "candidate_counts_before_action": {
                            "native_representative": len(cache["all_native_representatives"]),
                            "track": len(cache["all_track_ids"]),
                        },
                        "candidate_counts_after_action": {
                            "native_representative": len(cache["all_native_representatives"]),
                            "track": len(cache["all_track_ids"]),
                        },
                        "labels": {
                            "ground_truth_usage": "official_train_offline_marginal_ap_utility_only",
                            "global_metrics_after_action": metrics,
                            "delta_vs_fixed_coexist": delta,
                            "delta_official_ap": float(delta["official_ap"]),
                            "delta_ap50": float(delta["threshold_metrics"]["50"]["ap"]),
                            "delta_ap25": float(delta["threshold_metrics"]["25"]["ap"]),
                            "decision": decision,
                        },
                        "contracts": {
                            "all_candidates_retained": True,
                            "no_non_target_confidence_changed": True,
                            "target_track_confidence_changed_if_evaluator_present": True,
                            "evaluator_ignored_target_below_min_region": not confidence_audit[
                                "evaluator_candidate_present"
                            ],
                            "other_scenes_fixed_to_coexist": True,
                            "other_components_fixed_to_coexist": True,
                            "native_masks_scores_classes_bitwise_unchanged": True,
                            "candidate_geometry_modified": False,
                            "candidate_files_modified": False,
                            "global_ap_non_additive_warning": True,
                        },
                    }
                    rows.append(row)
                    track_count += 1

                scene_root = staging / scene
                scene_root.mkdir()
                _write_jsonl(scene_root / "track_marginal_harm_utilities.jsonl", rows)
                scene_summary = {
                    "scene_name": scene,
                    "controlled_track_count": len(rows),
                    "evaluator_accepted_controlled_track_count": sum(
                        row["confidence_audit"]["evaluator_candidate_present"] for row in rows
                    ),
                    "decision_counts": dict(sorted(Counter(
                        row["labels"]["decision"] for row in rows
                    ).items())),
                    "native_representative_count": len(cache["all_native_representatives"]),
                    "all_track_count": len(cache["all_track_ids"]),
                    "input_sha256": cache["input_sha256"],
                }
                (scene_root / "summary.json").write_text(
                    json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                )
                scene_summaries.append(scene_summary)
                print(
                    f"[track marginal utility] {scene_index}/{len(scenes)} {scene}: "
                    f"tracks={len(rows)}",
                    flush=True,
                )

            payload = {
                "version": "official100_track_marginal_harm_ledger_v1",
                "diagnostic_type": "official train GT-only fixed-coexist demote-one-track global AP utility ledger",
                "scene_count": len(scenes),
                "expected_full_scene_count": args.expected_scene_count,
                "is_smoke_subset": len(scenes) != args.expected_scene_count,
                "controlled_track_count": track_count,
                "evaluator_accepted_controlled_track_count": accepted_track_count,
                "decision_counts": dict(sorted(decision_counts.items())),
                "fixed_coexist_global_metrics": baseline_metrics,
                "score_context": score_context,
                "demoted_track_score": float(args.demoted_track_score),
                "official_ap_thresholds": list(OFFICIAL),
                "diagnostic_iou_thresholds": list(DIAGNOSTIC),
                "ground_truth_usage": "official_train_offline_marginal_ap_utility_labels_only",
                "feature_ground_truth_usage": "none; no model features are generated by this ledger",
                "global_ap_non_additive_warning": "each label is an isolated one-track counterfactual",
                "selection_head_trained": False,
                "inference_plan_generated": False,
                "candidate_files_modified": False,
                "safety60_evaluated": False,
                "input_provenance": {
                    "scene_list_path": str(args.scene_list.resolve()),
                    "scene_list_sha256": scene_list_sha,
                    "relation_feature_ledger_summary_path": str(relation_summary_path.resolve()),
                    "relation_feature_ledger_summary_sha256": _sha256(relation_summary_path),
                },
                "scene_summaries": scene_summaries,
                "params": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != "track_oof_scores"
                },
            }
            (staging / "summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            os.replace(staging, args.output_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
        caches.clear()
        gc.collect()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--track-score-mode", choices=("original", "oof_quality"), default="original")
    parser.add_argument("--oof-predictions", type=Path)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--demoted-track-score", type=float, default=DEMOTED_TRACK_SCORE)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "records_root", "relation_feature_ledger_root", "gt_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.oof_predictions is not None:
        args.oof_predictions = _resolve(args.oof_predictions)
    if args.expected_scene_count < 1:
        raise ValueError("--expected-scene-count must be positive")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be positive")
    if args.demoted_track_score != DEMOTED_TRACK_SCORE:
        raise ValueError("demoted track score is preregistered and fixed to 0.0")
    if not args.scene_list.is_file():
        raise FileNotFoundError(args.scene_list)
    for path in (args.records_root, args.relation_feature_ledger_root, args.gt_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output directory exists and is non-empty: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        key: summary[key] for key in (
            "scene_count", "controlled_track_count",
            "evaluator_accepted_controlled_track_count", "decision_counts",
        )
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

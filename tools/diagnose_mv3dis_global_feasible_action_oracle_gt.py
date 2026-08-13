#!/usr/bin/env python3
"""GT-only global-feasible oracle over the frozen boundary action ledger.

Actions are selected by their frozen, local fixed-target IoU utility.  The
selected actions are then applied only in memory; if their combined removals
would empty a proposal, the responsible actions deterministically fall back to
the ledger's ``unknown=no-op`` state.  No candidate, score, native prediction,
or inference rule is written or changed.

The reported AP keeps native and track scores frozen.  It is therefore a
global-feasible *geometry* oracle diagnostic, not an exact combinatorial AP
maximum and not the later fixed-mask scoring oracle.
"""

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS,
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    _merge_scan_matches,
    instance_eval,
)
from diagnose_mv3dis_boundary_action_oracle_gt import (  # noqa: E402
    UNKNOWN_NOOP_ACTION,
    _load_superpoints,
    _load_tracks,
    _read_scenes,
    _resolve,
    _write_jsonl,
)


DECISION_CONSTRAINT = (
    "GT selects only this offline frozen-action geometry oracle. It must not "
    "be converted into an inference selector, score, threshold, class, or "
    "materialized proposal."
)


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _utility(action):
    return float(sum(
        float(outcome["fixed_gt_iou_delta"])
        for outcome in action["proposal_outcomes"]
        if outcome["fixed_gt_available"]
    ))


def choose_local_oracle_action(action_results):
    """Maximize frozen local ΔIoU; ties deliberately retain unknown/no-op."""
    def key(action):
        target = action["target_owner_proposal_id"]
        return (
            _utility(action),
            action["action_kind"] == "unknown_noop",
            -(int(target) if target is not None else -1),
        )

    selected = max(action_results, key=key)
    return selected, _utility(selected)


def _apply_action(geometry, action):
    for outcome in action["proposal_outcomes"]:
        proposal_id = int(outcome["proposal_id"])
        final = set(geometry[proposal_id])
        final.difference_update(map(int, outcome["removed_superpoint_ids"]))
        final.update(map(int, outcome["added_superpoint_ids"]))
        geometry[proposal_id] = final


def select_global_feasible_actions(source_geometry, boundary_rows):
    """Apply local-oracle choices and atomically fall back only for empties."""
    selections = []
    for row in sorted(boundary_rows, key=lambda item: int(item["superpoint_id"])):
        action, utility = choose_local_oracle_action(row["action_results"])
        no_op = next(
            item for item in row["action_results"]
            if item["action_name"] == UNKNOWN_NOOP_ACTION
        )
        selections.append({
            "scene_name": row["scene_name"],
            "superpoint_id": int(row["superpoint_id"]),
            "frozen_planned_owner_proposal_id": int(
                row["frozen_planned_owner_proposal_id"]
            ),
            "selected_action_name": action["action_name"],
            "selected_action_kind": action["action_kind"],
            "selected_target_owner_proposal_id": action["target_owner_proposal_id"],
            "selected_local_fixed_target_iou_utility": utility,
            "selected_action": action,
            "unknown_noop_action": no_op,
            "global_status": "selected",
            "fallback_reasons": [],
        })
    geometry = {proposal_id: set(ids) for proposal_id, ids in source_geometry.items()}
    for selection in selections:
        _apply_action(geometry, selection["selected_action"])

    while True:
        empty = sorted(proposal_id for proposal_id, ids in geometry.items() if not ids)
        if not empty:
            break
        fallback_indices = set()
        for proposal_id in empty:
            candidates = []
            for index, selection in enumerate(selections):
                action = selection["selected_action"]
                if action["action_kind"] == "unknown_noop":
                    continue
                removed = any(
                    int(outcome["proposal_id"]) == proposal_id
                    and outcome["removed_superpoint_ids"]
                    for outcome in action["proposal_outcomes"]
                )
                if removed:
                    candidates.append((
                        float(selection["selected_local_fixed_target_iou_utility"]),
                        int(selection["superpoint_id"]),
                        index,
                    ))
            if not candidates:
                raise ValueError(f"cannot recover empty proposal {proposal_id}")
            fallback_indices.add(min(candidates)[2])
        for index in sorted(fallback_indices):
            selection = selections[index]
            no_op = selection["unknown_noop_action"]
            selection["selected_action"] = no_op
            selection["selected_action_name"] = no_op["action_name"]
            selection["selected_action_kind"] = no_op["action_kind"]
            selection["selected_target_owner_proposal_id"] = None
            selection["global_status"] = "fallback_would_empty"
            selection["fallback_reasons"].append("would_empty_proposal")
        geometry = {proposal_id: set(ids) for proposal_id, ids in source_geometry.items()}
        for selection in selections:
            _apply_action(geometry, selection["selected_action"])
    return geometry, selections


def _prepare_selections(source_geometry, boundary_rows):
    return select_global_feasible_actions(source_geometry, boundary_rows)


def _track_prediction(tracks, geometry, points_by_superpoint, point_count):
    proposal_ids = sorted(tracks)
    masks = np.zeros((point_count, len(proposal_ids)), dtype=bool)
    scores = np.zeros(len(proposal_ids), dtype=np.float32)
    for column, proposal_id in enumerate(proposal_ids):
        chunks = [points_by_superpoint[item] for item in sorted(geometry[proposal_id])]
        if not chunks:
            raise ValueError(f"oracle geometry contains empty proposal {proposal_id}")
        masks[np.concatenate(chunks), column] = True
        scores[column] = max(0.0, float(tracks[proposal_id]["track_score"]))
    return {
        "pred_masks": masks,
        "pred_scores": scores,
        "pred_classes": np.full(len(proposal_ids), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }


def _scene_ap_records(gt_by_label, pred_by_label, overlap_th):
    """The evaluator's chair-only matching, reduced to sufficient AP records."""
    label_name = "chair"
    gt_instances = [
        gt for gt in gt_by_label[label_name]
        if gt["instance_id"] >= 1000 and gt["vert_count"] >= 100
    ]
    pred_instances = pred_by_label[label_name]
    pred_visited = {pred["uuid"]: False for pred in pred_instances}
    true = np.ones(len(gt_instances), dtype=np.float64)
    score = np.full(len(gt_instances), -float("inf"), dtype=np.float64)
    matched = np.zeros(len(gt_instances), dtype=bool)
    hard_false_negatives = 0
    for index, gt in enumerate(gt_instances):
        found_match = False
        for pred in gt["matched_pred"]:
            if pred_visited[pred["uuid"]]:
                continue
            overlap = float(pred["intersection"]) / (
                gt["vert_count"] + pred["vert_count"] - pred["intersection"]
            )
            if overlap > overlap_th:
                confidence = pred["confidence"]
                if matched[index]:
                    maximum = max(score[index], confidence)
                    minimum = min(score[index], confidence)
                    score[index] = maximum
                    true = np.append(true, 0.0)
                    score = np.append(score, minimum)
                    matched = np.append(matched, True)
                else:
                    found_match = True
                    matched[index] = True
                    score[index] = confidence
                    pred_visited[pred["uuid"]] = True
        if not found_match:
            hard_false_negatives += 1
    true = true[matched]
    score = score[matched]
    for pred in pred_instances:
        found_gt = False
        for gt in pred["matched_gt"]:
            overlap = float(gt["intersection"]) / (
                gt["vert_count"] + pred["vert_count"] - gt["intersection"]
            )
            if overlap > overlap_th:
                found_gt = True
                break
        if not found_gt:
            ignored = pred["void_intersection"]
            for gt in pred["matched_gt"]:
                if gt["instance_id"] < 1000 or gt["vert_count"] < 100:
                    ignored += gt["intersection"]
            if float(ignored) / pred["vert_count"] <= overlap_th:
                true = np.append(true, 0.0)
                score = np.append(score, pred["confidence"])
    return true, score, hard_false_negatives, bool(gt_instances), bool(pred_instances)


def _average_precision(true_parts, score_parts, hard_false_negatives, has_gt, has_pred):
    if not has_gt:
        return float("nan")
    if not has_pred:
        return 0.0
    y_true = np.concatenate(true_parts) if true_parts else np.empty(0)
    y_score = np.concatenate(score_parts) if score_parts else np.empty(0)
    if not len(y_score):
        return 0.0
    order = np.argsort(y_score)
    scores = y_score[order]
    truth = y_true[order]
    cumulative = np.cumsum(truth)
    _, unique_indices = np.unique(scores, return_index=True)
    num_points = len(unique_indices) + 1
    num_true_examples = cumulative[-1] if len(cumulative) else 0.0
    cumulative = np.append(cumulative, 0)
    precision = np.zeros(num_points)
    recall = np.zeros(num_points)
    for index, score_index in enumerate(unique_indices):
        true_positive = num_true_examples - cumulative[score_index - 1]
        false_positive = len(scores) - score_index - true_positive
        false_negative = cumulative[score_index - 1] + hard_false_negatives
        precision[index] = float(true_positive) / (true_positive + false_positive)
        recall[index] = float(true_positive) / (true_positive + false_negative)
    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_conv = np.append(recall[0], recall)
    recall_for_conv = np.append(recall_for_conv, 0.0)
    return float(np.dot(precision, np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")))


def _detach_unused_match_cycles(gt_by_label, pred_by_label):
    """Drop only recursive references not read by ``evaluate_matches``.

    A copied GT entry inside ``pred['matched_gt']`` inherits a growing
    ``matched_pred`` list.  The evaluator never reads that nested list when
    processing false positives, but retaining it makes all scene matches grow
    quadratically.  The authoritative top-level GT ``matched_pred`` entries
    remain intact for greedy matching.
    """
    for rows in pred_by_label.values():
        for pred in rows:
            for gt in pred["matched_gt"]:
                gt.pop("matched_pred", None)


def evaluate_native_plus_geometry(
    name,
    cache_root,
    scenes,
    rows_by_scene,
    track_root,
    processed_scene_root,
    gt_dir,
    output_dir,
    use_oracle_geometry,
    memory_light,
    record_output=None,
):
    _configure_scannet200_instance_eval()
    matches = {}
    record_parts = {
        float(overlap): {"true": [], "score": [], "fn": 0, "has_gt": False, "has_pred": False}
        for overlap in instance_eval.opt["overlaps"]
    }
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for index, scene_name in enumerate(scenes, start=1):
            tracks, source, oracle, points_by_superpoint, _, _ = build_scene(
                scene_name,
                rows_by_scene[scene_name],
                track_root,
                processed_scene_root,
            )
            geometry = oracle if use_oracle_geometry else source
            prefix = cache_root / f"{scene_name}_pred_"
            native_masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
            native_scores = np.load(str(prefix) + "scores.npy", mmap_mode="r")
            native_prediction = {
                "pred_masks": native_masks,
                "pred_scores": np.asarray(native_scores, dtype=np.float32),
                "pred_classes": np.full(
                    len(native_scores), UNIFIED_PREDICTED_CLASS, dtype=np.int64
                ),
            }
            track_prediction = _track_prediction(
                tracks, geometry, points_by_superpoint, native_masks.shape[0]
            )
            gt_file = str(gt_dir / f"{scene_name}.txt")
            native_gt, native_pred = instance_eval.assign_instances_for_scan(
                native_prediction, gt_file
            )
            track_gt, track_pred = instance_eval.assign_instances_for_scan(
                track_prediction, gt_file
            )
            merged_gt, merged_pred = _merge_scan_matches(
                native_gt, native_pred, track_gt, track_pred
            )
            if memory_light:
                for overlap, records in record_parts.items():
                    true, score, fn, has_gt, has_pred = _scene_ap_records(
                        merged_gt, merged_pred, overlap
                    )
                    records["true"].append(true)
                    records["score"].append(score)
                    records["fn"] += fn
                    records["has_gt"] = records["has_gt"] or has_gt
                    records["has_pred"] = records["has_pred"] or has_pred
            else:
                _detach_unused_match_cycles(merged_gt, merged_pred)
                matches[os.path.abspath(gt_file)] = {"gt": merged_gt, "pred": merged_pred}
            del native_prediction, track_prediction, native_masks, native_gt, native_pred
            del track_gt, track_pred
            del tracks, source, oracle, geometry, points_by_superpoint
            gc.collect()
            print(f"[evaluate {name}] {index}/{len(scenes)} {scene_name}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    if memory_light:
        if record_output is not None:
            payload = {}
            for overlap, records in record_parts.items():
                key = str(overlap).replace(".", "_")
                payload[f"true_{key}"] = (
                    np.concatenate(records["true"]) if records["true"] else np.empty(0)
                )
                payload[f"score_{key}"] = (
                    np.concatenate(records["score"]) if records["score"] else np.empty(0)
                )
                payload[f"fn_{key}"] = np.asarray([records["fn"]], dtype=np.int64)
                payload[f"has_gt_{key}"] = np.asarray([records["has_gt"]], dtype=bool)
                payload[f"has_pred_{key}"] = np.asarray([records["has_pred"]], dtype=bool)
            np.savez_compressed(record_output, **payload)
        aps = {
            overlap: _average_precision(
                records["true"], records["score"], records["fn"],
                records["has_gt"], records["has_pred"],
            )
            for overlap, records in record_parts.items()
        }
        result = {
            "ap": float(np.mean([value for overlap, value in aps.items() if overlap != 0.25])),
            "ap50": float(aps[0.5]),
            "ap25": float(aps[0.25]),
        }
        (output_dir / f"{name}.csv").write_text(
            "metric,value\n" + "\n".join(f"{key},{value}" for key, value in result.items()) + "\n"
        )
        return result
    ap_scores, _, _, _, _, _ = instance_eval.evaluate_matches(matches)
    averages = instance_eval.compute_averages(ap_scores)
    instance_eval.write_result_file(averages, str(output_dir / f"{name}.csv"))
    chair = averages["classes"]["chair"]
    return {key: float(chair[source]) for key, source in (
        ("ap", "ap"), ("ap50", "ap50%"), ("ap25", "ap25%")
    )}


def build_scene(scene_name, rows, track_root, processed_scene_root):
    superpoint_sizes, points_by_superpoint = _load_superpoints(
        scene_name, processed_scene_root
    )
    tracks = _load_tracks(track_root, scene_name, superpoint_sizes)
    source_geometry = {
        proposal_id: set(track["superpoint_ids"]) for proposal_id, track in tracks.items()
    }
    geometry, selections = _prepare_selections(source_geometry, rows)
    summary = {
        "scene_name": scene_name,
        "frozen_boundary_action_count": len(rows),
        "selected_assign_count": sum(
            row["selected_action_kind"] == "assign_owner" for row in selections
        ),
        "selected_unknown_noop_count": sum(
            row["selected_action_kind"] == "unknown_noop" for row in selections
        ),
        "fallback_would_empty_count": sum(
            row["global_status"] == "fallback_would_empty" for row in selections
        ),
        "empty_proposal_count": sum(not ids for ids in geometry.values()),
        "geometry_changed_proposal_count": sum(
            geometry[proposal_id] != source_geometry[proposal_id]
            for proposal_id in source_geometry
        ),
    }
    if summary["empty_proposal_count"]:
        raise ValueError(f"{scene_name} global oracle retained empty proposals")
    return tracks, source_geometry, geometry, points_by_superpoint, selections, summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--action-ledger", type=Path, required=True)
    parser.add_argument("--f2-track-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--skip-ap", action="store_true")
    parser.add_argument("--memory-light-ap", action="store_true")
    parser.add_argument("--save-ap-records", action="store_true")
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "action_ledger", "f2_track_root", "processed_scene_root",
        "native_prediction_cache", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    rows_by_scene = {scene: [] for scene in scenes}
    for row in _read_jsonl(args.action_ledger):
        if row["scene_name"] in rows_by_scene:
            rows_by_scene[row["scene_name"]].append(row)
    args.output_dir.mkdir(parents=True)
    all_selections = []
    scene_summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        tracks, source, oracle, points, selections, summary = build_scene(
            scene_name, rows_by_scene[scene_name], args.f2_track_root,
            args.processed_scene_root,
        )
        del tracks, source, oracle, points
        all_selections.extend(selections)
        scene_summaries.append(summary)
        print(f"[build] {index}/{len(scenes)} {scene_name}", flush=True)
    _write_jsonl(
        args.output_dir / "global_feasible_action_selection.jsonl",
        [
            {key: value for key, value in row.items() if key != "unknown_noop_action"}
            for row in all_selections
        ],
    )
    result = {
        "diagnostic_type": "GT-only global-feasible frozen boundary geometry oracle",
        "decision_constraint": DECISION_CONSTRAINT,
        "selection_objective": (
            "per boundary, maximize the sum of affected proposals' frozen fixed-target "
            "IoU delta; tie-break to unknown/no-op; then enforce no-empty-proposal fallback"
        ),
        "oracle_scope": (
            "fixed-score feasible geometry oracle; not exact global AP maximization and "
            "not the fixed-mask scoring/ranking oracle"
        ),
        "proposal_materialization_applied": False,
        "native_prediction_mutation_applied": False,
        "scene_count": len(scenes),
        "scene_summaries": scene_summaries,
        "selected_assign_count": sum(
            row["selected_action_kind"] == "assign_owner" for row in all_selections
        ),
        "selected_unknown_noop_count": sum(
            row["selected_action_kind"] == "unknown_noop" for row in all_selections
        ),
        "fallback_would_empty_count": sum(
            row["global_status"] == "fallback_would_empty" for row in all_selections
        ),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if not args.skip_ap:
        baseline = evaluate_native_plus_geometry(
            "native_plus_frozen_f2_class_agnostic",
            args.native_prediction_cache,
            scenes,
            rows_by_scene,
            args.f2_track_root,
            args.processed_scene_root,
            args.gt_instance_dir,
            args.output_dir,
            False,
            args.memory_light_ap,
            (
                args.output_dir / "native_plus_frozen_f2_ap_records.npz"
                if args.save_ap_records and args.memory_light_ap else None
            ),
        )
        oracle = evaluate_native_plus_geometry(
            "native_plus_global_feasible_oracle_class_agnostic",
            args.native_prediction_cache,
            scenes,
            rows_by_scene,
            args.f2_track_root,
            args.processed_scene_root,
            args.gt_instance_dir,
            args.output_dir,
            True,
            args.memory_light_ap,
            (
                args.output_dir / "global_feasible_oracle_ap_records.npz"
                if args.save_ap_records and args.memory_light_ap else None
            ),
        )
        result.update({
            "ap_computed": True,
            "frozen_f2_fixed_score_class_agnostic": baseline,
            "global_feasible_oracle_fixed_score_class_agnostic": oracle,
            "oracle_minus_frozen_f2": {key: oracle[key] - baseline[key] for key in baseline},
        })
    else:
        result["ap_computed"] = False
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    gc.collect()


if __name__ == "__main__":
    main()

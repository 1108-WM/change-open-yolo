#!/usr/bin/env python3
"""GT-only per-boundary action oracle for the frozen F2 assignment plan.

This diagnostic never materializes a proposal.  For every frozen
``assign_unique_pareto_owner`` boundary superpoint it evaluates the exact
one-superpoint counterfactuals ``assign_to_candidate`` and
``unknown_noop_keep_current_ownership`` against ScanNet instance GT.  The
``unknown`` action deliberately means *no-op*: all current memberships stay
unchanged.  It never means deleting the superpoint from every proposal.

The output is an action-result ledger for the next, separate global-feasible
oracle step.  It is not a selector, does not compute AP, and must not be used
to choose inference actions, thresholds, scores, or classes.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_THRESHOLDS = (0.25, 0.50)
ASSIGN_ACTION = "assign_unique_pareto_owner"
UNKNOWN_NOOP_ACTION = "unknown_noop_keep_current_ownership"
GT_USAGE = "post-hoc action-space attribution only"
DECISION_CONSTRAINT = (
    "GT is used only to measure frozen action-space outcomes. This ledger must "
    "not choose inference actions, thresholds, scores, classes, or a selector."
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_superpoints(scene_name, processed_scene_root):
    path = processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    points_by_id = {
        int(item): np.flatnonzero(superpoints == item).astype(np.int64)
        for item in ids
    }
    return {int(item): int(count) for item, count in zip(ids, counts)}, points_by_id


def _load_tracks(track_root, scene_name, superpoint_sizes):
    payload = json.loads((track_root / scene_name / "automatic_tracks.json").read_text())
    tracks = payload.get("tracks", [])
    if int(payload.get("track_count", len(tracks))) != len(tracks):
        raise ValueError(f"{scene_name} F2 track count is inconsistent")
    result = {}
    for track in tracks:
        proposal_id = int(track["proposal_id"])
        if proposal_id in result or int(track.get("track_id", proposal_id)) != proposal_id:
            raise ValueError(f"{scene_name} has invalid proposal identity {proposal_id}")
        superpoint_ids = tuple(map(int, track.get("superpoint_ids", [])))
        if not superpoint_ids or list(superpoint_ids) != sorted(set(superpoint_ids)):
            raise ValueError(f"{scene_name}/{proposal_id} has invalid superpoint IDs")
        if set(superpoint_ids) - set(superpoint_sizes):
            raise ValueError(f"{scene_name}/{proposal_id} has unknown superpoints")
        expected_count = sum(superpoint_sizes[item] for item in superpoint_ids)
        if int(track.get("superpoint_count", -1)) != len(superpoint_ids):
            raise ValueError(f"{scene_name}/{proposal_id} superpoint count differs")
        if int(track.get("point_count", -1)) != expected_count:
            raise ValueError(f"{scene_name}/{proposal_id} point count differs")
        result[proposal_id] = {
            "proposal_id": proposal_id,
            "superpoint_ids": frozenset(superpoint_ids),
            "track_score": float(track.get("mean_node_quality", 0.0)),
        }
    return result


def _owners_by_superpoint(tracks):
    owners = {}
    for proposal_id, track in tracks.items():
        for superpoint_id in track["superpoint_ids"]:
            owners.setdefault(int(superpoint_id), set()).add(int(proposal_id))
    return owners


def _valid_gt_instances(gt_ids, valid_class_ids):
    ids, counts = np.unique(gt_ids, return_counts=True)
    return {
        int(instance_id): int(count)
        for instance_id, count in zip(ids, counts)
        if int(instance_id) > 0 and int(instance_id) // 1000 in valid_class_ids
    }


def _points_for_superpoints(superpoint_ids, points_by_superpoint):
    chunks = [points_by_superpoint[item] for item in sorted(superpoint_ids)]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64, copy=False)


def _best_gt_instance(points, gt_ids, gt_instances):
    if len(points) == 0:
        return None
    present, intersections = np.unique(gt_ids[points], return_counts=True)
    best = None
    for instance_id, intersection in zip(present, intersections):
        instance_id = int(instance_id)
        if instance_id not in gt_instances:
            continue
        intersection = int(intersection)
        instance_size = int(gt_instances[instance_id])
        iou = float(intersection / (len(points) + instance_size - intersection))
        candidate = (iou, intersection, -instance_id, instance_id, instance_size)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return None
    return {
        "instance_id": int(best[3]),
        "instance_size": int(best[4]),
        "iou": float(best[0]),
        "intersection": int(best[1]),
    }


def _iou_with_instance(points, gt_ids, instance_id, instance_size):
    intersection = int(np.count_nonzero(gt_ids[points] == instance_id))
    union = len(points) + int(instance_size) - intersection
    return float(intersection / max(1, union)), intersection


def _threshold_state(source_iou, action_iou, threshold):
    source_pass = source_iou >= threshold
    action_pass = action_iou >= threshold
    if not source_pass and action_pass:
        return "upcross"
    if source_pass and not action_pass:
        return "downcross"
    return "both_pass" if source_pass else "both_fail"


def _native_target_stats(native_masks, native_scores, gt_ids, instance_id, instance_size):
    counts = np.count_nonzero(native_masks, axis=0)
    intersections = np.count_nonzero(native_masks[gt_ids == instance_id], axis=0)
    ious = intersections / np.maximum(1, counts + int(instance_size) - intersections)
    result = {"best_native_iou": float(np.max(ious)) if len(ious) else 0.0}
    for threshold in EVAL_THRESHOLDS:
        key = f"iou{int(threshold * 100)}"
        eligible = ious >= threshold
        result[f"native_{key}_candidate_count"] = int(np.count_nonzero(eligible))
        result[f"max_native_score_at_{key}"] = (
            float(np.max(native_scores[eligible])) if np.any(eligible) else None
        )
    return result


def _proposal_outcome(
    proposal_id,
    source_superpoints,
    action_superpoints,
    source_cache,
    points_by_superpoint,
    gt_ids,
    gt_instances,
    native_masks,
    native_scores,
    native_cache,
):
    source = source_cache[proposal_id]
    source_points = source["points"]
    fixed = source["fixed_gt"]
    action_points = _points_for_superpoints(action_superpoints, points_by_superpoint)
    row = {
        "proposal_id": int(proposal_id),
        "track_score": float(source["track_score"]),
        "source_superpoint_count": len(source_superpoints),
        "action_superpoint_count": len(action_superpoints),
        "source_point_count": len(source_points),
        "action_point_count": len(action_points),
        "added_superpoint_ids": sorted(action_superpoints - source_superpoints),
        "removed_superpoint_ids": sorted(source_superpoints - action_superpoints),
        "would_be_empty": not action_superpoints,
        "fixed_gt_available": fixed is not None,
    }
    if fixed is None:
        return row
    instance_id = int(fixed["instance_id"])
    action_iou, action_intersection = _iou_with_instance(
        action_points, gt_ids, instance_id, fixed["instance_size"]
    )
    action_best = _best_gt_instance(action_points, gt_ids, gt_instances)
    if native_masks is None:
        native = None
    else:
        if instance_id not in native_cache:
            native_cache[instance_id] = _native_target_stats(
                native_masks, native_scores, gt_ids, instance_id, fixed["instance_size"]
            )
        native = native_cache[instance_id]
    row.update({
        "fixed_gt_instance_id": instance_id,
        "fixed_gt_point_count": int(fixed["instance_size"]),
        "source_fixed_gt_intersection": int(fixed["intersection"]),
        "action_fixed_gt_intersection": int(action_intersection),
        "source_fixed_gt_iou": float(fixed["iou"]),
        "action_fixed_gt_iou": float(action_iou),
        "fixed_gt_iou_delta": float(action_iou - fixed["iou"]),
        "action_best_gt_instance_id": (
            int(action_best["instance_id"]) if action_best is not None else None
        ),
        "action_best_gt_iou": float(action_best["iou"]) if action_best else 0.0,
        "best_gt_target_changed": bool(
            action_best is not None and int(action_best["instance_id"]) != instance_id
        ),
        "native_coverage_annotation_available": native is not None,
    })
    if native is None:
        row["best_native_iou"] = None
        for threshold in EVAL_THRESHOLDS:
            key = f"iou{int(threshold * 100)}"
            row[f"native_{key}_candidate_count"] = None
            row[f"max_native_score_at_{key}"] = None
    else:
        row.update(native)
    for threshold in EVAL_THRESHOLDS:
        key = f"iou{int(threshold * 100)}"
        row[f"fixed_gt_{key}_state"] = _threshold_state(
            float(fixed["iou"]), float(action_iou), threshold
        )
        row[f"native_already_covers_fixed_gt_at_{key}"] = (
            None if native is None else bool(native[f"native_{key}_candidate_count"] > 0)
        )
        row[f"native_higher_score_cover_at_{key}"] = (
            None if native is None else bool(
                native[f"max_native_score_at_{key}"] is not None
                and native[f"max_native_score_at_{key}"] >= source["track_score"]
            )
        )
    return row


def _summarize_outcomes(outcomes):
    valid = [row for row in outcomes if row["fixed_gt_available"]]
    deltas = [float(row["fixed_gt_iou_delta"]) for row in valid]
    result = {
        "affected_proposal_count": len(outcomes),
        "fixed_gt_available_proposal_count": len(valid),
        "would_empty_proposal_count": sum(row["would_be_empty"] for row in outcomes),
        "fixed_gt_iou_improved_proposal_count": sum(delta > 0 for delta in deltas),
        "fixed_gt_iou_declined_proposal_count": sum(delta < 0 for delta in deltas),
        "mean_fixed_gt_iou_delta": float(np.mean(deltas)) if deltas else None,
        "best_gt_target_changed_proposal_count": sum(
            row["best_gt_target_changed"] for row in valid
        ),
    }
    for threshold in EVAL_THRESHOLDS:
        key = f"iou{int(threshold * 100)}"
        states = Counter(row[f"fixed_gt_{key}_state"] for row in valid)
        for state in ("upcross", "downcross", "both_pass", "both_fail"):
            result[f"fixed_gt_{key}_{state}_proposal_count"] = states[state]
    return result


def evaluate_boundary_action(
    scene_name,
    plan_row,
    target_owner_proposal_id,
    tracks,
    source_owners,
    source_cache,
    points_by_superpoint,
    gt_ids,
    gt_instances,
    native_masks,
    native_scores,
    native_cache,
):
    """Evaluate one frozen boundary action without writing any proposal."""
    superpoint_id = int(plan_row["superpoint_id"])
    current_owners = tuple(sorted(map(int, plan_row["current_owner_proposal_ids"])))
    candidates = tuple(sorted(map(int, plan_row["adjacent_candidate_proposal_ids"])))
    actual_owners = tuple(sorted(source_owners.get(superpoint_id, set())))
    if actual_owners != current_owners:
        raise ValueError(
            f"{scene_name}/{superpoint_id} current owners differ: "
            f"plan={current_owners}, tracks={actual_owners}"
        )
    if set(candidates) - set(tracks):
        raise ValueError(f"{scene_name}/{superpoint_id} plan references unknown proposal")
    if target_owner_proposal_id is not None and target_owner_proposal_id not in candidates:
        raise ValueError("assignment target is not a frozen adjacent candidate")
    affected = tuple(sorted(set(current_owners) | set(candidates if target_owner_proposal_id is not None else [])))
    outcomes = []
    for proposal_id in affected:
        source_ids = tracks[proposal_id]["superpoint_ids"]
        final_ids = set(source_ids)
        if target_owner_proposal_id is not None:
            if proposal_id == target_owner_proposal_id:
                final_ids.add(superpoint_id)
            else:
                final_ids.discard(superpoint_id)
        outcomes.append(_proposal_outcome(
            proposal_id,
            source_ids,
            frozenset(final_ids),
            source_cache,
            points_by_superpoint,
            gt_ids,
            gt_instances,
            native_masks,
            native_scores,
            native_cache,
        ))
    action_name = (
        UNKNOWN_NOOP_ACTION
        if target_owner_proposal_id is None
        else f"assign_to_proposal_{int(target_owner_proposal_id)}"
    )
    target_owners = current_owners if target_owner_proposal_id is None else (int(target_owner_proposal_id),)
    return {
        "action_name": action_name,
        "action_kind": "unknown_noop" if target_owner_proposal_id is None else "assign_owner",
        "target_owner_proposal_id": target_owner_proposal_id,
        "source_owner_proposal_ids": list(current_owners),
        "action_owner_proposal_ids": list(target_owners),
        "resolves_current_multi_owner_conflict": bool(
            target_owner_proposal_id is not None and len(current_owners) > 1
        ),
        "changes_ownership": bool(target_owner_proposal_id is not None and target_owners != current_owners),
        "locally_feasible_no_empty_proposal": not any(
            outcome["would_be_empty"] for outcome in outcomes
        ),
        "proposal_outcomes": outcomes,
        **_summarize_outcomes(outcomes),
    }


def diagnose_scene(
    scene_name,
    plan_rows,
    tracks,
    points_by_superpoint,
    gt_ids,
    gt_instances,
    native_masks,
    native_scores,
):
    source_owners = _owners_by_superpoint(tracks)
    source_cache = {}
    for proposal_id, track in tracks.items():
        points = _points_for_superpoints(track["superpoint_ids"], points_by_superpoint)
        source_cache[proposal_id] = {
            "points": points,
            "track_score": track["track_score"],
            "fixed_gt": _best_gt_instance(points, gt_ids, gt_instances),
        }
    native_cache = {}
    rows = []
    for plan_row in sorted(plan_rows, key=lambda row: int(row["superpoint_id"])):
        if plan_row.get("planned_action") != ASSIGN_ACTION:
            continue
        if plan_row.get("ground_truth_usage") != "none":
            raise ValueError(f"{scene_name} frozen plan used ground truth")
        if bool(plan_row.get("assignment_applied")) or bool(plan_row.get("proposal_mutation_applied")):
            raise ValueError(f"{scene_name} frozen plan was materialized")
        candidates = sorted(map(int, plan_row["adjacent_candidate_proposal_ids"]))
        if len(candidates) < 2:
            raise ValueError(f"{scene_name} planned action lacks competition")
        action_results = [evaluate_boundary_action(
            scene_name, plan_row, None, tracks, source_owners, source_cache,
            points_by_superpoint, gt_ids, gt_instances, native_masks, native_scores,
            native_cache,
        )]
        action_results.extend(evaluate_boundary_action(
            scene_name, plan_row, candidate_id, tracks, source_owners, source_cache,
            points_by_superpoint, gt_ids, gt_instances, native_masks, native_scores,
            native_cache,
        ) for candidate_id in candidates)
        rows.append({
            "scene_name": scene_name,
            "superpoint_id": int(plan_row["superpoint_id"]),
            "competition_state": plan_row["competition_state"],
            "candidate_owner_proposal_ids": candidates,
            "frozen_planned_owner_proposal_id": int(plan_row["planned_owner_proposal_id"]),
            "frozen_plan_action": plan_row["planned_action"],
            "frozen_plan_evidence_state": plan_row["evidence_state"],
            "unknown_definition": "no-op; retain exact current owner memberships",
            "action_result_count": len(action_results),
            "action_results": action_results,
            "proposal_materialized": False,
            "score_used_for_decision": False,
            "ground_truth_usage": GT_USAGE,
        })
    return rows


def summarize(rows):
    all_actions = [action for row in rows for action in row["action_results"]]
    assign_actions = [action for action in all_actions if action["action_kind"] == "assign_owner"]
    return {
        "frozen_planned_boundary_action_count": len(rows),
        "counterfactual_action_result_count": len(all_actions),
        "assign_owner_counterfactual_count": len(assign_actions),
        "unknown_noop_counterfactual_count": len(all_actions) - len(assign_actions),
        "assign_actions_locally_feasible_no_empty_count": sum(
            action["locally_feasible_no_empty_proposal"] for action in assign_actions
        ),
        "assign_actions_with_any_fixed_gt_improvement_count": sum(
            action["fixed_gt_iou_improved_proposal_count"] > 0 for action in assign_actions
        ),
        "assign_actions_with_any_fixed_gt_decline_count": sum(
            action["fixed_gt_iou_declined_proposal_count"] > 0 for action in assign_actions
        ),
        "note": (
            "This is a per-action local counterfactual ledger only. It does not "
            "resolve cross-action ownership conflicts, empty-proposal fallback, "
            "global feasibility, or AP; those are the next oracle step."
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--f2-track-root", type=Path, required=True)
    parser.add_argument("--assignment-plan-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--native-prediction-cache", type=Path)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "f2_track_root", "assignment_plan_root", "processed_scene_root",
        "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.native_prediction_cache is not None:
        args.native_prediction_cache = _resolve(args.native_prediction_cache)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    if args.native_prediction_cache is not None:
        manifest = json.loads(
            (args.native_prediction_cache / "native_cache_no_gt_manifest.json").read_text()
        )
        if manifest.get("candidate_inputs", {}).get("mode") != "mask3d_yoloworld_only":
            raise ValueError("native cache is not mask3d_yoloworld_only")

    from evaluate.scannet200 import eval_semantic_instance as instance_eval

    valid_gt_classes = frozenset(
        int(value) for value in instance_eval.PRED_ID_TO_ID.values() if int(value) >= 0
    )
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_dir.mkdir(parents=True)
    all_rows = []
    scene_summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        superpoint_sizes, points_by_superpoint = _load_superpoints(
            scene_name, args.processed_scene_root
        )
        tracks = _load_tracks(args.f2_track_root, scene_name, superpoint_sizes)
        plan_rows = _read_jsonl(
            args.assignment_plan_root / scene_name / "global_boundary_assignment_plan.jsonl"
        )
        gt_ids = instance_eval.util_3d.load_ids(args.gt_instance_dir / f"{scene_name}.txt")
        if len(gt_ids) != sum(superpoint_sizes.values()):
            raise ValueError(f"{scene_name} GT point count differs from superpoints")
        if args.native_prediction_cache is None:
            native_masks, native_scores = None, None
        else:
            native_masks = np.load(
                args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
            )
            native_scores = np.load(
                args.native_prediction_cache / f"{scene_name}_pred_scores.npy", mmap_mode="r"
            )
            if native_masks.shape != (len(gt_ids), len(native_scores)):
                raise ValueError(f"{scene_name} native mask/score shape differs")
        rows = diagnose_scene(
            scene_name, plan_rows, tracks, points_by_superpoint, gt_ids,
            _valid_gt_instances(gt_ids, valid_gt_classes), native_masks, native_scores,
        )
        all_rows.extend(rows)
        scene_summary = {"scene_name": scene_name, **summarize(rows)}
        scene_summaries.append(scene_summary)
        print(
            f"[done] {index}/{len(scenes)} {scene_name}: "
            f"frozen actions {len(rows)}", flush=True
        )
    result = {
        "diagnostic_type": "GT-only frozen F2 per-boundary A/B/unknown action oracle ledger",
        "decision_constraint": DECISION_CONSTRAINT,
        "unknown_definition": "no-op; retain exact current owner memberships",
        "proposal_materialization_applied": False,
        "ap_computed": False,
        "native_coverage_annotations_available": args.native_prediction_cache is not None,
        "scene_count": len(scenes),
        "scene_summaries": scene_summaries,
        **summarize(all_rows),
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_jsonl(args.output_dir / "boundary_action_oracle_ledger.jsonl", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

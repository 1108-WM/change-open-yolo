#!/usr/bin/env python3
"""T1b GT-only local action oracle for the frozen track-family ledger.

For every requested T1a action this tool rebuilds the changed D1 tracklet with
the frozen Details consensus contract, then runs the complete frozen D2b
merge/refine loop in memory.  It writes only diagnostic JSONL; it never writes
proposals, changes scores/classes, or computes AP.  GT access is guarded by an
explicit command-line switch.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST
from tools.build_details_iterative_proposal_merges import (
    CONSENSUS_PARAMS,
    iterative_merge_proposals,
)
from tools.refine_details_automatic_tracks_consensus import (
    _load_observations,
    refine_tracklet_superpoints,
)


ORACLE_CONTRACT = (
    "GT-only T1b diagnostic. Each action rebuilds frozen Details consensus and "
    "runs frozen D2b merge/refine in memory. It never writes proposals, scores, "
    "classes, semantic decisions, inference thresholds, or AP."
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


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id <= 0 or instance_id // 1000 not in valid_classes:
            continue
        count = int(np.sum(gt_ids == instance_id))
        if count >= int(min_region_size):
            sizes[instance_id] = count
    return gt_ids, sizes


def _points_from_superpoints(superpoint_ids, points_by_superpoint):
    chunks = [points_by_superpoint[int(item)] for item in superpoint_ids]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _best_gt(points, gt_ids, gt_sizes):
    if len(points) == 0:
        return -1, 0.0
    ids, counts = np.unique(gt_ids[points], return_counts=True)
    best_id, best_iou = -1, 0.0
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        iou = float(intersection / max(1, len(points) + gt_sizes[instance_id] - int(intersection)))
        if iou > best_iou:
            best_id, best_iou = instance_id, iou
    return int(best_id), float(best_iou)


def _iou_for_gt(points, gt_ids, instance_id, gt_size):
    if len(points) == 0 or instance_id <= 0:
        return 0.0
    intersection = int(np.sum(gt_ids[points] == int(instance_id)))
    return float(intersection / max(1, len(points) + int(gt_size) - intersection))


def _track_template(track):
    return {
        "track_id": int(track["track_id"]),
        "node_ids": sorted(map(int, track.get("node_ids", []))),
        "frame_ids": sorted(map(str, track.get("frame_ids", []))),
        "mean_node_quality": float(track.get("mean_node_quality", 0.0)),
        "mean_predicted_iou": float(track.get("mean_predicted_iou", 0.0)),
        "mean_stability_score": float(track.get("mean_stability_score", 0.0)),
        "mean_edge_score": float(track.get("mean_edge_score", 0.0)),
    }


def _refine_association(track_id, observation_ids, templates, observations, superpoint_sizes):
    """Produce one D2b input proposal from one T1 association tracklet."""
    kept, diagnostics, frame_rows, initial = refine_tracklet_superpoints(
        observation_ids, observations, superpoint_sizes, **CONSENSUS_PARAMS
    )
    if not kept:
        return None
    template = dict(templates.get(int(track_id), {
        "track_id": int(track_id), "node_ids": [], "frame_ids": [],
        "mean_node_quality": 0.0, "mean_predicted_iou": 0.0,
        "mean_stability_score": 0.0, "mean_edge_score": 0.0,
    }))
    kept = sorted(map(int, kept))
    template.update({
        "track_id": int(track_id),
        "source_track_id": int(track_id),
        "observation_ids": sorted(map(int, observation_ids)),
        "frame_ids": sorted(str(row["frame_index"]) for row in frame_rows),
        "support_view_count": len(frame_rows),
        "superpoint_ids": kept,
        "superpoint_count": len(kept),
        "point_count": int(sum(superpoint_sizes[item] for item in kept)),
        "initial_superpoint_count": len(initial),
        "removed_superpoint_count": len(initial - set(kept)),
        "mean_consensus_rate": float(np.mean([diagnostics[item]["consensus_rate"] for item in kept])),
        "mean_supported_coverage": float(np.mean([diagnostics[item]["mean_supported_coverage"] for item in kept])),
        "support_score": float(sum(diagnostics[item]["support_frames"] for item in kept)),
    })
    return template


def _association_from_frozen_tracks(tracks):
    return {
        int(track["track_id"]): sorted(map(int, track["observation_ids"]))
        for track in tracks
    }


def apply_t1_action(association, action):
    """Return a one-action association counterfactual or a required no-op fallback."""
    result = {int(track_id): list(observation_ids) for track_id, observation_ids in association.items()}
    kind = action["action_type"]
    source = action.get("source_track_id")
    target = action.get("target_track_id")
    source = None if source is None else int(source)
    target = None if target is None else int(target)
    if kind == "keep":
        return result, {"action_applied": False, "fallback_reason": "exact_keep_no_op", "surviving_track_ids": [target]}
    if target not in result:
        raise ValueError(f"target track {target} is absent")
    if kind == "attach":
        observation_ids = sorted(set(map(int, action["source_observation_ids"])))
        if any(item in owned for owned in result.values() for item in observation_ids):
            raise ValueError("attach observation is already owned by a frozen track")
        result[target] = sorted(set(result[target]) | set(observation_ids))
        return result, {"action_applied": True, "fallback_reason": None, "surviving_track_ids": [target]}
    if source not in result or source == target:
        raise ValueError(f"invalid source/target for {kind}: {source}->{target}")
    if kind == "reassign":
        moved = sorted(set(map(int, action["source_observation_ids"])))
        if not set(moved) <= set(result[source]):
            raise ValueError("reassign observation does not belong to source track")
        remaining = sorted(set(result[source]) - set(moved))
        if not remaining:
            return association, {"action_applied": False, "fallback_reason": "empty_source_track_atomic_fallback", "surviving_track_ids": [source, target]}
        result[source] = remaining
        result[target] = sorted(set(result[target]) | set(moved))
        return result, {"action_applied": True, "fallback_reason": None, "surviving_track_ids": [source, target]}
    if kind == "merge":
        result[target] = sorted(set(result[target]) | set(result[source]))
        del result[source]
        return result, {"action_applied": True, "fallback_reason": None, "surviving_track_ids": [target]}
    raise ValueError(f"unsupported action type: {kind}")


def _source_tracks_for_association(association, templates, observations, superpoint_sizes):
    """Build frozen-consensus D2b inputs once for a complete association map."""
    source_tracks = {}
    empty_track_ids = []
    for track_id, observation_ids in sorted(association.items()):
        record = _refine_association(track_id, observation_ids, templates, observations, superpoint_sizes)
        if record is None:
            empty_track_ids.append(int(track_id))
        else:
            source_tracks[int(track_id)] = record
    return source_tracks, empty_track_ids


def _run_d2b_from_source_tracks(source_tracks_by_id, empty_track_ids, observations, superpoint_sizes):
    source_tracks = [source_tracks_by_id[key] for key in sorted(source_tracks_by_id)]
    if not source_tracks:
        raise ValueError("all proposals became empty after frozen consensus")
    final, merge_actions, rounds, _ = iterative_merge_proposals(source_tracks, observations, superpoint_sizes)
    return final, {
        "source_track_count": len(source_tracks),
        "empty_after_consensus_track_ids": empty_track_ids,
        "d2b_merge_action_count": len(merge_actions),
        "d2b_final_proposal_count": len(final),
        "d2b_round_count": len(rounds),
    }


def _action_source_tracks(
    association, changed_association, action, apply_diag, baseline_source_tracks,
    templates, observations, superpoint_sizes,
):
    """Overlay one action's changed consensus tracks on the frozen base inputs."""
    if not apply_diag["action_applied"]:
        return baseline_source_tracks, [], None
    result = dict(baseline_source_tracks)
    changed_ids = {
        int(item) for item in (action.get("source_track_id"), action.get("target_track_id"))
        if item is not None
    }
    intentionally_removed = {
        int(action["source_track_id"])
    } if action["action_type"] == "merge" else set()
    empty_changed = []
    for track_id in changed_ids:
        if track_id in intentionally_removed:
            result.pop(track_id, None)
            continue
        record = _refine_association(
            track_id, changed_association[track_id], templates, observations, superpoint_sizes
        )
        if record is None:
            empty_changed.append(track_id)
        else:
            result[track_id] = record
    # A T1 action cannot delete a previously usable proposal by making its
    # consensus empty.  Keep is the atomic fallback, rather than silently
    # changing the candidate set used by the oracle.
    if empty_changed:
        return baseline_source_tracks, empty_changed, "empty_after_consensus_atomic_fallback"
    return result, [], None


def _proposal_gt_rows(proposals, gt_ids, gt_sizes, points_by_superpoint):
    rows = []
    for proposal in proposals:
        points = _points_from_superpoints(proposal["superpoint_ids"], points_by_superpoint)
        best_id, best_iou = _best_gt(points, gt_ids, gt_sizes)
        rows.append({
            "proposal_id": int(proposal["proposal_id"]),
            "lineage_proposal_ids": sorted(map(int, proposal["lineage_proposal_ids"])),
            "points": points,
            "best_gt_instance_id": best_id,
            "best_gt_iou": best_iou,
        })
    return rows


def action_oracle_rows(action, baseline_rows, action_rows, gt_ids, gt_sizes, affected_track_ids, surviving_track_ids):
    affected_track_ids = {int(item) for item in affected_track_ids if item is not None}
    surviving_track_ids = {int(item) for item in surviving_track_ids if item is not None}
    source_rows = [
        row for row in baseline_rows
        if set(row["lineage_proposal_ids"]) & affected_track_ids and row["best_gt_instance_id"] > 0
    ]
    candidates = [
        row for row in action_rows
        if set(row["lineage_proposal_ids"]) & surviving_track_ids
    ]
    details = []
    for source in source_rows:
        instance_id = source["best_gt_instance_id"]
        gt_size = gt_sizes[instance_id]
        best = max(
            candidates,
            key=lambda item: _iou_for_gt(item["points"], gt_ids, instance_id, gt_size),
            default=None,
        )
        action_iou = _iou_for_gt(best["points"], gt_ids, instance_id, gt_size) if best else 0.0
        action_best_id, _ = _best_gt(best["points"], gt_ids, gt_sizes) if best else (-1, 0.0)
        details.append({
            "baseline_proposal_id": source["proposal_id"],
            "fixed_gt_instance_id": instance_id,
            "baseline_iou": source["best_gt_iou"],
            "action_proposal_id": None if best is None else best["proposal_id"],
            "action_iou_to_fixed_gt": action_iou,
            "delta_iou": float(action_iou - source["best_gt_iou"]),
            "iou25_up": bool(source["best_gt_iou"] < .25 <= action_iou),
            "iou25_down": bool(action_iou < .25 <= source["best_gt_iou"]),
            "iou50_up": bool(source["best_gt_iou"] < .50 <= action_iou),
            "iou50_down": bool(action_iou < .50 <= source["best_gt_iou"]),
            "best_target_switched": bool(action_best_id != instance_id),
        })
    return details


def _same_geometry(left, right):
    if len(left) != len(right):
        return False
    left_map = {int(row["proposal_id"]): row for row in left}
    right_map = {int(row["proposal_id"]): row for row in right}
    if set(left_map) != set(right_map):
        return False
    return all(
        list(map(int, left_map[key]["superpoint_ids"])) == list(map(int, right_map[key]["superpoint_ids"]))
        and list(map(int, left_map[key]["lineage_proposal_ids"])) == list(map(int, right_map[key]["lineage_proposal_ids"]))
        for key in left_map
    )


def _scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    points_by_superpoint = {
        int(item): np.flatnonzero(superpoints == item).astype(np.int64)
        for item in ids
    }
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    if len(gt_ids) != len(superpoints):
        raise ValueError(f"{scene_name} GT/point count mismatch")
    frozen_tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    templates = {int(track["track_id"]): _track_template(track) for track in frozen_tracks}
    association = _association_from_frozen_tracks(frozen_tracks)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations = _load_observations(args.d1_root / scene_name, superpoints, visibility)
    baseline_source_tracks, baseline_empty_track_ids = _source_tracks_for_association(
        association, templates, observations, superpoint_sizes
    )
    baseline_final, baseline_diag = _run_d2b_from_source_tracks(
        baseline_source_tracks, baseline_empty_track_ids, observations, superpoint_sizes
    )
    frozen_d2b = json.loads((args.d2b_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    if not _same_geometry(baseline_final, frozen_d2b):
        raise ValueError(f"{scene_name} frozen consensus/D2b parity check failed")
    baseline_gt = _proposal_gt_rows(baseline_final, gt_ids, gt_sizes, points_by_superpoint)
    actions = list(enumerate(_read_jsonl(args.t1_ledger_root / scene_name / "track_family_actions.jsonl")))
    if args.action_types is not None:
        actions = [(index, row) for index, row in actions if row["action_type"] in args.action_types]
    actions = actions[args.action_offset:]
    if args.max_actions_per_scene is not None:
        actions = actions[:args.max_actions_per_scene]
    result_rows, proposal_rows = [], []
    for action_index, action in actions:
        changed_association, apply_diag = apply_t1_action(association, action)
        if apply_diag["action_applied"]:
            source_tracks, empty_changed, empty_fallback = _action_source_tracks(
                association, changed_association, action, apply_diag, baseline_source_tracks,
                templates, observations, superpoint_sizes,
            )
            if empty_fallback is None:
                action_final, action_diag = _run_d2b_from_source_tracks(
                    source_tracks, baseline_empty_track_ids, observations, superpoint_sizes
                )
            else:
                apply_diag = {
                    **apply_diag, "action_applied": False,
                    "fallback_reason": empty_fallback,
                }
                action_final, action_diag = baseline_final, baseline_diag
                action_diag = {**action_diag, "empty_after_consensus_track_ids": empty_changed}
        else:
            action_final, action_diag = baseline_final, baseline_diag
        action_gt = _proposal_gt_rows(action_final, gt_ids, gt_sizes, points_by_superpoint)
        affected = [action.get("source_track_id"), action.get("target_track_id")]
        details = action_oracle_rows(
            action, baseline_gt, action_gt, gt_ids, gt_sizes, affected,
            apply_diag["surviving_track_ids"],
        )
        for row in details:
            proposal_rows.append({
                "scene_name": scene_name, "t1_action_index": action_index,
                "action_name": action["action_name"], "action_type": action["action_type"],
                **row, "ground_truth_usage": "GT-only diagnostic",
            })
        result_rows.append({
            "scene_name": scene_name, "t1_action_index": action_index,
            "action_name": action["action_name"], "action_type": action["action_type"],
            "source_track_id": action.get("source_track_id"), "target_track_id": action.get("target_track_id"),
            "action_applied": apply_diag["action_applied"], "fallback_reason": apply_diag["fallback_reason"],
            "affected_baseline_proposal_count": len(details),
            "fixed_target_count": len(details),
            "delta_iou_sum": float(sum(row["delta_iou"] for row in details)),
            "improved_fixed_target_count": sum(row["delta_iou"] > 1e-12 for row in details),
            "harmed_fixed_target_count": sum(row["delta_iou"] < -1e-12 for row in details),
            "iou25_up_count": sum(row["iou25_up"] for row in details),
            "iou25_down_count": sum(row["iou25_down"] for row in details),
            "iou50_up_count": sum(row["iou50_up"] for row in details),
            "iou50_down_count": sum(row["iou50_down"] for row in details),
            "target_switch_count": sum(row["best_target_switched"] for row in details),
            "empty_after_consensus_track_count": len(action_diag["empty_after_consensus_track_ids"]),
            "d2b_final_proposal_count": action_diag["d2b_final_proposal_count"],
            "d2b_merge_action_count": action_diag["d2b_merge_action_count"],
            "proposal_materialization_applied": False, "ap_computed": False,
            "ground_truth_usage": "GT-only diagnostic", "decision_constraint": ORACLE_CONTRACT,
        })
    summary = {
        "scene_name": scene_name, "baseline_d2b": baseline_diag,
        "frozen_d2b_parity_verified": True, "evaluated_action_count": len(result_rows),
        "selected_action_offset": args.action_offset,
        "action_counts": dict(sorted(Counter(row["action_type"] for row in result_rows).items())),
        "proposal_materialization_applied": False, "ap_computed": False,
        "ground_truth_usage": "GT-only diagnostic", "decision_constraint": ORACLE_CONTRACT,
    }
    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "t1_action_oracle_gt.jsonl", result_rows)
        _write_jsonl(staging / "t1_action_oracle_proposals_gt.jsonl", proposal_rows)
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--t1-ledger-root", type=Path, required=True)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--d2b-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--max-actions-per-scene", type=int)
    parser.add_argument("--action-offset", type=int, default=0)
    parser.add_argument("--action-types", nargs="+", choices=("keep", "attach", "reassign", "merge"))
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; T1b is GT-only offline diagnostics.")
    for name in (
        "scene_list", "t1_ledger_root", "d1_root", "track_root", "d2b_root",
        "processed_scene_root", "dataset_root", "config_path", "gt_instance_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    if args.scene_offset < 0:
        raise SystemExit("--scene-offset must be non-negative")
    if args.max_actions_per_scene is not None and args.max_actions_per_scene <= 0:
        raise SystemExit("--max-actions-per-scene must be positive")
    if args.action_offset < 0:
        raise SystemExit("--action-offset must be non-negative")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    scenes = scenes[args.scene_offset:]
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _scene(scene_name, args)
        print(f"[done] {index}/{len(scenes)} {scene_name}: actions {summary['evaluated_action_count']}", flush=True)
        summaries.append(summary)
    totals = Counter()
    for summary in summaries:
        totals.update(summary["action_counts"])
    payload = {
        "diagnostic_type": "T1b local action GT oracle; global feasible oracle/AP is a later stage",
        "scene_count": len(summaries), "action_counts": dict(sorted(totals.items())),
        "partial_action_slice": args.max_actions_per_scene is not None or args.action_offset != 0,
        "proposal_materialization_applied": False, "ap_computed": False,
        "ground_truth_usage": "GT-only diagnostic", "decision_constraint": ORACLE_CONTRACT,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "config"},
        "scene_summaries": summaries,
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scene_count": payload["scene_count"], "action_counts": payload["action_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

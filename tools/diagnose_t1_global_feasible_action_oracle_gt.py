#!/usr/bin/env python3
"""GT-only global-feasible oracle for frozen T1 track-family actions.

This consumes the complete T1b local action ledger.  It selects positive local
fixed-target utility actions subject to a *single* live association state per
scene: every observation has one owner, each track keeps at most one
observation per frame, and an action must still be applicable after every
earlier selection.  It then rebuilds frozen Details consensus and D2b in
memory.  If cumulative choices empty a previously usable consensus track, the
lowest-utility responsible action falls back atomically to unknown/no-op and
the scene is replayed.

All geometry is ephemeral.  GT is allowed only behind --allow-gt-diagnostics;
no proposal, score, class, selector, or inference decision is materialized.
"""

from __future__ import annotations

import argparse
import csv
import gc
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

from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS,
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    _merge_scan_matches,
    instance_eval,
)
from tools.diagnose_t1_track_family_action_oracle_gt import (  # noqa: E402
    _association_from_frozen_tracks,
    _load_gt,
    _load_observations,
    _read_jsonl,
    _read_scenes,
    _run_d2b_from_source_tracks,
    _same_geometry,
    _source_tracks_for_association,
    _track_template,
)


DECISION_CONSTRAINT = (
    "GT-only T1 global feasible oracle. GT chooses an offline diagnostic action "
    "set only; it must not become a selector, score, threshold, class, or "
    "materialized proposal."
)
EPSILON = 1e-12


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _published_local_rows(diagnostics_root: Path, prefix: str, scene_name: str) -> dict[int, dict]:
    """Read all non-smoke T1b rows and require duplicate publications to agree."""
    rows: dict[int, dict] = {}
    pattern = f"{prefix}*/{scene_name}/t1_action_oracle_gt.jsonl"
    for path in sorted(diagnostics_root.glob(pattern)):
        if "_smoke_" in path.parts[-3]:
            continue
        for row in _read_jsonl(path):
            if row.get("scene_name") != scene_name:
                raise ValueError(f"scene mismatch in {path}")
            index = int(row["t1_action_index"])
            previous = rows.get(index)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting local action rows: {scene_name} index={index}")
            rows[index] = row
    return rows


def _observation_owners(association: dict[int, list[int]]) -> dict[int, int]:
    owners: dict[int, int] = {}
    for track_id, observation_ids in association.items():
        for observation_id in observation_ids:
            if observation_id in owners:
                raise ValueError(f"observation {observation_id} has multiple frozen owners")
            owners[observation_id] = int(track_id)
    return owners


def _track_frames(observation_ids: list[int], frame_by_observation: dict[int, str]) -> set[str]:
    frames = [frame_by_observation[int(item)] for item in observation_ids]
    if len(frames) != len(set(frames)):
        raise ValueError("frozen association already violates same-frame mutual exclusion")
    return set(frames)


def _copy_association(association: dict[int, list[int]]) -> dict[int, list[int]]:
    return {int(track): list(map(int, observations)) for track, observations in association.items()}


def _try_apply_live_action(
    association: dict[int, list[int]],
    owners: dict[int, int],
    frames_by_track: dict[int, set[str]],
    action: dict,
    frame_by_observation: dict[int, str],
) -> tuple[bool, str | None]:
    """Apply one action to live association or return its deterministic conflict."""
    kind = action["action_type"]
    if kind == "keep":
        return False, "unknown_noop_keep"
    source = action.get("source_track_id")
    target = action.get("target_track_id")
    source = None if source is None else int(source)
    target = None if target is None else int(target)
    moved = sorted(set(map(int, action.get("source_observation_ids", []))))
    if target not in association:
        return False, "cross_action_target_absent"
    if kind == "attach":
        if any(item in owners for item in moved):
            return False, "observation_already_owned"
        move_frames = {frame_by_observation[item] for item in moved}
        if len(move_frames) != len(moved) or move_frames & frames_by_track[target]:
            return False, "same_frame_mutual_exclusion"
        association[target] = sorted(set(association[target]) | set(moved))
        frames_by_track[target].update(move_frames)
        owners.update({item: target for item in moved})
        return True, None
    if source is None or source == target or source not in association:
        return False, "cross_action_source_absent"
    if kind == "reassign":
        if not moved or any(owners.get(item) != source for item in moved):
            return False, "observation_owner_conflict"
        remaining = sorted(set(association[source]) - set(moved))
        if not remaining:
            return False, "cumulative_empty_track"
        move_frames = {frame_by_observation[item] for item in moved}
        if len(move_frames) != len(moved) or move_frames & frames_by_track[target]:
            return False, "same_frame_mutual_exclusion"
        association[source] = remaining
        association[target] = sorted(set(association[target]) | set(moved))
        frames_by_track[source] = _track_frames(remaining, frame_by_observation)
        frames_by_track[target].update(move_frames)
        owners.update({item: target for item in moved})
        return True, None
    if kind == "merge":
        source_frames = frames_by_track[source]
        if source_frames & frames_by_track[target]:
            return False, "same_frame_mutual_exclusion"
        association[target] = sorted(set(association[target]) | set(association[source]))
        frames_by_track[target].update(source_frames)
        for item in association[source]:
            owners[item] = target
        del association[source]
        del frames_by_track[source]
        return True, None
    raise ValueError(f"unsupported action type {kind}")


def _positive_candidates(actions: list[dict], local_rows: dict[int, dict]) -> list[dict]:
    candidates = []
    for index, action in enumerate(actions):
        row = local_rows[index]
        if row["action_type"] != action["action_type"] or row["action_name"] != action["action_name"]:
            raise ValueError(f"local row/action mismatch at index {index}")
        utility = float(row["delta_iou_sum"])
        if row["action_applied"] and utility > EPSILON:
            candidates.append({"index": index, "action": action, "utility": utility})
    return sorted(candidates, key=lambda row: (-row["utility"], row["index"]))


def _select_once(
    baseline_association: dict[int, list[int]],
    candidates: list[dict],
    frame_by_observation: dict[int, str],
    forbidden: set[int],
) -> tuple[dict[int, list[int]], list[dict], dict[int, str]]:
    association = _copy_association(baseline_association)
    owners = _observation_owners(association)
    frames_by_track = {
        track_id: _track_frames(observations, frame_by_observation)
        for track_id, observations in association.items()
    }
    selected, rejected = [], {}
    for candidate in candidates:
        index = candidate["index"]
        if index in forbidden:
            rejected[index] = "cumulative_empty_consensus_fallback"
            continue
        applied, reason = _try_apply_live_action(
            association, owners, frames_by_track, candidate["action"], frame_by_observation
        )
        if applied:
            selected.append(candidate)
        else:
            rejected[index] = reason or "unknown_noop"
    return association, selected, rejected


def select_global_feasible_actions(
    baseline_association: dict[int, list[int]],
    actions: list[dict],
    local_rows: dict[int, dict],
    frame_by_observation: dict[int, str],
    templates: dict[int, dict],
    observations: dict,
    superpoint_sizes: dict[int, int],
    baseline_active_track_ids: set[int],
) -> tuple[dict[int, list[int]], dict[int, dict], list[int], dict[int, str], list[dict], dict]:
    """Select live-feasible positive actions, then atomically repair consensus empties."""
    candidates = _positive_candidates(actions, local_rows)
    forbidden: set[int] = set()
    while True:
        association, selected, rejected = _select_once(
            baseline_association, candidates, frame_by_observation, forbidden
        )
        source_tracks, empty_track_ids = _source_tracks_for_association(
            association, templates, observations, superpoint_sizes
        )
        problematic = sorted(set(empty_track_ids) & set(baseline_active_track_ids))
        if not problematic:
            return association, source_tracks, empty_track_ids, rejected, selected, {
                "candidate_count": len(candidates),
                "forbidden_cumulative_empty_count": len(forbidden),
            }
        responsible = [
            candidate for candidate in selected
            if any(
                track_id in problematic
                for track_id in (candidate["action"].get("source_track_id"), candidate["action"].get("target_track_id"))
                if track_id is not None
            )
        ]
        if not responsible:
            raise ValueError(f"cannot recover cumulative empty consensus tracks {problematic}")
        forbidden.add(min(responsible, key=lambda row: (row["utility"], row["index"]))["index"])


def _selection_rows(actions: list[dict], local_rows: dict[int, dict], selected: list[dict], rejected: dict[int, str]) -> list[dict]:
    selected_by_index = {row["index"]: row for row in selected}
    rows = []
    for index, action in enumerate(actions):
        local = local_rows[index]
        candidate = selected_by_index.get(index)
        if candidate is not None:
            status = "selected"
            reason = None
        elif index in rejected:
            status = "unknown_noop"
            reason = rejected[index]
        elif not local["action_applied"]:
            status = "unknown_noop"
            reason = str(local["fallback_reason"])
        elif float(local["delta_iou_sum"]) <= EPSILON:
            status = "unknown_noop"
            reason = "nonpositive_local_utility"
        else:
            raise ValueError(f"unclassified action {index}")
        rows.append({
            "scene_name": action["scene_name"],
            "t1_action_index": index,
            "action_name": action["action_name"],
            "action_type": action["action_type"],
            "source_track_id": action.get("source_track_id"),
            "target_track_id": action.get("target_track_id"),
            "local_fixed_target_iou_utility": float(local["delta_iou_sum"]),
            "local_action_applied": bool(local["action_applied"]),
            "global_status": status,
            "unknown_noop_reason": reason,
            "ground_truth_usage": "GT-only diagnostic",
            "proposal_materialization_applied": False,
            "ap_computed": False,
        })
    return rows


def _track_prediction(proposals: list[dict], points_by_superpoint: dict[int, np.ndarray], point_count: int) -> dict:
    masks = np.zeros((point_count, len(proposals)), dtype=bool)
    scores = np.zeros(len(proposals), dtype=np.float32)
    for column, proposal in enumerate(proposals):
        ids = list(map(int, proposal["superpoint_ids"]))
        if not ids:
            raise ValueError("global oracle D2b emitted an empty proposal")
        masks[np.concatenate([points_by_superpoint[item] for item in ids]), column] = True
        scores[column] = max(0.0, float(proposal.get("mean_node_quality", 0.0)))
    return {
        "pred_masks": masks,
        "pred_scores": scores,
        "pred_classes": np.full(len(proposals), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }


def _scene_ap_records(gt_by_label, pred_by_label, overlap_th):
    label_name = "chair"
    gt_instances = [gt for gt in gt_by_label[label_name] if gt["instance_id"] >= 1000 and gt["vert_count"] >= 100]
    pred_instances = pred_by_label[label_name]
    visited = {pred["uuid"]: False for pred in pred_instances}
    true = np.ones(len(gt_instances), dtype=np.float64)
    score = np.full(len(gt_instances), -float("inf"), dtype=np.float64)
    matched = np.zeros(len(gt_instances), dtype=bool)
    hard_false_negatives = 0
    for index, gt in enumerate(gt_instances):
        found = False
        for pred in gt["matched_pred"]:
            if visited[pred["uuid"]]:
                continue
            overlap = float(pred["intersection"]) / (gt["vert_count"] + pred["vert_count"] - pred["intersection"])
            if overlap > overlap_th:
                confidence = pred["confidence"]
                if matched[index]:
                    maximum, minimum = max(score[index], confidence), min(score[index], confidence)
                    score[index] = maximum
                    true, score, matched = np.append(true, 0.0), np.append(score, minimum), np.append(matched, True)
                else:
                    found, matched[index], score[index], visited[pred["uuid"]] = True, True, confidence, True
        if not found:
            hard_false_negatives += 1
    true, score = true[matched], score[matched]
    for pred in pred_instances:
        if any(float(gt["intersection"]) / (gt["vert_count"] + pred["vert_count"] - gt["intersection"]) > overlap_th for gt in pred["matched_gt"]):
            continue
        ignored = pred["void_intersection"] + sum(
            gt["intersection"] for gt in pred["matched_gt"]
            if gt["instance_id"] < 1000 or gt["vert_count"] < 100
        )
        if float(ignored) / pred["vert_count"] <= overlap_th:
            true, score = np.append(true, 0.0), np.append(score, pred["confidence"])
    return true, score, hard_false_negatives, bool(gt_instances), bool(pred_instances)


def _average_precision(parts: dict) -> float:
    if not parts["has_gt"]:
        return float("nan")
    if not parts["has_pred"]:
        return 0.0
    true = np.concatenate(parts["true"]) if parts["true"] else np.empty(0)
    score = np.concatenate(parts["score"]) if parts["score"] else np.empty(0)
    if not len(score):
        return 0.0
    order = np.argsort(score)
    score, true = score[order], true[order]
    cumulative = np.cumsum(true)
    _, unique_indices = np.unique(score, return_index=True)
    num_true = cumulative[-1] if len(cumulative) else 0.0
    cumulative = np.append(cumulative, 0)
    precision, recall = np.zeros(len(unique_indices) + 1), np.zeros(len(unique_indices) + 1)
    for index, score_index in enumerate(unique_indices):
        true_positive = num_true - cumulative[score_index - 1]
        false_positive = len(score) - score_index - true_positive
        false_negative = cumulative[score_index - 1] + parts["fn"]
        precision[index] = float(true_positive) / (true_positive + false_positive)
        recall[index] = float(true_positive) / (true_positive + false_negative)
    precision[-1], recall[-1] = 1.0, 0.0
    recall_for_conv = np.append(np.append(recall[0], recall), 0.0)
    return float(np.dot(precision, np.convolve(recall_for_conv, [-0.5, 0.0, 0.5], "valid")))


def _evaluate_ap(name: str, scenes: list[str], args, rows_by_scene: dict[str, dict[int, dict]], include_native: bool) -> dict:
    _configure_scannet200_instance_eval()
    records = {float(overlap): {"true": [], "score": [], "fn": 0, "has_gt": False, "has_pred": False} for overlap in instance_eval.opt["overlaps"]}
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda filename: _class_agnostic_gt_ids(original_load_ids(filename))
    try:
        for position, scene_name in enumerate(scenes, start=1):
            payload = _build_scene(scene_name, args, rows_by_scene[scene_name])
            track_prediction = _track_prediction(payload["final_proposals"], payload["points_by_superpoint"], payload["point_count"])
            gt_file = str(args.gt_instance_dir / f"{scene_name}.txt")
            track_gt, track_pred = instance_eval.assign_instances_for_scan(track_prediction, gt_file)
            if include_native:
                prefix = args.native_prediction_cache / f"{scene_name}_pred_"
                native_masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
                native_scores = np.load(str(prefix) + "scores.npy", mmap_mode="r")
                native_prediction = {"pred_masks": native_masks, "pred_scores": np.asarray(native_scores, dtype=np.float32), "pred_classes": np.full(len(native_scores), UNIFIED_PREDICTED_CLASS, dtype=np.int64)}
                native_gt, native_pred = instance_eval.assign_instances_for_scan(native_prediction, gt_file)
                gt_by_label, pred_by_label = _merge_scan_matches(native_gt, native_pred, track_gt, track_pred)
            else:
                gt_by_label, pred_by_label = track_gt, track_pred
            for overlap, part in records.items():
                true, score, fn, has_gt, has_pred = _scene_ap_records(gt_by_label, pred_by_label, overlap)
                part["true"].append(true); part["score"].append(score); part["fn"] += fn
                part["has_gt"] = part["has_gt"] or has_gt; part["has_pred"] = part["has_pred"] or has_pred
            print(f"[evaluate {name}] {position}/{len(scenes)} {scene_name}", flush=True)
            gc.collect()
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    aps = {overlap: _average_precision(part) for overlap, part in records.items()}
    result = {"ap": float(np.mean([value for overlap, value in aps.items() if overlap != 0.25])), "ap50": float(aps[0.5]), "ap25": float(aps[0.25])}
    (args.output_dir / f"{name}.csv").write_text("metric,value\n" + "\n".join(f"{key},{value}" for key, value in result.items()) + "\n")
    return result


def _legacy_failure_join(path: Path, args, rows_by_scene: dict[str, dict[int, dict]]) -> dict:
    """Join the frozen 149/143 failure rows to the final global geometry only."""
    wanted = {"Details关联未形成主属轨迹", "当前共识选择损失"}
    with path.open(newline="") as handle:
        source = [row for row in csv.DictReader(handle) if row["loss_attribution"] in wanted]
    cached, result = {}, []
    for row in source:
        scene_name, instance_id = row["scene_name"], int(row["gt_instance_id"])
        if scene_name not in rows_by_scene:
            raise ValueError(f"legacy failure scene is absent from global oracle: {scene_name}")
        payload = cached.get(scene_name)
        if payload is None:
            payload = _build_scene(scene_name, args, rows_by_scene[scene_name]); cached[scene_name] = payload
        gt_ids = np.loadtxt(args.gt_instance_dir / f"{scene_name}.txt", dtype=np.int64)
        gt_size = int(np.sum(gt_ids == instance_id))
        best = 0.0
        for proposal in payload["final_proposals"]:
            points = np.concatenate([payload["points_by_superpoint"][int(item)] for item in proposal["superpoint_ids"]])
            intersection = int(np.sum(gt_ids[points] == instance_id))
            best = max(best, float(intersection / max(1, len(points) + gt_size - intersection)))
        baseline = float(row["best_consensus_iou"])
        result.append({"scene_name": scene_name, "gt_instance_id": instance_id, "loss_attribution": row["loss_attribution"], "baseline_consensus_iou": baseline, "global_oracle_best_iou": best, "iou25_recovered": baseline < .25 <= best, "iou50_recovered": baseline < .50 <= best})
    by_type = {}
    for kind in sorted(wanted):
        subset = [row for row in result if row["loss_attribution"] == kind]
        by_type[kind] = {"frozen_row_count": len(subset), "global_iou25_count": sum(row["global_oracle_best_iou"] >= .25 for row in subset), "global_iou50_count": sum(row["global_oracle_best_iou"] >= .50 for row in subset), "iou25_recovered_count": sum(row["iou25_recovered"] for row in subset), "iou50_recovered_count": sum(row["iou50_recovered"] for row in subset)}
    return {"source": str(path), "frozen_row_count": len(result), "by_loss_attribution": by_type, "rows": result}


def _build_scene(scene_name: str, args, local_rows: dict[int, dict]) -> dict:
    from utils import WORLD_2_CAM

    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    points_by_superpoint = {int(item): np.flatnonzero(superpoints == item).astype(np.int64) for item in ids}
    frozen_tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    templates = {int(track["track_id"]): _track_template(track) for track in frozen_tracks}
    association = _association_from_frozen_tracks(frozen_tracks)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    observations = _load_observations(args.d1_root / scene_name, superpoints, raw_visibility.detach().cpu().numpy().astype(bool, copy=False))
    frame_by_observation = {int(item): str(row["frame_index"]) for item, row in observations.items()}
    baseline_source, baseline_empty = _source_tracks_for_association(association, templates, observations, superpoint_sizes)
    baseline_final, baseline_diag = _run_d2b_from_source_tracks(baseline_source, baseline_empty, observations, superpoint_sizes)
    frozen_d2b = json.loads((args.d2b_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    if not _same_geometry(baseline_final, frozen_d2b):
        raise ValueError(f"{scene_name} frozen D2b parity failed")
    actions = _read_jsonl(args.t1_ledger_root / scene_name / "track_family_actions.jsonl")
    if set(local_rows) != set(range(len(actions))):
        raise ValueError(f"{scene_name} local T1b coverage is not exactly once")
    final_association, source_tracks, empty_ids, rejected, selected, selection_diag = select_global_feasible_actions(
        association, actions, local_rows, frame_by_observation, templates, observations, superpoint_sizes, set(baseline_source)
    )
    final_proposals, final_diag = _run_d2b_from_source_tracks(source_tracks, empty_ids, observations, superpoint_sizes)
    if any(not row["superpoint_ids"] for row in final_proposals):
        raise ValueError(f"{scene_name} global D2b has empty proposal")
    selection_rows = _selection_rows(actions, local_rows, selected, rejected)
    summary = {
        "scene_name": scene_name,
        "frozen_action_count": len(actions),
        "selected_action_count": len(selected),
        "selected_action_counts": dict(Counter(row["action"]["action_type"] for row in selected)),
        "unknown_noop_count": len(actions) - len(selected),
        "unknown_noop_reason_counts": dict(Counter(row["unknown_noop_reason"] for row in selection_rows if row["unknown_noop_reason"])),
        "selected_local_utility_sum": float(sum(row["utility"] for row in selected)),
        "cumulative_empty_track_ids": sorted(map(int, set(empty_ids) - set(baseline_empty))),
        "baseline_d2b": baseline_diag,
        "global_d2b": final_diag,
        "frozen_d2b_parity_verified": True,
        "global_empty_proposal_count": 0,
        **selection_diag,
    }
    return {"summary": summary, "selection_rows": selection_rows, "final_proposals": final_proposals, "points_by_superpoint": points_by_superpoint, "point_count": len(superpoints), "final_association": final_association}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--t1-ledger-root", type=Path, required=True)
    parser.add_argument("--local-diagnostics-root", type=Path, default=Path("docs/diagnostics"))
    parser.add_argument("--local-output-prefix", default="t1_track_family_action_oracle_gt_gvc_safety60_uniform30_d1_20260806")
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--d2b-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--legacy-instance-failure-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--skip-ap", action="store_true")
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; this is GT-only offline diagnostics")
    for name in ("scene_list", "t1_ledger_root", "local_diagnostics_root", "d1_root", "track_root", "d2b_root", "processed_scene_root", "dataset_root", "config_path", "gt_instance_dir", "native_prediction_cache", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.legacy_instance_failure_csv is not None:
        args.legacy_instance_failure_csv = _resolve(args.legacy_instance_failure_csv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    rows_by_scene = {scene: _published_local_rows(args.local_diagnostics_root, args.local_output_prefix, scene) for scene in scenes}
    args.output_dir.mkdir(parents=True)
    staging = args.output_dir / f".scenes.tmp.{os.getpid()}"
    staging.mkdir()
    all_rows, summaries = [], []
    try:
        for position, scene_name in enumerate(scenes, start=1):
            payload = _build_scene(scene_name, args, rows_by_scene[scene_name])
            all_rows.extend(payload["selection_rows"]); summaries.append(payload["summary"])
            print(f"[build] {position}/{len(scenes)} {scene_name}: selected {payload['summary']['selected_action_count']}", flush=True)
        _write_jsonl(staging / "global_feasible_action_selection.jsonl", all_rows)
        result = {
            "diagnostic_type": "GT-only T1 global-feasible association/consensus/D2b oracle",
            "decision_constraint": DECISION_CONSTRAINT,
            "selection_objective": "descending positive local fixed-target ΔIoU; ties by frozen action index; enforce live ownership, same-frame, cross-action and cumulative-empty constraints",
            "oracle_scope": "globally feasible frozen-utility geometry oracle, not an exact combinatorial AP maximum and not an inference rule",
            "ground_truth_usage": "GT-only diagnostic",
            "proposal_materialization_applied": False,
            "scene_count": len(scenes),
            "scene_summaries": summaries,
            "selected_action_count": sum(row["selected_action_count"] for row in summaries),
            "selected_local_utility_sum": float(sum(row["selected_local_utility_sum"] for row in summaries)),
            "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "config"},
        }
        if args.skip_ap:
            result["ap_computed"] = False
        else:
            result["ap_computed"] = True
            result["tracks_only_global_feasible_class_agnostic"] = _evaluate_ap("tracks_only_global_feasible_class_agnostic", scenes, args, rows_by_scene, False)
            result["native_plus_tracks_global_feasible_class_agnostic"] = _evaluate_ap("native_plus_tracks_global_feasible_class_agnostic", scenes, args, rows_by_scene, True)
        if args.legacy_instance_failure_csv is not None:
            joined = _legacy_failure_join(args.legacy_instance_failure_csv, args, rows_by_scene)
            _write_jsonl(staging / "legacy_instance_failure_global_join.jsonl", joined.pop("rows"))
            result["legacy_instance_failure_global_join"] = joined
        (staging / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_dir / "scenes")
        (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"scene_count": len(scenes), "selected_action_count": result["selected_action_count"], "ap_computed": result["ap_computed"]}, ensure_ascii=False))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()

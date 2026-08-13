#!/usr/bin/env python3
"""GT-only C1b score-order audit for C1a-selected frozen D2b tracks.

For each official threshold this consumes C1a's own GT-only keep subset.  The
native masks and scores never change.  C1b separates two different GT-only
questions which must not be conflated:

* ``gt_quality_track_rank_permutation``: assign the *existing selected-track
  score multiset* to tracks in best-GT-IoU order.  Thus native scores and the
  complete selected-track score distribution stay fixed; only the order of
  D2b tracks changes.
* ``gt_iou_track_score_rule``: replace each selected D2b score with its best
  GT IoU while native scores stay frozen.  This tests one particular
  cross-source calibration rule, but it changes the placement of tracks among
  native predictions.

Neither is an AP ceiling: the evaluator's greedy matching can make a
GT-quality ordering lower than the frozen ordering, and C1a's
maximum-matching keep set is not an AP-optimal assignment.  A later C1c
relation-component oracle is required for the general competition ceiling.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_track_native_competition_ledger import _native_cache_contract, _read_scenes  # noqa: E402
from diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS,
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from diagnose_mv3dis_global_feasible_action_oracle_gt import (  # noqa: E402
    _average_precision,
    _scene_ap_records,
)

AP25_THRESHOLD = 0.25
OFFICIAL_AP_THRESHOLDS = tuple(sorted(
    round(float(value), 2) for value in instance_eval.opt["overlaps"] if float(value) >= .50
))
EXTRA_DIAGNOSTIC_THRESHOLDS = (0.95,)
DIAGNOSTIC_THRESHOLDS = (AP25_THRESHOLD, *OFFICIAL_AP_THRESHOLDS, *EXTRA_DIAGNOSTIC_THRESHOLDS)


CONTRACT = (
    "GT-only C1b threshold-specific score-order audit. Native masks/scores and "
    "D2b geometry remain frozen; GT-only C1a keep subsets and GT-derived track "
    "orders/scores must not become an inference score, selector, threshold, or "
    "materialized candidate action.  Reported variants are diagnostic score "
    "rules, not an AP upper bound."
)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_tracks(track_root: Path, scene: str) -> dict[int, dict]:
    rows = json.loads((track_root / scene / "automatic_tracks.json").read_text())["tracks"]
    result = {int(row["track_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{scene}: duplicate filtered D2b track ID")
    return result


def _track_points(track: dict, point_count: int) -> np.ndarray:
    points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
    points = points[(points >= 0) & (points < point_count)]
    if len(points) != int(track["point_count"]):
        raise ValueError(f"track {track['track_id']}: point count differs")
    return points


def _native_prediction(root: Path, scene: str):
    masks = np.load(root / f"{scene}_pred_masks.npy", mmap_mode="r")
    scores = np.asarray(np.load(root / f"{scene}_pred_scores.npy", mmap_mode="r"), dtype=np.float32)
    if masks.ndim != 2 or masks.shape[1] != len(scores):
        raise ValueError(f"{scene}: native masks/scores differ")
    return masks, scores


def _track_rows(c1a_root: Path, scene: str) -> dict[int, dict]:
    rows = {int(row["track_id"]): row for row in _read_jsonl(c1a_root / scene / "track_filter_oracle_gt.jsonl")}
    if not rows:
        raise ValueError(f"{scene}: C1a track oracle is empty")
    return rows


def _selected_track_ids(oracle_rows: dict[int, dict], threshold_tag: str) -> list[int]:
    return [
        track_id for track_id, row in sorted(oracle_rows.items())
        if row[f"oracle_action_iou{threshold_tag}"] == "keep"
    ]


def _prediction(native_masks, native_scores, tracks, oracle_rows, threshold_tag, scene, score_map=None):
    selected = [
        track_id for track_id in _selected_track_ids(oracle_rows, threshold_tag)
    ]
    masks = np.zeros((native_masks.shape[0], len(selected)), dtype=bool)
    scores = np.zeros(len(selected), dtype=np.float32)
    for column, track_id in enumerate(selected):
        track = tracks[track_id]
        masks[_track_points(track, native_masks.shape[0]), column] = True
        scores[column] = float(
            oracle_rows[track_id]["frozen_track_score"]
            if score_map is None else score_map[(scene, track_id)]
        )
    return {
        "pred_masks": np.concatenate([native_masks, masks], axis=1),
        "pred_scores": np.concatenate([native_scores, scores]),
        "pred_classes": np.full(native_masks.shape[1] + len(selected), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }, selected


def _rescore_selected_tracks(prediction, native_scores, oracle_rows, selected, scene, score_map):
    """Reuse fixed masks while replacing only selected D2b scores in memory."""
    scores = np.asarray([
        score_map[(scene, track_id)] for track_id in selected
    ], dtype=np.float32)
    return {
        "pred_masks": prediction["pred_masks"],
        "pred_scores": np.concatenate([native_scores, scores]),
        "pred_classes": prediction["pred_classes"],
    }


def _score_maps(rows_by_scene: dict[str, dict[int, dict]], threshold_tag: str):
    """Build the two GT-only C1b track-score controls globally.

    The permutation control preserves every selected D2b score exactly, even
    across scenes.  It therefore isolates reordering the retained tracks from
    changing their score-scale distribution relative to frozen native scores.
    """
    entries = []
    for scene, rows in rows_by_scene.items():
        for track_id in _selected_track_ids(rows, threshold_tag):
            row = rows[track_id]
            entries.append((
                float(row["best_gt_iou"]),
                str(scene),
                int(track_id),
                float(row["frozen_track_score"]),
            ))
    # Ascending qualities receive ascending frozen scores.  Scene/ID break
    # equal-quality ties deterministically, without looking at evaluator AP.
    ranked = sorted(entries, key=lambda item: (item[0], item[1], item[2]))
    score_levels = sorted(item[3] for item in entries)
    permutation = {
        (scene, track_id): float(score)
        for score, (_, scene, track_id, _) in zip(score_levels, ranked)
    }
    raw_iou = {
        (scene, track_id): float(best_iou)
        for best_iou, scene, track_id, _ in entries
    }
    if len(permutation) != len(entries) or len(raw_iou) != len(entries):
        raise ValueError(f"IoU{threshold_tag}: duplicate selected D2b track key")
    return permutation, raw_iou, len(entries)


def _records(prediction, gt_file):
    """Run fixed-mask association once, then read every official threshold."""
    gt, pred = instance_eval.assign_instances_for_scan(prediction, str(gt_file))
    return {
        str(int(round(threshold * 100))): _scene_ap_records(gt, pred, threshold)
        for threshold in DIAGNOSTIC_THRESHOLDS
    }


def _assign(prediction, gt_file):
    """Associate fixed geometry once; confidence is changed separately below."""
    return instance_eval.assign_instances_for_scan(prediction, str(gt_file))


def _set_selected_track_confidences(
    gt_by_label,
    pred_by_label,
    native_masks,
    tracks,
    selected,
    scores,
):
    """Change only retained D2b confidences in an already associated scene.

    ``assign_instances_for_scan`` makes distinct shallow copies for top-level
    predictions and each GT's match list.  Updating both copies lets three
    score controls reuse exactly the same fixed mask--GT intersections.
    """
    min_points = int(instance_eval.opt["min_region_sizes"][0])
    native_valid_count = int(np.count_nonzero(np.count_nonzero(native_masks, axis=0) >= min_points))
    expected = {}
    valid_track_rank = 0
    for track_id, score in zip(selected, scores):
        if int(tracks[track_id]["point_count"]) < min_points:
            continue
        expected[native_valid_count + valid_track_rank] = float(score)
        valid_track_rank += 1
    by_uuid = {}
    for prediction in pred_by_label["chair"]:
        pred_id = int(prediction["pred_id"])
        if pred_id in expected:
            prediction["confidence"] = expected[pred_id]
            by_uuid[prediction["uuid"]] = expected[pred_id]
    if expected and len(by_uuid) != len(expected):
        observed_ids = {
            int(item["pred_id"]) for item in pred_by_label["chair"]
        }
        raise ValueError(
            "retained D2b score mapping differs from evaluator predictions: "
            f"expected_pred_ids={sorted(expected)}, "
            f"observed_track_pred_ids={sorted(set(expected) & observed_ids)}"
        )
    for gt in gt_by_label["chair"]:
        for matched_prediction in gt["matched_pred"]:
            confidence = by_uuid.get(matched_prediction["uuid"])
            if confidence is not None:
                matched_prediction["confidence"] = confidence


def _empty_record():
    return {"true": [], "score": [], "fn": 0, "has_gt": False, "has_pred": False}


def _append(record, values):
    true, score, fn, has_gt, has_pred = values
    record["true"].append(true)
    record["score"].append(score)
    record["fn"] += int(fn)
    record["has_gt"] = record["has_gt"] or bool(has_gt)
    record["has_pred"] = record["has_pred"] or bool(has_pred)


def _ap(record):
    return _average_precision(
        record["true"], record["score"], record["fn"], record["has_gt"], record["has_pred"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--c1a-oracle-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; GT is offline diagnosis only")
    for name in ("scene_list", "c1a_oracle_root", "filtered_d2b_track_root", "native_prediction_cache", "gt_instance_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    all_scenes = _read_scenes(args.scene_list)
    scenes = all_scenes[:args.max_scenes] if args.max_scenes is not None else all_scenes
    if not scenes:
        raise SystemExit("--max-scenes must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    cache_contract = _native_cache_contract(args.native_prediction_cache, len(all_scenes))
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_by_scene = {}
    tracks_by_scene = {}
    for scene in scenes:
        tracks = _load_tracks(args.filtered_d2b_track_root, scene)
        rows = _track_rows(args.c1a_oracle_root, scene)
        if set(rows) != set(tracks):
            raise ValueError(f"{scene}: C1a rows do not equal filtered D2b tracks")
        rows_by_scene[scene] = rows
        tracks_by_scene[scene] = tracks
    maps = {}
    for threshold in DIAGNOSTIC_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        maps[tag] = _score_maps(rows_by_scene, tag)
    records = {
        name: {str(int(round(threshold * 100))): _empty_record() for threshold in DIAGNOSTIC_THRESHOLDS}
        for name in (
            "native_frozen",
            "c1a_filter_frozen_track_score",
            "c1a_filter_gt_quality_track_rank_permutation",
            "c1a_filter_gt_iou_track_score_rule",
        )
    }
    selected_counts = {str(int(round(threshold * 100))): 0 for threshold in DIAGNOSTIC_THRESHOLDS}
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        for ordinal, scene in enumerate(scenes, 1):
            native_masks, native_scores = _native_prediction(args.native_prediction_cache, scene)
            tracks = tracks_by_scene[scene]
            c1a_rows = rows_by_scene[scene]
            gt_file = args.gt_instance_dir / f"{scene}.txt"
            native_prediction = {
                "pred_masks": native_masks,
                "pred_scores": native_scores,
                "pred_classes": np.full(native_masks.shape[1], UNIFIED_PREDICTED_CLASS, dtype=np.int64),
            }
            native_records = _records(native_prediction, gt_file)
            for threshold in DIAGNOSTIC_THRESHOLDS:
                tag = str(int(round(threshold * 100)))
                _append(records["native_frozen"][tag], native_records[tag])
            for threshold in DIAGNOSTIC_THRESHOLDS:
                tag = str(int(round(threshold * 100)))
                frozen, selected = _prediction(native_masks, native_scores, tracks, c1a_rows, tag, scene)
                permuted = _rescore_selected_tracks(
                    frozen, native_scores, c1a_rows, selected, scene, maps[tag][0]
                )
                raw_iou = _rescore_selected_tracks(
                    frozen, native_scores, c1a_rows, selected, scene, maps[tag][1]
                )
                selected_counts[tag] += len(selected)
                # The C1a keep set differs by threshold, so association is
                # still required once per threshold.  All three score controls
                # then reuse that identical fixed mask--GT association.
                frozen_gt, frozen_pred = _assign(frozen, gt_file)
                _append(
                    records["c1a_filter_frozen_track_score"][tag],
                    _scene_ap_records(frozen_gt, frozen_pred, threshold),
                )
                _set_selected_track_confidences(
                    frozen_gt, frozen_pred, native_masks, tracks, selected,
                    permuted["pred_scores"][native_masks.shape[1]:],
                )
                _append(
                    records["c1a_filter_gt_quality_track_rank_permutation"][tag],
                    _scene_ap_records(frozen_gt, frozen_pred, threshold),
                )
                _set_selected_track_confidences(
                    frozen_gt, frozen_pred, native_masks, tracks, selected,
                    raw_iou["pred_scores"][native_masks.shape[1]:],
                )
                _append(
                    records["c1a_filter_gt_iou_track_score_rule"][tag],
                    _scene_ap_records(frozen_gt, frozen_pred, threshold),
                )
                del frozen, permuted, raw_iou
            del native_prediction, native_masks, native_scores, tracks, c1a_rows
            gc.collect()
            print(f"[C1b track ranking oracle] {ordinal}/{len(scenes)} {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    metrics = {}
    for threshold in DIAGNOSTIC_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        native_ap = _ap(records["native_frozen"][tag])
        filtered_ap = _ap(records["c1a_filter_frozen_track_score"][tag])
        permutation_ap = _ap(records["c1a_filter_gt_quality_track_rank_permutation"][tag])
        raw_iou_ap = _ap(records["c1a_filter_gt_iou_track_score_rule"][tag])
        metrics[tag] = {
            "threshold": threshold,
            "native_frozen_ap": native_ap,
            "c1a_filter_frozen_track_score_ap": filtered_ap,
            "c1a_filter_gt_quality_track_rank_permutation_ap": permutation_ap,
            "c1a_filter_gt_iou_track_score_rule_ap": raw_iou_ap,
            "filter_gain_vs_native": filtered_ap - native_ap,
            "gt_quality_track_rank_permutation_gain_vs_filter": permutation_ap - filtered_ap,
            "gt_iou_track_score_rule_gain_vs_filter": raw_iou_ap - filtered_ap,
            "gt_iou_track_score_rule_gain_vs_native": raw_iou_ap - native_ap,
            "c1a_selected_track_count": selected_counts[tag],
        }
    official = [str(int(round(threshold * 100))) for threshold in OFFICIAL_AP_THRESHOLDS]
    summary = {
        "diagnostic_type": "GT-only C1b D2b/native threshold-specific track score-order audit",
        "decision_constraint": CONTRACT,
        "oracle_scope": (
            "C1a threshold-specific GT keep subsets; native scores frozen. "
            "The score-multiset permutation changes only selected D2b track order; "
            "the raw GT-IoU variant additionally tests a particular D2b/native score "
            "calibration. Neither result is a fixed-mask AP ceiling."
        ),
        "native_cache_contract": cache_contract,
        "official_ap_thresholds": list(OFFICIAL_AP_THRESHOLDS),
        "ap25_threshold": AP25_THRESHOLD,
        "extra_diagnostic_thresholds": list(EXTRA_DIAGNOSTIC_THRESHOLDS),
        "scene_count": len(scenes),
        "threshold_metrics": metrics,
        "aggregate_threshold_specific_ap": {
            "native": float(np.mean([metrics[tag]["native_frozen_ap"] for tag in official])),
            "filter_frozen_track_score": float(np.mean([metrics[tag]["c1a_filter_frozen_track_score_ap"] for tag in official])),
            "gt_quality_track_rank_permutation": float(np.mean([metrics[tag]["c1a_filter_gt_quality_track_rank_permutation_ap"] for tag in official])),
            "gt_iou_track_score_rule": float(np.mean([metrics[tag]["c1a_filter_gt_iou_track_score_rule_ap"] for tag in official])),
            "filter_gain_vs_native": float(np.mean([metrics[tag]["filter_gain_vs_native"] for tag in official])),
            "gt_quality_track_rank_permutation_gain_vs_filter": float(np.mean([metrics[tag]["gt_quality_track_rank_permutation_gain_vs_filter"] for tag in official])),
            "gt_iou_track_score_rule_gain_vs_filter": float(np.mean([metrics[tag]["gt_iou_track_score_rule_gain_vs_filter"] for tag in official])),
        },
        "proposal_materialization_applied": False,
        "ap_computed": True,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

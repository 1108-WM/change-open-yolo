#!/usr/bin/env python3
"""GT-only C1a oracle: filter frozen D2b tracks against frozen native masks.

Only a filtered D2b track may be kept or suppressed.  Native geometry,
classes, and scores are not read for a decision and remain fixed.  At each IoU
threshold GT selects a maximum-cardinality feasible keep subset; this measures
the *track-filter candidate-space coverage ceiling*, not AP and not a shared
inference rule.  It must be followed by the separate fixed-score and ranking
audits before any GVC implementation is considered.
"""
from __future__ import annotations

import argparse
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
from diagnose_n1_sampro3d_candidate_space_oracle_gt import _load_gt  # noqa: E402
from diagnose_n2_medoid_candidate_oracle_gt import (  # noqa: E402
    maximum_matching,
    native_edges,
    track_edges,
)
from diagnose_gvc_class_agnostic_ap import instance_eval  # noqa: E402

AP25_THRESHOLD = 0.25
OFFICIAL_AP_THRESHOLDS = tuple(sorted(
    round(float(value), 2) for value in instance_eval.opt["overlaps"] if float(value) >= .50
))
EXTRA_DIAGNOSTIC_THRESHOLDS = (0.95,)
DIAGNOSTIC_THRESHOLDS = (AP25_THRESHOLD, *OFFICIAL_AP_THRESHOLDS, *EXTRA_DIAGNOSTIC_THRESHOLDS)


CONTRACT = (
    "GT-only C1a track keep/suppress oracle. Native predictions remain frozen; "
    "GT only chooses threshold-specific offline track subsets and must not "
    "become a score, selector, threshold, or materialized candidate action."
)


def _resolve(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_tracks(track_root: Path, scene: str) -> list[dict]:
    tracks = json.loads((track_root / scene / "automatic_tracks.json").read_text())["tracks"]
    ids = [int(row["track_id"]) for row in tracks]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{scene}: filtered D2b track IDs are invalid")
    return tracks


def _best(edges: dict[int, float]) -> tuple[int, float]:
    if not edges:
        return -1, 0.0
    gt = min(edges, key=lambda item: (-edges[item], item))
    return int(gt), float(edges[gt])


def _scene_oracle(scene: str, args) -> tuple[list[dict], list[dict], dict]:
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)
    tracks = _load_tracks(args.filtered_d2b_track_root, scene)
    frozen_tracks = track_edges(args.filtered_d2b_track_root, scene, gt_ids, gt_sizes)
    frozen_native = native_edges(args.native_prediction_cache, scene, gt_ids, gt_sizes)
    expected_track_keys = {f"d2b_{int(row['track_id'])}" for row in tracks}
    if set(frozen_tracks) - expected_track_keys:
        raise ValueError(f"{scene}: track edge keys do not match filtered D2b tracks")
    combined = {**frozen_native, **frozen_tracks}

    selected_by_threshold = {}
    native_matched_gts_by_threshold = {}
    metrics = {}
    for threshold in DIAGNOSTIC_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        native_count, native_selected = maximum_matching(frozen_native, threshold)
        combined_count, combined_selected = maximum_matching(combined, threshold)
        selected_tracks = {
            int(key.removeprefix("d2b_")): int(gt)
            for key, gt in combined_selected.items() if key.startswith("d2b_")
        }
        selected_by_threshold[tag] = selected_tracks
        native_matched_gts_by_threshold[tag] = set(native_selected.values())
        metrics[tag] = {
            "threshold": threshold,
            "valid_gt_instance_count": len(gt_sizes),
            "native_frozen_maximum_matching": native_count,
            "native_plus_track_filter_oracle_maximum_matching": combined_count,
            "track_filter_oracle_increment_vs_native": combined_count - native_count,
            "track_keep_count": len(selected_tracks),
            "track_suppress_count": len(tracks) - len(selected_tracks),
            "threshold_specific_coverage_ceiling": combined_count / max(1, len(gt_sizes)),
            "native_selected_prediction_count": len(native_selected),
        }

    track_rows = []
    for track in tracks:
        track_id = int(track["track_id"])
        key = f"d2b_{track_id}"
        best_gt, best_iou = _best(frozen_tracks.get(key, {}))
        row = {
            "scene_name": scene,
            "track_id": track_id,
            "lineage_proposal_ids": list(map(int, track.get("lineage_proposal_ids", [track_id]))),
            "track_point_count": int(track["point_count"]),
            "frozen_track_score": float(track.get("mean_node_quality", 0.0)),
            "best_gt_instance_id": best_gt,
            "best_gt_iou": best_iou,
            "allowed_actions": ["keep", "suppress"],
            "native_geometry_mutation_applied": False,
            "track_geometry_mutation_applied": False,
            "track_score_mutation_applied": False,
            "ground_truth_usage": "offline_diagnostic_only",
            "proposal_materialization_applied": False,
            "ap_computed": False,
        }
        for threshold in DIAGNOSTIC_THRESHOLDS:
            tag = str(int(threshold * 100))
            row[f"oracle_action_iou{tag}"] = (
                "keep" if track_id in selected_by_threshold[tag] else "suppress"
            )
            row[f"oracle_matched_gt_iou{tag}"] = selected_by_threshold[tag].get(track_id)
        track_rows.append(row)

    gt_rows = []
    for gt in sorted(gt_sizes):
        row = {
            "scene_name": scene,
            "gt_instance_id": int(gt),
            "gt_point_count": int(gt_sizes[gt]),
            "ground_truth_usage": "offline_diagnostic_only",
            "proposal_materialization_applied": False,
            "ap_computed": False,
        }
        for threshold in DIAGNOSTIC_THRESHOLDS:
            tag = str(int(threshold * 100))
            matched_track = next(
                (track_id for track_id, target in selected_by_threshold[tag].items() if target == gt),
                None,
            )
            row[f"track_filter_oracle_selected_track_iou{tag}"] = matched_track
            row[f"native_frozen_matched_iou{tag}"] = gt in native_matched_gts_by_threshold[tag]
            row[f"track_filter_oracle_new_match_iou{tag}"] = (
                matched_track is not None and gt not in native_matched_gts_by_threshold[tag]
            )
        gt_rows.append(row)
    summary = {
        "scene_name": scene,
        "valid_gt_instance_count": len(gt_sizes),
        "frozen_native_candidate_count": len(frozen_native),
        "filtered_d2b_track_count": len(tracks),
        "threshold_metrics": metrics,
        "ground_truth_usage": "offline_diagnostic_only",
        "proposal_materialization_applied": False,
        "ap_computed": False,
    }
    return track_rows, gt_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; GT is offline diagnosis only")
    for name in ("scene_list", "filtered_d2b_track_root", "native_prediction_cache", "gt_instance_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max-scenes must be positive")
        scenes = scenes[:args.max_scenes]
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    cache_contract = _native_cache_contract(args.native_prediction_cache, len(_read_scenes(args.scene_list)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ordinal, scene in enumerate(scenes, 1):
        tracks, gts, summary = _scene_oracle(scene, args)
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        _write_jsonl(stage / "track_filter_oracle_gt.jsonl", tracks)
        _write_jsonl(stage / "gt_instance_oracle_gt.jsonl", gts)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_root / scene)
        summaries.append(summary)
        print(f"[C1a track filter oracle] {ordinal}/{len(scenes)} {scene}", flush=True)
    totals = {}
    for threshold in DIAGNOSTIC_THRESHOLDS:
        tag = str(int(round(threshold * 100)))
        keys = ("valid_gt_instance_count", "native_frozen_maximum_matching", "native_plus_track_filter_oracle_maximum_matching", "track_filter_oracle_increment_vs_native", "track_keep_count", "track_suppress_count")
        totals[tag] = {key: sum(row["threshold_metrics"][tag][key] for row in summaries) for key in keys}
        totals[tag]["threshold_specific_coverage_ceiling"] = totals[tag]["native_plus_track_filter_oracle_maximum_matching"] / max(1, totals[tag]["valid_gt_instance_count"])
    official = [str(int(round(value * 100))) for value in OFFICIAL_AP_THRESHOLDS]
    root = {
        "diagnostic_type": "GT-only C1a frozen D2b track keep/suppress oracle",
        "decision_constraint": CONTRACT,
        "oracle_scope": "threshold-specific maximum-cardinality track keep subsets; coverage ceiling only, not fixed-score AP, ranking, or an inference policy",
        "official_ap_thresholds": list(OFFICIAL_AP_THRESHOLDS),
        "ap25_threshold": AP25_THRESHOLD,
        "extra_diagnostic_thresholds": list(EXTRA_DIAGNOSTIC_THRESHOLDS),
        "native_cache_contract": cache_contract,
        "scene_count": len(summaries),
        "threshold_totals": totals,
        "aggregate_track_filter_coverage_ceiling": {
            "ap_like_coverage": float(np.mean([totals[tag]["threshold_specific_coverage_ceiling"] for tag in official])),
            "coverage50": totals["50"]["threshold_specific_coverage_ceiling"],
            "coverage25": totals["25"]["threshold_specific_coverage_ceiling"],
            "increment_like_ap": float(np.mean([totals[tag]["track_filter_oracle_increment_vs_native"] / max(1, totals[tag]["valid_gt_instance_count"]) for tag in official])),
        },
        "proposal_materialization_applied": False,
        "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the frozen hybrid with a GT-only Z4 winner-takes-local-component rule."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _Predictions,
    _evaluate,
    _load_native,
    _load_tracks,
    _load_union_rows,
    _points,
    _read_scenes,
    _resolve,
)


SOURCES = ("native_only", "track_only", "native_plus_track", "pair_union")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scene_prediction(scene: str, source: str, args, rows_by_scene_source, keep_by_key) -> dict:
    native = _load_native(args.stream_records_root, scene)
    point_count = native["pred_masks"].shape[0]
    native_scores = native["pred_scores"].copy()
    kept_native = set()
    for row in rows_by_scene_source[(scene, "native")]:
        candidate_id = int(row["candidate_id"])
        native_scores[candidate_id] = float(row["oof_predictions"]["C_joint_yolo_alpha"])
        if keep_by_key[(scene, "native", candidate_id)]:
            kept_native.add(candidate_id)
    native_mask = np.zeros(native["pred_masks"].shape[1], dtype=bool)
    for candidate_id in kept_native:
        native_mask[candidate_id] = True
    native_prediction = {
        "pred_masks": native["pred_masks"][:, native_mask],
        "pred_classes": native["pred_classes"][native_mask],
        "pred_scores": native_scores[native_mask],
    }

    tracks = _load_tracks(args.stream_records_root, scene)
    track_pieces = []
    for row in sorted(rows_by_scene_source[(scene, "track")], key=lambda item: int(item["candidate_id"])):
        candidate_id = int(row["candidate_id"])
        if not keep_by_key[(scene, "track", candidate_id)]:
            continue
        mask = np.zeros(point_count, dtype=bool)
        mask[_points(Path(tracks[candidate_id]["points_path"]), point_count)] = True
        track_pieces.append((mask, int(row["class_index"]), float(row["oof_predictions"]["C_joint_yolo_alpha"])))
    track_prediction = {
        "pred_masks": np.stack([item[0] for item in track_pieces], axis=1) if track_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([item[1] for item in track_pieces], dtype=np.int64),
        "pred_scores": np.asarray([item[2] for item in track_pieces], dtype=np.float32),
    }

    unions = _load_union_rows(args.combined_plan_root, scene)
    union_pieces = []
    for row in sorted(rows_by_scene_source[(scene, "pair_union")], key=lambda item: int(item["candidate_id"])):
        candidate_id = int(row["candidate_id"])
        if not keep_by_key[(scene, "pair_union", candidate_id)]:
            continue
        mask = np.zeros(point_count, dtype=bool)
        mask[_points(Path(unions[candidate_id]["points_path"]), point_count)] = True
        union_pieces.append((mask, int(row["class_index"]), float(row["original_score"])))
    union_prediction = {
        "pred_masks": np.stack([item[0] for item in union_pieces], axis=1) if union_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([item[1] for item in union_pieces], dtype=np.int64),
        "pred_scores": np.asarray([item[2] for item in union_pieces], dtype=np.float32),
    }
    pieces = {
        "native_only": [native_prediction], "track_only": [track_prediction],
        "native_plus_track": [native_prediction, track_prediction],
        "pair_union": [native_prediction, track_prediction, union_prediction],
    }[source]
    return {
        "pred_masks": np.concatenate([item["pred_masks"] for item in pieces], axis=1),
        "pred_classes": np.concatenate([item["pred_classes"] for item in pieces]).astype(np.int64),
        "pred_scores": np.concatenate([item["pred_scores"] for item in pieces]).astype(np.float32),
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    rows_by_scene_source = defaultdict(list)
    for row in rows:
        rows_by_scene_source[(str(row["scene_name"]), str(row["candidate_source"]))].append(row)
    plan_candidates = _read_jsonl(args.plan_root / "candidates.jsonl")
    keep_by_key = {}
    for row in plan_candidates:
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        if key in keep_by_key:
            raise ValueError(f"duplicate Z4 candidate plan key: {key}")
        if args.keep_field not in row:
            raise ValueError(f"Z4 plan row lacks --keep-field {args.keep_field}: {key}")
        keep_by_key[key] = bool(row[args.keep_field])
    expected_keys = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        for row in rows
    }
    if set(keep_by_key) != expected_keys:
        raise ValueError("Z4 plan does not exactly cover OOF candidates")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for source in SOURCES:
        mapping = _Predictions(
            scenes,
            lambda scene, source=source: _scene_prediction(
                scene, source, args, rows_by_scene_source, keep_by_key
            ),
        )
        results[source] = _evaluate(mapping, args.gt_instance_dir, args.output_dir / f"z4_winner__{source}.csv")
        print(f"[Z4 AP] {source}: {results[source]['ap']:.6f}", flush=True)
    hybrid_payload = json.loads(args.hybrid_summary.read_text())
    baseline = hybrid_payload["systems"]["C_joint_native_track_union_frozen_score"]
    deltas = {
        source: {name: float(results[source][name] - baseline[source][name]) for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")}
        for source in SOURCES
    }
    fold_results = []
    if args.split_manifest is not None:
        manifest = json.loads(args.split_manifest.read_text())
        baseline_folds = {int(row["fold_index"]): row for row in hybrid_payload["folds"]}
        for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
            fold_index = int(spec["fold_index"])
            fold_scenes = list(spec["validation_scenes"])
            mapping = _Predictions(
                fold_scenes,
                lambda scene: _scene_prediction(scene, "pair_union", args, rows_by_scene_source, keep_by_key),
            )
            current = _evaluate(
                mapping, args.gt_instance_dir, args.output_dir / f"fold_{fold_index}__z4_winner__pair_union.csv"
            )
            prior = baseline_folds[fold_index]["pair_union"]["C_joint_native_track_union_frozen_score"]
            fold_results.append({
                "fold_index": fold_index, "validation_scenes": fold_scenes, "pair_union": current,
                "delta_vs_union_frozen_hybrid": {
                    name: float(current[name] - prior[name])
                    for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
                },
            })
            print(f"[Z4 AP] fold {fold_index}: {current['ap']:.6f}", flush=True)
    summary = {
        "diagnostic_type": "Z4 GT-only evaluation of class-conditional local winner plan",
        "ground_truth_usage": "evaluation_only", "candidate_membership_modified": "evaluation_view_only",
        "geometry_modified": False, "class_modified": False,
        "plan_contract": args.plan_contract,
        "plan_keep_field": args.keep_field,
        "plan_candidate_count": len(plan_candidates),
        "plan_winner_count": sum(keep_by_key.values()),
        "results": results, "deltas_vs_union_frozen_hybrid": deltas,
        "folds": fold_results,
        "baseline_summary": str(args.hybrid_summary),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--hybrid-summary", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-field", default="component_winner")
    parser.add_argument("--plan-contract", default="same-class IoU>=0.50 connected component; keep highest hybrid competition score")
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("Z4 evaluator requires --allow-gt-evaluation")
    for name in (
        "scene_list", "oof_root", "plan_root", "stream_records_root", "combined_plan_root",
        "gt_instance_dir", "hybrid_summary", "split_manifest", "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _resolve(value))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

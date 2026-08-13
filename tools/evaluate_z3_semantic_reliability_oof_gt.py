#!/usr/bin/env python3
"""Evaluate OOF semantic reliability scores with the official ScanNet200 AP."""

from __future__ import annotations

import argparse
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


SYSTEMS = (
    "fixed_fusion_original_score", "A_base", "B_plus_yolo", "C_joint_yolo_alpha",
    "C_joint_native_track_union_frozen_score",
)
SOURCES = ("native_only", "track_only", "native_plus_track", "pair_union")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scene_prediction(scene: str, source: str, system: str, args, rows_by_scene_source) -> dict:
    native = _load_native(args.stream_records_root, scene)
    point_count = native["pred_masks"].shape[0]
    native_scores = native["pred_scores"].copy()
    prediction_group = "C_joint_yolo_alpha" if system == "C_joint_native_track_union_frozen_score" else system
    if system != "fixed_fusion_original_score":
        seen = set()
        for row in rows_by_scene_source[(scene, "native")]:
            candidate_id = int(row["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"{scene}: duplicate native OOF candidate {candidate_id}")
            seen.add(candidate_id)
            if int(native["pred_classes"][candidate_id]) != int(row["class_index"]):
                raise ValueError(f"{scene}: native OOF class mismatch {candidate_id}")
            native_scores[candidate_id] = float(row["oof_predictions"][prediction_group])
    native_prediction = {**native, "pred_scores": native_scores}

    tracks = _load_tracks(args.stream_records_root, scene)
    track_pieces = []
    for row in sorted(rows_by_scene_source[(scene, "track")], key=lambda item: int(item["candidate_id"])):
        candidate_id = int(row["candidate_id"])
        track = tracks[candidate_id]
        mask = np.zeros(point_count, dtype=bool)
        mask[_points(Path(track["points_path"]), point_count)] = True
        score = float(row["original_score"]) if system == "fixed_fusion_original_score" else float(row["oof_predictions"][prediction_group])
        track_pieces.append((mask, int(row["class_index"]), score))
    track_prediction = {
        "pred_masks": np.stack([piece[0] for piece in track_pieces], axis=1) if track_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([piece[1] for piece in track_pieces], dtype=np.int64),
        "pred_scores": np.asarray([piece[2] for piece in track_pieces], dtype=np.float32),
    }

    unions = _load_union_rows(args.combined_plan_root, scene)
    union_pieces = []
    for row in sorted(rows_by_scene_source[(scene, "pair_union")], key=lambda item: int(item["candidate_id"])):
        candidate_id = int(row["candidate_id"])
        union = unions[candidate_id]
        mask = np.zeros(point_count, dtype=bool)
        mask[_points(Path(union["points_path"]), point_count)] = True
        frozen_union_score = system in {"fixed_fusion_original_score", "C_joint_native_track_union_frozen_score"}
        score = float(row["original_score"]) if frozen_union_score else float(row["oof_predictions"][prediction_group])
        union_pieces.append((mask, int(row["class_index"]), score))
    union_prediction = {
        "pred_masks": np.stack([piece[0] for piece in union_pieces], axis=1) if union_pieces else np.zeros((point_count, 0), dtype=bool),
        "pred_classes": np.asarray([piece[1] for piece in union_pieces], dtype=np.int64),
        "pred_scores": np.asarray([piece[2] for piece in union_pieces], dtype=np.float32),
    }
    pieces = {
        "native_only": [native_prediction], "track_only": [track_prediction],
        "native_plus_track": [native_prediction, track_prediction],
        "pair_union": [native_prediction, track_prediction, union_prediction],
    }[source]
    return {
        "pred_masks": np.concatenate([piece["pred_masks"] for piece in pieces], axis=1),
        "pred_classes": np.concatenate([piece["pred_classes"] for piece in pieces]).astype(np.int64),
        "pred_scores": np.concatenate([piece["pred_scores"] for piece in pieces]).astype(np.float32),
    }


def _delta(current: dict, baseline: dict) -> dict:
    return {name: float(current[name] - baseline[name]) for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")}


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    rows_by_scene_source = defaultdict(list)
    for row in rows:
        rows_by_scene_source[(str(row["scene_name"]), str(row["candidate_source"]))].append(row)
    if {scene for scene, _ in rows_by_scene_source} != set(scenes):
        raise ValueError("OOF rows do not exactly cover requested scenes")
    manifest = json.loads(args.split_manifest.read_text())
    requested_systems = tuple(item.strip() for item in args.systems.split(",") if item.strip())
    unknown_systems = set(requested_systems) - set(SYSTEMS)
    if unknown_systems or "fixed_fusion_original_score" not in requested_systems:
        raise ValueError(
            "--systems must be a subset of known systems and include fixed_fusion_original_score"
        )
    validation_occurrences = defaultdict(int)
    for fold in manifest["folds"]:
        for scene in fold["validation_scenes"]:
            validation_occurrences[scene] += 1
    if set(validation_occurrences) != set(scenes) or any(value != 1 for value in validation_occurrences.values()):
        raise ValueError("split manifest does not provide exactly one OOF fold per scene")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for system in requested_systems:
        results[system] = {}
        for source in SOURCES:
            mapping = _Predictions(
                scenes,
                lambda scene, source=source, system=system: _scene_prediction(
                    scene, source, system, args, rows_by_scene_source
                ),
            )
            results[system][source] = _evaluate(
                mapping, args.gt_instance_dir, args.output_dir / f"{system}__{source}.csv"
            )
            print(f"[Z3 OOF AP] {system} {source}: {results[system][source]['ap']:.6f}", flush=True)

    expected_summary = json.loads(args.fixed_control_summary.read_text())
    expected = expected_summary["variants"]["frozen_context_equal_top1"]["sources"]["pair_union"]
    actual = results["fixed_fusion_original_score"]["pair_union"]
    control_max_abs_error = max(abs(float(actual[name]) - float(expected[name])) for name in ("ap", "ap50", "ap25"))
    if control_max_abs_error > args.control_tolerance:
        raise RuntimeError(f"fixed fusion control failed reproduction: max abs error {control_max_abs_error}")

    folds = []
    for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
        fold_scenes = list(spec["validation_scenes"])
        fold_result = {"fold_index": int(spec["fold_index"]), "validation_scenes": fold_scenes, "pair_union": {}}
        for system in requested_systems:
            mapping = _Predictions(
                fold_scenes,
                lambda scene, system=system: _scene_prediction(
                    scene, "pair_union", system, args, rows_by_scene_source
                ),
            )
            fold_result["pair_union"][system] = _evaluate(
                mapping, args.gt_instance_dir,
                args.output_dir / f"fold_{spec['fold_index']}__{system}__pair_union.csv",
            )
        baseline = fold_result["pair_union"]["fixed_fusion_original_score"]
        fold_result["deltas_vs_fixed_fusion"] = {
            system: _delta(fold_result["pair_union"][system], baseline)
            for system in requested_systems if system != "fixed_fusion_original_score"
        }
        folds.append(fold_result)
        print(f"[Z3 OOF AP] fold {spec['fold_index']} complete", flush=True)

    baseline = results["fixed_fusion_original_score"]
    summary = {
        "diagnostic_type": "official100 scene-isolated OOF open-vocabulary instance AP",
        "ground_truth_usage": "evaluation_only_after_OOF_training", "candidate_mutation": False,
        "scene_count": len(scenes), "systems": results,
        "deltas_vs_fixed_fusion": {
            system: {source: _delta(results[system][source], baseline[source]) for source in SOURCES}
            for system in requested_systems if system != "fixed_fusion_original_score"
        },
        "folds": folds,
        "control_reproduction": {
            "reference_summary": str(args.fixed_control_summary),
            "variant": "frozen_context_equal_top1", "max_abs_error": control_max_abs_error,
            "tolerance": args.control_tolerance, "valid": True,
        },
        "contracts": {
            "classes": "native unchanged; track/pair-union frozen-context-equal top1",
            "geometry": "unchanged", "membership": "unchanged except pre-existing invalid classes abstain",
            "scores": "each model score is from the scene's held-out fold only",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--fixed-control-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-tolerance", type=float, default=1e-10)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("OOF AP evaluation requires --allow-gt-evaluation")
    for name in (
        "scene_list", "oof_root", "split_manifest", "stream_records_root", "combined_plan_root",
        "gt_instance_dir", "fixed_control_summary", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

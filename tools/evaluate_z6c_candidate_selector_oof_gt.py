#!/usr/bin/env python3
"""Evaluate fixed Z6c OOF within-candidate selectors with official ScanNet200 AP."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_z3_semantic_reliability_oof_gt import _scene_prediction  # noqa: E402
from tools.evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _Predictions, _evaluate, _read_scenes, _resolve,
)


SYSTEMS = ("current_control", "semantic_only", "semantic_plus_dino")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _apply_selections(prediction: dict, selections: list[dict], system: str) -> dict:
    classes = np.asarray(prediction["pred_classes"], dtype=np.int64).copy()
    if system != "current_control":
        seen = set()
        for row in selections:
            index = int(row["prediction_index"])
            if index in seen or not 0 <= index < len(classes):
                raise ValueError(f"invalid or duplicate prediction index: {index}")
            seen.add(index)
            if int(row["current_class_index"]) != int(classes[index]):
                raise ValueError(f"current class mismatch at prediction {index}")
            selected = int(row["selectors"][system]["selected_class_index"])
            if not 0 <= selected < 198:
                raise ValueError(f"selected class outside registered space: {selected}")
            classes[index] = selected
    return {
        "pred_masks": prediction["pred_masks"],
        "pred_classes": classes,
        "pred_scores": prediction["pred_scores"],
    }


def _delta(current: dict, baseline: dict) -> dict:
    return {name: float(current[name] - baseline[name]) for name in (
        "ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap"
    )}


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    oof_rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    rows_by_scene_source = defaultdict(list)
    for row in oof_rows:
        rows_by_scene_source[(str(row["scene_name"]), str(row["candidate_source"]))].append(row)
    selections = defaultdict(list)
    for row in _read_jsonl(args.selector_root / "oof_selections.jsonl"):
        selections[str(row["scene_name"])].append(row)
    if set(selections) != set(scenes):
        raise ValueError("selector rows do not cover every official100 scene")
    for scene in scenes:
        selections[scene].sort(key=lambda row: int(row["prediction_index"]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    systems = {}
    for system in SYSTEMS:
        def build(scene, system=system):
            baseline = _scene_prediction(
                scene, "pair_union", "C_joint_native_track_union_frozen_score",
                args, rows_by_scene_source,
            )
            return _apply_selections(baseline, selections[scene], system)
        systems[system] = _evaluate(
            _Predictions(scenes, build), args.gt_instance_dir,
            args.output_dir / f"{system}.csv",
        )
        print(f"[Z6c selector AP] {system}: {systems[system]['ap']:.6f}", flush=True)

    reference = json.loads(args.hybrid_control_summary.read_text())[
        "systems"
    ]["C_joint_native_track_union_frozen_score"]["pair_union"]
    control_error = max(abs(float(systems["current_control"][name]) - float(reference[name])) for name in (
        "ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap"
    ))
    if control_error > args.control_tolerance:
        raise RuntimeError(f"current hybrid control reproduction failed: {control_error}")

    split = json.loads(args.split_manifest.read_text())
    occurrences = Counter(scene for fold in split["folds"] for scene in fold["validation_scenes"])
    if set(occurrences) != set(scenes) or any(value != 1 for value in occurrences.values()):
        raise ValueError("frozen split does not exactly cover official100")
    folds = []
    for spec in sorted(split["folds"], key=lambda row: int(row["fold_index"])):
        fold_scenes = list(spec["validation_scenes"])
        fold_systems = {}
        for system in SYSTEMS:
            def build_fold(scene, system=system):
                baseline = _scene_prediction(
                    scene, "pair_union", "C_joint_native_track_union_frozen_score",
                    args, rows_by_scene_source,
                )
                return _apply_selections(baseline, selections[scene], system)
            fold_systems[system] = _evaluate(
                _Predictions(fold_scenes, build_fold), args.gt_instance_dir,
                args.output_dir / f"fold_{spec['fold_index']}__{system}.csv",
            )
        folds.append({
            "fold_index": int(spec["fold_index"]), "validation_scenes": fold_scenes,
            "systems": fold_systems,
            "deltas_vs_current_control": {
                system: _delta(fold_systems[system], fold_systems["current_control"])
                for system in SYSTEMS if system != "current_control"
            },
        })
        print(f"[Z6c selector AP] fold {spec['fold_index']} complete", flush=True)
    summary = {
        "diagnostic_type": "official100 scene-isolated OOF direct within-candidate selector AP",
        "scene_count": len(scenes), "systems": systems,
        "deltas_vs_current_control": {
            system: _delta(systems[system], systems["current_control"])
            for system in SYSTEMS if system != "current_control"
        },
        "folds": folds,
        "positive_fold_counts_main_ap": {
            system: sum(fold["deltas_vs_current_control"][system]["ap"] > 0 for fold in folds)
            for system in SYSTEMS if system != "current_control"
        },
        "control_reproduction": {
            "max_abs_error": control_error, "tolerance": args.control_tolerance,
            "reference_summary": str(args.hybrid_control_summary), "valid": True,
        },
        "contracts": {
            "geometry": "unchanged", "membership": "unchanged",
            "scores": "unchanged current hybrid scores",
            "classes": "temporary OOF candidate argmax only; no inference plan written",
        },
        "ground_truth_usage": "evaluation_only_after_scene_isolated_oof_training",
        "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
        "inference_plan_written": False, "safety60_read": False, "even48_read": False,
        "test60_read": False,
        "params": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"))
    parser.add_argument("--selector-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_oof_official100_20260812"))
    parser.add_argument("--oof-root", type=Path, default=Path("docs/diagnostics/z3_semantic_reliability_oof_official100_20260811"))
    parser.add_argument("--split-manifest", type=Path, default=Path("output/train_candidate_quality_oof_official100_v2/split_manifest.json"))
    parser.add_argument("--stream-records-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/records"))
    parser.add_argument("--combined-plan-root", type=Path, default=Path("output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared/ground_truth"))
    parser.add_argument("--hybrid-control-summary", type=Path, default=Path("docs/diagnostics/z3_semantic_reliability_oof_ap_official100_20260811_v2_union_frozen_score_control/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_oof_ap_official100_20260812"))
    parser.add_argument("--control-tolerance", type=float, default=1e-10)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation: raise SystemExit("requires --allow-gt-evaluation")
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path): setattr(args, name, _resolve(value))
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()

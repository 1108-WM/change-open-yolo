#!/usr/bin/env python3
"""Evaluate one frozen Z6c selector system/subset in an isolated process."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_z3_semantic_reliability_oof_gt import _scene_prediction  # noqa: E402
from tools.evaluate_z3_yoloworld_control_group_gt import _Predictions, _evaluate, _read_scenes, _resolve  # noqa: E402
from tools.evaluate_z6c_candidate_selector_oof_gt import _apply_selections, _read_jsonl  # noqa: E402


SYSTEMS = (
    "current_control", "semantic_only", "semantic_plus_dino",
    "gated_semantic_only", "gated_semantic_plus_dino",
    "improvement_gated_semantic_only", "improvement_gated_semantic_plus_dino",
    "vlm_symmetric",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--fold-index", type=int, default=-1)
    parser.add_argument("--scene-list", type=Path, default=Path("output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"))
    parser.add_argument("--split-manifest", type=Path, default=Path("output/train_candidate_quality_oof_official100_v2/split_manifest.json"))
    parser.add_argument("--selector-root", type=Path, default=Path("docs/diagnostics/z6c_candidate_selector_oof_official100_20260812"))
    parser.add_argument("--oof-root", type=Path, default=Path("docs/diagnostics/z3_semantic_reliability_oof_official100_20260811"))
    parser.add_argument("--stream-records-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/records"))
    parser.add_argument("--combined-plan-root", type=Path, default=Path("output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/prepared/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("requires --allow-gt-evaluation")
    for name in (
        "scene_list", "split_manifest", "selector_root", "oof_root", "stream_records_root",
        "combined_plan_root", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    all_scenes = _read_scenes(args.scene_list)
    if args.fold_index >= 0:
        split = json.loads(args.split_manifest.read_text())
        matches = [fold for fold in split["folds"] if int(fold["fold_index"]) == args.fold_index]
        if len(matches) != 1:
            raise ValueError(f"unknown fold index {args.fold_index}")
        scenes = list(matches[0]["validation_scenes"])
    else:
        scenes = all_scenes
    selections = defaultdict(list)
    for row in _read_jsonl(args.selector_root / "oof_selections.jsonl"):
        selections[str(row["scene_name"])].append(row)
    for scene in scenes:
        selections[scene].sort(key=lambda row: int(row["prediction_index"]))
    oof_rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    rows_by_scene_source = defaultdict(list)
    for row in oof_rows:
        rows_by_scene_source[(str(row["scene_name"]), str(row["candidate_source"]))].append(row)

    def build(scene):
        baseline = _scene_prediction(
            scene, "pair_union", "C_joint_native_track_union_frozen_score", args, rows_by_scene_source
        )
        return _apply_selections(baseline, selections[scene], args.system)

    args.output_dir.mkdir(parents=True)
    metrics = _evaluate(
        _Predictions(scenes, build), args.gt_instance_dir, args.output_dir / "official_ap.csv"
    )
    payload = {
        "system": args.system, "fold_index": args.fold_index,
        "scene_count": len(scenes), "scenes": scenes, "metrics": metrics,
        "ground_truth_usage": "evaluation_only_after_scene_isolated_oof_training",
        "candidate_mutation": False, "geometry_mutation": False,
        "score_mutation": False, "inference_plan_written": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

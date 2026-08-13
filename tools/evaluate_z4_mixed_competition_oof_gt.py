#!/usr/bin/env python3
"""Evaluate independent native/track OOF with competition-aware union OOF."""

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
from tools.evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _Predictions, _evaluate, _read_scenes, _resolve,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    independent = _read_jsonl(args.independent_oof_root / "oof_predictions.jsonl")
    competition = _read_jsonl(args.competition_oof_root / "oof_predictions.jsonl")
    competition_union = {
        (str(row["scene_name"]), int(row["candidate_id"])): row
        for row in competition if str(row["candidate_source"]) == "pair_union"
    }
    rows_by_scene_source = defaultdict(list)
    replaced = 0
    for row in independent:
        current = row
        if str(row["candidate_source"]) == "pair_union":
            key = (str(row["scene_name"]), int(row["candidate_id"]))
            other = competition_union.get(key)
            if other is None:
                raise ValueError(f"missing competition-aware union OOF row: {key}")
            current = {**row, "oof_predictions": {
                **row["oof_predictions"],
                "C_joint_yolo_alpha": float(other["oof_predictions"]["C_joint_yolo_alpha"]),
            }}
            replaced += 1
        rows_by_scene_source[(str(current["scene_name"]), str(current["candidate_source"]))].append(current)
    if replaced != len(competition_union):
        raise ValueError("independent/competition union candidate contracts disagree")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mapping = _Predictions(
        scenes,
        lambda scene: _scene_prediction(
            scene, "pair_union", "C_joint_yolo_alpha", args, rows_by_scene_source
        ),
    )
    full = _evaluate(mapping, args.gt_instance_dir, args.output_dir / "mixed_competition__pair_union.csv")
    manifest = json.loads(args.split_manifest.read_text())
    reference = json.loads(args.hybrid_summary.read_text())
    reference_folds = {int(row["fold_index"]): row for row in reference["folds"]}
    folds = []
    for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
        fold_index = int(spec["fold_index"])
        fold_scenes = list(spec["validation_scenes"])
        fold_mapping = _Predictions(
            fold_scenes,
            lambda scene: _scene_prediction(
                scene, "pair_union", "C_joint_yolo_alpha", args, rows_by_scene_source
            ),
        )
        result = _evaluate(
            fold_mapping, args.gt_instance_dir,
            args.output_dir / f"fold_{fold_index}__mixed_competition__pair_union.csv",
        )
        baseline = reference_folds[fold_index]["pair_union"]["C_joint_native_track_union_frozen_score"]
        folds.append({
            "fold_index": fold_index, "validation_scenes": fold_scenes, "pair_union": result,
            "delta_vs_independent_union_frozen_hybrid": {
                name: float(result[name] - baseline[name])
                for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
            },
        })
        print(f"[Z4 mixed AP] fold {fold_index}: {result['ap']:.6f}", flush=True)
    baseline_full = reference["systems"]["C_joint_native_track_union_frozen_score"]["pair_union"]
    summary = {
        "diagnostic_type": "Z4 mixed independent-quality and one-to-one union competition OOF AP",
        "ground_truth_usage": "evaluation_only_after_two_scene-isolated_OOF_models",
        "candidate_mutation": False, "scene_count": len(scenes),
        "competition_union_row_count": replaced,
        "contract": "independent-quality joint OOF for native/track; one-to-one competition joint OOF for pair-union",
        "pair_union": full,
        "delta_vs_independent_union_frozen_hybrid": {
            name: float(full[name] - baseline_full[name])
            for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
        },
        "folds": folds,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--independent-oof-root", type=Path, required=True)
    parser.add_argument("--competition-oof-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--hybrid-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("mixed OOF evaluation requires --allow-gt-evaluation")
    for name in (
        "scene_list", "independent_oof_root", "competition_oof_root", "split_manifest",
        "stream_records_root", "combined_plan_root", "gt_instance_dir", "hybrid_summary", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

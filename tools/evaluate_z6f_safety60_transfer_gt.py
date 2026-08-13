#!/usr/bin/env python3
"""Evaluate the once-frozen Z6f safety60 transfer against its reproduced control."""

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

from tools.evaluate_z3_semantic_reliability_oof_gt import _scene_prediction  # noqa: E402
from tools.evaluate_z3_yoloworld_control_group_gt import _Predictions, _evaluate, _read_scenes, _resolve  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _delta(current: dict, control: dict) -> dict:
    return {name: float(current[name] - control[name]) for name in (
        "ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap"
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z3-prediction-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_evaluation:
        raise SystemExit("requires --allow-gt-evaluation")
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path): setattr(args, name, _resolve(value))
    if args.output_dir.exists(): raise SystemExit(f"refusing to overwrite {args.output_dir}")

    scenes = _read_scenes(args.scene_list)
    scene_set = set(scenes)
    rows_by_scene_source = defaultdict(list)
    z3_rows = _read_jsonl(args.z3_prediction_root / "oof_predictions.jsonl")
    z3_keys = set()
    for row in z3_rows:
        scene, source, candidate_id = (
            str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])
        )
        key = (scene, source, candidate_id)
        if scene not in scene_set or source not in {"native", "track", "pair_union"} or key in z3_keys:
            raise ValueError(f"invalid or duplicate Z3 prediction key: {key}")
        z3_keys.add(key)
        rows_by_scene_source[(scene, source)].append(row)
    expected_keys = set()
    for scene in scenes:
        prefix = args.stream_records_root / scene / "native_cache" / f"{scene}_pred_"
        native_count = len(np.load(str(prefix) + "classes.npy", mmap_mode="r"))
        native_classes = np.load(str(prefix) + "classes.npy", mmap_mode="r")
        expected_keys.update(
            (scene, "native", candidate_id)
            for candidate_id in range(native_count)
            if 0 <= int(native_classes[candidate_id]) < 198
        )
        track_path = (
            args.stream_records_root / scene / "d2b_tracks_filtered" / scene
            / "automatic_tracks.json"
        )
        tracks = json.loads(track_path.read_text())["tracks"]
        expected_keys.update((scene, "track", int(row["track_id"])) for row in tracks)
    for row in _read_jsonl(args.combined_plan_root / "pair_union_append_candidates.jsonl"):
        scene = str(row["scene_name"])
        if scene in scene_set:
            expected_keys.add((scene, "pair_union", int(row["candidate_id"])))
    if z3_keys != expected_keys:
        missing = sorted(expected_keys - z3_keys)[:5]
        extra = sorted(z3_keys - expected_keys)[:5]
        raise ValueError(f"Z3 transfer coverage mismatch; missing={missing}, extra={extra}")
    mutations = defaultdict(dict)
    review_rows = _read_jsonl(args.review_root / "review_outputs.jsonl")
    review_summary = json.loads((args.review_root / "summary.json").read_text())
    if int(review_summary.get("review_count", -1)) != len(review_rows):
        raise ValueError("Qwen review output count disagrees with its frozen summary")
    for row in review_rows:
        scene = str(row["scene_name"])
        if scene not in scene_set:
            raise ValueError(f"Qwen output references unknown scene: {scene}")
        if str(row["model_decision"]) == "PROPOSED":
            key = (str(row["candidate_source"]), int(row["candidate_id"]))
            if key in mutations[str(row["scene_name"])]:
                raise ValueError(f"duplicate VLM mutation: {row['scene_name']} {key}")
            mutations[str(row["scene_name"])][key] = (
                int(row["current_class_index"]), int(row["proposed_class_index"])
            )
            if not all(0 <= value < 198 for value in mutations[scene][key]):
                raise ValueError(f"Qwen mutation class outside registered space: {scene} {key}")

    def build(scene: str, apply_mutations: bool) -> dict:
        prediction = _scene_prediction(
            scene, "pair_union", "C_joint_native_track_union_frozen_score",
            args, rows_by_scene_source,
        )
        if not apply_mutations:
            return prediction
        classes = np.asarray(prediction["pred_classes"], dtype=np.int64).copy()
        prefix = args.stream_records_root / scene / "native_cache" / f"{scene}_pred_"
        native_count = len(np.load(str(prefix) + "classes.npy", mmap_mode="r"))
        index_by_key = {
            ("native", int(row["candidate_id"])): int(row["candidate_id"])
            for row in rows_by_scene_source[(scene, "native")]
        }
        offset = native_count
        for source in ("track", "pair_union"):
            source_rows = sorted(
                rows_by_scene_source[(scene, source)], key=lambda item: int(item["candidate_id"])
            )
            index_by_key.update({
                (source, int(row["candidate_id"])): offset + local_index
                for local_index, row in enumerate(source_rows)
            })
            offset += len(source_rows)
        if offset != len(classes):
            raise ValueError(f"{scene}: evaluator metadata length mismatch {offset} != {len(classes)}")
        for key, (current, proposed) in mutations.get(scene, {}).items():
            index = index_by_key.get(key)
            if index is None or int(classes[index]) != current:
                raise ValueError(f"{scene}: VLM mutation current-class mismatch {key}")
            classes[index] = proposed
        return {**prediction, "pred_classes": classes}

    args.output_dir.mkdir(parents=True)
    control = _evaluate(
        _Predictions(scenes, lambda scene: build(scene, False)),
        args.gt_instance_dir, args.output_dir / "control.csv",
    )
    z6f = _evaluate(
        _Predictions(scenes, lambda scene: build(scene, True)),
        args.gt_instance_dir, args.output_dir / "z6f_symmetric.csv",
    )
    payload = {
        "diagnostic_type": "one-way frozen Z6f safety60 open-vocabulary AP transfer",
        "scene_count": len(scenes), "mutation_count": sum(len(value) for value in mutations.values()),
        "current_control": control, "z6f_symmetric": z6f, "delta": _delta(z6f, control),
        "decision": "record only; no safety60-driven method, threshold, prompt, budget, or action changes",
        "ground_truth_usage": "evaluation_only_after_full_method_freeze",
        "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
        "safety60_read": True, "even48_read": False, "test60_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__": main()

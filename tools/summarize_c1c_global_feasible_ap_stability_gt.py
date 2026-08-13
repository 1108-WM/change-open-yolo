#!/usr/bin/env python3
"""Summarize stability of a completed C1c GT-only feasible AP construction.

This is read-only post-hoc diagnosis.  It compares the frozen append-only
``coexist`` state with the final state stored by the construction, using the
same evaluator-equivalent matching records.  Per-scene AP is descriptive only:
the optimization target remains a single global AP over all scenes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_track_native_competition_ledger import _read_scenes
from construct_c1c_global_feasible_ap_oracle_gt import (
    AP25,
    DIAGNOSTIC,
    OFFICIAL,
    _build_scene_cache,
    _configure_scannet200_instance_eval,
    _initial,
    _objective,
    _read_actions,
    _scene_records,
    _summary_records,
)
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids, instance_eval

EPS = 1e-12


def _resolve(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _metric_by_scene(records_by_scene: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for scene, records in records_by_scene.items():
        threshold_ap = _summary_records({scene: records})
        result[scene] = {
            "official_ap": _objective({scene: records}),
            "threshold_ap": threshold_ap,
        }
    return result


def _direction_counts(values: list[float]) -> dict[str, int]:
    return {
        "positive": sum(value > EPS for value in values),
        "negative": sum(value < -EPS for value in values),
        "zero": sum(abs(value) <= EPS for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--construction-summary", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path,
                        default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in (
        "construction_summary", "scene_list", "relation_ledger_root",
        "action_ledger_root", "filtered_d2b_track_root",
        "native_prediction_cache", "gt_instance_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("output root is non-empty")

    construction = json.loads(args.construction_summary.read_text())
    expected_thresholds = list(OFFICIAL)
    if construction.get("official_ap_thresholds") != expected_thresholds:
        raise ValueError("construction does not use evaluator official thresholds")
    scenes = _read_scenes(args.scene_list)
    if construction.get("scene_count") != len(scenes):
        raise ValueError("construction scene count differs from scene list")
    final_state = construction["final_component_actions"]
    if set(final_state) != set(scenes):
        raise ValueError("construction final actions do not cover the requested scenes")

    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_scannet200_instance_eval()
    original = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original(path))
    try:
        baseline_records: dict[str, dict] = {}
        final_records: dict[str, dict] = {}
        scene_details: dict[str, dict] = {}
        action_counts: Counter[str] = Counter()
        changed_components = 0
        total_components = 0
        for scene in scenes:
            actions = _read_actions(args.action_ledger_root, scene)
            baseline_state = _initial(actions, "coexist")
            selected_state = {int(cid): action for cid, action in final_state[scene].items()}
            if set(selected_state) != set(actions):
                raise ValueError(f"{scene}: final action components differ from ledger")
            cache = _build_scene_cache(scene, args)
            baseline_records[scene], baseline_detail = _scene_records(
                scene, actions, baseline_state, cache
            )
            final_records[scene], final_detail = _scene_records(
                scene, actions, selected_state, cache
            )
            total_components += len(actions)
            changed_components += sum(
                selected_state[cid] != baseline_state[cid] for cid in actions
            )
            action_counts.update(selected_state.values())
            scene_details[scene] = {
                "baseline_candidate_counts": baseline_detail,
                "final_candidate_counts": final_detail,
            }
    finally:
        instance_eval.util_3d.load_ids = original

    baseline_threshold = _summary_records(baseline_records)
    final_threshold = _summary_records(final_records)
    threshold_delta = {
        tag: final_threshold[tag] - baseline_threshold[tag]
        for tag in sorted(final_threshold, key=int)
    }
    baseline_scene = _metric_by_scene(baseline_records)
    final_scene = _metric_by_scene(final_records)
    scenes_payload = {}
    scene_deltas = []
    threshold_scene_deltas = {str(int(round(t * 100))): [] for t in OFFICIAL}
    for scene in scenes:
        delta = final_scene[scene]["official_ap"] - baseline_scene[scene]["official_ap"]
        scene_deltas.append(delta)
        per_threshold = {}
        for tag in threshold_scene_deltas:
            value = (final_scene[scene]["threshold_ap"][tag]
                     - baseline_scene[scene]["threshold_ap"][tag])
            per_threshold[tag] = value
            threshold_scene_deltas[tag].append(value)
        scenes_payload[scene] = {
            **scene_details[scene],
            "baseline_official_ap": baseline_scene[scene]["official_ap"],
            "final_official_ap": final_scene[scene]["official_ap"],
            "official_ap_delta": delta,
            "threshold_ap_delta": per_threshold,
        }

    baseline_ap = _objective(baseline_records)
    final_ap = _objective(final_records)
    if abs(final_ap - construction["official_ap"]) > 1e-10:
        raise ValueError("final replay does not reproduce construction official AP")
    payload = {
        "diagnostic_type": "GT-only C1c stability summary; no candidate materialization",
        "proposal_materialization_applied": False,
        "official_ap_thresholds": list(OFFICIAL),
        "ap25_threshold": AP25,
        "extra_diagnostic_thresholds": [0.95],
        "scene_count": len(scenes),
        "baseline_coexist_official_ap": baseline_ap,
        "final_official_ap": final_ap,
        "official_ap_delta": final_ap - baseline_ap,
        "baseline_threshold_ap": baseline_threshold,
        "final_threshold_ap": final_threshold,
        "threshold_ap_delta": threshold_delta,
        "official_scene_direction_counts": _direction_counts(scene_deltas),
        "official_scene_delta_mean": float(np.mean(scene_deltas)),
        "official_scene_delta_median": float(np.median(scene_deltas)),
        "threshold_scene_direction_counts": {
            tag: _direction_counts(values) for tag, values in threshold_scene_deltas.items()
        },
        "component_count": total_components,
        "changed_component_count_from_coexist": changed_components,
        "selected_action_counts": dict(sorted(action_counts.items())),
        "scenes": scenes_payload,
        "params": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        key: payload[key] for key in (
            "baseline_coexist_official_ap", "final_official_ap", "official_ap_delta",
            "official_scene_direction_counts", "threshold_ap_delta",
            "threshold_scene_direction_counts", "changed_component_count_from_coexist",
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

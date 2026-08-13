#!/usr/bin/env python3
"""Validate C1d attribution branches and isolate R-only AP25 trade-offs.

This GT-only post-hoc diagnostic replays the frozen evaluator records.  It
does not select new actions: branch states are read verbatim from the C1d
attribution summary.  For every selected R-only replacement it additionally
computes a one-action reversion margin against the final R-only construction.
Those margins are non-additive influence diagnostics, not new selections.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_track_native_competition_ledger import _read_scenes
from construct_c1c_global_feasible_ap_oracle_gt import AP25, DIAGNOSTIC, OFFICIAL, _build_scene_cache, _objective, _summary_records
from construct_c1d_attribution_oracle_gt import (
    FROZEN_SCORE, _assign_global_quality_ranks, _baseline_state,
    _quality_by_uuid, _records_for_state,
)
from construct_c1d_global_feasible_replacement_oracle_gt import _canonical_native_map, _read_actions
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval

EPS = 1e-12


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _normal_state(state: dict) -> dict[int, str]:
    return {int(component_id): str(action_name) for component_id, action_name in state.items()}


def _action_row(actions: dict[int, list[dict]], component_id: int, action_name: str) -> dict:
    return next(row for row in actions[component_id] if row["action_name"] == action_name)


def _branch_records(branch: dict, scenes, actions, caches, canonical):
    records, details = {}, {}
    for scene in scenes:
        state = _normal_state(branch["state"][scene])
        records[scene], details[scene] = _records_for_state(
            scene, actions[scene], state, caches[scene], canonical[scene],
            branch["native_geometry_mode"], branch["score_mode"],
        )
    return records, details


def _delta_map(reference: dict[str, float], value: dict[str, float]) -> dict[str, float]:
    return {key: float(value[key] - reference[key]) for key in value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--attribution-summary", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--native-fold-audit-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in (
        "scene_list", "attribution_summary", "relation_ledger_root", "action_ledger_root",
        "native_fold_audit_root", "filtered_d2b_track_root", "native_prediction_cache",
        "gt_instance_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise SystemExit("--max-scenes 必须为正数")
        scenes = scenes[:args.max_scenes]
    if not scenes:
        raise SystemExit("场景列表为空")
    attribution = json.loads(args.attribution_summary.read_text())
    if int(attribution["scene_count"]) < len(scenes):
        raise ValueError("归因摘要不覆盖所请求的场景")
    by_branch = {row["branch"]: row for row in attribution["results"]}
    required = {"F-only", "R-only", "Q-only", "F+R", "R+Q", "F+R+Q"}
    if set(by_branch) != required:
        raise ValueError(f"归因分支不完整：{sorted(by_branch)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        actions = {scene: _read_actions(args.action_ledger_root, scene) for scene in scenes}
        caches = {scene: _build_scene_cache(scene, args) for scene in scenes}
        for cache in caches.values():
            cache["gt_quality_by_uuid"] = _quality_by_uuid(cache)
        _assign_global_quality_ranks(caches)
        canonical = {}
        for scene in scenes:
            scores = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy"), dtype=np.float64)
            canonical[scene] = _canonical_native_map(scene, args.native_fold_audit_root, scores)
        baseline_states = {scene: _baseline_state(actions[scene]) for scene in scenes}
        baseline_records, baseline_details = {}, {}
        for scene in scenes:
            baseline_records[scene], baseline_details[scene] = _records_for_state(
                scene, actions[scene], baseline_states[scene], caches[scene], canonical[scene],
                "raw", FROZEN_SCORE,
            )
        baseline_threshold_ap = _summary_records(baseline_records)
        baseline_official_ap = _objective(baseline_records)
        branch_payload = {}
        branch_records = {}
        for name, branch in by_branch.items():
            records, details = _branch_records(branch, scenes, actions, caches, canonical)
            branch_records[name] = records
            threshold_ap = _summary_records(records)
            branch_payload[name] = {
                "official_ap": _objective(records),
                "threshold_ap": threshold_ap,
                "delta_vs_raw_frozen_baseline": {
                    "official_ap": _objective(records) - baseline_official_ap,
                    "threshold_ap": _delta_map(baseline_threshold_ap, threshold_ap),
                },
                "scene_candidate_counts": details,
            }
        r_branch = by_branch["R-only"]
        r_records = branch_records["R-only"]
        scene_rows = []
        official_counter = Counter()
        ap25_counter = Counter()
        for scene in scenes:
            base_official = _objective({scene: baseline_records[scene]})
            value_official = _objective({scene: r_records[scene]})
            base_ap25 = _summary_records({scene: baseline_records[scene]})["25"]
            value_ap25 = _summary_records({scene: r_records[scene]})["25"]
            delta_official = value_official - base_official
            delta_ap25 = value_ap25 - base_ap25
            official_counter["positive" if delta_official > EPS else "negative" if delta_official < -EPS else "zero"] += 1
            ap25_counter["positive" if delta_ap25 > EPS else "negative" if delta_ap25 < -EPS else "zero"] += 1
            scene_rows.append({
                "scene_name": scene,
                "baseline_official_ap": base_official,
                "r_only_official_ap": value_official,
                "delta_official_ap": delta_official,
                "baseline_ap25": base_ap25,
                "r_only_ap25": value_ap25,
                "delta_ap25": delta_ap25,
            })
        r_states = {scene: _normal_state(r_branch["state"][scene]) for scene in scenes}
        r_global_ap25 = branch_payload["R-only"]["threshold_ap"]["25"]
        r_global_official = branch_payload["R-only"]["official_ap"]
        action_rows = []
        for scene in scenes:
            current = r_states[scene]
            for component_id, action_name in current.items():
                baseline_name = baseline_states[scene][component_id]
                if action_name == baseline_name:
                    continue
                trial_state = dict(current)
                trial_state[component_id] = baseline_name
                trial, _ = _records_for_state(
                    scene, actions[scene], trial_state, caches[scene], canonical[scene],
                    "raw", FROZEN_SCORE,
                )
                combined = dict(r_records)
                combined[scene] = trial
                trial_ap25 = _summary_records(combined)["25"]
                trial_official = _objective(combined)
                selected = _action_row(actions[scene], component_id, action_name)
                action_rows.append({
                    "scene_name": scene,
                    "component_id": component_id,
                    "action_name": action_name,
                    "action_kind": selected["action_kind"],
                    "delta_ap25_if_kept_vs_reverted": r_global_ap25 - trial_ap25,
                    "delta_official_ap_if_kept_vs_reverted": r_global_official - trial_official,
                    "kept_native_candidate_count": len(selected["kept_native_candidate_ids"]),
                    "kept_track_count": len(selected["kept_track_ids"]),
                })
        action_rows.sort(key=lambda row: (row["delta_ap25_if_kept_vs_reverted"], row["scene_name"], row["component_id"]))
        by_kind, by_scene = defaultdict(lambda: {"count": 0, "ap25_margin_sum": 0.0, "official_margin_sum": 0.0}), defaultdict(lambda: {"count": 0, "ap25_margin_sum": 0.0, "official_margin_sum": 0.0})
        for row in action_rows:
            for group, key in ((by_kind, row["action_kind"]), (by_scene, row["scene_name"])):
                group[key]["count"] += 1
                group[key]["ap25_margin_sum"] += row["delta_ap25_if_kept_vs_reverted"]
                group[key]["official_margin_sum"] += row["delta_official_ap_if_kept_vs_reverted"]
        (args.output_root / "r_only_selected_action_reversion_margins.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in action_rows)
        )
        (args.output_root / "r_only_scene_stability.json").write_text(
            json.dumps(scene_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        payload = {
            "diagnostic_type": "GT-only C1d attribution replay and R-only stability/reversion margins",
            "not_mathematical_ap_upper_bound": True,
            "proposal_materialization_applied": False,
            "inference_rule_or_score_created": False,
            "official_ap_thresholds": list(OFFICIAL),
            "ap25_threshold": AP25,
            "extra_diagnostic_thresholds": [.95],
            "scene_count": len(scenes),
            "raw_frozen_baseline": {"official_ap": baseline_official_ap, "threshold_ap": baseline_threshold_ap, "scene_candidate_counts": baseline_details},
            "branches": branch_payload,
            "r_only_scene_stability": {
                "official_ap_sign_counts": dict(official_counter),
                "ap25_sign_counts": dict(ap25_counter),
                "rows_path": "r_only_scene_stability.json",
            },
            "r_only_ap25_action_reversion": {
                "action_count": len(action_rows),
                "harmful_if_kept_count": sum(row["delta_ap25_if_kept_vs_reverted"] < -EPS for row in action_rows),
                "beneficial_if_kept_count": sum(row["delta_ap25_if_kept_vs_reverted"] > EPS for row in action_rows),
                "neutral_if_kept_count": sum(abs(row["delta_ap25_if_kept_vs_reverted"]) <= EPS for row in action_rows),
                "by_action_kind_nonadditive": dict(sorted(by_kind.items())),
                "by_scene_nonadditive": dict(sorted(by_scene.items())),
                "five_most_negative_actions": action_rows[:5],
                "rows_path": "r_only_selected_action_reversion_margins.jsonl",
                "interpretation": "Each margin reverts one final R-only action while all others remain fixed; margins are not additive.",
            },
            "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        }
        (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    print(json.dumps({
        "r_only_official_ap": payload["branches"]["R-only"]["official_ap"],
        "r_only_delta_ap": payload["branches"]["R-only"]["delta_vs_raw_frozen_baseline"]["official_ap"],
        "r_only_scene_signs": payload["r_only_scene_stability"]["official_ap_sign_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

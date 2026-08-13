#!/usr/bin/env python3
"""Five-fold C1d-R GVC-only replacement audit.

This is deliberately a small, fixed no-training audit.  It uses public-view
source-frame-excluded relative GVC and/or the frozen raw score difference to
choose only pre-registered C1d replacement actions.  Missing public GVC means
coexist.  Each fold chooses a direction and one of three fixed quantile
thresholds on its other four scene folds, then evaluates a frozen plan on its
held-out scenes.  No proposal, score, NMS, or model is materialized.
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
from construct_c1d_attribution_oracle_gt import FROZEN_SCORE, _baseline_state, _records_for_state
from construct_c1d_global_feasible_replacement_oracle_gt import _canonical_native_map, _read_actions
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval

EPS = 1e-12
QUANTILES = (.50, .75, .90)
FEATURES = ("gvc_only", "raw_score_difference_only", "equal_gvc_raw_score_difference")
FAMILIES = ("strict_coverage_replacement", "track_only_one", "track_only_all")
KIND_TO_FAMILY = {
    "replace_native_covered_099_with_track": "strict_coverage_replacement",
    "replace_native_covered_099_with_all_tracks": "strict_coverage_replacement",
    "track_only_one": "track_only_one",
    "track_only_all": "track_only_all",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _normal_state(state: dict) -> dict[int, str]:
    return {int(component_id): str(action_name) for component_id, action_name in state.items()}


def _score_differences(scene: str, cache: dict, relation_root: Path, native_cache: Path) -> dict[int, float | None]:
    """Track minus max-overlapping-native frozen score, without score calibration."""
    native_scores = np.asarray(np.load(native_cache / f"{scene}_pred_scores.npy"), dtype=np.float64)
    by_track = defaultdict(list)
    for row in _jsonl(relation_root / scene / "track_native_relations.jsonl"):
        if float(row["point_iou"]) > 0.0:
            by_track[int(row["proposal_id"])].append(int(row["native_candidate_id"]))
    output = {}
    for track_id, track in cache["tracks"].items():
        natives = by_track.get(track_id, [])
        output[track_id] = (
            float(track["mean_node_quality"]) - float(np.max(native_scores[natives])) if natives else None
        )
    return output


def _relative_features(scene: str, relative_root: Path, score_diff: dict[int, float | None]) -> dict[int, dict]:
    output = {}
    for row in _jsonl(relative_root / scene / "track_relative_gvc.jsonl"):
        track_id = int(row["track_id"])
        pair = row["public_comparison_pair"]
        gvc = None if pair is None else float(pair["track_minus_native_gvc"])
        output[track_id] = {
            "gvc_difference": gvc,
            "raw_score_difference": score_diff.get(track_id),
            "public_comparison_available": row["comparison_state"] == "relative_public_evidence_available",
            "comparison_state": row["comparison_state"],
        }
    return output


def _fit_standardizer(values: list[float]) -> tuple[float, float]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(values):
        raise ValueError("训练折没有可用特征")
    mean, std = float(values.mean()), float(values.std())
    return mean, std if std > 1e-12 else 1.0


def _track_value(feature: str, row: dict, standardizer: dict | None) -> float | None:
    gvc, raw = row["gvc_difference"], row["raw_score_difference"]
    if feature == "gvc_only":
        return gvc
    if feature == "raw_score_difference_only":
        return raw
    if feature == "equal_gvc_raw_score_difference":
        if gvc is None or raw is None or standardizer is None:
            return None
        return .5 * ((gvc - standardizer["gvc"][0]) / standardizer["gvc"][1] + (raw - standardizer["raw"][0]) / standardizer["raw"][1])
    raise ValueError(feature)


def _action_value(action: dict, track_features: dict[int, dict], feature: str, standardizer: dict | None) -> float | None:
    values = []
    for track_id in action["kept_track_ids"]:
        row = track_features.get(int(track_id))
        # Missing independent public GVC always forces coexist, including in score-only controls.
        if row is None or not row["public_comparison_available"]:
            return None
        value = _track_value(feature, row, standardizer)
        if value is None or not np.isfinite(value):
            return None
        values.append(value)
    if not values:
        return None
    # An all-track replacement is admissible only if every retained track has evidence;
    # its weakest member determines the conservative action evidence.
    return float(min(values)) if len(values) > 1 else float(values[0])


def _plan_scene(actions: dict[int, list[dict]], track_features: dict[int, dict], family: str, feature: str, standardizer: dict | None, direction: str, threshold: float):
    state = _baseline_state(actions)
    decisions = []
    sign = 1.0 if direction == "higher" else -1.0
    for component_id, rows in actions.items():
        candidates = []
        for row in rows:
            if KIND_TO_FAMILY.get(row["action_kind"]) != family:
                continue
            value = _action_value(row, track_features, feature, standardizer)
            if value is not None:
                candidates.append((sign * value, value, row))
        if not candidates:
            continue
        _, value, selected = max(candidates, key=lambda item: (item[0], item[2]["action_name"]))
        if sign * value >= sign * threshold - EPS:
            state[component_id] = selected["action_name"]
            decisions.append({
                "component_id": component_id, "action_name": selected["action_name"],
                "action_kind": selected["action_kind"], "feature_value": value,
            })
    return state, decisions


def _training_values(scenes, actions, features, family, feature, standardizer):
    values = []
    for scene in scenes:
        for rows in actions[scene].values():
            for row in rows:
                if KIND_TO_FAMILY.get(row["action_kind"]) == family:
                    value = _action_value(row, features[scene], feature, standardizer)
                    if value is not None:
                        values.append(value)
    return values


def _records_for_plan(scenes, states, actions, caches, canonical):
    records, details = {}, {}
    for scene in scenes:
        records[scene], details[scene] = _records_for_state(
            scene, actions[scene], states[scene], caches[scene], canonical[scene], "raw", FROZEN_SCORE
        )
    return records, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relative-gvc-root", type=Path, required=True)
    parser.add_argument("--attribution-summary", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--native-fold-audit-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in (
        "scene_list", "relative_gvc_root", "attribution_summary", "relation_ledger_root", "action_ledger_root",
        "native_fold_audit_root", "filtered_d2b_track_root", "native_prediction_cache", "gt_instance_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.fold_count != 5:
        raise SystemExit("本审计固定为五折场景隔离")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    if len(scenes) < args.fold_count:
        raise SystemExit("场景数不足五折")
    attribution = json.loads(args.attribution_summary.read_text())
    r_state = {scene: _normal_state(state) for scene, state in next(row for row in attribution["results"] if row["branch"] == "R-only")["state"].items()}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        actions = {scene: _read_actions(args.action_ledger_root, scene) for scene in scenes}
        caches = {scene: _build_scene_cache(scene, args) for scene in scenes}
        canonical = {}
        feature_rows = {}
        for scene in scenes:
            scores = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy"), dtype=np.float64)
            canonical[scene] = _canonical_native_map(scene, args.native_fold_audit_root, scores)
            feature_rows[scene] = _relative_features(
                scene, args.relative_gvc_root,
                _score_differences(scene, caches[scene], args.relation_ledger_root, args.native_prediction_cache),
            )
        baseline_states = {scene: _baseline_state(actions[scene]) for scene in scenes}
        baseline_records, baseline_details = _records_for_plan(scenes, baseline_states, actions, caches, canonical)
        baseline_ap = _objective(baseline_records)
        baseline_thresholds = _summary_records(baseline_records)
        feature_ledger, fold_rows, oof_rows = [], [], []
        for scene_index, scene in enumerate(scenes):
            for component_id, rows in actions[scene].items():
                for row in rows:
                    family = KIND_TO_FAMILY.get(row["action_kind"])
                    if family is None:
                        continue
                    track_id = row.get("selected_track_id")
                    if track_id is None and len(row["kept_track_ids"]) == 1:
                        track_id = row["kept_track_ids"][0]
                    track_info = feature_rows[scene].get(int(track_id)) if track_id is not None else None
                    feature_ledger.append({
                        "scene_name": scene, "scene_fold": scene_index % args.fold_count,
                        "component_id": component_id, "action_name": row["action_name"], "action_family": family,
                        "selected_track_id": track_id,
                        "public_comparison_available": None if track_info is None else track_info["public_comparison_available"],
                        "gvc_difference": None if track_info is None else track_info["gvc_difference"],
                        "raw_score_difference": None if track_info is None else track_info["raw_score_difference"],
                        "r_only_full_construct_selected": r_state.get(scene, {}).get(component_id) == row["action_name"],
                    })
        for feature in FEATURES:
            for family in FAMILIES:
                heldout_records = {}
                for fold in range(args.fold_count):
                    train_scenes = [scene for index, scene in enumerate(scenes) if index % args.fold_count != fold]
                    test_scenes = [scene for index, scene in enumerate(scenes) if index % args.fold_count == fold]
                    standardizer = None
                    if feature == "equal_gvc_raw_score_difference":
                        train_gvc, train_raw = [], []
                        for scene in train_scenes:
                            for row in feature_rows[scene].values():
                                if row["public_comparison_available"] and row["gvc_difference"] is not None and row["raw_score_difference"] is not None:
                                    train_gvc.append(row["gvc_difference"]); train_raw.append(row["raw_score_difference"])
                        standardizer = {"gvc": _fit_standardizer(train_gvc), "raw": _fit_standardizer(train_raw)}
                    values = _training_values(train_scenes, actions, feature_rows, family, feature, standardizer)
                    if not values:
                        raise ValueError(f"fold={fold} {feature} {family}: 无可用训练动作特征")
                    candidates = []
                    for direction in ("higher", "lower"):
                        for quantile in QUANTILES:
                            threshold = float(np.quantile(values, quantile))
                            states, _ = {}, {}
                            for scene in train_scenes:
                                states[scene], _ = _plan_scene(actions[scene], feature_rows[scene], family, feature, standardizer, direction, threshold)
                            records, _ = _records_for_plan(train_scenes, states, actions, caches, canonical)
                            candidates.append((_objective(records), direction, quantile, threshold))
                    train_ap, direction, quantile, threshold = max(candidates, key=lambda row: (row[0], row[1] == "higher", -row[2]))
                    test_states, decisions = {}, {}
                    for scene in test_scenes:
                        test_states[scene], decisions[scene] = _plan_scene(actions[scene], feature_rows[scene], family, feature, standardizer, direction, threshold)
                    records, details = _records_for_plan(test_scenes, test_states, actions, caches, canonical)
                    heldout_records.update(records)
                    test_ap, test_threshold = _objective(records), _summary_records(records)
                    fold_rows.append({
                        "feature": feature, "action_family": family, "fold": fold,
                        "train_scene_count": len(train_scenes), "test_scene_count": len(test_scenes),
                        "training_selected_direction": direction, "training_selected_quantile": quantile,
                        "training_selected_threshold": threshold, "training_official_ap": train_ap,
                        "held_out_official_ap": test_ap, "held_out_threshold_ap": test_threshold,
                        "held_out_selected_action_count": sum(len(rows) for rows in decisions.values()),
                    })
                    for scene, rows in decisions.items():
                        for row in rows:
                            oof_rows.append({"feature": feature, "action_family": family, "fold": fold, "scene_name": scene, **row})
                total_threshold = _summary_records(heldout_records)
                # Keep an aggregate row after all five held-out partitions, not a mean of fold APs.
                fold_rows.append({
                    "feature": feature, "action_family": family, "fold": "oof_aggregate",
                    "held_out_official_ap": _objective(heldout_records), "held_out_threshold_ap": total_threshold,
                    "delta_vs_raw_frozen_baseline": {
                        "official_ap": _objective(heldout_records) - baseline_ap,
                        "threshold_ap": {key: total_threshold[key] - baseline_thresholds[key] for key in total_threshold},
                    },
                    "held_out_selected_action_count": sum(1 for row in oof_rows if row["feature"] == feature and row["action_family"] == family),
                })
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("c1dr_gvc_action_features.jsonl", feature_ledger), ("fold_results.jsonl", fold_rows), ("oof_actions.jsonl", oof_rows)):
        (args.output_root / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    aggregate = [row for row in fold_rows if row["fold"] == "oof_aggregate"]
    payload = {
        "diagnostic_type": "GT-only C1d-R GVC-only replacement five-fold AP audit",
        "proposal_materialization_applied": False, "model_training_applied": False,
        "missing_public_relative_gvc_action": "coexist",
        "official_ap_thresholds": list(OFFICIAL), "ap25_threshold": AP25, "extra_diagnostic_thresholds": [.95],
        "fold_count": args.fold_count, "scene_count": len(scenes),
        "fixed_features": list(FEATURES), "fixed_action_families": list(FAMILIES),
        "fixed_training_quantiles": list(QUANTILES),
        "baseline": {"official_ap": baseline_ap, "threshold_ap": baseline_thresholds, "candidate_counts": baseline_details},
        "oof_aggregate": aggregate,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({f"{row['feature']}::{row['action_family']}": row["delta_vs_raw_frozen_baseline"]["official_ap"] for row in aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build GT-only global-AP action margins around a completed C1c construction.

For every pre-registered component action, this tool replaces only that
component in the frozen final C1c state and recomputes the single shared
global official AP.  It therefore distinguishes strict improvements, neutral
equivalences, and harmful alternatives without turning the final action itself
into a binary learning label.  No prediction is materialized or modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_track_native_competition_ledger import _read_scenes
from construct_c1c_global_feasible_ap_oracle_gt import (
    OFFICIAL,
    _build_scene_cache,
    _objective,
    _read_actions,
    _scene_records,
)
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval

EPS = 1e-10


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _action_track_ids(row: dict) -> set[int]:
    kind = row["action_kind"]
    if kind in ("coexist", "keep_all_tracks"):
        return {int(value) for value in row["component_track_ids"]}
    if kind in ("native_plus_one_track", "keep_one_track"):
        return {int(row["selected_track_id"])}
    if kind in ("native_only", "suppress_all_tracks"):
        return set()
    raise ValueError(f"未知动作类型：{kind}")


def _decision(delta: float) -> str:
    if delta > EPS:
        return "strict_positive"
    if delta < -EPS:
        return "harmful"
    return "neutral"


def _track_label(track_id: int, rows: list[dict], final_name: str) -> dict:
    final = next(row for row in rows if row["action_name"] == final_name)
    final_has_track = track_id in _action_track_ids(final)
    alternatives = [
        row for row in rows
        if row["action_name"] != final_name
        and ((track_id not in _action_track_ids(row)) if final_has_track else (track_id in _action_track_ids(row)))
    ]
    if not alternatives:
        return {
            "track_id": track_id, "final_state_keeps_track": final_has_track,
            "label": "neutral", "counterfactual_best_delta": None,
            "equivalent_alternative_count": 0,
        }
    best_delta = max(float(row["global_official_ap_delta"]) for row in alternatives)
    equivalent_count = sum(abs(float(row["global_official_ap_delta"])) <= EPS for row in alternatives)
    if equivalent_count:
        label = "one_of_equivalent"
    elif final_has_track and best_delta < -EPS:
        label = "necessary_keep"
    elif not final_has_track and best_delta < -EPS:
        label = "beneficial_suppress"
    else:
        label = "neutral"
    return {
        "track_id": track_id,
        "final_state_keeps_track": final_has_track,
        "label": label,
        "counterfactual_best_delta": best_delta,
        "equivalent_alternative_count": equivalent_count,
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
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in (
        "construction_summary", "scene_list", "relation_ledger_root", "action_ledger_root",
        "filtered_d2b_track_root", "native_prediction_cache", "gt_instance_dir", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    construction = json.loads(args.construction_summary.read_text())
    if construction.get("official_ap_thresholds") != list(OFFICIAL):
        raise ValueError("C1c 构造的正式 AP 阈值与 evaluator 不一致")
    all_scenes = _read_scenes(args.scene_list)
    scenes = all_scenes if args.max_scenes is None else all_scenes[:args.max_scenes]
    final_state = construction["final_component_actions"]
    if any(scene not in final_state for scene in scenes):
        raise ValueError("构造结果缺少场景最终动作")
    args.output_root.mkdir(parents=True)
    _configure_scannet200_instance_eval()
    original = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original(path))
    try:
        actions = {scene: _read_actions(args.action_ledger_root, scene) for scene in scenes}
        states = {
            scene: {int(component): action for component, action in final_state[scene].items()}
            for scene in scenes
        }
        if any(set(states[scene]) != set(actions[scene]) for scene in scenes):
            raise ValueError("最终动作与预注册组件不一致")
        caches = {scene: _build_scene_cache(scene, args) for scene in scenes}
        final_records, final_details = {}, {}
        for scene in scenes:
            final_records[scene], final_details[scene] = _scene_records(
                scene, actions[scene], states[scene], caches[scene]
            )
        final_ap = _objective(final_records)
        if args.max_scenes is None and abs(final_ap - float(construction["official_ap"])) > EPS:
            raise ValueError("最终状态重放未复现 C1c AP")
        component_count = 0
        action_count = 0
        track_label_counts: Counter[str] = Counter()
        scene_summaries = []
        for scene_index, scene in enumerate(scenes, start=1):
            scene_rows, track_rows = [], []
            for component_id in sorted(actions[scene]):
                component_count += 1
                final_name = states[scene][component_id]
                evaluated = []
                for candidate in actions[scene][component_id]:
                    candidate_name = candidate["action_name"]
                    trial_state = dict(states[scene])
                    trial_state[component_id] = candidate_name
                    trial_records, trial_detail = _scene_records(
                        scene, actions[scene], trial_state, caches[scene]
                    )
                    all_records = dict(final_records)
                    all_records[scene] = trial_records
                    trial_ap = _objective(all_records)
                    row = {
                        "scene_name": scene,
                        "component_id": component_id,
                        "action_name": candidate_name,
                        "action_kind": candidate["action_kind"],
                        "selected_track_id": candidate.get("selected_track_id"),
                        "component_track_ids": candidate["component_track_ids"],
                        "component_native_candidate_ids": candidate["component_native_candidate_ids"],
                        "is_c1c_final_action": candidate_name == final_name,
                        "global_official_ap_after_action": trial_ap,
                        "global_official_ap_delta": trial_ap - final_ap,
                        "decision": _decision(trial_ap - final_ap),
                        "candidate_counts": trial_detail,
                    }
                    evaluated.append(row)
                    action_count += 1
                ordered = sorted(evaluated, key=lambda row: (-row["global_official_ap_after_action"], row["action_name"]))
                best, second = ordered[0], ordered[1] if len(ordered) > 1 else None
                for row in evaluated:
                    row["component_best_action_name"] = best["action_name"]
                    row["component_best_minus_second_margin"] = (
                        best["global_official_ap_after_action"] - second["global_official_ap_after_action"]
                        if second is not None else None
                    )
                    row["component_final_action_name"] = final_name
                    scene_rows.append(row)
                for track_id in sorted({int(track) for row in evaluated for track in row["component_track_ids"]}):
                    label = _track_label(track_id, evaluated, final_name)
                    label.update({"scene_name": scene, "component_id": component_id})
                    track_rows.append(label)
                    track_label_counts[label["label"]] += 1
            scene_root = args.output_root / scene
            scene_root.mkdir()
            (scene_root / "component_action_margins.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scene_rows)
            )
            (scene_root / "track_action_labels.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in track_rows)
            )
            summary = {
                "scene_name": scene,
                "component_count": len(actions[scene]),
                "action_count": len(scene_rows),
                "track_label_count": len(track_rows),
                "final_candidate_counts": final_details[scene],
            }
            (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            scene_summaries.append(summary)
            print(f"[场景完成] {scene_index}/{len(scenes)} {scene}: {len(scene_rows)} 条动作边际", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original
    payload = {
        "diagnostic_type": "GT-only C1c global official-AP single-component action margins",
        "not_mathematical_ap_upper_bound": True,
        "proposal_materialization_applied": False,
        "official_ap_thresholds": list(OFFICIAL),
        "final_c1c_official_ap": final_ap,
        "scene_count": len(scenes),
        "component_count": component_count,
        "action_count": action_count,
        "track_label_counts": dict(sorted(track_label_counts.items())),
        "scenes": scene_summaries,
        "params": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "final_c1c_official_ap", "scene_count", "component_count", "action_count", "track_label_counts",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

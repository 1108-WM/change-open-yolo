#!/usr/bin/env python3
"""GT-only C1d global feasible AP construction for folding and replacement.

This diagnostic keeps raw-native and exact-native-folded representations
separate.  Within either representation it only selects pre-registered C1d
component actions by global official AP coordinate ascent.  It is not a
mathematical AP upper bound and never materializes a prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_track_native_competition_ledger import _read_scenes
from construct_c1c_global_feasible_ap_oracle_gt import (
    AP25, DIAGNOSTIC, OFFICIAL, _build_scene_cache, _objective, _record_from_matches, _summary_records,
)
from diagnose_gvc_class_agnostic_ap import _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval

EPS = 1e-12


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_actions(root: Path, scene: str) -> dict[int, list[dict]]:
    rows = [json.loads(line) for line in (root / scene / "c1d_expanded_actions.jsonl").read_text().splitlines() if line]
    result = {}
    for row in rows:
        result.setdefault(int(row["component_id"]), []).append(row)
    return {key: sorted(value, key=lambda row: row["action_name"]) for key, value in result.items()}


def _canonical_native_map(scene: str, audit_root: Path, native_scores: np.ndarray) -> dict[int, int]:
    result = {index: index for index in range(len(native_scores))}
    for line in (audit_root / scene / "exact_geometry_groups.jsonl").read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        candidates = [int(value) for value in row["candidate_ids"]]
        representative = max(candidates, key=lambda item: (float(native_scores[item]), -item))
        for candidate in candidates:
            result[candidate] = representative
    return result


def _initial_state(actions: dict[int, list[dict]], start: str) -> dict[int, str]:
    state = {}
    for component_id, rows in actions.items():
        by_kind = {}
        for row in rows:
            by_kind.setdefault(row["action_kind"], []).append(row)
        if start == "coexist":
            pick = by_kind.get("coexist", by_kind.get("keep_all_tracks"))[0]
        elif start == "native_only":
            pick = by_kind.get("native_only", by_kind.get("suppress_all_tracks"))[0]
        elif start == "track_only_one":
            pick = (by_kind.get("track_only_one") or by_kind.get("keep_one_track")
                    or by_kind.get("coexist") or by_kind.get("keep_all_tracks"))[0]
        else:
            raise ValueError(start)
        state[component_id] = pick["action_name"]
    return state


def _state_candidate_sets(actions, state, native_count, canonical, native_mode):
    if native_mode == "raw":
        kept_native = set(range(native_count))
    elif native_mode == "exact_folded":
        kept_native = set(canonical.values())
    else:
        raise ValueError(native_mode)
    kept_track = set()
    controlled = {}
    for component_id, rows in actions.items():
        raw_native = {int(value) for value in rows[0]["component_native_candidate_ids"]}
        canonical_native = {canonical[value] for value in raw_native}
        for value in canonical_native:
            previous = controlled.setdefault(value, component_id)
            if previous != component_id:
                raise ValueError(f"exact native representative {value} is controlled by components {previous}/{component_id}")
        kept_native.difference_update(canonical_native)
        selected = next(row for row in rows if row["action_name"] == state[component_id])
        kept_native.update(canonical[int(value)] for value in selected["kept_native_candidate_ids"])
        kept_track.update(int(value) for value in selected["kept_track_ids"])
    return kept_native, kept_track


def _scene_records(scene, actions, state, cache, canonical, native_mode):
    kept_native, kept_track = _state_candidate_sets(
        actions, state, cache["native_count"], canonical, native_mode
    )
    uuid_by_candidate = cache["uuid_by_candidate"]
    keep_uuid = {uuid_by_candidate[("native", item)] for item in kept_native if ("native", item) in uuid_by_candidate}
    keep_uuid.update(uuid_by_candidate[("track", item)] for item in kept_track if ("track", item) in uuid_by_candidate)
    pred = {"chair": [row for row in cache["pred"]["chair"] if row["uuid"] in keep_uuid]}
    gt = {"chair": [dict(row, matched_pred=[match for match in row["matched_pred"] if match["uuid"] in keep_uuid]) for row in cache["gt"]["chair"]]}
    records = {str(int(round(threshold * 100))): _record_from_matches(gt, pred, threshold) for threshold in DIAGNOSTIC}
    return records, {"kept_native_count": len(kept_native), "kept_track_count": len(kept_track)}


def _run_mode(mode, scenes, actions, caches, canonical, max_sweeps):
    outcomes = []
    for start in ("coexist", "native_only", "track_only_one"):
        print(f"[开始] native={mode}, 起点={start}, 场景={len(scenes)}", flush=True)
        state = {scene: _initial_state(actions[scene], start) for scene in scenes}
        records, details = {}, {}
        for scene in scenes:
            records[scene], details[scene] = _scene_records(scene, actions[scene], state[scene], caches[scene], canonical[scene], mode)
        initial_ap = _objective(records)
        score, history = initial_ap, []
        for sweep in range(max_sweeps):
            accepted = 0
            for scene in scenes:
                for component_id in sorted(actions[scene]):
                    old = state[scene][component_id]
                    for candidate in actions[scene][component_id]:
                        name = candidate["action_name"]
                        if name == old:
                            continue
                        state[scene][component_id] = name
                        trial, trial_detail = _scene_records(scene, actions[scene], state[scene], caches[scene], canonical[scene], mode)
                        merged = dict(records); merged[scene] = trial
                        value = _objective(merged)
                        if value > score + EPS:
                            score, records[scene], details[scene], old = value, trial, trial_detail, name
                            accepted += 1
                        else:
                            state[scene][component_id] = old
            history.append({"sweep": sweep + 1, "accepted_action_count": accepted, "global_official_ap": score})
            print(
                f"[轮次完成] native={mode}, 起点={start}, 轮次={sweep + 1}, "
                f"接受={accepted}, 正式AP={score:.9f}",
                flush=True,
            )
            if not accepted:
                break
        final_records, final_details = {}, {}
        for scene in scenes:
            final_records[scene], final_details[scene] = _scene_records(scene, actions[scene], state[scene], caches[scene], canonical[scene], mode)
        outcomes.append({
            "native_geometry_mode": mode, "start": start, "initial_official_ap": initial_ap,
            "official_ap": _objective(final_records), "threshold_ap": _summary_records(final_records),
            "history": history, "state": state, "scene_candidate_counts": final_details,
        })
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--native-fold-audit-root", type=Path, required=True)
    parser.add_argument("--filtered-d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sweeps", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, help="仅用于可恢复的 smoke 验证；默认运行完整场景列表")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in ("scene_list", "relation_ledger_root", "action_ledger_root", "native_fold_audit_root", "filtered_d2b_track_root", "native_prediction_cache", "gt_instance_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_sweeps < 1:
        raise SystemExit("--max-sweeps 必须为正数")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise SystemExit("--max-scenes 必须为正数")
        scenes = scenes[:args.max_scenes]
    if not scenes:
        raise SystemExit("场景列表为空")
    # An earlier interrupted preflight may have created an empty directory.
    # The non-empty guard above still prevents overwriting any diagnostic.
    args.output_root.mkdir(parents=True, exist_ok=True)
    _configure_scannet200_instance_eval()
    original = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original(path))
    try:
        actions = {scene: _read_actions(args.action_ledger_root, scene) for scene in scenes}
        caches = {scene: _build_scene_cache(scene, args) for scene in scenes}
        canonical = {}
        for scene in scenes:
            scores = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy"), dtype=np.float64)
            canonical[scene] = _canonical_native_map(scene, args.native_fold_audit_root, scores)
        outcomes = []
        for mode in ("raw", "exact_folded"):
            outcomes.extend(_run_mode(mode, scenes, actions, caches, canonical, args.max_sweeps))
    finally:
        instance_eval.util_3d.load_ids = original
    best_by_mode = {}
    for mode in ("raw", "exact_folded"):
        rows = [row for row in outcomes if row["native_geometry_mode"] == mode]
        best_by_mode[mode] = max(rows, key=lambda row: (row["official_ap"], -("coexist", "native_only", "track_only_one").index(row["start"])))
    payload = {
        "diagnostic_type": "GT-only C1d global feasible replacement/folding AP construction",
        "not_mathematical_ap_upper_bound": True, "proposal_materialization_applied": False,
        "official_ap_thresholds": list(OFFICIAL), "ap25_threshold": AP25, "extra_diagnostic_thresholds": [.95],
        "scene_count": len(scenes), "max_sweeps": args.max_sweeps,
        "all_mode_start_results": outcomes, "best_by_native_geometry_mode": best_by_mode,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({mode: {"official_ap": row["official_ap"], "start": row["start"]} for mode, row in best_by_mode.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

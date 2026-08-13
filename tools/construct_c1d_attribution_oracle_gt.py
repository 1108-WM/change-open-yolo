#!/usr/bin/env python3
"""GT-only C1d attribution construction: folding, replacement, and ranking.

The six reported branches have fixed, non-overlapping meanings:

* F-only: exact-native geometry folding only;
* R-only: track replaces native in mixed components, frozen scores;
* Q-only: frozen candidate set, one threshold-independent GT-IoU quality rank;
* F+R, R+Q, F+R+Q: the corresponding combinations.

It never writes a proposal, inference score, or deployment decision.  The
coordinate constructions are feasible GT-guided diagnostics, not AP bounds.
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
    AP25, DIAGNOSTIC, OFFICIAL, _build_scene_cache, _objective,
    _record_from_matches, _summary_records,
)
from construct_c1d_global_feasible_replacement_oracle_gt import (
    _canonical_native_map, _read_actions, _state_candidate_sets,
)
from diagnose_gvc_class_agnostic_ap import (
    _class_agnostic_gt_ids, _configure_scannet200_instance_eval, instance_eval,
)

EPS = 1e-12
FROZEN_SCORE = "frozen_score"
GT_QUALITY_SCORE = "gt_best_iou_shared_across_thresholds"
REPLACEMENT_KINDS = {
    "track_only_all", "track_only_one",
    "replace_native_covered_099_with_track",
    "replace_native_covered_099_with_all_tracks",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _baseline_state(actions: dict[int, list[dict]]) -> dict[int, str]:
    """Frozen strict-filter input: coexist mixed components, all pure tracks."""
    result = {}
    for component_id, rows in actions.items():
        by_kind = {}
        for row in rows:
            by_kind.setdefault(row["action_kind"], []).append(row)
        result[component_id] = (
            by_kind.get("coexist", by_kind.get("keep_all_tracks"))[0]["action_name"]
        )
    return result


def _replacement_options(rows: list[dict], current: str) -> list[dict]:
    """Only true mixed-component native→track replacement alternatives."""
    if not rows[0]["component_native_candidate_ids"]:
        return []
    return [
        row for row in rows
        if row["action_name"] != current
        and row["action_kind"] in (REPLACEMENT_KINDS | {"coexist"})
    ]


def _quality_by_uuid(cache: dict) -> dict[str, float]:
    """One candidate-level GT quality, shared by all official thresholds."""
    quality = {row["uuid"]: 0.0 for row in cache["pred"]["chair"]}
    for gt in cache["gt"]["chair"]:
        if int(gt["instance_id"]) < 1000 or int(gt["vert_count"]) < 100:
            continue
        for pred in gt["matched_pred"]:
            uuid = pred["uuid"]
            if uuid not in quality:
                continue
            union = int(gt["vert_count"]) + int(pred["vert_count"]) - int(pred["intersection"])
            if union > 0:
                quality[uuid] = max(quality[uuid], float(pred["intersection"]) / union)
    return quality


def _assign_global_quality_ranks(caches: dict[str, dict]) -> None:
    """Make the shared oracle quality order strict without changing its primary key.

    Exact native geometry expansion creates many candidates with identical GT
    quality.  The evaluator treats equal confidence as one score group, which
    would measure score ties rather than a unified ranking.  A fixed scene /
    source / candidate-ID secondary key is no-GT and applies *only* after the
    GT-IoU primary key; it makes the requested ranking auditable and total.
    """
    entries = []
    for scene, cache in caches.items():
        by_uuid = {uuid: key for key, uuid in cache["uuid_by_candidate"].items()}
        for uuid, quality in cache["gt_quality_by_uuid"].items():
            source, candidate_id = by_uuid[uuid]
            entries.append((float(quality), str(scene), str(source), int(candidate_id), uuid))
    entries.sort()
    total = len(entries)
    for index, (_, scene, _, _, uuid) in enumerate(entries, start=1):
        caches[scene].setdefault("gt_quality_rank_by_uuid", {})[uuid] = float(index) / total


def _records_for_state(
    scene: str,
    actions: dict[int, list[dict]],
    state: dict[int, str],
    cache: dict,
    canonical: dict[int, int],
    native_mode: str,
    score_mode: str,
) -> tuple[dict, dict]:
    kept_native, kept_track = _state_candidate_sets(
        actions, state, cache["native_count"], canonical, native_mode
    )
    uuid_by_candidate = cache["uuid_by_candidate"]
    keep_uuid = {
        uuid_by_candidate[("native", item)] for item in kept_native
        if ("native", item) in uuid_by_candidate
    }
    keep_uuid.update(
        uuid_by_candidate[("track", item)] for item in kept_track
        if ("track", item) in uuid_by_candidate
    )
    if score_mode == FROZEN_SCORE:
        pred = {"chair": [row for row in cache["pred"]["chair"] if row["uuid"] in keep_uuid]}
        gt = {"chair": [
            dict(row, matched_pred=[match for match in row["matched_pred"] if match["uuid"] in keep_uuid])
            for row in cache["gt"]["chair"]
        ]}
    elif score_mode == GT_QUALITY_SCORE:
        quality = cache["gt_quality_rank_by_uuid"]
        pred_rows = []
        for row in cache["pred"]["chair"]:
            if row["uuid"] not in keep_uuid:
                continue
            pred_rows.append(dict(row, confidence=quality[row["uuid"]]))
        pred = {"chair": pred_rows}
        gt = {"chair": [
            dict(
                row,
                matched_pred=[
                    dict(match, confidence=quality[match["uuid"]])
                    for match in row["matched_pred"] if match["uuid"] in keep_uuid
                ],
            )
            for row in cache["gt"]["chair"]
        ]}
    else:
        raise ValueError(score_mode)
    records = {
        str(int(round(threshold * 100))): _record_from_matches(gt, pred, threshold)
        for threshold in DIAGNOSTIC
    }
    return records, {
        "kept_native_count": len(kept_native),
        "kept_track_count": len(kept_track),
    }


def _selection_counts(actions: dict[int, list[dict]], state: dict[int, str]) -> dict[str, int]:
    counts = Counter()
    for component_id, name in state.items():
        row = next(row for row in actions[component_id] if row["action_name"] == name)
        counts[row["action_kind"]] += 1
    return dict(sorted(counts.items()))


def _run_replacement(
    name: str,
    native_mode: str,
    score_mode: str,
    scenes: list[str],
    actions_by_scene: dict[str, dict[int, list[dict]]],
    caches: dict[str, dict],
    canonical: dict[str, dict[int, int]],
    max_sweeps: int,
) -> dict:
    states = {scene: _baseline_state(actions_by_scene[scene]) for scene in scenes}
    records, details = {}, {}
    for scene in scenes:
        records[scene], details[scene] = _records_for_state(
            scene, actions_by_scene[scene], states[scene], caches[scene], canonical[scene],
            native_mode, score_mode,
        )
    initial_ap = _objective(records)
    score, history = initial_ap, []
    print(f"[开始] {name}: 场景={len(scenes)}", flush=True)
    for sweep in range(max_sweeps):
        accepted = 0
        for scene in scenes:
            actions = actions_by_scene[scene]
            for component_id in sorted(actions):
                old = states[scene][component_id]
                for candidate in _replacement_options(actions[component_id], old):
                    states[scene][component_id] = candidate["action_name"]
                    trial, trial_detail = _records_for_state(
                        scene, actions, states[scene], caches[scene], canonical[scene],
                        native_mode, score_mode,
                    )
                    combined = dict(records)
                    combined[scene] = trial
                    value = _objective(combined)
                    if value > score + EPS:
                        score, records[scene], details[scene], old = value, trial, trial_detail, candidate["action_name"]
                        accepted += 1
                    else:
                        states[scene][component_id] = old
        history.append({"sweep": sweep + 1, "accepted_action_count": accepted, "official_ap": score})
        print(f"[轮次完成] {name}: 轮次={sweep + 1}, 接受={accepted}, 正式AP={score:.9f}", flush=True)
        if not accepted:
            break
    final_records, final_details = {}, {}
    for scene in scenes:
        final_records[scene], final_details[scene] = _records_for_state(
            scene, actions_by_scene[scene], states[scene], caches[scene], canonical[scene],
            native_mode, score_mode,
        )
    return {
        "branch": name,
        "selection_applied": True,
        "native_geometry_mode": native_mode,
        "score_mode": score_mode,
        "initial_official_ap": initial_ap,
        "official_ap": _objective(final_records),
        "threshold_ap": _summary_records(final_records),
        "history": history,
        "state": states,
        "scene_candidate_counts": final_details,
        "selected_action_kinds": {
            scene: _selection_counts(actions_by_scene[scene], states[scene]) for scene in scenes
        },
    }


def _run_fixed(
    name: str,
    native_mode: str,
    score_mode: str,
    scenes: list[str],
    actions_by_scene: dict[str, dict[int, list[dict]]],
    caches: dict[str, dict],
    canonical: dict[str, dict[int, int]],
) -> dict:
    states = {scene: _baseline_state(actions_by_scene[scene]) for scene in scenes}
    records, details = {}, {}
    for scene in scenes:
        records[scene], details[scene] = _records_for_state(
            scene, actions_by_scene[scene], states[scene], caches[scene], canonical[scene],
            native_mode, score_mode,
        )
    return {
        "branch": name,
        "selection_applied": False,
        "native_geometry_mode": native_mode,
        "score_mode": score_mode,
        "official_ap": _objective(records),
        "threshold_ap": _summary_records(records),
        "state": states,
        "scene_candidate_counts": details,
        "selected_action_kinds": {
            scene: _selection_counts(actions_by_scene[scene], states[scene]) for scene in scenes
        },
    }


def main() -> None:
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
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in (
        "scene_list", "relation_ledger_root", "action_ledger_root", "native_fold_audit_root",
        "filtered_d2b_track_root", "native_prediction_cache", "gt_instance_dir", "output_root",
    ):
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
        results = [
            _run_fixed("F-only", "exact_folded", FROZEN_SCORE, scenes, actions, caches, canonical),
            _run_replacement("R-only", "raw", FROZEN_SCORE, scenes, actions, caches, canonical, args.max_sweeps),
            _run_fixed("Q-only", "raw", GT_QUALITY_SCORE, scenes, actions, caches, canonical),
            _run_replacement("F+R", "exact_folded", FROZEN_SCORE, scenes, actions, caches, canonical, args.max_sweeps),
            _run_replacement("R+Q", "raw", GT_QUALITY_SCORE, scenes, actions, caches, canonical, args.max_sweeps),
            _run_replacement("F+R+Q", "exact_folded", GT_QUALITY_SCORE, scenes, actions, caches, canonical, args.max_sweeps),
        ]
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    payload = {
        "diagnostic_type": "GT-only C1d attribution: folding, replacement, shared quality ranking",
        "not_mathematical_ap_upper_bound": True,
        "proposal_materialization_applied": False,
        "inference_score_or_rule_created": False,
        "official_ap_thresholds": list(OFFICIAL),
        "ap25_threshold": AP25,
        "extra_diagnostic_thresholds": [.95],
        "scene_count": len(scenes),
        "max_sweeps": args.max_sweeps,
        "replacement_action_kinds": sorted(REPLACEMENT_KINDS),
        "quality_contract": "candidate best valid-GT IoU primary rank; fixed scene/source/ID tie break only for equal IoU; one shared total order for all thresholds; no threshold-specific re-ranking",
        "results": results,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({row["branch"]: row["official_ap"] for row in results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

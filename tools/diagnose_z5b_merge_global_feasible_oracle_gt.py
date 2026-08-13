#!/usr/bin/env python3
"""Run the GT-only Z5b oracle over the frozen official100 merge action space.

The only admissible action is append-only insertion of an already materialized
pair-union from Z5a.  GT selects at most one action per target instance when
the child creates a new official IoU-threshold crossing over the complete
native+track same-class no-op baseline.  The resulting target-wise selection
is globally feasible and is evaluated with frozen classes and frozen hybrid
scores.  It is a deterministic feasible construction, not a mathematical AP
upper bound and never becomes an inference plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST  # noqa: E402
from tools.evaluate_z3_yoloworld_control_group_gt import (  # noqa: E402
    _Predictions,
    _evaluate,
    _load_native,
    _load_tracks,
    _load_union_rows,
    _points,
    _read_scenes,
)


VERSION = "z5b_merge_targetwise_global_feasible_oracle_official100_v1"
EXPECTED_SPLIT_SHA256 = "aa657449965bc76164a1a1b77c7785aa705a0295eeed1307163b325f7233fe3e"
OFFICIAL_THRESHOLDS = tuple(
    round(float(value), 2) for value in instance_eval.opt["overlaps"] if float(value) >= 0.50
)
AP25_THRESHOLD = 0.25


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _valid_gt_instances(gt_ids: np.ndarray) -> dict[int, int]:
    valid_classes = set(map(int, VALID_CLASS_IDS_200_INST))
    result = {}
    ids, counts = np.unique(gt_ids, return_counts=True)
    for instance_id, count in zip(ids, counts):
        instance_id = int(instance_id)
        if (
            instance_id >= 1000
            and instance_id // 1000 in valid_classes
            and int(count) >= int(instance_eval.opt["min_region_sizes"][0])
        ):
            result[instance_id] = int(count)
    return result


def _best_gt(points: np.ndarray, gt_ids: np.ndarray, gt_sizes: dict[int, int], class_id: int | None = None):
    ids, counts = np.unique(gt_ids[points], return_counts=True)
    best_id, best_iou = -1, 0.0
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        if class_id is not None and instance_id // 1000 != int(class_id):
            continue
        iou = float(intersection / max(1, len(points) + gt_sizes[instance_id] - int(intersection)))
        if iou > best_iou:
            best_id, best_iou = instance_id, iou
    return best_id, best_iou


def _iou_to_target(points: np.ndarray, gt_ids: np.ndarray, gt_sizes: dict[int, int], target: int) -> float:
    if target not in gt_sizes:
        return 0.0
    intersection = int(np.count_nonzero(gt_ids[points] == target))
    return float(intersection / max(1, len(points) + gt_sizes[target] - intersection))


def threshold_crossings(noop_iou: float, action_iou: float) -> dict[str, int]:
    thresholds = (AP25_THRESHOLD, *OFFICIAL_THRESHOLDS)
    return {
        str(int(round(threshold * 100))): int(noop_iou <= threshold < action_iou)
        for threshold in thresholds
    }


def choose_targetwise_actions(rows: list[dict]) -> list[dict]:
    """Choose at most one positive official-threshold crossing per GT target."""
    by_target = defaultdict(list)
    for row in rows:
        target = int(row["action_best_same_class_gt_instance_id"])
        official_gain = int(sum(
            int(row["threshold_crossings"][str(int(round(value * 100)))])
            for value in OFFICIAL_THRESHOLDS
        ))
        if target > 0 and bool(row["semantic_correct"]) and official_gain > 0:
            by_target[(str(row["scene_name"]), target)].append((row, official_gain))
    selected = []
    for _, candidates in sorted(by_target.items()):
        row, gain = max(candidates, key=lambda item: (
            item[1],
            int(item[0]["threshold_crossings"]["50"]),
            int(item[0]["threshold_crossings"]["25"]),
            float(item[0]["action_best_same_class_iou"] - item[0]["noop_best_same_class_iou"]),
            float(item[0]["action_best_same_class_iou"]),
            -int(item[0]["child_candidate_id"]),
        ))
        selected.append({
            "action_id": str(row["action_id"]),
            "scene_name": str(row["scene_name"]),
            "child_candidate_id": int(row["child_candidate_id"]),
            "target_gt_instance_id": int(row["action_best_same_class_gt_instance_id"]),
            "target_gt_class_id": int(row["predicted_semantic_class_id"]),
            "official_threshold_crossing_count": int(gain),
            "tp50_crossing": int(row["threshold_crossings"]["50"]),
            "tp25_crossing": int(row["threshold_crossings"]["25"]),
            "noop_best_same_class_iou": float(row["noop_best_same_class_iou"]),
            "action_best_same_class_iou": float(row["action_best_same_class_iou"]),
            "selection_contract": "one action per scene/GT target; maximize new official threshold crossings, then deterministic IoU tie-breaks",
            "ground_truth_usage": "oracle_only",
            "written_to_inference_plan": False,
        })
    return selected


def _base_prediction(scene: str, args, rows_by_scene_source) -> dict:
    native = _load_native(args.stream_records_root, scene)
    native_scores = native["pred_scores"].copy()
    seen = set()
    for row in rows_by_scene_source[(scene, "native")]:
        candidate_id = int(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"{scene}: duplicate native OOF candidate {candidate_id}")
        seen.add(candidate_id)
        if int(native["pred_classes"][candidate_id]) != int(row["class_index"]):
            raise ValueError(f"{scene}: native class mismatch {candidate_id}")
        native_scores[candidate_id] = float(row["oof_predictions"]["C_joint_yolo_alpha"])
    native_prediction = {**native, "pred_scores": native_scores}

    tracks = _load_tracks(args.stream_records_root, scene)
    pieces = []
    for row in sorted(rows_by_scene_source[(scene, "track")], key=lambda item: int(item["candidate_id"])):
        candidate_id = int(row["candidate_id"])
        track = tracks[candidate_id]
        mask = np.zeros(native["pred_masks"].shape[0], dtype=bool)
        mask[_points(Path(track["points_path"]), len(mask))] = True
        pieces.append((mask, int(row["class_index"]), float(row["oof_predictions"]["C_joint_yolo_alpha"])))
    track_prediction = {
        "pred_masks": np.stack([row[0] for row in pieces], axis=1) if pieces else np.zeros((native["pred_masks"].shape[0], 0), dtype=bool),
        "pred_classes": np.asarray([row[1] for row in pieces], dtype=np.int64),
        "pred_scores": np.asarray([row[2] for row in pieces], dtype=np.float32),
    }
    return {
        "pred_masks": np.concatenate([native_prediction["pred_masks"], track_prediction["pred_masks"]], axis=1),
        "pred_classes": np.concatenate([native_prediction["pred_classes"], track_prediction["pred_classes"]]).astype(np.int64),
        "pred_scores": np.concatenate([native_prediction["pred_scores"], track_prediction["pred_scores"]]).astype(np.float32),
    }


def _scene_prediction(scene: str, selected: set[int] | None, args, rows_by_scene_source) -> dict:
    base = _base_prediction(scene, args, rows_by_scene_source)
    if selected is not None and not selected:
        return base
    unions = _load_union_rows(args.combined_plan_root, scene)
    pieces = []
    for row in sorted(rows_by_scene_source[(scene, "pair_union")], key=lambda item: int(item["candidate_id"])):
        candidate_id = int(row["candidate_id"])
        if selected is not None and candidate_id not in selected:
            continue
        union = unions[candidate_id]
        mask = np.zeros(base["pred_masks"].shape[0], dtype=bool)
        mask[_points(Path(union["points_path"]), len(mask))] = True
        pieces.append((mask, int(row["class_index"]), float(row["original_score"])))
    if not pieces:
        return base
    return {
        "pred_masks": np.concatenate([base["pred_masks"], np.stack([row[0] for row in pieces], axis=1)], axis=1),
        "pred_classes": np.concatenate([base["pred_classes"], np.asarray([row[1] for row in pieces], dtype=np.int64)]),
        "pred_scores": np.concatenate([base["pred_scores"], np.asarray([row[2] for row in pieces], dtype=np.float32)]),
    }


def _best_base_iou_for_target(
    class_index: int, target: int, gt_ids: np.ndarray, gt_sizes: dict[int, int],
    native: dict, track_rows: list[dict], tracks: dict[int, dict], track_points: dict[int, np.ndarray],
) -> tuple[float, str | None]:
    best_iou, best_key = 0.0, None
    native_ids = np.flatnonzero(native["pred_classes"] == class_index)
    if len(native_ids):
        intersections = np.asarray(native["pred_masks"][gt_ids == target][:, native_ids].sum(axis=0), dtype=np.int64)
        sizes = np.asarray(native["pred_masks"][:, native_ids].sum(axis=0), dtype=np.int64)
        ious = intersections / np.maximum(1, sizes + gt_sizes[target] - intersections)
        index = int(np.argmax(ious))
        if float(ious[index]) > best_iou:
            best_iou = float(ious[index])
            best_key = f"native:{int(native_ids[index])}"
    for row in track_rows:
        if int(row["class_index"]) != class_index:
            continue
        track_id = int(row["candidate_id"])
        iou = _iou_to_target(track_points[track_id], gt_ids, gt_sizes, target)
        if iou > best_iou:
            best_iou, best_key = iou, f"track:{track_id}"
    return best_iou, best_key


def _local_scene_rows(scene: str, actions: list[dict], args, rows_by_scene_source) -> list[dict]:
    gt_ids = np.loadtxt(args.gt_instance_dir / f"{scene}.txt", dtype=np.int64)
    gt_sizes = _valid_gt_instances(gt_ids)
    native = _load_native(args.stream_records_root, scene)
    if native["pred_masks"].shape[0] != len(gt_ids):
        raise ValueError(f"{scene}: prediction/GT point count mismatch")
    tracks = _load_tracks(args.stream_records_root, scene)
    track_points = {
        track_id: _points(Path(track["points_path"]), len(gt_ids))
        for track_id, track in tracks.items()
    }
    track_rows = rows_by_scene_source[(scene, "track")]
    union_rows = {int(row["candidate_id"]): row for row in rows_by_scene_source[(scene, "pair_union")]}
    base_cache = {}
    result = []
    for action in sorted(actions, key=lambda row: int(row["child"]["candidate_id"])):
        candidate_id = int(action["child"]["candidate_id"])
        oof = union_rows[candidate_id]
        class_index = int(oof["class_index"])
        semantic_class_id = int(instance_eval.PRED_ID_TO_ID[class_index])
        child = _points(Path(action["child"]["points_path"]), len(gt_ids))
        same_id, same_iou = _best_gt(child, gt_ids, gt_sizes, semantic_class_id)
        any_id, any_iou = _best_gt(child, gt_ids, gt_sizes, None)
        semantic_correct = bool(any_id > 0 and any_id // 1000 == semantic_class_id)
        if same_id > 0:
            cache_key = (class_index, same_id)
            if cache_key not in base_cache:
                base_cache[cache_key] = _best_base_iou_for_target(
                    class_index, same_id, gt_ids, gt_sizes, native, track_rows, tracks, track_points
                )
            noop_iou, noop_key = base_cache[cache_key]
        else:
            noop_iou, noop_key = 0.0, None
        track_id = int(action["parents"]["track"]["candidate_id"])
        native_id = int(action["parents"]["native_exact_group"]["representative_candidate_id"])
        native_points = np.flatnonzero(np.asarray(native["pred_masks"][:, native_id], dtype=bool)).astype(np.int64)
        parent_track_iou = _iou_to_target(track_points[track_id], gt_ids, gt_sizes, same_id) if same_id > 0 else 0.0
        parent_native_iou = _iou_to_target(native_points, gt_ids, gt_sizes, same_id) if same_id > 0 else 0.0
        crossings = threshold_crossings(noop_iou, same_iou)
        result.append({
            "action_id": str(action["action_id"]), "scene_name": scene,
            "action_type": "merge", "child_candidate_id": candidate_id,
            "predicted_class_index": class_index, "predicted_semantic_class_id": semantic_class_id,
            "predicted_class_name": instance_eval.ID_TO_LABEL[semantic_class_id],
            "frozen_child_score": float(oof["original_score"]),
            "best_any_class_gt_instance_id": int(any_id),
            "best_any_class_gt_class_id": int(any_id // 1000) if any_id > 0 else -1,
            "best_any_class_iou": float(any_iou), "semantic_correct": semantic_correct,
            "action_best_same_class_gt_instance_id": int(same_id),
            "action_best_same_class_iou": float(same_iou),
            "noop_best_same_class_iou": float(noop_iou),
            "noop_best_same_class_candidate_key": noop_key,
            "parent_track_same_target_iou": float(parent_track_iou),
            "parent_native_same_target_iou": float(parent_native_iou),
            "action_delta_vs_noop_best_iou": float(same_iou - noop_iou),
            "threshold_crossings": crossings,
            "official_threshold_crossing_count": int(sum(
                crossings[str(int(round(value * 100)))] for value in OFFICIAL_THRESHOLDS
            )),
            "common_visibility_all_three_view_count": int(action["common_visibility"]["all_three_common_view_count"]),
            "ground_truth_usage": "oracle_only",
            "candidate_mutation": False,
            "action_applied_to_inference": False,
        })
    return result


def _delta(current: dict, baseline: dict) -> dict:
    return {
        name: float(current[name] - baseline[name])
        for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
    }


def run(args: argparse.Namespace) -> dict:
    scenes = _read_scenes(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("Z5b requires official100")
    split_sha = _sha256(args.split_manifest)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen split manifest SHA-256 mismatch")
    z5a_summary = json.loads((args.z5a_root / "summary.json").read_text())
    if (
        z5a_summary.get("ground_truth_usage") != "none"
        or z5a_summary.get("merge_action_count") != 1501
        or z5a_summary.get("action_family_status", {}).get("merge") != "available"
    ):
        raise ValueError("Z5a merge action contract is not valid")
    actions = _read_jsonl(args.z5a_root / "actions.jsonl")
    actions_by_scene = defaultdict(list)
    for row in actions:
        actions_by_scene[str(row["scene_name"])].append(row)
    if set(actions_by_scene) != set(scenes):
        raise ValueError("Z5a actions do not cover official100")

    oof_rows = _read_jsonl(args.z3_oof_root / "oof_predictions.jsonl")
    rows_by_scene_source = defaultdict(list)
    for row in oof_rows:
        rows_by_scene_source[(str(row["scene_name"]), str(row["candidate_source"]))].append(row)
    local_rows = []
    for index, scene in enumerate(scenes, 1):
        scene_rows = _local_scene_rows(scene, actions_by_scene[scene], args, rows_by_scene_source)
        local_rows.extend(scene_rows)
        print(f"[Z5b local oracle] {index}/100 {scene}: actions={len(scene_rows)}", flush=True)
    selected_rows = choose_targetwise_actions(local_rows)
    selected_by_scene = defaultdict(set)
    for row in selected_rows:
        selected_by_scene[str(row["scene_name"])].add(int(row["child_candidate_id"]))

    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        systems = {}
        for name, selector in (
            ("noop_native_plus_track", {scene: set() for scene in scenes}),
            ("all_frozen_pair_union_control", {scene: None for scene in scenes}),
            ("targetwise_global_feasible_oracle_frozen_score", selected_by_scene),
        ):
            mapping = _Predictions(
                scenes,
                lambda scene, selector=selector: _scene_prediction(
                    scene, selector.get(scene, set()), args, rows_by_scene_source
                ),
            )
            systems[name] = _evaluate(mapping, args.gt_instance_dir, staging / f"{name}.csv")
            print(f"[Z5b AP] {name}: {systems[name]['ap']:.6f}", flush=True)

        expected = json.loads(args.hybrid_control_summary.read_text())["systems"]["C_joint_native_track_union_frozen_score"]
        reproduction = {
            "noop_native_plus_track_max_abs_error": max(
                abs(systems["noop_native_plus_track"][name] - expected["native_plus_track"][name])
                for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
            ),
            "all_frozen_pair_union_max_abs_error": max(
                abs(systems["all_frozen_pair_union_control"][name] - expected["pair_union"][name])
                for name in ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
            ),
        }
        if max(reproduction.values()) > args.control_tolerance:
            raise RuntimeError(f"Z5b control reproduction failed: {reproduction}")

        manifest = json.loads(args.split_manifest.read_text())
        folds = []
        for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
            fold_scenes = list(spec["validation_scenes"])
            fold_systems = {}
            for name, selector in (
                ("noop_native_plus_track", {scene: set() for scene in fold_scenes}),
                ("all_frozen_pair_union_control", {scene: None for scene in fold_scenes}),
                ("targetwise_global_feasible_oracle_frozen_score", selected_by_scene),
            ):
                mapping = _Predictions(
                    fold_scenes,
                    lambda scene, selector=selector: _scene_prediction(
                        scene, selector.get(scene, set()), args, rows_by_scene_source
                    ),
                )
                fold_systems[name] = _evaluate(
                    mapping, args.gt_instance_dir, staging / f"fold_{spec['fold_index']}__{name}.csv"
                )
            baseline = fold_systems["noop_native_plus_track"]
            folds.append({
                "fold_index": int(spec["fold_index"]), "validation_scenes": fold_scenes,
                "selected_action_count": sum(len(selected_by_scene[scene]) for scene in fold_scenes),
                "systems": fold_systems,
                "deltas_vs_noop": {
                    name: _delta(value, baseline) for name, value in fold_systems.items()
                    if name != "noop_native_plus_track"
                },
            })
            print(f"[Z5b AP] fold {spec['fold_index']} complete", flush=True)

        selected_ids = {row["action_id"] for row in selected_rows}
        scene_attribution = []
        for scene in scenes:
            scene_rows = [row for row in local_rows if row["scene_name"] == scene]
            scene_attribution.append({
                "scene_name": scene, "action_count": len(scene_rows),
                "selected_action_count": sum(row["action_id"] in selected_ids for row in scene_rows),
                "official_threshold_crossing_count": sum(
                    row["official_threshold_crossing_count"] for row in scene_rows if row["action_id"] in selected_ids
                ),
            })
        class_groups = defaultdict(list)
        for row in local_rows:
            class_groups[(int(row["predicted_class_index"]), str(row["predicted_class_name"]))].append(row)
        class_attribution = []
        for (class_index, class_name), rows in sorted(class_groups.items()):
            class_attribution.append({
                "predicted_class_index": class_index, "predicted_class_name": class_name,
                "action_count": len(rows),
                "semantic_correct_count": sum(bool(row["semantic_correct"]) for row in rows),
                "selected_action_count": sum(row["action_id"] in selected_ids for row in rows),
                "selected_official_threshold_crossing_count": sum(
                    row["official_threshold_crossing_count"] for row in rows if row["action_id"] in selected_ids
                ),
            })

        crossing_counts = Counter()
        for row in local_rows:
            for tag, value in row["threshold_crossings"].items():
                crossing_counts[tag] += int(value)
        selected_crossing_counts = Counter()
        for row in local_rows:
            if row["action_id"] in selected_ids:
                for tag, value in row["threshold_crossings"].items():
                    selected_crossing_counts[tag] += int(value)
        summary = {
            "version": VERSION,
            "diagnostic_type": "Z5b GT-only target-wise global-feasible append-only merge oracle",
            "not_mathematical_ap_upper_bound": True,
            "scene_count": len(scenes), "action_count": len(local_rows),
            "selected_action_count": len(selected_rows),
            "selected_scene_count": len({row["scene_name"] for row in selected_rows}),
            "semantic_correct_action_count": sum(bool(row["semantic_correct"]) for row in local_rows),
            "actions_with_any_official_threshold_crossing": sum(row["official_threshold_crossing_count"] > 0 for row in local_rows),
            "threshold_crossing_counts": dict(sorted(crossing_counts.items(), key=lambda item: int(item[0]))),
            "selected_threshold_crossing_counts": dict(sorted(selected_crossing_counts.items(), key=lambda item: int(item[0]))),
            "systems": systems,
            "deltas_vs_noop": {
                name: _delta(value, systems["noop_native_plus_track"])
                for name, value in systems.items() if name != "noop_native_plus_track"
            },
            "control_reproduction": {**reproduction, "tolerance": args.control_tolerance, "valid": True},
            "folds": folds,
            "positive_fold_count_main_ap_vs_noop": sum(
                fold["deltas_vs_noop"]["targetwise_global_feasible_oracle_frozen_score"]["ap"] > 0
                for fold in folds
            ),
            "ground_truth_usage": "official_train_oracle_and_evaluation_only",
            "candidate_mutation": False, "proposal_materialization_applied": False,
            "inference_plan_written": False, "model_trained": False,
            "safety60_read": False, "even48_read": False, "test60_read": False,
            "contracts": {
                "noop": "current native+track C_joint OOF hybrid scores and frozen classes",
                "action": "append one already materialized pair-union with frozen class and original frozen low score",
                "selection": "at most one action per scene/GT instance; require a new official IoU threshold crossing over the complete same-class native+track baseline",
                "oracle_isolation": "GT selections stay only in this diagnostic directory and are forbidden as inference labels/rules",
            },
            "input_provenance": {
                "scene_list_sha256": _sha256(args.scene_list),
                "split_manifest_sha256": split_sha,
                "z5a_summary_sha256": _sha256(args.z5a_root / "summary.json"),
                "z5a_actions_sha256": _sha256(args.z5a_root / "actions.jsonl"),
                "z3_oof_summary_sha256": _sha256(args.z3_oof_root / "summary.json"),
                "z3_oof_predictions_sha256": _sha256(args.z3_oof_root / "oof_predictions.jsonl"),
                "hybrid_control_summary_sha256": _sha256(args.hybrid_control_summary),
                "combined_plan_summary_sha256": _sha256(args.combined_plan_root / "summary.json"),
            },
        }
        _write_jsonl(staging / "action_oracle_gt.jsonl", local_rows)
        _write_jsonl(staging / "global_feasible_selection_gt.jsonl", selected_rows)
        _write_jsonl(staging / "scene_attribution.jsonl", scene_attribution)
        _write_jsonl(staging / "class_attribution.jsonl", class_attribution)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--z5a-root", type=Path, required=True)
    parser.add_argument("--z3-oof-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, required=True)
    parser.add_argument("--hybrid-control-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("Z5b requires --allow-gt-diagnostics; GT is oracle/evaluation only")
    for name in (
        "scene_list", "split_manifest", "z5a_root", "z3_oof_root", "stream_records_root",
        "combined_plan_root", "gt_instance_dir", "hybrid_control_summary", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    result = run(args)
    print(json.dumps({
        "action_count": result["action_count"],
        "selected_action_count": result["selected_action_count"],
        "systems": result["systems"],
        "positive_fold_count_main_ap_vs_noop": result["positive_fold_count_main_ap_vs_noop"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

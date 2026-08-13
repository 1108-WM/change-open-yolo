#!/usr/bin/env python3
"""Audit pair-label separability using only fixed candidate OOF scores.

This diagnostic has no fitting stage.  It aggregates native class expansions
by exact geometry (median is frozen as the primary aggregation), compares
fixed C/D candidate-quality score differences, and reports ranking safety.
It never chooses a threshold, emits a replacement plan, changes candidates,
or evaluates AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE, read_jsonl, read_scene_list


MODEL_GROUPS = ("C_plus_geometry_track_structure", "D_plus_gvc")
TARGETS = ("q", "valid25", "valid50")
TOP_FRACTIONS = (.005, .01, .02, .05)
OFFICIAL100_SCENE_COUNT = 100
OFFICIAL100_EXISTING_SCENE_COUNT = 20
OFFICIAL100_NEW_SCENE_COUNT = 80


def _load_split_folds(path: Path, scene_names: list[str]) -> tuple[dict[str, int], dict[str, set[str]], dict]:
    payload = json.loads(path.read_text())
    expected_scenes = set(scene_names)
    if len(scene_names) != OFFICIAL100_SCENE_COUNT or len(expected_scenes) != len(scene_names):
        raise ValueError("official100 pair audit requires exactly 100 unique scenes")
    if payload.get("scene_count") != OFFICIAL100_SCENE_COUNT:
        raise ValueError("frozen split manifest does not declare 100 scenes")
    if payload.get("existing_scene_count") != OFFICIAL100_EXISTING_SCENE_COUNT or payload.get("new_scene_count") != OFFICIAL100_NEW_SCENE_COUNT:
        raise ValueError("frozen split manifest must declare the fixed official20/new80 composition")
    folds = payload.get("folds", [])
    scene_to_fold = {}
    for expected_index, fold in enumerate(sorted(folds, key=lambda value: value.get("fold_index", -1))):
        index = int(fold["fold_index"])
        if index != expected_index:
            raise ValueError("frozen split fold indices must be exactly 0..4")
        train, validation = set(fold.get("train_scenes", [])), set(fold.get("validation_scenes", []))
        if len(train) != 80 or len(validation) != 20:
            raise ValueError("every frozen official100 fold must have 80 train and 20 validation scenes")
        if train & validation or train | validation != expected_scenes or train != expected_scenes - validation:
            raise ValueError(f"frozen fold {index} does not exactly partition the official100 scenes")
        for scene in fold["validation_scenes"]:
            if scene in scene_to_fold:
                raise ValueError(f"scene appears in multiple OOF validation folds: {scene}")
            scene_to_fold[scene] = index
    if len(folds) != 5 or set(scene_to_fold) != expected_scenes or len(scene_to_fold) != OFFICIAL100_SCENE_COUNT:
        raise ValueError("every official100 scene must appear in exactly one OOF validation fold")
    cohorts = {
        "existing20": set(scene_names[:OFFICIAL100_EXISTING_SCENE_COUNT]),
        "new80": set(scene_names[OFFICIAL100_EXISTING_SCENE_COUNT:]),
    }
    return scene_to_fold, cohorts, payload


def _score_lookup(path: Path) -> dict[tuple[str, str, int], dict]:
    rows = read_jsonl(path)
    lookup = {}
    for row in rows:
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        if key in lookup:
            raise ValueError(f"duplicate OOF candidate: {key}")
        if row["candidate_source"] not in (NATIVE_SOURCE, TRACK_SOURCE):
            raise ValueError(f"unexpected OOF source: {row['candidate_source']}")
        for model in MODEL_GROUPS:
            for target in TARGETS:
                try:
                    value = float(row["predictions"][model][target])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"missing OOF prediction {model}/{target}: {key}") from error
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"OOF prediction outside [0, 1] for {model}/{target}: {key}")
        lookup[key] = row
    if len(lookup) != len(rows):
        raise AssertionError("OOF lookup does not conserve rows")
    return lookup


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()), "max": float(array.max()), "median": float(np.median(array)),
        "mean": float(array.mean()), "std": float(array.std()), "range": float(array.max() - array.min()),
    }


def _pair_rows(pair_root: Path) -> list[dict]:
    summary = json.loads((pair_root / "summary.json").read_text())
    rows = []
    for scene_summary in summary["scene_summaries"]:
        scene = scene_summary["scene_name"]
        rows.extend(read_jsonl(pair_root / scene / "pair_labels_v2.jsonl"))
    if len(rows) != int(summary["v2_geometry_relation_count"]):
        raise ValueError("pair v2 summary count differs from pair rows")
    if int(summary.get("folded_source_relation_count", -1)) != int(summary["v1_relation_count"]):
        raise ValueError("pair v2 exact-geometry folding did not conserve source relations")
    return rows


def build_scored_pairs(oof_lookup: dict, scene_to_fold: dict[str, int], pair_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    group_cache = {}
    scored = []
    for pair in pair_rows:
        scene = str(pair["scene_name"])
        if scene not in scene_to_fold:
            raise ValueError(f"pair scene is absent from OOF validation split: {scene}")
        track_key = (scene, TRACK_SOURCE, int(pair["track_id"]))
        if track_key not in oof_lookup:
            raise ValueError(f"missing track OOF score: {track_key}")
        group_key = (scene, str(pair["native_exact_geometry_group_id"]))
        if group_key not in group_cache:
            members = [int(value) for value in pair["native_member_candidate_ids"]]
            native_rows = []
            for member in members:
                key = (scene, NATIVE_SOURCE, member)
                if key not in oof_lookup:
                    raise ValueError(f"missing native OOF score: {key}")
                native_rows.append(oof_lookup[key])
            aggregate = {
                "scene_name": scene,
                "fold_index": scene_to_fold[scene],
                "native_exact_geometry_group_id": group_key[1],
                "native_member_candidate_ids": members,
                "member_count": len(members),
                "scores": {},
            }
            for model in MODEL_GROUPS:
                for target in TARGETS:
                    aggregate["scores"][f"{model}:{target}"] = _stats([
                        float(row["predictions"][model][target]) for row in native_rows
                    ])
            group_cache[group_key] = aggregate
        if pair["label_pair_preference"] != "unknown" and not bool(pair.get("reliable_pair")):
            raise ValueError("unreliable pair was assigned a same-target or preference label")
        if pair["label_pair_preference"] != "unknown" and pair.get("same_best_gt") not in (True, False):
            raise ValueError("reliable pair must have a boolean same_best_gt label")
        track = oof_lookup[track_key]
        aggregate = group_cache[group_key]
        item = {
            "scene_name": scene,
            "fold_index": scene_to_fold[scene],
            "track_id": int(pair["track_id"]),
            "native_exact_geometry_group_id": group_key[1],
            "native_member_count": aggregate["member_count"],
            "label_pair_preference": pair["label_pair_preference"],
            "same_best_gt": pair["same_best_gt"],
            "reliable_pair": bool(pair["reliable_pair"]),
            "reliability_state": pair["reliability_state"],
            "ground_truth_usage": "label_only_for_diagnostic",
        }
        item["delta_raw_original_score"] = float(pair["track_original_score"]) - float(pair["native_original_score_median"])
        for model in MODEL_GROUPS:
            for target in TARGETS:
                key = f"{model}:{target}"
                native_median = aggregate["scores"][key]["median"]
                track_score = float(track["predictions"][model][target])
                item[f"track_{model}_{target}"] = track_score
                item[f"native_median_{model}_{target}"] = native_median
                item[f"delta_{model}_{target}"] = track_score - native_median
        scored.append(item)
    return scored, [group_cache[key] for key in sorted(group_cache)]


def _wilson(successes: int, total: int) -> dict | None:
    if total == 0:
        return None
    z = 1.959963984540054
    value = successes / total
    denominator = 1 + z * z / total
    centre = (value + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(value * (1 - value) / total + z * z / (4 * total * total)) / denominator
    return {"level": 0.95, "lower": max(0.0, centre - radius), "upper": min(1.0, centre + radius)}


def _task_rows(rows: list[dict], task: str) -> tuple[list[dict], np.ndarray]:
    if task == "same_target_vs_coexist":
        selected = [row for row in rows if row["label_pair_preference"] != "unknown"]
        return selected, np.asarray([int(row["same_best_gt"] is True) for row in selected], dtype=np.int64)
    if task == "same_target_relative_quality":
        selected = [row for row in rows if row["label_pair_preference"] in ("prefer_track", "prefer_native")]
        return selected, np.asarray([int(row["label_pair_preference"] == "prefer_track") for row in selected], dtype=np.int64)
    if task == "replacement_safety":
        selected = list(rows)
        return selected, np.asarray([int(row["label_pair_preference"] == "prefer_track") for row in selected], dtype=np.int64)
    raise ValueError(f"unknown task: {task}")


def _top_metrics(rows: list[dict], labels: np.ndarray, score_name: str) -> dict:
    order = sorted(range(len(rows)), key=lambda index: (-float(rows[index][score_name]), rows[index]["scene_name"], rows[index]["track_id"], rows[index]["native_exact_geometry_group_id"]))
    positives = int(labels.sum())
    result = {}
    for fraction in TOP_FRACTIONS:
        count = min(len(rows), max(1, math.ceil(len(rows) * fraction)))
        selected = order[:count]
        selected_labels = labels[selected]
        correct = int(selected_labels.sum())
        error_labels = Counter(rows[index]["label_pair_preference"] for index in selected if not labels[index])
        positive_scenes = {rows[index]["scene_name"] for index in selected if labels[index]}
        fraction_label = f"top_{fraction * 100:g}pct".replace(".", "_")
        result[fraction_label] = {
            "selected_count": count, "precision": correct / count, "recall": correct / positives if positives else None,
            "precision_wilson_95": _wilson(correct, count), "positive_scene_count": len(positive_scenes),
            "error_label_counts": dict(sorted(error_labels.items())),
        }
    scene_selected = []
    for scene in sorted({row["scene_name"] for row in rows}):
        indexes = [index for index in order if rows[index]["scene_name"] == scene]
        if indexes:
            scene_selected.append(indexes[0])
    correct = int(labels[scene_selected].sum())
    result["per_scene_top1"] = {
        "selected_count": len(scene_selected), "precision": correct / len(scene_selected) if scene_selected else None,
        "precision_wilson_95": _wilson(correct, len(scene_selected)),
        "positive_scene_count": len({rows[index]["scene_name"] for index in scene_selected if labels[index]}),
        "error_label_counts": dict(sorted(Counter(rows[index]["label_pair_preference"] for index in scene_selected if not labels[index]).items())),
    }
    return result


def _score_metrics(rows: list[dict], labels: np.ndarray, score_name: str) -> dict:
    scores = np.asarray([float(row[score_name]) for row in rows], dtype=np.float64)
    base = {"count": len(rows), "positive_count": int(labels.sum()), "positive_rate": float(labels.mean()) if len(labels) else None}
    if len(rows) and len(np.unique(labels)) == 2:
        base["roc_auc"] = float(roc_auc_score(labels, scores))
        base["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        base["roc_auc"] = None
        base["pr_auc"] = None
    base["top_selection"] = _top_metrics(rows, labels, score_name) if len(rows) else {}
    return base


def analyze(scored_pairs: list[dict], cohorts: dict[str, set[str]]) -> tuple[dict, list[dict], list[dict], dict]:
    score_names = ["delta_raw_original_score"] + [f"delta_{model}_{target}" for model in MODEL_GROUPS for target in TARGETS]
    tasks = ("same_target_vs_coexist", "same_target_relative_quality", "replacement_safety")
    summary_tasks, fold_metrics, scene_metrics = {}, [], []
    for task in tasks:
        rows, labels = _task_rows(scored_pairs, task)
        summary_tasks[task] = {score: _score_metrics(rows, labels, score) for score in score_names}
        fold_data = {"task": task, "folds": []}
        for fold_index in range(5):
            fold_rows, fold_labels = _task_rows([row for row in scored_pairs if row["fold_index"] == fold_index], task)
            fold_data["folds"].append({"fold_index": fold_index, "scores": {score: _score_metrics(fold_rows, fold_labels, score) for score in score_names}})
        fold_metrics.append(fold_data)
        for scene in sorted({row["scene_name"] for row in scored_pairs}):
            scene_rows, scene_labels = _task_rows([row for row in scored_pairs if row["scene_name"] == scene], task)
            scene_metrics.append({"scene_name": scene, "task": task, "scores": {score: _score_metrics(scene_rows, scene_labels, score) for score in score_names}})
    cohort_metrics = {}
    for cohort_name, cohort_scenes in cohorts.items():
        cohort_metrics[cohort_name] = {
            "scene_count": len(cohort_scenes),
            "tasks": {
                task: {
                    score: _score_metrics(*_task_rows([row for row in scored_pairs if row["scene_name"] in cohort_scenes], task), score)
                    for score in score_names
                }
                for task in tasks
            },
        }
    return summary_tasks, fold_metrics, scene_metrics, cohort_metrics


def _validate_oof_summary(path: Path, expected_protocol: str, oof_row_count: int) -> dict:
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing OOF summary beside predictions: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("protocol_name") != expected_protocol:
        raise ValueError(f"OOF protocol is not {expected_protocol}")
    if int(summary.get("candidate_count", -1)) != oof_row_count:
        raise ValueError("OOF summary candidate count differs from OOF predictions")
    if summary.get("prediction_contract", {}).get("q") != "clipped_to_[0,1]_before_metrics_and_oof_export":
        raise ValueError("OOF q prediction contract is not the required v2 clipped contract")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--pair-ledger-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--expected-oof-protocol", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    actual_split_sha256 = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    if actual_split_sha256 != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 does not match --expected-split-sha256")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_split_folds(args.split_manifest, scenes)
    oof_lookup = _score_lookup(args.oof_predictions)
    oof_summary = _validate_oof_summary(args.oof_predictions, args.expected_oof_protocol, len(oof_lookup))
    scored_pairs, group_consistency = build_scored_pairs(oof_lookup, scene_to_fold, _pair_rows(args.pair_ledger_root))
    task_summary, fold_metrics, scene_metrics, cohort_metrics = analyze(scored_pairs, cohorts)
    summary = {
        "version": f"{args.protocol_name}_pair_fixed_oof_score_separability_v1",
        "protocol_name": args.protocol_name,
        "pair_count": len(scored_pairs), "native_geometry_group_count": len(group_consistency),
        "primary_native_aggregation": "median", "models": list(MODEL_GROUPS), "targets": list(TARGETS),
        "tasks": task_summary, "cohort_metrics": cohort_metrics,
        "input_provenance": {
            "oof_predictions_path": str(args.oof_predictions.resolve()),
            "oof_predictions_sha256": hashlib.sha256(args.oof_predictions.read_bytes()).hexdigest(),
            "oof_protocol_name": oof_summary["protocol_name"],
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": actual_split_sha256,
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": hashlib.sha256(args.scene_list.read_bytes()).hexdigest(),
            "pair_ledger_summary_sha256": hashlib.sha256((args.pair_ledger_root / "summary.json").read_bytes()).hexdigest(),
            "cohort_definition": "existing20 is the first 20 ordered scenes; new80 is the remaining 80 in official_train100.txt",
            "frozen_manifest_existing_scene_count": split_payload["existing_scene_count"],
            "frozen_manifest_new_scene_count": split_payload["new_scene_count"],
        },
        "pair_model_trained": False, "replacement_plan_generated": False,
        "candidate_geometry_modified": False, "ap_evaluation_run": False,
        "ground_truth_usage": "label_only_for_offline_diagnostic",
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "fold_metrics.json").write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "scene_metrics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scene_metrics))
        (staging / "native_group_consistency.json").write_text(json.dumps(group_consistency, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "scored_pairs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored_pairs))
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

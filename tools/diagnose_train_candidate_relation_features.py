#!/usr/bin/env python3
"""Audit frozen candidate-relation features without fitting a relation model.

The primary task asks whether a reliable track--native relation belongs to the
same ground-truth object or to different objects that must coexist.  Unknown
relations are excluded from that task rather than being fabricated as
negatives.  The diagnostic reports single-feature separability, fold and
cohort stability, plus scene/track-cluster bootstrap intervals.  It never
selects a threshold or emits a replacement action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list


TASKS = (
    "same_target_vs_coexist",
    "same_target_relative_quality",
    "replacement_safety",
)
BOOTSTRAP_TOP_FEATURE_COUNT = 15
BOOTSTRAP_SEED = 20260809


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_partition(
    split_manifest: Path,
    scene_names: list[str],
    existing_scene_list: Path,
) -> tuple[dict[str, int], dict[str, set[str]], dict]:
    payload = json.loads(split_manifest.read_text())
    expected = set(scene_names)
    if len(scene_names) != 100 or len(expected) != 100:
        raise ValueError("relation feature audit requires exactly 100 unique scenes")
    existing = set(read_scene_list(existing_scene_list))
    if len(existing) != 20 or not existing <= expected:
        raise ValueError("explicit existing-scene list must contain 20 official100 scenes")
    folds = payload.get("folds", [])
    scene_to_fold = {}
    for expected_index, fold in enumerate(sorted(folds, key=lambda row: row.get("fold_index", -1))):
        index = int(fold["fold_index"])
        if index != expected_index:
            raise ValueError("frozen fold indices must be exactly 0..4")
        train = set(fold.get("train_scenes", []))
        validation = set(fold.get("validation_scenes", []))
        if len(train) != 80 or len(validation) != 20:
            raise ValueError("every frozen fold must contain 80 train and 20 validation scenes")
        if train & validation or train | validation != expected or train != expected - validation:
            raise ValueError(f"frozen fold {index} does not exactly partition official100")
        for scene in validation:
            if scene in scene_to_fold:
                raise ValueError(f"scene appears in multiple validation folds: {scene}")
            scene_to_fold[scene] = index
    if len(folds) != 5 or set(scene_to_fold) != expected:
        raise ValueError("every official100 scene must appear once in validation")
    return scene_to_fold, {"existing20": existing, "new80": expected - existing}, payload


def _load_rows(feature_root: Path, scenes: list[str]) -> list[dict]:
    summary = json.loads((feature_root / "summary.json").read_text())
    if int(summary.get("scene_count", -1)) != len(scenes):
        raise ValueError("feature ledger summary scene count differs from requested scenes")
    if summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("feature ledger does not declare GT-free features")
    rows = []
    identities = set()
    for scene in scenes:
        for row in read_jsonl(feature_root / scene / "relation_features.jsonl"):
            if row.get("scene_name") != scene:
                raise ValueError(f"{scene}: relation row scene mismatch")
            if row.get("contracts", {}).get("feature_ground_truth_usage") != "none":
                raise ValueError(f"{scene}: relation row does not declare GT-free features")
            key = (
                scene,
                int(row["track_id"]),
                str(row["native_exact_geometry_group_id"]),
            )
            if key in identities:
                raise ValueError(f"duplicate relation feature identity: {key}")
            identities.add(key)
            rows.append(row)
    if len(rows) != int(summary.get("relation_count", -1)):
        raise ValueError("feature ledger summary relation count differs from rows")
    return rows


def _numeric_feature_names(rows: list[dict]) -> list[str]:
    names = set(rows[0]["features"]) if rows else set()
    for row in rows[1:]:
        names &= set(row["features"])
    result = []
    for name in sorted(names):
        values = [row["features"][name] for row in rows]
        if all(isinstance(value, (int, float, bool)) for value in values):
            numeric = np.asarray(values, dtype=np.float64)
            if np.all(np.isfinite(numeric)):
                result.append(name)
    if not result:
        raise ValueError("feature ledger has no common finite numeric features")
    return result


def _task_rows(rows: list[dict], task: str) -> tuple[list[dict], np.ndarray]:
    if task == "same_target_vs_coexist":
        selected = [
            row for row in rows
            if row["labels"]["target_state"] in ("same_target", "different_target_coexist")
        ]
        labels = [row["labels"]["target_state"] == "same_target" for row in selected]
    elif task == "same_target_relative_quality":
        selected = [
            row for row in rows
            if row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native")
        ]
        labels = [row["labels"]["relative_quality_state"] == "prefer_track" for row in selected]
    elif task == "replacement_safety":
        selected = list(rows)
        labels = [row["labels"]["relative_quality_state"] == "prefer_track" for row in selected]
    else:
        raise ValueError(f"unknown task: {task}")
    return selected, np.asarray(labels, dtype=np.int64)


def _metric(rows: list[dict], labels: np.ndarray, feature: str) -> dict:
    scores = np.asarray([float(row["features"][feature]) for row in rows], dtype=np.float64)
    result = {
        "count": len(rows),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "unique_score_count": int(len(np.unique(scores))),
        "positive_median": float(np.median(scores[labels == 1])) if np.any(labels == 1) else None,
        "negative_median": float(np.median(scores[labels == 0])) if np.any(labels == 0) else None,
    }
    if len(rows) and len(np.unique(labels)) == 2 and len(np.unique(scores)) > 1:
        natural_auc = float(roc_auc_score(labels, scores))
        direction = 1 if natural_auc >= 0.5 else -1
        oriented = scores * direction
        result.update({
            "natural_roc_auc": natural_auc,
            "separability_roc_auc": max(natural_auc, 1.0 - natural_auc),
            "positive_direction": "higher" if direction > 0 else "lower",
            "directional_pr_auc": float(average_precision_score(labels, oriented)),
        })
    else:
        result.update({
            "natural_roc_auc": None,
            "separability_roc_auc": None,
            "positive_direction": None,
            "directional_pr_auc": None,
        })
    return result


def _cluster_bootstrap(
    rows: list[dict],
    labels: np.ndarray,
    feature: str,
    direction: int,
    cluster_kind: str,
    repetitions: int,
    seed: int,
) -> dict:
    if cluster_kind == "scene":
        cluster_ids = [str(row["scene_name"]) for row in rows]
    elif cluster_kind == "track":
        cluster_ids = [f"{row['scene_name']}:{int(row['track_id'])}" for row in rows]
    else:
        raise ValueError(f"unknown cluster kind: {cluster_kind}")
    unique = sorted(set(cluster_ids))
    cluster_index = {value: index for index, value in enumerate(unique)}
    row_clusters = np.asarray([cluster_index[value] for value in cluster_ids], dtype=np.int64)
    scores = np.asarray([float(row["features"][feature]) * direction for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    aucs, prs = [], []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(unique), size=len(unique))
        cluster_weights = np.bincount(sampled, minlength=len(unique)).astype(np.float64)
        weights = cluster_weights[row_clusters]
        active = weights > 0
        if len(np.unique(labels[active])) < 2:
            continue
        aucs.append(float(roc_auc_score(labels, scores, sample_weight=weights)))
        prs.append(float(average_precision_score(labels, scores, sample_weight=weights)))

    def interval(values: list[float]) -> dict | None:
        if not values:
            return None
        return {
            "median": float(np.median(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    return {
        "cluster_kind": cluster_kind,
        "cluster_count": len(unique),
        "requested_repetitions": repetitions,
        "valid_repetitions": len(aucs),
        "directional_roc_auc": interval(aucs),
        "directional_pr_auc": interval(prs),
    }


def analyze(
    rows: list[dict],
    features: list[str],
    scene_to_fold: dict[str, int],
    cohorts: dict[str, set[str]],
    bootstrap_repetitions: int,
) -> tuple[dict, list[dict], dict]:
    task_summary, fold_rows, cohort_summary = {}, [], {}
    for task in TASKS:
        selected, labels = _task_rows(rows, task)
        metrics = {feature: _metric(selected, labels, feature) for feature in features}
        ranked = sorted(
            features,
            key=lambda feature: (
                -(metrics[feature]["separability_roc_auc"] or -1.0), feature
            ),
        )
        task_summary[task] = {
            "sample_count": len(selected),
            "positive_count": int(labels.sum()),
            "unknown_relations_used_as_negative": False if task == "same_target_vs_coexist" else None,
            "feature_metrics": metrics,
            "features_ranked_by_univariate_separability": ranked,
            "bootstrap_top_features": {},
        }
        for rank, feature in enumerate(ranked[:BOOTSTRAP_TOP_FEATURE_COUNT]):
            direction = 1 if metrics[feature]["positive_direction"] == "higher" else -1
            if metrics[feature]["positive_direction"] is None:
                continue
            task_summary[task]["bootstrap_top_features"][feature] = {
                "rank": rank + 1,
                "scene_cluster": _cluster_bootstrap(
                    selected, labels, feature, direction, "scene",
                    bootstrap_repetitions, BOOTSTRAP_SEED + rank,
                ),
                "track_cluster": _cluster_bootstrap(
                    selected, labels, feature, direction, "track",
                    bootstrap_repetitions, BOOTSTRAP_SEED + 1000 + rank,
                ),
            }
        for fold_index in range(5):
            fold_selected, fold_labels = _task_rows([
                row for row in rows if scene_to_fold[row["scene_name"]] == fold_index
            ], task)
            fold_rows.append({
                "task": task,
                "fold_index": fold_index,
                "sample_count": len(fold_selected),
                "positive_count": int(fold_labels.sum()),
                "feature_metrics": {
                    feature: _metric(fold_selected, fold_labels, feature) for feature in features
                },
            })
        cohort_summary[task] = {}
        for cohort_name, cohort_scenes in cohorts.items():
            cohort_rows, cohort_labels = _task_rows([
                row for row in rows if row["scene_name"] in cohort_scenes
            ], task)
            cohort_summary[task][cohort_name] = {
                "scene_count": len(cohort_scenes),
                "sample_count": len(cohort_rows),
                "positive_count": int(cohort_labels.sum()),
                "feature_metrics": {
                    feature: _metric(cohort_rows, cohort_labels, feature) for feature in features
                },
            }
    return task_summary, fold_rows, cohort_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    actual_split_sha = _sha256(args.split_manifest)
    if actual_split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    rows = _load_rows(args.feature_ledger_root, scenes)
    features = _numeric_feature_names(rows)
    tasks, fold_metrics, cohort_metrics = analyze(
        rows, features, scene_to_fold, cohorts, args.bootstrap_repetitions
    )
    label_counts = Counter(row["labels"]["relative_quality_state"] for row in rows)
    summary = {
        "version": f"{args.protocol_name}_candidate_relation_feature_separability_v1",
        "protocol_name": args.protocol_name,
        "scene_count": len(scenes),
        "relation_count": len(rows),
        "numeric_feature_count": len(features),
        "numeric_feature_names": features,
        "label_counts": dict(sorted(label_counts.items())),
        "tasks": tasks,
        "cohort_metrics": cohort_metrics,
        "diagnostic_contract": {
            "relation_model_trained": False,
            "threshold_selected": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "ap_evaluation_run": False,
            "unknown_used_as_same_target_negative": False,
            "feature_direction_selected_for_action": False,
            "univariate_metrics_are_diagnostic_only": True,
        },
        "input_provenance": {
            "feature_ledger_summary_path": str((args.feature_ledger_root / "summary.json").resolve()),
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": _sha256(args.scene_list),
            "existing20_scene_list_path": str(args.existing20_scene_list.resolve()),
            "existing20_scene_list_sha256": _sha256(args.existing20_scene_list),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": actual_split_sha,
            "split_existing_scene_count": split_payload.get("existing_scene_count"),
            "split_new_scene_count": split_payload.get("new_scene_count"),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "fold_metrics.json").write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

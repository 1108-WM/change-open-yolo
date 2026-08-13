#!/usr/bin/env python3
"""审计“仅一条轨迹候选”组件动作的场景隔离折外可分性。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import EPS, _sha256, _write_jsonl  # noqa: E402
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_SPLIT_SHA256,
    MODEL_PARAMS,
    RANDOM_SEED,
    feature_matrix,
    load_training_rows,
)
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def top_precision(scores: np.ndarray, labels: np.ndarray, fraction: float) -> dict:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    order = np.argsort(-scores, kind="mergesort")[:count]
    positive = int(labels[order].sum())
    return {
        "fraction": fraction,
        "selected_count": count,
        "positive_count": positive,
        "precision": float(positive / count),
        "positive_recall": float(positive / max(1, int(labels.sum()))),
    }


def training_weights(rows: list[dict], indexes: np.ndarray, balance_positive: bool) -> np.ndarray:
    weights = np.ones(len(rows), dtype=np.float64)
    component_counts = {}
    for index in indexes:
        key = (rows[index]["scene_name"], int(rows[index]["relation_component_id"]))
        component_counts[key] = component_counts.get(key, 0) + 1
    labels = np.asarray([int(row["label_positive"]) for row in rows], dtype=np.int64)
    positive = int(labels[indexes].sum())
    negative = len(indexes) - positive
    positive_multiplier = negative / max(1, positive) if balance_positive else 1.0
    for index in indexes:
        key = (rows[index]["scene_name"], int(rows[index]["relation_component_id"]))
        weights[index] = (positive_multiplier if labels[index] else 1.0) / component_counts[key]
    weights[indexes] /= weights[indexes].mean()
    return weights


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("必须使用冻结 official100 五折")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    rows, _, feature_contract = load_training_rows(
        scenes, args.action_ledger_root, args.relation_feature_ledger_root
    )
    rows = [row for row in rows if row["action_kind"] == "track_only_one"]
    matrix, feature_names = feature_matrix(rows)
    labels = np.asarray([int(row["label_positive"]) for row in rows], dtype=np.int64)
    row_scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    variants = ("component_balanced", "component_and_positive_balanced")
    predictions = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in variants}
    fold_metrics = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        train = np.flatnonzero(np.isin(row_scenes, fold["train_scenes"]))
        validation = np.flatnonzero(np.isin(row_scenes, fold["validation_scenes"]))
        row = {"fold_index": fold_index, "validation_action_count": len(validation), "validation_positive_count": int(labels[validation].sum()), "variants": {}}
        for variant in variants:
            weights = training_weights(
                rows, train, balance_positive=variant == "component_and_positive_balanced"
            )
            model = HistGradientBoostingClassifier(
                loss="log_loss", **{**MODEL_PARAMS, "random_state": RANDOM_SEED + fold_index}
            )
            model.fit(matrix[train], labels[train], sample_weight=weights[train])
            score = model.predict_proba(matrix[validation])[:, 1]
            predictions[variant][validation] = score
            row["variants"][variant] = {
                "roc_auc": float(roc_auc_score(labels[validation], score)),
                "pr_auc": float(average_precision_score(labels[validation], score)),
                "top_precision": [top_precision(score, labels[validation], value) for value in FRACTIONS],
            }
        fold_metrics.append(row)
    if any(np.isnan(values).any() for values in predictions.values()):
        raise AssertionError("每条仅轨迹动作必须得到一次折外预测")
    overall = {}
    for variant, score in predictions.items():
        overall[variant] = {
            "roc_auc": float(roc_auc_score(labels, score)),
            "pr_auc": float(average_precision_score(labels, score)),
            "top_precision": [top_precision(score, labels, value) for value in FRACTIONS],
        }
    oof_rows = [{
        "scene_name": row["scene_name"],
        "relation_component_id": int(row["relation_component_id"]),
        "action_name": row["action_name"],
        "selected_track_id": row["selected_track_id"],
        "label_positive": bool(labels[index]),
        "label_utility": float(row["label_utility"]),
        "predictions": {name: float(values[index]) for name, values in predictions.items()},
    } for index, row in enumerate(rows)]
    summary = {
        "version": "official100_component_track_only_oof_separability_v1",
        "scene_count": 100,
        "action_count": len(rows),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "feature_contract": {**feature_contract, "feature_names": feature_names},
        "candidate_quality_oof_features_used": False,
        "overall": overall,
        "fold_metrics": fold_metrics,
        "action_threshold_selected": False,
        "inference_action_generated": False,
        "candidate_files_modified": False,
        "ap_evaluation_run": False,
        "ground_truth_usage": "official train OOF label only",
        "input_provenance": {
            "split_manifest_sha256": _sha256(args.split_manifest),
            "action_ledger_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "relation_feature_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "oof_predictions.jsonl", oof_rows)
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        if isinstance(getattr(args, name), Path):
            setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，拒绝覆盖：{args.output_root}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

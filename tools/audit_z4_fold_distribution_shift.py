#!/usr/bin/env python3
"""Audit frozen-fold feature, label, and calibration distribution shift.

This tool is read-only.  Official-train labels are used only for an explicit
post-OOF diagnostic; no model is fit and no candidate action is generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from evaluate.scannet200.scannet_constants import (  # noqa: E402
    COMMON_CATS_SCANNET_200,
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
)


SOURCES = ("native", "track", "pair_union")
BANDS = ("head", "common", "tail")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() > 0 else 0.0


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    if not len(values) or weights.sum() <= 0:
        return 0.0
    mean = _weighted_mean(values, weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def _distribution_stats(values: np.ndarray, weights: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "weight_sum": 0.0, "mean": None, "std": None, "p10": None, "median": None, "p90": None}
    return {
        "count": int(len(values)), "weight_sum": float(weights.sum()),
        "mean": _weighted_mean(values, weights),
        "std": math.sqrt(max(0.0, _weighted_var(values, weights))),
        "p10": float(np.quantile(values, 0.10)), "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _shift(train_values, validation_values, train_weights, validation_weights) -> dict:
    train_mean = _weighted_mean(train_values, train_weights)
    validation_mean = _weighted_mean(validation_values, validation_weights)
    pooled = math.sqrt(max(1e-12, 0.5 * (
        _weighted_var(train_values, train_weights) + _weighted_var(validation_values, validation_weights)
    )))
    ks = float(ks_2samp(train_values, validation_values).statistic) if len(train_values) and len(validation_values) else 0.0
    return {
        "train_mean": train_mean, "validation_mean": validation_mean,
        "mean_delta": validation_mean - train_mean,
        "absolute_standardized_mean_difference": abs(validation_mean - train_mean) / pooled,
        "ks_statistic": ks,
    }


def _band(class_index: int) -> str:
    semantic_id = int(instance_eval.PRED_ID_TO_ID[class_index])
    label = str(instance_eval.ID_TO_LABEL[semantic_id])
    if label in HEAD_CATS_SCANNET_200:
        return "head"
    if label in COMMON_CATS_SCANNET_200:
        return "common"
    if label in TAIL_CATS_SCANNET_200:
        return "tail"
    raise ValueError(f"class lacks ScanNet200 frequency band: {class_index} {label}")


def _subset(indexes: np.ndarray, rows: list[dict], source: str | None = None, band: str | None = None) -> np.ndarray:
    return np.asarray([
        index for index in indexes
        if (source is None or str(rows[index]["candidate_source"]) == source)
        and (band is None or str(rows[index]["frequency_band"]) == band)
    ], dtype=np.int64)


def _group_stats(indexes, rows, target, tp50, original, prediction, weights) -> dict:
    result = {}
    for source in (None, *SOURCES):
        source_name = "overall" if source is None else source
        source_result = {}
        for band in (None, *BANDS):
            band_name = "overall" if band is None else band
            selected = _subset(indexes, rows, source, band)
            if not len(selected):
                continue
            source_result[band_name] = {
                "row_count": int(len(selected)), "weight_sum": float(weights[selected].sum()),
                "target_mean": _weighted_mean(target[selected], weights[selected]),
                "tp50_rate": _weighted_mean(tp50[selected], weights[selected]),
                "original_score_mean": _weighted_mean(original[selected], weights[selected]),
                "prediction_mean": _weighted_mean(prediction[selected], weights[selected]) if prediction is not None else None,
                "prediction_minus_target": (
                    _weighted_mean(prediction[selected], weights[selected]) - _weighted_mean(target[selected], weights[selected])
                    if prediction is not None else None
                ),
                "mean_absolute_error": (
                    _weighted_mean(np.abs(prediction[selected] - target[selected]), weights[selected])
                    if prediction is not None else None
                ),
            }
        result[source_name] = source_result
    return result


def _binary_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def _unweighted_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, scores))


def _read_ap_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="") as handle:
        return {str(row["class"]): row for row in csv.DictReader(handle)}


def _category_ranking_audit(
    fold_index: int, validation: np.ndarray, rows: list[dict], target: np.ndarray,
    tp50: np.ndarray, original: np.ndarray, applied_prediction: np.ndarray,
    weights: np.ndarray, ap_csv_root: Path,
) -> list[dict]:
    baseline = _read_ap_csv(
        ap_csv_root / f"fold_{fold_index}__fixed_fusion_original_score__pair_union.csv"
    )
    hybrid = _read_ap_csv(
        ap_csv_root / f"fold_{fold_index}__C_joint_native_track_union_frozen_score__pair_union.csv"
    )
    id_to_pred = {int(semantic_id): int(class_index) for class_index, semantic_id in instance_eval.PRED_ID_TO_ID.items()}
    records = []
    for label in sorted(set(baseline) & set(hybrid)):
        semantic_id = int(instance_eval.LABEL_TO_ID[label])
        class_index = id_to_pred[semantic_id]
        selected = np.asarray(
            [index for index in validation if int(rows[index]["class_index"]) == class_index],
            dtype=np.int64,
        )
        source_statistics = {}
        for source in ("overall", *SOURCES):
            source_selected = selected if source == "overall" else _subset(selected, rows, source, None)
            if not len(source_selected):
                continue
            source_statistics[source] = {
                "row_count": int(len(source_selected)),
                "weight_sum": float(weights[source_selected].sum()),
                "tp50_count": int(tp50[source_selected].sum()),
                "tp50_rate": _weighted_mean(tp50[source_selected], weights[source_selected]),
                "target_mean": _weighted_mean(target[source_selected], weights[source_selected]),
                "original_score_mean": _weighted_mean(original[source_selected], weights[source_selected]),
                "applied_prediction_mean": _weighted_mean(
                    applied_prediction[source_selected], weights[source_selected]
                ),
                "original_tp50_roc_auc": _binary_auc(
                    tp50[source_selected], original[source_selected], weights[source_selected]
                ),
                "applied_tp50_roc_auc": _binary_auc(
                    tp50[source_selected], applied_prediction[source_selected], weights[source_selected]
                ),
                "original_tp50_roc_auc_unweighted": _unweighted_binary_auc(
                    tp50[source_selected], original[source_selected]
                ),
                "applied_tp50_roc_auc_unweighted": _unweighted_binary_auc(
                    tp50[source_selected], applied_prediction[source_selected]
                ),
                "tp50_scenes": sorted({
                    str(rows[index]["scene_name"]) for index in source_selected if tp50[index] > 0
                }),
            }
        baseline_ap = float(baseline[label]["ap"])
        hybrid_ap = float(hybrid[label]["ap"])
        if not math.isfinite(baseline_ap) or not math.isfinite(hybrid_ap):
            continue
        records.append({
            "fold_index": fold_index,
            "class_label": label,
            "semantic_id": semantic_id,
            "class_index": class_index,
            "frequency_band": _band(class_index),
            "baseline_ap": baseline_ap,
            "hybrid_ap": hybrid_ap,
            "ap_delta": hybrid_ap - baseline_ap,
            "ap50_delta": float(hybrid[label]["ap50"]) - float(baseline[label]["ap50"]),
            "ap25_delta": float(hybrid[label]["ap25"]) - float(baseline[label]["ap25"]),
            "source_statistics": source_statistics,
        })
    return records


def run(args: argparse.Namespace) -> dict:
    rows = _read_jsonl(args.dataset_root / "rows.jsonl")
    with np.load(args.dataset_root / "dataset.npz") as payload:
        features = np.asarray(payload["features"], dtype=np.float64)
        target = np.asarray(payload["label_ap_quality"], dtype=np.float64)
        tp50 = np.asarray(payload["label_tp50"], dtype=np.float64)
    schema = json.loads((args.dataset_root / "feature_schema.json").read_text())
    feature_names = list(schema["feature_names"])
    oof_rows = _read_jsonl(args.oof_root / "oof_predictions.jsonl")
    if len(rows) != len(features) or len(rows) != len(oof_rows):
        raise ValueError("dataset and OOF row counts disagree")
    for index, (row, oof) in enumerate(zip(rows, oof_rows)):
        identity = (row["scene_name"], row["candidate_source"], int(row["candidate_id"]))
        other = (oof["scene_name"], oof["candidate_source"], int(oof["candidate_id"]))
        if identity != other or int(row["row_index"]) != index:
            raise ValueError(f"dataset/OOF row identity mismatch at {index}")
        row["frequency_band"] = _band(int(row["class_index"]))
    joint_model_prediction = np.asarray([
        float(row["oof_predictions"]["C_joint_yolo_alpha"]) for row in oof_rows
    ], dtype=np.float64)
    weights = np.asarray([float(row["sample_weight"]) for row in oof_rows], dtype=np.float64)
    original = np.asarray([float(row["original_score"]) for row in rows], dtype=np.float64)
    applied_prediction = np.asarray([
        original[index] if str(row["candidate_source"]) == "pair_union" else joint_model_prediction[index]
        for index, row in enumerate(rows)
    ], dtype=np.float64)
    manifest = json.loads(args.split_manifest.read_text())
    ap_summary = json.loads(args.ap_summary.read_text())
    ap_folds = {int(row["fold_index"]): row for row in ap_summary["folds"]}

    feature_audit_rows, fold_summaries, scene_rows, category_rows = [], [], [], []
    for spec in sorted(manifest["folds"], key=lambda row: int(row["fold_index"])):
        fold_index = int(spec["fold_index"])
        train_scenes, validation_scenes = set(spec["train_scenes"]), set(spec["validation_scenes"])
        train = np.asarray([i for i, row in enumerate(rows) if row["scene_name"] in train_scenes], dtype=np.int64)
        validation = np.asarray([i for i, row in enumerate(rows) if row["scene_name"] in validation_scenes], dtype=np.int64)
        fold_feature_rows = []
        for source in (None, *SOURCES):
            train_source = _subset(train, rows, source, None)
            validation_source = _subset(validation, rows, source, None)
            for column, name in enumerate(feature_names):
                shift = _shift(
                    features[train_source, column], features[validation_source, column],
                    weights[train_source], weights[validation_source],
                )
                record = {
                    "fold_index": fold_index, "source": "overall" if source is None else source,
                    "feature_index": column, "feature_name": name, **shift,
                }
                feature_audit_rows.append(record); fold_feature_rows.append(record)
        top_smd = sorted(fold_feature_rows, key=lambda row: (-row["absolute_standardized_mean_difference"], -row["ks_statistic"]))[:20]
        top_ks = sorted(fold_feature_rows, key=lambda row: (-row["ks_statistic"], -row["absolute_standardized_mean_difference"]))[:20]
        ap_fold = ap_folds[fold_index]
        hybrid_delta = ap_fold["deltas_vs_fixed_fusion"]["C_joint_native_track_union_frozen_score"]
        fold_summary = {
            "fold_index": fold_index, "train_scene_count": len(train_scenes),
            "validation_scene_count": len(validation_scenes),
            "validation_scenes": sorted(validation_scenes),
            "pair_union_ap_delta_vs_fixed_fusion": hybrid_delta,
            "train_statistics": _group_stats(train, rows, target, tp50, original, None, weights),
            "validation_statistics": _group_stats(
                validation, rows, target, tp50, original, applied_prediction, weights
            ),
            "joint_model_validation_statistics": _group_stats(
                validation, rows, target, tp50, original, joint_model_prediction, weights
            ),
            "top_feature_shifts_by_smd": top_smd,
            "top_feature_shifts_by_ks": top_ks,
        }
        fold_summaries.append(fold_summary)
        category_rows.extend(_category_ranking_audit(
            fold_index, validation, rows, target, tp50, original, applied_prediction,
            weights, args.ap_summary.parent,
        ))
        for scene in sorted(validation_scenes):
            selected = np.asarray([i for i in validation if rows[i]["scene_name"] == scene], dtype=np.int64)
            scene_rows.append({
                "fold_index": fold_index, "scene_name": scene,
                "statistics": _group_stats(
                    selected, rows, target, tp50, original, applied_prediction, weights
                ),
            })

    fold4 = next(row for row in fold_summaries if row["fold_index"] == args.focus_fold)
    other_folds = [row for row in fold_summaries if row["fold_index"] != args.focus_fold]
    comparative = []
    for source in ("overall", *SOURCES):
        for band in ("overall", *BANDS):
            current = fold4["validation_statistics"].get(source, {}).get(band)
            peers = [row["validation_statistics"].get(source, {}).get(band) for row in other_folds]
            peers = [row for row in peers if row is not None]
            if current is None or not peers:
                continue
            comparative.append({
                "source": source, "band": band,
                "fold4_target_mean": current["target_mean"],
                "other_fold_target_mean": float(np.mean([row["target_mean"] for row in peers])),
                "fold4_prediction_mean": current["prediction_mean"],
                "other_fold_prediction_mean": float(np.mean([row["prediction_mean"] for row in peers])),
                "fold4_calibration_bias": current["prediction_minus_target"],
                "other_fold_calibration_bias_mean": float(np.mean([row["prediction_minus_target"] for row in peers])),
                "fold4_mae": current["mean_absolute_error"],
                "other_fold_mae_mean": float(np.mean([row["mean_absolute_error"] for row in peers])),
                "fold4_weight_fraction": current["weight_sum"] / fold4["validation_statistics"]["overall"]["overall"]["weight_sum"],
                "other_fold_weight_fraction_mean": float(np.mean([
                    row["weight_sum"] / peer["validation_statistics"]["overall"]["overall"]["weight_sum"]
                    for row, peer in zip(peers, other_folds)
                ])),
            })
    comparative.sort(key=lambda row: -abs(row["fold4_calibration_bias"] - row["other_fold_calibration_bias_mean"]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "feature_shifts.jsonl").open("w") as handle:
        for row in feature_audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "validation_scene_statistics.jsonl").open("w") as handle:
        for row in scene_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "category_ranking_audit.jsonl").open("w") as handle:
        for row in category_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    focus_category_rows = [row for row in category_rows if row["fold_index"] == args.focus_fold]
    summary = {
        "diagnostic_type": "Z4b frozen-fold feature/label/calibration distribution audit",
        "ground_truth_usage": "official_train_post_OOF_diagnostic_only",
        "model_fit_run": False, "ap_rule_generated": False, "candidate_mutation": False,
        "scene_count": len({row["scene_name"] for row in rows}), "row_count": len(rows),
        "feature_count": len(feature_names), "focus_fold": args.focus_fold,
        "fold_summaries": fold_summaries,
        "focus_fold_vs_other_validation_folds": comparative,
        "focus_fold_largest_category_losses": sorted(
            focus_category_rows, key=lambda row: row["ap_delta"]
        )[:20],
        "focus_fold_largest_category_gains": sorted(
            focus_category_rows, key=lambda row: -row["ap_delta"]
        )[:20],
        "global_source_counts": dict(Counter(str(row["candidate_source"]) for row in rows)),
        "global_band_counts": dict(Counter(str(row["frequency_band"]) for row in rows)),
        "score_contract": {
            "native": "C_joint_yolo_alpha held-out OOF prediction",
            "track": "C_joint_yolo_alpha held-out OOF prediction",
            "pair_union": "frozen original score",
        },
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--ap-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-fold", type=int, default=4)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("fold audit requires --allow-gt-diagnostics")
    for name in ("dataset_root", "oof_root", "split_manifest", "ap_summary", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

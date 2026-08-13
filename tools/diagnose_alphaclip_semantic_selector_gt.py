#!/usr/bin/env python3
"""仅离线 GT 诊断：检查 Alpha-CLIP 能否选择性纠正强基线 MVPDist 语义。

只分析与当前最终预测类别逐条一致的已有 Alpha-CLIP 特征。GT 只用于事后统计，
不会写入推理缓存、候选、模型或任何选择阈值。
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        class_id = instance_id // 1000
        if instance_id <= 0 or class_id not in valid_classes:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) >= min_region_size:
            instances.append((class_id, indices))
    return gt_ids, instances


def _load_prediction(root, scene_name, point_count):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    classes = np.load(f"{prefix}_pred_classes.npy", mmap_mode="r")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.ndim != 2 or masks.shape[0] != point_count or masks.shape[1] != len(classes):
        raise ValueError(f"{scene_name} 的预测缓存形状异常")
    return masks, np.asarray(classes, dtype=np.int64)


def _load_features(root, scene_name):
    path = root / scene_name / "multiview_object_clip_features.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("features", [])


def _top_and_margin(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        return -1, 0.0, 0.0
    order = np.argsort(-values)
    first = int(order[0])
    second = float(values[order[1]]) if len(order) > 1 else 0.0
    return first, float(values[first]), float(values[first] - second)


def _quantiles(values):
    if not values:
        return None
    return {name: float(np.quantile(values, quantile)) for name, quantile in (("q25", 0.25), ("median", 0.50), ("q75", 0.75))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--alphaclip_feature_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in ("scene_list", "prediction_cache_dir", "alphaclip_feature_root", "output_dir", "gt_instance_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    semantic_to_prediction = {int(value): int(key) for key, value in PRED_ID_TO_ID.items() if int(value) >= 0}
    rows = []
    coverage = Counter()
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, gt_instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks, classes = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
        features = _load_features(args.alphaclip_feature_root, scene_name)
        mask_sizes = np.asarray(masks.sum(axis=0), dtype=np.int64)
        best_iou = np.zeros(masks.shape[1], dtype=np.float32)
        best_semantic = np.full(masks.shape[1], -1, dtype=np.int64)
        for semantic_id, indices in gt_instances:
            intersections = np.asarray(masks[indices].sum(axis=0), dtype=np.int64)
            ious = intersections / np.maximum(1, mask_sizes + len(indices) - intersections)
            better = ious > best_iou
            best_iou[better] = ious[better]
            best_semantic[better] = semantic_id
        for record in features:
            prediction_id = int(record.get("prediction_id", -1))
            if prediction_id < 0 or prediction_id >= len(classes):
                coverage["特征预测编号越界"] += 1
                continue
            feature_class = int(record.get("pred_class_id", -1))
            if feature_class != int(classes[prediction_id]):
                coverage["特征类别与当前缓存不一致"] += 1
                continue
            alpha_class, alpha_probability, alpha_margin = _top_and_margin(record.get("clip_probs", []))
            _, alpha_logit, alpha_logit_margin = _top_and_margin(record.get("clip_logits", []))
            if alpha_class < 0:
                coverage["无有效Alpha-CLIP分数"] += 1
                continue
            gt_semantic = int(best_semantic[prediction_id])
            gt_prediction = semantic_to_prediction.get(gt_semantic, -1)
            coverage["有效对齐特征"] += 1
            rows.append(
                {
                    "scene_name": scene_name,
                    "prediction_id": prediction_id,
                    "source_kind": str(record.get("source_kind", "")),
                    "best_gt_iou": float(best_iou[prediction_id]),
                    "gt_class": str(ID_TO_LABEL.get(gt_semantic, "无有效GT")),
                    "mvpdist_class": int(classes[prediction_id]),
                    "alphaclip_class": alpha_class,
                    "mvpdist_correct": bool(gt_prediction >= 0 and int(classes[prediction_id]) == gt_prediction),
                    "alphaclip_correct": bool(gt_prediction >= 0 and alpha_class == gt_prediction),
                    "models_agree": bool(alpha_class == int(classes[prediction_id])),
                    "alphaclip_top_probability": alpha_probability,
                    "alphaclip_probability_margin": alpha_margin,
                    "alphaclip_top_logit": alpha_logit,
                    "alphaclip_logit_margin": alpha_logit_margin,
                    "mvpdist_raw_score": float(record.get("pred_score", 0.0)),
                }
            )

    fieldnames = list(rows[0]) if rows else []
    with (args.output_dir / "aligned_prediction_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"diagnostic_only": True, "coverage": dict(sorted(coverage.items())), "thresholds": {}}
    for threshold in (0.25, 0.50):
        eligible = [row for row in rows if row["best_gt_iou"] >= threshold and row["gt_class"] != "无有效GT"]
        groups = Counter()
        margins = {"Alpha纠正MVPDist错误": [], "Alpha破坏MVPDist正确": []}
        for row in eligible:
            if row["mvpdist_correct"] and row["alphaclip_correct"]:
                group = "两者正确"
            elif not row["mvpdist_correct"] and row["alphaclip_correct"]:
                group = "Alpha纠正MVPDist错误"
            elif row["mvpdist_correct"] and not row["alphaclip_correct"]:
                group = "Alpha破坏MVPDist正确"
            else:
                group = "两者错误"
            groups[group] += 1
            if group in margins:
                margins[group].append(row["alphaclip_logit_margin"])
        disagreements = [row for row in eligible if not row["models_agree"]]
        summary["thresholds"][f"{threshold:.2f}"] = {
            "eligible_geometrically_correct_predictions": len(eligible),
            "groups": dict(sorted(groups.items())),
            "group_rates": {name: float(count / max(1, len(eligible)) ) for name, count in sorted(groups.items())},
            "model_disagreement_count": len(disagreements),
            "model_disagreement_rate": float(len(disagreements) / max(1, len(eligible))),
            "alphaclip_logit_margin_quantiles": {name: _quantiles(values) for name, values in margins.items()},
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

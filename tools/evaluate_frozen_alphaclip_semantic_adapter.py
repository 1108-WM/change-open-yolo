#!/usr/bin/env python3
"""在冻结最终预测上评测 Alpha-CLIP 语义适配器。

输入 mask、候选、融合结果和评分均来自已有缓存；本脚本只按已有 Alpha-CLIP
特征改类别。GT 只由官方评测器在最终 AP 计算时读取，不参与类别决策。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import evaluate_scannet200


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_prediction(root, scene_name):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    scores = np.load(f"{prefix}_pred_scores.npy", mmap_mode="r")
    classes = np.load(f"{prefix}_pred_classes.npy", mmap_mode="r")
    if masks.ndim != 2 or masks.shape[1] != len(classes) or len(classes) != len(scores):
        raise ValueError(f"{scene_name} 的冻结预测缓存形状异常")
    return masks, np.asarray(scores, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def _load_features(root, scene_name):
    path = root / scene_name / "multiview_object_clip_features.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("features", [])


def _top_and_margin(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        return -1, 0.0
    order = np.argsort(-values)
    top = int(order[0])
    second = float(values[order[1]]) if len(order) > 1 else 0.0
    return top, float(values[top] - second)


def _metric_summary(path):
    values = {"ap": [], "ap50": [], "ap25": []}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            for name in values:
                value = float(row[name])
                if not np.isnan(value):
                    values[name].append(value)
    return {name: float(np.mean(items)) if items else float("nan") for name, items in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--alphaclip_feature_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "conservative"), required=True)
    parser.add_argument("--min_logit_margin", type=float, default=1.0)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    args = parser.parse_args()
    if args.min_logit_margin < 0.0:
        raise SystemExit("--min_logit_margin 必须非负。")
    for name in ("scene_list", "prediction_cache_dir", "alphaclip_feature_root", "output_dir", "gt_instance_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    scene_reports = {}
    total_changes = 0
    total_features = 0
    for scene_name in _read_scenes(args.scene_list):
        masks, scores, classes = _load_prediction(args.prediction_cache_dir, scene_name)
        adapted_classes = classes.copy()
        report = {"features": 0, "changed": [], "skipped": []}
        for record in _load_features(args.alphaclip_feature_root, scene_name):
            prediction_id = int(record.get("prediction_id", -1))
            if prediction_id < 0 or prediction_id >= len(classes):
                report["skipped"].append({"prediction_id": prediction_id, "reason": "prediction_id_out_of_range"})
                continue
            if int(record.get("pred_class_id", -1)) != int(classes[prediction_id]):
                raise ValueError(f"{scene_name} 的 Alpha-CLIP 特征类别与冻结预测不一致，拒绝混用。")
            report["features"] += 1
            alpha_class, logit_margin = _top_and_margin(record.get("clip_logits", []))
            if alpha_class < 0:
                report["skipped"].append({"prediction_id": prediction_id, "reason": "missing_logits"})
                continue
            if args.mode == "conservative" and logit_margin < args.min_logit_margin:
                report["skipped"].append(
                    {"prediction_id": prediction_id, "reason": "below_logit_margin", "logit_margin": logit_margin}
                )
                continue
            if alpha_class != int(classes[prediction_id]):
                adapted_classes[prediction_id] = alpha_class
                report["changed"].append(
                    {
                        "prediction_id": prediction_id,
                        "old_class": int(classes[prediction_id]),
                        "new_class": alpha_class,
                        "logit_margin": logit_margin,
                    }
                )
        total_features += report["features"]
        total_changes += len(report["changed"])
        scene_reports[scene_name] = report
        predictions[scene_name] = {
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_classes": adapted_classes,
        }

    suffix = "all" if args.mode == "all" else f"margin{args.min_logit_margin:g}"
    csv_path = args.output_dir / f"alphaclip_{suffix}.csv"
    evaluate_scannet200(predictions, str(args.gt_instance_dir), output_file=str(csv_path), dataset="scannet200")
    summary = {
        "inference_without_gt": True,
        "mode": args.mode,
        "min_logit_margin": args.min_logit_margin if args.mode == "conservative" else None,
        "feature_count": total_features,
        "changed_class_count": total_changes,
        "metrics": _metric_summary(csv_path),
        "scene_reports": scene_reports,
    }
    (args.output_dir / f"alphaclip_{suffix}_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "scene_reports"}, indent=2))


if __name__ == "__main__":
    main()

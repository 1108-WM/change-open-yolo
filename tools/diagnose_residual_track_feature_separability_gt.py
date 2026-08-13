#!/usr/bin/env python3
"""仅离线 GT 诊断：检查无 GT 轨迹特征能否区分可用残差。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "support_view_count",
    "point_count",
    "mean_edge_score",
    "min_mutual_projection_support",
    "mean_node_quality",
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _auc_high_value(values, labels):
    positives = values[labels]
    negatives = values[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def _feature_summary(rows):
    labels = np.asarray([row["来源归因"] == "同类可用覆盖" for row in rows], dtype=bool)
    payload = {}
    for feature in FEATURES:
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        high_value_auc = _auc_high_value(values, labels)
        direction = "高值更像可用轨迹" if high_value_auc >= 0.5 else "低值更像可用轨迹"
        payload[feature] = {
            "可用轨迹中位数": float(np.median(values[labels])) if labels.any() else None,
            "其余轨迹中位数": float(np.median(values[~labels])) if (~labels).any() else None,
            "高值方向 AUC": high_value_auc,
            "最佳单调方向 AUC": max(high_value_auc, 1.0 - high_value_auc),
            "更有利方向": direction,
        }
    return {
        "可用轨迹数": int(labels.sum()),
        "其余轨迹数": int((~labels).sum()),
        "特征": payload,
        "限制": (
            "可用轨迹数量很少。AUC 只用于离线判断特征是否值得进入后续质量模型，"
            "不能据此选择推理阈值、训练标签或最终候选。"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_attribution_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT-only 诊断绝不进入推理。")
    args.source_attribution_csv = _resolve(args.source_attribution_csv)
    args.output_dir = _resolve(args.output_dir)
    if not args.source_attribution_csv.is_file():
        raise SystemExit(f"缺少残差来源归因账本：{args.source_attribution_csv}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")

    with args.source_attribution_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    payload = {
        "诊断限定": "输入与输出均为 GT-only 离线诊断，绝不进入推理、候选、融合、评分或阈值。",
        "输入来源归因账本": str(args.source_attribution_csv),
        "汇总": _feature_summary(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

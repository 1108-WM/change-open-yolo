#!/usr/bin/env python3
"""仅离线 GT 归因：拆解残差轨迹的错误类别、碎片与背景来源。

输入是 ``diagnose_residual_tracks_gt.py`` 生成的 GT-only 账本。本脚本不读取原始
GT，但其输出仍然只能用于研究归因，绝不允许被推理、关联、候选、融合或阈值代码读取。
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _float(row, key):
    return float(row[key])


def _int(row, key):
    return int(row[key])


def _source_type(row):
    """给轨迹做互斥的离线来源归因，不返回任何推理规则。"""
    has_same_class_gt = _int(row, "best_same_class_gt_instance_id") > 0
    has_any_gt = _int(row, "best_gt_instance_id") > 0
    if has_same_class_gt and _float(row, "best_same_class_gt_iou") >= 0.25:
        return "同类可用覆盖"
    if has_same_class_gt:
        return "同类局部碎片"
    if has_any_gt and _float(row, "best_gt_precision") >= 0.50:
        return "主导异类实例"
    if has_any_gt:
        return "混合或低精度区域"
    return "无有效实例区域"


def _summary(rows):
    source_counts = Counter(row["来源归因"] for row in rows)
    state_by_source = defaultdict(Counter)
    for row in rows:
        state_by_source[row["来源归因"]][row["matched_gt_residual_type"]] += 1

    confusion = Counter(
        (row["class_name"], row["best_gt_class"])
        for row in rows
        if row["来源归因"] == "主导异类实例"
    )
    total = len(rows)
    categories = {
        name: {
            "轨迹数": source_counts[name],
            "比例": float(source_counts[name] / max(1, total)),
            "强基线状态": dict(sorted(state_by_source[name].items())),
        }
        for name in ("无有效实例区域", "混合或低精度区域", "主导异类实例", "同类局部碎片", "同类可用覆盖")
    }
    return {
        "轨迹总数": total,
        "来源归因": categories,
        "主导异类实例的常见类别混淆": [
            {"预测类别": prediction, "真实主导类别": gt_class, "轨迹数": count}
            for (prediction, gt_class), count in confusion.most_common(20)
        ],
        "结论": (
            "该划分只说明当前残差源的失败来源，不能转化为推理阈值。若无有效实例、混合区域和"
            "主导异类实例占多数，优先改进二维对象证据与残差排他定义；若同类局部碎片占多数，"
            "才优先研究跨帧覆盖补全或三维提升原子。"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_ledger_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT-only 归因绝不进入推理。")
    args.track_ledger_csv = _resolve(args.track_ledger_csv)
    args.output_dir = _resolve(args.output_dir)
    if not args.track_ledger_csv.is_file():
        raise SystemExit(f"缺少轨迹 GT-only 账本：{args.track_ledger_csv}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")

    with args.track_ledger_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["来源归因"] = _source_type(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "residual_track_source_attribution_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "输入与输出均为 GT-only 离线归因，绝不进入推理、关联、候选、融合、评分或阈值。",
        "输入轨迹账本": str(args.track_ledger_csv),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

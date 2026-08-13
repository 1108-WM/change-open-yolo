#!/usr/bin/env python3
"""GT-only：审计正反可见性证据是否区分几何可靠的目标候选区域。

此工具只关联已经固定的无 GT 证据账本和既有自动轨迹 GT 账本。GT 不参与
观测匹配、superpoint 证据、候选形成、阈值、排序、融合或 AP。
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_evidence(root):
    rows = []
    for path in sorted(root.glob("*/visibility_counterevidence_ledger.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


def _join_rows(evidence_rows, track_gt_rows):
    gt_by_key = {(row["scene_name"], int(row["track_id"])): row for row in track_gt_rows}
    joined = []
    for evidence in evidence_rows:
        key = (evidence["scene_name"], int(evidence["track_id"]))
        gt = gt_by_key.get(key)
        if gt is None:
            raise ValueError(f"无 GT 证据轨迹未找到对应的既有 GT 账本：{key}")
        joined.append({
            **evidence,
            "best_gt_instance_id": int(gt["best_gt_instance_id"]),
            "best_gt_iou": float(gt["best_gt_iou"]),
            "best_gt_precision": float(gt["best_gt_precision"]),
            "matched_gt_residual_type": gt["matched_gt_residual_type"],
        })
    return joined


def _describe(rows, fields):
    result = {"轨迹数": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "均值": float(statistics.fmean(values)) if values else 0.0,
            "中位数": float(statistics.median(values)) if values else 0.0,
        }
    return result


def _summary(rows):
    good = [row for row in rows if row["best_gt_iou"] >= 0.25 and row["best_gt_instance_id"] > 0]
    target = [
        row for row in good
        if row["matched_gt_residual_type"] in {"无合格三维候选", "边界不足"}
    ]
    fields = (
        "track_support_view_count",
        "visible_frame_count",
        "matched_class_observation_frame_count",
        "positive_support_frame_count",
        "counterevidence_eligible_frame_count",
        "mean_selected_coverage",
        "mean_selected_quality",
        "positive_superpoint_count",
        "negative_margin_superpoint_count",
        "mean_superpoint_evidence_margin",
        "top_native_iou",
        "track_inside_top_native_ratio",
        "native_candidate_overlap_count",
    )
    return {
        "说明": "所有分组只作固定无 GT 特征的离线描述；不得据此用 GT 选择阈值或写候选。",
        "全部轨迹": _describe(rows, fields),
        "几何合格轨迹（最佳 GT IoU≥25%）": _describe(good, fields),
        "几何合格且对应基线缺口的轨迹": _describe(target, fields),
        "基线缺口类型计数": {
            name: sum(row["matched_gt_residual_type"] == name for row in target)
            for name in ("无合格三维候选", "边界不足")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_root", type=Path, required=True)
    parser.add_argument("--track_gt_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线特征审计。")
    for name in ("evidence_root", "track_gt_ledger", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = _join_rows(_load_evidence(args.evidence_root), _read_csv(args.track_gt_ledger))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "track_id"]
    with (args.output_dir / "visibility_counterevidence_gt_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于审计固定正反可见性证据的可分性；不反向修改观测、候选、阈值、排序、融合或 AP。",
        "轨迹数": len(rows),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

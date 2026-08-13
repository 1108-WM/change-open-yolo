#!/usr/bin/env python3
"""仅离线 GT 账本：检查自动轨迹与候选的多视角关系特征是否有区分力。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_RESIDUALS = {"无合格三维候选", "边界不足"}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_relation_rows(root):
    rows = []
    for path in sorted(root.glob("*/automatic_track_multiview_relations.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


def _join_rows(relation_rows, ledger_rows):
    ledger = {(row["scene_name"], int(row["track_id"])): row for row in ledger_rows}
    joined = []
    for relation in relation_rows:
        key = (relation["scene_name"], int(relation["track_id"]))
        row = ledger.get(key)
        if row is None:
            raise ValueError(f"多视角关系账本中的轨迹未在 GT 账本中找到：{key}")
        joined.append({
            **relation,
            "best_gt_instance_id": int(row["best_gt_instance_id"]),
            "best_gt_iou": float(row["best_gt_iou"]),
            "matched_gt_residual_type": row["matched_gt_residual_type"],
        })
    return joined


def _target(row):
    return row["best_gt_iou"] >= 0.25 and row["matched_gt_residual_type"] in TARGET_RESIDUALS


def _bucket_summary(rows, name, bucket):
    groups = defaultdict(list)
    for row in rows:
        groups[bucket(row)].append(row)
    result = {}
    for group, subset in sorted(groups.items()):
        positives = [row for row in subset if _target(row)]
        instances = {(row["scene_name"], row["best_gt_instance_id"]) for row in positives if row["best_gt_instance_id"] > 0}
        result[group] = {
            "轨迹数": len(subset),
            "目标几何合格轨迹数": len(positives),
            "目标独立 GT 实例数": len(instances),
            "目标轨迹比例": float(len(positives) / max(1, len(subset))),
        }
    return {name: result}


def _summary(rows):
    result = {
        "轨迹数": len(rows),
        "目标定义": "最佳 GT IoU 不低于 25%，且强基线状态为无合格三维候选或边界不足。",
        "目标几何合格轨迹数": sum(_target(row) for row in rows),
    }
    result.update(_bucket_summary(
        rows, "与同一候选的跨帧支持比例", lambda row: (
            "无候选共现" if row["top_candidate_support_view_count"] == 0 else
            "低于一半视角" if row["top_candidate_support_view_ratio"] < 0.50 else
            "至少一半但未覆盖全部视角" if row["top_candidate_support_view_ratio"] < 1.0 else "全部支持视角"
        )
    ))
    result.update(_bucket_summary(
        rows, "候选身份间隔", lambda row: (
            "无候选共现" if row["top_candidate_support_view_count"] == 0 else
            "并列或无间隔" if row["candidate_identity_margin"] == 0 else
            "低于一半" if row["candidate_identity_margin"] < 0.50 else "至少一半"
        )
    ))
    result.update(_bucket_summary(
        rows, "竞争候选数", lambda row: (
            "0" if row["candidate_with_support_count"] == 0 else
            "1" if row["candidate_with_support_count"] == 1 else
            "2至4" if row["candidate_with_support_count"] <= 4 else "5及以上"
        )
    ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation_root", type=Path, required=True)
    parser.add_argument("--track_gt_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线特征审计。")
    for name in ("relation_root", "track_gt_ledger", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = _join_rows(_read_relation_rows(args.relation_root), _read_csv(args.track_gt_ledger))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "automatic_track_multiview_relation_gt_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于检查固定关系特征是否可分；不反向挑选阈值，不进入自动 mask、轨迹、候选、融合、评分或评测。",
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""仅离线 GT 账本：审计自动轨迹相对现有候选的几何分流是否可信。"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("新增实例", "边界竞争", "内部重复或冲突")
RESIDUAL_TYPES = ("无合格三维候选", "边界不足", "已有严格三维候选")


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_relation_rows(root):
    rows = []
    for path in sorted(root.glob("*/automatic_track_geometry_relations.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


def _join_rows(relation_rows, ledger_rows):
    ledger_by_key = {(row["scene_name"], int(row["track_id"])): row for row in ledger_rows}
    joined = []
    for relation in relation_rows:
        key = (relation["scene_name"], int(relation["track_id"]))
        ledger = ledger_by_key.get(key)
        if ledger is None:
            raise ValueError(f"关系账本中的轨迹未在 GT 账本中找到：{key}")
        joined.append({
            **relation,
            "best_gt_instance_id": int(ledger["best_gt_instance_id"]),
            "best_gt_iou": float(ledger["best_gt_iou"]),
            "matched_gt_residual_type": ledger["matched_gt_residual_type"],
        })
    return joined


def _route_summary(rows):
    result = {}
    for route in ROUTES:
        subset = [row for row in rows if row["route"] == route]
        good = [row for row in subset if row["best_gt_iou"] >= 0.25 and row["best_gt_instance_id"] > 0]
        instances = defaultdict(set)
        for row in good:
            residual_type = row["matched_gt_residual_type"]
            if residual_type in RESIDUAL_TYPES:
                instances[residual_type].add((row["scene_name"], row["best_gt_instance_id"]))
        result[route] = {
            "轨迹数": len(subset),
            "几何合格轨迹数": len(good),
            "几何合格轨迹中强基线状态": dict(sorted(Counter(row["matched_gt_residual_type"] for row in good).items())),
            "几何合格轨迹对应的独立 GT 实例数": {
                residual_type: len(instances[residual_type]) for residual_type in RESIDUAL_TYPES
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation_root", type=Path, required=True)
    parser.add_argument("--track_gt_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线分流审计。")
    for name in ("relation_root", "track_gt_ledger", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")

    rows = _join_rows(_read_relation_rows(args.relation_root), _read_csv(args.track_gt_ledger))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "automatic_track_geometry_relation_gt_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线审计固定的无 GT 几何分流；不反向修改阈值，绝不进入自动 mask、轨迹、候选、融合、评分或评测。",
        "轨迹数": len(rows),
        "按分流的 GT 审计": _route_summary(rows),
        "解释": {
            "新增实例": "理想情况下，几何合格轨迹应主要对应‘无合格三维候选’。",
            "边界竞争": "理想情况下，几何合格轨迹应主要对应‘边界不足’。",
            "内部重复或冲突": "理想情况下，几何合格轨迹应主要对应‘已有严格三维候选’。",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

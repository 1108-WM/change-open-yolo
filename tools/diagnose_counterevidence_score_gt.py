#!/usr/bin/env python3
"""GT-only：按独立场景划分审计可信反证据分数的排序与安全性。

无 GT 分数已经固定后，本工具只联接既有逐 superpoint GT 审计。场景以字典序
交替分入 A/B，避免同一场景的重复 superpoint 同时出现在两个组。结果仅用于
验证排序，不反向修改分数、权重、阈值、候选或 AP。
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_scores(root):
    rows = []
    for path in sorted(root.glob("*/counterevidence_reliability_scores.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _scene_split(scene_name, all_scenes):
    return "A" if sorted(all_scenes).index(scene_name) % 2 == 0 else "B"


def _quintiles(rows):
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["counterevidence_reliability_score"]))
    result = []
    for index in range(5):
        left = math.floor(index * len(ordered) / 5)
        right = math.floor((index + 1) * len(ordered) / 5)
        subset = ordered[left:right]
        usable = [row for row in subset if row["gt_removal_outcome"] != "无固定有效 GT"]
        result.append({
            "score_quintile_low_to_high": index + 1,
            "superpoint_count": len(subset),
            "mean_score": float(np.mean([row["counterevidence_reliability_score"] for row in subset])) if subset else 0.0,
            "mean_gt_iou_delta_if_removed": float(np.mean([row["gt_iou_delta_if_removed"] for row in usable])) if usable else 0.0,
            "beneficial_removal_rate": float(np.mean([row["gt_removal_outcome"] == "移除有益" for row in usable])) if usable else 0.0,
        })
    return result


def _describe(rows):
    usable = [row for row in rows if row["gt_removal_outcome"] != "无固定有效 GT"]
    if len(usable) >= 3:
        correlation, pvalue = spearmanr(
            [row["counterevidence_reliability_score"] for row in usable],
            [row["gt_iou_delta_if_removed"] for row in usable],
        )
    else:
        correlation, pvalue = 0.0, 1.0
    return {
        "superpoint_count": len(rows),
        "usable_gt_count": len(usable),
        "score_to_gt_iou_delta_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "pvalue": float(pvalue) if np.isfinite(pvalue) else 1.0,
        "quintiles": _quintiles(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_root", type=Path, required=True)
    parser.add_argument("--gt_audit_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线排序审计。")
    for name in ("score_root", "gt_audit_csv", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    gt_rows = _read_csv(args.gt_audit_csv)
    gt_by_key = {(row["scene_name"], int(row["track_id"]), int(row["superpoint_id"])): row for row in gt_rows}
    rows = []
    for score in _load_scores(args.score_root):
        key = (score["scene_name"], int(score["track_id"]), int(score["superpoint_id"]))
        gt = gt_by_key.get(key)
        if gt is None:
            raise ValueError(f"固定 score 找不到 GT-only 审计记录：{key}")
        rows.append({
            **score,
            "gt_iou_delta_if_removed": float(gt["gt_iou_delta_if_removed"]),
            "gt_removal_outcome": gt["gt_removal_outcome"],
            "original_track_best_gt_iou": float(gt["original_track_best_gt_iou"]),
            "matched_gt_residual_type": gt["matched_gt_residual_type"],
        })
    scenes = sorted({row["scene_name"] for row in rows})
    for row in rows:
        row["scene_split"] = _scene_split(row["scene_name"], scenes)
    negative = [row for row in rows if row["is_negative_margin"]]
    payload = {
        "诊断限定": "分数完全由无 GT 特征固定；GT 仅离线验证按场景拆分后的排序，不能回调权重、阈值、删点、候选、融合或 AP。",
        "scene_split_rule": "按 scene_name 字典序交替分为 A/B；同一场景不会跨组。",
        "all_negative_margin": {split: _describe([row for row in negative if row["scene_split"] == split]) for split in ("A", "B")},
        "target_negative_margin": {split: _describe([
            row for row in negative if row["scene_split"] == split and row["original_track_best_gt_iou"] >= 0.25 and row["matched_gt_residual_type"] in {"无合格三维候选", "边界不足"}
        ]) for split in ("A", "B")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "track_id", "superpoint_id"]
    with (args.output_dir / "counterevidence_score_gt_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GT-only：审计固定反证据 superpoint 选择对原轨迹几何的影响。

本工具只比较已经固定的无 GT 原轨迹与几何变体；GT 不参与 superpoint 选择、
候选形成、类别、阈值、排序、融合或 AP。每条变体只针对原轨迹已匹配的 GT
实例计算 IoU，避免重新匹配后掩盖几何变化。
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_variants(root):
    rows = []
    for path in sorted(root.glob("*/counterevidence_superpoint_variants.json")):
        rows.extend(json.loads(path.read_text()))
    return rows


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {
        int(instance_id): int(np.sum(ids == instance_id))
        for instance_id in np.unique(ids)
        if int(instance_id) > 0
        and int(instance_id) // 1000 in valid_classes
        and int(np.sum(ids == instance_id)) >= min_region_size
    }
    return ids, sizes


def _iou_to_fixed_gt(points, gt_ids, instance_id, instance_size):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if instance_id <= 0 or instance_id not in instance_size:
        return 0.0
    intersection = int(np.sum(gt_ids[points] == instance_id)) if len(points) else 0
    return float(intersection / max(1, len(points) + instance_size[instance_id] - intersection))


def _scene_rows(scene_name, variants, gt_by_key, args):
    gt_ids, sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    rows = []
    for variant in variants:
        key = (scene_name, int(variant["track_id"]))
        original = gt_by_key.get(key)
        if original is None:
            raise ValueError(f"变体未找到既有原轨迹 GT 账本：{key}")
        instance_id = int(original["best_gt_instance_id"])
        points = np.asarray(np.load(variant["variant_points_path"])["point_indices"], dtype=np.int64)
        variant_iou = _iou_to_fixed_gt(points, gt_ids, instance_id, sizes)
        original_iou = float(original["best_gt_iou"])
        rows.append({
            **variant,
            "fixed_gt_instance_id": instance_id,
            "original_best_gt_iou": original_iou,
            "variant_iou_to_fixed_gt": variant_iou,
            "iou_delta": float(variant_iou - original_iou),
            "matched_gt_residual_type": original["matched_gt_residual_type"],
        })
    return rows


def _describe(rows):
    deltas = [float(row["iou_delta"]) for row in rows]
    return {
        "轨迹数": len(rows),
        "变体 IoU 均值": float(statistics.fmean(float(row["variant_iou_to_fixed_gt"]) for row in rows)) if rows else 0.0,
        "IoU 变化均值": float(statistics.fmean(deltas)) if deltas else 0.0,
        "IoU 变化中位数": float(statistics.median(deltas)) if deltas else 0.0,
        "IoU 提升轨迹数": sum(delta > 0 for delta in deltas),
        "IoU 不变轨迹数": sum(delta == 0 for delta in deltas),
        "IoU 下降轨迹数": sum(delta < 0 for delta in deltas),
        "变体保持 IoU≥25% 的轨迹数": sum(float(row["variant_iou_to_fixed_gt"]) >= 0.25 for row in rows),
    }


def _summary(rows):
    geometric = [row for row in rows if float(row["original_best_gt_iou"]) >= 0.25 and int(row["fixed_gt_instance_id"]) > 0]
    target = [row for row in geometric if row["matched_gt_residual_type"] in {"无合格三维候选", "边界不足"}]
    changed = [row for row in rows if int(row["removed_point_count"]) > 0]
    return {
        "说明": "以下仅描述已固定无 GT 变体的离线几何影响；不得据此将 GT 用于挑选规则或形成候选。",
        "全部变体": _describe(rows),
        "发生几何裁剪的变体": _describe(changed),
        "原轨迹几何合格（IoU≥25%）": _describe(geometric),
        "原轨迹几何合格且对应基线缺口": _describe(target),
        "基线缺口类型计数": {name: sum(row["matched_gt_residual_type"] == name for row in target) for name in ("无合格三维候选", "边界不足")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant_root", type=Path, required=True)
    parser.add_argument("--track_gt_ledger", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线几何审计。")
    for name in ("variant_root", "track_gt_ledger", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    gt_rows = _read_csv(args.track_gt_ledger)
    gt_by_key = {(row["scene_name"], int(row["track_id"])): row for row in gt_rows}
    variants_by_scene = {}
    for row in _load_variants(args.variant_root):
        variants_by_scene.setdefault(row["scene_name"], []).append(row)
    rows = []
    for scene_name in sorted(variants_by_scene):
        rows.extend(_scene_rows(scene_name, variants_by_scene[scene_name], gt_by_key, args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "track_id"]
    with (args.output_dir / "counterevidence_superpoint_variants_gt_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于审计固定无 GT 几何变体，不反向影响 superpoint 选择、候选、类别、阈值、排序、融合或 AP。",
        "轨迹数": len(rows),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

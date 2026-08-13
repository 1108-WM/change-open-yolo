#!/usr/bin/env python3
"""仅离线 GT 账本：评估类别无关自动 mask 轨迹的三维对象覆盖。"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id > 0 and instance_id // 1000 in valid:
            size = int(np.sum(gt_ids == instance_id))
            if size >= min_region_size:
                sizes[instance_id] = size
    return gt_ids, sizes


def _load_prediction(root, scene_name, point_count):
    masks = np.load(f"{root / scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的最终预测点数不匹配")
    return np.asarray(masks, dtype=bool)


def _final_iou_by_gt(masks, gt_ids, gt_sizes):
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    result = {}
    for instance_id, size in gt_sizes.items():
        indices = np.flatnonzero(gt_ids == instance_id)
        if masks.shape[1] == 0:
            result[instance_id] = 0.0
            continue
        intersection = masks[indices].sum(axis=0, dtype=np.int64)
        result[instance_id] = float(np.max(intersection / np.maximum(1, mask_sizes + size - intersection)))
    return result


def _best_gt(points, gt_ids, gt_sizes):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if len(points) == 0:
        return {"instance_id": -1, "iou": 0.0, "precision": 0.0, "coverage": 0.0}
    ids, counts = np.unique(gt_ids[points], return_counts=True)
    candidates = []
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        intersection = int(intersection)
        size = gt_sizes[instance_id]
        candidates.append({
            "instance_id": instance_id,
            "iou": float(intersection / max(1, len(points) + size - intersection)),
            "precision": float(intersection / max(1, len(points))),
            "coverage": float(intersection / max(1, size)),
        })
    return max(candidates, key=lambda item: item["iou"], default={"instance_id": -1, "iou": 0.0, "precision": 0.0, "coverage": 0.0})


def _residual_type(iou):
    if iou < 0.25:
        return "无合格三维候选"
    if iou < 0.50:
        return "边界不足"
    return "已有严格三维候选"


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
    final_iou = _final_iou_by_gt(masks, gt_ids, gt_sizes)
    payload = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())
    rows = []
    for track in payload["tracks"]:
        points = np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        match = _best_gt(points, gt_ids, gt_sizes)
        instance_id = int(match["instance_id"])
        baseline_iou = float(final_iou.get(instance_id, 0.0)) if instance_id > 0 else 0.0
        rows.append({
            "scene_name": scene_name,
            "track_id": int(track["track_id"]),
            "support_view_count": int(track["support_view_count"]),
            "point_count": int(track["point_count"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "mean_predicted_iou": float(track["mean_predicted_iou"]),
            "mean_stability_score": float(track["mean_stability_score"]),
            "mean_edge_score": float(track["mean_edge_score"]),
            "best_gt_instance_id": instance_id,
            "best_gt_class": ID_TO_LABEL.get(instance_id // 1000, "无有效实例") if instance_id > 0 else "无有效实例",
            "best_gt_iou": float(match["iou"]),
            "best_gt_precision": float(match["precision"]),
            "best_gt_coverage": float(match["coverage"]),
            "matched_gt_final_iou": baseline_iou,
            "matched_gt_residual_type": _residual_type(baseline_iou) if instance_id > 0 else "无有效 GT",
        })
    return rows


def _summary(rows):
    def count(predicate):
        return sum(bool(predicate(row)) for row in rows)

    total = len(rows)
    indicators = {}
    for name, predicate in {
        "最佳 GT IoU 不低于 25%": lambda row: row["best_gt_iou"] >= 0.25,
        "最佳 GT IoU 不低于 50%": lambda row: row["best_gt_iou"] >= 0.50,
        "最佳 GT 点精度不低于 50%": lambda row: row["best_gt_precision"] >= 0.50,
    }.items():
        value = count(predicate)
        indicators[name] = {"轨迹数": value, "比例": float(value / max(1, total))}
    recovered = defaultdict(set)
    for row in rows:
        if row["best_gt_iou"] >= 0.25 and row["best_gt_instance_id"] > 0:
            recovered[row["matched_gt_residual_type"]].add((row["scene_name"], row["best_gt_instance_id"]))
    views = {}
    for name, predicate in {"仅两视角": lambda row: row["support_view_count"] == 2, "至少三视角": lambda row: row["support_view_count"] >= 3}.items():
        subset = [row for row in rows if predicate(row)]
        views[name] = {
            "轨迹数": len(subset),
            "最佳 GT IoU 不低于 25% 的轨迹数": count(lambda row: predicate(row) and row["best_gt_iou"] >= 0.25),
        }
    return {
        "类别无关自动 mask 轨迹数": total,
        "指标": indicators,
        "匹配 GT 后的强基线状态": dict(sorted(Counter(row["matched_gt_residual_type"] for row in rows).items())),
        "最佳 GT IoU 不低于 25% 的独立实例数": {key: len(value) for key, value in sorted(recovered.items())},
        "按视角数分组": views,
        "说明": "GT 仅用于离线评估无 GT 轨迹；轨迹未赋类别、未生成候选，不能代表 AP。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int, help="仅用于小规模 GT-only 冒烟诊断。")
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线轨迹归因。")
    for name in ("scene_list", "track_root", "prediction_cache_dir", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = []
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条自动轨迹", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "automatic_mask_track_features_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线轨迹归因，绝不进入自动 mask 导出、关联、语义、候选、融合、评分或阈值。",
        "scene_count": len(scenes),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""仅离线 GT 账本：评估多视角残差轨迹能否补足 native 强基线。

本脚本只读取已经构建完成的残差轨迹。GT 只用于比较轨迹与真实实例的对应、覆盖和
强基线既有几何状态；输出不得被推理、关联、阈值或候选生成代码读取。
"""

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

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        class_id = instance_id // 1000
        if instance_id > 0 and class_id in valid_classes:
            size = int(np.sum(gt_ids == instance_id))
            if size >= min_region_size:
                sizes[instance_id] = size
    return gt_ids, sizes


def _load_prediction(root, scene_name, point_count):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    classes = np.load(f"{prefix}_pred_classes.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count or masks.shape[1] != len(classes):
        raise ValueError(f"{scene_name} 的最终预测缓存与点数不一致")
    semantic_classes = np.asarray([PRED_ID_TO_ID.get(int(value), -1) for value in classes], dtype=np.int64)
    return np.asarray(masks, dtype=bool), semantic_classes


def _final_iou_by_gt(masks, classes, gt_ids, gt_sizes):
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    result = {}
    for instance_id, size in gt_sizes.items():
        indices = np.flatnonzero(gt_ids == instance_id)
        if masks.shape[1] == 0:
            result[instance_id] = {"any": 0.0, "same_class": 0.0}
            continue
        intersection = masks[indices].sum(axis=0, dtype=np.int64)
        ious = intersection / np.maximum(1, mask_sizes + size - intersection)
        class_id = instance_id // 1000
        same = ious[classes == class_id]
        result[instance_id] = {
            "any": float(np.max(ious)) if len(ious) else 0.0,
            "same_class": float(np.max(same)) if len(same) else 0.0,
        }
    return result


def _best_gt_for_points(point_indices, gt_ids, gt_sizes, prediction_class_id):
    point_indices = np.unique(np.asarray(point_indices, dtype=np.int64))
    if len(point_indices) == 0:
        return {"best_gt_instance_id": -1, "best_gt_iou": 0.0, "best_gt_precision": 0.0, "best_gt_coverage": 0.0,
                "best_same_class_gt_instance_id": -1, "best_same_class_gt_iou": 0.0,
                "best_same_class_gt_precision": 0.0, "best_same_class_gt_coverage": 0.0}
    ids, counts = np.unique(gt_ids[point_indices], return_counts=True)
    candidates = []
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        size = int(gt_sizes[instance_id])
        intersection = int(intersection)
        candidates.append({
            "instance_id": instance_id,
            "class_id": instance_id // 1000,
            "iou": float(intersection / max(1, len(point_indices) + size - intersection)),
            "precision": float(intersection / max(1, len(point_indices))),
            "coverage": float(intersection / max(1, size)),
        })
    best = max(candidates, key=lambda item: item["iou"], default=None)
    same = max((item for item in candidates if item["class_id"] == prediction_class_id), key=lambda item: item["iou"], default=None)
    return {
        "best_gt_instance_id": int(best["instance_id"]) if best else -1,
        "best_gt_iou": float(best["iou"]) if best else 0.0,
        "best_gt_precision": float(best["precision"]) if best else 0.0,
        "best_gt_coverage": float(best["coverage"]) if best else 0.0,
        "best_same_class_gt_instance_id": int(same["instance_id"]) if same else -1,
        "best_same_class_gt_iou": float(same["iou"]) if same else 0.0,
        "best_same_class_gt_precision": float(same["precision"]) if same else 0.0,
        "best_same_class_gt_coverage": float(same["coverage"]) if same else 0.0,
    }


def _residual_type(final_iou):
    if final_iou < 0.25:
        return "无合格三维候选"
    if final_iou < 0.50:
        return "边界不足"
    return "已有严格三维候选"


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks, classes = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
    final_iou = _final_iou_by_gt(masks, classes, gt_ids, gt_sizes)
    payload = json.loads((args.graph_root / scene_name / "residual_tracks.json").read_text())
    rows = []
    for track in payload["tracks"]:
        points = np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        predicted_class_id = int(PRED_ID_TO_ID.get(int(track["class_id"]), -1))
        match = _best_gt_for_points(points, gt_ids, gt_sizes, predicted_class_id)
        matched_id = int(match["best_same_class_gt_instance_id"])
        matched_final = final_iou.get(matched_id, {"any": 0.0, "same_class": 0.0})
        rows.append({
            "scene_name": scene_name,
            "track_id": int(track["track_id"]),
            "class_name": str(track["class_name"]),
            "support_view_count": int(track["support_view_count"]),
            "point_count": int(track["point_count"]),
            "mean_edge_score": float(track["mean_edge_score"]),
            "min_mutual_projection_support": float(track["min_mutual_projection_support"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "best_gt_instance_id": int(match["best_gt_instance_id"]),
            "best_gt_class": str(ID_TO_LABEL.get(int(match["best_gt_instance_id"]) // 1000, "无有效实例")) if match["best_gt_instance_id"] > 0 else "无有效实例",
            "best_gt_iou": match["best_gt_iou"],
            "best_gt_precision": match["best_gt_precision"],
            "best_gt_coverage": match["best_gt_coverage"],
            "best_same_class_gt_instance_id": matched_id,
            "best_same_class_gt_iou": match["best_same_class_gt_iou"],
            "best_same_class_gt_precision": match["best_same_class_gt_precision"],
            "best_same_class_gt_coverage": match["best_same_class_gt_coverage"],
            "matched_gt_final_any_iou": float(matched_final["any"]),
            "matched_gt_final_same_class_iou": float(matched_final["same_class"]),
            "matched_gt_residual_type": _residual_type(float(matched_final["any"])) if matched_id > 0 else "无同类有效 GT",
        })
    return rows


def _count(rows, predicate):
    return sum(bool(predicate(row)) for row in rows)


def _summary(rows):
    groups = {}
    for name, predicate in {
        "同类 IoU 不低于 25%": lambda row: row["best_same_class_gt_iou"] >= 0.25,
        "同类 IoU 不低于 50%": lambda row: row["best_same_class_gt_iou"] >= 0.50,
        "同类精度不低于 50%": lambda row: row["best_same_class_gt_precision"] >= 0.50,
        "同类实例覆盖不低于 25%": lambda row: row["best_same_class_gt_coverage"] >= 0.25,
    }.items():
        count = _count(rows, predicate)
        groups[name] = {"轨迹数": count, "比例": float(count / max(1, len(rows)))}
    support_groups = {}
    for name, predicate in {"仅两视角": lambda row: row["support_view_count"] == 2, "至少三视角": lambda row: row["support_view_count"] >= 3}.items():
        subset = [row for row in rows if predicate(row)]
        support_groups[name] = {
            "轨迹数": len(subset),
            "同类 IoU 不低于 25% 的轨迹数": _count(subset, lambda row: row["best_same_class_gt_iou"] >= 0.25),
            "同类精度不低于 50% 的轨迹数": _count(subset, lambda row: row["best_same_class_gt_precision"] >= 0.50),
        }
    recovered = defaultdict(set)
    for row in rows:
        if row["best_same_class_gt_iou"] >= 0.25 and row["best_same_class_gt_instance_id"] > 0:
            recovered[row["matched_gt_residual_type"]].add((row["scene_name"], row["best_same_class_gt_instance_id"]))
    return {
        "多视角残差轨迹数": len(rows),
        "指标": groups,
        "按视角数分组": support_groups,
        "匹配同类 GT 后的强基线状态": dict(sorted(Counter(row["matched_gt_residual_type"] for row in rows).items())),
        "同类 IoU 不低于 25% 的独立 GT 实例数": {key: len(value) for key, value in sorted(recovered.items())},
        "说明": "轨迹仅作离线 GT 归因；未被写回最终候选，更不代表 AP 提升。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--graph_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    for name in ("scene_list", "graph_root", "prediction_cache_dir", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")

    rows = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条残差轨迹", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "residual_track_features_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线归因；绝不进入残差构图、关联、候选生成、融合、评分或阈值。",
        "scene_count": len(scenes),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

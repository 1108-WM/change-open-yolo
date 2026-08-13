#!/usr/bin/env python3
"""GT-only 审计已冻结 GVC append-only 候选的几何、语义与 native 重复风险。

此工具只读取已导出的候选和 native 缓存并做事后统计。GT 不会参与候选导出、类别、
分数、去重或融合，输出也不提供接受阈值或 AP。
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


TARGET_RESIDUALS = {"无合格三维候选", "边界不足"}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("场景列表含重复场景")
    return scenes


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id, count in zip(*np.unique(ids, return_counts=True)):
        instance_id = int(instance_id)
        if instance_id > 0 and instance_id // 1000 in valid_classes and int(count) >= min_region_size:
            sizes[instance_id] = int(count)
    return ids, sizes


def _load_native(root, scene_name, point_count):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    classes = np.load(root / f"{scene_name}_pred_classes.npy", mmap_mode="r")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count or masks.shape[1] != len(classes):
        raise ValueError(f"{scene_name} native 缓存维度不匹配")
    # 保持 mmap，避免 60 场景审计时把每个 native mask 矩阵复制进常驻内存。
    return masks, np.asarray(classes, dtype=np.int64)


def _best_gt(points, gt_ids, gt_sizes):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if len(points) == 0:
        return -1, 0.0
    ids, counts = np.unique(gt_ids[points], return_counts=True)
    best = (-1, 0.0)
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        iou = float(intersection / max(1, len(points) + gt_sizes[instance_id] - int(intersection)))
        if iou > best[1]:
            best = (instance_id, iou)
    return best


def _native_iou_by_gt(native_masks, gt_ids, gt_sizes):
    sizes = native_masks.sum(axis=0, dtype=np.int64)
    output = {}
    for instance_id, gt_size in gt_sizes.items():
        intersection = native_masks[gt_ids == instance_id].sum(axis=0, dtype=np.int64)
        output[instance_id] = float(np.max(intersection / np.maximum(1, sizes + gt_size - intersection))) if len(intersection) else 0.0
    return output


def _native_overlap(points, native_masks, native_classes, predicted_class):
    # 交集仅需读取该 GVC 候选覆盖的点，不能为每个候选构造完整点云大小的二维临时阵列。
    intersections = native_masks[points].sum(axis=0, dtype=np.int64)
    unions = native_masks.sum(axis=0, dtype=np.int64) + len(points) - intersections
    ious = intersections / np.maximum(1, unions)
    same_class = native_classes == int(predicted_class)
    return {
        "native_max_iou": float(ious.max(initial=0.0)),
        "same_class_native_max_iou": float(ious[same_class].max(initial=0.0)),
        "same_class_native_overlap_count": int(np.sum((ious > 0.0) & same_class)),
    }


def _residual_type(iou):
    if iou < 0.25:
        return "无合格三维候选"
    if iou < 0.50:
        return "边界不足"
    return "已有严格三维候选"


def _scene_rows(scene_name, args):
    candidate_path = args.candidate_root / scene_name / "backprojection_candidates.json"
    payload = json.loads(candidate_path.read_text())
    if payload.get("source_kind") != "gvc_append_only":
        raise ValueError(f"{scene_name} 不是 GVC append-only 候选")
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    native_masks, native_classes = _load_native(args.native_prediction_cache, scene_name, len(gt_ids))
    native_iou_by_gt = _native_iou_by_gt(native_masks, gt_ids, gt_sizes)
    rows = []
    for candidate in payload.get("candidates", []):
        seed_path = candidate_path.parent / candidate["seed_points_path"]
        points = np.unique(np.asarray(np.load(seed_path)["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(gt_ids))]
        instance_id, geometric_iou = _best_gt(points, gt_ids, gt_sizes)
        gt_class_id = instance_id // 1000 if instance_id > 0 else -1
        predicted_index = int(candidate["class_id"])
        predicted_class_id = int(PRED_ID_TO_ID.get(predicted_index, -1))
        native_overlap = _native_overlap(points, native_masks, native_classes, predicted_index)
        baseline_iou = float(native_iou_by_gt.get(instance_id, 0.0)) if instance_id > 0 else 0.0
        rows.append({
            "scene_name": scene_name,
            "candidate_id": int(candidate["candidate_id"]),
            "candidate_class": str(candidate["class_name"]),
            "candidate_score": float(candidate["score"]),
            "gvc_score": float(candidate["gvc_score"]),
            "semantic_reliability": float(candidate["semantic_reliability"]),
            "support_view_count": int(candidate["support_view_count"]),
            "num_seed_points": int(len(points)),
            "best_gt_instance_id": int(instance_id),
            "best_gt_class": ID_TO_LABEL.get(gt_class_id, "无有效实例"),
            "best_gt_iou": float(geometric_iou),
            "semantic_correct": bool(instance_id > 0 and predicted_class_id == gt_class_id),
            "matched_gt_final_iou": baseline_iou,
            "matched_gt_residual_type": _residual_type(baseline_iou) if instance_id > 0 else "无有效 GT",
            **native_overlap,
        })
    return rows


def _rank_audit(rows, predicate):
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return {"样本数": 0, "说明": "无满足条件的冻结候选。"}
    scores = np.asarray([row["candidate_score"] for row in subset], dtype=np.float64)
    ious = np.asarray([row["best_gt_iou"] for row in subset], dtype=np.float64)
    rho = float(spearmanr(scores, ious).statistic) if len(subset) >= 2 else 0.0
    order = np.argsort(-scores, kind="stable")
    top_count = max(1, int(np.ceil(len(subset) / 5)))
    top = [subset[index] for index in order[:top_count]]
    return {
        "样本数": len(subset),
        "候选分数与最佳 GT IoU 的 Spearman": rho if np.isfinite(rho) else 0.0,
        "最高分五分位候选数": len(top),
        "最高分五分位几何 IoU≥25%比例": float(sum(row["best_gt_iou"] >= 0.25 for row in top) / len(top)),
        "最高分五分位几何且语义正确比例": float(sum(row["best_gt_iou"] >= 0.25 and row["semantic_correct"] for row in top) / len(top)),
    }


def _summary(rows):
    target = [row for row in rows if row["matched_gt_residual_type"] in TARGET_RESIDUALS]
    geometric_semantic = [row for row in rows if row["best_gt_iou"] >= 0.25 and row["semantic_correct"]]
    same_class_duplicate = [row for row in rows if row["same_class_native_max_iou"] >= 0.50]
    target_geometric_semantic = [row for row in target if row["best_gt_iou"] >= 0.25 and row["semantic_correct"]]
    return {
        "全部 GVC append-only 候选数": len(rows),
        "几何 IoU≥25% 候选数": sum(row["best_gt_iou"] >= 0.25 for row in rows),
        "几何 IoU≥25% 且语义正确候选数": len(geometric_semantic),
        "目标基线缺口候选数": len(target),
        "目标基线缺口中几何且语义正确候选数": len(target_geometric_semantic),
        "与同类 native 候选 IoU≥50% 的候选数": len(same_class_duplicate),
        "其中几何且语义正确候选数": sum(row["best_gt_iou"] >= 0.25 and row["semantic_correct"] for row in same_class_duplicate),
        "候选分数排序审计": {
            "全部": _rank_audit(rows, lambda row: True),
            "目标基线缺口": _rank_audit(rows, lambda row: row["matched_gt_residual_type"] in TARGET_RESIDUALS),
        },
        "限制": "GT 只事后审计固定候选；本报告不提供阈值、不修改候选、不进行 AP。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；GT 只能用于离线安全审计。")
    for name in ("scene_list", "candidate_root", "native_prediction_cache", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条 GVC 候选", flush=True)
        del scene_rows
        gc.collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "gvc_append_only_safety_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于事后审计已冻结的 GVC append-only 候选；绝不进入候选、类别、分数、阈值、去重、融合或 AP。",
        "scene_count": len(scenes),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

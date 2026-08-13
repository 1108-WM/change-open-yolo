#!/usr/bin/env python3
"""仅 GT-only 审计 GVC 候选级账本的几何/语义可分性。

GT 仅在无 GT 的 GVC 特征写完后，事后计算固定轨迹的最佳实例、几何 IoU、
语义是否正确及基线残差类型；不定义或回写阈值、候选、类别或分数。
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {
        int(instance_id): int(np.sum(ids == instance_id))
        for instance_id in np.unique(ids)
        if int(instance_id) > 0 and int(instance_id) // 1000 in valid
        and int(np.sum(ids == instance_id)) >= min_region_size
    }
    return ids, sizes


def _load_masks(root, scene_name, point_count):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} native mask 点数不匹配")
    return np.asarray(masks, dtype=bool)


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
        iou = float(intersection / max(1, len(points) + gt_sizes[instance_id] - intersection))
        if iou > best[1]:
            best = (instance_id, iou)
    return best


def _native_iou_by_gt(masks, gt_ids, gt_sizes):
    sizes = masks.sum(axis=0, dtype=np.int64)
    result = {}
    for instance_id, gt_size in gt_sizes.items():
        intersection = masks[gt_ids == instance_id].sum(axis=0, dtype=np.int64)
        result[instance_id] = float(np.max(intersection / np.maximum(1, sizes + gt_size - intersection))) if len(intersection) else 0.0
    return result


def _residual_type(iou):
    if iou < 0.25:
        return "无合格三维候选"
    if iou < 0.50:
        return "边界不足"
    return "已有严格三维候选"


def _scene_rows(scene_name, args):
    gvc_rows = json.loads((args.gvc_root / scene_name / "track_gvc_feature_ledger.json").read_text())
    tracks = {int(row["track_id"]): row for row in json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )["tracks"]}
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    native_iou = _native_iou_by_gt(_load_masks(args.prediction_cache_dir, scene_name, len(gt_ids)), gt_ids, gt_sizes)
    rows = []
    for gvc in gvc_rows:
        track = tracks.get(int(gvc["track_id"]))
        if track is None:
            raise KeyError(f"{scene_name} 缺少轨迹 {gvc['track_id']}")
        points = np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        instance_id, geometric_iou = _best_gt(points, gt_ids, gt_sizes)
        gt_class_id = instance_id // 1000 if instance_id > 0 else -1
        predicted_class_id = int(PRED_ID_TO_ID.get(int(gvc["voted_class_index"]), -1))
        baseline_iou = float(native_iou.get(instance_id, 0.0)) if instance_id > 0 else 0.0
        rows.append({
            "scene_name": scene_name,
            "track_id": int(gvc["track_id"]),
            "best_gt_instance_id": instance_id,
            "best_gt_class": ID_TO_LABEL.get(gt_class_id, "无有效实例"),
            "best_gt_iou": geometric_iou,
            "matched_gt_final_iou": baseline_iou,
            "matched_gt_residual_type": _residual_type(baseline_iou) if instance_id > 0 else "无有效 GT",
            "semantic_correct": bool(instance_id > 0 and gt_class_id == predicted_class_id),
            "gvc_score": float(gvc["gvc_score"]),
            "gvc_selected_match_ratio": float(gvc["gvc_selected_match_ratio"]),
            "gvc_selected_view_count": int(gvc["gvc_selected_view_count"]),
            "gvc_eligible_view_count": int(gvc["gvc_eligible_view_count"]),
            "gvc_box_iou_mean": float(gvc["gvc_box_iou"]["mean"]),
            "gvc_mask_support_mean": float(gvc["gvc_mask_point_support"]["mean"]),
            "yoloworld_vote_margin": float(gvc["yoloworld_vote_margin"]),
            "native_top_iou": float(gvc["native_top_iou"]),
            "cer_mean": float(gvc["cer"]["mean"]),
            "pes_mean": float(gvc["pes"]["mean"]),
        })
    return rows


def _rank_audit(rows, name, predicate):
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return {"样本数": 0, "说明": "无满足条件的固定轨迹。"}
    scores = np.asarray([row["gvc_score"] for row in subset], dtype=np.float64)
    labels = np.asarray([row["best_gt_iou"] for row in subset], dtype=np.float64)
    rho = float(spearmanr(scores, labels).statistic) if len(subset) >= 2 else 0.0
    order = np.argsort(-scores, kind="stable")
    count = max(1, int(np.ceil(len(subset) / 5)))
    top = [subset[index] for index in order[:count]]
    return {
        "样本数": len(subset),
        "GVC 与轨迹最佳 GT IoU 的 Spearman": rho if np.isfinite(rho) else 0.0,
        "最高 GVC 五分位轨迹数": len(top),
        "最高 GVC 五分位几何 IoU≥25%比例": float(sum(row["best_gt_iou"] >= 0.25 for row in top) / len(top)),
        "最高 GVC 五分位几何且语义正确比例": float(sum(row["best_gt_iou"] >= 0.25 and row["semantic_correct"] for row in top) / len(top)),
        "说明": name,
    }


def _summary(rows):
    target = [row for row in rows if row["matched_gt_residual_type"] in TARGET_RESIDUALS]
    ordered_scenes = sorted({row["scene_name"] for row in rows})
    scene_group = {scene_name: "A" if index % 2 == 0 else "B" for index, scene_name in enumerate(ordered_scenes)}
    return {
        "全部固定轨迹数": len(rows),
        "几何 IoU≥25% 轨迹数": sum(row["best_gt_iou"] >= 0.25 for row in rows),
        "几何 IoU≥25% 且语义正确轨迹数": sum(row["best_gt_iou"] >= 0.25 and row["semantic_correct"] for row in rows),
        "目标基线缺口轨迹数": len(target),
        "目标基线缺口中几何 IoU≥25% 且语义正确轨迹数": sum(row["best_gt_iou"] >= 0.25 and row["semantic_correct"] for row in target),
        "GVC 排序审计": {
            "全部": _rank_audit(rows, "全量固定轨迹，仅作探索性相关性。", lambda row: True),
            "目标基线缺口": _rank_audit(target, "仅无合格候选/边界不足；这是候选安全性的关键子集。", lambda row: True),
            "目标基线缺口_场景A": _rank_audit(
                [row for row in target if scene_group[row["scene_name"]] == "A"],
                "字典序交替的 A 场景；仅验证固定 GVC 排序方向，不选阈值。", lambda row: True,
            ),
            "目标基线缺口_场景B": _rank_audit(
                [row for row in target if scene_group[row["scene_name"]] == "B"],
                "字典序交替的 B 场景；仅验证固定 GVC 排序方向，不选阈值。", lambda row: True,
            ),
        },
        "限制": "同一 even48 只用于提出假设；任何有用特征不得在此调阈值，必须原样在保留场景复验。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--gvc_root", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线 GVC 归因。")
    for name in ("scene_list", "gvc_root", "track_root", "prediction_cache_dir", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    rows = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条 GVC 轨迹", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "track_gvc_features_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线评估已经固定的 GVC 特征；不得进入视图选择、候选、类别、阈值、NMS 或 AP。",
        "scene_count": len(scenes),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

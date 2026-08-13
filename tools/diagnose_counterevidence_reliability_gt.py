#!/usr/bin/env python3
"""GT-only：审计“负边际 superpoint”是有益反证还是目标真表面。

该工具只读取已经固定的无 GT 可靠性账本，逐 superpoint 计算其属于原轨迹固定
GT 实例的比例，以及从原轨迹移除该部分所带来的固定实例 IoU 变化。GT 不参与
特征导出、阈值、删点、候选、类别、融合或 AP。
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

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_rows(root):
    rows = []
    for path in sorted(root.glob("*/counterevidence_superpoint_reliability.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {
        int(instance_id): int(np.sum(ids == instance_id))
        for instance_id in np.unique(ids)
        if int(instance_id) > 0 and int(instance_id) // 1000 in valid and int(np.sum(ids == instance_id)) >= min_region_size
    }
    return ids, sizes


def _iou(point_count, intersection, target_size):
    return float(intersection / max(1, point_count + target_size - intersection))


def _scene_rows(scene_name, reliability_rows, track_gt, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    track_points = {}
    track_intersection = {}
    for track in tracks:
        track_id = int(track["track_id"])
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(gt_ids))]
        track_points[track_id] = points
        target_id = int(track_gt[(scene_name, track_id)]["best_gt_instance_id"])
        track_intersection[track_id] = int(np.sum(gt_ids[points] == target_id)) if target_id in gt_sizes else 0
    rows = []
    valid_ids = set(gt_sizes)
    for item in reliability_rows:
        track_id, sp = int(item["track_id"]), int(item["superpoint_id"])
        source = track_gt.get((scene_name, track_id))
        if source is None:
            raise ValueError(f"缺少原轨迹 GT 账本：{scene_name}, {track_id}")
        target_id = int(source["best_gt_instance_id"])
        points = track_points[track_id]
        sp_points = points[superpoints[points] == sp]
        if target_id not in gt_sizes or len(sp_points) == 0:
            target_fraction, other_fraction, delta, outcome = 0.0, 0.0, 0.0, "无固定有效 GT"
        else:
            target_intersection = int(np.sum(gt_ids[sp_points] == target_id))
            other_intersection = int(np.sum(np.isin(gt_ids[sp_points], list(valid_ids - {target_id}))))
            target_fraction = float(target_intersection / len(sp_points))
            other_fraction = float(other_intersection / len(sp_points))
            before = _iou(len(points), track_intersection[track_id], gt_sizes[target_id])
            after = _iou(len(points) - len(sp_points), track_intersection[track_id] - target_intersection, gt_sizes[target_id])
            delta = float(after - before)
            outcome = "移除有益" if delta > 1e-12 else "移除有害" if delta < -1e-12 else "移除无影响"
        rows.append({
            **item,
            "fixed_gt_instance_id": target_id,
            "original_track_best_gt_iou": float(source["best_gt_iou"]),
            "matched_gt_residual_type": source["matched_gt_residual_type"],
            "gt_fixed_instance_point_fraction": target_fraction,
            "gt_other_valid_instance_point_fraction": other_fraction,
            "gt_iou_delta_if_removed": delta,
            "gt_removal_outcome": outcome,
        })
    return rows


def _describe(rows):
    usable = [row for row in rows if row["gt_removal_outcome"] != "无固定有效 GT"]
    return {
        "superpoint 数": len(rows),
        "有固定有效 GT 的 superpoint 数": len(usable),
        "固定 GT 点占比均值": float(np.mean([row["gt_fixed_instance_point_fraction"] for row in usable])) if usable else 0.0,
        "若移除的 IoU 变化均值": float(np.mean([row["gt_iou_delta_if_removed"] for row in usable])) if usable else 0.0,
        "移除有益数": sum(row["gt_removal_outcome"] == "移除有益" for row in usable),
        "移除有害数": sum(row["gt_removal_outcome"] == "移除有害" for row in usable),
        "移除无影响数": sum(row["gt_removal_outcome"] == "移除无影响" for row in usable),
    }


def _feature_associations(rows):
    usable = [row for row in rows if row["gt_removal_outcome"] != "无固定有效 GT"]
    fields = (
        "counterevidence_eligible_view_rate", "negative_weight_ratio", "mask_coverage_mean",
        "mask_coverage_std", "uncovered_near_2px_mask_boundary_ratio", "covered_mask_2px_interior_ratio",
        "counter_view_max_camera_baseline_m", "counter_view_max_view_angle_deg", "depth_residual_mean_m",
        "depth_discontinuity_median_m", "yoloworld_vote_margin", "centroid_knn_track_connectivity_proxy",
        "sp_planarity", "sp_linearity", "track_superpoint_coverage", "sp_inside_top_native_ratio",
    )
    result = []
    target = np.asarray([row["gt_iou_delta_if_removed"] for row in usable], dtype=np.float64)
    for field in fields:
        values = np.asarray([row[field] for row in usable], dtype=np.float64)
        if len(values) < 3 or np.allclose(values, values[0]):
            continue
        correlation, pvalue = spearmanr(values, target)
        if np.isfinite(correlation):
            result.append({"feature": field, "spearman_to_gt_iou_delta_if_removed": float(correlation), "pvalue": float(pvalue)})
    return sorted(result, key=lambda item: -abs(item["spearman_to_gt_iou_delta_if_removed"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reliability_root", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--track_gt_ledger", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线反证真伪审计。")
    for name in ("reliability_root", "track_root", "track_gt_ledger", "processed_scene_root", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    track_gt = {(row["scene_name"], int(row["track_id"])): row for row in _read_csv(args.track_gt_ledger)}
    by_scene = defaultdict(list)
    for row in _load_rows(args.reliability_root):
        by_scene[row["scene_name"]].append(row)
    rows = []
    for scene_name in sorted(by_scene):
        rows.extend(_scene_rows(scene_name, by_scene[scene_name], track_gt, args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "track_id", "superpoint_id"]
    with (args.output_dir / "counterevidence_reliability_gt_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    negative = [row for row in rows if row["is_negative_margin"]]
    payload = {
        "诊断限定": "GT 仅用于审计固定无 GT 特征；不得反向用于删点、候选、类别、阈值、排序、融合或 AP。",
        "全部 superpoint": _describe(rows),
        "负边际 superpoint": _describe(negative),
        "负边际且原轨迹几何合格并对应基线缺口": _describe([
            row for row in negative if row["original_track_best_gt_iou"] >= 0.25 and row["matched_gt_residual_type"] in {"无合格三维候选", "边界不足"}
        ]),
        "特征与移除固定 GT IoU 变化的 Spearman 关联": _feature_associations(negative),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

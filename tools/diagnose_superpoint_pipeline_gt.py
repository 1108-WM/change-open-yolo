#!/usr/bin/env python3
"""仅离线真值诊断：定位实例覆盖在超点到 SAM2 链路的哪一层下降。

比较原始 ScanNet superpoint、f30 IBSp、当前可靠种子、实际采样种子和最终
SAM2 三维候选。所有 GT 只在本脚本内统计，输出绝不回流到推理、候选、融合、
打分或阈值选择。
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnose_sam2_seed_coverage_gt import _best_mask_iou, _load_gt, _load_masks, _read_scenes


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scene_array(root, scene_name):
    scene_id = scene_name.removeprefix("scene")
    return np.load(root / scene_name / f"{scene_id}.npy", mmap_mode="r")


def _superpoints(root, scene_name, point_count):
    values = _scene_array(root, scene_name)[:, 9].astype(np.int64, copy=False)
    if len(values) != point_count or np.any(values < 0):
        raise ValueError(f"{scene_name} 的 superpoint 数组不合法")
    return values


def _load_seed_sets(track_root, scene_name):
    reliable, sampled = set(), set()
    for round_name in ("tracks_round1", "tracks_round2", "tracks_round3"):
        path = track_root / round_name / scene_name / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少轨迹摘要：{path}")
        summary = json.loads(path.read_text())
        reliable.update(int(value) for value in summary.get("reliable_superpoint_ids", []))
        sampled.update(int(value) for value in summary.get("attempted_seed_superpoint_ids", []))
    return reliable, sampled


def _load_candidates(root, scene_name, num_points):
    path = root / scene_name / "backprojection_candidates.json"
    payload = json.loads(path.read_text())
    candidates = []
    for item in payload.get("candidates", []):
        indices = np.unique(np.load(_resolve(item["seed_points_path"]))["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < num_points)].astype(np.int32, copy=False)
        candidates.append(indices)
    return candidates


def _best_candidate_iou(gt_ids, instance_id, gt_size, candidates):
    best = 0.0
    for indices in candidates:
        intersection = int(np.count_nonzero(gt_ids[indices] == instance_id)) if len(indices) else 0
        best = max(best, intersection / max(1, len(indices) + gt_size - intersection))
    return float(best)


def _superpoint_quality(superpoints, indices, purity_threshold):
    sizes = np.bincount(superpoints, minlength=int(superpoints.max(initial=-1)) + 1)
    segment_ids, in_gt_counts = np.unique(superpoints[indices], return_counts=True)
    purity = in_gt_counts / np.maximum(1, sizes[segment_ids])
    pure = purity >= purity_threshold
    return {
        "segment_ids": segment_ids,
        "pure_segment_ids": set(int(value) for value in segment_ids[pure]),
        "best_purity": float(purity.max(initial=0.0)),
        "pure_point_fraction": float(in_gt_counts[pure].sum() / max(1, len(indices))),
        "has_pure_segment": bool(np.any(pure)),
    }


def _group_summary(rows, threshold):
    selected = [row for row in rows if row["baseline_best_iou"] < threshold]
    if threshold == 0.0:
        selected = rows
    if not selected:
        return {"gt_instance_count": 0}
    original_fraction = np.asarray([row["original_pure_point_fraction"] for row in selected])
    ibsp_fraction = np.asarray([row["ibsp_pure_point_fraction"] for row in selected])
    delta = ibsp_fraction - original_fraction
    def _rate(key):
        return float(sum(bool(row[key]) for row in selected) / len(selected))
    return {
        "gt_instance_count": len(selected),
        "original": {
            "mean_gt_point_coverage_by_pure_superpoints": float(original_fraction.mean()),
            "mean_best_superpoint_purity": float(np.mean([row["original_best_purity"] for row in selected])),
            "has_pure_superpoint_rate": _rate("original_has_pure_segment"),
            "at_least_half_gt_points_in_pure_superpoints_rate": float(np.mean(original_fraction >= 0.5)),
        },
        "f30_ibsp": {
            "mean_gt_point_coverage_by_pure_superpoints": float(ibsp_fraction.mean()),
            "mean_best_superpoint_purity": float(np.mean([row["ibsp_best_purity"] for row in selected])),
            "has_pure_superpoint_rate": _rate("ibsp_has_pure_segment"),
            "at_least_half_gt_points_in_pure_superpoints_rate": float(np.mean(ibsp_fraction >= 0.5)),
        },
        "f30_minus_original": {
            "mean_coverage_change": float(delta.mean()),
            "improved_instance_count": int(np.count_nonzero(delta > 1e-6)),
            "unchanged_instance_count": int(np.count_nonzero(np.abs(delta) <= 1e-6)),
            "worsened_instance_count": int(np.count_nonzero(delta < -1e-6)),
        },
        "f30_to_sam2": {
            "has_pure_ibsp_rate": _rate("ibsp_has_pure_segment"),
            "has_pure_reliable_ibsp_rate": _rate("has_pure_reliable_ibsp"),
            "has_pure_sampled_ibsp_rate": _rate("has_pure_sampled_ibsp"),
            "sam2_iou25_coverage_rate": float(np.mean([row["sam2_best_iou"] >= 0.25 for row in selected])),
            "sam2_iou50_coverage_rate": float(np.mean([row["sam2_best_iou"] >= 0.50 for row in selected])),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--original_superpoint_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--ibsp_superpoint_root", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--purity_threshold", type=float, default=0.50)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断。")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    rows = []
    for scene_name in scenes:
        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        original = _superpoints(args.original_superpoint_root, scene_name, len(gt_ids))
        ibsp = _superpoints(args.ibsp_superpoint_root, scene_name, len(gt_ids))
        masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
        mask_sizes = masks.sum(axis=0, dtype=np.int64)
        reliable, sampled = _load_seed_sets(args.track_root, scene_name)
        candidates = _load_candidates(args.candidate_root, scene_name, len(gt_ids))
        for instance in instances:
            original_quality = _superpoint_quality(original, instance["indices"], args.purity_threshold)
            ibsp_quality = _superpoint_quality(ibsp, instance["indices"], args.purity_threshold)
            pure_ids = ibsp_quality["pure_segment_ids"]
            rows.append(
                {
                    "scene_name": scene_name,
                    "gt_instance_id": instance["gt_instance_id"],
                    "gt_class": instance["gt_class"],
                    "gt_point_count": instance["point_count"],
                    "baseline_best_iou": _best_mask_iou(masks, mask_sizes, instance["indices"], instance["point_count"]),
                    "original_best_purity": original_quality["best_purity"],
                    "original_pure_point_fraction": original_quality["pure_point_fraction"],
                    "original_has_pure_segment": original_quality["has_pure_segment"],
                    "ibsp_best_purity": ibsp_quality["best_purity"],
                    "ibsp_pure_point_fraction": ibsp_quality["pure_point_fraction"],
                    "ibsp_has_pure_segment": ibsp_quality["has_pure_segment"],
                    "pure_ibsp_segment_count": len(pure_ids),
                    "has_pure_reliable_ibsp": bool(pure_ids & reliable),
                    "has_pure_sampled_ibsp": bool(pure_ids & sampled),
                    "sam2_best_iou": _best_candidate_iou(
                        gt_ids, instance["gt_instance_id"], instance["point_count"], candidates
                    ),
                }
            )
        print(f"[场景完成] {scene_name}: {len(instances)} 个有效 GT 实例", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "superpoint_pipeline_gt.csv").open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或阈值选择。",
        "purity_threshold": args.purity_threshold,
        "definitions": {
            "pure_superpoint": "某 GT 实例在该 superpoint 内占至少一半点。",
            "pure_point_coverage": "GT 实例中落在纯 superpoint 内的点比例。",
            "reliable": "当前无 GT 种子资格：规模与 f30 可见帧条件同时满足。",
        },
        "scene_count": len(scenes),
        "all_valid_gt": _group_summary(rows, 0.0),
        "mask3d_missed_iou25": _group_summary(rows, 0.25),
        "mask3d_missed_iou50": _group_summary(rows, 0.50),
        "f30_change_direction_all_valid_gt": dict(sorted(Counter(
            "improved" if row["ibsp_pure_point_fraction"] > row["original_pure_point_fraction"] + 1e-6
            else "worsened" if row["ibsp_pure_point_fraction"] < row["original_pure_point_fraction"] - 1e-6
            else "unchanged"
            for row in rows
        ).items())),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

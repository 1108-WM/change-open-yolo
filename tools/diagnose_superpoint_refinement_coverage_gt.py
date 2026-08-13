#!/usr/bin/env python3
"""仅离线 GT 账本：检查新版 superpoint 是否仍保持原始实例边界覆盖。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnose_sam2_seed_coverage_gt import _best_mask_iou, _load_gt, _load_masks, _read_scenes
from diagnose_superpoint_pipeline_gt import _superpoint_quality


def _scene_superpoints(root, scene_name, point_count):
    scene_id = scene_name.removeprefix("scene")
    values = np.load(root / scene_name / f"{scene_id}.npy", mmap_mode="r")[:, 9]
    if len(values) != point_count or not np.all(np.isfinite(values)):
        raise ValueError(f"{scene_name} 的 superpoint 标签不合法")
    values = np.rint(values).astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError(f"{scene_name} 的 superpoint 标签不能为负数")
    return values


def _partition_is_refinement(original, refined):
    pairs = np.stack((refined, original), axis=1)
    return len(np.unique(pairs, axis=0)) == len(np.unique(refined))


def _summary(rows, mask_iou_limit=None):
    selected = [
        row for row in rows
        if mask_iou_limit is None or float(row["baseline_best_iou"]) < mask_iou_limit
    ]
    if not selected:
        return {"instances": 0}
    original = np.asarray([float(row["original_pure_point_fraction"]) for row in selected])
    refined = np.asarray([float(row["refined_pure_point_fraction"]) for row in selected])
    return {
        "instances": len(selected),
        "original_pure_point_coverage_percent": float(original.mean() * 100.0),
        "refined_pure_point_coverage_percent": float(refined.mean() * 100.0),
        "coverage_change_percentage_points": float((refined.mean() - original.mean()) * 100.0),
        "improved_instances": int(np.count_nonzero(refined > original + 1e-6)),
        "unchanged_instances": int(np.count_nonzero(np.abs(refined - original) <= 1e-6)),
        "worsened_instances": int(np.count_nonzero(refined < original - 1e-6)),
        "refined_has_pure_superpoint_percent": float(
            np.mean([row["refined_has_pure_segment"] for row in selected]) * 100.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--original_superpoint_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--refined_superpoint_root", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--purity_threshold", type=float, default=0.50)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断。")

    scenes = _read_scenes(args.scene_list)
    rows = []
    scene_checks = []
    for scene_name in scenes:
        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        original = _scene_superpoints(args.original_superpoint_root, scene_name, len(gt_ids))
        refined = _scene_superpoints(args.refined_superpoint_root, scene_name, len(gt_ids))
        if not _partition_is_refinement(original, refined):
            raise RuntimeError(f"{scene_name} 违反原始 superpoint 仅切分约束")
        masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
        mask_sizes = masks.sum(axis=0, dtype=np.int64)
        for instance in instances:
            original_quality = _superpoint_quality(original, instance["indices"], args.purity_threshold)
            refined_quality = _superpoint_quality(refined, instance["indices"], args.purity_threshold)
            rows.append({
                "scene_name": scene_name,
                "gt_instance_id": instance["gt_instance_id"],
                "gt_class": instance["gt_class"],
                "gt_point_count": instance["point_count"],
                "baseline_best_iou": _best_mask_iou(
                    masks, mask_sizes, instance["indices"], instance["point_count"]
                ),
                "original_pure_point_fraction": original_quality["pure_point_fraction"],
                "original_has_pure_segment": original_quality["has_pure_segment"],
                "refined_pure_point_fraction": refined_quality["pure_point_fraction"],
                "refined_has_pure_segment": refined_quality["has_pure_segment"],
            })
        scene_checks.append({
            "scene_name": scene_name,
            "original_segments": int(np.unique(original).size),
            "refined_segments": int(np.unique(refined).size),
            "partition_is_refinement": True,
        })
        print(f"[场景完成] {scene_name}：{len(instances)} 个有效 GT 实例", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "superpoint_refinement_coverage_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或阈值选择。",
        "definition": "实例主属超点覆盖：GT 实例中，落在该实例占该 superpoint 至少一半点的区域内的点比例。",
        "purity_threshold": args.purity_threshold,
        "scene_count": len(scenes),
        "all_valid_gt": _summary(rows),
        "mask3d_missed_iou25": _summary(rows, 0.25),
        "mask3d_missed_iou50": _summary(rows, 0.50),
        "scene_partition_checks": scene_checks,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

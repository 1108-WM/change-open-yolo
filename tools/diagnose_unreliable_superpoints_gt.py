#!/usr/bin/env python3
"""仅离线真值诊断：拆解 Mask3D 漏检实例为何没有可靠 IBSp。"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from diagnose_sam2_seed_coverage_gt import _best_mask_iou, _load_gt, _load_masks, _read_scenes
from export_any3dis_sam2_tracks import (
    _load_scene_arrays,
    _load_visibility,
    _superpoint_stats,
    _visibility_by_superpoint,
)


def _reason(size_ok, visible_ok, reliable_count):
    if reliable_count:
        return "存在可靠超点"
    if not size_ok:
        return "超点规模不足"
    if not visible_ok:
        return "可见帧不足"
    return "规模与可见性未在同一超点满足"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--superpoint_root", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--purity_threshold", type=float, default=0.50)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断。")

    rows_by_threshold = defaultdict(list)
    for scene_name in _read_scenes(args.scene_list):
        track_summary = json.loads((args.track_root / scene_name / "summary.json").read_text())
        params = track_summary["params"]
        min_sp_points = int(params["min_superpoint_points"])
        min_prompt_points = int(params["min_prompt_points"])
        min_visible_frames = int(params["min_visible_frames"])
        depth_tolerance = float(params["depth_tolerance"])
        frame_ids = track_summary["frame_ids"]
        scene_dir = args.dataset_root / scene_name
        points, superpoints, intrinsics, _ = _load_scene_arrays(scene_dir, args.superpoint_root, scene_name)
        visible_points, _, _ = _load_visibility(scene_dir, frame_ids, points, intrinsics, depth_tolerance)
        sizes, _ = _superpoint_stats(points, superpoints)
        distribution = _visibility_by_superpoint(visible_points, superpoints, len(sizes))
        visible_frame_count = (distribution >= min_prompt_points).sum(axis=0)
        reliable = (sizes >= min_sp_points) & (visible_frame_count >= min_visible_frames)

        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
        mask_sizes = masks.sum(axis=0, dtype=np.int64)
        for instance in instances:
            baseline_iou = _best_mask_iou(masks, mask_sizes, instance["indices"], instance["point_count"])
            segment_ids, segment_counts = np.unique(superpoints[instance["indices"]], return_counts=True)
            segment_sizes = sizes[segment_ids]
            segment_visible = visible_frame_count[segment_ids]
            purity = segment_counts / np.maximum(1, segment_sizes)
            size_ok = segment_sizes >= min_sp_points
            visible_ok = segment_visible >= min_visible_frames
            reliable_count = int(np.count_nonzero(reliable[segment_ids]))
            best_index = int(np.argmax(segment_counts))
            pure_gt_fraction = float(segment_counts[purity >= args.purity_threshold].sum() / instance["point_count"])
            row_base = {
                "scene_name": scene_name,
                "gt_instance_id": instance["gt_instance_id"],
                "gt_class": instance["gt_class"],
                "gt_point_count": instance["point_count"],
                "baseline_best_iou": baseline_iou,
                "gt_superpoint_count": int(len(segment_ids)),
                "reliable_superpoint_count": reliable_count,
                "size_eligible_superpoint_count": int(np.count_nonzero(size_ok)),
                "visible_eligible_superpoint_count": int(np.count_nonzero(visible_ok)),
                "best_superpoint_id": int(segment_ids[best_index]),
                "best_superpoint_gt_purity": float(purity[best_index]),
                "max_superpoint_gt_purity": float(purity.max(initial=0.0)),
                "gt_fraction_in_pure_superpoints": pure_gt_fraction,
                "max_visible_frame_count": int(segment_visible.max(initial=0)),
                "unreliable_reason": _reason(bool(size_ok.any()), bool(visible_ok.any()), reliable_count),
                "boundary_mixed": bool(float(purity.max(initial=0.0)) < args.purity_threshold),
            }
            for threshold, suffix in ((0.25, "iou25"), (0.50, "iou50")):
                if baseline_iou < threshold:
                    rows_by_threshold[suffix].append(row_base)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、打分、融合、阈值选择或候选生成。",
        "purity_threshold": args.purity_threshold,
        "thresholds": {},
    }
    for suffix, rows in rows_by_threshold.items():
        with (args.output_dir / f"mask3d_missed_{suffix}_superpoint_causes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        unreliable = [row for row in rows if row["reliable_superpoint_count"] == 0]
        summary["thresholds"][suffix] = {
            "mask3d_missed_gt_count": len(rows),
            "no_reliable_superpoint_count": len(unreliable),
            "unreliable_reason_counts": dict(sorted(Counter(row["unreliable_reason"] for row in unreliable).items())),
            "boundary_mixed_count": int(sum(row["boundary_mixed"] for row in unreliable)),
            "mean_max_superpoint_gt_purity": float(np.mean([row["max_superpoint_gt_purity"] for row in unreliable])) if unreliable else 0.0,
            "mean_gt_fraction_in_pure_superpoints": float(
                np.mean([row["gt_fraction_in_pure_superpoints"] for row in unreliable])
            ) if unreliable else 0.0,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

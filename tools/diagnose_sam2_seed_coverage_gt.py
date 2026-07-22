#!/usr/bin/env python3
"""仅离线真值诊断：定位 Mask3D 漏检实例在 SAM2 前端的停止位置。

将每个 Mask3D 未覆盖的 GT 实例依次归为：没有可靠超点、有可靠超点但
没有被采样、已采样但未形成足够好的 SAM2 候选、或已被 SAM2 覆盖。
输出只用于分析，绝不参与任何推理、阈值或候选选择。
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid_classes:
            continue
        indices = np.flatnonzero(ids == instance_id).astype(np.int32)
        if len(indices) < min_region_size:
            continue
        instances.append(
            {
                "gt_instance_id": instance_id,
                "gt_class": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
                "indices": indices,
                "point_count": int(len(indices)),
            }
        )
    return ids, instances


def _load_masks(root, scene_name, num_points):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = _as_numpy(masks).astype(bool, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的基础 mask 维度异常：{masks.shape}")
    if masks.shape[0] != num_points and masks.shape[1] == num_points:
        masks = masks.T
    if masks.shape[0] != num_points:
        raise ValueError(f"{scene_name} 的点数不一致：{masks.shape[0]} 与 {num_points}")
    return masks


def _load_candidates(root, scene_name, num_points):
    payload = json.loads((root / scene_name / "backprojection_candidates.json").read_text())
    candidates = []
    for item in payload.get("candidates", []):
        indices = np.unique(np.load(_resolve(item["seed_points_path"]))["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < num_points)].astype(np.int32, copy=False)
        candidates.append((int(item["candidate_id"]), indices))
    return candidates


def _best_mask_iou(masks, mask_sizes, indices, point_count):
    if masks.shape[1] == 0:
        return 0.0
    intersections = masks[indices].sum(axis=0, dtype=np.int64)
    return float(np.max(intersections / np.maximum(1, mask_sizes + point_count - intersections)))


def _best_candidate_iou(gt_ids, gt_instance_id, point_count, candidates):
    best_iou, best_id = 0.0, -1
    for candidate_id, indices in candidates:
        intersection = int(np.count_nonzero(gt_ids[indices] == gt_instance_id)) if len(indices) else 0
        iou = intersection / max(1, len(indices) + point_count - intersection)
        if iou > best_iou:
            best_iou, best_id = float(iou), candidate_id
    return best_iou, best_id


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


def _failure_stage(reliable_count, sampled_count, candidate_iou, threshold):
    if candidate_iou >= threshold:
        return "sam2_已覆盖"
    if reliable_count == 0:
        return "无可靠超点"
    if sampled_count == 0:
        return "有可靠超点但未采样"
    return "已采样但轨迹或三维提升未恢复"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--track_run_root", type=Path, required=True)
    parser.add_argument("--superpoint_root", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断。")

    rows_by_threshold = defaultdict(list)
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
        mask_sizes = masks.sum(axis=0, dtype=np.int64)
        candidates = _load_candidates(args.candidate_root, scene_name, len(gt_ids))
        superpoints = np.load(
            args.superpoint_root / scene_name / f"{scene_name.removeprefix('scene')}.npy", mmap_mode="r"
        )[:, 9].astype(np.int64, copy=False)
        if len(superpoints) != len(gt_ids):
            raise ValueError(f"{scene_name} 的超点与 GT 点数不一致")
        reliable, sampled = _load_seed_sets(args.track_run_root, scene_name)
        for instance in instances:
            indices = instance["indices"]
            baseline_iou = _best_mask_iou(masks, mask_sizes, indices, instance["point_count"])
            candidate_iou, candidate_id = _best_candidate_iou(
                gt_ids, instance["gt_instance_id"], instance["point_count"], candidates
            )
            instance_superpoints = set(int(value) for value in np.unique(superpoints[indices]))
            reliable_count = len(instance_superpoints & reliable)
            sampled_count = len(instance_superpoints & sampled)
            for threshold, suffix in ((0.25, "iou25"), (0.50, "iou50")):
                if baseline_iou >= threshold:
                    continue
                rows_by_threshold[suffix].append(
                    {
                        "scene_name": scene_name,
                        "gt_instance_id": instance["gt_instance_id"],
                        "gt_class": instance["gt_class"],
                        "gt_point_count": instance["point_count"],
                        "baseline_best_iou": baseline_iou,
                        "sam2_best_iou": candidate_iou,
                        "sam2_best_candidate_id": candidate_id,
                        "gt_superpoint_count": len(instance_superpoints),
                        "reliable_superpoint_count": reliable_count,
                        "sampled_seed_count": sampled_count,
                        "failure_stage": _failure_stage(reliable_count, sampled_count, candidate_iou, threshold),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、打分、融合、阈值选择或候选生成。",
        "thresholds": {},
    }
    for suffix, rows in rows_by_threshold.items():
        with (args.output_dir / f"mask3d_missed_{suffix}_seed_coverage.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        counts = Counter(row["failure_stage"] for row in rows)
        summary["thresholds"][suffix] = {
            "mask3d_missed_gt_count": len(rows),
            "failure_stage_counts": dict(sorted(counts.items())),
            "failure_stage_rates": {
                name: float(count / max(1, len(rows))) for name, count in sorted(counts.items())
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

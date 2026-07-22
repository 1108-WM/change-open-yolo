#!/usr/bin/env python3
"""仅离线真值诊断：量化强基线与 SAM2 候选的实例级互补性。

本工具只把 GT 用于事后分析，绝不向候选生成、融合、打分、阈值选择
或测试时推理返回任何数据。它回答的不是候选整体质量，而是每个 GT
实例究竟由谁覆盖：强基线、SAM2、二者，还是二者都遗漏。
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

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _load_baseline_masks(root, scene_name, num_points):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = _as_numpy(masks).astype(bool, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"Unexpected baseline mask shape for {scene_name}: {masks.shape}")
    if masks.shape[0] != num_points and masks.shape[1] == num_points:
        masks = masks.T
    if masks.shape[0] != num_points:
        raise ValueError(f"Point count mismatch for {scene_name}: {masks.shape[0]} vs {num_points}")
    return masks


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid_classes:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) < min_region_size:
            continue
        instances.append(
            {
                "gt_instance_id": instance_id,
                "class_name": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
                "indices": indices,
                "point_count": int(len(indices)),
            }
        )
    return gt_ids, instances


def _load_sam2_candidates(root, scene_name, num_points):
    payload = json.loads((root / scene_name / "backprojection_candidates.json").read_text())
    candidates = []
    for item in payload.get("candidates", []):
        indices = np.unique(np.load(_resolve(item["seed_points_path"]))["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < num_points)].astype(np.int32, copy=False)
        candidates.append(
            {
                "candidate_id": int(item["candidate_id"]),
                "class_name": str(item["class_name"]),
                "score": float(item["score"]),
                "indices": indices,
                "point_count": int(len(indices)),
            }
        )
    return candidates


def _best_baseline_iou(baseline_masks, baseline_sizes, gt_indices, gt_size):
    if baseline_masks.shape[1] == 0:
        return 0.0, -1
    intersections = baseline_masks[gt_indices].sum(axis=0, dtype=np.int64)
    unions = baseline_sizes + gt_size - intersections
    scores = intersections / np.maximum(1, unions)
    best_index = int(np.argmax(scores))
    return float(scores[best_index]), best_index


def _best_sam2_iou(gt_ids, gt_instance_id, gt_size, candidates):
    best = {"iou": 0.0, "candidate_id": -1, "class_name": "", "score": 0.0}
    for candidate in candidates:
        indices = candidate["indices"]
        if not len(indices):
            continue
        intersection = int(np.count_nonzero(gt_ids[indices] == gt_instance_id))
        iou = intersection / max(1, candidate["point_count"] + gt_size - intersection)
        if iou > best["iou"]:
            best = {
                "iou": float(iou),
                "candidate_id": int(candidate["candidate_id"]),
                "class_name": candidate["class_name"],
                "score": float(candidate["score"]),
            }
    return best


def _coverage_group(baseline_iou, sam2_iou, threshold):
    baseline_covers = baseline_iou >= threshold
    sam2_covers = sam2_iou >= threshold
    if not baseline_covers and sam2_covers:
        return "baseline_missed_sam2_covered"
    if baseline_covers and sam2_covers:
        return "both_covered"
    if baseline_covers:
        return "baseline_covered_sam2_missed"
    return "both_missed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--improvement_delta", type=float, default=0.10)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断，不能进入推理或选择。")

    rows = []
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, gt_instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        baseline_masks = _load_baseline_masks(args.baseline_masks_root, scene_name, len(gt_ids))
        baseline_sizes = baseline_masks.sum(axis=0, dtype=np.int64)
        candidates = _load_sam2_candidates(args.candidate_root, scene_name, len(gt_ids))
        for gt in gt_instances:
            baseline_iou, baseline_mask_id = _best_baseline_iou(
                baseline_masks, baseline_sizes, gt["indices"], gt["point_count"]
            )
            sam2 = _best_sam2_iou(gt_ids, gt["gt_instance_id"], gt["point_count"], candidates)
            row = {
                "scene_name": scene_name,
                "gt_instance_id": gt["gt_instance_id"],
                "gt_class": gt["class_name"],
                "gt_point_count": gt["point_count"],
                "baseline_best_iou": baseline_iou,
                "baseline_best_mask_id": baseline_mask_id,
                "sam2_best_iou": sam2["iou"],
                "sam2_best_candidate_id": sam2["candidate_id"],
                "sam2_best_class": sam2["class_name"],
                "sam2_best_score": sam2["score"],
                "sam2_semantic_exact": bool(sam2["class_name"] == gt["class_name"]),
                "sam2_minus_baseline_iou": float(sam2["iou"] - baseline_iou),
            }
            for threshold, suffix in ((0.25, "iou25"), (0.50, "iou50")):
                row[f"coverage_{suffix}"] = _coverage_group(baseline_iou, sam2["iou"], threshold)
                row[f"sam2_improves_{suffix}"] = bool(
                    sam2["iou"] >= threshold and sam2["iou"] - baseline_iou >= args.improvement_delta
                )
            rows.append(row)

    def _summary_for(threshold, suffix):
        groups = Counter(row[f"coverage_{suffix}"] for row in rows)
        improved = [row for row in rows if row[f"sam2_improves_{suffix}"]]
        complement = [row for row in rows if row[f"coverage_{suffix}"] == "baseline_missed_sam2_covered"]
        return {
            "coverage_groups": dict(sorted(groups.items())),
            "sam2_genuine_complement_count": len(complement),
            "sam2_genuine_complement_rate": float(len(complement) / max(1, len(rows))),
            "sam2_improves_baseline_by_delta_count": len(improved),
            "sam2_improves_baseline_by_delta_rate": float(len(improved) / max(1, len(rows))),
            "sam2_complement_semantic_exact_accuracy": float(
                sum(row["sam2_semantic_exact"] for row in complement) / max(1, len(complement))
            ),
        }

    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、打分、融合、阈值选择或候选生成。",
        "num_gt_instances": len(rows),
        "baseline_masks_root": str(args.baseline_masks_root),
        "candidate_root": str(args.candidate_root),
        "improvement_delta": args.improvement_delta,
        "iou25": _summary_for(0.25, "iou25"),
        "iou50": _summary_for(0.50, "iou50"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gt_instance_complementarity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""仅离线真值诊断：量化 SAM2 对 Mask3D 局部补全或裁剪的 oracle 上界。

每个 SAM2 候选与其三维 IoU 最高的原始 Mask3D mask 配对，比较保留原 mask、
并集补全和交集裁剪对该原 mask 最佳对应 GT 的 IoU。GT 只在本脚本中读取，
不会输出给推理、融合、阈值或候选生成路径。
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


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


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


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes, classes = {}, {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid_classes:
            continue
        count = int(np.count_nonzero(gt_ids == instance_id))
        if count < min_region_size:
            continue
        sizes[instance_id] = count
        classes[instance_id] = str(ID_TO_LABEL.get(semantic_id, semantic_id))
    return gt_ids, sizes, classes


def _load_candidates(root, scene_name, num_points):
    payload = json.loads((root / scene_name / "backprojection_candidates.json").read_text())
    candidates = []
    for item in payload.get("candidates", []):
        indices = np.unique(np.load(_resolve(item["seed_points_path"]))["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < num_points)].astype(np.int32, copy=False)
        if len(indices):
            candidates.append((item, indices))
    return candidates


def _best_gt_for_baseline(gt_ids, valid_sizes, baseline_indices):
    ids, counts = np.unique(gt_ids[baseline_indices], return_counts=True)
    best_id, best_iou = -1, 0.0
    baseline_size = len(baseline_indices)
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in valid_sizes:
            continue
        iou = int(intersection) / max(1, baseline_size + valid_sizes[instance_id] - int(intersection))
        if iou > best_iou:
            best_id, best_iou = instance_id, float(iou)
    return best_id, best_iou


def _feature_summary(rows):
    fields = ["score", "fusion_score", "support_score", "best_existing_iou", "seed_in_existing_mask_ratio", "source_track_count", "candidate_point_count"]
    result = {}
    for name, group in (("oracle_improves", [row for row in rows if row["oracle_improves"]]), ("not_oracle_improves", [row for row in rows if not row["oracle_improves"]])):
        result[name] = {"count": len(group), "features": {}}
        for field in fields:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            result[name]["features"][field] = {
                "mean": float(values.mean()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
            }
    return result


def _diagnose_scene(scene_name, args):
    gt_ids, valid_sizes, classes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks = _load_masks(args.baseline_masks_root, scene_name, len(gt_ids))
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    baseline_indices = [np.flatnonzero(masks[:, index]) for index in range(masks.shape[1])]
    baseline_targets = [_best_gt_for_baseline(gt_ids, valid_sizes, indices) for indices in baseline_indices]
    rows = []
    for item, candidate_indices in _load_candidates(args.candidate_root, scene_name, len(gt_ids)):
        intersections = masks[candidate_indices].sum(axis=0, dtype=np.int64)
        candidate_size = len(candidate_indices)
        parent_ious = intersections / np.maximum(1, mask_sizes + candidate_size - intersections)
        parent_id = int(np.argmax(parent_ious)) if len(parent_ious) else -1
        parent_iou = float(parent_ious[parent_id]) if parent_id >= 0 else 0.0
        target_id, keep_iou = baseline_targets[parent_id] if parent_id >= 0 else (-1, 0.0)
        if target_id < 0:
            continue
        parent_mask = masks[:, parent_id]
        union_mask = np.logical_or(parent_mask, np.isin(np.arange(len(gt_ids)), candidate_indices))
        intersection_mask = np.logical_and(parent_mask, np.isin(np.arange(len(gt_ids)), candidate_indices))
        target_size = valid_sizes[target_id]
        union_iou = float(np.count_nonzero(gt_ids[union_mask] == target_id) / max(1, union_mask.sum() + target_size - np.count_nonzero(gt_ids[union_mask] == target_id)))
        intersection_iou = float(np.count_nonzero(gt_ids[intersection_mask] == target_id) / max(1, intersection_mask.sum() + target_size - np.count_nonzero(gt_ids[intersection_mask] == target_id)))
        action_scores = {"keep": keep_iou, "union": union_iou, "intersection": intersection_iou}
        best_action = max(action_scores, key=action_scores.get)
        best_iou = action_scores[best_action]
        non_keep_scores = [union_iou, intersection_iou]
        gain = best_iou - keep_iou
        row = {
            "scene_name": scene_name,
            "candidate_id": int(item["candidate_id"]),
            "candidate_class_name": str(item.get("class_name", "")),
            "candidate_point_count": candidate_size,
            "parent_mask_id": parent_id,
            "parent_candidate_iou": parent_iou,
            "parent_target_gt_instance_id": target_id,
            "parent_target_gt_class": classes[target_id],
            "keep_iou": keep_iou,
            "union_iou": union_iou,
            "intersection_iou": intersection_iou,
            "oracle_best_action": best_action,
            "oracle_best_iou": best_iou,
            "oracle_gain": gain,
            "oracle_improves": bool(best_action != "keep" and gain >= args.improvement_delta),
            "both_nonkeep_worsen": bool(max(non_keep_scores) <= keep_iou - args.improvement_delta),
            "local_overlap_pair": bool(parent_iou >= args.min_parent_iou),
            "score": float(item.get("score", 0.0)),
            "fusion_score": float(item.get("fusion_score", item.get("score", 0.0))),
            "support_score": float(item.get("support_score", 0.0)),
            "best_existing_iou": float(item.get("best_existing_iou", 0.0)),
            "seed_in_existing_mask_ratio": float(item.get("seed_in_existing_mask_ratio", 0.0)),
            "source_track_count": len(item.get("source_track_ids", [])),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_parent_iou", type=float, default=0.05)
    parser.add_argument("--improvement_delta", type=float, default=0.02)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能离线诊断。")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    rows = []
    for scene_name in scenes:
        scene_rows = _diagnose_scene(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {scene_name}: 新增 {len(scene_rows)} 条候选动作记录", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "sam2_local_correction_oracle.csv").open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    local_rows = [row for row in rows if row["local_overlap_pair"]]
    summary = {
        "gt_usage": "仅限离线 GT 诊断；绝不进入推理、候选生成、融合、打分或阈值选择。",
        "decision_rule": "只有局部重叠候选中 oracle 改善明显多于双动作均恶化，且无 GT 特征有可分层信号，才实现 MV3DIS 式局部修正。",
        "scene_count": len(scenes),
        "improvement_delta": args.improvement_delta,
        "min_parent_iou": args.min_parent_iou,
        "all_candidates": {
            "count": len(rows),
            "oracle_best_action_counts": dict(sorted(Counter(row["oracle_best_action"] for row in rows).items())),
            "oracle_improves_count": sum(row["oracle_improves"] for row in rows),
            "both_nonkeep_worsen_count": sum(row["both_nonkeep_worsen"] for row in rows),
        },
        "local_overlap_candidates": {
            "count": len(local_rows),
            "oracle_best_action_counts": dict(sorted(Counter(row["oracle_best_action"] for row in local_rows).items())),
            "oracle_improves_count": sum(row["oracle_improves"] for row in local_rows),
            "oracle_improves_rate": float(sum(row["oracle_improves"] for row in local_rows) / max(1, len(local_rows))),
            "both_nonkeep_worsen_count": sum(row["both_nonkeep_worsen"] for row in local_rows),
            "both_nonkeep_worsen_rate": float(sum(row["both_nonkeep_worsen"] for row in local_rows) / max(1, len(local_rows))),
            "feature_groups": _feature_summary(local_rows),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

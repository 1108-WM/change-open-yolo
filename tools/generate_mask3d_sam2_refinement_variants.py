#!/usr/bin/env python3
"""生成仅用于受控评测的 Mask3D + SAM2 局部修正变体。

本工具不读取 GT，也不改写原始 Mask3D 或 SAM2 候选。每个基础 Mask3D
实例至多关联一个 SAM2 候选，并输出并集、交集或保守自适应的独立 mask 根目录。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_base_masks(root, scene_name):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"{scene_name} 的 Mask3D 文件必须是 (masks, scores)：{root}")
    masks, scores = payload
    if not torch.is_tensor(masks) or masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 Mask3D mask 维度异常：{getattr(masks, 'shape', None)}")
    if not torch.is_tensor(scores) or scores.ndim != 1 or masks.shape[1] != len(scores):
        raise ValueError(f"{scene_name} 的 Mask3D score 与 mask 数量不一致")
    return masks.cpu(), scores.cpu()


def _load_candidates(root, scene_name, point_count):
    path = root / scene_name / "backprojection_candidates.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    candidates = []
    for item in payload.get("candidates", []):
        seed_path = _resolve(item["seed_points_path"])
        indices = np.unique(np.load(seed_path)["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < point_count)].astype(np.int64, copy=False)
        if len(indices):
            candidates.append((item, indices))
    return candidates


def _global_quality_thresholds(candidate_root, scenes, support_quantile, fusion_quantile):
    supports, fusions = [], []
    for scene_name in scenes:
        path = candidate_root / scene_name / "backprojection_candidates.json"
        if not path.exists():
            continue
        for item in json.loads(path.read_text()).get("candidates", []):
            supports.append(float(item.get("support_score", 0.0)))
            fusions.append(float(item.get("fusion_score", item.get("score", 0.0))))
    if not supports or not fusions:
        raise ValueError("未找到可用 SAM2 候选，无法计算自适应质量阈值")
    return {
        "support_score": float(np.quantile(np.asarray(supports), support_quantile)),
        "fusion_score": float(np.quantile(np.asarray(fusions), fusion_quantile)),
    }


def _candidate_action(mode, parent_iou, item, args, thresholds):
    if parent_iou < args.min_parent_iou:
        return None
    if mode == "union":
        return "union"
    if mode == "intersection":
        return "intersection"
    support = float(item.get("support_score", 0.0))
    fusion = float(item.get("fusion_score", item.get("score", 0.0)))
    if support < thresholds["support_score"] or fusion < thresholds["fusion_score"]:
        return None
    return "union" if parent_iou < args.adaptive_switch_iou else "intersection"


def _process_scene(scene_name, args, thresholds):
    masks, scores = _load_base_masks(args.baseline_masks_root, scene_name)
    point_count, mask_count = masks.shape
    base_bool = masks.numpy() > 0
    mask_sizes = base_bool.sum(axis=0, dtype=np.int64)
    selected = {}
    counters = Counter()

    for item, indices in _load_candidates(args.candidate_root, scene_name, point_count):
        intersections = base_bool[indices].sum(axis=0, dtype=np.int64)
        candidate_size = len(indices)
        ious = intersections / np.maximum(1, mask_sizes + candidate_size - intersections)
        parent_id = int(np.argmax(ious)) if len(ious) else -1
        parent_iou = float(ious[parent_id]) if parent_id >= 0 else 0.0
        action = _candidate_action(args.mode, parent_iou, item, args, thresholds)
        if action is None:
            counters["candidate_not_eligible"] += 1
            continue
        # 同一基础实例只使用质量最高的一条证据，避免多次并集/交集累积失控。
        quality = (
            float(item.get("fusion_score", item.get("score", 0.0))),
            float(item.get("support_score", 0.0)),
            len(item.get("source_track_ids", [])),
            candidate_size,
        )
        previous = selected.get(parent_id)
        if previous is None or quality > previous["quality"]:
            selected[parent_id] = {
                "item": item,
                "indices": indices,
                "parent_iou": parent_iou,
                "action": action,
                "quality": quality,
            }
        counters["candidate_eligible"] += 1

    refined_bool = base_bool.copy()
    actions = []
    for parent_id, record in sorted(selected.items()):
        candidate_mask = np.zeros(point_count, dtype=bool)
        candidate_mask[record["indices"]] = True
        if record["action"] == "union":
            updated = np.logical_or(refined_bool[:, parent_id], candidate_mask)
        else:
            updated = np.logical_and(refined_bool[:, parent_id], candidate_mask)
        if not np.any(updated):
            counters["empty_intersection_kept_original"] += 1
            continue
        changed_points = int(np.count_nonzero(updated != refined_bool[:, parent_id]))
        if not changed_points:
            counters["selected_but_unchanged"] += 1
            continue
        refined_bool[:, parent_id] = updated
        item = record["item"]
        actions.append(
            {
                "parent_mask_id": int(parent_id),
                "candidate_id": int(item.get("candidate_id", -1)),
                "candidate_class_name": str(item.get("class_name", "")),
                "action": record["action"],
                "parent_candidate_iou": record["parent_iou"],
                "candidate_point_count": int(len(record["indices"])),
                "changed_point_count": changed_points,
                "fusion_score": float(item.get("fusion_score", item.get("score", 0.0))),
                "support_score": float(item.get("support_score", 0.0)),
                "source_track_count": int(len(item.get("source_track_ids", []))),
            }
        )
        counters[f"applied_{record['action']}"] += 1

    output_path = args.output_root / args.mode / f"{scene_name}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((torch.from_numpy(refined_bool).to(dtype=masks.dtype), scores), output_path)
    report_path = args.output_root / args.mode / "actions" / f"{scene_name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "gt_usage": "未读取 GT。",
                "scene_name": scene_name,
                "mode": args.mode,
                "base_mask_count": int(mask_count),
                "candidate_count": int(sum(counters[key] for key in ("candidate_eligible", "candidate_not_eligible"))),
                "thresholds": thresholds,
                "actions": actions,
                "counters": dict(sorted(counters.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return counters, len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, default=Path("output/scannet200/scannet200_masks"))
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--mode", choices=("union", "intersection", "adaptive"), required=True)
    parser.add_argument("--min_parent_iou", type=float, default=0.30)
    parser.add_argument("--adaptive_switch_iou", type=float, default=0.70)
    parser.add_argument("--adaptive_support_quantile", type=float, default=0.50)
    parser.add_argument("--adaptive_fusion_quantile", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.min_parent_iou <= args.adaptive_switch_iou <= 1.0:
        raise SystemExit("IoU 阈值必须满足 0 <= min_parent_iou <= adaptive_switch_iou <= 1")
    if not 0.0 <= args.adaptive_support_quantile <= 1.0 or not 0.0 <= args.adaptive_fusion_quantile <= 1.0:
        raise SystemExit("质量分位数必须在 [0, 1] 内")
    args.baseline_masks_root = _resolve(args.baseline_masks_root)
    args.candidate_root = _resolve(args.candidate_root)
    args.output_root = _resolve(args.output_root)
    scenes = _read_scenes(_resolve(args.scene_list))
    thresholds = _global_quality_thresholds(
        args.candidate_root,
        scenes,
        args.adaptive_support_quantile,
        args.adaptive_fusion_quantile,
    )
    counters, applied = Counter(), 0
    for scene_name in scenes:
        scene_counters, scene_applied = _process_scene(scene_name, args, thresholds)
        counters.update(scene_counters)
        applied += scene_applied
        print(f"[场景完成] {scene_name}: 应用 {scene_applied} 条修正", flush=True)
    summary = {
        "gt_usage": "未读取 GT；输出仅用于后续冻结评测。",
        "mode": args.mode,
        "scene_count": len(scenes),
        "association": "每条 SAM2 候选关联三维 IoU 最大的基础 Mask3D；每个基础实例最多保留一条最高质量候选。",
        "quality_note": "旧候选没有 support_view_count；support_score 仅作为已有无 GT 支持分数，不等同于可见帧数。",
        "thresholds": {
            "min_parent_iou": args.min_parent_iou,
            "adaptive_switch_iou": args.adaptive_switch_iou,
            "adaptive_support_quantile": args.adaptive_support_quantile,
            "adaptive_fusion_quantile": args.adaptive_fusion_quantile,
            "resolved": thresholds,
        },
        "applied_action_count": applied,
        "counters": dict(sorted(counters.items())),
    }
    summary_path = args.output_root / args.mode / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

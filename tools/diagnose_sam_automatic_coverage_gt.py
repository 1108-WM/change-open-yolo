#!/usr/bin/env python3
"""仅离线 GT 账本：评估类别无关 SAM 自动 mask 的多视角对象覆盖。

GT 仅用于回答自动 mask 是否有能力观察强基线严格漏掉的真实实例。关联、语义、候选
生成和阈值不得读取本脚本输出。
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id <= 0 or instance_id // 1000 not in valid:
            continue
        size = int(np.sum(gt_ids == instance_id))
        if size >= min_region_size:
            sizes[instance_id] = size
    return gt_ids, sizes


def _load_prediction(root, scene_name, point_count):
    masks = np.load(f"{root / scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的最终预测点数不匹配")
    return np.asarray(masks, dtype=bool)


def _final_iou_by_gt(masks, gt_ids, gt_sizes):
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    result = {}
    for instance_id, size in gt_sizes.items():
        indices = np.flatnonzero(gt_ids == instance_id)
        if masks.shape[1] == 0:
            result[instance_id] = 0.0
            continue
        intersection = masks[indices].sum(axis=0, dtype=np.int64)
        result[instance_id] = float(np.max(intersection / np.maximum(1, mask_sizes + size - intersection)))
    return result


def _read_observations(scene_root):
    path = scene_root / "automatic_observations.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少自动 SAM 观测：{path}")
    records = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            points = np.asarray(np.load(row["point_indices_path"])["point_indices"], dtype=np.int64)
            row["points"] = np.unique(points)
            records.append(row)
    return records


def _collect_support(observations, gt_ids, eligible_ids, min_points, min_precision):
    """GT-only 关联：每帧只保留对同一真实实例最有解释力的自动 mask。"""
    by_gt_frame = defaultdict(dict)
    eligible = set(eligible_ids)
    for observation in observations:
        points = observation["points"]
        if len(points) == 0:
            continue
        ids, counts = np.unique(gt_ids[points], return_counts=True)
        for instance_id, intersection in zip(ids, counts):
            instance_id = int(instance_id)
            intersection = int(intersection)
            if instance_id not in eligible or intersection < min_points:
                continue
            precision = float(intersection / len(points))
            if precision < min_precision:
                continue
            candidate = {
                "points": points,
                "intersection": intersection,
                "precision": precision,
                "predicted_iou": float(observation["predicted_iou"]),
                "stability_score": float(observation["stability_score"]),
            }
            frame_id = str(observation["frame_id"])
            previous = by_gt_frame[instance_id].get(frame_id)
            if previous is None or (candidate["intersection"], candidate["precision"]) > (
                previous["intersection"], previous["precision"]
            ):
                by_gt_frame[instance_id][frame_id] = candidate
    return by_gt_frame


def _row_for_gt(scene_name, instance_id, size, final_iou, frame_support, min_support_frames):
    selected = list(frame_support.values())
    union_points = np.unique(np.concatenate([item["points"] for item in selected])) if selected else np.empty(0, dtype=np.int64)
    return {
        "scene_name": scene_name,
        "gt_instance_id": int(instance_id),
        "gt_point_count": int(size),
        "native_best_iou": float(final_iou),
        "residual_type": "无合格三维候选" if final_iou < 0.25 else "边界不足",
        "support_frame_count": len(frame_support),
        "support_observation_count": len(selected),
        "multi_view_supported": len(frame_support) >= min_support_frames,
        "auto_union_point_count": int(len(union_points)),
        "auto_union_points": union_points,
        "mean_auto_precision": float(np.mean([item["precision"] for item in selected])) if selected else 0.0,
        "mean_predicted_iou": float(np.mean([item["predicted_iou"] for item in selected])) if selected else 0.0,
        "mean_stability_score": float(np.mean([item["stability_score"] for item in selected])) if selected else 0.0,
    }


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks = _load_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
    final_iou = _final_iou_by_gt(masks, gt_ids, gt_sizes)
    strict_residuals = {instance_id: value for instance_id, value in final_iou.items() if value < 0.50}
    observations = _read_observations(args.automatic_root / scene_name)
    support = _collect_support(
        observations,
        gt_ids,
        strict_residuals,
        args.min_observation_gt_points,
        args.min_observation_precision,
    )
    rows = []
    for instance_id, iou in strict_residuals.items():
        row = _row_for_gt(
            scene_name,
            instance_id,
            gt_sizes[instance_id],
            iou,
            support.get(instance_id, {}),
            args.min_support_frames,
        )
        points = row.pop("auto_union_points")
        if len(points):
            intersection = int(np.sum(gt_ids[points] == instance_id))
            union = len(points) + gt_sizes[instance_id] - intersection
            row["auto_union_iou"] = float(intersection / max(1, union))
            row["auto_union_precision"] = float(intersection / max(1, len(points)))
            row["auto_union_coverage"] = float(intersection / max(1, gt_sizes[instance_id]))
        else:
            row["auto_union_iou"] = 0.0
            row["auto_union_precision"] = 0.0
            row["auto_union_coverage"] = 0.0
        rows.append(row)
    return rows


def _summary(rows):
    def count(predicate):
        return sum(bool(predicate(row)) for row in rows)

    total = len(rows)
    groups = {}
    for name, predicate in {
        "至少一帧高纯度自动 mask": lambda row: row["support_frame_count"] >= 1,
        "至少两帧高纯度自动 mask": lambda row: row["multi_view_supported"],
        "多视角后自动 mask 并集 IoU 不低于 25%": lambda row: row["multi_view_supported"] and row["auto_union_iou"] >= 0.25,
        "多视角后自动 mask 并集 IoU 不低于 50%": lambda row: row["multi_view_supported"] and row["auto_union_iou"] >= 0.50,
    }.items():
        value = count(predicate)
        groups[name] = {"实例数": value, "比例": float(value / max(1, total))}
    by_type = {}
    for residual_type in ("无合格三维候选", "边界不足"):
        subset = [row for row in rows if row["residual_type"] == residual_type]
        by_type[residual_type] = {
            "严格残差实例数": len(subset),
            "至少两帧高纯度自动 mask": count(
                lambda row: row["residual_type"] == residual_type and row["multi_view_supported"]
            ),
            "多视角自动 mask 并集 IoU 不低于 25%": count(
                lambda row: row["residual_type"] == residual_type
                and row["multi_view_supported"]
                and row["auto_union_iou"] >= 0.25
            ),
        }
    return {
        "严格残差实例数": total,
        "指标": groups,
        "按强基线残差类型": by_type,
        "说明": "这是 GT-only 的对象观测上限账本。GT 在每帧选择最匹配自动 mask，故结果不能直接视作无 GT 方法性能。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_observation_gt_points", type=int, default=20)
    parser.add_argument("--min_observation_precision", type=float, default=0.50)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线对象覆盖归因。")
    for name in ("scene_list", "automatic_root", "prediction_cache_dir", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")

    rows = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 个严格残差实例", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "sam_automatic_coverage_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线对象覆盖账本，绝不进入自动 mask 导出、语义、关联、候选、融合、评分或阈值。",
        "scene_count": len(scenes),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

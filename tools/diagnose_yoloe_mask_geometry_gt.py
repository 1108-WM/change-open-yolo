#!/usr/bin/env python3
"""GT-only：审计 YOLOE 独有二维证据的分割 mask 几何上限。

GT 仅在既有框级账本判定为“仅 YOLOE 有可靠二维证据”的严格残差上，逐帧选择
最匹配的同类预测 mask 并计算其多视角点并集。这是对象观测上限，不能回流到推理。
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
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


def _load_observations(scene_root):
    path = scene_root / "mask_observations.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 YOLOE mask 观测：{path}")
    result = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["points"] = np.asarray(np.load(row["point_indices_path"])["point_indices"], dtype=np.int64)
            result.append(row)
    return result


def _load_yoloe_only_targets(path):
    result = defaultdict(set)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] == "仅 YOLOE 有可靠二维证据":
                result[row["scene_name"]].add(int(row["gt_instance_id"]))
    return result


def _collect_support(observations, gt_ids, instance_id, label_id, min_points, min_precision):
    """GT-only 地为每个帧保留最解释该实例的一个同类预测 mask。"""
    by_frame = {}
    for observation in observations:
        if int(observation["label_id"]) != label_id:
            continue
        points = observation["points"]
        if len(points) == 0:
            continue
        intersection = int(np.sum(gt_ids[points] == instance_id))
        if intersection < min_points:
            continue
        precision = float(intersection / len(points))
        if precision < min_precision:
            continue
        candidate = {
            "points": points,
            "intersection": intersection,
            "precision": precision,
            "score": float(observation["score"]),
        }
        frame_id = str(observation["frame_id"])
        previous = by_frame.get(frame_id)
        if previous is None or (candidate["intersection"], candidate["precision"], candidate["score"]) > (
            previous["intersection"], previous["precision"], previous["score"]
        ):
            by_frame[frame_id] = candidate
    return by_frame


def _row(scene_name, instance_id, gt_ids, size, coverage_row, observations, label_id, args):
    support = _collect_support(
        observations,
        gt_ids,
        instance_id,
        label_id,
        args.min_mask_gt_points,
        args.min_mask_precision,
    )
    selected = list(support.values())
    points = np.unique(np.concatenate([item["points"] for item in selected])) if selected else np.empty(0, dtype=np.int64)
    intersection = int(np.sum(gt_ids[points] == instance_id)) if len(points) else 0
    union = int(len(points) + size - intersection)
    native_iou = float(coverage_row["native_best_iou"])
    return {
        "scene_name": scene_name,
        "gt_instance_id": instance_id,
        "gt_class": coverage_row["gt_class"],
        "gt_point_count": size,
        "native_best_iou": native_iou,
        "residual_type": "无合格三维候选" if native_iou < 0.25 else "边界不足",
        "support_frame_count": len(support),
        "multi_view_supported": len(support) >= args.min_support_frames,
        "mask_union_point_count": int(len(points)),
        "mask_union_iou": float(intersection / max(1, union)),
        "mask_union_precision": float(intersection / max(1, len(points))),
        "mask_union_coverage": float(intersection / max(1, size)),
        "mean_mask_precision": float(np.mean([item["precision"] for item in selected])) if selected else 0.0,
        "mean_mask_score": float(np.mean([item["score"] for item in selected])) if selected else 0.0,
    }


def _summary(rows):
    total = len(rows)

    def count(predicate):
        return sum(bool(predicate(row)) for row in rows)

    measures = {
        "至少两帧高纯度 YOLOE mask": lambda row: row["multi_view_supported"],
        "多视角 YOLOE mask 并集 IoU 不低于 25%": lambda row: row["multi_view_supported"] and row["mask_union_iou"] >= 0.25,
        "多视角 YOLOE mask 并集 IoU 不低于 50%": lambda row: row["multi_view_supported"] and row["mask_union_iou"] >= 0.50,
    }
    by_type = {}
    for residual_type in ("无合格三维候选", "边界不足"):
        subset = [row for row in rows if row["residual_type"] == residual_type]
        by_type[residual_type] = {
            "YOLOE 独有严格残差实例数": len(subset),
            **{
                name: sum(bool(predicate(row)) for row in subset)
                for name, predicate in measures.items()
            },
        }
    return {
        "YOLOE 独有严格残差实例数": total,
        "指标": {
            name: {"实例数": count(predicate), "比例": float(count(predicate) / max(1, total))}
            for name, predicate in measures.items()
        },
        "按强基线残差类型": by_type,
        "说明": "GT 在每帧选择最匹配同类 YOLOE mask；这是几何观测上限，不是无 GT 候选性能。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--mask_observation_root", type=Path, required=True)
    parser.add_argument("--box_coverage_csv", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_mask_gt_points", type=int, default=20)
    parser.add_argument("--min_mask_precision", type=float, default=0.50)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线几何上限诊断。")
    for name in ("scene_list", "mask_observation_root", "box_coverage_csv", "config_path", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    targets = _load_yoloe_only_targets(args.box_coverage_csv)
    with args.config_path.open() as handle:
        config = yaml.safe_load(handle)
    prompt_to_id = {str(name): index for index, name in enumerate(config["network2d"]["text_prompts"])}
    coverage_rows = {}
    with args.box_coverage_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] == "仅 YOLOE 有可靠二维证据":
                coverage_rows[(row["scene_name"], int(row["gt_instance_id"]))] = row
    rows = []
    for index, scene_name in enumerate(_read_scenes(args.scene_list), start=1):
        selected = targets.get(scene_name, set())
        if not selected:
            continue
        gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        observations = _load_observations(args.mask_observation_root / scene_name)
        for instance_id in sorted(selected):
            if instance_id not in gt_sizes:
                continue
            coverage_row = coverage_rows[(scene_name, instance_id)]
            label_id = prompt_to_id.get(coverage_row["gt_class"])
            if label_id is None:
                continue
            rows.append(
                _row(
                    scene_name,
                    instance_id,
                    gt_ids,
                    gt_sizes[instance_id],
                    coverage_row,
                    observations,
                    label_id,
                    args,
                )
            )
        print(f"[场景完成] {index} {scene_name}: {len(selected)} 个 YOLOE 独有严格残差", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scene_name", "gt_instance_id", "gt_class"]
    with (args.output_dir / "yoloe_mask_geometry_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "诊断限定": "GT 仅用于离线几何上限账本；绝不进入 YOLOE mask 导出、候选生成、融合、评分、阈值或 AP。",
        "scene_count": len(_read_scenes(args.scene_list)),
        "汇总": _summary(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

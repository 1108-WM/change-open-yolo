#!/usr/bin/env python3
"""仅离线 GT 诊断：建立最终强基线遗漏实例的多视角残差覆盖账本。

账本不产生候选，也不修改任何推理结果。它只回答一个前置问题：最终三维
预测没有以足够 IoU 恢复的真实实例，是否已在多个视图中拥有同类别的
YOLO-World 二维检测框证据。若有，该实例是后续“多视角残差候选补全”可
尝试解决的对象；若没有，优先问题是二维观测而不是三维形成。

GT 仅用于离线归因，绝不进入推理、候选生成、融合、打分或阈值选择。
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


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


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    instances = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        class_id = instance_id // 1000
        if instance_id <= 0 or class_id not in valid_classes:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) < min_region_size:
            continue
        instances.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": str(ID_TO_LABEL.get(class_id, class_id)),
                "indices": indices,
                "point_count": int(len(indices)),
            }
        )
    return gt_ids, instances


def _load_final_prediction(root, scene_name, point_count):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的最终预测点数不一致：{masks.shape[0]} 与 {point_count}")
    return np.asarray(masks, dtype=bool)


def _load_2d_predictions(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        raise ValueError(f"二维缓存格式异常：{path}")
    return payload


def _best_iou(masks, mask_sizes, instance):
    if masks.shape[1] == 0:
        return 0.0
    intersections = masks[instance["indices"]].sum(axis=0, dtype=np.int64)
    unions = mask_sizes + instance["point_count"] - intersections
    return float(np.max(intersections / np.maximum(1, unions)))


def _same_class_box_coverage(coords_depth, boxes, labels, scores, prompt_id, scaling_params):
    eligible = labels == prompt_id
    if not np.any(eligible):
        return 0.0, 0.0
    boxes = boxes[eligible]
    scores = scores[eligible]
    xs = coords_depth[:, 0].astype(np.float32) / float(scaling_params[1])
    ys = coords_depth[:, 1].astype(np.float32) / float(scaling_params[0])
    inside = (
        (xs[:, None] >= boxes[None, :, 0])
        & (xs[:, None] <= boxes[None, :, 2])
        & (ys[:, None] >= boxes[None, :, 1])
        & (ys[:, None] <= boxes[None, :, 3])
    )
    coverages = inside.mean(axis=0)
    best = int(np.argmax(coverages))
    return float(coverages[best]), float(scores[best])


def _size_bucket(point_count, boundaries):
    if point_count <= boundaries[0]:
        return "最小四分位"
    if point_count <= boundaries[1]:
        return "第二四分位"
    if point_count <= boundaries[2]:
        return "第三四分位"
    return "最大四分位"


def _diagnose_scene(scene_name, args, prompt_to_id, size_boundaries):
    # 延迟导入，保持 --help 和静态检查不依赖三维投影环境。
    from utils import WORLD_2_CAM

    gt_ids, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    masks = _load_final_prediction(args.prediction_cache_dir, scene_name, len(gt_ids))
    mask_sizes = masks.sum(axis=0, dtype=np.int64)
    predictions_2d = _load_2d_predictions(args.bboxes_2d_root / f"{scene_name}.pt")
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = _as_numpy(projections)
    visibility = _as_numpy(visibility).astype(bool, copy=False)
    frame_count = min(int(args.max_frames), len(world.color_paths), projections.shape[0])
    scaling_params = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )

    rows = []
    for instance in instances:
        best_iou = _best_iou(masks, mask_sizes, instance)
        # 只有最终基线在严格几何阈值下未恢复的实例才属于“残差”。
        if best_iou >= args.residual_iou:
            continue

        prompt_id = prompt_to_id.get(instance["class_name"])
        usable_frames = 0
        matched_frames = []
        best_coverage = 0.0
        best_score = 0.0
        if prompt_id is not None:
            for frame_index in range(frame_count):
                visible_indices = instance["indices"][visibility[frame_index, instance["indices"]]]
                if len(visible_indices) < args.min_visible_points:
                    continue
                frame_id = Path(world.color_paths[frame_index]).stem
                prediction = predictions_2d.get(frame_id)
                if prediction is None:
                    continue
                usable_frames += 1
                coverage, score = _same_class_box_coverage(
                    projections[frame_index, visible_indices],
                    _as_numpy(prediction["bbox"]).astype(np.float32, copy=False),
                    _as_numpy(prediction["labels"]).astype(np.int64, copy=False),
                    _as_numpy(prediction["scores"]).astype(np.float32, copy=False),
                    prompt_id,
                    scaling_params,
                )
                best_coverage = max(best_coverage, coverage)
                best_score = max(best_score, score)
                if coverage >= args.min_box_point_coverage:
                    matched_frames.append(frame_id)

        coarse_geometry_exists = best_iou >= args.coarse_iou
        if prompt_id is None:
            evidence_group = "类别不在二维提示词中"
        elif len(matched_frames) >= args.min_matched_frames:
            evidence_group = "稳定二维残差证据"
        elif matched_frames:
            evidence_group = "仅单帧二维证据"
        else:
            evidence_group = "无可靠二维证据"
        rows.append(
            {
                "scene_name": scene_name,
                "gt_instance_id": instance["instance_id"],
                "gt_class": instance["class_name"],
                "gt_point_count": instance["point_count"],
                "size_bucket": _size_bucket(instance["point_count"], size_boundaries),
                "best_final_prediction_iou": best_iou,
                "residual_type": "边界不足" if coarse_geometry_exists else "无合格三维候选",
                "class_in_yoloworld_prompts": prompt_id is not None,
                "usable_visible_frame_count": usable_frames,
                "box_covered_frame_count": len(matched_frames),
                "box_covered_frame_ids": ";".join(matched_frames),
                "best_box_point_coverage": best_coverage,
                "best_box_score": best_score,
                "evidence_group": evidence_group,
                "residual_completion_candidate": evidence_group == "稳定二维残差证据",
            }
        )

    del world, projections, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _summarize(rows):
    evidence_counts = Counter(row["evidence_group"] for row in rows)
    residual_counts = Counter(row["residual_type"] for row in rows)
    size_counts = {}
    for bucket in ("最小四分位", "第二四分位", "第三四分位", "最大四分位"):
        group = [row for row in rows if row["size_bucket"] == bucket]
        stable = sum(row["residual_completion_candidate"] for row in group)
        size_counts[bucket] = {
            "残差实例数": len(group),
            "稳定二维残差证据数": stable,
            "稳定二维残差证据比例": stable / max(1, len(group)),
        }
    by_class = {}
    for class_name, count in Counter(row["gt_class"] for row in rows).most_common():
        group = [row for row in rows if row["gt_class"] == class_name]
        stable = sum(row["residual_completion_candidate"] for row in group)
        by_class[class_name] = {
            "残差实例数": count,
            "稳定二维残差证据数": stable,
            "稳定二维残差证据比例": stable / max(1, count),
        }
    stable_count = sum(row["residual_completion_candidate"] for row in rows)
    return {
        "残差实例数": len(rows),
        "稳定二维残差证据数": stable_count,
        "稳定二维残差证据比例": stable_count / max(1, len(rows)),
        "按二维证据分组": dict(sorted(evidence_counts.items())),
        "按三维残差类型分组": dict(sorted(residual_counts.items())),
        "按实例大小分组": size_counts,
        "按类别分组": by_class,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--bboxes_2d_root", type=Path, default=Path("output/scannet200/bboxes_2d"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--min_box_point_coverage", type=float, default=0.50)
    parser.add_argument("--min_matched_frames", type=int, default=2)
    parser.add_argument("--coarse_iou", type=float, default=0.25)
    parser.add_argument("--residual_iou", type=float, default=0.50)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow_gt_diagnostics；GT 只能用于离线诊断。")
    if not 0.0 < args.coarse_iou < args.residual_iou <= 1.0:
        raise SystemExit("要求 0 < --coarse_iou < --residual_iou <= 1。")

    for name in (
        "scene_list", "prediction_cache_dir", "dataset_root", "bboxes_2d_root",
        "gt_instance_dir", "config_path", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    prompts = [str(item) for item in args.config["network2d"]["text_prompts"]]
    prompt_to_id = {name: index for index, name in enumerate(prompts)}
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]

    # 固定全体 GT 实例大小分位边界，只用于报告切片，不参与任何判断。
    all_sizes = []
    for scene_name in scenes:
        _, instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        all_sizes.extend(item["point_count"] for item in instances)
    size_boundaries = np.percentile(all_sizes, [25, 50, 75]).astype(np.int64).tolist()

    rows = []
    for scene_index, scene_name in enumerate(scenes, start=1):
        scene_rows = _diagnose_scene(scene_name, args, prompt_to_id, size_boundaries)
        rows.extend(scene_rows)
        print(f"[场景完成] {scene_index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条严格残差", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "gt_multiview_residual_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "诊断限定": "GT 仅用于离线归因；绝不进入推理、候选生成、融合、打分或阈值选择。",
        "证据限定": "当前活动输出未保存逐帧二值 SAM mask；本版以缓存 YOLO-World 同类别二维检测框覆盖 GT 可见点的比例近似二维证据。",
        "方法决策": "若稳定二维残差证据占比足够，再重建二值 SAM mask 残差并实现多视角三维补全；若占比很低，优先改进二维观测。",
        "scene_count": len(scenes),
        "max_frames": args.max_frames,
        "min_visible_points": args.min_visible_points,
        "min_box_point_coverage": args.min_box_point_coverage,
        "min_matched_frames": args.min_matched_frames,
        "coarse_iou": args.coarse_iou,
        "residual_iou": args.residual_iou,
        "size_quartile_boundaries_point_count": size_boundaries,
        "汇总": _summarize(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

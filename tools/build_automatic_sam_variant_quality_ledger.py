#!/usr/bin/env python3
"""为自动 SAM 的局部生长变体建立类别无关二维--三维质量账本。

本工具只计算自动 SAM 多视图一致性、深度可见性和 native 的类别无关几何关系。
它不读取 GT、不使用 YOLO-World 类别、不生成最终候选、不做 NMS 或 AP。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _box_iou(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return float(intersection / max(union, 1e-8))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "p90": 0.0}
    return {"mean": float(values.mean()), "std": float(values.std()), "p90": float(np.quantile(values, 0.90))}


def class_agnostic_gvc_from_views(view_rows, max_views):
    """保留 GVC 的可见视图汇总形式，但不在几何阶段使用类别。"""
    selected = sorted(view_rows, key=lambda row: (-row["visible_point_count"], row["frame_index"]))[:max_views]
    scores = [row["gvc_frame_score"] for row in selected]
    boxes = [row["box_iou"] for row in selected]
    supports = [row["mask_point_support"] for row in selected]
    return {
        "gvc_eligible_view_count": len(view_rows),
        "gvc_selected_view_count": len(selected),
        "gvc_matched_view_count": sum(row["matched_observation_id"] >= 0 for row in selected),
        "gvc_selected_match_ratio": float(sum(row["matched_observation_id"] >= 0 for row in selected) / max(1, len(selected))),
        "gvc_score": float(np.mean(scores)) if scores else 0.0,
        "gvc_box_iou": _summary(boxes),
        "gvc_mask_point_support": _summary(supports),
        "gvc_frame_score": _summary(scores),
        "gvc_selected_frames": selected,
    }


def _load_automatic_observations(scene_root):
    by_frame = defaultdict(list)
    with (scene_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            point_path = Path(row["point_indices_path"])
            if not point_path.is_absolute():
                point_path = scene_root / point_path
            points = np.unique(np.asarray(np.load(point_path)["point_indices"], dtype=np.int64))
            x, y, width, height = (float(value) for value in row["bbox_xywh"])
            by_frame[int(row["frame_index"])].append({
                "observation_id": int(row["observation_id"]),
                "bbox": np.asarray([x, y, x + width, y + height], dtype=np.float64),
                "points": points,
                "quality": float(row["predicted_iou"]) * float(row["stability_score"]),
            })
    return by_frame


def _load_selected_frames(scene_root):
    payload = json.loads((scene_root / "summary.json").read_text())
    return sorted(int(row["frame_index"]) for row in payload["frames"])


def _load_native_masks(root, scene_name, point_count):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 native mask 维度异常：{masks.shape}")
    if masks.shape[0] != point_count and masks.shape[1] == point_count:
        masks = masks.T
    if masks.shape[0] != point_count:
        raise ValueError(f"{scene_name} 的 native mask 点数不匹配")
    return np.asarray(masks, dtype=bool)


def _native_relation(points, masks):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if len(points) == 0 or masks.shape[1] == 0:
        return {
            "native_top_candidate_id": -1, "native_top_iou": 0.0,
            "variant_inside_top_native_ratio": 0.0, "native_overlap_count": 0,
        }
    sizes = masks.sum(axis=0, dtype=np.int64)
    intersection = masks[points].sum(axis=0, dtype=np.int64)
    ious = intersection / np.maximum(1, len(points) + sizes - intersection)
    top = int(np.argmax(ious))
    return {
        "native_top_candidate_id": top,
        "native_top_iou": float(ious[top]),
        "variant_inside_top_native_ratio": float(intersection[top] / max(1, len(points))),
        "native_overlap_count": int(np.sum(ious >= 0.10)),
    }


def _raw_superpoint_points(processed):
    raw_ids = np.asarray(processed[:, 9], dtype=np.int64)
    order = np.argsort(raw_ids, kind="mergesort")
    sorted_ids = raw_ids[order]
    ids, start = np.unique(sorted_ids, return_index=True)
    end = np.append(start[1:], len(order))
    return {
        int(superpoint_id): np.asarray(order[left:right], dtype=np.int64)
        for superpoint_id, left, right in zip(ids, start, end)
    }


def _variant_points(variant, superpoint_points):
    superpoint_ids = [int(item) for item in variant["base_superpoint_ids"]]
    superpoint_ids.extend(int(item) for item in variant.get("added_superpoint_ids", []))
    chunks = [superpoint_points[item] for item in sorted(set(superpoint_ids)) if item in superpoint_points]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64, copy=False)


def _view_row(points, frame_index, projections, visibility, scaling, observations, min_visible_points):
    visible_points = points[visibility[frame_index, points]]
    if len(visible_points) < int(min_visible_points):
        return None
    coords = projections[frame_index, visible_points]
    xs, ys = coords[:, 0] / scaling[1], coords[:, 1] / scaling[0]
    projected_box = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    candidates = observations.get(int(frame_index), [])
    if not candidates:
        return {
            "frame_index": int(frame_index), "visible_point_count": int(len(visible_points)),
            "matched_observation_id": -1, "box_iou": 0.0, "mask_point_support": 0.0, "gvc_frame_score": 0.0,
        }
    selected = max(
        candidates,
        key=lambda item: (_box_iou(projected_box, item["bbox"]), item["quality"], -item["observation_id"]),
    )
    box_iou = _box_iou(projected_box, selected["bbox"])
    support = float(
        len(np.intersect1d(visible_points, selected["points"], assume_unique=True)) / max(1, len(visible_points))
    )
    return {
        "frame_index": int(frame_index), "visible_point_count": int(len(visible_points)),
        "matched_observation_id": int(selected["observation_id"]), "box_iou": box_iou,
        "mask_point_support": support, "gvc_frame_score": float(box_iou * support),
    }


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    variants = []
    with (args.variant_plan_root / scene_name / "automatic_sam_growth_variant_plan.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                variants.append(json.loads(line))
    if args.max_variants_per_scene is not None:
        variants = variants[: args.max_variants_per_scene]
    processed = np.load(
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r"
    )
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoint_points = _raw_superpoint_points(processed)
    automatic_scene = args.automatic_root / scene_name
    observations = _load_automatic_observations(automatic_scene)
    selected_frames = _load_selected_frames(automatic_scene)
    native_masks = _load_native_masks(args.native_prediction_cache, scene_name, len(processed))
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.float64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    records = []
    for variant in variants:
        points = _variant_points(variant, superpoint_points)
        view_rows = []
        for frame_index in selected_frames:
            row = _view_row(points, frame_index, projections, visibility, scaling, observations, args.min_visible_points)
            if row is not None:
                view_rows.append(row)
        gvc = class_agnostic_gvc_from_views(view_rows, args.max_views)
        relation = _native_relation(points, native_masks)
        records.append({
            "scene_name": scene_name,
            "variant_id": variant["variant_id"],
            "variant_type": variant["variant_type"],
            "source_track_id": int(variant["source_track_id"]),
            "source_view_count": int(variant["source_view_count"]),
            "base_superpoint_count": int(variant["base_superpoint_count"]),
            "added_superpoint_count": len(variant.get("added_superpoint_ids", [])),
            "variant_point_count": int(len(points)),
            "internal_cross_view_edge_count": int(variant["internal_cross_view_edge_count"]),
            "mean_internal_reprojection_support": float(variant["mean_internal_reprojection_support"]),
            "growth_geometry": {
                key: variant[key] for key in (
                    "adjacent_seed_superpoint_count", "boundary_contact_count_sum", "mean_normal_difference",
                    "mean_color_difference", "mean_boundary_distance", "cross_view_link_count",
                    "mean_cross_view_reprojection_support", "linked_seed_node_count",
                ) if key in variant
            },
            "candidate_source": "automatic_sam_seed_or_one_hop_geometry_variant",
            "semantic_state": "类别无关；不在此账本使用 YOLO-World 投票。",
            "decision_state": "连续质量特征，不含接受阈值、类别、最终分数或候选写回。",
            **relation,
            **gvc,
        })
    del world, projections, visibility, native_masks, processed
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--variant_plan_root", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--native_prediction_cache", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_views", type=int, default=10)
    parser.add_argument("--min_visible_points", type=int, default=30)
    parser.add_argument("--max_variants_per_scene", type=int)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in (
        "scene_list", "variant_plan_root", "automatic_root", "native_prediction_cache", "dataset_root",
        "processed_scene_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_views <= 0 or args.min_visible_points <= 0:
        raise SystemExit("--max_views 与 --min_visible_points 必须为正数。")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        records = _scene_records(scene_name, args)
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        (scene_root / "automatic_sam_variant_quality_ledger.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "scene_name": scene_name,
            "variant_count": len(records),
            "with_gvc_match_count": sum(row["gvc_matched_view_count"] > 0 for row in records),
        }
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条变体质量记录", flush=True)
    payload = {
        "purpose": "为自动 SAM 的局部几何变体记录类别无关多视图和 native 连续质量特征。",
        "gt_usage": "不读取 GT；不生成最终候选、不赋类别、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "variant_count": sum(item["variant_count"] for item in summaries),
        "with_gvc_match_count": sum(item["with_gvc_match_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_sam_variant_quality_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

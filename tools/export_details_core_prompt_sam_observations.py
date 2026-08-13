#!/usr/bin/env python3
"""执行固定三维核心点计划，导出类别无关 SAM 点提示观测。

每次单点提示保留 SAM ``multimask_output`` 返回的三个假设，保存精确二维 RLE、
三维回投点以及对共同三维核心的连续覆盖指标。工具不读取 GT、类别、语义或
native，不选择最佳假设、不聚合轨迹、不形成候选，也不运行 AP。
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from export_sam_automatic_observations import (
    encode_binary_mask_rle,
    mask_to_visible_points,
)
from refine_details_automatic_tracks_consensus import _superpoint_points
from refine_details_consensus_all_view_reobservation import _read_scenes, _resolve


def group_prompt_requests(plans):
    """按帧组织提示，并拒绝同一轨迹--帧重复请求。"""
    grouped = defaultdict(list)
    seen = set()
    for plan in plans:
        track_id = int(plan["track_id"])
        for frame in plan.get("prompt_frames", []):
            frame_index = int(frame["frame_index"])
            key = (track_id, frame_index)
            if key in seen:
                raise ValueError(f"重复提示请求：track={track_id}, frame={frame_index}")
            seen.add(key)
            grouped[frame_index].append({
                "track_id": track_id,
                "prompt_point_index": int(plan["prompt_point_index"]),
                "prompt_superpoint_id": int(plan["prompt_superpoint_id"]),
                "common_core_superpoint_ids": [
                    int(item) for item in plan["common_core_superpoint_ids"]
                ],
                "frame_id": str(frame["frame_id"]),
                "frame_index": frame_index,
                "prompt_xy": [float(value) for value in frame["prompt_xy"]],
                "planned_visible_common_core_point_count": int(
                    frame["visible_common_core_point_count"]
                ),
            })
    return {
        frame_index: sorted(rows, key=lambda row: row["track_id"])
        for frame_index, rows in sorted(grouped.items())
    }


def hypothesis_core_metrics(point_indices, visible_core_points, prompt_point_index):
    points = np.unique(np.asarray(point_indices, dtype=np.int64))
    core = np.unique(np.asarray(visible_core_points, dtype=np.int64))
    intersection = int(np.intersect1d(points, core, assume_unique=True).size)
    return {
        "visible_core_point_count": int(len(core)),
        "visible_core_covered_point_count": intersection,
        "visible_core_support_ratio": float(intersection / max(1, len(core))),
        "observation_core_purity_ratio": float(intersection / max(1, len(points))),
        "prompt_point_backprojected": bool(
            np.searchsorted(points, int(prompt_point_index)) < len(points)
            and points[np.searchsorted(points, int(prompt_point_index))]
            == int(prompt_point_index)
        ),
    }


def _load_predictor(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "当前 CUDA 不可用；拒绝让 SAM 点提示静默回退 CPU。恢复 GPU 后按固定计划重试。"
        )
    source = str(args.sam_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    model.to(device=args.device)
    model.eval()
    return SamPredictor(model)


def _export_scene(scene_name, predictor, args):
    from utils import WORLD_2_CAM

    plans = json.loads((args.plan_root / scene_name / "core_prompt_plan.json").read_text())
    grouped = group_prompt_requests(plans)
    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint = _superpoint_points(superpoints)

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections_raw, visibility_raw = world.get_mesh_projections()
    projections = projections_raw.detach().cpu().numpy().astype(np.float64, copy=False)
    visibility = visibility_raw.detach().cpu().numpy().astype(bool, copy=False)
    scaling_params = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    points_root = staging_root / "points"
    points_root.mkdir(parents=True)
    records = []
    frame_summaries = []
    for frame_index, requests in grouped.items():
        image_path = Path(world.color_paths[frame_index])
        if image_path.stem != requests[0]["frame_id"]:
            raise ValueError(f"{scene_name} frame {frame_index} 的 frame_id 与计划不一致")
        image = np.asarray(imageio.imread(image_path))
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"RGB 图像格式异常：{image_path}")
        predictor.set_image(np.ascontiguousarray(image[:, :, :3]))
        frame_record_count = 0
        for request in requests:
            prompt_xy = np.asarray([request["prompt_xy"]], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                point_coords=prompt_xy,
                point_labels=np.asarray([1], dtype=np.int64),
                multimask_output=True,
            )
            if masks.ndim != 3 or len(masks) != len(scores):
                raise ValueError("SAM 点提示输出维度异常")
            common_core_points = [
                points_by_superpoint[superpoint_id]
                for superpoint_id in request["common_core_superpoint_ids"]
                if superpoint_id in points_by_superpoint
            ]
            common_core_points = np.unique(
                np.concatenate(common_core_points).astype(np.int64, copy=False)
            )
            visible_core_points = common_core_points[
                visibility[frame_index, common_core_points]
            ]
            if len(visible_core_points) != request["planned_visible_common_core_point_count"]:
                raise ValueError("执行时共同核心可见点数与固定计划不一致")
            for hypothesis_index, (mask, score) in enumerate(zip(masks, scores)):
                mask = np.asarray(mask, dtype=bool)
                point_indices = mask_to_visible_points(
                    mask,
                    frame_index,
                    projections,
                    visibility,
                    scaling_params,
                )
                observation_id = len(records)
                points_path = points_root / f"obs{observation_id:06d}_points.npz"
                np.savez_compressed(points_path, point_indices=point_indices)
                metrics = hypothesis_core_metrics(
                    point_indices,
                    visible_core_points,
                    request["prompt_point_index"],
                )
                records.append({
                    "observation_id": observation_id,
                    "scene_name": scene_name,
                    "track_id": request["track_id"],
                    "frame_id": request["frame_id"],
                    "frame_index": frame_index,
                    "prompt_point_index": request["prompt_point_index"],
                    "prompt_superpoint_id": request["prompt_superpoint_id"],
                    "prompt_xy": request["prompt_xy"],
                    "hypothesis_index": int(hypothesis_index),
                    "sam_predicted_iou": float(score),
                    "mask_area": int(mask.sum()),
                    "point_count": int(len(point_indices)),
                    "point_indices_path": str(
                        args.output_root / scene_name / "points" / points_path.name
                    ),
                    "mask_rle": encode_binary_mask_rle(mask),
                    **metrics,
                })
                frame_record_count += 1
        frame_summaries.append({
            "frame_id": image_path.stem,
            "frame_index": int(frame_index),
            "prompt_request_count": len(requests),
            "hypothesis_count": frame_record_count,
        })

    with (staging_root / "prompt_observations.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "scene_name": scene_name,
        "planned_track_count": len(plans),
        "prompt_request_count": sum(len(rows) for rows in grouped.values()),
        "prompt_frame_count": len(grouped),
        "hypothesis_count": len(records),
        "frames": frame_summaries,
        "decision_state": "保留全部SAM点提示假设；未选择、聚合或形成候选。",
    }
    (staging_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    del world, projections_raw, visibility_raw, projections, visibility, processed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument(
        "--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")
    )
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list",
        "plan_root",
        "processed_scene_root",
        "dataset_root",
        "config_path",
        "sam_source",
        "sam_checkpoint",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if not args.sam_source.is_dir() or not args.sam_checkpoint.is_file():
        raise SystemExit("SAM 源码目录或权重不存在")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictor = _load_predictor(args)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
        else:
            summary = _export_scene(scene_name, predictor, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['prompt_request_count']} 次提示，"
            f"{summary['hypothesis_count']} 个假设",
            flush=True,
        )

    root_summary = {
        "gt_usage": "不读取GT；不使用类别、语义或native候选。",
        "decision_state": "固定三维核心点提示的全部SAM二维/三维假设；尚未选择或形成候选。",
        "scene_count": len(summaries),
        "planned_track_count": sum(item["planned_track_count"] for item in summaries),
        "prompt_request_count": sum(item["prompt_request_count"] for item in summaries),
        "prompt_frame_count": sum(item["prompt_frame_count"] for item in summaries),
        "hypothesis_count": sum(item["hypothesis_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "core_prompt_observations_summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

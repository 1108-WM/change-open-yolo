#!/usr/bin/env python3
"""导出固定 f30 帧上 YOLOE 分割 mask 的三维观测，不读取 GT。

本工具只把 YOLOE 分割 mask 回投为每帧可见三维点证据，用于后续离线几何上限账本。
输出不是三维候选，不能用于候选融合、评分或 AP 评测。
"""

import argparse
import json
import sys
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


def _mask_to_visible_points(mask, frame_index, projections, visibility, scaling):
    """将一个与原始 RGB 同分辨率的二值 mask 回投为可见点编号。"""
    point_indices = np.flatnonzero(visibility[frame_index])
    if len(point_indices) == 0:
        return np.empty(0, dtype=np.int64)
    coords = projections[frame_index, point_indices].astype(np.float32, copy=False)
    xs = np.round(coords[:, 0] / float(scaling[1])).astype(np.int64)
    ys = np.round(coords[:, 1] / float(scaling[0])).astype(np.int64)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    return point_indices[valid][mask[ys[valid], xs[valid]]].astype(np.int64, copy=False)


def _result_arrays(result, image_shape):
    """取得与检测顺序一致的原始分辨率 mask、类别、分数和框。"""
    boxes = result.boxes
    masks = getattr(result, "masks", None)
    if boxes is None or len(boxes) == 0:
        return (
            np.empty((0, image_shape[0], image_shape[1]), dtype=bool),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
    if masks is None or getattr(masks, "data", None) is None:
        raise ValueError("YOLOE 未返回分割 mask；请确认使用的是 -seg 权重。")
    mask_data = masks.data
    if torch.is_tensor(mask_data):
        mask_data = mask_data.detach().float().cpu()
        if tuple(mask_data.shape[-2:]) != tuple(image_shape):
            mask_data = torch.nn.functional.interpolate(
                mask_data[:, None], size=image_shape, mode="nearest"
            )[:, 0]
        mask_array = mask_data.numpy() > 0.5
    else:
        mask_array = np.asarray(mask_data, dtype=np.float32)
        if tuple(mask_array.shape[-2:]) != tuple(image_shape):
            raise ValueError("非 Tensor 的 YOLOE mask 分辨率与原始 RGB 不一致。")
        mask_array = mask_array > 0.5
    labels = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
    if len(mask_array) != len(labels):
        raise ValueError("YOLOE mask 数量与检测数量不一致。")
    return mask_array, labels, scores, xyxy


def _load_model(args, labels):
    source = str(args.yoloe_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from ultralytics import YOLOE

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("当前 CUDA 不可用；拒绝让 YOLOE 回退到 CPU。")
    model = YOLOE(str(args.checkpoint))
    model.to(args.device)
    model.set_classes(labels, model.get_text_pe(labels))
    return model


def _export_scene(scene_name, model, args):
    from utils import WORLD_2_CAM

    scene_root = args.output_root / scene_name
    if scene_root.exists():
        raise FileExistsError(f"输出场景目录已存在：{scene_root}")
    points_root = scene_root / "points"
    points_root.mkdir(parents=True)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy()
    visibility = visibility.detach().cpu().numpy().astype(bool, copy=False)
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    frame_indices = list(range(0, len(world.color_paths), max(1, args.frame_stride)))[: args.max_frames]
    records = []
    frames = []
    for frame_index in frame_indices:
        image_path = Path(world.color_paths[frame_index])
        result = model.predict(
            str(image_path),
            conf=args.confidence,
            iou=args.nms_iou,
            max_det=args.max_detections,
            imgsz=args.image_size,
            verbose=False,
        )[0]
        image_shape = tuple(int(value) for value in result.orig_shape)
        masks, labels, scores, boxes = _result_arrays(result, image_shape)
        kept = 0
        for detection_index, mask in enumerate(masks):
            points = _mask_to_visible_points(mask, frame_index, projections, visibility, scaling)
            if len(points) < args.min_visible_points:
                continue
            observation_id = len(records)
            points_path = points_root / f"obs{observation_id:06d}_points.npz"
            np.savez_compressed(points_path, point_indices=np.unique(points))
            records.append(
                {
                    "observation_id": observation_id,
                    "scene_name": scene_name,
                    "frame_id": image_path.stem,
                    "frame_index": frame_index,
                    "detection_index": detection_index,
                    "label_id": int(labels[detection_index]),
                    "score": float(scores[detection_index]),
                    "bbox_xyxy": [float(value) for value in boxes[detection_index]],
                    "mask_area": int(mask.sum()),
                    "point_indices_path": str(points_path),
                }
            )
            kept += 1
        frames.append(
            {
                "frame_id": image_path.stem,
                "frame_index": frame_index,
                "detection_count": int(len(labels)),
                "kept_observation_count": kept,
            }
        )
    payload = {
        "scene_name": scene_name,
        "frame_count": len(frames),
        "observation_count": len(records),
        "frames": frames,
        "decision_state": "不读取 GT；不生成三维候选、不融合、不评分、不评测 AP。",
    }
    with (scene_root / "mask_observations.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (scene_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    del world, projections, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _load_existing_scene(output_root, scene_name):
    path = output_root / scene_name / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"{scene_name} 的已有输出不完整：{path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--yoloe_source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--scene_offset", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--nms_iou", type=float, default=0.30)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--min_visible_points", type=int, default=20)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "dataset_root", "config_path", "yoloe_source", "checkpoint", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.skip_existing:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    if not args.yoloe_source.is_dir() or not args.checkpoint.is_file():
        raise SystemExit("YOLOE 源码目录或 checkpoint 不存在。")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    labels = [str(item) for item in args.config["network2d"]["text_prompts"]]
    scenes = _read_scenes(args.scene_list)
    scenes = scenes[max(0, args.scene_offset):]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = _load_model(args, labels)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        if (args.output_root / scene_name).exists() and args.skip_existing:
            summary = _load_existing_scene(args.output_root, scene_name)
            state = "复用"
        else:
            summary = _export_scene(scene_name, model, args)
            state = "完成"
        summaries.append(summary)
        print(f"[场景{state}] {index}/{len(scenes)} {scene_name}: {summary['frame_count']} 帧，{summary['observation_count']} 条 mask 观测", flush=True)
    payload = {
        "gt_usage": "不读取 GT；输出不生成三维候选、融合、评分或 AP 评测。",
        "decision_state": "仅供 GT-only 的 YOLOE 分割 mask 几何上限账本读取。",
        "scene_count": len(summaries),
        "frame_count": int(sum(item["frame_count"] for item in summaries)),
        "observation_count": int(sum(item["observation_count"] for item in summaries)),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "mask_observations_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

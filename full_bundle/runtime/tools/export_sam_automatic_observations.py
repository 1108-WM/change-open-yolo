#!/usr/bin/env python3
"""导出不依赖 YOLO-World 提示的类别无关 SAM 自动 mask 三维观测。

输出只保存自动 mask 回投后的三维点和 SAM 自带质量信息，用于后续离线可行性账本。
不读取 GT、不赋类别、不生成 3D 候选、不参与融合或 AP 评测。
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
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


def select_frame_indices(total_frames, max_frames, frame_stride, selection):
    """从配置已加载的视角中选择帧；默认 first 保持历史 f30 行为。"""
    available = np.arange(0, total_frames, max(1, frame_stride), dtype=np.int64)
    if max_frames is None or len(available) <= max_frames:
        return available.tolist()
    if selection == "first":
        return available[:max_frames].tolist()
    if selection == "uniform":
        positions = np.rint(np.linspace(0, len(available) - 1, max_frames)).astype(np.int64)
        return available[positions].tolist()
    raise ValueError(f"未知帧采样方式：{selection}")


def mask_to_visible_points(mask, frame_index, projections, visibility, scaling_params):
    """将二维自动 mask 映射到该帧实际可见的三维点编号。"""
    point_indices = np.flatnonzero(visibility[frame_index])
    coords = projections[frame_index, point_indices].astype(np.float32)
    xs = np.round(coords[:, 0] / float(scaling_params[1])).astype(np.int64)
    ys = np.round(coords[:, 1] / float(scaling_params[0])).astype(np.int64)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    return point_indices[valid][mask[ys[valid], xs[valid]]].astype(np.int64)


def encode_binary_mask_rle(mask):
    """以 COCO 兼容的列优先游程保存二值 mask，避免保存完整二维数组。"""
    mask = np.asarray(mask, dtype=bool)
    pixels = np.asfortranarray(mask).reshape(-1, order="F").astype(np.uint8)
    changes = np.flatnonzero(np.diff(np.concatenate((np.asarray([0], dtype=np.uint8), pixels, np.asarray([0], dtype=np.uint8)))))
    counts = np.diff(np.concatenate((np.asarray([0], dtype=np.int64), changes, np.asarray([len(pixels)], dtype=np.int64))))
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(value) for value in counts]}


def decode_binary_mask_rle(payload):
    """测试和后续离线关系构建使用的 RLE 还原函数。"""
    height, width = (int(value) for value in payload["size"])
    values, value = [], False
    for count in payload["counts"]:
        values.extend([value] * int(count))
        value = not value
    return np.asarray(values, dtype=bool).reshape((height, width), order="F")


def exact_mask_relation(left, right):
    """保存同帧 mask 的连续关系，不把其解释为 parent/child 决定。"""
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return {
        "intersection_pixel_count": intersection,
        "iou": float(intersection / max(1, union)),
        "left_coverage": float(intersection / max(1, int(left.sum()))),
        "right_coverage": float(intersection / max(1, int(right.sum()))),
    }


def _load_generator(args):
    source = str(args.sam_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("当前 CUDA 不可用；拒绝让 SAM 自动 mask 回退到 CPU。请恢复 GPU 后重试，或显式传入 --device cpu。")
    device = args.device
    model = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    model.to(device=device)
    model.eval()
    return SamAutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_threshold,
        stability_score_thresh=args.stability_score_threshold,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_area,
        output_mode="binary_mask",
    )


def _select_masks(items, max_masks_per_frame):
    """SAM 内部已做 NMS；这里只限制存储预算，不做类别或 GT 选择。"""
    ordered = sorted(
        items,
        key=lambda item: (
            -float(item["predicted_iou"] * item["stability_score"]),
            int(item["area"]),
        ),
    )
    return ordered[:max_masks_per_frame]


def _export_scene(scene_name, generator, args):
    from utils import WORLD_2_CAM

    scene_root = args.output_root / scene_name
    if scene_root.exists():
        raise FileExistsError(f"输出场景目录已存在：{scene_root}")
    points_dir = scene_root / "points"
    points_dir.mkdir(parents=True)

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    scaling_params = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    frame_indices = select_frame_indices(
        len(world.color_paths), args.max_frames, args.frame_stride, args.frame_selection
    )
    records = []
    frames = []
    same_frame_relations = []
    for frame_index in frame_indices:
        image_path = Path(world.color_paths[frame_index])
        image = np.asarray(imageio.imread(image_path))
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"RGB 图像格式异常：{image_path}")
        generated = generator.generate(np.ascontiguousarray(image[:, :, :3]))
        candidates = []
        for item in generated:
            mask = np.asarray(item["segmentation"], dtype=bool)
            area = int(mask.sum())
            if area < args.min_mask_area:
                continue
            points = mask_to_visible_points(mask, frame_index, projections, visibility, scaling_params)
            if len(points) < args.min_visible_points:
                continue
            candidates.append(
                {
                    "area": area,
                    "predicted_iou": float(item["predicted_iou"]),
                    "stability_score": float(item["stability_score"]),
                    "bbox_xywh": [int(value) for value in item["bbox"]],
                    "crop_box_xywh": [int(value) for value in item["crop_box"]],
                    "mask": mask,
                    "points": points,
                }
            )
        kept = _select_masks(candidates, args.max_masks_per_frame)
        frame_id = image_path.stem
        if args.save_exact_same_frame_relations:
            frame_start = len(records)
            for left_index in range(len(kept)):
                for right_index in range(left_index + 1, len(kept)):
                    relation = exact_mask_relation(kept[left_index]["mask"], kept[right_index]["mask"])
                    if relation["intersection_pixel_count"] == 0:
                        continue
                    same_frame_relations.append({
                        "scene_name": scene_name,
                        "frame_id": str(frame_id), "frame_index": int(frame_index),
                        "left_observation_id": frame_start + left_index,
                        "right_observation_id": frame_start + right_index,
                        "left_area": int(kept[left_index]["area"]), "right_area": int(kept[right_index]["area"]),
                        **relation,
                        "relation_state": "真实二维二值 mask 连续关系；不直接决定 parent/child、分割、合并或删除。",
                    })
        for item in kept:
            observation_id = len(records)
            points_path = points_dir / f"obs{observation_id:06d}_points.npz"
            np.savez_compressed(points_path, point_indices=item.pop("points"))
            record = {
                "observation_id": observation_id,
                "scene_name": scene_name,
                "frame_id": str(frame_id),
                "frame_index": int(frame_index),
                "point_indices_path": str(points_path),
                **item,
            }
            mask = record.pop("mask")
            if args.save_mask_rle:
                record["mask_rle"] = encode_binary_mask_rle(mask)
            records.append(record)
        frames.append(
            {
                "frame_id": str(frame_id),
                "frame_index": int(frame_index),
                "generated_mask_count": len(generated),
                "kept_observation_count": len(kept),
            }
        )
    with (scene_root / "automatic_observations.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if args.save_exact_same_frame_relations:
        with (scene_root / "same_frame_mask_relations.jsonl").open("w") as handle:
            for relation in same_frame_relations:
                handle.write(json.dumps(relation, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "scene_name": scene_name,
        "frame_count": len(frames),
        "observation_count": len(records),
        "same_frame_mask_relation_count": len(same_frame_relations),
        "mask_rle_saved": bool(args.save_mask_rle),
        "exact_same_frame_relations_saved": bool(args.save_exact_same_frame_relations),
        "frames": frames,
        "decision_state": "不赋类别、不生成候选、不融合、不评分、不评测。",
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    del world, projections, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--sam_source", type=Path, required=True)
    parser.add_argument("--sam_checkpoint", type=Path, required=True)
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_selection", choices=("first", "uniform"), default="first")
    parser.add_argument("--points_per_side", type=int, default=16)
    parser.add_argument("--pred_iou_threshold", type=float, default=0.88)
    parser.add_argument("--stability_score_threshold", type=float, default=0.95)
    parser.add_argument("--crop_n_layers", type=int, default=0)
    parser.add_argument("--min_mask_area", type=int, default=64)
    parser.add_argument("--min_visible_points", type=int, default=20)
    parser.add_argument("--max_masks_per_frame", type=int, default=40)
    parser.add_argument("--save-mask-rle", action="store_true", help="保存自动 SAM 二值 mask 的压缩 RLE，供真实层级关系审计。")
    parser.add_argument("--save-exact-same-frame-relations", action="store_true", help="保存同帧自动 mask 的连续像素关系；不产生 parent/child 决定。")
    parser.add_argument("--resume", action="store_true", help="跳过已完整导出的场景，继续未完成的独立输出根。")
    args = parser.parse_args()
    for name in ("scene_list", "dataset_root", "config_path", "sam_source", "sam_checkpoint", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    if not args.sam_source.is_dir() or not args.sam_checkpoint.is_file():
        raise SystemExit("SAM 源码目录或权重不存在。")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = _load_generator(args)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing_summary = args.output_root / scene_name / "summary.json"
        if args.resume and existing_summary.is_file():
            summary = json.loads(existing_summary.read_text())
            if not (args.output_root / scene_name / "automatic_observations.jsonl").is_file():
                raise SystemExit(f"场景 {scene_name} 的 resume 输出不完整，拒绝跳过：缺少 automatic_observations.jsonl")
            print(f"[场景跳过] {index}/{len(scenes)} {scene_name}: 已有 {summary.get('observation_count', 0)} 条自动观测", flush=True)
        else:
            summary = _export_scene(scene_name, generator, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['frame_count']} 帧，{summary['observation_count']} 条自动观测",
            flush=True,
        )
    payload = {
        "gt_usage": "不读取 GT；输出仅供后续 GT-only 可行性账本读取。",
        "decision_state": "不赋类别、不生成候选、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "observation_count": sum(item["observation_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_observations_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

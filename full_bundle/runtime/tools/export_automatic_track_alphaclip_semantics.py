#!/usr/bin/env python3
"""为类别无关自动 mask 轨迹导出多视角 Alpha-CLIP 语义，不读取 GT。"""

import argparse
import gc
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
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from export_multiview_object_clip_features import (
    _aggregate_rows,
    _bbox_from_visible_points,
    _encode_alpha_clip_images,
    _load_alpha_clip,
    _make_crop_alpha_mask,
    _make_crop_image,
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _select_track_views(
    points,
    frame_ids,
    frame_lookup,
    projections,
    visibility,
    scaling,
    image_shape,
    top_views,
    min_visible_points,
    crop_padding_ratio,
):
    """只在形成该轨迹的自动 mask 帧中选择可见点最多的视角。"""
    image_h, image_w = image_shape
    selected = []
    for frame_id in frame_ids:
        frame_index = frame_lookup.get(str(frame_id))
        if frame_index is None:
            continue
        visible_points = points[visibility[frame_index, points]]
        if len(visible_points) < min_visible_points:
            continue
        coords = projections[frame_index, visible_points].astype(np.int64)
        bbox = _bbox_from_visible_points(
            coords,
            scaling,
            (image_h, image_w),
            padding_ratio=crop_padding_ratio,
        )
        if bbox is None:
            continue
        selected.append({
            "frame_id": str(frame_id),
            "frame_index": int(frame_index),
            "visible_points": int(len(visible_points)),
            "visible_point_ids": visible_points,
            "bbox_xyxy": [int(value) for value in bbox],
        })
    return sorted(selected, key=lambda item: (-item["visible_points"], item["frame_index"]))[:top_views]


def _flush(alpha_state, device, images, masks, pending):
    if not images:
        return
    encoded = _encode_alpha_clip_images(alpha_state, images, masks, device)
    for record, view, probs, logits in zip(pending["records"], pending["views"], encoded["probs"], encoded["logits"]):
        view["clip_logits"] = [float(value) for value in logits.tolist()]
        view["clip_top_class_id"] = int(np.argmax(probs))
        record["_probs"].append(probs)
        record["_logits"].append(logits)
    images.clear()
    masks.clear()
    pending["records"].clear()
    pending["views"].clear()


def _finalize_record(record, labels):
    probs = _aggregate_rows(record.pop("_probs"), "mean", probability=True)
    logits = _aggregate_rows(record.pop("_logits"), "mean", probability=False)
    if probs is None or logits is None:
        record.update({
            "alphaclip_class_index": -1,
            "alphaclip_class_name": "无语义结果",
            "alphaclip_top_probability": 0.0,
            "alphaclip_logit_margin": 0.0,
            "clip_logits": [],
        })
        return record
    order = np.argsort(-logits)
    top = int(order[0])
    second = float(logits[order[1]]) if len(order) > 1 else 0.0
    record.update({
        "alphaclip_class_index": top,
        "alphaclip_class_name": labels[top],
        "alphaclip_top_probability": float(probs[top]),
        "alphaclip_logit_margin": float(logits[top] - second),
        "clip_logits": [float(value) for value in logits.tolist()],
    })
    return record


def _scene_records(scene_name, args, alpha_state, labels, device):
    from utils import WORLD_2_CAM

    track_path = args.track_root / scene_name / "automatic_tracks.json"
    if not track_path.is_file():
        track_path = args.track_root / scene_name / "d2b_tracks_filtered" / scene_name / "automatic_tracks.json"
    if not track_path.is_file():
        raise FileNotFoundError(f"{scene_name}: automatic_tracks.json not found under {args.track_root}")
    tracks = json.loads(track_path.read_text())["tracks"]
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    records = []
    pending_images, pending_masks = [], []
    pending = {"records": [], "views": []}
    for track in tracks:
        points = np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        selected = _select_track_views(
            points, track["frame_ids"], frame_lookup, projections, visibility, scaling,
            world.image_resolution, args.top_views, args.min_visible_points, args.crop_padding_ratio,
        )
        record = {
            "scene_name": scene_name,
            "track_id": int(track["track_id"]),
            "support_view_count": int(track["support_view_count"]),
            "point_count": int(track["point_count"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "mean_edge_score": float(track["mean_edge_score"]),
            "crop_contract": {
                "crop_padding_ratio": float(args.crop_padding_ratio),
                "alpha_mask_dilation_iters": int(args.alpha_mask_dilation_iters),
                "mask_background": False,
            },
            "views": [],
            "_probs": [],
            "_logits": [],
        }
        for view in selected:
            image = np.asarray(imageio.imread(world.color_paths[view["frame_index"]]))
            x1, y1, x2, y2 = view["bbox_xyxy"]
            coords = projections[view["frame_index"], view["visible_point_ids"]]
            coords_color = np.stack([
                np.round(coords[:, 0] / scaling[1]).astype(np.int64),
                np.round(coords[:, 1] / scaling[0]).astype(np.int64),
            ], axis=1)
            crop = _make_crop_image(image, (x1, y1, x2, y2), coords=coords_color, mask_background=False)
            alpha_mask = _make_crop_alpha_mask(
                image.shape[:2], (x1, y1, x2, y2), coords=coords_color, dilation_iters=args.alpha_mask_dilation_iters,
            )
            view_record = {
                "frame_id": view["frame_id"],
                "visible_points": view["visible_points"],
                "bbox_xyxy": view["bbox_xyxy"],
            }
            record["views"].append(view_record)
            pending_images.append(crop)
            pending_masks.append(alpha_mask)
            pending["records"].append(record)
            pending["views"].append(view_record)
            if len(pending_images) >= args.batch_size:
                _flush(alpha_state, device, pending_images, pending_masks, pending)
        records.append(record)
    _flush(alpha_state, device, pending_images, pending_masks, pending)
    records = [_finalize_record(record, labels) for record in records]
    del world, projections, visibility
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--alpha_clip_source", type=Path, default=Path("_external/AlphaCLIP/AlphaCLIP-main"))
    parser.add_argument("--alpha_clip_base_model", type=Path, default=Path("pretrained/alpha_clip/checkpoints/ViT-L-14.pt"))
    parser.add_argument("--alpha_clip_checkpoint", type=Path, default=Path("pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth"))
    parser.add_argument("--prompt_template", default="a photo of a {label}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_views", type=int, default=3)
    parser.add_argument("--min_visible_points", type=int, default=20)
    parser.add_argument("--crop_padding_ratio", type=float, default=0.15)
    parser.add_argument("--alpha_mask_dilation_iters", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    if args.crop_padding_ratio < 0:
        raise SystemExit("--crop_padding_ratio must be non-negative")
    for name in (
        "scene_list", "track_root", "dataset_root", "config_path", "alpha_clip_source", "alpha_clip_base_model",
        "alpha_clip_checkpoint", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("当前 CUDA 不可用；拒绝让 Alpha-CLIP 回退到 CPU。")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    labels = list(args.config["network2d"]["text_prompts"])
    alpha_state = _load_alpha_clip(args, labels, args.device)
    args.output_root.mkdir(parents=True)
    all_records = []
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("--max_scenes must be positive")
        scenes = scenes[: args.max_scenes]
    for index, scene_name in enumerate(scenes, start=1):
        records = _scene_records(scene_name, args, alpha_state, labels, args.device)
        all_records.extend(records)
        root = args.output_root / scene_name
        root.mkdir()
        (root / "automatic_track_alphaclip_semantics.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条轨迹", flush=True)
    payload = {
        "gt_usage": "不读取 GT；不写候选、不融合、不评分、不评测。",
        "track_count": len(all_records),
        "with_semantics_count": sum(record["alphaclip_class_index"] >= 0 for record in all_records),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_track_alphaclip_semantic_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

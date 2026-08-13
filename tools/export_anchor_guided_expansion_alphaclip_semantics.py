#!/usr/bin/env python3
"""为锚点引导扩展区域导出多视角 Alpha-CLIP 语义，不读取 GT。"""

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


def _select_views(points, selected_frames, frame_lookup, projections, visibility, scaling, image_shape, top_views, min_visible_points):
    """只在锚点扩展实际得到二维支持的帧中选择裁剪。"""
    image_h, image_w = image_shape
    views = []
    for frame_id in selected_frames:
        frame_index = frame_lookup.get(str(frame_id))
        if frame_index is None:
            continue
        visible_points = points[visibility[frame_index, points]]
        if len(visible_points) < min_visible_points:
            continue
        coords = projections[frame_index, visible_points].astype(np.int64)
        bbox = _bbox_from_visible_points(coords, scaling, (image_h, image_w))
        if bbox is not None:
            views.append({"frame_id": str(frame_id), "frame_index": int(frame_index), "visible_points": visible_points, "bbox_xyxy": bbox})
    return sorted(views, key=lambda item: (-len(item["visible_points"]), item["frame_index"]))[:top_views]


def _flush(alpha_state, device, images, masks, pending_records, pending_views):
    if not images:
        return
    encoded = _encode_alpha_clip_images(alpha_state, images, masks, device)
    for record, view, probs, logits in zip(pending_records, pending_views, encoded["probs"], encoded["logits"]):
        view["clip_logits"] = [float(value) for value in logits.tolist()]
        record["_probs"].append(probs)
        record["_logits"].append(logits)
    images.clear()
    masks.clear()
    pending_records.clear()
    pending_views.clear()


def _finalize(record, labels):
    probabilities = _aggregate_rows(record.pop("_probs"), "mean", probability=True)
    logits = _aggregate_rows(record.pop("_logits"), "mean", probability=False)
    if probabilities is None or logits is None:
        record.update({"alphaclip_class_index": -1, "alphaclip_class_name": "无语义结果", "alphaclip_logit_margin": 0.0, "clip_logits": []})
        return record
    order = np.argsort(-logits)
    top = int(order[0])
    second = float(logits[order[1]]) if len(order) > 1 else 0.0
    record.update({
        "alphaclip_class_index": top,
        "alphaclip_class_name": labels[top],
        "alphaclip_logit_margin": float(logits[top] - second),
        "clip_logits": [float(value) for value in logits.tolist()],
    })
    return record


def _scene_records(scene_name, args, alpha_state, labels):
    from utils import WORLD_2_CAM

    expansions = json.loads((args.expansion_root / scene_name / "anchor_guided_expansions.json").read_text())
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    scaling = (world.depth_resolution[0] / world.image_resolution[0], world.depth_resolution[1] / world.image_resolution[1])
    records, images, masks, pending_records, pending_views = [], [], [], [], []
    for expansion in expansions:
        if not expansion["meets_minimum_support"]:
            continue
        points = np.asarray(np.load(expansion["expanded_points_path"])["point_indices"], dtype=np.int64)
        selected_frames = [item["frame_id"] for item in expansion["selected_observations"]]
        views = _select_views(points, selected_frames, frame_lookup, projections, visibility, scaling, world.image_resolution, args.top_views, args.min_visible_points)
        record = {
            "scene_name": scene_name,
            "track_id": int(expansion["track_id"]),
            "expanded_points_path": str(expansion["expanded_points_path"]),
            "expanded_point_count": int(expansion["expanded_point_count"]),
            "added_point_count": int(expansion["added_point_count"]),
            "selected_frame_count": int(expansion["selected_frame_count"]),
            "views": [], "_probs": [], "_logits": [],
        }
        for view in views:
            image = np.asarray(imageio.imread(world.color_paths[view["frame_index"]]))
            x1, y1, x2, y2 = view["bbox_xyxy"]
            coords = projections[view["frame_index"], view["visible_points"]]
            coords_color = np.stack([
                np.round(coords[:, 0] / scaling[1]).astype(np.int64),
                np.round(coords[:, 1] / scaling[0]).astype(np.int64),
            ], axis=1)
            crop = _make_crop_image(image, (x1, y1, x2, y2), coords=coords_color, mask_background=False)
            alpha_mask = _make_crop_alpha_mask(image.shape[:2], (x1, y1, x2, y2), coords=coords_color, dilation_iters=args.alpha_mask_dilation_iters)
            view_record = {"frame_id": view["frame_id"], "visible_points": int(len(view["visible_points"])), "bbox_xyxy": [int(value) for value in view["bbox_xyxy"]]}
            record["views"].append(view_record)
            images.append(crop)
            masks.append(alpha_mask)
            pending_records.append(record)
            pending_views.append(view_record)
            if len(images) >= args.batch_size:
                _flush(alpha_state, args.device, images, masks, pending_records, pending_views)
        records.append(record)
    _flush(alpha_state, args.device, images, masks, pending_records, pending_views)
    records = [_finalize(record, labels) for record in records]
    del world, projections, visibility
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--expansion_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--alpha_clip_source", type=Path, default=Path("_external/AlphaCLIP/AlphaCLIP-main"))
    parser.add_argument("--alpha_clip_base_model", type=Path, default=Path("pretrained/alpha_clip/checkpoints/ViT-L-14.pt"))
    parser.add_argument("--alpha_clip_checkpoint", type=Path, default=Path("pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth"))
    parser.add_argument("--prompt_template", default="a photo of a {label}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_views", type=int, default=3)
    parser.add_argument("--min_visible_points", type=int, default=20)
    parser.add_argument("--alpha_mask_dilation_iters", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "expansion_root", "dataset_root", "config_path", "alpha_clip_source", "alpha_clip_base_model", "alpha_clip_checkpoint", "output_root"):
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
    for index, scene_name in enumerate(scenes, start=1):
        records = _scene_records(scene_name, args, alpha_state, labels)
        all_records.extend(records)
        root = args.output_root / scene_name
        root.mkdir()
        (root / "anchor_guided_expansion_alphaclip_semantics.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条扩展区域", flush=True)
    payload = {
        "gt_usage": "不读取 GT；不生成候选、不融合、不评分、不评测。",
        "expansion_count": len(all_records),
        "with_semantics_count": sum(record["alphaclip_class_index"] >= 0 for record in all_records),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "anchor_guided_expansion_alphaclip_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

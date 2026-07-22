#!/usr/bin/env python3
"""Export GT-free Any3DIS-style SAM2 tracks from IBSp superpoint prompts.

Run this script with the dedicated ``sam2`` Conda environment.  It deliberately
does not import the Open-YOLO evaluation stack: its only inputs are ScanNet RGB-D
frames/poses and the inference-time superpoint ids in processed-array column 9.
The output is an intermediate track artifact, not an AP submission.
"""

import argparse
import json
import os
import os.path as osp
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _natural_key(value):
    return (not str(value).isdigit(), int(value) if str(value).isdigit() else str(value))


def _read_scenes(scene_names, scene_split, max_scenes):
    if scene_names:
        scenes = [item.strip() for item in scene_names.split(",") if item.strip()]
    elif scene_split:
        scenes = [line.strip() for line in Path(scene_split).read_text().splitlines() if line.strip()]
    else:
        raise ValueError("Provide --scene_names or --scene_split.")
    return scenes[:max_scenes] if max_scenes is not None else scenes


def _scene_id(scene_name):
    return scene_name.replace("scene", "")


def _load_scene_arrays(scene_dir, superpoint_root, scene_name):
    array_path = Path(superpoint_root) / scene_name / f"{_scene_id(scene_name)}.npy"
    if not array_path.is_file():
        raise FileNotFoundError(f"Missing IBSp processed scene: {array_path}")
    array = np.load(array_path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] < 10:
        raise ValueError(f"Expected processed scene with superpoint ids in column 9: {array_path}")
    points = np.asarray(array[:, :3], dtype=np.float32)
    superpoints = np.asarray(array[:, 9], dtype=np.int64)
    if np.any(superpoints < 0):
        raise ValueError(f"Negative superpoint ids are unsupported: {array_path}")
    intrinsics = np.loadtxt(Path(scene_dir) / "intrinsics.txt", dtype=np.float32)
    if intrinsics.shape != (4, 4):
        raise ValueError(f"Expected 4x4 intrinsics.txt in {scene_dir}")
    return points, superpoints, intrinsics[:3, :3], array_path


def _dense_frame_ids(scene_dir, max_frames):
    color_dir = Path(scene_dir) / "color"
    ids = sorted((path.stem for path in color_dir.glob("*.jpg")), key=_natural_key)
    if not ids:
        raise FileNotFoundError(f"No RGB JPEG frames in {color_dir}")
    return ids[:max_frames] if max_frames is not None else ids


def _sample_frame_ids(scene_dir, stride, max_frames):
    ids = _dense_frame_ids(scene_dir, None)[:: max(1, int(stride))]
    return ids[:max_frames] if max_frames is not None else ids


def _make_video_dir(output_scene_dir, scene_dir, frame_ids, overwrite):
    """SAM2 requires integer-named JPEGs; retain source ids in metadata instead."""
    video_dir = output_scene_dir / "sam2_frames"
    if video_dir.exists() and overwrite:
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_id in enumerate(frame_ids):
        target = video_dir / f"{index:05d}.jpg"
        source = Path(scene_dir) / "color" / f"{frame_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.exists():
            os.symlink(source.resolve(), target)
    return video_dir


def _project_visible_points(points, pose, intrinsics, image_shape, depth_image, depth_tolerance):
    """Project mesh points to RGB, retain front-most points consistent with RGB-D depth."""
    world_to_camera = np.linalg.inv(pose).astype(np.float32)
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera_points[:, 2]
    valid = np.isfinite(z) & (z > 1e-4)
    pixels = camera_points @ intrinsics.T
    x = pixels[:, 0] / np.maximum(pixels[:, 2], 1e-6)
    y = pixels[:, 1] / np.maximum(pixels[:, 2], 1e-6)
    height, width = image_shape
    px = np.rint(x).astype(np.int64)
    py = np.rint(y).astype(np.int64)
    valid &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
    candidate = np.flatnonzero(valid)
    if not len(candidate):
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int32)

    flat = py[candidate] * width + px[candidate]
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, flat, z[candidate])
    keep = z[candidate] <= z_buffer[flat] + float(depth_tolerance)
    candidate = candidate[keep]

    if depth_image is not None and len(candidate):
        depth_height, depth_width = depth_image.shape[:2]
        depth_x = np.clip(np.rint(px[candidate] * depth_width / width).astype(np.int64), 0, depth_width - 1)
        depth_y = np.clip(np.rint(py[candidate] * depth_height / height).astype(np.int64), 0, depth_height - 1)
        observed_depth = depth_image[depth_y, depth_x].astype(np.float32) / 1000.0
        depth_ok = (observed_depth > 1e-4) & (np.abs(observed_depth - z[candidate]) <= float(depth_tolerance))
        candidate = candidate[depth_ok]
    return candidate, np.column_stack((px[candidate], py[candidate])).astype(np.int32)


def _load_visibility(scene_dir, frame_ids, points, intrinsics, depth_tolerance):
    first_image = Image.open(Path(scene_dir) / "color" / f"{frame_ids[0]}.jpg")
    width, height = first_image.size
    visible = []
    pixels = []
    for frame_id in tqdm(frame_ids, desc="project RGB-D visibility", leave=False):
        pose_path = Path(scene_dir) / "poses" / f"{frame_id}.txt"
        depth_path = Path(scene_dir) / "depth" / f"{frame_id}.png"
        if not pose_path.is_file() or not depth_path.is_file():
            visible.append(np.empty(0, dtype=np.int64))
            pixels.append(np.empty((0, 2), dtype=np.int32))
            continue
        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            visible.append(np.empty(0, dtype=np.int64))
            pixels.append(np.empty((0, 2), dtype=np.int32))
            continue
        depth = imageio.imread(depth_path)
        indices, projected = _project_visible_points(
            points, pose, intrinsics, (height, width), depth, depth_tolerance
        )
        visible.append(indices)
        pixels.append(projected)
    return visible, pixels, (height, width)


def _superpoint_stats(points, superpoints):
    count = int(superpoints.max(initial=-1)) + 1
    sizes = np.bincount(superpoints, minlength=count).astype(np.int64)
    centroids = np.zeros((count, 3), dtype=np.float32)
    np.add.at(centroids, superpoints, points)
    centroids /= np.maximum(sizes[:, None], 1)
    return sizes, centroids


def _visibility_by_superpoint(visible_points, superpoints, count):
    distribution = np.zeros((len(visible_points), count), dtype=np.int32)
    for frame_index, indices in enumerate(visible_points):
        if len(indices):
            distribution[frame_index] = np.bincount(superpoints[indices], minlength=count)
    return distribution


def _farthest_sample(points, candidates, limit):
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) <= limit:
        return candidates
    chosen = [int(candidates[np.argmin(np.linalg.norm(points[candidates] - points[candidates].mean(0), axis=1))])]
    nearest = np.linalg.norm(points[candidates] - points[chosen[0]], axis=1)
    for _ in range(1, int(limit)):
        next_index = int(np.argmax(nearest))
        chosen.append(int(candidates[next_index]))
        nearest = np.minimum(nearest, np.linalg.norm(points[candidates] - points[chosen[-1]], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def _select_pivot_and_prompts(seed_id, superpoints, centroids, sizes, distribution, visible_points, pixels, args):
    visible_ratio = distribution / np.maximum(sizes[None, :], 1)
    distances = np.linalg.norm(centroids - centroids[int(seed_id)], axis=1)
    neighbors = np.argsort(distances)[: min(int(args.neighbor_superpoints), len(centroids))]
    weights = 1.0 / np.maximum(distances[neighbors], 0.05)
    frame_score = (visible_ratio[:, neighbors] * weights[None, :]).sum(axis=1)
    frame_score[distribution[:, int(seed_id)] < int(args.min_prompt_points)] = -np.inf
    pivot = int(np.argmax(frame_score))
    if not np.isfinite(frame_score[pivot]):
        return None
    visible = visible_points[pivot]
    seed_locations = np.flatnonzero(superpoints[visible] == int(seed_id))
    if len(seed_locations) < int(args.min_prompt_points):
        return None
    seed_pixels = pixels[pivot][seed_locations]
    chosen = [int(np.argmin(np.linalg.norm(seed_pixels - seed_pixels.mean(axis=0), axis=1)))]
    nearest = np.linalg.norm(seed_pixels - seed_pixels[chosen[0]], axis=1)
    for _ in range(1, min(int(args.prompt_points), len(seed_pixels))):
        next_index = int(np.argmax(nearest))
        chosen.append(next_index)
        nearest = np.minimum(nearest, np.linalg.norm(seed_pixels - seed_pixels[next_index], axis=1))
    return pivot, seed_pixels[np.asarray(chosen, dtype=np.int64)], float(frame_score[pivot])


def _load_predictors(args):
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    video_predictor = build_sam2_video_predictor(
        args.sam2_config,
        args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=False,
    )
    image_predictor = None
    if args.initialization_mode == "image_mask" or int(args.reobservation_stride) > 0:
        image_predictor = SAM2ImagePredictor(
            build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device, apply_postprocessing=False)
        )
    return video_predictor, image_predictor


def _initial_mask(image_predictor, image_path, prompt_points, args):
    image = np.array(Image.open(image_path).convert("RGB"), copy=True)
    image_predictor.set_image(image)
    masks, scores, _ = image_predictor.predict(
        point_coords=prompt_points.astype(np.float32),
        point_labels=np.ones(len(prompt_points), dtype=np.int32),
        multimask_output=bool(args.image_multimask),
    )
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)
    if int(mask.sum()) < int(args.min_mask_area):
        return None, float(scores[best])
    return mask, float(scores[best])


def _seed_prompt_at_frame(seed_id, superpoints, visible_points, pixels, frame_index):
    visible = visible_points[frame_index]
    locations = np.flatnonzero(superpoints[visible] == int(seed_id))
    if not len(locations):
        return None
    seed_pixels = pixels[frame_index][locations]
    center = seed_pixels.mean(axis=0)
    return seed_pixels[int(np.argmin(np.linalg.norm(seed_pixels - center, axis=1)))][None, :]


def _mask_iou(left, right):
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _independent_reobservations(
    image_predictor, video_dir, masks, pivot, seed_id, superpoints,
    distribution, visible_points, pixels, args,
):
    """以独立图像预测确认传播 mask；只记录证据，不直接篡改原始轨迹。"""
    if image_predictor is None or int(args.reobservation_stride) <= 0:
        return []
    checked = {int(pivot)}
    stride = int(args.reobservation_stride)
    checked.update(range(0, len(masks), stride))
    records = []
    for frame_index in sorted(checked):
        if distribution[frame_index, int(seed_id)] < int(args.min_prompt_points):
            continue
        point = _seed_prompt_at_frame(seed_id, superpoints, visible_points, pixels, frame_index)
        if point is None:
            continue
        observed_mask, observed_score = _initial_mask(
            image_predictor, Path(video_dir) / f"{frame_index:05d}.jpg", point, args
        )
        if observed_mask is None:
            continue
        propagated = masks[frame_index]
        agreement = _mask_iou(propagated, observed_mask)
        records.append(
            {
                "frame_index": int(frame_index),
                "agreement_iou": agreement,
                "image_mask_score": float(observed_score),
                "accepted": bool(agreement >= float(args.reobservation_min_iou)),
            }
        )
    return records


def _reappearance_prompts(seed_id, pivot, superpoints, distribution, visible_points, pixels, args):
    """Add one 3D-aware point when a seed becomes visible after SAM2's memory window."""
    visible = distribution[:, int(seed_id)] >= int(args.min_prompt_points)
    runs = []
    start = None
    for index, value in enumerate(visible.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    prompts = []
    for run_start, run_end in runs:
        if run_start <= int(pivot) <= run_end:
            continue
        if run_start > int(pivot):
            prompt_frame = run_start
            gap = run_start - int(pivot)
            earlier = [end for _, end in runs if end < run_start]
            if earlier:
                gap = run_start - max(earlier)
        else:
            prompt_frame = run_end
            gap = int(pivot) - run_end
            later = [start for start, _ in runs if start > run_end]
            if later:
                gap = min(later) - run_end
        if gap < int(args.reappearance_memory_window):
            continue
        point = _seed_prompt_at_frame(seed_id, superpoints, visible_points, pixels, prompt_frame)
        if point is not None:
            prompts.append((int(prompt_frame), point))
    return prompts


def _write_track(path, masks, metadata):
    flattened = masks.reshape(masks.shape[0], -1)
    np.savez_compressed(
        path,
        packed_masks=np.packbits(flattened, axis=1),
        mask_shape=np.asarray(masks.shape, dtype=np.int32),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def _track_seed(
    predictor, image_predictor, video_dir, pivot, prompt_points, seed_id, superpoints,
    distribution, visible_points, pixels, frame_count, image_shape, args,
):
    state = predictor.init_state(
        video_path=str(video_dir), offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    initial_score = None
    if image_predictor is not None:
        initial_mask, initial_score = _initial_mask(
            image_predictor, Path(video_dir) / f"{int(pivot):05d}.jpg", prompt_points, args
        )
    else:
        initial_mask = None
    if initial_mask is None:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(pivot),
            obj_id=1,
            points=prompt_points.astype(np.float32),
            labels=np.ones(len(prompt_points), dtype=np.int32),
        )
        initialization = "points"
    else:
        predictor.add_new_mask(inference_state=state, frame_idx=int(pivot), obj_id=1, mask=initial_mask)
        initialization = "image_mask"
    reappearance = _reappearance_prompts(
        seed_id, pivot, superpoints, distribution, visible_points, pixels, args
    )
    for frame_index, point in reappearance:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=1,
            points=point.astype(np.float32),
            labels=np.ones(len(point), dtype=np.int32),
        )
    outputs = {}
    for reverse in (False, True):
        for frame_index, _, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=int(pivot), reverse=reverse
        ):
            outputs[int(frame_index)] = (mask_logits[0] > 0).squeeze(0).cpu().numpy().astype(bool)
    predictor.reset_state(state)
    del state
    torch.cuda.empty_cache()
    masks = np.zeros((frame_count, image_shape[0], image_shape[1]), dtype=bool)
    for frame_index, mask in outputs.items():
        masks[frame_index] = mask
    reobservations = _independent_reobservations(
        image_predictor, video_dir, masks, pivot, seed_id, superpoints,
        distribution, visible_points, pixels, args,
    )
    return masks, initialization, initial_score, [frame_index for frame_index, _ in reappearance], reobservations


def _read_seed_ids(path):
    if path is None:
        return None
    payload = Path(path).read_text().strip()
    if not payload:
        return np.empty(0, dtype=np.int64)
    if payload.startswith("["):
        return np.asarray(json.loads(payload), dtype=np.int64)
    return np.asarray([int(line.strip()) for line in payload.splitlines() if line.strip()], dtype=np.int64)


def export_scene(scene_name, args, predictor, image_predictor):
    scene_dir = Path(args.dataset_root) / scene_name
    output_scene = Path(args.output_root) / scene_name
    if output_scene.exists() and args.overwrite:
        shutil.rmtree(output_scene)
    if output_scene.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_scene}; use --overwrite")
    output_scene.mkdir(parents=True)
    frame_ids = _sample_frame_ids(scene_dir, args.frame_stride, args.max_frames)
    points, superpoints, intrinsics, source_array = _load_scene_arrays(scene_dir, args.superpoint_root, scene_name)
    visible_points, pixels, image_shape = _load_visibility(
        scene_dir, frame_ids, points, intrinsics, args.depth_tolerance
    )
    sizes, centroids = _superpoint_stats(points, superpoints)
    distribution = _visibility_by_superpoint(visible_points, superpoints, len(sizes))
    reliable = np.flatnonzero(
        (sizes >= int(args.min_superpoint_points))
        & ((distribution >= int(args.min_prompt_points)).sum(axis=0) >= int(args.min_visible_frames))
    )
    seed_path = args.seed_ids_file
    if args.seed_ids_root:
        scene_seed_path = Path(args.seed_ids_root) / f"{scene_name}.txt"
        if not scene_seed_path.is_file():
            raise FileNotFoundError(f"Missing scene seed file: {scene_seed_path}")
        seed_path = scene_seed_path
    seed_override = _read_seed_ids(seed_path)
    if seed_override is not None:
        seeds = np.intersect1d(reliable, seed_override, assume_unique=False)
        if not len(seeds):
            raise ValueError(f"No requested seed ids are reliable for {scene_name}: {seed_path}")
        seeds = _farthest_sample(centroids, seeds, args.max_tracks)
    else:
        seeds = _farthest_sample(centroids, reliable, args.max_tracks)
    video_dir = _make_video_dir(output_scene, scene_dir, frame_ids, args.overwrite)
    track_dir = output_scene / "tracks"
    track_dir.mkdir()
    records = []
    for track_index, seed_id in enumerate(tqdm(seeds, desc=f"SAM2 {scene_name}")):
        selection = _select_pivot_and_prompts(
            seed_id, superpoints, centroids, sizes, distribution, visible_points, pixels, args
        )
        if selection is None:
            continue
        pivot, prompt_points, pivot_score = selection
        masks, initialization, initial_score, reappearance_frames, reobservations = _track_seed(
            predictor, image_predictor, video_dir, pivot, prompt_points, seed_id, superpoints,
            distribution, visible_points, pixels, len(frame_ids), image_shape, args,
        )
        nonempty = np.flatnonzero(masks.reshape(len(frame_ids), -1).sum(axis=1) >= int(args.min_mask_area))
        if len(nonempty) < int(args.min_track_frames):
            continue
        track_path = track_dir / f"track{track_index:04d}.npz"
        record = {
            "track_id": int(track_index),
            "seed_superpoint_id": int(seed_id),
            "pivot_frame_index": int(pivot),
            "pivot_frame_id": str(frame_ids[pivot]),
            "pivot_score": float(pivot_score),
            "prompt_points_xy": prompt_points.astype(int).tolist(),
            "initialization": initialization,
            "initial_mask_score": initial_score,
            "reappearance_prompt_frame_indices": reappearance_frames,
            "reobservations": reobservations,
            "reobservation_checked_frame_indices": [item["frame_index"] for item in reobservations],
            "reobservation_rejected_frame_indices": [
                item["frame_index"] for item in reobservations if not item["accepted"]
            ],
            "nonempty_frame_indices": nonempty.astype(int).tolist(),
            "mask_path": str(track_path),
        }
        _write_track(track_path, masks, record)
        records.append(record)
    metadata = {
        "scene_name": scene_name,
        "frame_ids": [str(frame_id) for frame_id in frame_ids],
        "image_shape": list(image_shape),
        "source_superpoint_array": str(source_array),
        "superpoint_column": 9,
        "reliable_superpoint_count": int(len(reliable)),
        "reliable_superpoint_ids": reliable.astype(int).tolist(),
        "attempted_seed_count": int(len(seeds)),
        "attempted_seed_superpoint_ids": seeds.astype(int).tolist(),
        "saved_track_count": int(len(records)),
        "params": {key: value for key, value in vars(args).items() if key != "scene_names"},
    }
    (output_scene / "tracks.jsonl").write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    (output_scene / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export GT-free Any3DIS-style SAM2 superpoint tracks.")
    parser.add_argument("--dataset_root", default="data/scannet200")
    parser.add_argument("--superpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--max_tracks", type=int, default=8)
    parser.add_argument("--seed_ids_file", default=None, help="Optional JSON array or newline-delimited reliable SP ids for a later sampling round.")
    parser.add_argument("--seed_ids_root", default=None, help="Directory with <scene>.txt files for a multi-scene later sampling round.")
    parser.add_argument("--prompt_points", type=int, default=3)
    parser.add_argument("--min_superpoint_points", type=int, default=40)
    parser.add_argument("--min_visible_frames", type=int, default=3)
    parser.add_argument("--min_prompt_points", type=int, default=3)
    parser.add_argument("--neighbor_superpoints", type=int, default=32)
    parser.add_argument("--depth_tolerance", type=float, default=0.10)
    parser.add_argument("--min_mask_area", type=int, default=64)
    parser.add_argument("--min_track_frames", type=int, default=2)
    parser.add_argument("--initialization_mode", choices=("image_mask", "points"), default="image_mask")
    parser.add_argument("--image_multimask", action="store_true")
    parser.add_argument("--reappearance_memory_window", type=int, default=7)
    parser.add_argument("--reobservation_stride", type=int, default=0, help="大于 0 时每隔该帧数使用独立图像预测器确认传播 mask。")
    parser.add_argument("--reobservation_min_iou", type=float, default=0.30, help="独立重观测与传播 mask 的最小一致 IoU。")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seed_ids_file and args.seed_ids_root:
        raise ValueError("Use only one of --seed_ids_file and --seed_ids_root.")
    if args.reobservation_stride < 0:
        raise ValueError("--reobservation_stride 必须为非负整数。")
    if not 0.0 <= args.reobservation_min_iou <= 1.0:
        raise ValueError("--reobservation_min_iou 必须在 0 到 1 之间。")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM2 tracking but is unavailable.")
    scenes = _read_scenes(args.scene_names, args.scene_split, args.max_scenes)
    predictor, image_predictor = _load_predictors(args)
    summaries = [export_scene(scene, args, predictor, image_predictor) for scene in scenes]
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "sam2_track_export_summary.json").write_text(
        json.dumps({"scenes": summaries, "params": vars(args)}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

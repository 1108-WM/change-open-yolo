#!/usr/bin/env python3
"""Reproject frozen automatic-SAM RLE masks with MV3DIS relative depth.

The output is a parallel, no-GT cache.  It never overwrites the source
hierarchy-safe observations or Open-YOLO's absolute-0.05m visibility.  For
each used frame it stores points satisfying the paper's strict
``|zc-d| < 0.05*d`` condition and their continuous depth weights.  Each source
RLE is then reprojected to a new point/weight file using those frame results.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_details_same_frame_hierarchy_preprocessor import (
    decode_binary_mask_rle,
)


RELATIVE_DEPTH_ALPHA = 0.05
PROJECTION_CONTRACT = "mv3dis_relative_depth_rle_reference"


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def relative_depth_weights(projected_depth, measured_depth, in_image, alpha=RELATIVE_DEPTH_ALPHA):
    """Return paper visibility and Eq. (2) weights with strict inequality."""
    projected_depth = np.asarray(projected_depth, dtype=np.float64)
    measured_depth = np.asarray(measured_depth, dtype=np.float64)
    in_image = np.asarray(in_image, dtype=bool)
    if projected_depth.shape != measured_depth.shape or projected_depth.shape != in_image.shape:
        raise ValueError("depth and in-image arrays must have identical shapes")
    tolerance = float(alpha) * measured_depth
    error = np.abs(projected_depth - measured_depth)
    visible = (
        in_image
        & np.isfinite(projected_depth)
        & np.isfinite(measured_depth)
        & (projected_depth > 0.0)
        & (measured_depth > 0.0)
        & (error < tolerance)
    )
    weights = np.zeros(projected_depth.shape, dtype=np.float32)
    weights[visible] = (1.0 - error[visible] / tolerance[visible]).astype(np.float32)
    if np.any(weights[visible] <= 0.0) or np.any(weights > 1.0):
        raise ValueError("relative depth weights are outside (0, 1]")
    return visible, weights


def project_relative_depth_frame(points_h, camera_to_world, intrinsic, depth_map, alpha=RELATIVE_DEPTH_ALPHA):
    """Project one frame using MV3DIS Eqs. (1)-(2)."""
    points_h = np.asarray(points_h, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if points_h.ndim != 2 or points_h.shape[1] != 4:
        raise ValueError("homogeneous points must have shape N x 4")
    if camera_to_world.shape != (4, 4) or intrinsic.shape != (4, 4):
        raise ValueError("pose and intrinsic must be 4 x 4")
    if depth_map.ndim != 2:
        raise ValueError("depth map must be two-dimensional")

    world_to_camera = np.linalg.inv(camera_to_world)
    camera = points_h @ world_to_camera.T
    projected = camera @ intrinsic.T
    depth = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth) & (depth > 0.0)
    pixels = np.zeros((len(points_h), 2), dtype=np.int64)
    pixels[finite, 0] = np.floor(u[finite]).astype(np.int64)
    pixels[finite, 1] = np.floor(v[finite]).astype(np.int64)
    height, width = depth_map.shape
    in_image = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    measured = np.zeros(len(points_h), dtype=np.float64)
    selected = np.flatnonzero(in_image)
    measured[selected] = depth_map[pixels[selected, 1], pixels[selected, 0]]
    visible, weights = relative_depth_weights(depth, measured, in_image, alpha)
    return pixels, visible, weights


def mask_points_from_projection(mask, pixels, visible, weights, depth_shape):
    """Map depth-grid projections into the source RGB/RLE resolution."""
    mask = np.asarray(mask, dtype=bool)
    pixels = np.asarray(pixels, dtype=np.int64)
    visible = np.asarray(visible, dtype=bool)
    weights = np.asarray(weights, dtype=np.float32)
    if pixels.shape != (len(visible), 2) or weights.shape != visible.shape:
        raise ValueError("projection arrays have inconsistent shapes")
    point_indices = np.flatnonzero(visible)
    if not len(point_indices):
        return point_indices.astype(np.int64), np.empty(0, dtype=np.float32)
    depth_height, depth_width = map(int, depth_shape)
    xs = np.rint(
        pixels[point_indices, 0].astype(np.float64) * mask.shape[1] / depth_width
    ).astype(np.int64)
    ys = np.rint(
        pixels[point_indices, 1].astype(np.float64) * mask.shape[0] / depth_height
    ).astype(np.int64)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    selected = valid & mask[ys.clip(0, mask.shape[0] - 1), xs.clip(0, mask.shape[1] - 1)]
    result_points = point_indices[selected].astype(np.int64, copy=False)
    result_weights = weights[result_points].astype(np.float32, copy=False)
    if np.any(result_weights <= 0.0) or np.any(result_weights > 1.0):
        raise ValueError("mask point weights are outside (0, 1]")
    return result_points, result_weights


def _build_scene(scene_name, args):
    from utils import WORLD_2_CAM

    source_scene = args.automatic_root / scene_name
    observations = []
    with (source_scene / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                observations.append(json.loads(line))
    observation_ids = [int(row["observation_id"]) for row in observations]
    if observation_ids != sorted(set(observation_ids)):
        raise ValueError(f"{scene_name} observation IDs are not ordered and unique")
    if any("mask_rle" not in row for row in observations):
        raise ValueError(f"{scene_name} lacks exact RLE masks")

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    points_h, _ = world.load_ply(world.mesh)
    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if len(points_h) != len(processed) or not np.allclose(
        points_h[:, :3], np.asarray(processed[:, :3]), atol=1e-5, rtol=0.0
    ):
        raise ValueError(f"{scene_name} mesh and processed point order differ")
    intrinsic = world.adjust_intrinsic(
        np.loadtxt(world.intrinsics[0]), world.image_resolution, world.depth_resolution
    )
    frame_indices = sorted({int(row["frame_index"]) for row in observations})
    if any(item < 0 or item >= len(world.poses) for item in frame_indices):
        raise ValueError(f"{scene_name} observation frame index is invalid")

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    (staging / "points").mkdir(parents=True)
    (staging / "visibility").mkdir()
    try:
        projections = {}
        frame_rows = []
        for frame_index in frame_indices:
            depth_map = np.asarray(
                imageio.imread(world.depth_maps_paths[frame_index]), dtype=np.float64
            ) / args.depth_scale
            pixels, visible, weights = project_relative_depth_frame(
                points_h,
                np.loadtxt(world.poses[frame_index]),
                intrinsic,
                depth_map,
            )
            projections[frame_index] = (pixels, visible, weights)
            visible_points = np.flatnonzero(visible).astype(np.int64)
            filename = f"frame{frame_index:04d}_visibility.npz"
            np.savez_compressed(
                staging / "visibility" / filename,
                point_indices=visible_points,
                depth_weights=weights[visible_points].astype(np.float32),
            )
            frame_rows.append({
                "frame_index": frame_index,
                "frame_id": str(Path(world.color_paths[frame_index]).stem),
                "visible_point_count": len(visible_points),
                "visibility_path": str(
                    args.output_root / scene_name / "visibility" / filename
                ),
                "relative_depth_alpha": RELATIVE_DEPTH_ALPHA,
                "projection_contract": PROJECTION_CONTRACT,
                "gt_usage": "none",
            })

        public_observations = []
        totals = Counter()
        for row in observations:
            observation_id = int(row["observation_id"])
            frame_index = int(row["frame_index"])
            pixels, visible, weights = projections[frame_index]
            mask = decode_binary_mask_rle(row["mask_rle"])
            points, point_weights = mask_points_from_projection(
                mask, pixels, visible, weights, world.depth_resolution
            )
            filename = f"obs{observation_id:06d}_points.npz"
            np.savez_compressed(
                staging / "points" / filename,
                point_indices=points,
                depth_weights=point_weights,
            )
            with np.load(row["point_indices_path"]) as payload:
                source_points = np.unique(
                    np.asarray(payload["point_indices"], dtype=np.int64)
                )
            intersection = int(np.intersect1d(source_points, points).size)
            totals.update({
                "source_observation_point_count": len(source_points),
                "relative_observation_point_count": len(points),
                "shared_observation_point_count": intersection,
                "source_only_observation_point_count": len(source_points) - intersection,
                "relative_only_observation_point_count": len(points) - intersection,
                "empty_relative_observation_count": int(len(points) == 0),
            })
            public = dict(row)
            public.update({
                "point_indices_path": str(
                    args.output_root / scene_name / "points" / filename
                ),
                "depth_weights_in_point_file": True,
                "relative_depth_alpha": RELATIVE_DEPTH_ALPHA,
                "projection_contract": PROJECTION_CONTRACT,
                "source_point_indices_path": str(row["point_indices_path"]),
            })
            public_observations.append(public)

        _write_jsonl(staging / "automatic_observations.jsonl", public_observations)
        _write_jsonl(staging / "relative_visibility_frames.jsonl", frame_rows)
        summary = {
            "scene_name": scene_name,
            "frame_count": len(frame_rows),
            "observation_count": len(public_observations),
            **dict(totals),
            "relative_depth_alpha": RELATIVE_DEPTH_ALPHA,
            "visibility_contract": "strict |zc-d| < 0.05*d",
            "depth_weight_contract": "wpd = 1-|zc-d|/(0.05*d)",
            "projection_contract": PROJECTION_CONTRACT,
            "source_observation_mutation_count": 0,
            "ground_truth_usage": "none",
            "decision_state": (
                "Parallel paper-reference reprojection cache; source observations and "
                "Open-YOLO visibility unchanged."
            ),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    del world, points_h, processed, projections
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "automatic_root", "processed_scene_root", "dataset_root",
        "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("projection_contract") != PROJECTION_CONTRACT:
                raise SystemExit(f"resume projection contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: observations "
                f"{summary['observation_count']}, points "
                f"{summary['source_observation_point_count']} -> "
                f"{summary['relative_observation_point_count']}",
                flush=True,
            )
        summaries.append(summary)
    count_keys = (
        "frame_count", "observation_count", "source_observation_point_count",
        "relative_observation_point_count", "shared_observation_point_count",
        "source_only_observation_point_count", "relative_only_observation_point_count",
        "empty_relative_observation_count", "source_observation_mutation_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{key: sum(int(row.get(key, 0)) for row in summaries) for key in count_keys},
        "relative_depth_alpha": RELATIVE_DEPTH_ALPHA,
        "visibility_contract": "strict |zc-d| < 0.05*d",
        "depth_weight_contract": "wpd = 1-|zc-d|/(0.05*d)",
        "projection_contract": PROJECTION_CONTRACT,
        "ground_truth_usage": "none",
        "params": {key: value for key, value in vars(args).items() if key != "config"},
        "scene_summaries": summaries,
    }
    (args.output_root / "mv3dis_relative_depth_observation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "observation_count": payload["observation_count"],
        "source_observation_point_count": payload["source_observation_point_count"],
        "relative_observation_point_count": payload["relative_observation_point_count"],
        "source_observation_mutation_count": payload["source_observation_mutation_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

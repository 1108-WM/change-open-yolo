#!/usr/bin/env python3
"""为固定 v0/v1.2 轨迹构建三维核心点提示 SAM 的无 GT 执行计划。

每条轨迹只从 v0 与 v1.2 的共同 superpoint 核心中选择一个三维点。该点优先
覆盖最多未使用 uniform30 帧，再以到共同核心几何中位点的距离和点编号确定性
打破平局。提示帧排除原成员及重观测帧，按共同核心可见点数排序，最多保留五帧。
本工具不加载 SAM，不读取 GT、类别、语义或 native，也不生成 mask 或候选。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from refine_details_automatic_tracks_consensus import _superpoint_points
from refine_details_consensus_all_view_reobservation import _read_scenes, _resolve
from score_details_consensus_track_out_holdout import load_selected_frame_indices


def select_consistent_prompt_point(point_indices, available_frames, visibility, xyz):
    """选择跨未使用帧可见次数最多、且位于共同核心中央的三维点。"""
    points = np.unique(np.asarray(point_indices, dtype=np.int64))
    frames = np.unique(np.asarray(available_frames, dtype=np.int64))
    if len(points) == 0 or len(frames) == 0:
        return None
    visible_counts = visibility[frames][:, points].sum(axis=0, dtype=np.int64)
    best_count = int(visible_counts.max(initial=0))
    if best_count <= 0:
        return None
    candidates = points[visible_counts == best_count]
    center = np.median(np.asarray(xyz[points], dtype=np.float64), axis=0)
    distances = np.linalg.norm(np.asarray(xyz[candidates], dtype=np.float64) - center, axis=1)
    order = np.lexsort((candidates, distances))
    return int(candidates[int(order[0])])


def select_prompt_frames(
    prompt_point,
    common_core_points,
    available_frames,
    visibility,
    projections,
    scaling_params,
    image_shape,
    max_prompt_frames,
):
    """按共同核心可见点数选择可投影的新帧，并返回 RGB 像素提示。"""
    height, width = (int(value) for value in image_shape)
    core_points = np.unique(np.asarray(common_core_points, dtype=np.int64))
    rows = []
    for frame_index in sorted(set(int(item) for item in available_frames)):
        if not bool(visibility[frame_index, int(prompt_point)]):
            continue
        coords = np.asarray(projections[frame_index, int(prompt_point)], dtype=np.float64)
        x = float(coords[0] / float(scaling_params[1]))
        y = float(coords[1] / float(scaling_params[0]))
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue
        rows.append({
            "frame_index": int(frame_index),
            "prompt_xy": [x, y],
            "visible_common_core_point_count": int(
                visibility[frame_index, core_points].sum(dtype=np.int64)
            ),
        })
    return sorted(
        rows,
        key=lambda row: (-row["visible_common_core_point_count"], row["frame_index"]),
    )[: int(max_prompt_frames)]


def _track_map(payload, scene_name, source_name):
    tracks = payload.get("tracks", [])
    mapped = {int(track["track_id"]): track for track in tracks}
    if len(mapped) != len(tracks):
        raise ValueError(f"{scene_name} 的 {source_name} 含重复 track_id")
    return mapped


def _build_scene(scene_name, args):
    from utils import WORLD_2_CAM

    base_payload = json.loads(
        (args.base_track_root / scene_name / "automatic_tracks.json").read_text()
    )
    candidate_payload = json.loads(
        (args.candidate_track_root / scene_name / "automatic_tracks.json").read_text()
    )
    base_by_id = _track_map(base_payload, scene_name, "v0")
    candidate_by_id = _track_map(candidate_payload, scene_name, "v1.2")
    if set(base_by_id) != set(candidate_by_id):
        raise ValueError(f"{scene_name} 的 v0/v1.2 track_id 集合不一致")

    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少 XYZ 或原始 superpoint 列")
    xyz = np.asarray(processed[:, :3], dtype=np.float64)
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint = _superpoint_points(superpoints)

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections_raw, visibility_raw = world.get_mesh_projections()
    projections = projections_raw.detach().cpu().numpy().astype(np.float64, copy=False)
    visibility = visibility_raw.detach().cpu().numpy().astype(bool, copy=False)
    selected_frames = load_selected_frame_indices(args.automatic_root / scene_name)
    scaling_params = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    observation_frames = {}
    observation_path = args.automatic_root / scene_name / "automatic_observations.jsonl"
    for line in observation_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            observation_frames[int(row["observation_id"])] = int(row["frame_index"])

    plans = []
    no_common_core_count = 0
    no_new_visible_frame_count = 0
    for track_id in sorted(base_by_id):
        base = base_by_id[track_id]
        candidate = candidate_by_id[track_id]
        common_superpoints = sorted(
            set(int(item) for item in base.get("superpoint_ids", []))
            & set(int(item) for item in candidate.get("superpoint_ids", []))
        )
        common_points = [
            points_by_superpoint[superpoint_id]
            for superpoint_id in common_superpoints
            if superpoint_id in points_by_superpoint
        ]
        if not common_points:
            no_common_core_count += 1
            continue
        common_points = np.unique(np.concatenate(common_points).astype(np.int64, copy=False))
        observation_ids = [int(item) for item in candidate.get("observation_ids", [])]
        missing = sorted(set(observation_ids) - set(observation_frames))
        if missing:
            raise ValueError(
                f"{scene_name} track {track_id} 引用了未知自动观测：{missing[:5]}"
            )
        excluded_frames = sorted({observation_frames[item] for item in observation_ids})
        available_frames = sorted(set(selected_frames) - set(excluded_frames))
        prompt_point = select_consistent_prompt_point(
            common_points, available_frames, visibility, xyz
        )
        if prompt_point is None:
            no_new_visible_frame_count += 1
            continue
        prompt_frames = select_prompt_frames(
            prompt_point,
            common_points,
            available_frames,
            visibility,
            projections,
            scaling_params,
            world.image_resolution,
            args.max_prompt_frames,
        )
        if not prompt_frames:
            no_new_visible_frame_count += 1
            continue
        for frame in prompt_frames:
            frame["frame_id"] = Path(world.color_paths[frame["frame_index"]]).stem
        prompt_superpoint = int(superpoints[prompt_point])
        if prompt_superpoint not in common_superpoints:
            raise AssertionError("选出的提示点不属于共同核心")
        plans.append({
            "scene_name": scene_name,
            "track_id": int(track_id),
            "prompt_point_index": int(prompt_point),
            "prompt_superpoint_id": prompt_superpoint,
            "common_core_superpoint_ids": common_superpoints,
            "common_core_superpoint_count": int(len(common_superpoints)),
            "common_core_point_count": int(len(common_points)),
            "excluded_frame_indices": excluded_frames,
            "prompt_frames": prompt_frames,
        })

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    staging_root.mkdir(parents=True)
    scene_summary = {
        "scene_name": scene_name,
        "source_track_count": len(candidate_by_id),
        "planned_track_count": len(plans),
        "prompt_request_count": sum(len(item["prompt_frames"]) for item in plans),
        "no_common_core_count": no_common_core_count,
        "no_new_visible_frame_count": no_new_visible_frame_count,
        "selected_frame_count": len(selected_frames),
    }
    (staging_root / "core_prompt_plan.json").write_text(
        json.dumps(plans, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (staging_root / "summary.json").write_text(
        json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    del world, projections_raw, visibility_raw, projections, visibility, processed
    return scene_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--base-track-root", type=Path, required=True)
    parser.add_argument("--candidate-track-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument(
        "--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-prompt-frames", type=int, default=5)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list",
        "base_track_root",
        "candidate_track_root",
        "automatic_root",
        "processed_scene_root",
        "dataset_root",
        "config_path",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_prompt_frames <= 0:
        raise SystemExit("--max-prompt-frames 必须为正数")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
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
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
        else:
            summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['planned_track_count']} 条轨迹，"
            f"{summary['prompt_request_count']} 次提示",
            flush=True,
        )

    root_summary = {
        "gt_usage": "不读取GT；不使用类别、语义或native候选。",
        "decision_state": "固定v0/v1.2共同三维核心的类别无关SAM点提示计划；尚未生成mask。",
        "scene_count": len(summaries),
        "source_track_count": sum(item["source_track_count"] for item in summaries),
        "planned_track_count": sum(item["planned_track_count"] for item in summaries),
        "prompt_request_count": sum(item["prompt_request_count"] for item in summaries),
        "no_common_core_count": sum(item["no_common_core_count"] for item in summaries),
        "no_new_visible_frame_count": sum(
            item["no_new_visible_frame_count"] for item in summaries
        ),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "core_prompt_plan_summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

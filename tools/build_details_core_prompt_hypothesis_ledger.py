#!/usr/bin/env python3
"""建立固定 v1.2 mask 与三维核心点提示 SAM 假设的无 GT 成对账本。

逐假设将三维回投提升到原始 ScanNet superpoint，在共同可见域比较固定 v1.2
mask，记录新增、保留和遗漏 superpoint。轨迹级只汇总跨帧支持：分别统计每帧
任一假设支持的上界，以及同帧三个假设全部支持的保守交集。本工具不选择假设、
不生成并集或候选，不读取 GT、类别、语义或 native，也不运行 AP。
"""

import argparse
import json
import os
import sys
from collections import defaultdict
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
from refine_details_consensus_all_view_reobservation import (
    _read_scenes,
    _resolve,
    frame_match_metrics,
)


def lift_prompt_points_to_superpoints(
    point_indices,
    superpoints,
    superpoint_sizes,
    visible_counts,
    min_visible_ratio,
    min_mask_support,
):
    points = np.unique(np.asarray(point_indices, dtype=np.int64))
    points = points[(points >= 0) & (points < len(superpoints))]
    ids, counts = np.unique(superpoints[points], return_counts=True)
    lifted = []
    for superpoint_id, inside_count in zip(ids, counts):
        superpoint_id = int(superpoint_id)
        visible_count = int(visible_counts.get(superpoint_id, 0))
        total_count = int(superpoint_sizes.get(superpoint_id, 0))
        if visible_count <= 0 or total_count <= 0:
            continue
        if visible_count / total_count < float(min_visible_ratio):
            continue
        if int(inside_count) / visible_count >= float(min_mask_support):
            lifted.append(superpoint_id)
    return sorted(lifted)


def aggregate_track_hypotheses(rows, base_superpoints, min_support_frames):
    """以 distinct frame 聚合任一假设上界和三假设交集，不选择单个假设。"""
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame_index"])].append(row)
    any_support = defaultdict(set)
    all_support = defaultdict(set)
    frame_rows = []
    base = set(int(item) for item in base_superpoints)
    for frame_index, frame_hypotheses in sorted(by_frame.items()):
        if len(frame_hypotheses) != 3:
            raise ValueError(f"frame {frame_index} 不是三个固定 SAM 假设")
        hypothesis_sets = [set(int(item) for item in row["lifted_superpoints"]) for row in frame_hypotheses]
        union = set.union(*hypothesis_sets) if hypothesis_sets else set()
        intersection = set.intersection(*hypothesis_sets) if hypothesis_sets else set()
        added_union = union - base
        added_intersection = intersection - base
        for superpoint_id in added_union:
            any_support[superpoint_id].add(frame_index)
        for superpoint_id in added_intersection:
            all_support[superpoint_id].add(frame_index)
        frame_rows.append({
            "frame_index": int(frame_index),
            "added_superpoints_any_hypothesis": sorted(added_union),
            "added_superpoints_all_hypotheses": sorted(added_intersection),
            "added_superpoint_count_any_hypothesis": len(added_union),
            "added_superpoint_count_all_hypotheses": len(added_intersection),
        })

    def support_rows(mapping):
        return [
            {"superpoint_id": int(superpoint_id), "support_frame_count": len(frames)}
            for superpoint_id, frames in sorted(mapping.items())
        ]

    any_rows = support_rows(any_support)
    all_rows = support_rows(all_support)
    return {
        "prompt_frame_count": len(by_frame),
        "new_superpoint_support_any_hypothesis": any_rows,
        "new_superpoint_support_all_hypotheses": all_rows,
        "stable_new_superpoints_any_hypothesis": [
            row["superpoint_id"]
            for row in any_rows
            if row["support_frame_count"] >= int(min_support_frames)
        ],
        "stable_new_superpoints_all_hypotheses": [
            row["superpoint_id"]
            for row in all_rows
            if row["support_frame_count"] >= int(min_support_frames)
        ],
        "frames": frame_rows,
    }


def _visible_counts(superpoints, visibility, frame_indices):
    output = {}
    for frame_index in sorted(set(int(item) for item in frame_indices)):
        ids, counts = np.unique(
            superpoints[np.flatnonzero(visibility[frame_index])], return_counts=True
        )
        output[frame_index] = {
            int(superpoint_id): int(count)
            for superpoint_id, count in zip(ids, counts)
        }
    return output


def _build_scene(scene_name, args):
    from utils import WORLD_2_CAM

    track_payload = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    tracks = {int(row["track_id"]): row for row in track_payload.get("tracks", [])}
    if len(tracks) != len(track_payload.get("tracks", [])):
        raise ValueError(f"{scene_name} 含重复 track_id")
    observation_path = args.prompt_root / scene_name / "prompt_observations.jsonl"
    observations = [
        json.loads(line) for line in observation_path.read_text().splitlines() if line.strip()
    ]
    observations_by_track = defaultdict(list)
    for row in observations:
        track_id = int(row["track_id"])
        if track_id not in tracks:
            raise ValueError(f"{scene_name} 提示观测引用未知 track {track_id}")
        observations_by_track[track_id].append(row)

    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {
        int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)
    }
    points_by_superpoint = _superpoint_points(superpoints)

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, visibility_raw = world.get_mesh_projections()
    visibility = visibility_raw.detach().cpu().numpy().astype(bool, copy=False)
    frame_indices = [int(row["frame_index"]) for row in observations]
    visible_counts_by_frame = _visible_counts(superpoints, visibility, frame_indices)

    hypothesis_ledger = []
    track_ledger = []
    tracks_with_stable_any = 0
    tracks_with_stable_all = 0
    for track_id, raw_rows in sorted(observations_by_track.items()):
        track = tracks[track_id]
        base_superpoints = set(int(item) for item in track.get("superpoint_ids", []))
        base_points = [
            points_by_superpoint[superpoint_id]
            for superpoint_id in base_superpoints
            if superpoint_id in points_by_superpoint
        ]
        base_points = (
            np.unique(np.concatenate(base_points).astype(np.int64, copy=False))
            if base_points
            else np.empty(0, dtype=np.int64)
        )
        ledger_rows = []
        for raw in sorted(
            raw_rows,
            key=lambda row: (int(row["frame_index"]), int(row["hypothesis_index"])),
        ):
            points = np.unique(
                np.asarray(np.load(raw["point_indices_path"])["point_indices"], dtype=np.int64)
            )
            points = points[(points >= 0) & (points < len(superpoints))]
            frame_index = int(raw["frame_index"])
            visible_counts = visible_counts_by_frame[frame_index]
            lifted = lift_prompt_points_to_superpoints(
                points,
                superpoints,
                superpoint_sizes,
                visible_counts,
                args.min_superpoint_visible_ratio,
                args.min_mask_support,
            )
            metrics = frame_match_metrics(
                base_superpoints,
                lifted,
                visible_counts,
                args.min_visible_points_per_superpoint,
            )
            visible_base = {
                superpoint_id
                for superpoint_id in base_superpoints
                if int(visible_counts.get(superpoint_id, 0))
                >= args.min_visible_points_per_superpoint
            }
            lifted_set = set(lifted)
            added = sorted(lifted_set - base_superpoints)
            omitted = sorted(visible_base - lifted_set)
            intersection_points = int(
                np.intersect1d(points, base_points, assume_unique=True).size
            )
            row = {
                "scene_name": scene_name,
                "track_id": int(track_id),
                "observation_id": int(raw["observation_id"]),
                "frame_id": str(raw["frame_id"]),
                "frame_index": frame_index,
                "hypothesis_index": int(raw["hypothesis_index"]),
                "sam_predicted_iou": float(raw["sam_predicted_iou"]),
                "prompt_point_backprojected": bool(raw["prompt_point_backprojected"]),
                "visible_core_support_ratio": float(raw["visible_core_support_ratio"]),
                "observation_core_purity_ratio": float(raw["observation_core_purity_ratio"]),
                "point_count": int(len(points)),
                "base_point_intersection_count": intersection_points,
                "base_inside_observation_ratio": float(
                    intersection_points / max(1, len(base_points))
                ),
                "observation_inside_base_ratio": float(
                    intersection_points / max(1, len(points))
                ),
                "lifted_superpoints": lifted,
                "lifted_superpoint_count": len(lifted),
                "added_superpoints": added,
                "added_superpoint_count": len(added),
                "omitted_visible_base_superpoints": omitted,
                "omitted_visible_base_superpoint_count": len(omitted),
                **metrics,
            }
            ledger_rows.append(row)
            hypothesis_ledger.append(row)
        aggregate = aggregate_track_hypotheses(
            ledger_rows, base_superpoints, args.min_support_frames
        )
        stable_any = aggregate["stable_new_superpoints_any_hypothesis"]
        stable_all = aggregate["stable_new_superpoints_all_hypotheses"]
        tracks_with_stable_any += int(bool(stable_any))
        tracks_with_stable_all += int(bool(stable_all))
        track_ledger.append({
            "scene_name": scene_name,
            "track_id": int(track_id),
            "base_superpoint_count": len(base_superpoints),
            "base_point_count": len(base_points),
            "hypothesis_count": len(ledger_rows),
            **aggregate,
        })

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    staging_root.mkdir(parents=True)
    with (staging_root / "prompt_hypothesis_ledger.jsonl").open("w") as handle:
        for row in hypothesis_ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (staging_root / "prompt_track_ledger.json").write_text(
        json.dumps(track_ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "scene_name": scene_name,
        "prompt_track_count": len(track_ledger),
        "hypothesis_count": len(hypothesis_ledger),
        "tracks_with_stable_new_superpoints_any_hypothesis": tracks_with_stable_any,
        "tracks_with_stable_new_superpoints_all_hypotheses": tracks_with_stable_all,
    }
    (staging_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    del world, visibility_raw, visibility, processed
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument(
        "--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-superpoint-visible-ratio", type=float, default=0.10)
    parser.add_argument("--min-mask-support", type=float, default=0.30)
    parser.add_argument("--min-visible-points-per-superpoint", type=int, default=3)
    parser.add_argument("--min-support-frames", type=int, default=2)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list",
        "track_root",
        "prompt_root",
        "processed_scene_root",
        "dataset_root",
        "config_path",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
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
            f"稳定增量(any/all)="
            f"{summary['tracks_with_stable_new_superpoints_any_hypothesis']}/"
            f"{summary['tracks_with_stable_new_superpoints_all_hypotheses']}",
            flush=True,
        )

    root_summary = {
        "gt_usage": "不读取GT；不使用类别、语义或native候选。",
        "decision_state": "固定提示假设相对v1.2的连续几何账本；未选择、聚合或形成候选。",
        "scene_count": len(summaries),
        "prompt_track_count": sum(item["prompt_track_count"] for item in summaries),
        "hypothesis_count": sum(item["hypothesis_count"] for item in summaries),
        "tracks_with_stable_new_superpoints_any_hypothesis": sum(
            item["tracks_with_stable_new_superpoints_any_hypothesis"] for item in summaries
        ),
        "tracks_with_stable_new_superpoints_all_hypotheses": sum(
            item["tracks_with_stable_new_superpoints_all_hypotheses"] for item in summaries
        ),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "prompt_hypothesis_ledger_summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

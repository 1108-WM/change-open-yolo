#!/usr/bin/env python3
"""以多视图 superpoint 共识细化固定的自动 SAM 轨迹。

输入必须是已经形成的类别无关轨迹及其原始自动 SAM 观测。每条轨迹先恢复关联时
使用的 superpoint 闭包，再按 Details Matter 后端已有的“支持帧/可见帧”定义生成
一个并行细化变体。原轨迹不会被覆盖；本工具不合并轨迹、不删除包含候选、不处理
轨迹间竞争，也不读取 GT、类别或 native 候选。
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


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复场景")
    return scenes


def initial_candidate_superpoints(
    frame_rows,
    superpoint_sizes,
    min_visible_ratio,
    min_mask_support,
):
    """恢复轨迹关联阶段认可过的逐帧 superpoint，并取其并集。"""
    selected = set()
    for row in frame_rows:
        for superpoint_id, inside_count in row["inside_counts"].items():
            visible_count = int(row["visible_counts"].get(superpoint_id, 0))
            total_count = int(superpoint_sizes.get(superpoint_id, 0))
            if visible_count <= 0 or total_count <= 0:
                continue
            if visible_count / total_count < float(min_visible_ratio):
                continue
            if inside_count / visible_count >= float(min_mask_support):
                selected.add(int(superpoint_id))
    return selected


def consensus_superpoints(
    candidate_superpoints,
    frame_rows,
    min_visible_points,
    frame_superpoint_coverage,
    min_support_frames,
    min_consensus_rate,
    mean_superpoint_coverage,
):
    """计算每个候选 superpoint 的支持帧/可见帧共识。"""
    support = defaultdict(int)
    visible = defaultdict(int)
    coverage_sum = defaultdict(float)
    for row in frame_rows:
        for superpoint_id in candidate_superpoints:
            visible_count = int(row["visible_counts"].get(superpoint_id, 0))
            if visible_count < int(min_visible_points):
                continue
            ratio = float(row["inside_counts"].get(superpoint_id, 0) / visible_count)
            visible[superpoint_id] += 1
            if ratio >= float(frame_superpoint_coverage):
                support[superpoint_id] += 1
                coverage_sum[superpoint_id] += ratio

    kept = {
        int(superpoint_id)
        for superpoint_id in candidate_superpoints
        if support[superpoint_id] >= int(min_support_frames)
        and support[superpoint_id] / max(1, visible[superpoint_id]) >= float(min_consensus_rate)
        and coverage_sum[superpoint_id] / max(1, support[superpoint_id]) >= float(mean_superpoint_coverage)
    }
    diagnostics = {
        int(superpoint_id): {
            "support_frames": int(support[superpoint_id]),
            "visible_frames": int(visible[superpoint_id]),
            "consensus_rate": float(support[superpoint_id] / max(1, visible[superpoint_id])),
            "mean_supported_coverage": float(
                coverage_sum[superpoint_id] / max(1, support[superpoint_id])
            ),
        }
        for superpoint_id in sorted(candidate_superpoints)
    }
    return kept, diagnostics


def collapse_tracklet_rows_by_frame(frame_rows):
    """Collapse multiple masks from one merged tracklet frame into one view vote.

    Details defines consensus over tracked frames, not over the number of masks that
    happen to survive in a frame. When point provenance is available, the mask union
    is computed exactly per superpoint; count-only test callers use a conservative max.
    """
    grouped = {}
    for row in frame_rows:
        frame_index = int(row["frame_index"])
        visible_counts = {
            int(key): int(value) for key, value in row["visible_counts"].items()
        }
        inside_counts = {
            int(key): int(value) for key, value in row["inside_counts"].items()
        }
        point_support = row.get("inside_point_indices_by_superpoint")
        if point_support is not None:
            point_support = {
                int(key): np.unique(np.asarray(value, dtype=np.int64))
                for key, value in point_support.items()
            }
        if frame_index not in grouped:
            grouped[frame_index] = {
                "frame_index": frame_index,
                "visible_counts": visible_counts,
                "inside_counts": inside_counts,
                "_inside_point_indices_by_superpoint": point_support,
            }
            continue
        current = grouped[frame_index]
        if current["visible_counts"] != visible_counts:
            raise ValueError(f"frame {frame_index} has inconsistent visibility counts")
        current_support = current["_inside_point_indices_by_superpoint"]
        if (current_support is None) != (point_support is None):
            raise ValueError(f"frame {frame_index} mixes exact and count-only mask support")
        if point_support is None:
            for superpoint_id, count in inside_counts.items():
                current["inside_counts"][superpoint_id] = max(
                    int(count), int(current["inside_counts"].get(superpoint_id, 0))
                )
        else:
            for superpoint_id, points in point_support.items():
                current_support[superpoint_id] = np.union1d(
                    current_support.get(superpoint_id, np.empty(0, dtype=np.int64)),
                    points,
                )
            current["inside_counts"] = {
                int(superpoint_id): int(len(points))
                for superpoint_id, points in current_support.items()
            }
    result = []
    for frame_index in sorted(grouped):
        row = grouped[frame_index]
        row.pop("_inside_point_indices_by_superpoint", None)
        result.append(row)
    return result


def refine_tracklet_superpoints(
    observation_ids,
    observations,
    superpoint_sizes,
    min_visible_ratio,
    min_mask_support,
    min_visible_points,
    frame_superpoint_coverage,
    min_support_frames,
    min_consensus_rate,
    mean_superpoint_coverage,
    candidate_superpoints=None,
):
    """Run the frozen D1 consensus contract on a possibly merged tracklet."""
    ids = sorted(set(int(item) for item in observation_ids))
    missing = [item for item in ids if item not in observations]
    if missing:
        raise ValueError(f"tracklet references missing observations: {missing[:5]}")
    frame_rows = collapse_tracklet_rows_by_frame([observations[item] for item in ids])
    if candidate_superpoints is None:
        initial = initial_candidate_superpoints(
            frame_rows, superpoint_sizes, min_visible_ratio, min_mask_support
        )
    else:
        initial = set(int(item) for item in candidate_superpoints)
        missing_superpoints = sorted(initial - set(superpoint_sizes))
        if missing_superpoints:
            raise ValueError(
                f"tracklet references missing superpoints: {missing_superpoints[:5]}"
            )
    kept, diagnostics = consensus_superpoints(
        initial,
        frame_rows,
        min_visible_points,
        frame_superpoint_coverage,
        min_support_frames,
        min_consensus_rate,
        mean_superpoint_coverage,
    )
    return kept, diagnostics, frame_rows, initial


def _load_observations(scene_root, superpoints, visibility):
    used_frames = set()
    raw_rows = []
    with (scene_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            frame_index = int(raw["frame_index"])
            used_frames.add(frame_index)
            raw_rows.append((raw, frame_index))

    visible_counts_by_frame = {}
    for frame_index in sorted(used_frames):
        ids, counts = np.unique(superpoints[np.flatnonzero(visibility[frame_index])], return_counts=True)
        visible_counts_by_frame[frame_index] = {
            int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)
        }

    observations = {}
    for raw, frame_index in raw_rows:
        points = np.unique(np.asarray(np.load(raw["point_indices_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(superpoints))]
        ids, counts = np.unique(superpoints[points], return_counts=True)
        observations[int(raw["observation_id"])] = {
            "frame_index": frame_index,
            "inside_counts": {
                int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)
            },
            "inside_point_indices_by_superpoint": {
                int(superpoint_id): points[superpoints[points] == superpoint_id]
                for superpoint_id in ids
            },
            "visible_counts": visible_counts_by_frame[frame_index],
        }
    return observations


def _superpoint_points(superpoints):
    order = np.argsort(superpoints, kind="mergesort")
    sorted_ids = superpoints[order]
    ids, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], len(order))
    return {
        int(superpoint_id): np.asarray(order[start:end], dtype=np.int64)
        for superpoint_id, start, end in zip(ids, starts, ends)
    }


def _refine_scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
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
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations = _load_observations(args.automatic_root / scene_name, superpoints, visibility)
    source = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    staging_root.mkdir(parents=True)
    point_root = staging_root / "track_points"
    point_root.mkdir()

    records = []
    empty_count = 0
    removed_superpoint_count = 0
    for track in source.get("tracks", []):
        kept, diagnostics, frame_rows, initial = refine_tracklet_superpoints(
            track.get("observation_ids", []),
            observations,
            superpoint_sizes,
            args.min_superpoint_visible_ratio,
            args.min_initial_mask_support,
            args.min_visible_points_per_superpoint,
            args.frame_superpoint_coverage,
            args.min_support_frames,
            args.min_consensus_rate,
            args.mean_superpoint_coverage,
        )
        removed_superpoint_count += len(initial - kept)
        if not kept:
            empty_count += 1
            continue
        chunks = [points_by_superpoint[superpoint_id] for superpoint_id in sorted(kept)]
        points = np.concatenate(chunks).astype(np.int64, copy=False)
        source_track_id = int(track["track_id"])
        filename = f"track{source_track_id:04d}_points.npz"
        np.savez_compressed(point_root / filename, point_indices=points)
        support_values = [diagnostics[item]["support_frames"] for item in kept]
        consensus_values = [diagnostics[item]["consensus_rate"] for item in kept]
        coverage_values = [diagnostics[item]["mean_supported_coverage"] for item in kept]
        record = dict(track)
        record.update({
            "track_id": source_track_id,
            "source_track_id": source_track_id,
            "point_count": int(len(points)),
            "points_path": str(args.output_root / scene_name / "track_points" / filename),
            "initial_superpoint_count": int(len(initial)),
            "superpoint_count": int(len(kept)),
            "removed_superpoint_count": int(len(initial - kept)),
            "mean_consensus_rate": float(np.mean(consensus_values)),
            "mean_supported_coverage": float(np.mean(coverage_values)),
            "support_score": float(np.sum(support_values)),
            "superpoint_ids": sorted(int(item) for item in kept),
        })
        records.append(record)

    payload = {
        "scene_name": scene_name,
        "source_track_count": len(source.get("tracks", [])),
        "track_count": len(records),
        "empty_after_consensus_count": empty_count,
        "removed_superpoint_count": removed_superpoint_count,
        "tracks": records,
    }
    (staging_root / "automatic_tracks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    del world, raw_visibility, visibility, observations, processed
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min_superpoint_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_initial_mask_support", type=float, default=0.30)
    parser.add_argument("--frame_superpoint_coverage", type=float, default=0.50)
    parser.add_argument("--mean_superpoint_coverage", type=float, default=0.55)
    parser.add_argument("--min_visible_points_per_superpoint", type=int, default=3)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--min_consensus_rate", type=float, default=0.30)
    args = parser.parse_args()
    for name in (
        "scene_list", "track_root", "automatic_root", "processed_scene_root",
        "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    for name in (
        "min_superpoint_visible_ratio", "min_initial_mask_support", "frame_superpoint_coverage",
        "mean_superpoint_coverage", "min_consensus_rate",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name} 必须在零到一之间")
    if args.min_visible_points_per_superpoint <= 0 or args.min_support_frames <= 0:
        raise SystemExit("可见点数与支持帧数必须为正数")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "automatic_tracks.json"
        if existing.is_file():
            if not args.resume:
                raise SystemExit(f"输出场景已存在：{scene_name}")
            summary = json.loads(existing.read_text())
            summaries.append(summary)
            print(f"[跳过已有] {index}/{len(scenes)} {scene_name}: 轨迹 {summary['track_count']}", flush=True)
            continue
        summary = _refine_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['source_track_count']} -> {summary['track_count']} 条轨迹",
            flush=True,
        )

    payload = {
        "gt_usage": "不读取 GT；不使用类别、语义或 native 候选。",
        "decision_state": "固定 Details 轨迹的并行多视图 superpoint 共识变体；原轨迹保持可回退。",
        "scene_count": len(summaries),
        "source_track_count": sum(item["source_track_count"] for item in summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "empty_after_consensus_count": sum(item["empty_after_consensus_count"] for item in summaries),
        "removed_superpoint_count": sum(item["removed_superpoint_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_track_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

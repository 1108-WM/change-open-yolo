#!/usr/bin/env python3
"""在轨迹形成视角之外，为固定 v0/v1.2 mask 建立成对质量账本。

对每条轨迹排除原成员观测和新增重观测所在帧，只在双方都可见的剩余
uniform 帧上比较 v0 与固定 v1.2 mask。没有自动 SAM 观测的可见帧显式记零。
工具不读取 GT、类别、语义或 native，不修改 mask；输出轨迹仍引用 v1.2 的
固定点索引，只新增预注册的 ``track_out_holdout_quality`` 排序字段。
"""

import argparse
import json
import math
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

from refine_details_consensus_all_view_reobservation import (
    _load_observations,
    _read_scenes,
    _resolve,
    frame_match_metrics,
    proposal_dominates_base,
)


def load_selected_frame_indices(scene_root):
    summary_path = scene_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"自动 SAM 场景缺少 summary.json：{scene_root}")
    payload = json.loads(summary_path.read_text())
    frames = [int(row["frame_index"]) for row in payload.get("frames", [])]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f"{scene_root.name} 的选择帧为空或重复")
    return sorted(frames)


def visible_counts_for_frames(superpoints, visibility, frame_indices):
    counts_by_frame = {}
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(visibility):
            raise ValueError(f"选择帧 {frame_index} 超出可见性数组范围")
        ids, counts = np.unique(
            superpoints[np.flatnonzero(visibility[frame_index])],
            return_counts=True,
        )
        counts_by_frame[int(frame_index)] = {
            int(superpoint_id): int(count)
            for superpoint_id, count in zip(ids, counts)
        }
    return counts_by_frame


def _mask_is_visible(superpoint_ids, visible_counts, min_visible_points):
    return any(
        int(visible_counts.get(int(superpoint_id), 0)) >= int(min_visible_points)
        for superpoint_id in superpoint_ids
    )


def _best_frame_siou(superpoint_ids, observations, min_visible_points):
    return max(
        (
            frame_match_metrics(
                superpoint_ids,
                observation["lifted_superpoints"],
                observation["visible_counts"],
                min_visible_points,
            )["siou"]
            for observation in observations
        ),
        default=0.0,
    )


def paired_track_out_consistency(
    base_superpoints,
    candidate_superpoints,
    selected_frame_indices,
    excluded_frame_indices,
    observations_by_frame,
    visible_counts_by_frame,
    min_visible_points,
    support_siou,
):
    """在相同的轨迹外共同可见帧域内比较两个固定 mask。"""
    base_superpoints = tuple(sorted(set(int(item) for item in base_superpoints)))
    candidate_superpoints = tuple(sorted(set(int(item) for item in candidate_superpoints)))
    excluded = set(int(item) for item in excluded_frame_indices)
    frame_rows = []
    for frame_index in sorted(set(int(item) for item in selected_frame_indices)):
        if frame_index in excluded:
            continue
        visible_counts = visible_counts_by_frame[frame_index]
        if not _mask_is_visible(base_superpoints, visible_counts, min_visible_points):
            continue
        if not _mask_is_visible(candidate_superpoints, visible_counts, min_visible_points):
            continue
        observations = observations_by_frame.get(frame_index, [])
        frame_rows.append({
            "frame_index": int(frame_index),
            "observation_count": int(len(observations)),
            "base_best_siou": float(_best_frame_siou(
                base_superpoints, observations, min_visible_points
            )),
            "candidate_best_siou": float(_best_frame_siou(
                candidate_superpoints, observations, min_visible_points
            )),
        })

    def summarize(field):
        scores = np.asarray([row[field] for row in frame_rows], dtype=np.float64)
        if len(scores) == 0:
            return {"mean_best_siou": 0.0, "support_frame_rate": 0.0}
        return {
            "mean_best_siou": float(scores.mean()),
            "support_frame_rate": float(np.mean(scores >= float(support_siou))),
        }

    return {
        "eligible_frame_count": int(len(frame_rows)),
        "zero_observation_frame_count": int(sum(
            row["observation_count"] == 0 for row in frame_rows
        )),
        "base": summarize("base_best_siou"),
        "candidate": summarize("candidate_best_siou"),
        "frames": frame_rows,
    }


def track_out_holdout_quality(mean_node_quality, consistency, min_reliable_frames):
    """证据不足时保持旧分数；充足时沿用既有几何均值公式。"""
    source_quality = max(0.0, float(mean_node_quality))
    eligible = int(consistency["eligible_frame_count"])
    if eligible < int(min_reliable_frames):
        return source_quality, "mean_node_quality_fallback"
    mean_siou = max(0.0, float(consistency["candidate"]["mean_best_siou"]))
    return float(math.sqrt(source_quality * mean_siou)), "track_out_geometric_mean"


def _track_map(payload, scene_name, source_name):
    tracks = payload.get("tracks", [])
    mapped = {int(track["track_id"]): track for track in tracks}
    if len(mapped) != len(tracks):
        raise ValueError(f"{scene_name} 的 {source_name} 含重复 track_id")
    return mapped


def _score_scene(scene_name, args):
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
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {
        int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)
    }

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    automatic_scene_root = args.automatic_root / scene_name
    selected_frames = load_selected_frame_indices(automatic_scene_root)
    visible_counts = visible_counts_for_frames(superpoints, visibility, selected_frames)
    observations, observations_by_frame = _load_observations(
        automatic_scene_root,
        superpoints,
        superpoint_sizes,
        visibility,
        args,
    )
    observation_frames = {
        int(observation_id): int(row["frame_index"])
        for observation_id, row in observations.items()
    }

    scored_tracks = []
    ledger = []
    fallback_count = 0
    changed_candidate_count = 0
    candidate_dominates_count = 0
    base_dominates_count = 0
    for track_id in sorted(base_by_id):
        base = base_by_id[track_id]
        candidate = candidate_by_id[track_id]
        observation_ids = [int(item) for item in candidate.get("observation_ids", [])]
        missing = sorted(set(observation_ids) - set(observation_frames))
        if missing:
            raise ValueError(
                f"{scene_name} track {track_id} 引用了未知自动观测：{missing[:5]}"
            )
        excluded_frames = sorted({observation_frames[item] for item in observation_ids})
        base_superpoints = base.get("superpoint_ids", [])
        candidate_superpoints = candidate.get("superpoint_ids", [])
        consistency = paired_track_out_consistency(
            base_superpoints,
            candidate_superpoints,
            selected_frames,
            excluded_frames,
            observations_by_frame,
            visible_counts,
            args.min_visible_points_per_superpoint,
            args.support_siou,
        )
        quality, quality_mode = track_out_holdout_quality(
            candidate.get("mean_node_quality", 0.0),
            consistency,
            args.min_reliable_holdout_frames,
        )
        candidate_dominates = proposal_dominates_base(
            consistency["base"], consistency["candidate"]
        )
        base_dominates = proposal_dominates_base(
            consistency["candidate"], consistency["base"]
        )
        changed = set(int(item) for item in base_superpoints) != set(
            int(item) for item in candidate_superpoints
        )
        fallback_count += int(quality_mode == "mean_node_quality_fallback")
        changed_candidate_count += int(changed)
        candidate_dominates_count += int(candidate_dominates)
        base_dominates_count += int(base_dominates)

        scored = dict(candidate)
        scored.update({
            "track_out_holdout_quality": float(quality),
            "track_out_quality_mode": quality_mode,
            "track_out_excluded_frame_count": int(len(excluded_frames)),
            "track_out_eligible_frame_count": consistency["eligible_frame_count"],
            "track_out_zero_observation_frame_count": consistency[
                "zero_observation_frame_count"
            ],
            "track_out_base_mean_best_siou": consistency["base"]["mean_best_siou"],
            "track_out_base_support_frame_rate": consistency["base"][
                "support_frame_rate"
            ],
            "track_out_candidate_mean_best_siou": consistency["candidate"][
                "mean_best_siou"
            ],
            "track_out_candidate_support_frame_rate": consistency["candidate"][
                "support_frame_rate"
            ],
            "track_out_candidate_dominates_base": bool(candidate_dominates),
            "track_out_base_dominates_candidate": bool(base_dominates),
        })
        scored_tracks.append(scored)
        ledger.append({
            "scene_name": scene_name,
            "track_id": int(track_id),
            "mask_changed_from_v0": bool(changed),
            "excluded_frame_indices": excluded_frames,
            "quality": float(quality),
            "quality_mode": quality_mode,
            "candidate_dominates_base": bool(candidate_dominates),
            "base_dominates_candidate": bool(base_dominates),
            **consistency,
        })

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    staging_root.mkdir(parents=True)
    scene_summary = {
        "scene_name": scene_name,
        "track_count": len(scored_tracks),
        "changed_candidate_count": changed_candidate_count,
        "fallback_count": fallback_count,
        "candidate_dominates_count": candidate_dominates_count,
        "base_dominates_count": base_dominates_count,
        "selected_frame_count": len(selected_frames),
    }
    output_payload = dict(candidate_payload)
    output_payload.update(scene_summary)
    output_payload["tracks"] = scored_tracks
    (staging_root / "automatic_tracks.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (staging_root / "track_out_holdout_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging_root, published_root)
    del world, raw_visibility, visibility, processed
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
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-superpoint-visible-ratio", type=float, default=0.10)
    parser.add_argument("--min-initial-mask-support", type=float, default=0.30)
    parser.add_argument("--min-visible-points-per-superpoint", type=int, default=3)
    parser.add_argument("--support-siou", type=float, default=0.30)
    parser.add_argument("--min-reliable-holdout-frames", type=int, default=2)
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
    if args.min_reliable_holdout_frames <= 0:
        raise SystemExit("--min-reliable-holdout-frames 必须为正数")
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
        existing = args.output_root / scene_name / "automatic_tracks.json"
        if existing.is_file() and args.resume:
            payload = json.loads(existing.read_text())
            summary = {
                key: payload[key]
                for key in (
                    "scene_name",
                    "track_count",
                    "changed_candidate_count",
                    "fallback_count",
                    "candidate_dominates_count",
                    "base_dominates_count",
                    "selected_frame_count",
                )
            }
        else:
            summary = _score_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['track_count']} 条，{summary['fallback_count']} 条回退旧分数",
            flush=True,
        )

    root_summary = {
        "gt_usage": "不读取GT；不使用类别、语义或native候选。",
        "decision_state": "固定v1.2 mask的轨迹外共同留出视角质量校准。",
        "scene_count": len(summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "changed_candidate_count": sum(
            item["changed_candidate_count"] for item in summaries
        ),
        "fallback_count": sum(item["fallback_count"] for item in summaries),
        "candidate_dominates_count": sum(
            item["candidate_dominates_count"] for item in summaries
        ),
        "base_dominates_count": sum(
            item["base_dominates_count"] for item in summaries
        ),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "track_out_holdout_summary.json").write_text(
        json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(root_summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

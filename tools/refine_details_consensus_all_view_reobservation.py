#!/usr/bin/env python3
"""以固定共识核心在全部自动 SAM 帧中重观测，并重新计算 superpoint 共识。

输入是 Details 帧级 sIoU 轨迹、共识 v0 和同一批独立自动 SAM 观测。对每个非空
共识核心，在原轨迹未覆盖的 uniform 帧中寻找未被其他有效轨迹占用的自动观测；
只有轨迹和观测互为最佳匹配且共同可见 sIoU 达标时才加入。随后沿用 v0 的共识
定义重新形成并行变体。工具不读取 GT、类别、native 或语义，不覆盖任何输入。
"""

import argparse
import json
import math
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

from refine_details_automatic_tracks_consensus import (
    _superpoint_points,
    consensus_superpoints,
    initial_candidate_superpoints,
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复场景")
    return scenes


def _node_quality(predicted_iou, stability_score, point_count):
    size = min(1.0, math.log1p(point_count) / math.log(2001.0))
    return float(max(0.0, predicted_iou) * max(0.0, stability_score) * (0.40 + 0.60 * size))


def _load_observations(scene_root, superpoints, superpoint_sizes, visibility, args):
    raw_rows = [json.loads(line) for line in (scene_root / "automatic_observations.jsonl").read_text().splitlines() if line.strip()]
    frame_indices = sorted({int(raw["frame_index"]) for raw in raw_rows})
    visible_counts_by_frame = {}
    for frame_index in frame_indices:
        ids, counts = np.unique(superpoints[np.flatnonzero(visibility[frame_index])], return_counts=True)
        visible_counts_by_frame[frame_index] = {
            int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)
        }

    observations = {}
    by_frame = defaultdict(list)
    for raw in raw_rows:
        frame_index = int(raw["frame_index"])
        points = np.unique(np.asarray(np.load(raw["point_indices_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < len(superpoints))]
        ids, counts = np.unique(superpoints[points], return_counts=True)
        visible_counts = visible_counts_by_frame[frame_index]
        inside_counts = {int(superpoint_id): int(count) for superpoint_id, count in zip(ids, counts)}
        lifted = []
        for superpoint_id, inside_count in inside_counts.items():
            visible_count = int(visible_counts.get(superpoint_id, 0))
            total_count = int(superpoint_sizes.get(superpoint_id, 0))
            if visible_count <= 0 or total_count <= 0:
                continue
            if visible_count / total_count < float(args.min_superpoint_visible_ratio):
                continue
            if inside_count / visible_count >= float(args.min_initial_mask_support):
                lifted.append(int(superpoint_id))
        observation_id = int(raw["observation_id"])
        row = {
            "observation_id": observation_id,
            "frame_id": str(raw["frame_id"]),
            "frame_index": frame_index,
            "inside_counts": inside_counts,
            "visible_counts": visible_counts,
            "lifted_superpoints": np.asarray(sorted(lifted), dtype=np.int64),
            "quality": _node_quality(raw["predicted_iou"], raw["stability_score"], len(points)),
        }
        observations[observation_id] = row
        by_frame[frame_index].append(row)
    return observations, by_frame


def frame_match_metrics(core_superpoints, lifted_superpoints, visible_counts, min_visible_points):
    """在单帧可见域比较固定三维核心和独立自动 SAM 观测。"""
    if isinstance(core_superpoints, set):
        core_superpoints = sorted(core_superpoints)
    if isinstance(lifted_superpoints, set):
        lifted_superpoints = sorted(lifted_superpoints)
    visible_domain = np.asarray(
        sorted(int(item) for item, count in visible_counts.items() if int(count) >= int(min_visible_points)),
        dtype=np.int64,
    )
    if len(visible_domain) == 0:
        return {"siou": 0.0, "core_coverage": 0.0, "observation_purity": 0.0}
    core = np.intersect1d(np.unique(core_superpoints), visible_domain, assume_unique=True)
    lifted = np.intersect1d(np.unique(lifted_superpoints), visible_domain, assume_unique=True)
    if len(core) == 0 or len(lifted) == 0:
        return {"siou": 0.0, "core_coverage": 0.0, "observation_purity": 0.0}
    intersection = int(np.intersect1d(core, lifted, assume_unique=True).size)
    union = int(len(core) + len(lifted) - intersection)
    return {
        "siou": float(intersection / max(1, union)),
        "core_coverage": float(intersection / len(core)),
        "observation_purity": float(intersection / len(lifted)),
    }


def reciprocal_frame_matches(track_states, frame_observations, owned_observation_ids, min_siou, min_visible_points):
    """返回一帧内轨迹核心与未占用观测的互为最佳匹配。"""
    pairs = []
    for track_id, state in sorted(track_states.items()):
        if int(state["frame_index"]) in state["member_frame_indices"]:
            continue
        for observation in frame_observations:
            observation_id = int(observation["observation_id"])
            if observation_id in owned_observation_ids:
                continue
            metrics = frame_match_metrics(
                state["core_superpoints"],
                observation["lifted_superpoints"],
                observation["visible_counts"],
                min_visible_points,
            )
            if metrics["siou"] < float(min_siou):
                continue
            pairs.append({
                "track_id": int(track_id),
                "observation_id": observation_id,
                "siou": float(metrics["siou"]),
                "core_coverage": float(metrics["core_coverage"]),
                "observation_purity": float(metrics["observation_purity"]),
                "observation_quality": float(observation["quality"]),
            })

    def rank(pair, other_id):
        return (
            pair["siou"], pair["core_coverage"], pair["observation_purity"],
            pair["observation_quality"], -int(pair[other_id]),
        )

    best_by_track = {}
    best_by_observation = {}
    for pair in pairs:
        track_id, observation_id = pair["track_id"], pair["observation_id"]
        if track_id not in best_by_track or rank(pair, "observation_id") > rank(best_by_track[track_id], "observation_id"):
            best_by_track[track_id] = pair
        if observation_id not in best_by_observation or rank(pair, "track_id") > rank(best_by_observation[observation_id], "track_id"):
            best_by_observation[observation_id] = pair
    return [
        pair for track_id, pair in sorted(best_by_track.items())
        if best_by_observation[pair["observation_id"]]["track_id"] == track_id
    ]


def all_view_consistency(superpoint_ids, observations_by_frame, min_visible_points, support_siou):
    """以全部可见帧的最佳自动观测评估固定mask；无匹配帧显式保留零分。"""
    frame_scores = []
    for _, observations in sorted(observations_by_frame.items()):
        if not observations:
            continue
        visible_counts = observations[0]["visible_counts"]
        visible_core = [
            int(item) for item in superpoint_ids
            if int(visible_counts.get(int(item), 0)) >= int(min_visible_points)
        ]
        if not visible_core:
            continue
        best = max(
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
        frame_scores.append(float(best))
    mean_score = float(np.mean(frame_scores)) if frame_scores else 0.0
    support_rate = float(np.mean(np.asarray(frame_scores) >= float(support_siou))) if frame_scores else 0.0
    return {
        "visible_frame_count": int(len(frame_scores)),
        "mean_best_siou": mean_score,
        "support_frame_rate": support_rate,
    }


def proposal_dominates_base(base_consistency, proposal_consistency, tolerance=1e-12):
    """仅在两项全视角证据均不差且至少一项严格更好时接受几何提案。"""
    fields = ("mean_best_siou", "support_frame_rate")
    no_worse = all(
        float(proposal_consistency[field]) + tolerance >= float(base_consistency[field])
        for field in fields
    )
    strictly_better = any(
        float(proposal_consistency[field]) > float(base_consistency[field]) + tolerance
        for field in fields
    )
    return bool(no_worse and strictly_better)


def _all_view_matches(source_tracks, consensus_tracks, observations_by_frame, args):
    source_by_id = {int(track["track_id"]): track for track in source_tracks}
    owned = {
        int(observation_id)
        for track in source_tracks
        for observation_id in track.get("observation_ids", [])
    }
    owner_by_observation = {
        int(observation_id): int(track["track_id"])
        for track in source_tracks
        for observation_id in track.get("observation_ids", [])
    }
    states = {}
    for track in consensus_tracks:
        source_id = int(track.get("source_track_id", track["track_id"]))
        source = source_by_id.get(source_id)
        core = np.asarray(track.get("superpoint_ids", []), dtype=np.int64)
        if source is None or len(core) == 0:
            continue
        states[source_id] = {
            "core_superpoints": core,
            "member_frame_indices": {
                int(frame_index) for frame_index in source.get("_member_frame_indices", [])
            },
        }

    accepted = defaultdict(list)
    for frame_index, frame_observations in sorted(observations_by_frame.items()):
        frame_states = {
            track_id: {
                **state,
                "frame_index": int(frame_index),
            }
            for track_id, state in states.items()
        }
        blocked_observations = owned if args.unowned_only else set()
        for pair in reciprocal_frame_matches(
            frame_states,
            frame_observations,
            blocked_observations,
            args.min_reobservation_siou,
            args.min_visible_points_per_superpoint,
        ):
            pair["source_owner_track_id"] = int(owner_by_observation.get(pair["observation_id"], -1))
            accepted[int(pair["track_id"])].append(pair)
    return accepted


def _refine_scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    points_by_superpoint = _superpoint_points(superpoints)

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations, observations_by_frame = _load_observations(
        args.automatic_root / scene_name,
        superpoints,
        superpoint_sizes,
        visibility,
        args,
    )
    source_payload = json.loads((args.details_track_root / scene_name / "automatic_tracks.json").read_text())
    consensus_payload = json.loads((args.consensus_track_root / scene_name / "automatic_tracks.json").read_text())
    source_tracks = source_payload.get("tracks", [])
    for track in source_tracks:
        track["_member_frame_indices"] = [
            observations[int(observation_id)]["frame_index"]
            for observation_id in track.get("observation_ids", [])
            if int(observation_id) in observations
        ]
    consensus_tracks = consensus_payload.get("tracks", [])
    source_by_id = {int(track["track_id"]): track for track in source_tracks}
    matches = _all_view_matches(source_tracks, consensus_tracks, observations_by_frame, args)

    staging_root = args.output_root / f".{scene_name}.writing"
    published_root = args.output_root / scene_name
    if staging_root.exists() or published_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published_root}")
    staging_root.mkdir(parents=True)
    point_root = staging_root / "track_points"
    point_root.mkdir()

    records = []
    empty_count = 0
    added_observation_count = 0
    changed_mask_count = 0
    proposal_changed_mask_count = 0
    accepted_proposal_count = 0
    for base in consensus_tracks:
        source_id = int(base.get("source_track_id", base["track_id"]))
        source = source_by_id[source_id]
        extra_pairs = matches.get(source_id, [])
        observation_ids = list(dict.fromkeys(
            [int(item) for item in source.get("observation_ids", [])]
            + [int(pair["observation_id"]) for pair in extra_pairs]
        ))
        frame_rows = [observations[item] for item in observation_ids if item in observations]
        initial = initial_candidate_superpoints(
            frame_rows,
            superpoint_sizes,
            args.min_superpoint_visible_ratio,
            args.min_initial_mask_support,
        )
        proposal, diagnostics = consensus_superpoints(
            initial,
            frame_rows,
            args.min_visible_points_per_superpoint,
            args.frame_superpoint_coverage,
            args.min_support_frames,
            args.min_consensus_rate,
            args.mean_superpoint_coverage,
        )
        previous = set(int(item) for item in base.get("superpoint_ids", []))
        if not proposal:
            empty_count += 1
            proposal = previous
        proposal_changed_mask_count += int(previous != proposal)
        base_consistency = all_view_consistency(
            previous,
            observations_by_frame,
            args.min_visible_points_per_superpoint,
            args.min_reobservation_siou,
        )
        proposal_consistency = all_view_consistency(
            proposal,
            observations_by_frame,
            args.min_visible_points_per_superpoint,
            args.min_reobservation_siou,
        )
        accept_proposal = previous != proposal and proposal_dominates_base(
            base_consistency,
            proposal_consistency,
        )
        kept = proposal if accept_proposal else previous
        consistency = proposal_consistency if accept_proposal else base_consistency
        accepted_proposal_count += int(accept_proposal)
        changed_mask_count += int(previous != kept)
        source_track_id = int(source["track_id"])
        filename = f"track{source_track_id:04d}_points.npz"
        points = np.concatenate([points_by_superpoint[item] for item in sorted(kept)]).astype(np.int64, copy=False)
        np.savez_compressed(point_root / filename, point_indices=points)
        added_observation_count += len(extra_pairs)
        consensus_values = [diagnostics[item]["consensus_rate"] for item in kept if item in diagnostics]
        coverage_values = [diagnostics[item]["mean_supported_coverage"] for item in kept if item in diagnostics]
        support_values = [diagnostics[item]["support_frames"] for item in kept if item in diagnostics]
        source_quality = max(0.0, float(base.get("mean_node_quality", 0.0)))
        consensus_quality = float(math.sqrt(source_quality * consistency["mean_best_siou"]))
        record = dict(base)
        record.update({
            "track_id": source_track_id,
            "source_track_id": source_track_id,
            "observation_ids": observation_ids,
            "frame_ids": [observations[item]["frame_id"] for item in observation_ids],
            "support_view_count": len({observations[item]["frame_index"] for item in observation_ids}),
            "point_count": int(len(points)),
            "points_path": str(args.output_root / scene_name / "track_points" / filename),
            "superpoint_ids": sorted(int(item) for item in kept),
            "superpoint_count": int(len(kept)),
            "mean_consensus_rate": float(np.mean(consensus_values)) if consensus_values else float(base.get("mean_consensus_rate", 0.0)),
            "mean_supported_coverage": float(np.mean(coverage_values)) if coverage_values else float(base.get("mean_supported_coverage", 0.0)),
            "support_score": float(np.sum(support_values)) if support_values else float(base.get("support_score", 0.0)),
            "reobservation_count": int(len(extra_pairs)),
            "mean_reobservation_siou": float(np.mean([item["siou"] for item in extra_pairs])) if extra_pairs else 0.0,
            "reobservation_pairs": extra_pairs,
            "all_view_visible_frame_count": consistency["visible_frame_count"],
            "all_view_mean_best_siou": consistency["mean_best_siou"],
            "all_view_support_frame_rate": consistency["support_frame_rate"],
            "consensus_quality": consensus_quality,
            "proposal_changed_mask": bool(previous != proposal),
            "proposal_accepted": bool(accept_proposal),
            "proposal_superpoint_count": int(len(proposal)),
            "base_all_view_mean_best_siou": base_consistency["mean_best_siou"],
            "base_all_view_support_frame_rate": base_consistency["support_frame_rate"],
            "proposal_all_view_mean_best_siou": proposal_consistency["mean_best_siou"],
            "proposal_all_view_support_frame_rate": proposal_consistency["support_frame_rate"],
        })
        records.append(record)

    payload = {
        "scene_name": scene_name,
        "source_consensus_track_count": len(consensus_tracks),
        "track_count": len(records),
        "empty_after_reobservation_consensus_count": empty_count,
        "added_reobservation_count": added_observation_count,
        "changed_mask_count": changed_mask_count,
        "proposal_changed_mask_count": proposal_changed_mask_count,
        "accepted_proposal_count": accepted_proposal_count,
        "tracks": records,
    }
    (staging_root / "automatic_tracks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(staging_root, published_root)
    del world, raw_visibility, visibility, processed
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--details-track-root", type=Path, required=True)
    parser.add_argument("--consensus-track-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--unowned-only",
        action="store_true",
        help="只允许未进入任何有效Details轨迹的观测；默认保留跨轨迹双假设，不从原轨迹移走观测。",
    )
    parser.add_argument("--min-superpoint-visible-ratio", type=float, default=0.10)
    parser.add_argument("--min-initial-mask-support", type=float, default=0.30)
    parser.add_argument("--min-reobservation-siou", type=float, default=0.30)
    parser.add_argument("--frame-superpoint-coverage", type=float, default=0.50)
    parser.add_argument("--mean-superpoint-coverage", type=float, default=0.55)
    parser.add_argument("--min-visible-points-per-superpoint", type=int, default=3)
    parser.add_argument("--min-support-frames", type=int, default=2)
    parser.add_argument("--min-consensus-rate", type=float, default=0.30)
    args = parser.parse_args()
    for name in (
        "scene_list", "details_track_root", "consensus_track_root", "automatic_root",
        "processed_scene_root", "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
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
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
        else:
            summary = _refine_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"重观测 {summary['added_reobservation_count']}，改变 {summary['changed_mask_count']} 条mask",
            flush=True,
        )
    payload = {
        "gt_usage": "不读取GT；不使用类别、语义或native候选。",
        "decision_state": "固定共识v0核心引导的全自动SAM可见帧互为最佳重观测变体。",
        "scene_count": len(summaries),
        "source_consensus_track_count": sum(item["source_consensus_track_count"] for item in summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "added_reobservation_count": sum(item["added_reobservation_count"] for item in summaries),
        "changed_mask_count": sum(item["changed_mask_count"] for item in summaries),
        "proposal_changed_mask_count": sum(item["proposal_changed_mask_count"] for item in summaries),
        "accepted_proposal_count": sum(item["accepted_proposal_count"] for item in summaries),
        "empty_after_reobservation_consensus_count": sum(item["empty_after_reobservation_consensus_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_track_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

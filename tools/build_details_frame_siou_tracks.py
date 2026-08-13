#!/usr/bin/env python3
"""以 Details Matter 的共同可见 superpoint 帧级 sIoU 关联自动 SAM 观测。

这是一个严格受控的关联替换实验：输入仍是既有的类别无关 SAM 自动观测和原始
ScanNet superpoint；不做帧内重叠删除、不做轨迹合并/细化/删除、不读取 GT，也不
产生三维候选或 AP。唯一改变是跨帧观测如何组成轨迹。
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

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


def _node_quality(predicted_iou, stability_score, point_count):
    size = min(1.0, math.log1p(point_count) / math.log(2001.0))
    return float(max(0.0, predicted_iou) * max(0.0, stability_score) * (0.40 + 0.60 * size))


def framewise_siou(left_superpoints, right_superpoints, jointly_visible_superpoints):
    """Details 式 sIoU：仅在两帧共同可见的 superpoint 域内比较。"""
    domain = np.unique(np.asarray(jointly_visible_superpoints, dtype=np.int64))
    if len(domain) == 0:
        return 0.0
    left = np.intersect1d(np.unique(left_superpoints), domain, assume_unique=True)
    right = np.intersect1d(np.unique(right_superpoints), domain, assume_unique=True)
    union = np.union1d(left, right)
    if len(union) == 0:
        return 0.0
    return float(len(np.intersect1d(left, right, assume_unique=True)) / len(union))


def _frame_visibility_superpoints(visibility, superpoints, frame_indices, min_visible_ratio):
    """计算每个已用帧中可见比例足够的原始 superpoint 集合。"""
    ids, total_counts = np.unique(superpoints, return_counts=True)
    total = {int(sp): int(count) for sp, count in zip(ids, total_counts)}
    result = {}
    for frame_index in sorted(set(int(item) for item in frame_indices)):
        visible_points = np.flatnonzero(visibility[frame_index])
        visible_ids, visible_counts = np.unique(superpoints[visible_points], return_counts=True)
        kept = [
            int(sp) for sp, count in zip(visible_ids, visible_counts)
            if int(count) / max(1, total[int(sp)]) >= min_visible_ratio
        ]
        result[frame_index] = np.asarray(kept, dtype=np.int64)
    return result


def _load_nodes(scene_name, automatic_root, superpoints, frame_visible_superpoints, min_mask_support):
    path = automatic_root / scene_name / "automatic_observations.jsonl"
    nodes = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            points = np.unique(np.asarray(np.load(raw["point_indices_path"])["point_indices"], dtype=np.int64))
            points = points[(points >= 0) & (points < len(superpoints))]
            if len(points) == 0:
                continue
            frame_index = int(raw["frame_index"])
            visible_sp = frame_visible_superpoints.get(frame_index, np.empty(0, dtype=np.int64))
            if len(visible_sp) == 0:
                continue
            point_sp, support_counts = np.unique(superpoints[points], return_counts=True)
            # 每个 observation 的 mask support 使用该帧已可见的该 SP 点作分母。
            # frame_visible_superpoints 已滤过可见比例，故从 visibility 重新取分母。
            lifted = []
            for sp, count in zip(point_sp, support_counts):
                position = np.searchsorted(visible_sp, sp)
                if position >= len(visible_sp) or visible_sp[position] != sp:
                    continue
                # 这里临时记录分子；分母由调用方补到 raw，避免点级 GT 或候选信息。
                lifted.append((int(sp), int(count)))
            node = {
                "node_id": len(nodes),
                "observation_id": int(raw["observation_id"]),
                "scene_name": scene_name,
                "frame_id": str(raw["frame_id"]),
                "frame_index": frame_index,
                "point_count": int(len(points)),
                "predicted_iou": float(raw["predicted_iou"]),
                "stability_score": float(raw["stability_score"]),
                "points_path": str(raw["point_indices_path"]),
                "points": points,
                "_support_counts": lifted,
            }
            node["quality"] = _node_quality(node["predicted_iou"], node["stability_score"], len(points))
            nodes.append(node)
    return nodes


def _assign_lifted_superpoints(nodes, visibility, superpoints, frame_visible_superpoints, min_mask_support):
    """按 Details Matter 的可见性 r 与 mask support c 生成每帧观测的 SP 集。"""
    denominators = {}
    for frame_index, visible_sp in frame_visible_superpoints.items():
        visible_points = np.flatnonzero(visibility[frame_index])
        ids, counts = np.unique(superpoints[visible_points], return_counts=True)
        count_map = {int(sp): int(count) for sp, count in zip(ids, counts)}
        denominators[frame_index] = count_map
    for node in nodes:
        denom = denominators[node["frame_index"]]
        kept = [
            sp for sp, numerator in node.pop("_support_counts")
            if numerator / max(1, denom.get(sp, 0)) >= min_mask_support
        ]
        node["lifted_superpoints"] = np.asarray(sorted(kept), dtype=np.int64)


def _best_track_score(node, track, frame_visible_superpoints):
    best = 0.0
    for prior in track["nodes"]:
        jointly_visible = np.intersect1d(
            frame_visible_superpoints[node["frame_index"]],
            frame_visible_superpoints[prior["frame_index"]],
            assume_unique=True,
        )
        score = framewise_siou(node["lifted_superpoints"], prior["lifted_superpoints"], jointly_visible)
        best = max(best, score)
    return best


def associate_sequentially(nodes, frame_visible_superpoints, min_tracking_siou, min_track_views):
    """按时间顺序关联；同一帧对同一轨迹最多保留一个观测，避免帧内重复吸附。"""
    by_frame = defaultdict(list)
    for node in nodes:
        by_frame[node["frame_index"]].append(node)
    tracks = []
    for frame_index in sorted(by_frame):
        frame_nodes = sorted(by_frame[frame_index], key=lambda item: item["observation_id"])
        pairs = []
        for node in frame_nodes:
            for track_index, track in enumerate(tracks):
                score = _best_track_score(node, track, frame_visible_superpoints)
                if score >= min_tracking_siou:
                    pairs.append((score, -node["quality"], node["observation_id"], track_index, node))
        used_nodes, used_tracks = set(), set()
        for score, _, observation_id, track_index, node in sorted(pairs, key=lambda item: (-item[0], item[1], item[2], item[3])):
            if observation_id in used_nodes or track_index in used_tracks:
                continue
            tracks[track_index]["nodes"].append(node)
            tracks[track_index]["edge_scores"].append(float(score))
            used_nodes.add(observation_id)
            used_tracks.add(track_index)
        for node in frame_nodes:
            if node["observation_id"] not in used_nodes:
                tracks.append({"nodes": [node], "edge_scores": []})
    return [
        track for track in tracks
        if len({node["frame_index"] for node in track["nodes"]}) >= min_track_views
    ]


def _write_scene(scene_name, nodes, tracks, output_root):
    root = output_root / scene_name
    staging_root = output_root / f".{scene_name}.writing"
    if root.exists() or staging_root.exists():
        raise FileExistsError(f"输出或临时目录已存在：{root}")
    staging_root.mkdir(parents=True, exist_ok=False)
    root = staging_root
    point_root = root / "track_points"
    point_root.mkdir()
    records = []
    for track_id, track in enumerate(tracks):
        points = np.unique(np.concatenate([node["points"] for node in track["nodes"]])).astype(np.int64)
        points_path = point_root / f"track{track_id:04d}_points.npz"
        np.savez_compressed(points_path, point_indices=points)
        published_points_path = output_root / scene_name / "track_points" / points_path.name
        records.append({
            "track_id": track_id,
            "node_ids": [int(node["node_id"]) for node in track["nodes"]],
            "observation_ids": [int(node["observation_id"]) for node in track["nodes"]],
            "frame_ids": [str(node["frame_id"]) for node in track["nodes"]],
            "support_view_count": len({node["frame_index"] for node in track["nodes"]}),
            "point_count": int(len(points)),
            "mean_node_quality": float(np.mean([node["quality"] for node in track["nodes"]])),
            "mean_predicted_iou": float(np.mean([node["predicted_iou"] for node in track["nodes"]])),
            "mean_stability_score": float(np.mean([node["stability_score"] for node in track["nodes"]])),
            "mean_edge_score": float(np.mean(track["edge_scores"])) if track["edge_scores"] else 0.0,
            "points_path": str(published_points_path),
        })
    payload = {
        "scene_name": scene_name,
        "node_count": len(nodes),
        "track_count": len(records),
        "tracks": records,
    }
    (root / "automatic_tracks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(staging_root, output_root / scene_name)
    return payload


def _build_scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    if processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    # 只按自动观测已经使用的帧关联，绝不借用未导出的帧。
    observation_path = args.automatic_root / scene_name / "automatic_observations.jsonl"
    used_frames = {int(json.loads(line)["frame_index"]) for line in observation_path.read_text().splitlines() if line.strip()}
    visible_sp = _frame_visibility_superpoints(visibility, superpoints, used_frames, args.min_superpoint_visible_ratio)
    nodes = _load_nodes(scene_name, args.automatic_root, superpoints, visible_sp, args.min_superpoint_mask_support)
    _assign_lifted_superpoints(nodes, visibility, superpoints, visible_sp, args.min_superpoint_mask_support)
    tracks = associate_sequentially(nodes, visible_sp, args.min_tracking_siou, args.min_track_views)
    del world, raw_visibility, visibility, processed
    return _write_scene(scene_name, nodes, tracks, args.output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--resume", action="store_true", help="只跳过已有完整场景，用于中断后的安全续跑。")
    parser.add_argument("--min_superpoint_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_superpoint_mask_support", type=float, default=0.30)
    parser.add_argument("--min_tracking_siou", type=float, default=0.30)
    parser.add_argument("--min_track_views", type=int, default=2)
    args = parser.parse_args()
    for name in ("scene_list", "automatic_root", "processed_scene_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    for name in ("min_superpoint_visible_ratio", "min_superpoint_mask_support", "min_tracking_siou"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise SystemExit(f"--{name} 必须在零到一之间。")
    if args.min_track_views < 2:
        raise SystemExit("轨迹至少需要两个视角。")
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
        if existing.is_file():
            if not args.resume:
                raise SystemExit(f"输出场景已存在：{scene_name}")
            summary = json.loads(existing.read_text())
            summaries.append(summary)
            print(f"[跳过已有] {index}/{len(scenes)} {scene_name}: 轨迹 {summary['track_count']}", flush=True)
            continue
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: 节点 {summary['node_count']}，轨迹 {summary['track_count']}", flush=True)
    payload = {
        "gt_usage": "不读取 GT；输出只供后续 GT-only 轨迹归因。",
        "decision_state": "仅替换跨帧关联；不做帧内去重、合并、细化、删除、候选、融合或评测。",
        "scene_count": len(summaries),
        "node_count": sum(item["node_count"] for item in summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_track_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({key: payload[key] for key in ("scene_count", "node_count", "track_count", "decision_state")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

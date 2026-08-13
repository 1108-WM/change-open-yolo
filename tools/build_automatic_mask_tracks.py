#!/usr/bin/env python3
"""从类别无关 SAM 自动 mask 三维点构建无 GT 多视角轨迹。

该脚本只使用自动 mask 回投点、SAM 质量、空间关系和原始 superpoint。输出是待诊断的
类别无关轨迹，不赋类别、不生成最终候选、不融合、不评测。
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _point_iou(left, right):
    if len(left) == 0 or len(right) == 0:
        return 0.0, 0
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    return float(intersection / max(1, len(left) + len(right) - intersection)), intersection


def _node_quality(predicted_iou, stability_score, point_count):
    size = min(1.0, math.log1p(point_count) / math.log(2001.0))
    return float(max(0.0, predicted_iou) * max(0.0, stability_score) * (0.40 + 0.60 * size))


def _load_nodes(scene_name, automatic_root, points_xyz, superpoints):
    path = automatic_root / scene_name / "automatic_observations.jsonl"
    nodes = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            points = np.unique(np.asarray(np.load(raw["point_indices_path"])["point_indices"], dtype=np.int64))
            points = points[(points >= 0) & (points < len(points_xyz))]
            if len(points) == 0:
                continue
            local = points_xyz[points]
            node = {
                "node_id": len(nodes),
                "observation_id": int(raw["observation_id"]),
                "scene_name": scene_name,
                "frame_id": str(raw["frame_id"]),
                "frame_index": int(raw["frame_index"]),
                "point_count": int(len(points)),
                "predicted_iou": float(raw["predicted_iou"]),
                "stability_score": float(raw["stability_score"]),
                "bbox_xywh": list(raw["bbox_xywh"]),
                "points_path": str(raw["point_indices_path"]),
                "points": points,
                "centroid": local.mean(axis=0).astype(np.float32),
                "bbox_min": local.min(axis=0).astype(np.float32),
                "bbox_max": local.max(axis=0).astype(np.float32),
                "superpoint_ids": np.unique(superpoints[points]).astype(np.int64),
            }
            node["quality"] = _node_quality(node["predicted_iou"], node["stability_score"], len(points))
            nodes.append(node)
    return nodes


def _candidate_pairs(nodes, knn, max_centroid_distance, max_nodes_per_point):
    pairs = set()
    if len(nodes) < 2:
        return []
    centroids = np.stack([node["centroid"] for node in nodes])
    tree = cKDTree(centroids)
    count = min(len(nodes), max(2, int(knn) + 1))
    distances, neighbors = tree.query(centroids, k=count)
    distances = np.atleast_2d(distances)
    neighbors = np.atleast_2d(neighbors)
    for left_id in range(len(nodes)):
        for distance, right_id in zip(distances[left_id], neighbors[left_id]):
            right_id = int(right_id)
            if left_id == right_id or distance > max_centroid_distance:
                continue
            if nodes[left_id]["frame_index"] != nodes[right_id]["frame_index"]:
                pairs.add(tuple(sorted((left_id, right_id))))
    point_to_nodes = defaultdict(list)
    for node in nodes:
        for point_id in node["points"]:
            bucket = point_to_nodes[int(point_id)]
            if len(bucket) < max_nodes_per_point:
                bucket.append(node["node_id"])
    for shared_nodes in point_to_nodes.values():
        for offset, left_id in enumerate(shared_nodes):
            for right_id in shared_nodes[offset + 1:]:
                if nodes[left_id]["frame_index"] != nodes[right_id]["frame_index"]:
                    pairs.add(tuple(sorted((left_id, right_id))))
    return sorted(pairs)


def _shared_superpoints(left, right):
    return int(np.intersect1d(left["superpoint_ids"], right["superpoint_ids"], assume_unique=True).size)


def _build_edge(left, right, args):
    point_iou, intersection = _point_iou(left["points"], right["points"])
    distance = float(np.linalg.norm(left["centroid"] - right["centroid"]))
    shared_superpoints = _shared_superpoints(left, right)
    direct = point_iou >= args.min_direct_point_iou and intersection >= args.min_shared_points
    superpoint_bridge = shared_superpoints >= args.min_shared_superpoints and distance <= args.max_centroid_distance
    accepted = bool(direct or superpoint_bridge)
    spatial = math.exp(-distance / max(args.max_centroid_distance, 1e-6))
    score = float(
        0.45 * min(1.0, point_iou / max(args.min_direct_point_iou, 1e-6))
        + 0.25 * min(1.0, shared_superpoints / max(1, args.min_shared_superpoints))
        + 0.15 * spatial
        + 0.15 * min(left["quality"], right["quality"])
    )
    return {
        "left_node_id": left["node_id"],
        "right_node_id": right["node_id"],
        "left_frame_id": left["frame_id"],
        "right_frame_id": right["frame_id"],
        "point_iou": point_iou,
        "shared_point_count": intersection,
        "centroid_distance": distance,
        "shared_superpoint_count": shared_superpoints,
        "edge_score": score,
        "accepted": accepted,
    }


def _build_tracks(nodes, edges, min_track_views, min_track_support_edges):
    """种子-验证-扩展，避免类别无关图的弱链传播。"""
    adjacency = defaultdict(list)
    for edge in edges:
        if edge["accepted"]:
            adjacency[edge["left_node_id"]].append(edge)
            adjacency[edge["right_node_id"]].append(edge)
    node_by_id = {node["node_id"]: node for node in nodes}
    assigned = set()
    tracks = []
    for seed in sorted(nodes, key=lambda node: (-node["quality"], node["node_id"])):
        seed_id = seed["node_id"]
        if seed_id in assigned:
            continue
        first = next(
            (
                edge for edge in sorted(adjacency[seed_id], key=lambda item: -item["edge_score"])
                if (edge["right_node_id"] if edge["left_node_id"] == seed_id else edge["left_node_id"]) not in assigned
            ),
            None,
        )
        if first is None:
            continue
        other_id = first["right_node_id"] if first["left_node_id"] == seed_id else first["left_node_id"]
        members = {seed_id, other_id}
        frames = {node_by_id[seed_id]["frame_index"], node_by_id[other_id]["frame_index"]}
        changed = True
        while changed:
            changed = False
            best = None
            candidates = set()
            for member in members:
                for edge in adjacency[member]:
                    candidate = edge["right_node_id"] if edge["left_node_id"] == member else edge["left_node_id"]
                    if candidate not in members and candidate not in assigned and node_by_id[candidate]["frame_index"] not in frames:
                        candidates.add(candidate)
            for candidate in candidates:
                support_edges = [
                    edge for edge in adjacency[candidate]
                    if (edge["right_node_id"] if edge["left_node_id"] == candidate else edge["left_node_id"]) in members
                ]
                required = min(min_track_support_edges, len(members))
                if len(support_edges) < required:
                    continue
                score = float(np.mean(sorted((edge["edge_score"] for edge in support_edges), reverse=True)[:required]))
                if best is None or score > best[0]:
                    best = (score, candidate)
            if best is not None:
                members.add(best[1])
                frames.add(node_by_id[best[1]]["frame_index"])
                changed = True
        if len(frames) >= min_track_views:
            assigned.update(members)
            track_edges = [
                edge for edge in edges
                if edge["accepted"] and edge["left_node_id"] in members and edge["right_node_id"] in members
            ]
            tracks.append({"nodes": [node_by_id[node_id] for node_id in sorted(members)], "edges": track_edges})
    return tracks


def _write_scene(scene_name, nodes, edges, tracks, output_root):
    root = output_root / scene_name
    root.mkdir(parents=True, exist_ok=False)
    track_dir = root / "track_points"
    track_dir.mkdir()
    records = []
    for track_id, track in enumerate(tracks):
        points = np.unique(np.concatenate([node["points"] for node in track["nodes"]])).astype(np.int64)
        points_path = track_dir / f"track{track_id:04d}_points.npz"
        np.savez_compressed(points_path, point_indices=points)
        records.append({
            "track_id": track_id,
            "node_ids": [node["node_id"] for node in track["nodes"]],
            "observation_ids": [node["observation_id"] for node in track["nodes"]],
            "frame_ids": [node["frame_id"] for node in track["nodes"]],
            "support_view_count": len({node["frame_index"] for node in track["nodes"]}),
            "point_count": int(len(points)),
            "mean_node_quality": float(np.mean([node["quality"] for node in track["nodes"]])),
            "mean_predicted_iou": float(np.mean([node["predicted_iou"] for node in track["nodes"]])),
            "mean_stability_score": float(np.mean([node["stability_score"] for node in track["nodes"]])),
            "mean_edge_score": float(np.mean([edge["edge_score"] for edge in track["edges"]])) if track["edges"] else 0.0,
            "points_path": str(points_path),
        })
    summary = {
        "scene_name": scene_name,
        "node_count": len(nodes),
        "candidate_edge_count": len(edges),
        "accepted_edge_count": sum(edge["accepted"] for edge in edges),
        "track_count": len(records),
        "tracks": records,
    }
    (root / "automatic_tracks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def _build_scene(scene_name, args):
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    points_xyz = np.asarray(processed[:, :3], dtype=np.float32)
    if processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    nodes = _load_nodes(scene_name, args.automatic_root, points_xyz, superpoints)
    pairs = _candidate_pairs(nodes, args.knn, args.max_centroid_distance, args.max_nodes_per_point)
    edges = [_build_edge(nodes[left], nodes[right], args) for left, right in pairs]
    tracks = _build_tracks(nodes, edges, args.min_track_views, args.min_track_support_edges)
    return _write_scene(scene_name, nodes, edges, tracks, args.output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--max_centroid_distance", type=float, default=0.40)
    parser.add_argument("--max_nodes_per_point", type=int, default=16)
    parser.add_argument("--min_direct_point_iou", type=float, default=0.08)
    parser.add_argument("--min_shared_points", type=int, default=20)
    parser.add_argument("--min_shared_superpoints", type=int, default=2)
    parser.add_argument("--min_track_views", type=int, default=2)
    parser.add_argument("--min_track_support_edges", type=int, default=2)
    args = parser.parse_args()
    if args.min_track_views < 2:
        raise SystemExit("自动 mask 轨迹至少需要两个视角。")
    for name in ("scene_list", "automatic_root", "processed_scene_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: 节点 {summary['node_count']}，轨迹 {summary['track_count']}", flush=True)
    payload = {
        "gt_usage": "不读取 GT；输出仅供后续 GT-only 轨迹账本读取。",
        "decision_state": "不赋类别、不生成候选、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "node_count": sum(item["node_count"] for item in summaries),
        "accepted_edge_count": sum(item["accepted_edge_count"] for item in summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "params": vars(args),
    }
    (args.output_root / "automatic_track_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({key: payload[key] for key in ("scene_count", "node_count", "accepted_edge_count", "track_count", "decision_state")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

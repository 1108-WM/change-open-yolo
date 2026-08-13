#!/usr/bin/env python3
"""构建不含 GT 的多视角残差证据图和残差轨迹。

节点是单帧 SAM 残差观测，不是已有 3D 候选。边要求同类别、不同帧，并通过
双向重投影确认：一方残差点在另一帧真实可见且落入对方 SAM mask，反向也成立。
三维点重叠、空间邻近和原始 superpoint 关系只用于召回与记录，不能单独形成边。

输出的轨迹是后续 GT-only 诊断对象，不会进入候选融合、评分或 AP 评测。
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_points(path, point_count):
    points = np.asarray(np.load(path)["point_indices"], dtype=np.int64)
    points = points[(points >= 0) & (points < point_count)]
    return np.unique(points)


def _point_iou(left, right):
    if len(left) == 0 or len(right) == 0:
        return 0.0, 0
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    union = len(left) + len(right) - intersection
    return float(intersection / max(1, union)), intersection


def _node_quality(node):
    semantic = max(0.0, min(1.0, float(node["detection_score"])))
    sam = max(0.0, min(1.0, float(node["sam_score"])))
    size = min(1.0, math.log1p(len(node["points"])) / math.log(1001.0))
    novelty = max(0.0, min(1.0, 1.0 - float(node["candidate_seed_coverage"])))
    return float(semantic * sam * (0.45 + 0.55 * size) * (0.50 + 0.50 * novelty))


def _load_scene_nodes(scene_name, residual_root, points_xyz, superpoints, residual_mode):
    mode_fields = {
        "after_any": (
            "residual_after_any_points_path",
            "any_candidate_seed_coverage",
            "best_any_seed_coverage",
            "best_any_candidate_id",
        ),
        "after_same_class": (
            "residual_after_same_class_points_path",
            "same_class_candidate_seed_coverage",
            "best_same_class_seed_coverage",
            "best_same_class_candidate_id",
        ),
    }
    residual_key, coverage_key, best_coverage_key, best_candidate_key = mode_fields[residual_mode]
    rows = []
    path = residual_root / scene_name / "residual_observations.jsonl"
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            residual_path = raw.get(residual_key)
            if not residual_path or not Path(residual_path).is_file():
                continue
            points = _load_points(residual_path, len(points_xyz))
            if len(points) == 0:
                continue
            local_points = points_xyz[points]
            node = {
                "node_id": len(rows),
                "observation_id": int(raw["observation_id"]),
                "scene_name": scene_name,
                "frame_id": str(raw["frame_id"]),
                "frame_index": int(raw["frame_index"]),
                "class_id": int(raw["class_id"]),
                "class_name": str(raw["class_name"]),
                "detection_score": float(raw["detection_score"]),
                "sam_score": float(raw["sam_score"]),
                "source_mask_path": str(raw["source_mask_path"]),
                "residual_mode": residual_mode,
                "residual_points_path": str(residual_path),
                "point_count": int(len(points)),
                "candidate_seed_coverage": float(raw[coverage_key]),
                "best_candidate_seed_coverage": float(raw[best_coverage_key]),
                "best_candidate_id": int(raw[best_candidate_key]),
                "points": points,
                "centroid": local_points.mean(axis=0).astype(np.float32),
                "bbox_min": local_points.min(axis=0).astype(np.float32),
                "bbox_max": local_points.max(axis=0).astype(np.float32),
                "superpoint_ids": (
                    np.unique(superpoints[points]).astype(np.int64) if superpoints is not None else np.asarray([], dtype=np.int64)
                ),
            }
            node["quality"] = _node_quality(node)
            rows.append(node)
    return rows


def _candidate_pairs(nodes, knn, max_centroid_distance, max_nodes_per_point):
    """同类别内用近邻与共享点召回，不做全量两两比较。"""
    pairs = set()
    by_class = defaultdict(list)
    for node in nodes:
        by_class[node["class_id"]].append(node["node_id"])
    for node_ids in by_class.values():
        if len(node_ids) < 2:
            continue
        centroids = np.stack([nodes[index]["centroid"] for index in node_ids])
        tree = cKDTree(centroids)
        neighbor_count = min(len(node_ids), max(2, int(knn) + 1))
        distances, neighbors = tree.query(centroids, k=neighbor_count)
        distances = np.atleast_2d(distances)
        neighbors = np.atleast_2d(neighbors)
        for local_left, left_id in enumerate(node_ids):
            for distance, local_right in zip(distances[local_left], neighbors[local_left]):
                right_id = node_ids[int(local_right)]
                if left_id == right_id or distance > float(max_centroid_distance):
                    continue
                if nodes[left_id]["frame_index"] == nodes[right_id]["frame_index"]:
                    continue
                pairs.add(tuple(sorted((left_id, right_id))))

        point_to_nodes = defaultdict(list)
        for node_id in node_ids:
            for point_id in nodes[node_id]["points"]:
                bucket = point_to_nodes[int(point_id)]
                if len(bucket) < int(max_nodes_per_point):
                    bucket.append(node_id)
        for shared_nodes in point_to_nodes.values():
            for left_offset, left_id in enumerate(shared_nodes):
                for right_id in shared_nodes[left_offset + 1:]:
                    if nodes[left_id]["frame_index"] != nodes[right_id]["frame_index"]:
                        pairs.add(tuple(sorted((left_id, right_id))))
    return sorted(pairs)


class _MaskCache:
    def __init__(self):
        self._items = {}

    def load(self, path):
        if path not in self._items:
            image = imageio.imread(path)
            self._items[path] = np.asarray(image > 0, dtype=bool)
        return self._items[path]


def _project_support(source, target, projections, visibility, scaling_params, mask_cache):
    source_points = source["points"]
    frame_index = int(target["frame_index"])
    visible_points = source_points[visibility[frame_index, source_points]]
    stats = {
        "input_points": int(len(source_points)),
        "visible_points": int(len(visible_points)),
        "inside_points": 0,
        "inside_visible_ratio": 0.0,
    }
    if len(visible_points) == 0:
        return stats
    mask = mask_cache.load(target["source_mask_path"])
    coords = projections[frame_index, visible_points].astype(np.float32)
    xs = np.round(coords[:, 0] / float(scaling_params[1])).astype(np.int64)
    ys = np.round(coords[:, 1] / float(scaling_params[0])).astype(np.int64)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not valid.any():
        return stats
    inside = mask[ys[valid], xs[valid]]
    stats["inside_points"] = int(inside.sum())
    stats["inside_visible_ratio"] = float(inside.sum() / max(1, int(valid.sum())))
    return stats


def _shared_superpoint_count(left, right):
    if len(left["superpoint_ids"]) == 0 or len(right["superpoint_ids"]) == 0:
        return 0
    return int(np.intersect1d(left["superpoint_ids"], right["superpoint_ids"], assume_unique=True).size)


def _build_edge(left, right, projections, visibility, scaling_params, mask_cache, args):
    direct_iou, direct_intersection = _point_iou(left["points"], right["points"])
    centroid_distance = float(np.linalg.norm(left["centroid"] - right["centroid"]))
    left_to_right = _project_support(left, right, projections, visibility, scaling_params, mask_cache)
    right_to_left = _project_support(right, left, projections, visibility, scaling_params, mask_cache)
    mutual_support = min(
        float(left_to_right["inside_visible_ratio"]),
        float(right_to_left["inside_visible_ratio"]),
    )
    enough_visible = (
        left_to_right["visible_points"] >= args.min_projection_visible_points
        and right_to_left["visible_points"] >= args.min_projection_visible_points
    )
    accepted = bool(
        enough_visible
        and mutual_support >= args.min_mutual_projection_support
        and (
            direct_iou >= args.min_direct_point_iou
            or centroid_distance <= args.max_centroid_distance
        )
    )
    spatial_score = math.exp(-centroid_distance / max(1e-6, args.max_centroid_distance))
    score = float(
        0.40 * mutual_support
        + 0.25 * min(1.0, direct_iou / max(1e-6, args.min_direct_point_iou))
        + 0.20 * spatial_score
        + 0.15 * min(left["quality"], right["quality"])
    )
    return {
        "left_node_id": left["node_id"],
        "right_node_id": right["node_id"],
        "class_name": left["class_name"],
        "left_frame_id": left["frame_id"],
        "right_frame_id": right["frame_id"],
        "direct_point_intersection": direct_intersection,
        "centroid_distance": centroid_distance,
        "shared_superpoint_count": _shared_superpoint_count(left, right),
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "mutual_projection_support": mutual_support,
        "enough_visible_points": enough_visible,
        "edge_score": score,
        "accepted": accepted,
    }


def _build_tracks(nodes, edges, min_track_views, min_track_support_edges):
    """用保守的种子-验证-扩展替代连通分量直接合并。"""
    adjacency = defaultdict(list)
    for edge in edges:
        if edge["accepted"]:
            adjacency[edge["left_node_id"]].append(edge)
            adjacency[edge["right_node_id"]].append(edge)
    node_by_id = {node["node_id"]: node for node in nodes}
    assigned = set()
    tracks = []
    for seed in sorted(nodes, key=lambda item: (-item["quality"], item["node_id"])):
        seed_id = seed["node_id"]
        if seed_id in assigned:
            continue
        seed_edges = sorted(adjacency[seed_id], key=lambda item: -item["edge_score"])
        first_edge = next(
            (
                edge for edge in seed_edges
                if (edge["right_node_id"] if edge["left_node_id"] == seed_id else edge["left_node_id"]) not in assigned
            ),
            None,
        )
        if first_edge is None:
            continue
        other_id = first_edge["right_node_id"] if first_edge["left_node_id"] == seed_id else first_edge["left_node_id"]
        member_ids = {seed_id, other_id}
        frame_ids = {node_by_id[seed_id]["frame_index"], node_by_id[other_id]["frame_index"]}
        changed = True
        while changed:
            changed = False
            candidates = []
            for member_id in list(member_ids):
                for edge in adjacency[member_id]:
                    candidate_id = edge["right_node_id"] if edge["left_node_id"] == member_id else edge["left_node_id"]
                    if candidate_id in assigned or candidate_id in member_ids:
                        continue
                    if node_by_id[candidate_id]["frame_index"] in frame_ids:
                        continue
                    candidates.append(candidate_id)
            best = None
            for candidate_id in set(candidates):
                support_edges = [
                    edge for edge in adjacency[candidate_id]
                    if (edge["right_node_id"] if edge["left_node_id"] == candidate_id else edge["left_node_id"]) in member_ids
                ]
                required = min(int(min_track_support_edges), len(member_ids))
                if len(support_edges) < required:
                    continue
                score = float(np.mean(sorted((edge["edge_score"] for edge in support_edges), reverse=True)[:required]))
                candidate = (score, candidate_id, support_edges)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                _, candidate_id, _ = best
                member_ids.add(candidate_id)
                frame_ids.add(node_by_id[candidate_id]["frame_index"])
                changed = True
        if len(frame_ids) < int(min_track_views):
            continue
        assigned.update(member_ids)
        track_edges = [
            edge for edge in edges
            if edge["accepted"] and edge["left_node_id"] in member_ids and edge["right_node_id"] in member_ids
        ]
        track_nodes = [node_by_id[node_id] for node_id in sorted(member_ids)]
        tracks.append({"nodes": track_nodes, "edges": track_edges})
    return tracks


def _serializable_node(node):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in node.items()
        if key not in {"points"}
    }


def _scene_output(scene_name, nodes, edges, tracks, output_root):
    scene_root = output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=False)
    with (scene_root / "nodes.jsonl").open("w") as handle:
        for node in nodes:
            handle.write(json.dumps(_serializable_node(node), ensure_ascii=False, sort_keys=True) + "\n")
    with (scene_root / "edges.jsonl").open("w") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False, sort_keys=True) + "\n")
    track_dir = scene_root / "track_points"
    track_dir.mkdir()
    records = []
    for track_id, track in enumerate(tracks):
        points = np.unique(np.concatenate([node["points"] for node in track["nodes"]])).astype(np.int64)
        point_path = track_dir / f"track{track_id:04d}_points.npz"
        np.savez_compressed(point_path, point_indices=points)
        records.append(
            {
                "track_id": track_id,
                "class_id": int(track["nodes"][0]["class_id"]),
                "class_name": str(track["nodes"][0]["class_name"]),
                "node_ids": [int(node["node_id"]) for node in track["nodes"]],
                "observation_ids": [int(node["observation_id"]) for node in track["nodes"]],
                "frame_ids": [str(node["frame_id"]) for node in track["nodes"]],
                "support_view_count": len({node["frame_index"] for node in track["nodes"]}),
                "point_count": int(len(points)),
                "mean_node_quality": float(np.mean([node["quality"] for node in track["nodes"]])),
                "mean_edge_score": float(np.mean([edge["edge_score"] for edge in track["edges"]])) if track["edges"] else 0.0,
                "min_mutual_projection_support": float(min(edge["mutual_projection_support"] for edge in track["edges"])) if track["edges"] else 0.0,
                "points_path": str(point_path),
            }
        )
    payload = {
        "scene_name": scene_name,
        "node_count": len(nodes),
        "candidate_edge_count": len(edges),
        "accepted_edge_count": sum(edge["accepted"] for edge in edges),
        "track_count": len(records),
        "tracks": records,
    }
    (scene_root / "residual_tracks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def _build_scene(scene_name, args):
    from utils import WORLD_2_CAM

    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    points, _ = world.load_ply(world.mesh)
    points_xyz = np.asarray(points[:, :3], dtype=np.float32)
    superpoints = None
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    if processed_path.is_file():
        processed = np.load(processed_path, mmap_mode="r")
        if processed.shape[0] == len(points_xyz) and processed.shape[1] >= 10:
            superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    nodes = _load_scene_nodes(scene_name, args.residual_root, points_xyz, superpoints, args.residual_mode)
    pairs = _candidate_pairs(nodes, args.knn, args.max_centroid_distance, args.max_nodes_per_point)
    scaling_params = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    mask_cache = _MaskCache()
    edges = [
        _build_edge(nodes[left], nodes[right], projections, visibility, scaling_params, mask_cache, args)
        for left, right in pairs
    ]
    tracks = _build_tracks(nodes, edges, args.min_track_views, args.min_track_support_edges)
    summary = _scene_output(scene_name, nodes, edges, tracks, args.output_root)
    del world, projections, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--residual_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--residual_mode", choices=("after_any", "after_same_class"), default="after_any")
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--max_centroid_distance", type=float, default=0.35)
    parser.add_argument("--max_nodes_per_point", type=int, default=12)
    parser.add_argument("--min_projection_visible_points", type=int, default=8)
    parser.add_argument("--min_mutual_projection_support", type=float, default=0.50)
    parser.add_argument("--min_direct_point_iou", type=float, default=0.05)
    parser.add_argument("--min_track_views", type=int, default=2)
    parser.add_argument("--min_track_support_edges", type=int, default=2)
    args = parser.parse_args()
    if args.min_track_views < 2:
        raise SystemExit("--min_track_views 必须不小于 2；单帧残差不能形成轨迹。")
    for name in ("scene_list", "residual_root", "dataset_root", "processed_scene_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists():
        existing_names = {path.name for path in args.output_root.iterdir()}
        allowed_preflight = {"entry_preflight_manifest.json"}
        if existing_names - allowed_preflight:
            raise SystemExit(f"输出目录已存在真实结果，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = _build_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"节点 {summary['node_count']}，边 {summary['accepted_edge_count']}，轨迹 {summary['track_count']}",
            flush=True,
        )
    payload = {
        "gt_usage": "不读取 GT；输出只用于后续 GT-only 轨迹诊断。",
        "decision_state": "未生成最终候选、未融合、未评分、未评测。",
        "params": {key: value for key, value in vars(args).items() if key not in {"config"}},
        "scene_count": len(summaries),
        "node_count": sum(item["node_count"] for item in summaries),
        "candidate_edge_count": sum(item["candidate_edge_count"] for item in summaries),
        "accepted_edge_count": sum(item["accepted_edge_count"] for item in summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "scenes": summaries,
    }
    (args.output_root / "residual_evidence_graph_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "scene_count",
                    "node_count",
                    "candidate_edge_count",
                    "accepted_edge_count",
                    "track_count",
                    "decision_state",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

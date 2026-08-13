#!/usr/bin/env python3
"""为自动 SAM 轨迹建立原始 superpoint 的受约束生长证据账本。

输入是既有自动轨迹和全局证据图。本工具只记录轨迹核心、原始 superpoint 邻接关系
及跨视角正支持的潜在生长前沿；不扩张点、不生成候选、不赋类别、不融合、不评测。
"""

import argparse
import json
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


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_rgb(colors):
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size and float(colors.max(initial=0.0)) > 1.5:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _normalize_normals(normals):
    normals = np.asarray(normals, dtype=np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-6)


def _raw_superpoint_context(processed, adjacency_knn, adjacency_max_distance, min_contact_points, min_contact_ratio):
    """按原始 superpoint 编号构建局部接触图，绝不重编号或修改几何原子。"""
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError("处理点云缺少原始 superpoint 列")
    points = np.asarray(processed[:, :3], dtype=np.float32)
    raw_ids, inverse = np.unique(np.asarray(processed[:, 9], dtype=np.int64), return_inverse=True)
    inverse = inverse.astype(np.int64)
    sizes = np.bincount(inverse, minlength=len(raw_ids)).astype(np.int64)
    colors = _normalize_rgb(processed[:, 3:6])
    normals = _normalize_normals(processed[:, 6:9])
    centers = np.zeros((len(raw_ids), 3), dtype=np.float32)
    mean_colors = np.zeros((len(raw_ids), 3), dtype=np.float32)
    mean_normals = np.zeros((len(raw_ids), 3), dtype=np.float32)
    np.add.at(centers, inverse, points)
    np.add.at(mean_colors, inverse, colors)
    np.add.at(mean_normals, inverse, normals)
    centers /= np.maximum(sizes[:, None], 1)
    mean_colors /= np.maximum(sizes[:, None], 1)
    mean_normals /= np.maximum(sizes[:, None], 1)
    mean_normals = _normalize_normals(mean_normals)

    neighbors = defaultdict(list)
    if len(points) < 2:
        return {
            "raw_ids": raw_ids,
            "sizes": sizes,
            "centers": centers,
            "mean_colors": mean_colors,
            "mean_normals": mean_normals,
            "neighbors": neighbors,
        }
    neighbor_count = min(max(1, int(adjacency_knn)), len(points) - 1)
    tree = cKDTree(points)
    distances, point_neighbors = tree.query(points, k=neighbor_count + 1, workers=-1)
    left = np.repeat(np.arange(len(points), dtype=np.int64), neighbor_count)
    right = np.asarray(point_neighbors, dtype=np.int64)[:, 1:].reshape(-1)
    distances = np.asarray(distances, dtype=np.float32)[:, 1:].reshape(-1)
    valid = (left != right) & (distances <= float(adjacency_max_distance))
    left, right, distances = left[valid], right[valid], distances[valid]
    left_segments, right_segments = inverse[left], inverse[right]
    cross_segment = left_segments != right_segments
    left, right, distances = left[cross_segment], right[cross_segment], distances[cross_segment]
    left_segments, right_segments = left_segments[cross_segment], right_segments[cross_segment]
    if len(left_segments) == 0:
        return {
            "raw_ids": raw_ids,
            "sizes": sizes,
            "centers": centers,
            "mean_colors": mean_colors,
            "mean_normals": mean_normals,
            "neighbors": neighbors,
        }
    low, high = np.minimum(left_segments, right_segments), np.maximum(left_segments, right_segments)
    packed = low * len(raw_ids) + high
    unique_pairs, pair_inverse = np.unique(packed, return_inverse=True)
    contact_counts = np.bincount(pair_inverse, minlength=len(unique_pairs)).astype(np.int64)
    distance_sums = np.bincount(pair_inverse, weights=distances, minlength=len(unique_pairs))
    normal_differences = 1.0 - np.abs(np.sum(normals[left] * normals[right], axis=1))
    color_differences = np.linalg.norm(colors[left] - colors[right], axis=1) / np.sqrt(3.0)
    normal_sums = np.bincount(pair_inverse, weights=normal_differences, minlength=len(unique_pairs))
    color_sums = np.bincount(pair_inverse, weights=color_differences, minlength=len(unique_pairs))
    for packed_pair, count, distance_sum, normal_sum, color_sum in zip(
        unique_pairs, contact_counts, distance_sums, normal_sums, color_sums
    ):
        low_id, high_id = divmod(int(packed_pair), len(raw_ids))
        contact_ratio = float(count / max(1, min(sizes[low_id], sizes[high_id])))
        if int(count) < int(min_contact_points) or contact_ratio < float(min_contact_ratio):
            continue
        record = {
            "neighbor_superpoint_id": int(raw_ids[high_id]),
            "boundary_contact_count": int(count),
            "boundary_contact_ratio": contact_ratio,
            "mean_boundary_distance": float(distance_sum / count),
            "mean_normal_difference": float(normal_sum / count),
            "mean_color_difference": float(color_sum / count),
        }
        reverse = {**record, "neighbor_superpoint_id": int(raw_ids[low_id])}
        neighbors[int(raw_ids[low_id])].append(record)
        neighbors[int(raw_ids[high_id])].append(reverse)
    return {
        "raw_ids": raw_ids,
        "sizes": sizes,
        "centers": centers,
        "mean_colors": mean_colors,
        "mean_normals": mean_normals,
        "neighbors": {key: sorted(value, key=lambda item: item["neighbor_superpoint_id"]) for key, value in neighbors.items()},
    }


def _node_superpoint_support(node, size_by_superpoint):
    ids = np.asarray(node.get("superpoint_ids", []), dtype=np.int64)
    counts = np.asarray(node.get("superpoint_point_counts", []), dtype=np.int64)
    if len(ids) != len(counts):
        raise ValueError(f"观测 {node.get('observation_id')} 的 superpoint 支持数组长度不一致")
    return {
        int(superpoint_id): float(count / max(1, size_by_superpoint[int(superpoint_id)]))
        for superpoint_id, count in zip(ids, counts)
        if int(superpoint_id) in size_by_superpoint
    }


def _summarize_track_support(track_id, nodes, size_by_superpoint):
    support = defaultdict(lambda: {"frames": set(), "nodes": set(), "occupancies": [], "quality_weighted": 0.0})
    for node in nodes:
        for superpoint_id, occupancy in _node_superpoint_support(node, size_by_superpoint).items():
            item = support[superpoint_id]
            item["frames"].add(int(node["frame_index"]))
            item["nodes"].add(int(node["node_id"]))
            item["occupancies"].append(float(occupancy))
            item["quality_weighted"] += float(occupancy) * float(node["quality"])
    records = []
    for superpoint_id, item in sorted(support.items()):
        occupancies = item["occupancies"]
        records.append({
            "superpoint_id": int(superpoint_id),
            "point_count": int(size_by_superpoint[superpoint_id]),
            "support_view_count": len(item["frames"]),
            "support_node_count": len(item["nodes"]),
            "max_observation_occupancy": float(max(occupancies)),
            "mean_observation_occupancy": float(np.mean(occupancies)),
            "quality_weighted_occupancy_sum": float(item["quality_weighted"]),
            "track_id": int(track_id),
        })
    return records


def _track_membership(tracks):
    membership = defaultdict(list)
    for track in tracks:
        track_id = int(track["track_id"])
        for observation_id in track.get("observation_ids", []):
            membership[int(observation_id)].append(track_id)
    return {key: sorted(value) for key, value in membership.items()}


def _cross_view_frontier_links(edges, node_by_id, membership):
    """将图中跨视角边保留为生长前沿的正证据，不以边分数或阈值作决定。"""
    links = defaultdict(lambda: {"edge_count": 0, "reprojection_support": [], "source_node_ids": set()})
    for edge in edges:
        left = node_by_id.get(int(edge["left_node_id"]))
        right = node_by_id.get(int(edge["right_node_id"]))
        if left is None or right is None:
            continue
        for source, target, support in (
            (left, right, float(edge.get("left_to_right_reprojection_support_ratio", 0.0))),
            (right, left, float(edge.get("right_to_left_reprojection_support_ratio", 0.0))),
        ):
            for track_id in membership.get(int(source["observation_id"]), []):
                for superpoint_id in target.get("superpoint_ids", []):
                    item = links[(int(track_id), int(superpoint_id))]
                    item["edge_count"] += 1
                    item["reprojection_support"].append(support)
                    item["source_node_ids"].add(int(source["node_id"]))
    return links


def _internal_edge_summary(track_node_ids, edges):
    internal = [
        edge for edge in edges
        if int(edge["left_node_id"]) in track_node_ids and int(edge["right_node_id"]) in track_node_ids
    ]
    if not internal:
        return {"internal_cross_view_edge_count": 0, "mean_internal_reprojection_support": 0.0}
    supports = [
        min(
            float(edge.get("left_to_right_reprojection_support_ratio", 0.0)),
            float(edge.get("right_to_left_reprojection_support_ratio", 0.0)),
        )
        for edge in internal
    ]
    return {
        "internal_cross_view_edge_count": len(internal),
        "mean_internal_reprojection_support": float(np.mean(supports)),
    }


def _growth_frontier(track_id, support_records, neighbors, frontier_links):
    supported_ids = {int(item["superpoint_id"]) for item in support_records}
    frontier = {}
    for source in support_records:
        source_id = int(source["superpoint_id"])
        for adjacency in neighbors.get(source_id, []):
            target_id = int(adjacency["neighbor_superpoint_id"])
            if target_id in supported_ids:
                continue
            link = frontier_links.get((int(track_id), target_id))
            if link is None:
                continue
            item = frontier.setdefault(target_id, {
                "superpoint_id": target_id,
                "adjacent_seed_superpoint_ids": [],
                "boundary_contact_count_sum": 0,
                "mean_normal_differences": [],
                "mean_color_differences": [],
                "mean_boundary_distances": [],
                "cross_view_link_count": 0,
                "cross_view_reprojection_support": [],
                "linked_seed_node_ids": set(),
            })
            item["adjacent_seed_superpoint_ids"].append(source_id)
            item["boundary_contact_count_sum"] += int(adjacency["boundary_contact_count"])
            item["mean_normal_differences"].append(float(adjacency["mean_normal_difference"]))
            item["mean_color_differences"].append(float(adjacency["mean_color_difference"]))
            item["mean_boundary_distances"].append(float(adjacency["mean_boundary_distance"]))
            item["cross_view_link_count"] += int(link["edge_count"])
            item["cross_view_reprojection_support"].extend(link["reprojection_support"])
            item["linked_seed_node_ids"].update(link["source_node_ids"])
    records = []
    for target_id, item in sorted(frontier.items()):
        records.append({
            "superpoint_id": target_id,
            "adjacent_seed_superpoint_ids": sorted(set(item["adjacent_seed_superpoint_ids"])),
            "adjacent_seed_superpoint_count": len(set(item["adjacent_seed_superpoint_ids"])),
            "boundary_contact_count_sum": int(item["boundary_contact_count_sum"]),
            "mean_normal_difference": float(np.mean(item["mean_normal_differences"])),
            "mean_color_difference": float(np.mean(item["mean_color_differences"])),
            "mean_boundary_distance": float(np.mean(item["mean_boundary_distances"])),
            "cross_view_link_count": int(item["cross_view_link_count"]),
            "mean_cross_view_reprojection_support": float(np.mean(item["cross_view_reprojection_support"])),
            "linked_seed_node_count": len(item["linked_seed_node_ids"]),
            "growth_status": "仅记录具有图中跨视角正证据的相邻原子；不加入任何 mask。",
        })
    return records


def _build_scene(scene_name, args):
    graph_scene = args.evidence_graph_root / scene_name
    nodes = _read_jsonl(graph_scene / "nodes.jsonl")
    edges = _read_jsonl(graph_scene / "cross_view_edges.jsonl")
    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text()).get("tracks", [])
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    context = _raw_superpoint_context(
        processed, args.adjacency_knn, args.adjacency_max_distance,
        args.min_contact_points, args.min_contact_ratio,
    )
    size_by_superpoint = {int(item): int(size) for item, size in zip(context["raw_ids"], context["sizes"])}
    node_by_observation = {int(node["observation_id"]): node for node in nodes}
    node_by_id = {int(node["node_id"]): node for node in nodes}
    membership = _track_membership(tracks)
    frontier_links = _cross_view_frontier_links(edges, node_by_id, membership)
    records = []
    for track in tracks:
        track_id = int(track["track_id"])
        track_nodes = [node_by_observation[int(item)] for item in track.get("observation_ids", []) if int(item) in node_by_observation]
        if not track_nodes:
            continue
        support_records = _summarize_track_support(track_id, track_nodes, size_by_superpoint)
        track_node_ids = {int(node["node_id"]) for node in track_nodes}
        records.append({
            "scene_name": scene_name,
            "track_id": track_id,
            "source_observation_ids": [int(item["observation_id"]) for item in track_nodes],
            "source_node_ids": sorted(track_node_ids),
            "source_view_count": len({int(item["frame_index"]) for item in track_nodes}),
            "supported_superpoints": support_records,
            "growth_frontier_superpoints": _growth_frontier(
                track_id, support_records, context["neighbors"], frontier_links
            ),
            **_internal_edge_summary(track_node_ids, edges),
            "decision_state": "轨迹仅为 seed；本记录不扩张 superpoint、不生成候选。",
        })
    scene_root = args.output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=False)
    (scene_root / "automatic_sam_track_growth_ledger.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "scene_name": scene_name,
        "graph_node_count": len(nodes),
        "graph_cross_view_edge_count": len(edges),
        "track_count": len(records),
        "supported_superpoint_count": sum(len(item["supported_superpoints"]) for item in records),
        "positive_growth_frontier_count": sum(len(item["growth_frontier_superpoints"]) for item in records),
        "decision_state": "不读取 GT；不扩张点、不生成候选、不赋类别、不融合、不评分、不评测。",
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--evidence_graph_root", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--adjacency_knn", type=int, default=12)
    parser.add_argument("--adjacency_max_distance", type=float, default=0.05)
    parser.add_argument("--min_contact_points", type=int, default=3)
    parser.add_argument("--min_contact_ratio", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("scene_list", "evidence_graph_root", "track_root", "processed_scene_root", "output_root"):
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
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: 轨迹 {summary['track_count']}，"
            f"正证据前沿 {summary['positive_growth_frontier_count']}",
            flush=True,
        )
    payload = {
        "purpose": "在生成几何变体前记录自动 SAM 轨迹的原始 superpoint 生长证据。",
        "gt_usage": "不读取 GT；不扩张点、不生成候选、不赋类别、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "track_count": sum(item["track_count"] for item in summaries),
        "supported_superpoint_count": sum(item["supported_superpoint_count"] for item in summaries),
        "positive_growth_frontier_count": sum(item["positive_growth_frontier_count"] for item in summaries),
        "params": vars(args),
    }
    (args.output_root / "automatic_sam_track_growth_ledger_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

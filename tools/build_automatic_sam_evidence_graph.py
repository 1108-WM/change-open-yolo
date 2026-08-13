#!/usr/bin/env python3
"""构建独立自动 SAM 观测的无 GT 全局证据图。

该工具只组织观测间的连续几何、深度可见性和同帧粒度关系。它不生成三维候选、
不赋类别、不融合、不评分、不评测，也不会为任意边输出接受或拒绝决定。
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import yaml


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


def _quality(predicted_iou, stability_score, point_count):
    size = min(1.0, math.log1p(point_count) / math.log(2001.0))
    return float(max(0.0, predicted_iou) * max(0.0, stability_score) * (0.40 + 0.60 * size))


def _bbox_iou_xywh(left, right):
    left_x, left_y, left_w, left_h = (float(value) for value in left)
    right_x, right_y, right_w, right_h = (float(value) for value in right)
    intersection_w = max(0.0, min(left_x + left_w, right_x + right_w) - max(left_x, right_x))
    intersection_h = max(0.0, min(left_y + left_h, right_y + right_h) - max(left_y, right_y))
    intersection = intersection_w * intersection_h
    union = left_w * left_h + right_w * right_h - intersection
    return float(intersection / max(union, 1e-12))


def _bbox_containment(inner, outer):
    inner_x, inner_y, inner_w, inner_h = (float(value) for value in inner)
    outer_x, outer_y, outer_w, outer_h = (float(value) for value in outer)
    intersection_w = max(0.0, min(inner_x + inner_w, outer_x + outer_w) - max(inner_x, outer_x))
    intersection_h = max(0.0, min(inner_y + inner_h, outer_y + outer_h) - max(inner_y, outer_y))
    return float((intersection_w * intersection_h) / max(inner_w * inner_h, 1e-12))


def _superpoint_support(points, superpoints):
    ids, counts = np.unique(superpoints[points], return_counts=True)
    return ids.astype(np.int64), counts.astype(np.int64)


def _resolve_points_path(value, observation_root):
    path = Path(value)
    return path if path.is_absolute() else observation_root / path


def _load_nodes(scene_name, automatic_root, points_xyz, superpoints):
    observation_root = automatic_root / scene_name
    nodes = []
    with (observation_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            points_path = _resolve_points_path(raw["point_indices_path"], observation_root)
            points = np.unique(np.asarray(np.load(points_path)["point_indices"], dtype=np.int64))
            points = points[(points >= 0) & (points < len(points_xyz))]
            if len(points) == 0:
                continue
            superpoint_ids, superpoint_counts = _superpoint_support(points, superpoints)
            local = points_xyz[points]
            node = {
                "node_id": len(nodes),
                "observation_id": int(raw["observation_id"]),
                "scene_name": scene_name,
                "frame_id": str(raw["frame_id"]),
                "frame_index": int(raw["frame_index"]),
                "point_count": int(len(points)),
                "area": int(raw.get("area", 0)),
                "predicted_iou": float(raw["predicted_iou"]),
                "stability_score": float(raw["stability_score"]),
                "bbox_xywh": [int(value) for value in raw.get("bbox_xywh", [0, 0, 0, 0])],
                "crop_box_xywh": [int(value) for value in raw.get("crop_box_xywh", [0, 0, 0, 0])],
                "point_indices_path": str(points_path),
                "points": points,
                "centroid": local.mean(axis=0).astype(np.float32),
                "bbox3d_min": local.min(axis=0).astype(np.float32),
                "bbox3d_max": local.max(axis=0).astype(np.float32),
                "superpoint_ids": superpoint_ids,
                "superpoint_point_counts": superpoint_counts,
            }
            node["quality"] = _quality(node["predicted_iou"], node["stability_score"], len(points))
            nodes.append(node)
    return nodes


def _load_track_membership(track_root, scene_name):
    if track_root is None:
        return {}
    path = track_root / scene_name / "automatic_tracks.json"
    if not path.is_file():
        return {}
    membership = defaultdict(list)
    for track in json.loads(path.read_text()).get("tracks", []):
        track_id = int(track["track_id"])
        for observation_id in track.get("observation_ids", []):
            membership[int(observation_id)].append(track_id)
    return {key: sorted(value) for key, value in membership.items()}


def _shared_superpoints(left, right):
    return np.intersect1d(left["superpoint_ids"], right["superpoint_ids"], assume_unique=True)


def _superpoint_jaccard(left, right):
    shared = _shared_superpoints(left, right)
    union = len(left["superpoint_ids"]) + len(right["superpoint_ids"]) - len(shared)
    return float(len(shared) / max(1, union)), shared


def _same_frame_relation(left, right):
    point_iou, shared_points = _point_iou(left["points"], right["points"])
    superpoint_iou, shared_superpoints = _superpoint_jaccard(left, right)
    left_coverage = float(shared_points / max(1, left["point_count"]))
    right_coverage = float(shared_points / max(1, right["point_count"]))
    bbox_iou = _bbox_iou_xywh(left["bbox_xywh"], right["bbox_xywh"])
    bbox_left_in_right = _bbox_containment(left["bbox_xywh"], right["bbox_xywh"])
    bbox_right_in_left = _bbox_containment(right["bbox_xywh"], left["bbox_xywh"])
    return {
        "relation_type": "same_frame_overlap_or_granularity",
        "left_node_id": left["node_id"],
        "right_node_id": right["node_id"],
        "frame_id": left["frame_id"],
        "frame_index": left["frame_index"],
        "point_iou": point_iou,
        "shared_point_count": shared_points,
        "left_point_coverage": left_coverage,
        "right_point_coverage": right_coverage,
        "superpoint_iou": superpoint_iou,
        "shared_superpoint_count": int(len(shared_superpoints)),
        "bbox_iou": bbox_iou,
        "left_bbox_in_right": bbox_left_in_right,
        "right_bbox_in_left": bbox_right_in_left,
        "larger_area_node_id": left["node_id"] if left["area"] >= right["area"] else right["node_id"],
        "smaller_area_node_id": right["node_id"] if left["area"] >= right["area"] else left["node_id"],
        "relation_data_limit": "自动观测未保存二维二值 mask；包含关系仅基于回投点、superpoint 与 bbox 连续值。",
    }


def _same_frame_relations(nodes):
    by_frame = defaultdict(list)
    for node in nodes:
        by_frame[node["frame_index"]].append(node)
    relations = []
    for frame_nodes in by_frame.values():
        for offset, left in enumerate(frame_nodes):
            for right in frame_nodes[offset + 1:]:
                relation = _same_frame_relation(left, right)
                if (
                    relation["shared_point_count"]
                    or relation["shared_superpoint_count"]
                    or relation["bbox_iou"] > 0.0
                ):
                    relations.append(relation)
    return relations


def _cross_view_candidate_pairs(nodes, knn, max_centroid_distance, max_nodes_per_entity):
    """仅做可审计的稀疏召回；原因记录在边上，不是边接受规则。"""
    if len(nodes) < 2:
        return {}
    pairs = defaultdict(set)

    def add_pair(left_id, right_id, reason):
        if left_id == right_id or nodes[left_id]["frame_index"] == nodes[right_id]["frame_index"]:
            return
        left_id, right_id = sorted((int(left_id), int(right_id)))
        pairs[(left_id, right_id)].add(reason)

    centroids = np.stack([node["centroid"] for node in nodes])
    tree = cKDTree(centroids)
    neighbor_count = min(len(nodes), max(2, int(knn) + 1))
    distances, neighbors = tree.query(centroids, k=neighbor_count)
    distances, neighbors = np.atleast_2d(distances), np.atleast_2d(neighbors)
    for left_id, (row_distances, row_neighbors) in enumerate(zip(distances, neighbors)):
        for distance, right_id in zip(row_distances, row_neighbors):
            if distance <= float(max_centroid_distance):
                add_pair(left_id, int(right_id), "centroid_knn")

    for entity_name, values_key, reason in (
        ("point", "points", "shared_point"),
        ("superpoint", "superpoint_ids", "shared_superpoint"),
    ):
        buckets = defaultdict(list)
        for node in nodes:
            for entity_id in node[values_key]:
                bucket = buckets[int(entity_id)]
                if len(bucket) < int(max_nodes_per_entity):
                    bucket.append(node["node_id"])
        for entity_nodes in buckets.values():
            for offset, left_id in enumerate(entity_nodes):
                for right_id in entity_nodes[offset + 1:]:
                    add_pair(left_id, right_id, reason)
    return {key: sorted(value) for key, value in sorted(pairs.items())}


def _frame_visible_superpoints(visibility, superpoints, frame_indices):
    if visibility is None:
        return {}
    output = {}
    for frame_index in sorted(set(int(value) for value in frame_indices)):
        visible_points = np.flatnonzero(visibility[frame_index])
        output[frame_index] = np.unique(superpoints[visible_points]).astype(np.int64)
    return output


def _cross_view_edge(left, right, recall_reasons, visibility, frame_visible_superpoints):
    point_iou, shared_points = _point_iou(left["points"], right["points"])
    superpoint_iou, shared_superpoints = _superpoint_jaccard(left, right)
    left_visible_count = right_visible_count = 0
    jointly_visible_superpoints = np.empty(0, dtype=np.int64)
    visibility_status = "not_loaded"
    if visibility is not None:
        left_visible_count = int(visibility[right["frame_index"], left["points"]].sum())
        right_visible_count = int(visibility[left["frame_index"], right["points"]].sum())
        left_frame_visible = frame_visible_superpoints[left["frame_index"]]
        right_frame_visible = frame_visible_superpoints[right["frame_index"]]
        jointly_visible_superpoints = np.intersect1d(
            shared_superpoints,
            np.intersect1d(left_frame_visible, right_frame_visible, assume_unique=True),
            assume_unique=True,
        )
        visibility_status = "rgbd_depth_zbuffer"
    centroid_distance = float(np.linalg.norm(left["centroid"] - right["centroid"]))
    return {
        "relation_type": "cross_view_evidence",
        "left_node_id": left["node_id"],
        "right_node_id": right["node_id"],
        "left_frame_id": left["frame_id"],
        "right_frame_id": right["frame_id"],
        "left_frame_index": left["frame_index"],
        "right_frame_index": right["frame_index"],
        "frame_index_gap": abs(int(left["frame_index"]) - int(right["frame_index"])),
        "recall_reasons": recall_reasons,
        "point_iou": point_iou,
        "shared_point_count": shared_points,
        "left_point_coverage": float(shared_points / max(1, left["point_count"])),
        "right_point_coverage": float(shared_points / max(1, right["point_count"])),
        "shared_superpoint_count": int(len(shared_superpoints)),
        "superpoint_iou": superpoint_iou,
        "centroid_distance": centroid_distance,
        "left_visible_in_right_frame_count": left_visible_count,
        "right_visible_in_left_frame_count": right_visible_count,
        "left_visible_in_right_frame_ratio": float(left_visible_count / max(1, left["point_count"])),
        "right_visible_in_left_frame_ratio": float(right_visible_count / max(1, right["point_count"])),
        "left_to_right_reprojection_support_ratio": float(shared_points / max(1, left_visible_count)),
        "right_to_left_reprojection_support_ratio": float(shared_points / max(1, right_visible_count)),
        "jointly_visible_shared_superpoint_count": int(len(jointly_visible_superpoints)),
        "visibility_source": visibility_status,
        "decision_state": "连续证据记录；不含 accepted、阈值筛除或候选写回决定。",
    }


def _load_visibility(scene_name, dataset_root, config_path):
    from utils import WORLD_2_CAM

    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    world = WORLD_2_CAM(str(dataset_root / scene_name), float(config["openyolo3d"]["depth_scale"]), config)
    _, visibility = world.get_mesh_projections()
    return visibility.detach().cpu().numpy().astype(bool)


def _serializable_node(node, track_membership):
    return {
        "node_id": node["node_id"],
        "observation_id": node["observation_id"],
        "scene_name": node["scene_name"],
        "frame_id": node["frame_id"],
        "frame_index": node["frame_index"],
        "point_count": node["point_count"],
        "area": node["area"],
        "predicted_iou": node["predicted_iou"],
        "stability_score": node["stability_score"],
        "quality": node["quality"],
        "bbox_xywh": node["bbox_xywh"],
        "crop_box_xywh": node["crop_box_xywh"],
        "point_indices_path": node["point_indices_path"],
        "centroid": node["centroid"].tolist(),
        "bbox3d_min": node["bbox3d_min"].tolist(),
        "bbox3d_max": node["bbox3d_max"].tolist(),
        "superpoint_ids": node["superpoint_ids"].tolist(),
        "superpoint_point_counts": node["superpoint_point_counts"].tolist(),
        "existing_track_ids": track_membership.get(node["observation_id"], []),
        "node_role": "独立自动 SAM 单帧证据，不是最终实例或候选 mask。",
    }


def _write_jsonl(path, records):
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _build_scene(scene_name, args):
    processed_path = args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 ScanNet superpoint 列")
    points_xyz = np.asarray(processed[:, :3], dtype=np.float32)
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    nodes = _load_nodes(scene_name, args.automatic_root, points_xyz, superpoints)
    visibility = _load_visibility(scene_name, args.dataset_root, args.config_path) if args.with_visibility else None
    frame_visible_superpoints = _frame_visible_superpoints(
        visibility, superpoints, [node["frame_index"] for node in nodes]
    )
    track_membership = _load_track_membership(args.track_root, scene_name)
    same_frame_relations = _same_frame_relations(nodes)
    pairs = _cross_view_candidate_pairs(
        nodes, args.knn, args.max_centroid_distance, args.max_nodes_per_entity
    )
    cross_view_edges = [
        _cross_view_edge(nodes[left_id], nodes[right_id], reasons, visibility, frame_visible_superpoints)
        for (left_id, right_id), reasons in pairs.items()
    ]

    scene_root = args.output_root / scene_name
    scene_root.mkdir(parents=True, exist_ok=False)
    _write_jsonl(scene_root / "nodes.jsonl", [_serializable_node(node, track_membership) for node in nodes])
    _write_jsonl(scene_root / "same_frame_relations.jsonl", same_frame_relations)
    _write_jsonl(scene_root / "cross_view_edges.jsonl", cross_view_edges)
    summary = {
        "scene_name": scene_name,
        "node_count": len(nodes),
        "same_frame_relation_count": len(same_frame_relations),
        "cross_view_edge_count": len(cross_view_edges),
        "nodes_with_existing_track_membership": sum(
            bool(track_membership.get(node["observation_id"])) for node in nodes
        ),
        "visibility_source": "rgbd_depth_zbuffer" if visibility is not None else "not_loaded",
        "decision_state": "不读取 GT；不生成候选、不赋类别、不融合、不评分、不评测。",
    }
    (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--processed_scene_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--track_root", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--max_centroid_distance", type=float, default=0.40)
    parser.add_argument("--max_nodes_per_entity", type=int, default=24)
    parser.add_argument("--with_visibility", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "automatic_root", "processed_scene_root", "dataset_root", "config_path", "output_root"
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.track_root is not None:
        args.track_root = _resolve(args.track_root)
    existing_names = set()
    if args.output_root.exists():
        existing_names = {child.name for child in args.output_root.iterdir()}
    if existing_names - {"entry_preflight_manifest.json"}:
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
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"节点 {summary['node_count']}，跨帧证据边 {summary['cross_view_edge_count']}",
            flush=True,
        )
    payload = {
        "purpose": "为后续受约束候选生长和多视图聚合保留自动 SAM 的全局连续证据。",
        "gt_usage": "不读取 GT；不生成候选、不赋类别、不融合、不评分、不评测。",
        "scene_count": len(summaries),
        "node_count": sum(item["node_count"] for item in summaries),
        "same_frame_relation_count": sum(item["same_frame_relation_count"] for item in summaries),
        "cross_view_edge_count": sum(item["cross_view_edge_count"] for item in summaries),
        "params": {key: value for key, value in vars(args).items()},
    }
    (args.output_root / "automatic_sam_evidence_graph_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

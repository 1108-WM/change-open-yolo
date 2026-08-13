#!/usr/bin/env python3
"""建立自动 SAM Pareto 候选的无 GT 候选族关系图。

本工具只保存候选之间及候选与冻结 native 之间的连续关系。它不定义 IoU 门槛、
不筛除、不合并、不改类别，也不输出最终预测。自动观测的同帧层级信息尚未保存
真实二维二值 mask，因此仅作为回投点、superpoint 与 bbox 的近似粒度证据写入。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复项")
    return scenes


def _read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, records):
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _semantic_distribution(record):
    values = {
        int(item["class_index"]): max(0.0, float(item["probability"]))
        for item in record.get("semantic_class_distribution", [])
    }
    values[-1] = max(0.0, 1.0 - sum(values.values()))
    total = sum(values.values())
    return {key: value / max(total, 1e-12) for key, value in values.items()}


def semantic_js_divergence(left, right):
    """将保存分布以 other 桶补齐后计算 Jensen-Shannon divergence。"""
    left_values, right_values = _semantic_distribution(left), _semantic_distribution(right)
    keys = sorted(set(left_values) | set(right_values))
    left_array = np.asarray([left_values.get(key, 0.0) for key in keys], dtype=np.float64)
    right_array = np.asarray([right_values.get(key, 0.0) for key in keys], dtype=np.float64)
    midpoint = 0.5 * (left_array + right_array)
    left_valid, right_valid = left_array > 0.0, right_array > 0.0
    left_kl = np.sum(left_array[left_valid] * np.log(left_array[left_valid] / midpoint[left_valid]))
    right_kl = np.sum(right_array[right_valid] * np.log(right_array[right_valid] / midpoint[right_valid]))
    return float(0.5 * (left_kl + right_kl) / np.log(2.0))


def geometry_relation(left_size, right_size, intersection):
    """返回无阈值的 IoU 与双向覆盖。"""
    union = int(left_size + right_size - intersection)
    return {
        "intersection_point_count": int(intersection),
        "iou": float(intersection / max(1, union)),
        "left_coverage": float(intersection / max(1, left_size)),
        "right_coverage": float(intersection / max(1, right_size)),
    }


def _track_memberships(nodes, source_tracks):
    node_tracks = {}
    for node in nodes:
        node_tracks[int(node["node_id"])] = sorted(
            set(int(track) for track in node.get("existing_track_ids", []) if int(track) in source_tracks)
        )
    return node_tracks


def _append_pair_stat(stats, key, values):
    item = stats.setdefault(key, {"count": 0, "reprojection": [], "joint_visible_superpoints": [], "hierarchy": []})
    item["count"] += 1
    item["reprojection"].extend(values.get("reprojection", []))
    item["joint_visible_superpoints"].extend(values.get("joint_visible_superpoints", []))
    item["hierarchy"].extend(values.get("hierarchy", []))


def build_track_pair_evidence(nodes, cross_view_edges, same_frame_relations, source_tracks):
    """将观测级跨视角/粒度关系提升至候选的 source_track 对。"""
    node_tracks = _track_memberships(nodes, source_tracks)
    cross_stats, hierarchy_stats = {}, {}
    for edge in cross_view_edges:
        left_tracks = node_tracks.get(int(edge["left_node_id"]), [])
        right_tracks = node_tracks.get(int(edge["right_node_id"]), [])
        support = min(
            float(edge.get("left_to_right_reprojection_support_ratio", 0.0)),
            float(edge.get("right_to_left_reprojection_support_ratio", 0.0)),
        )
        for left_track in left_tracks:
            for right_track in right_tracks:
                if left_track == right_track:
                    continue
                _append_pair_stat(cross_stats, tuple(sorted((left_track, right_track))), {
                    "reprojection": [support],
                    "joint_visible_superpoints": [int(edge.get("jointly_visible_shared_superpoint_count", 0))],
                })
    for relation in same_frame_relations:
        left_tracks = node_tracks.get(int(relation["left_node_id"]), [])
        right_tracks = node_tracks.get(int(relation["right_node_id"]), [])
        nesting = max(float(relation.get("left_bbox_in_right", 0.0)), float(relation.get("right_bbox_in_left", 0.0)))
        for left_track in left_tracks:
            for right_track in right_tracks:
                if left_track == right_track:
                    continue
                _append_pair_stat(hierarchy_stats, tuple(sorted((left_track, right_track))), {
                    "hierarchy": [nesting],
                })
    return cross_stats, hierarchy_stats


def _pair_evidence(track_pair, cross_stats, hierarchy_stats):
    cross = cross_stats.get(track_pair, {})
    hierarchy = hierarchy_stats.get(track_pair, {})
    return {
        "cross_view_edge_count": int(cross.get("count", 0)),
        "mean_bidirectional_reprojection_support": float(np.mean(cross.get("reprojection", [0.0]))),
        "mean_jointly_visible_shared_superpoint_count": float(np.mean(cross.get("joint_visible_superpoints", [0.0]))),
        "same_frame_granularity_relation_count": int(hierarchy.get("count", 0)),
        "max_same_frame_bbox_nesting": float(max(hierarchy.get("hierarchy", [0.0]))),
        "hierarchy_evidence_limit": "近似：自动观测未保存二维二值 mask。",
    }


def _relation_tags(same_class, geometry, evidence):
    tags = ["geometric_overlap"] if geometry["intersection_point_count"] else []
    tags.append("same_class_relation" if same_class else "cross_class_relation")
    if evidence["cross_view_edge_count"]:
        tags.append("cross_view_evidence")
    if evidence["same_frame_granularity_relation_count"]:
        tags.append("same_frame_granularity_evidence")
    if not geometry["intersection_point_count"] and evidence["cross_view_edge_count"]:
        tags.append("cross_view_evidence_without_geometric_overlap")
    return tags


def _candidate_points(scene_root, candidates, point_count):
    rows, point_lists = [], []
    for candidate in candidates:
        points = np.unique(np.asarray(np.load(scene_root / candidate["seed_points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < point_count)]
        point_lists.append(points)
        rows.extend([len(point_lists) - 1] * len(points))
    if not candidates:
        return sparse.csr_matrix((0, point_count), dtype=np.int8), point_lists
    columns = np.concatenate(point_lists) if point_lists else np.empty(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(columns), dtype=np.int32), (rows, columns)), shape=(len(candidates), point_count)), point_lists


def _auto_nodes(candidates, quality_by_id, semantic_by_id, point_lists, native_union):
    nodes = []
    for candidate, points in zip(candidates, point_lists):
        variant_id = str(candidate["selected_variant_id"])
        quality, semantic = quality_by_id.get(variant_id, {}), semantic_by_id.get(variant_id, {})
        outside_native = np.setdiff1d(points, native_union, assume_unique=True).size
        nodes.append({
            "node_kind": "automatic_candidate",
            "candidate_id": int(candidate["candidate_id"]),
            "source_track_id": int(candidate["source_track_id"]),
            "class_id": int(candidate["class_id"]),
            "class_name": str(candidate["class_name"]),
            "selected_variant_id": variant_id,
            "point_count": int(len(points)),
            "gvc_score": float(candidate.get("gvc_score", quality.get("gvc_score", 0.0))),
            "semantic_vote_margin": float(candidate.get("semantic_vote_margin", semantic.get("semantic_vote_margin", 0.0))),
            "semantic_normalized_entropy": float(candidate.get("semantic_normalized_entropy", semantic.get("semantic_normalized_entropy", 1.0))),
            "semantic_evidence_top_class_index": int(semantic.get("semantic_evidence_top_class_index", candidate["class_id"])),
            "semantic_class_distribution": semantic.get("semantic_class_distribution", []),
            "independent_from_native_point_count": int(outside_native),
            "independent_from_native_ratio": float(outside_native / max(1, len(points))),
            "node_role": "自动候选关系图节点；不代表最终预测。",
        })
    return nodes


def _family_records(auto_nodes, auto_relations, native_relations):
    auto_neighbors, native_neighbors, tags = defaultdict(set), defaultdict(set), defaultdict(Counter)
    for relation in auto_relations:
        left, right = int(relation["left_candidate_id"]), int(relation["right_candidate_id"])
        auto_neighbors[left].add(right)
        auto_neighbors[right].add(left)
        for candidate_id in (left, right):
            tags[candidate_id].update(relation["relation_tags"])
    for relation in native_relations:
        candidate_id = int(relation["automatic_candidate_id"])
        native_neighbors[candidate_id].add(int(relation["native_candidate_index"]))
        tags[candidate_id].update(relation["relation_tags"])
    records = []
    for node in auto_nodes:
        candidate_id = int(node["candidate_id"])
        states = []
        if "same_class_relation" in tags[candidate_id]:
            states.append("same_class_relation_observed")
        if "cross_class_relation" in tags[candidate_id]:
            states.append("cross_class_relation_observed")
        if native_neighbors[candidate_id]:
            states.append("native_related")
        if node["independent_from_native_point_count"]:
            states.append("independent_from_native_observed")
        if "cross_view_evidence_without_geometric_overlap" in tags[candidate_id]:
            states.append("cross_view_complementarity_hypothesis")
        if "same_frame_granularity_evidence" in tags[candidate_id]:
            states.append("same_frame_granularity_observed")
        records.append({
            "family_id": f"auto_candidate_{candidate_id}",
            "anchor_candidate_id": candidate_id,
            "automatic_neighbor_candidate_ids": sorted(auto_neighbors[candidate_id]),
            "native_neighbor_candidate_indices": sorted(native_neighbors[candidate_id]),
            "audit_states": states or ["isolated_automatic_candidate"],
            "decision_state": "仅记录候选族关系；不保留、抑制、合并、删除、改类别或输出预测。",
        })
    return records


def build_scene_graph(candidates, scene_root, native_masks, native_classes, native_scores, nodes, cross_view_edges, same_frame_relations, quality_rows, semantic_rows):
    """构造单场景的全量非零几何边和跨视角证据边。"""
    if native_masks.ndim != 2 or native_masks.shape[1] != len(native_classes) or len(native_classes) != len(native_scores):
        raise ValueError("native mask/class/score 维度不一致")
    point_count = native_masks.shape[0]
    auto_matrix, point_lists = _candidate_points(scene_root, candidates, point_count)
    native_sparse = sparse.csc_matrix(native_masks.astype(np.int32, copy=False))
    native_union = np.flatnonzero(np.any(native_masks, axis=1))
    quality_by_id = {str(row["variant_id"]): row for row in quality_rows}
    semantic_by_id = {str(row["variant_id"]): row for row in semantic_rows}
    auto_nodes = _auto_nodes(candidates, quality_by_id, semantic_by_id, point_lists, native_union)
    by_track = {int(node["source_track_id"]): node for node in auto_nodes}
    cross_stats, hierarchy_stats = build_track_pair_evidence(
        nodes, cross_view_edges, same_frame_relations, set(by_track)
    )

    auto_relations = []
    intersections = (auto_matrix @ auto_matrix.T).tocoo()
    for left, right, intersection in zip(intersections.row, intersections.col, intersections.data):
        if left >= right:
            continue
        left_node, right_node = auto_nodes[int(left)], auto_nodes[int(right)]
        track_pair = tuple(sorted((left_node["source_track_id"], right_node["source_track_id"])))
        evidence = _pair_evidence(track_pair, cross_stats, hierarchy_stats)
        geometry = geometry_relation(left_node["point_count"], right_node["point_count"], int(intersection))
        auto_relations.append({
            "relation_kind": "automatic_to_automatic",
            "left_candidate_id": left_node["candidate_id"],
            "right_candidate_id": right_node["candidate_id"],
            "left_source_track_id": left_node["source_track_id"],
            "right_source_track_id": right_node["source_track_id"],
            "same_class": left_node["class_id"] == right_node["class_id"],
            "semantic_js_divergence": semantic_js_divergence(left_node, right_node),
            "geometry": geometry,
            "cross_view_and_granularity_evidence": evidence,
            "relation_tags": _relation_tags(left_node["class_id"] == right_node["class_id"], geometry, evidence),
            "decision_state": "连续关系记录；不按阈值竞争或改写候选。",
        })
    # 保留跨视角正证据但点集尚不重叠的候选对，供互补性审计。
    recorded_pairs = {
        tuple(sorted((item["left_source_track_id"], item["right_source_track_id"])))
        for item in auto_relations
    }
    for track_pair, stats in sorted(cross_stats.items()):
        if track_pair in recorded_pairs or track_pair[0] not in by_track or track_pair[1] not in by_track:
            continue
        left_node, right_node = by_track[track_pair[0]], by_track[track_pair[1]]
        geometry = geometry_relation(left_node["point_count"], right_node["point_count"], 0)
        evidence = _pair_evidence(track_pair, cross_stats, hierarchy_stats)
        auto_relations.append({
            "relation_kind": "automatic_to_automatic",
            "left_candidate_id": left_node["candidate_id"], "right_candidate_id": right_node["candidate_id"],
            "left_source_track_id": left_node["source_track_id"], "right_source_track_id": right_node["source_track_id"],
            "same_class": left_node["class_id"] == right_node["class_id"],
            "semantic_js_divergence": semantic_js_divergence(left_node, right_node),
            "geometry": geometry, "cross_view_and_granularity_evidence": evidence,
            "relation_tags": _relation_tags(left_node["class_id"] == right_node["class_id"], geometry, evidence),
            "decision_state": "跨视角但无点重叠的连续关系；不产生合并。",
        })

    native_relations, native_neighbors = [], {}
    auto_native = (auto_matrix @ native_sparse).tocoo()
    for auto_index, native_index, intersection in zip(auto_native.row, auto_native.col, auto_native.data):
        node = auto_nodes[int(auto_index)]
        native_index = int(native_index)
        native_size = int(native_masks[:, native_index].sum())
        geometry = geometry_relation(node["point_count"], native_size, int(intersection))
        native_neighbors[native_index] = {
            "node_kind": "frozen_native_candidate",
            "native_candidate_index": native_index,
            "class_id": int(native_classes[native_index]),
            "score": float(native_scores[native_index]),
            "point_count": native_size,
            "node_role": "冻结 native 邻居；不由本图修改。",
        }
        native_relations.append({
            "relation_kind": "automatic_to_native",
            "automatic_candidate_id": node["candidate_id"],
            "automatic_source_track_id": node["source_track_id"],
            "native_candidate_index": native_index,
            "same_class": node["class_id"] == int(native_classes[native_index]),
            "geometry": geometry,
            "automatic_independent_from_full_native_ratio": node["independent_from_native_ratio"],
            "relation_tags": ["geometric_overlap", "same_class_relation" if node["class_id"] == int(native_classes[native_index]) else "cross_class_relation"],
            "decision_state": "native 只作为冻结关系节点；不删除、替换或降权。",
        })
    families = _family_records(auto_nodes, auto_relations, native_relations)
    return auto_nodes, list(native_neighbors.values()), auto_relations, native_relations, families


def _scene_inputs(scene_name, args):
    scene_root = args.candidate_root / scene_name
    candidates = json.loads((scene_root / "backprojection_candidates.json").read_text())["candidates"]
    prefix = args.native_prediction_cache / f"{scene_name}_pred_"
    native_masks = np.asarray(np.load(str(prefix) + "masks.npy", mmap_mode="r"), dtype=bool)
    native_classes = np.asarray(np.load(str(prefix) + "classes.npy", mmap_mode="r"), dtype=np.int64)
    native_scores = np.asarray(np.load(str(prefix) + "scores.npy", mmap_mode="r"), dtype=np.float32)
    graph_root = args.evidence_graph_root / scene_name
    return (
        candidates, scene_root, native_masks, native_classes, native_scores,
        _read_jsonl(graph_root / "nodes.jsonl"), _read_jsonl(graph_root / "cross_view_edges.jsonl"),
        _read_jsonl(graph_root / "same_frame_relations.jsonl"),
        json.loads((args.quality_ledger_root / scene_name / "automatic_sam_variant_quality_ledger.json").read_text()),
        json.loads((args.semantic_ledger_root / scene_name / "automatic_sam_variant_semantic_ledger.json").read_text()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--evidence-graph-root", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--semantic-ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "candidate_root", "native_prediction_cache", "evidence_graph_root", "quality_ledger_root", "semantic_ledger_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True)
    totals = Counter()
    for index, scene_name in enumerate(scenes, start=1):
        records = build_scene_graph(*_scene_inputs(scene_name, args))
        auto_nodes, native_nodes, auto_relations, native_relations, families = records
        scene_output = args.output_root / scene_name
        scene_output.mkdir()
        _write_jsonl(scene_output / "automatic_candidate_nodes.jsonl", auto_nodes)
        _write_jsonl(scene_output / "native_neighbor_nodes.jsonl", native_nodes)
        _write_jsonl(scene_output / "automatic_to_automatic_relations.jsonl", auto_relations)
        _write_jsonl(scene_output / "automatic_to_native_relations.jsonl", native_relations)
        (scene_output / "local_candidate_families.json").write_text(json.dumps(families, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        totals.update({"automatic_candidate_count": len(auto_nodes), "native_neighbor_count": len(native_nodes), "automatic_relation_count": len(auto_relations), "native_relation_count": len(native_relations), "local_family_count": len(families)})
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: 自动关系 {len(auto_relations)}，native 关系 {len(native_relations)}", flush=True)
    payload = {
        "purpose": "保留自动 Pareto 候选与冻结 native 的候选族连续关系。",
        "gt_usage": "none",
        "decision_state": "不筛除、不合并、不改类别、不写回 native、不输出预测或 AP。",
        "scene_count": len(scenes), **dict(totals), "params": vars(args),
    }
    (args.output_root / "candidate_family_graph_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

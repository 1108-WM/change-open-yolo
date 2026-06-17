import argparse
import gc
import json
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = osp.dirname(osp.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import numpy as np
import torch
from tqdm import tqdm

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from run_evaluation import load_yaml
from export_sam_fused_proposals import (
    _clamp_box,
    _erode_binary_mask,
    _existing_mask_metrics,
    _filter_seed_indices_by_adaptive_internal_seed,
    _filter_seed_indices_by_depth_cluster,
    _label_consensus_metrics,
    _load_geometry_discriminator,
    _load_sam_predictor,
    _load_scene_points_xyz,
    _merge_label_consensus,
    _parse_class_names,
    _prepare_image,
    _predict_geometry_discriminator_score,
    _resolve_scene_names,
    _safe_label,
    _sam_mask_discriminator_row,
    _sam_mask_to_indices,
    _sam_seed_geometry_quality,
    _sam_view_quality_score,
    _save_mask,
    _save_overlay,
    _select_2d_nms_indices,
    _to_numpy,
)
from utils import OpenYolo3D


def _seed_overlap(left, right):
    if len(left) == 0 or len(right) == 0:
        return 0, 0, 0.0, 0.0
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    intersection = int(np.intersect1d(left, right, assume_unique=False).size)
    union = int(len(left) + len(right) - intersection)
    iou = float(intersection / max(1, union))
    containment = float(intersection / max(1, min(len(left), len(right))))
    return intersection, union, iou, containment


def _seed_centroid(points_xyz, seed_indices):
    if points_xyz is None or len(seed_indices) == 0:
        return None
    return points_xyz[np.asarray(seed_indices, dtype=np.int64)].mean(axis=0)


def _safe_mean(values, default=0.0):
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return float(default)
    return float(np.mean(values))


def _graph_quality_score(observation):
    return float(
        0.35 * observation.get("view_quality_score", 0.0)
        + 0.25 * observation.get("sam_score", 0.0)
        + 0.20 * observation.get("score", 0.0)
        + 0.10 * observation.get("seed_depth_support_ratio", 0.0)
        + 0.10 * observation.get("sam_mask_geometry", {}).get("quality_score", 0.0)
    )


def _cluster_reference_key(observation, min_reference_coverage):
    ref_id = observation.get("best_existing_mask_id")
    ref_coverage = float(observation.get("best_existing_seed_coverage", 0.0) or 0.0)
    if ref_id is None or ref_coverage < float(min_reference_coverage):
        return None
    return int(ref_id)


def _visible_seed_stats(seed_indices, visible_mask):
    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    if len(seed_indices) == 0:
        return np.asarray([], dtype=np.int64), 0, 0.0
    visible_seed_indices = seed_indices[visible_mask]
    visible_count = int(len(visible_seed_indices))
    visible_ratio = float(visible_count / max(1, len(seed_indices)))
    return visible_seed_indices.astype(np.int64), visible_count, visible_ratio


def _relation_from_pair(
    left,
    right,
    reference_counts,
    min_seed_iou,
    min_seed_containment,
    min_reference_coverage,
    spatial_sigma,
    view_consensus_scale,
):
    left_visible = left.get("_visible_seed_indices", left["_seed_indices"])
    right_visible = right.get("_visible_seed_indices", right["_seed_indices"])
    visible_intersection, visible_union, visible_iou, visible_containment = _seed_overlap(left_visible, right_visible)

    left_ref = _cluster_reference_key(left, min_reference_coverage)
    right_ref = _cluster_reference_key(right, min_reference_coverage)
    reference_match = left_ref is not None and left_ref == right_ref
    reference_support = 0.0
    if reference_match:
        reference_support = min(
            1.0,
            float(reference_counts.get((left["class_id"], left_ref), 0)) / max(1.0, float(view_consensus_scale)),
        )
        reference_support = max(
            reference_support,
            min(
                1.0,
                min(
                    float(left.get("best_existing_seed_coverage", 0.0) or 0.0),
                    float(right.get("best_existing_seed_coverage", 0.0) or 0.0),
                ),
            ),
        )

    left_depth_support = float(left.get("seed_depth_support_ratio", 0.0))
    right_depth_support = float(right.get("seed_depth_support_ratio", 0.0))
    pair_depth_support = min(left_depth_support, right_depth_support)

    spatial_consistency = 0.0
    left_centroid = left.get("seed_centroid")
    right_centroid = right.get("seed_centroid")
    if left_centroid is not None and right_centroid is not None:
        distance = float(np.linalg.norm(np.asarray(left_centroid) - np.asarray(right_centroid)))
        spatial_consistency = float(np.exp(-distance / max(1e-6, float(spatial_sigma))))

    view_consensus = max(reference_support, min(1.0, (int(reference_match) + visible_containment) / 2.0))
    support_score = float(
        0.30 * min(1.0, visible_iou / max(1e-6, float(min_seed_iou)))
        + 0.20 * min(1.0, visible_containment / max(1e-6, float(min_seed_containment)))
        + 0.20 * reference_support
        + 0.20 * spatial_consistency
        + 0.10 * pair_depth_support
    )
    same_object_support = (
        visible_iou >= float(min_seed_iou)
        or reference_match
        or support_score >= 0.45
    )

    left_count = len(left_visible)
    right_count = len(right_visible)
    size_ratio = float(max(left_count, right_count) / max(1, min(left_count, right_count)))
    if left_count >= right_count:
        parent_idx, child_idx = "left", "right"
        parent_count, child_count = left_count, right_count
    else:
        parent_idx, child_idx = "right", "left"
        parent_count, child_count = right_count, left_count
    containment_support = visible_containment >= float(min_seed_containment) and size_ratio >= 1.25
    containment_strength = float(max(visible_containment, support_score))

    return {
        "same_object_support": bool(same_object_support),
        "containment_support": bool(containment_support),
        "containment_parent": parent_idx,
        "containment_parent_count": int(parent_count),
        "containment_child_count": int(child_count),
        "containment_strength": containment_strength,
        "visible_intersection": int(visible_intersection),
        "visible_union": int(visible_union),
        "visible_iou": float(visible_iou),
        "visible_containment": float(visible_containment),
        "support_score": support_score,
        "reference_match": bool(reference_match),
        "reference_support": float(reference_support),
        "depth_support": float(pair_depth_support),
        "spatial_consistency": float(spatial_consistency),
        "view_consensus": float(view_consensus),
        "same_class": int(left["class_id"]) == int(right["class_id"]),
        "class_mismatch": int(left["class_id"]) != int(right["class_id"]),
        "left_reference_id": int(left_ref) if left_ref is not None else None,
        "right_reference_id": int(right_ref) if right_ref is not None else None,
    }


def _build_mask_graph(
    observations,
    points_xyz=None,
    same_class_only=True,
    min_seed_iou=0.03,
    min_seed_containment=0.18,
    min_reference_coverage=0.20,
    spatial_sigma=0.35,
    view_consensus_scale=4.0,
    edge_score_threshold=0.35,
    weak_edge_threshold=None,
    conflict_edge_threshold=None,
):
    reference_counts = defaultdict(int)
    for obs in observations:
        key = (obs["class_id"], _cluster_reference_key(obs, min_reference_coverage))
        if key[1] is not None:
            reference_counts[key] += 1

    centroids = [_seed_centroid(points_xyz, obs["_seed_indices"]) for obs in observations]
    for obs, centroid in zip(observations, centroids):
        obs["seed_centroid"] = centroid

    relation_edges = []
    support_edges = []
    weak_edges = []
    conflict_edges = []
    adjacency = [[] for _ in observations]
    weak_threshold = float(max(0.15, edge_score_threshold * 0.70)) if weak_edge_threshold is None else float(weak_edge_threshold)
    conflict_threshold = float(max(0.20, edge_score_threshold * 0.60)) if conflict_edge_threshold is None else float(conflict_edge_threshold)
    for left_idx in range(len(observations)):
        left = observations[left_idx]
        for right_idx in range(left_idx + 1, len(observations)):
            right = observations[right_idx]
            relation = _relation_from_pair(
                left,
                right,
                reference_counts=reference_counts,
                min_seed_iou=min_seed_iou,
                min_seed_containment=min_seed_containment,
                min_reference_coverage=min_reference_coverage,
                spatial_sigma=spatial_sigma,
                view_consensus_scale=view_consensus_scale,
            )
            if int(left["frame_index"]) == int(right["frame_index"]):
                continue

            if relation["class_mismatch"]:
                if relation["reference_match"] or relation["support_score"] >= conflict_threshold:
                    conflict_edge = {
                        "left": int(left_idx),
                        "right": int(right_idx),
                        "relation_type": "conflict",
                        "same_object_score": 0.0,
                        "containment_score": 0.0,
                        "conflict_score": float(max(relation["support_score"], relation["spatial_consistency"], relation["reference_support"])),
                        "edge_score": float(max(relation["support_score"], relation["spatial_consistency"], relation["reference_support"])),
                        "seed_iou": float(relation["visible_iou"]),
                        "seed_containment": float(relation["visible_containment"]),
                        "coarse_reference_overlap": float(relation["reference_support"]),
                        "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                        "depth_consistency": float(relation["depth_support"]),
                        "view_consensus_score": float(relation["view_consensus"]),
                        "same_class": False,
                    }
                    conflict_edges.append(conflict_edge)
                    relation_edges.append(conflict_edge)
                continue

            strong_same_object = relation["same_object_support"] and relation["support_score"] >= float(edge_score_threshold)
            if not strong_same_object and not relation["containment_support"]:
                if relation["support_score"] >= weak_threshold:
                    weak_edge = {
                        "left": int(left_idx),
                        "right": int(right_idx),
                        "relation_type": "weak",
                        "same_object_score": float(relation["support_score"]),
                        "containment_score": float(relation["containment_strength"]) if relation["containment_support"] else 0.0,
                        "conflict_score": 0.0,
                        "relation_score": float(relation["support_score"]),
                        "edge_score": float(relation["support_score"]),
                        "seed_intersection": int(relation["visible_intersection"]),
                        "seed_union": int(relation["visible_union"]),
                        "seed_iou": float(relation["visible_iou"]),
                        "seed_containment": float(relation["visible_containment"]),
                        "class_compatible": True,
                        "coarse_reference_overlap": float(relation["reference_support"]),
                        "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                        "depth_consistency": float(relation["depth_support"]),
                        "view_consensus_score": float(relation["view_consensus"]),
                        "same_class": True,
                    }
                    weak_edges.append(weak_edge)
                    relation_edges.append(weak_edge)
                continue

            relation_type = "same_object" if strong_same_object else "containment"
            relation_edge = {
                "left": int(left_idx),
                "right": int(right_idx),
                "relation_type": relation_type,
                "same_object_score": float(relation["support_score"]) if relation["same_object_support"] else 0.0,
                "containment_score": float(relation["containment_strength"]) if relation["containment_support"] else 0.0,
                "conflict_score": 0.0,
                "relation_score": float(relation["support_score"]),
                "edge_score": float(relation["support_score"]),
                "seed_intersection": int(relation["visible_intersection"]),
                "seed_union": int(relation["visible_union"]),
                "seed_iou": float(relation["visible_iou"]),
                "seed_containment": float(relation["visible_containment"]),
                "class_compatible": True,
                "coarse_reference_overlap": float(relation["reference_support"]),
                "coarse_reference_id": relation["left_reference_id"] if relation["reference_match"] else None,
                "depth_consistency": float(relation["depth_support"]),
                "view_consensus_score": float(relation["view_consensus"]),
                "same_class": True,
            }
            if relation["containment_support"]:
                if relation["containment_parent"] == "left":
                    relation_edge["parent"] = int(left_idx)
                    relation_edge["child"] = int(right_idx)
                else:
                    relation_edge["parent"] = int(right_idx)
                    relation_edge["child"] = int(left_idx)
                relation_edge["containment_strength"] = float(relation["containment_strength"])
            relation_edges.append(relation_edge)
            if relation_type == "same_object":
                support_edges.append(relation_edge)
                adjacency[left_idx].append((right_idx, relation_edge))
                adjacency[right_idx].append((left_idx, relation_edge))
    return relation_edges, adjacency, conflict_edges, weak_edges


def _connected_components(num_nodes, adjacency):
    visited = np.zeros((num_nodes,), dtype=bool)
    components = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor, _ in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _component_edge_stats(component, adjacency, relation_edges=None):
    component_set = set(component)
    edge_scores = []
    seed_ious = []
    containments = []
    depth_scores = []
    consensus_scores = []
    relation_kind_counts = defaultdict(int)
    relation_kind_scores = defaultdict(list)
    seen = set()
    for node in component:
        for neighbor, edge in adjacency[node]:
            if neighbor not in component_set:
                continue
            key = tuple(sorted((node, neighbor)))
            if key in seen:
                continue
            seen.add(key)
            edge_scores.append(edge["edge_score"])
            seed_ious.append(edge["seed_iou"])
            containments.append(edge["seed_containment"])
            depth_scores.append(edge["depth_consistency"])
            consensus_scores.append(edge["view_consensus_score"])
            relation_kind = edge.get("relation_type", "same_object")
            relation_kind_counts[relation_kind] += 1
            relation_kind_scores[relation_kind].append(float(edge.get("relation_score", edge.get("edge_score", 0.0))))
            if float(edge.get("containment_score", 0.0) or 0.0) > 0.0:
                relation_kind_counts["containment"] += 1
                relation_kind_scores["containment"].append(float(edge.get("containment_score", edge.get("containment_strength", 0.0))))
    conflict_count = 0
    weak_count = 0
    if relation_edges is not None:
        for edge in relation_edges:
            if edge["left"] not in component_set or edge["right"] not in component_set:
                continue
            relation_kind = edge.get("relation_type", "same_object")
            if relation_kind == "conflict":
                conflict_count += 1
                relation_kind_scores[relation_kind].append(float(edge.get("conflict_score", edge.get("edge_score", 0.0))))
            elif relation_kind == "weak":
                weak_count += 1
                relation_kind_scores[relation_kind].append(float(edge.get("relation_score", edge.get("edge_score", 0.0))))
            elif relation_kind == "containment":
                relation_kind_counts["containment"] += 1
                relation_kind_scores["containment"].append(float(edge.get("containment_score", edge.get("containment_strength", 0.0))))
    return {
        "edge_count": int(len(edge_scores)),
        "edge_mean_score": _safe_mean(edge_scores),
        "seed_mean_iou": _safe_mean(seed_ious),
        "seed_mean_containment": _safe_mean(containments),
        "depth_consistency_score": _safe_mean(depth_scores),
        "graph_consensus_score": _safe_mean(consensus_scores),
        "same_object_edge_count": int(relation_kind_counts.get("same_object", 0)),
        "containment_edge_count": int(relation_kind_counts.get("containment", 0)),
        "weak_edge_count": int(max(weak_count, relation_kind_counts.get("weak", 0))),
        "conflict_edge_count": int(max(conflict_count, relation_kind_counts.get("conflict", 0))),
        "same_object_edge_mean_score": _safe_mean(relation_kind_scores.get("same_object", [])),
        "containment_edge_mean_score": _safe_mean(relation_kind_scores.get("containment", [])),
        "weak_edge_mean_score": _safe_mean(relation_kind_scores.get("weak", [])),
        "conflict_edge_mean_score": _safe_mean(relation_kind_scores.get("conflict", [])),
    }


def _select_cluster_views(
    observations,
    component,
    adjacency,
    max_views=4,
    min_new_seed_ratio=0.05,
    redundancy_penalty=0.20,
):
    ranked = sorted(component, key=lambda idx: (-_graph_quality_score(observations[idx]), -len(observations[idx]["_seed_indices"])))
    selected = []
    selected_seed = np.asarray([], dtype=np.int64)
    max_views = int(max(1, max_views))
    while ranked and len(selected) < max_views:
        best_idx = None
        best_score = None
        best_seed = None
        for idx in ranked:
            seed = observations[idx]["_seed_indices"]
            if len(selected_seed) == 0:
                new_seed_ratio = 1.0
                redundancy = 0.0
            else:
                intersection = np.intersect1d(selected_seed, seed, assume_unique=False).size
                new_seed_ratio = float((len(seed) - intersection) / max(1, len(seed)))
                redundancy = float(intersection / max(1, min(len(selected_seed), len(seed))))
            if selected and new_seed_ratio < float(min_new_seed_ratio):
                continue
            graph_support = 0.0
            for neighbor, edge in adjacency[idx]:
                if neighbor in selected:
                    graph_support = max(graph_support, float(edge.get("edge_score", 0.0)))
            score = (
                0.45 * _graph_quality_score(observations[idx])
                + 0.30 * new_seed_ratio
                + 0.20 * graph_support
                - float(redundancy_penalty) * redundancy
                + 0.05 * min(1.0, math.log1p(len(seed)) / math.log(10000.0))
            )
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
                best_seed = seed
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_seed = np.union1d(selected_seed, best_seed).astype(np.int64)
        ranked = [idx for idx in ranked if idx != best_idx]

    if not selected and component:
        selected = [component[0]]
        selected_seed = observations[component[0]]["_seed_indices"].copy()
    return selected, selected_seed.astype(np.int64)


def _merge_cluster_label_consensus(cluster_record, observations, selected_indices):
    first = True
    for idx in selected_indices:
        obs = observations[idx]
        label_record = {
            key: obs.get(key)
            for key in (
                "label_consensus_score",
                "label_conflict_score",
                "label_margin",
                "label_entropy",
                "label_consensus_view_count",
                "label_conflict_view_count",
                "label_evidence_view_count",
                "label_target_evidence",
                "label_total_evidence",
                "top_conflicting_class_id",
                "top_conflicting_evidence",
            )
            if key in obs
        }
        if first:
            cluster_record.update(label_record)
            first = False
        else:
            _merge_label_consensus(cluster_record, label_record)
    return cluster_record


def _cluster_to_candidate(
    cluster_id,
    observations,
    component,
    selected_indices,
    selected_seed,
    adjacency,
    relation_edges,
    existing_masks,
):
    selected_observations = [observations[idx] for idx in selected_indices]
    best_observation = max(
        selected_observations,
        key=lambda obs: (_graph_quality_score(obs), obs.get("proposal_priority", 0.0), len(obs["_seed_indices"])),
    )
    class_votes = defaultdict(float)
    for idx in component:
        obs = observations[idx]
        class_votes[int(obs["class_id"])] += float(obs.get("fusion_score", obs.get("score", 0.0))) * max(
            0.1,
            float(obs.get("sam_score", 0.0)),
        )
    class_id = max(class_votes.items(), key=lambda item: item[1])[0]
    class_name = best_observation["class_name"] if int(best_observation["class_id"]) == int(class_id) else best_observation["class_name"]
    stats = _component_edge_stats(component, adjacency, relation_edges=relation_edges)
    existing_metrics = _existing_mask_metrics(existing_masks, selected_seed)
    support_views = []
    for rank, obs in enumerate(selected_observations):
        view = dict(obs["support_views"][0])
        view["graph_observation_id"] = int(obs["graph_observation_id"])
        view["graph_selected_rank"] = int(rank)
        support_views.append(view)

    conflict_penalty = max(0.50, 1.0 - 0.20 * min(2, int(stats["conflict_edge_count"])))
    candidate = {
        "scene_name": best_observation["scene_name"],
        "source_kind": "mask_graph_multi_view" if int(len(component)) >= 2 and int(stats["edge_count"]) > 0 else "mask_graph_single_view",
        "frame_id": best_observation["frame_id"],
        "frame_index": int(best_observation["frame_index"]),
        "detection_id": int(best_observation["detection_id"]),
        "sam_mask_id": int(best_observation["sam_mask_id"]),
        "sam_mask_rank": int(best_observation.get("sam_mask_rank", 0)),
        "sam_score_rank": int(best_observation.get("sam_score_rank", 0)),
        "class_id": int(class_id),
        "class_name": class_name,
        "score": float(max(obs.get("score", 0.0) for obs in selected_observations)),
        "bbox_xyxy": best_observation["bbox_xyxy"],
        "box_area_ratio": float(best_observation.get("box_area_ratio", 0.0)),
        "fusion_score": float(_safe_mean([obs.get("fusion_score", obs.get("score", 0.0)) for obs in selected_observations])),
        "proposal_priority": float(
            max(0.05, stats["graph_consensus_score"])
            * max(obs.get("proposal_priority", 0.0) for obs in selected_observations)
            * (0.5 + 0.5 * min(1.0, len(selected_indices) / 4.0))
            * conflict_penalty
        ),
        "support_view_count": int(len(selected_indices)),
        "support_mean_iou": float(stats["seed_mean_iou"]),
        "support_best_iou": float(
            max(
                [edge["seed_iou"] for idx in component for _, edge in adjacency[idx] if edge["right"] in component or edge["left"] in component]
                or [0.0]
            )
        ),
        "support_mean_score": float(_safe_mean([obs.get("score", 0.0) for obs in selected_observations])),
        "support_best_score": float(max(obs.get("score", 0.0) for obs in selected_observations)),
        "view_quality_score": float(_safe_mean([obs.get("view_quality_score", 0.0) for obs in selected_observations])),
        "support_views": support_views,
        "merged_observations": int(len(component)),
        "selected_seed_view_count": int(len(selected_indices)),
        "available_seed_view_count": int(len(component)),
        "graph_cluster_id": int(cluster_id),
        "graph_cluster_observation_ids": [int(observations[idx]["graph_observation_id"]) for idx in component],
        "graph_selected_observation_ids": [int(observations[idx]["graph_observation_id"]) for idx in selected_indices],
        "cluster_observation_count": int(len(component)),
        "selected_view_count": int(len(selected_indices)),
        "graph_edge_count": int(stats["edge_count"]),
        "graph_edge_mean_score": float(stats["edge_mean_score"]),
        "graph_consensus_score": float(stats["graph_consensus_score"]),
        "depth_consistency_score": float(stats["depth_consistency_score"]),
        "seed_mean_containment": float(stats["seed_mean_containment"]),
        "same_object_edge_count": int(stats["same_object_edge_count"]),
        "containment_edge_count": int(stats["containment_edge_count"]),
        "weak_edge_count": int(stats["weak_edge_count"]),
        "conflict_edge_count": int(stats["conflict_edge_count"]),
        "same_object_edge_mean_score": float(stats["same_object_edge_mean_score"]),
        "containment_edge_mean_score": float(stats["containment_edge_mean_score"]),
        "weak_edge_mean_score": float(stats["weak_edge_mean_score"]),
        "conflict_edge_mean_score": float(stats["conflict_edge_mean_score"]),
        "conflict_penalty": float(conflict_penalty),
        "sam_mask_selection_policy": best_observation.get("sam_mask_selection_policy"),
        "sam_mask_selection_score": best_observation.get("sam_mask_selection_score"),
        "sam_mask_geometry": best_observation.get("sam_mask_geometry"),
        "evidence": best_observation.get("evidence", {}),
        "_seed_indices": selected_seed,
        **existing_metrics,
    }
    _merge_cluster_label_consensus(candidate, observations, selected_indices)
    return candidate


def collect_scene_mask_observations(
    openyolo3d,
    predictor,
    scene_name,
    output_dir,
    detection_score_th=0.45,
    min_seed_points=80,
    max_box_area_ratio=0.30,
    frame_stride=5,
    max_frames=None,
    max_detections_per_frame=8,
    blocked_classes=None,
    sam_multimask_topk=1,
    sam_mask_selection_policy="sam_score",
    sam_mask_geometry_model=None,
    sam_mask_geometry_cc_radius=0.03,
    sam_mask_geometry_plane_threshold=0.02,
    sam_mask_geometry_max_points=50000,
    seed_depth_cluster=False,
    seed_depth_cluster_bin_size=0.10,
    seed_depth_cluster_window_bins=1,
    seed_depth_cluster_min_keep_ratio=0.25,
    seed_depth_cluster_min_removed_ratio=0.0,
    seed_depth_cluster_max_removed_ratio=1.0,
    sam_adaptive_internal_seed=False,
    sam_adaptive_internal_keep_ratio=0.70,
    sam_adaptive_internal_min_keep_ratio=0.35,
    sam_adaptive_internal_boundary_weight=0.45,
    sam_adaptive_internal_depth_weight=0.55,
    sam_adaptive_internal_depth_bin_size=0.10,
    sam_adaptive_internal_depth_window_bins=1,
    sam_mask_erode_pixels=0,
    sam_mask_erode_min_area_ratio=0.15,
    label_consensus_iou_th=0.25,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
):
    labels = openyolo3d.openyolo3d_config["network2d"]["text_prompts"]
    blocked_classes = _parse_class_names(blocked_classes)
    projections, keep_visible_points = openyolo3d.mesh_projections
    projections_np = _to_numpy(projections).astype(np.int64)
    visible_np = _to_numpy(keep_visible_points).astype(bool)
    existing_masks = _to_numpy(openyolo3d.preds_3d[0]).astype(bool)
    if existing_masks.shape[0] != projections_np.shape[1]:
        existing_masks = existing_masks.T
    points_xyz = _load_scene_points_xyz(openyolo3d)
    geometry_model_bundle = sam_mask_geometry_model
    if isinstance(geometry_model_bundle, str):
        geometry_model_bundle = _load_geometry_discriminator(geometry_model_bundle)
    if str(sam_mask_selection_policy) == "learned_geometry" and geometry_model_bundle is None:
        raise ValueError("--sam_mask_geometry_model is required when using learned_geometry selection.")

    scene_dir = osp.join(output_dir, scene_name)
    image_dir = osp.join(scene_dir, "mask_graph_images")
    mask_dir = osp.join(scene_dir, "mask_graph_masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_height, image_width = openyolo3d.world2cam.image_resolution
    frame_indices = list(range(0, len(openyolo3d.world2cam.color_paths), max(1, int(frame_stride))))
    if max_frames is not None:
        frame_indices = frame_indices[: int(max_frames)]

    observations = []
    skipped = []
    raw_observations = 0
    for frame_idx in frame_indices:
        image_path = openyolo3d.world2cam.color_paths[frame_idx]
        frame_id = osp.basename(image_path).split(".")[0]
        frame_pred = openyolo3d.preds_2d.get(frame_id)
        if frame_pred is None:
            continue
        boxes = _to_numpy(frame_pred["bbox"]).astype(np.float32)
        class_ids = _to_numpy(frame_pred["labels"]).astype(np.int64)
        scores = _to_numpy(frame_pred["scores"]).astype(np.float32)
        nms_indices = _select_2d_nms_indices(
            boxes,
            scores,
            class_ids,
            iou_threshold=box_nms_iou,
            same_class_only=box_nms_same_class_only,
        )
        order = nms_indices[np.argsort(-scores[nms_indices])[:max_detections_per_frame]]

        image = _prepare_image(image_path)
        predictor.set_image(image)
        for det_rank, det_id in enumerate(order):
            class_id = int(class_ids[det_id])
            class_name = _safe_label(labels, class_id)
            score = float(scores[det_id])
            if class_name in blocked_classes:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "class_blocked"})
                continue
            if score < detection_score_th:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "low_score"})
                continue
            box = np.asarray(_clamp_box(boxes[det_id], image_width, image_height), dtype=np.float32)
            box_area_ratio = float(((box[2] - box[0]) * (box[3] - box[1])) / max(1, image_width * image_height))
            if box_area_ratio > max_box_area_ratio:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "large_2d_box"})
                continue

            masks, sam_scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            sam_score_order = np.argsort(-sam_scores)
            sam_score_rank_by_id = {int(mask_id): int(rank) for rank, mask_id in enumerate(sam_score_order)}
            mask_order = sam_score_order if str(sam_mask_selection_policy) in {"geometry", "learned_geometry"} else sam_score_order[: max(1, int(sam_multimask_topk))]
            mask_items = []
            for mask_rank, mask_id in enumerate(mask_order):
                mask_id = int(mask_id)
                sam_score = float(sam_scores[mask_id])
                sam_mask = masks[mask_id].astype(bool)
                core_mask, erosion_info = _erode_binary_mask(
                    sam_mask,
                    erode_pixels=sam_mask_erode_pixels,
                    min_area_ratio=sam_mask_erode_min_area_ratio,
                )
                seed_indices = _sam_mask_to_indices(openyolo3d, frame_idx, core_mask, projections_np, visible_np)
                adaptive_internal_info = {"enabled": False}
                if sam_adaptive_internal_seed:
                    seed_indices, adaptive_internal_info = _filter_seed_indices_by_adaptive_internal_seed(
                        openyolo3d,
                        frame_idx,
                        seed_indices,
                        core_mask,
                        projections_np,
                        keep_ratio=sam_adaptive_internal_keep_ratio,
                        min_keep_ratio=sam_adaptive_internal_min_keep_ratio,
                        boundary_weight=sam_adaptive_internal_boundary_weight,
                        depth_weight=sam_adaptive_internal_depth_weight,
                        depth_bin_size=sam_adaptive_internal_depth_bin_size,
                        depth_window_bins=sam_adaptive_internal_depth_window_bins,
                        min_points=min_seed_points,
                    )
                depth_cluster_info = {"enabled": False}
                if seed_depth_cluster:
                    seed_indices, depth_cluster_info = _filter_seed_indices_by_depth_cluster(
                        openyolo3d,
                        frame_idx,
                        seed_indices,
                        projections_np,
                        bin_size=seed_depth_cluster_bin_size,
                        window_bins=seed_depth_cluster_window_bins,
                        min_keep_ratio=seed_depth_cluster_min_keep_ratio,
                        min_removed_ratio=seed_depth_cluster_min_removed_ratio,
                        max_removed_ratio=seed_depth_cluster_max_removed_ratio,
                        min_points=min_seed_points,
                    )
                geometry_info = _sam_seed_geometry_quality(
                    seed_indices,
                    points_xyz,
                    existing_masks,
                    min_seed_points=min_seed_points,
                    cc_radius=sam_mask_geometry_cc_radius,
                    plane_threshold=sam_mask_geometry_plane_threshold,
                    max_points=sam_mask_geometry_max_points,
                )
                learned_geometry_score = None
                if str(sam_mask_selection_policy) == "learned_geometry":
                    existing_metrics = _existing_mask_metrics(existing_masks, seed_indices)
                    feature_row = _sam_mask_discriminator_row(
                        geometry_info,
                        existing_metrics,
                        detection_score=score,
                        sam_score=sam_score,
                        box_area_ratio=box_area_ratio,
                        num_seed_points=len(seed_indices),
                        num_mask_points=len(seed_indices),
                    )
                    learned_geometry_score = _predict_geometry_discriminator_score(feature_row, geometry_model_bundle)
                    selection_score = float(learned_geometry_score)
                else:
                    selection_score = float(0.45 * max(0.0, min(1.0, sam_score)) + 0.55 * float(geometry_info.get("quality_score", 0.0)))
                if len(seed_indices) < min_seed_points:
                    skipped.append(
                        {
                            "frame_id": frame_id,
                            "detection_id": int(det_id),
                            "sam_mask_id": mask_id,
                            "reason": "few_seed_points",
                            "num_seed_points": int(len(seed_indices)),
                            "sam_mask_geometry": geometry_info,
                        }
                    )
                    continue
                visible_seed_mask = visible_np[frame_idx][seed_indices]
                visible_seed_indices, visible_seed_count, visible_seed_ratio = _visible_seed_stats(seed_indices, visible_seed_mask)
                mask_items.append(
                    {
                        "mask_id": mask_id,
                        "sam_score": sam_score,
                        "core_mask": core_mask,
                        "erosion_info": erosion_info,
                        "seed_indices": seed_indices,
                        "adaptive_internal_info": adaptive_internal_info,
                        "depth_cluster_info": depth_cluster_info,
                        "geometry_info": geometry_info,
                        "learned_geometry_score": learned_geometry_score,
                        "selection_score": selection_score,
                        "sam_score_rank": int(sam_score_rank_by_id.get(mask_id, mask_rank)),
                        "sam_mask_rank": int(mask_rank),
                        "visible_seed_indices": visible_seed_indices,
                        "seed_visible_count": int(visible_seed_count),
                        "seed_depth_support_ratio": float(visible_seed_ratio),
                    }
                )

            if str(sam_mask_selection_policy) in {"geometry", "learned_geometry"}:
                mask_items = sorted(
                    mask_items,
                    key=lambda item: (
                        -float(item["selection_score"]),
                        -float(item["geometry_info"].get("quality_score", 0.0)),
                        -float(item["sam_score"]),
                        -len(item["seed_indices"]),
                    ),
                )[: max(1, int(sam_multimask_topk))]
                for rank, item in enumerate(mask_items):
                    item["sam_mask_rank"] = int(rank)

            if not mask_items:
                skipped.append({"frame_id": frame_id, "detection_id": int(det_id), "reason": "no_valid_sam_mask"})
                continue

            for item in mask_items:
                mask_id = int(item["mask_id"])
                sam_score = float(item["sam_score"])
                seed_indices = item["seed_indices"]
                visible_seed_indices = item["visible_seed_indices"]
                geometry_info = item["geometry_info"]
                learned_geometry_score = item.get("learned_geometry_score")
                priority_factor = 1.0
                if str(sam_mask_selection_policy) == "geometry":
                    priority_factor = 0.75 + 0.50 * float(geometry_info.get("quality_score", 0.0))
                elif str(sam_mask_selection_policy) == "learned_geometry":
                    priority_factor = 0.75 + 0.50 * float(learned_geometry_score or 0.0)
                view_quality_score = _sam_view_quality_score(
                    geometry_info,
                    detection_score=score,
                    sam_score=sam_score,
                    box_area_ratio=box_area_ratio,
                    num_seed_points=len(seed_indices),
                    min_seed_points=min_seed_points,
                )
                evidence_prefix = osp.join(
                    image_dir,
                    f"obs{raw_observations:05d}_frame{frame_id}_det{int(det_id):03d}_sam{mask_id}",
                )
                mask_prefix = osp.join(
                    mask_dir,
                    f"obs{raw_observations:05d}_frame{frame_id}_det{int(det_id):03d}_sam{mask_id}",
                )
                evidence = _save_overlay(image, item["core_mask"], box, evidence_prefix)
                mask_path = _save_mask(item["core_mask"], mask_prefix)
                observation = {
                    "scene_name": scene_name,
                    "graph_observation_id": int(raw_observations),
                    "frame_id": frame_id,
                    "frame_index": int(frame_idx),
                    "detection_id": int(det_id),
                    "sam_mask_id": mask_id,
                    "sam_mask_rank": int(item["sam_mask_rank"]),
                    "sam_score_rank": int(item["sam_score_rank"]),
                    "sam_score": sam_score,
                    "sam_mask_selection_policy": str(sam_mask_selection_policy),
                    "sam_mask_selection_score": float(item["selection_score"]),
                    "sam_mask_learned_geometry_score": learned_geometry_score,
                    "sam_mask_geometry": geometry_info,
                    "sam_mask_erosion": item["erosion_info"],
                    "sam_adaptive_internal_seed": item["adaptive_internal_info"],
                    "seed_depth_cluster": item["depth_cluster_info"],
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "box_area_ratio": box_area_ratio,
                    "num_seed_points": int(len(seed_indices)),
                    "seed_visible_count": int(item.get("seed_visible_count", 0)),
                    "seed_depth_support_ratio": float(item.get("seed_depth_support_ratio", 0.0)),
                    "proposal_priority": float(score * max(0.1, sam_score) * np.log1p(len(seed_indices)) * priority_factor),
                    "fusion_score": float(score * max(0.1, sam_score)),
                    "support_view_count": 1,
                    "support_mean_iou": 1.0,
                    "support_best_iou": 1.0,
                    "support_mean_score": score,
                    "support_best_score": score,
                    "view_quality_score": view_quality_score,
                    "support_views": [
                        {
                            "frame_id": frame_id,
                            "frame_index": int(frame_idx),
                            "visible_seed_points": int(item.get("seed_visible_count", len(seed_indices))),
                            "iou": 1.0,
                            "score": score,
                            "sam_score": sam_score,
                            "sam_mask_id": mask_id,
                            "sam_mask_rank": int(item["sam_mask_rank"]),
                            "sam_score_rank": int(item["sam_score_rank"]),
                            "bbox_xyxy": [float(v) for v in box.tolist()],
                            "sam_mask_path": mask_path,
                            "sam_mask_selection_policy": str(sam_mask_selection_policy),
                            "sam_mask_selection_score": float(item["selection_score"]),
                            "sam_mask_learned_geometry_score": learned_geometry_score,
                            "sam_mask_geometry_quality_score": float(geometry_info.get("quality_score", 0.0)),
                            "view_quality_score": view_quality_score,
                            "sam_mask_erode_pixels": int(sam_mask_erode_pixels or 0),
                            "sam_mask_core_area_ratio": float(item["erosion_info"].get("area_ratio", 1.0)),
                            "seed_depth_cluster_keep_ratio": float(item["depth_cluster_info"].get("keep_ratio", 1.0)),
                            "seed_depth_support_ratio": float(item.get("seed_depth_support_ratio", 0.0)),
                        }
                    ],
                    **_label_consensus_metrics(box, class_id, boxes, class_ids, scores, label_consensus_iou_th),
                    "evidence": {
                        "color_path": image_path,
                        "bbox_xyxy": [int(round(v)) for v in box.tolist()],
                        "sam_mask_path": mask_path,
                        **evidence,
                    },
                    "_seed_indices": seed_indices,
                    "_visible_seed_indices": visible_seed_indices,
                    **_existing_mask_metrics(existing_masks, seed_indices),
                }
                observations.append(observation)
                raw_observations += 1

    return observations, skipped, existing_masks, points_xyz


def export_scene_mask_graph_proposals(
    openyolo3d,
    predictor,
    scene_name,
    output_dir,
    detection_score_th=0.45,
    min_seed_points=80,
    max_box_area_ratio=0.30,
    frame_stride=5,
    max_frames=None,
    max_detections_per_frame=8,
    max_candidates=30,
    blocked_classes=None,
    ranking_policy="graph_priority",
    sam_multimask_topk=1,
    sam_mask_selection_policy="sam_score",
    sam_mask_geometry_model=None,
    sam_mask_geometry_cc_radius=0.03,
    sam_mask_geometry_plane_threshold=0.02,
    sam_mask_geometry_max_points=50000,
    seed_depth_cluster=False,
    seed_depth_cluster_bin_size=0.10,
    seed_depth_cluster_window_bins=1,
    seed_depth_cluster_min_keep_ratio=0.25,
    seed_depth_cluster_min_removed_ratio=0.0,
    seed_depth_cluster_max_removed_ratio=1.0,
    sam_adaptive_internal_seed=False,
    sam_adaptive_internal_keep_ratio=0.70,
    sam_adaptive_internal_min_keep_ratio=0.35,
    sam_adaptive_internal_boundary_weight=0.45,
    sam_adaptive_internal_depth_weight=0.55,
    sam_adaptive_internal_depth_bin_size=0.10,
    sam_adaptive_internal_depth_window_bins=1,
    sam_mask_erode_pixels=0,
    sam_mask_erode_min_area_ratio=0.15,
    label_consensus_iou_th=0.25,
    box_nms_iou=0.0,
    box_nms_same_class_only=True,
    graph_same_class_only=True,
    graph_min_seed_iou=0.03,
    graph_min_seed_containment=0.18,
    graph_min_reference_coverage=0.20,
    graph_spatial_sigma=0.35,
    graph_view_consensus_scale=4.0,
    graph_edge_score_threshold=0.35,
    graph_min_cluster_observations=2,
    graph_keep_singletons=False,
    graph_max_views_per_cluster=4,
    graph_min_new_seed_ratio=0.05,
    export_max_existing_iou=None,
    export_max_seed_in_existing_mask_ratio=None,
):
    scene_dir = osp.join(output_dir, scene_name)
    seed_dir = osp.join(scene_dir, "seed_points")
    os.makedirs(seed_dir, exist_ok=True)

    observations, skipped, existing_masks, points_xyz = collect_scene_mask_observations(
        openyolo3d,
        predictor,
        scene_name,
        output_dir,
        detection_score_th=detection_score_th,
        min_seed_points=min_seed_points,
        max_box_area_ratio=max_box_area_ratio,
        frame_stride=frame_stride,
        max_frames=max_frames,
        max_detections_per_frame=max_detections_per_frame,
        blocked_classes=blocked_classes,
        sam_multimask_topk=sam_multimask_topk,
        sam_mask_selection_policy=sam_mask_selection_policy,
        sam_mask_geometry_model=sam_mask_geometry_model,
        sam_mask_geometry_cc_radius=sam_mask_geometry_cc_radius,
        sam_mask_geometry_plane_threshold=sam_mask_geometry_plane_threshold,
        sam_mask_geometry_max_points=sam_mask_geometry_max_points,
        seed_depth_cluster=seed_depth_cluster,
        seed_depth_cluster_bin_size=seed_depth_cluster_bin_size,
        seed_depth_cluster_window_bins=seed_depth_cluster_window_bins,
        seed_depth_cluster_min_keep_ratio=seed_depth_cluster_min_keep_ratio,
        seed_depth_cluster_min_removed_ratio=seed_depth_cluster_min_removed_ratio,
        seed_depth_cluster_max_removed_ratio=seed_depth_cluster_max_removed_ratio,
        sam_adaptive_internal_seed=sam_adaptive_internal_seed,
        sam_adaptive_internal_keep_ratio=sam_adaptive_internal_keep_ratio,
        sam_adaptive_internal_min_keep_ratio=sam_adaptive_internal_min_keep_ratio,
        sam_adaptive_internal_boundary_weight=sam_adaptive_internal_boundary_weight,
        sam_adaptive_internal_depth_weight=sam_adaptive_internal_depth_weight,
        sam_adaptive_internal_depth_bin_size=sam_adaptive_internal_depth_bin_size,
        sam_adaptive_internal_depth_window_bins=sam_adaptive_internal_depth_window_bins,
        sam_mask_erode_pixels=sam_mask_erode_pixels,
        sam_mask_erode_min_area_ratio=sam_mask_erode_min_area_ratio,
        label_consensus_iou_th=label_consensus_iou_th,
        box_nms_iou=box_nms_iou,
        box_nms_same_class_only=box_nms_same_class_only,
    )

    relation_edges, adjacency, conflict_edges, weak_edges = _build_mask_graph(
        observations,
        points_xyz=points_xyz,
        same_class_only=graph_same_class_only,
        min_seed_iou=graph_min_seed_iou,
        min_seed_containment=graph_min_seed_containment,
        min_reference_coverage=graph_min_reference_coverage,
        spatial_sigma=graph_spatial_sigma,
        view_consensus_scale=graph_view_consensus_scale,
        edge_score_threshold=graph_edge_score_threshold,
    )
    components = _connected_components(len(observations), adjacency)
    candidates = []
    prefilter_skipped = []
    cluster_skipped = []
    for cluster_id, component in enumerate(components):
        if len(component) < int(graph_min_cluster_observations):
            if not graph_keep_singletons:
                cluster_skipped.append({"graph_cluster_id": int(cluster_id), "reason": "few_observations", "observation_count": int(len(component))})
                continue
        selected_indices, selected_seed = _select_cluster_views(
            observations,
            component,
            adjacency,
            max_views=graph_max_views_per_cluster,
            min_new_seed_ratio=graph_min_new_seed_ratio,
        )
        if len(selected_seed) < int(min_seed_points):
            cluster_skipped.append({"graph_cluster_id": int(cluster_id), "reason": "few_cluster_seed_points", "num_seed_points": int(len(selected_seed))})
            continue
        candidate = _cluster_to_candidate(
            len(candidates),
            observations,
            component,
            selected_indices,
            selected_seed,
            adjacency,
            relation_edges,
            existing_masks,
        )
        if export_max_existing_iou is not None and float(candidate.get("best_existing_iou", 0.0)) > float(export_max_existing_iou):
            prefilter_skipped.append(
                {
                    "reason": "export_matched_existing_3d_mask",
                    "graph_cluster_id": int(cluster_id),
                    "class_name": candidate.get("class_name"),
                    "best_existing_iou": float(candidate.get("best_existing_iou", 0.0)),
                }
            )
            continue
        if (
            export_max_seed_in_existing_mask_ratio is not None
            and float(candidate.get("seed_in_existing_mask_ratio", 0.0)) > float(export_max_seed_in_existing_mask_ratio)
        ):
            prefilter_skipped.append(
                {
                    "reason": "export_mostly_covered_by_existing_masks",
                    "graph_cluster_id": int(cluster_id),
                    "class_name": candidate.get("class_name"),
                    "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", 0.0)),
                }
            )
            continue
        candidates.append(candidate)

    if ranking_policy == "graph_priority":
        candidates = sorted(
            candidates,
            key=lambda item: (
                -int(item.get("support_view_count", 0)),
                -float(item.get("graph_consensus_score", 0.0)),
                -float(item.get("graph_edge_mean_score", 0.0)),
                -float(item.get("proposal_priority", 0.0)),
            ),
        )
    elif ranking_policy == "novelty":
        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item.get("seed_in_existing_mask_ratio", 1.0)),
                float(item.get("best_existing_iou", 1.0)),
                -float(item.get("graph_consensus_score", 0.0)),
                -float(item.get("proposal_priority", 0.0)),
            ),
        )
    else:
        candidates = sorted(candidates, key=lambda item: -float(item.get("proposal_priority", 0.0)))

    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]

    output_candidates = []
    for candidate_id, candidate in enumerate(candidates):
        seed_indices = candidate.pop("_seed_indices")
        seed_path = osp.join(seed_dir, f"candidate{candidate_id:04d}_points.npz")
        np.savez_compressed(seed_path, point_indices=seed_indices)
        candidate["candidate_id"] = int(candidate_id)
        candidate["num_seed_points"] = int(len(seed_indices))
        candidate["seed_points_path"] = seed_path
        output_candidates.append(candidate)

    json_path = osp.join(scene_dir, "backprojection_candidates.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "scene_name": scene_name,
                "source_kind": "mask_graph",
                "num_candidates": len(output_candidates),
                "raw_observations": len(observations),
                "graph_edges": len(relation_edges),
                "graph_support_edges": len(adjacency) and sum(len(node_edges) for node_edges in adjacency) // 2,
                "graph_weak_edges": len(weak_edges),
                "graph_conflict_edges": len(conflict_edges),
                "graph_components": len(components),
                "skipped": skipped,
                "cluster_skipped": cluster_skipped,
                "prefilter_skipped": prefilter_skipped,
                "filters": {
                    "detection_score_th": detection_score_th,
                    "min_seed_points": min_seed_points,
                    "max_box_area_ratio": max_box_area_ratio,
                    "frame_stride": frame_stride,
                    "max_frames": max_frames,
                    "max_detections_per_frame": max_detections_per_frame,
                    "max_candidates": max_candidates,
                    "blocked_classes": sorted(_parse_class_names(blocked_classes)),
                    "ranking_policy": ranking_policy,
                    "sam_multimask_topk": sam_multimask_topk,
                    "sam_mask_selection_policy": sam_mask_selection_policy,
                    "sam_mask_geometry_cc_radius": sam_mask_geometry_cc_radius,
                    "sam_mask_geometry_plane_threshold": sam_mask_geometry_plane_threshold,
                    "sam_mask_geometry_max_points": sam_mask_geometry_max_points,
                    "seed_depth_cluster": seed_depth_cluster,
                    "sam_adaptive_internal_seed": sam_adaptive_internal_seed,
                    "sam_mask_erode_pixels": sam_mask_erode_pixels,
                    "sam_mask_erode_min_area_ratio": sam_mask_erode_min_area_ratio,
                    "label_consensus_iou_th": label_consensus_iou_th,
                    "box_nms_iou": box_nms_iou,
                    "box_nms_same_class_only": box_nms_same_class_only,
                    "graph_same_class_only": graph_same_class_only,
                    "graph_min_seed_iou": graph_min_seed_iou,
                    "graph_min_seed_containment": graph_min_seed_containment,
                    "graph_min_reference_coverage": graph_min_reference_coverage,
                    "graph_spatial_sigma": graph_spatial_sigma,
                    "graph_view_consensus_scale": graph_view_consensus_scale,
                    "graph_edge_score_threshold": graph_edge_score_threshold,
                    "graph_min_cluster_observations": graph_min_cluster_observations,
                    "graph_keep_singletons": graph_keep_singletons,
                    "graph_max_views_per_cluster": graph_max_views_per_cluster,
                    "graph_min_new_seed_ratio": graph_min_new_seed_ratio,
                    "export_max_existing_iou": export_max_existing_iou,
                    "export_max_seed_in_existing_mask_ratio": export_max_seed_in_existing_mask_ratio,
                },
                "graph_edge_preview": relation_edges[:200],
                "candidates": output_candidates,
            },
            f,
            indent=2,
        )
    return json_path, output_candidates, {
        "raw_observations": len(observations),
        "graph_edges": len(relation_edges),
        "graph_support_edges": len(adjacency) and sum(len(node_edges) for node_edges in adjacency) // 2,
        "graph_weak_edges": len(weak_edges),
        "graph_conflict_edges": len(conflict_edges),
        "graph_components": len(components),
        "num_candidates": len(output_candidates),
    }


def export_dataset_mask_graph_proposals(
    dataset_name,
    path_to_3d_masks,
    output_dir,
    sam_checkpoint,
    sam_source,
    sam_model_type="vit_b",
    scene_name=None,
    path_to_2d_preds=None,
    reuse_2d_preds=True,
    scene_list=None,
    max_scenes=None,
    **kwargs,
):
    config = load_yaml(osp.join(f"./pretrained/config_{dataset_name}.yaml"))
    path_2_dataset = osp.join("./data", dataset_name)
    depth_scale = config["openyolo3d"]["depth_scale"]

    if dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype = "point cloud"
    elif dataset_name == "scannet200":
        scene_names = SCENE_NAMES_SCANNET200
        datatype = "mesh"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if scene_name is not None:
        scene_names = [scene_name]
    scene_names = _resolve_scene_names(scene_names, scene_list=scene_list, max_scenes=max_scenes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _load_sam_predictor(sam_checkpoint, sam_model_type, device, sam_source)
    geometry_model_bundle = _load_geometry_discriminator(kwargs.get("sam_mask_geometry_model"))
    kwargs["sam_mask_geometry_model"] = geometry_model_bundle
    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_name}.yaml")
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    start = time.time()
    for current_scene in tqdm(scene_names):
        scene_id = current_scene.replace("scene", "")
        processed_file = osp.join(path_2_dataset, current_scene, f"{scene_id}.npy") if dataset_name == "scannet200" else None
        openyolo3d.predict(
            path_2_scene_data=osp.join(path_2_dataset, current_scene),
            depth_scale=depth_scale,
            datatype=datatype,
            processed_scene=processed_file,
            path_to_3d_masks=path_to_3d_masks,
            is_gt=False,
            path_to_2d_preds=path_to_2d_preds,
            save_2d_preds=False,
            reuse_2d_preds=reuse_2d_preds,
        )
        json_path, _, summary = export_scene_mask_graph_proposals(
            openyolo3d,
            predictor,
            current_scene,
            output_dir,
            **kwargs,
        )
        summaries.append({"scene_name": current_scene, "json_path": json_path, **summary})

        for attr in (
            "world2cam",
            "mesh_projections",
            "preds_3d",
            "preds_2d",
            "predicted_masks",
            "predicated_scores",
            "predicated_classes",
        ):
            if hasattr(openyolo3d, attr):
                setattr(openyolo3d, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = osp.join(output_dir, "mask_graph_proposals_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "elapsed_seconds": time.time() - start,
                "params": kwargs,
                "scenes": summaries,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved mask-graph proposal summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--output_dir", default="./output/mask_graph_proposals_replica")
    parser.add_argument("--sam_checkpoint", default="./pretrained/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_source", default="./_external/segment-anything/segment-anything-main")
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h", "default"])
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)

    parser.add_argument("--detection_score_th", default=0.45, type=float)
    parser.add_argument("--min_seed_points", default=80, type=int)
    parser.add_argument("--max_box_area_ratio", default=0.30, type=float)
    parser.add_argument("--frame_stride", default=5, type=int)
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--max_detections_per_frame", default=8, type=int)
    parser.add_argument("--max_candidates_per_scene", default=30, type=int)
    parser.add_argument("--blocked_classes", default="rug")
    parser.add_argument("--ranking_policy", default="graph_priority", choices=["graph_priority", "novelty", "priority"])

    parser.add_argument("--sam_multimask_topk", default=1, type=int)
    parser.add_argument("--sam_mask_selection_policy", default="sam_score", choices=["sam_score", "geometry", "learned_geometry"])
    parser.add_argument("--sam_mask_geometry_model", default=None)
    parser.add_argument("--sam_mask_geometry_cc_radius", default=0.03, type=float)
    parser.add_argument("--sam_mask_geometry_plane_threshold", default=0.02, type=float)
    parser.add_argument("--sam_mask_geometry_max_points", default=50000, type=int)
    parser.add_argument("--seed_depth_cluster", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed_depth_cluster_bin_size", default=0.10, type=float)
    parser.add_argument("--seed_depth_cluster_window_bins", default=1, type=int)
    parser.add_argument("--seed_depth_cluster_min_keep_ratio", default=0.25, type=float)
    parser.add_argument("--seed_depth_cluster_min_removed_ratio", default=0.0, type=float)
    parser.add_argument("--seed_depth_cluster_max_removed_ratio", default=1.0, type=float)
    parser.add_argument("--sam_adaptive_internal_seed", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sam_adaptive_internal_keep_ratio", default=0.70, type=float)
    parser.add_argument("--sam_adaptive_internal_min_keep_ratio", default=0.35, type=float)
    parser.add_argument("--sam_adaptive_internal_boundary_weight", default=0.45, type=float)
    parser.add_argument("--sam_adaptive_internal_depth_weight", default=0.55, type=float)
    parser.add_argument("--sam_adaptive_internal_depth_bin_size", default=0.10, type=float)
    parser.add_argument("--sam_adaptive_internal_depth_window_bins", default=1, type=int)
    parser.add_argument("--sam_mask_erode_pixels", default=0, type=int)
    parser.add_argument("--sam_mask_erode_min_area_ratio", default=0.15, type=float)
    parser.add_argument("--label_consensus_iou_th", default=0.25, type=float)
    parser.add_argument("--box_nms_iou", default=0.0, type=float)
    parser.add_argument("--box_nms_same_class_only", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--graph_same_class_only", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_min_seed_iou", default=0.03, type=float)
    parser.add_argument("--graph_min_seed_containment", default=0.18, type=float)
    parser.add_argument("--graph_min_reference_coverage", default=0.20, type=float)
    parser.add_argument("--graph_spatial_sigma", default=0.35, type=float)
    parser.add_argument("--graph_view_consensus_scale", default=4.0, type=float)
    parser.add_argument("--graph_edge_score_threshold", default=0.35, type=float)
    parser.add_argument("--graph_min_cluster_observations", default=2, type=int)
    parser.add_argument("--graph_keep_singletons", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--graph_max_views_per_cluster", default=4, type=int)
    parser.add_argument("--graph_min_new_seed_ratio", default=0.05, type=float)
    parser.add_argument("--export_max_existing_iou", default=None, type=float)
    parser.add_argument("--export_max_seed_in_existing_mask_ratio", default=None, type=float)
    args = parser.parse_args()

    kwargs = vars(args).copy()
    dataset_name = kwargs.pop("dataset_name")
    path_to_3d_masks = kwargs.pop("path_to_3d_masks")
    output_dir = kwargs.pop("output_dir")
    sam_checkpoint = kwargs.pop("sam_checkpoint")
    sam_source = kwargs.pop("sam_source")
    sam_model_type = kwargs.pop("sam_model_type")
    scene_name = kwargs.pop("scene_name")
    path_to_2d_preds = kwargs.pop("path_to_2d_preds")
    reuse_2d_preds = kwargs.pop("reuse_2d_preds")
    scene_list = kwargs.pop("scene_list")
    max_scenes = kwargs.pop("max_scenes")
    kwargs["max_candidates"] = kwargs.pop("max_candidates_per_scene")

    export_dataset_mask_graph_proposals(
        dataset_name=dataset_name,
        path_to_3d_masks=path_to_3d_masks,
        output_dir=output_dir,
        sam_checkpoint=sam_checkpoint,
        sam_source=sam_source,
        sam_model_type=sam_model_type,
        scene_name=scene_name,
        path_to_2d_preds=path_to_2d_preds,
        reuse_2d_preds=reuse_2d_preds,
        scene_list=scene_list,
        max_scenes=max_scenes,
        **kwargs,
    )


if __name__ == "__main__":
    main()

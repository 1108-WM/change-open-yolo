#!/usr/bin/env python3
"""Build a small evidence-tree v0 diagnostic from exported mask-graph traces.

This is a prototype-only analyzer. It does not write predictions for AP and it
does not modify the fusion path. The input is an export_only mask-graph directory
that already contains mask_graph_trace.json, superpoint observation evidence, and
candidate seed point files.
"""

import argparse
import csv
import json
import os
import os.path as osp
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SCENES = ("scene0011_00", "scene0077_00", "scene0608_01")
PLANE_RISK_CLASSES = {
    "whiteboard",
    "tv",
    "door",
    "curtain",
    "mattress",
    "projector screen",
    "bulletin board",
    "mirror",
    "mat",
    "poster",
    "calendar",
    "paper",
    "picture",
    "blanket",
    "laptop",
}


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path, rows, fieldnames):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _slug(text):
    out = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _load_npz_indices(path):
    if not path or not osp.exists(path):
        return np.zeros((0,), dtype=np.int64)
    data = np.load(path)
    if "point_indices" not in data:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(data["point_indices"], dtype=np.int64)


def _load_superpoint_cache(scene_dir):
    path = osp.join(scene_dir, "superpoint_cache", "superpoint_cache.npz")
    if not osp.exists(path):
        return None
    data = np.load(path)
    return {
        "path": path,
        "scene_points": np.asarray(data["scene_points"], dtype=np.float32),
        "superpoint_ids": np.asarray(data["superpoint_ids"], dtype=np.int64),
        "superpoint_centers": np.asarray(data["superpoint_centers"], dtype=np.float32),
        "superpoint_planarity": np.asarray(data["superpoint_planarity"], dtype=np.float32),
        "superpoint_point_counts": np.asarray(data["superpoint_point_counts"], dtype=np.int64),
    }


def _node_area(obs):
    erosion = obs.get("sam_mask_erosion") if isinstance(obs.get("sam_mask_erosion"), dict) else {}
    if "input_area" in erosion:
        return _safe_int(erosion.get("input_area"), 0)
    box = obs.get("bbox_xyxy") or [0, 0, 0, 0]
    if len(box) >= 4:
        return max(0.0, (_safe_float(box[2]) - _safe_float(box[0])) * (_safe_float(box[3]) - _safe_float(box[1])))
    return 0


def _load_observation_superpoints(scene_dir, obs_id):
    path = osp.join(scene_dir, "superpoint_observation_evidence", f"observation{obs_id:05d}_superpoints.json")
    if not osp.exists(path):
        return {
            "path": "",
            "status": "missing_superpoint_evidence",
            "candidate": set(),
            "core": set(),
            "boundary": set(),
            "conflict": set(),
            "summary": {},
        }
    payload = _load_json(path)
    core = set()
    boundary = set()
    conflict = set()
    candidate = set(int(item) for item in payload.get("candidate_superpoint_ids", []))
    for item in payload.get("superpoint_evidence", []):
        seg = int(item.get("segment_id", -1))
        if seg < 0:
            continue
        label = str(item.get("support_label", ""))
        if label == "strong_support":
            core.add(seg)
        elif label in {"partial_support", "touched_only"}:
            boundary.add(seg)
        elif "reject" in label or "conflict" in label:
            conflict.add(seg)
    return {
        "path": path,
        "status": "available",
        "candidate": candidate,
        "core": core,
        "boundary": boundary,
        "conflict": conflict,
        "summary": payload.get("summary", {}),
    }


def _mask_relations(trace):
    parents = defaultdict(set)
    children = defaultdict(set)
    conflicts = defaultdict(set)
    relation_rows = []
    for edge in trace.get("relation_edges", []):
        relation_type = str(edge.get("relation_type", ""))
        left = _safe_int(edge.get("left"), -1)
        right = _safe_int(edge.get("right"), -1)
        if left < 0 or right < 0:
            continue
        if "containment" in relation_type and edge.get("parent") is not None and edge.get("child") is not None:
            parent = _safe_int(edge.get("parent"), -1)
            child = _safe_int(edge.get("child"), -1)
            if parent >= 0 and child >= 0:
                children[parent].add(child)
                parents[child].add(parent)
                relation_rows.append(
                    {
                        "relation_type": relation_type,
                        "parent": parent,
                        "child": child,
                        "score": _safe_float(edge.get("containment_score", edge.get("containment_strength", 0.0))),
                    }
                )
        if "conflict" in relation_type or relation_type == "same_frame_mutex":
            conflicts[left].add(right)
            conflicts[right].add(left)
    return parents, children, conflicts, relation_rows


def _tree_role(obs, parent_ids, child_ids, conflict_ids):
    if child_ids and (len(child_ids) >= 2 or _safe_int(obs.get("num_seed_points")) >= 500):
        return "possible_undersegmented_parent"
    if parent_ids:
        return "possible_oversegmented_child"
    if conflict_ids or _safe_int(obs.get("same_frame_conflict_count")) > 0:
        return "same_frame_conflict_node"
    return "leaf_or_singleton"


def _build_tree_nodes(scene_dir, trace):
    parents, children, conflicts, relation_rows = _mask_relations(trace)
    nodes = []
    node_by_obs = {}
    for obs in trace.get("observations", []):
        obs_id = _safe_int(obs.get("graph_observation_id"), -1)
        if obs_id < 0:
            continue
        sp = _load_observation_superpoints(scene_dir, obs_id)
        parent_ids = sorted(parents.get(obs_id, set()))
        child_ids = sorted(children.get(obs_id, set()))
        conflict_ids = sorted(conflicts.get(obs_id, set()))
        role = _tree_role(obs, parent_ids, child_ids, conflict_ids)
        row = {
            "scene_name": trace.get("scene_name") or osp.basename(scene_dir),
            "node_id": f"observation{obs_id:05d}",
            "observation_id": obs_id,
            "frame_index": _safe_int(obs.get("frame_index"), -1),
            "frame_id": obs.get("frame_id", ""),
            "class_name": obs.get("class_name", ""),
            "class_id": _safe_int(obs.get("class_id"), -1),
            "score": _safe_float(obs.get("score")),
            "sam_score": _safe_float(obs.get("sam_score")),
            "mask_area": _node_area(obs),
            "box_area_ratio": _safe_float(obs.get("box_area_ratio")),
            "parent_count": len(parent_ids),
            "child_count": len(child_ids),
            "conflict_count": len(conflict_ids),
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "conflict_ids": conflict_ids,
            "tree_role": role,
            "clip_feature_status": "missing_in_source_trace",
            "clip_semantic_proxy": obs.get("class_name", ""),
            "superpoint_evidence_status": sp["status"],
            "core_superpoints": sorted(sp["core"]),
            "boundary_superpoints": sorted(sp["boundary"]),
            "conflict_superpoints": sorted(sp["conflict"]),
            "candidate_superpoint_count": len(sp["candidate"]),
            "core_superpoint_count": len(sp["core"]),
            "boundary_superpoint_count": len(sp["boundary"]),
            "conflict_superpoint_count": len(sp["conflict"]),
            "overlay_path": (obs.get("evidence") or {}).get("overlay_path", ""),
            "context_path": (obs.get("evidence") or {}).get("context_path", ""),
            "crop_path": (obs.get("evidence") or {}).get("crop_path", ""),
        }
        nodes.append(row)
        node_by_obs[obs_id] = row
    return nodes, node_by_obs, relation_rows


def _semantic_proxy_similarity(left, right):
    left = str(left or "").lower()
    right = str(right or "").lower()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = set(left.replace("_", " ").split())
    right_tokens = set(right.replace("_", " ").split())
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _candidate_relation_to_mask3d(candidate):
    best_iou = _safe_float(candidate.get("best_existing_iou"))
    seed_in_existing = _safe_float(candidate.get("seed_in_existing_mask_ratio"))
    label_conflict = _safe_float(candidate.get("label_conflict_score"))
    class_name = str(candidate.get("class_name", ""))
    is_plane_risk = class_name in PLANE_RISK_CLASSES
    if label_conflict >= 0.30 or _safe_int(candidate.get("conflict_edge_count")) > 0:
        return "class_or_relation_conflict_risk"
    if best_iou < 0.15 and seed_in_existing < 0.50:
        return "new_instance_like"
    if best_iou < 0.35 and seed_in_existing >= 0.70:
        return "more_complete_than_mask3d_like"
    if is_plane_risk and best_iou < 0.35:
        return "large_plane_risk_needs_review"
    if best_iou >= 0.50 or seed_in_existing >= 0.90:
        return "local_or_duplicate_of_mask3d"
    return "ambiguous_mask3d_relation"


def _component_relation_score(trace, members):
    members = set(int(item) for item in members)
    support = 0
    containment = 0
    conflict = 0
    weak = 0
    scores = []
    for edge in trace.get("relation_edges", []):
        left = _safe_int(edge.get("left"), -1)
        right = _safe_int(edge.get("right"), -1)
        if left not in members or right not in members:
            continue
        relation_type = str(edge.get("relation_type", ""))
        if relation_type == "same_object":
            support += 1
            scores.append(_safe_float(edge.get("same_object_score", edge.get("edge_score", 0.0))))
        elif "containment" in relation_type:
            containment += 1
            scores.append(_safe_float(edge.get("containment_score", edge.get("edge_score", 0.0))))
        elif "conflict" in relation_type or relation_type == "same_frame_mutex":
            conflict += 1
        elif relation_type in {"weak", "uncertain"}:
            weak += 1
    return {
        "same_object_edge_count": support,
        "containment_edge_count": containment,
        "conflict_edge_count": conflict,
        "weak_edge_count": weak,
        "graph_edge_mean_score": float(np.mean(scores)) if scores else 0.0,
    }


def _trace_components(trace):
    observations = trace.get("observations", [])
    obs_ids = [_safe_int(obs.get("graph_observation_id"), -1) for obs in observations]
    obs_ids = [item for item in obs_ids if item >= 0]
    parent = {item: item for item in obs_ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        if left not in parent or right not in parent:
            return
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in trace.get("relation_edges", []):
        relation_type = str(edge.get("relation_type", ""))
        if relation_type == "same_object" or "containment" in relation_type:
            union(_safe_int(edge.get("left"), -1), _safe_int(edge.get("right"), -1))

    components = defaultdict(list)
    for obs_id in obs_ids:
        components[find(obs_id)].append(obs_id)
    return list(components.values())


def _candidate_from_trace_component(scene_name, component_id, members, trace, node_by_obs):
    member_nodes = [node_by_obs[item] for item in members if item in node_by_obs]
    roles = Counter(node.get("tree_role", "") for node in member_nodes)
    classes = Counter(node.get("class_name", "") for node in member_nodes)
    class_name = classes.most_common(1)[0][0] if classes else ""
    scores = [_safe_float(node.get("score")) for node in member_nodes]
    best_ious = []
    seed_existing = []
    for obs in trace.get("observations", []):
        obs_id = _safe_int(obs.get("graph_observation_id"), -1)
        if obs_id not in members:
            continue
        best_ious.append(_safe_float(obs.get("best_existing_iou")))
        seed_existing.append(_safe_float(obs.get("seed_in_existing_mask_ratio")))
    edge_stats = _component_relation_score(trace, members)
    pseudo = {
        "class_name": class_name,
        "best_existing_iou": max(best_ious) if best_ious else 0.0,
        "seed_in_existing_mask_ratio": float(np.mean(seed_existing)) if seed_existing else 0.0,
        "label_conflict_score": 1.0 - (classes.most_common(1)[0][1] / max(1, len(member_nodes))) if classes else 0.0,
        "conflict_edge_count": edge_stats["conflict_edge_count"],
    }
    relation = _candidate_relation_to_mask3d(pseudo)
    return {
        "scene_name": scene_name,
        "candidate_id": int(10000 + component_id),
        "candidate_source_kind": "trace_relation_component",
        "class_name": class_name,
        "score": float(np.mean(scores)) if scores else 0.0,
        "support_view_count": len(set(_safe_int(node.get("frame_index"), -1) for node in member_nodes)),
        "member_observation_ids": sorted(int(item) for item in members),
        "member_count": len(members),
        "tree_role_counts": dict(roles),
        "dominant_tree_role": roles.most_common(1)[0][0] if roles else "",
        "class_consistency": float(classes.most_common(1)[0][1] / max(1, len(member_nodes))) if classes else 0.0,
        "clip_feature_status": "missing_in_source_trace",
        "clip_semantic_proxy_mean_similarity": 1.0 if len(classes) <= 1 else 0.0,
        "core_superpoints": [],
        "boundary_superpoints": [],
        "conflict_superpoints": [],
        "core_superpoint_count": 0,
        "boundary_superpoint_count": 0,
        "conflict_superpoint_count": 0,
        "largest_cc_point_count": 0,
        "point_count": 0,
        "existing_mask_iou": pseudo["best_existing_iou"],
        "seed_in_existing_mask_ratio": pseudo["seed_in_existing_mask_ratio"],
        "mask3d_relation": relation,
        "is_new_instance_like": relation == "new_instance_like",
        "is_local_or_duplicate": relation == "local_or_duplicate_of_mask3d",
        "is_plane_risk": class_name in PLANE_RISK_CLASSES,
        "superpoint_connected_proxy": "",
        "superpoint_evidence_status": "missing_superpoint_evidence",
        "same_object_edge_count": edge_stats["same_object_edge_count"],
        "containment_edge_count": edge_stats["containment_edge_count"],
        "conflict_edge_count": edge_stats["conflict_edge_count"],
        "weak_edge_count": edge_stats["weak_edge_count"],
        "graph_edge_mean_score": edge_stats["graph_edge_mean_score"],
        "source_trace": "mask_graph_trace_relation_component_without_superpoint_evidence",
    }


def _candidate_from_backprojection(scene_name, candidate, node_by_obs, superpoint_cache):
    member_ids = [_safe_int(item, -1) for item in candidate.get("graph_selected_observation_ids", [])]
    member_ids = [item for item in member_ids if item >= 0]
    if not member_ids:
        member_ids = [_safe_int(item, -1) for item in candidate.get("graph_cluster_observation_ids", [])]
        member_ids = [item for item in member_ids if item >= 0]
    member_nodes = [node_by_obs[item] for item in member_ids if item in node_by_obs]

    core = set()
    boundary = set()
    conflict = set()
    roles = Counter()
    classes = Counter()
    for node in member_nodes:
        core.update(int(item) for item in node.get("core_superpoints", []))
        boundary.update(int(item) for item in node.get("boundary_superpoints", []))
        conflict.update(int(item) for item in node.get("conflict_superpoints", []))
        roles[node.get("tree_role", "")] += 1
        classes[node.get("class_name", "")] += 1

    point_indices = _load_npz_indices(candidate.get("superpoint_candidate_largest_cc_seed_points_path") or candidate.get("seed_points_path"))
    connected = True
    plane_risk = str(candidate.get("class_name", "")) in PLANE_RISK_CLASSES
    if superpoint_cache is not None and core:
        centers = superpoint_cache["superpoint_centers"]
        valid = np.asarray([sid for sid in core if 0 <= sid < len(centers)], dtype=np.int64)
        if len(valid) >= 2:
            span = centers[valid].max(axis=0) - centers[valid].min(axis=0)
            connected = bool(np.max(span) < 4.0)
        planarity = float(np.mean(superpoint_cache["superpoint_planarity"][valid])) if len(valid) else 0.0
        plane_risk = bool(plane_risk or planarity >= 0.65)

    relation = _candidate_relation_to_mask3d(candidate)
    return {
        "scene_name": scene_name,
        "candidate_id": _safe_int(candidate.get("candidate_id"), -1),
        "candidate_source_kind": candidate.get("source_kind", "backprojection_candidate"),
        "class_name": candidate.get("class_name", ""),
        "score": _safe_float(candidate.get("score")),
        "support_view_count": _safe_int(candidate.get("support_view_count")),
        "member_observation_ids": member_ids,
        "member_count": len(member_ids),
        "tree_role_counts": dict(roles),
        "dominant_tree_role": roles.most_common(1)[0][0] if roles else "",
        "class_consistency": float(classes.most_common(1)[0][1] / max(1, sum(classes.values()))) if classes else 0.0,
        "clip_feature_status": "missing_in_source_trace",
        "clip_semantic_proxy_mean_similarity": 1.0,
        "core_superpoints": sorted(core),
        "boundary_superpoints": sorted(boundary - core),
        "conflict_superpoints": sorted(conflict),
        "core_superpoint_count": len(core),
        "boundary_superpoint_count": len(boundary - core),
        "conflict_superpoint_count": len(conflict),
        "largest_cc_point_count": _safe_int(candidate.get("superpoint_candidate_largest_cc_seed_point_count"), len(point_indices)),
        "point_count": len(point_indices),
        "existing_mask_iou": _safe_float(candidate.get("best_existing_iou")),
        "seed_in_existing_mask_ratio": _safe_float(candidate.get("seed_in_existing_mask_ratio")),
        "mask3d_relation": relation,
        "is_new_instance_like": relation == "new_instance_like",
        "is_local_or_duplicate": relation == "local_or_duplicate_of_mask3d",
        "is_plane_risk": plane_risk,
        "superpoint_connected_proxy": connected,
        "superpoint_evidence_status": "available" if any(node.get("superpoint_evidence_status") == "available" for node in member_nodes) else "missing_superpoint_evidence",
        "same_object_edge_count": _safe_int(candidate.get("same_object_edge_count")),
        "containment_edge_count": _safe_int(candidate.get("containment_edge_count")),
        "conflict_edge_count": _safe_int(candidate.get("conflict_edge_count")),
        "weak_edge_count": _safe_int(candidate.get("weak_edge_count")),
        "graph_edge_mean_score": _safe_float(candidate.get("graph_edge_mean_score")),
        "source_trace": "mask_graph_trace_plus_superpoint_observation_evidence",
    }


def _plot_scene_tree(path, scene_name, nodes, relation_rows):
    os.makedirs(osp.dirname(path), exist_ok=True)
    role_colors = {
        "possible_undersegmented_parent": "#e45756",
        "possible_oversegmented_child": "#72b7b2",
        "same_frame_conflict_node": "#f58518",
        "leaf_or_singleton": "#4c78a8",
    }
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    max_area = max([_safe_float(node.get("mask_area")) for node in nodes] + [1.0])
    for node in nodes:
        x = _safe_int(node.get("frame_index"))
        y = _safe_float(node.get("mask_area")) / max_area
        color = role_colors.get(node.get("tree_role"), "#8c8c8c")
        size = 20 + 120 * min(1.0, y)
        ax.scatter([x], [y], s=size, c=color, alpha=0.82, linewidths=0)
    obs_to_node = {_safe_int(node["observation_id"]): node for node in nodes}
    for edge in relation_rows:
        parent = obs_to_node.get(_safe_int(edge.get("parent"), -1))
        child = obs_to_node.get(_safe_int(edge.get("child"), -1))
        if not parent or not child:
            continue
        ax.plot(
            [_safe_int(parent["frame_index"]), _safe_int(child["frame_index"])],
            [_safe_float(parent["mask_area"]) / max_area, _safe_float(child["mask_area"]) / max_area],
            c="#999999",
            linewidth=0.8,
            alpha=0.5,
        )
    ax.set_title(f"{scene_name} evidence tree nodes by frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("relative mask area")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=role, markerfacecolor=color, markersize=7)
        for role, color in role_colors.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_candidate_superpoints(path, scene_name, candidate, superpoint_cache):
    os.makedirs(osp.dirname(path), exist_ok=True)
    centers = superpoint_cache["superpoint_centers"] if superpoint_cache is not None else np.zeros((0, 3), dtype=np.float32)
    dims = (0, 2)
    sets = [
        ("core", candidate.get("core_superpoints", []), "#54a24b", 16),
        ("boundary", candidate.get("boundary_superpoints", []), "#f58518", 12),
        ("conflict", candidate.get("conflict_superpoints", []), "#e45756", 14),
    ]
    arrays = []
    for _, ids, _, _ in sets:
        valid = np.asarray([sid for sid in ids if 0 <= int(sid) < len(centers)], dtype=np.int64)
        arrays.append(centers[valid] if len(valid) else np.zeros((0, 3), dtype=np.float32))
    non_empty = [arr[:, dims] for arr in arrays if len(arr)]
    if non_empty:
        stacked = np.concatenate(non_empty, axis=0)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        span = np.maximum(maxs - mins, 1e-3)
        xlim = (mins[0] - 0.08 * span[0], maxs[0] + 0.08 * span[0])
        ylim = (mins[1] - 0.08 * span[1], maxs[1] + 0.08 * span[1])
    else:
        xlim, ylim = (0.0, 1.0), (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for (label, _, color, size), arr in zip(sets, arrays):
        if len(arr):
            ax.scatter(arr[:, dims[0]], arr[:, dims[1]], s=size, c=color, alpha=0.8, linewidths=0, label=f"{label} ({len(arr)})")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    if any(len(arr) for arr in arrays):
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "superpoint evidence missing\nsource trace still supports tree/montage review",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(
        f"{scene_name} cand{candidate['candidate_id']:04d} {candidate['class_name']} | "
        f"{candidate['mask3d_relation']}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_source_montage(path, scene_name, candidate, node_by_obs):
    image_paths = []
    for obs_id in candidate.get("member_observation_ids", [])[:4]:
        node = node_by_obs.get(int(obs_id))
        if node and node.get("overlay_path") and osp.exists(node["overlay_path"]):
            image_paths.append(node["overlay_path"])
    if not image_paths:
        return False
    os.makedirs(osp.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, len(image_paths), figsize=(4 * len(image_paths), 3.2), constrained_layout=True)
    if len(image_paths) == 1:
        axes = [axes]
    for ax, img_path in zip(axes, image_paths):
        ax.imshow(imageio.imread(img_path))
        ax.set_title(osp.basename(img_path).split("_")[0], fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{scene_name} cand{candidate['candidate_id']:04d} source tree nodes", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _summarize_scene(scene_name, nodes, candidates):
    relation_counts = Counter(c["mask3d_relation"] for c in candidates)
    return {
        "scene_name": scene_name,
        "tree_node_count": len(nodes),
        "undersegmented_parent_nodes": sum(1 for node in nodes if node["tree_role"] == "possible_undersegmented_parent"),
        "oversegmented_child_nodes": sum(1 for node in nodes if node["tree_role"] == "possible_oversegmented_child"),
        "same_frame_conflict_nodes": sum(1 for node in nodes if node["tree_role"] == "same_frame_conflict_node"),
        "candidate_count": len(candidates),
        "new_instance_like_count": relation_counts.get("new_instance_like", 0),
        "more_complete_like_count": relation_counts.get("more_complete_than_mask3d_like", 0),
        "local_or_duplicate_count": relation_counts.get("local_or_duplicate_of_mask3d", 0),
        "plane_risk_candidate_count": sum(1 for item in candidates if item["is_plane_risk"]),
        "clip_feature_status": "missing_in_source_trace",
    }


def run(args):
    scenes = args.scenes or list(DEFAULT_SCENES)
    os.makedirs(args.output_diagnostics, exist_ok=True)
    os.makedirs(args.output_visuals, exist_ok=True)
    all_nodes = []
    all_candidates = []
    scene_summaries = []
    visual_index = []

    for scene_name in scenes:
        scene_dir = osp.join(args.candidates_dir, scene_name)
        trace_path = osp.join(scene_dir, "mask_graph_trace.json")
        backprojection_path = osp.join(scene_dir, "backprojection_candidates.json")
        if not osp.exists(trace_path) or not osp.exists(backprojection_path):
            raise FileNotFoundError(f"missing exported trace/candidates for {scene_name}: {scene_dir}")
        trace = _load_json(trace_path)
        backprojection = _load_json(backprojection_path)
        superpoint_cache = _load_superpoint_cache(scene_dir)
        nodes, node_by_obs, relation_rows = _build_tree_nodes(scene_dir, trace)
        candidates = [
            _candidate_from_backprojection(scene_name, candidate, node_by_obs, superpoint_cache)
            for candidate in backprojection.get("candidates", [])
        ]
        represented = {tuple(sorted(item.get("member_observation_ids", []))) for item in candidates}
        component_candidates = []
        for component in _trace_components(trace):
            if len(component) < 2:
                continue
            key = tuple(sorted(component))
            if key in represented:
                continue
            component_candidates.append((len(component), component))
        component_candidates.sort(reverse=True, key=lambda item: item[0])
        for component_index, (_, component) in enumerate(component_candidates[: args.max_trace_component_candidates_per_scene]):
            candidates.append(_candidate_from_trace_component(scene_name, component_index, component, trace, node_by_obs))
        all_nodes.extend(nodes)
        all_candidates.extend(candidates)
        scene_summaries.append(_summarize_scene(scene_name, nodes, candidates))

        _write_json(osp.join(args.output_diagnostics, scene_name, "tree_nodes.json"), nodes)
        _write_json(osp.join(args.output_diagnostics, scene_name, "candidate_instances.json"), candidates)
        _write_json(osp.join(args.output_diagnostics, scene_name, "tree_relations.json"), relation_rows)

        tree_png = osp.join(args.output_visuals, scene_name, f"{scene_name}_tree_nodes.png")
        _plot_scene_tree(tree_png, scene_name, nodes, relation_rows)
        visual_index.append({"scene_name": scene_name, "kind": "tree_nodes", "path": tree_png})

        ranked = sorted(
            candidates,
            key=lambda item: (
                item["mask3d_relation"] not in {"new_instance_like", "more_complete_than_mask3d_like"},
                -item["support_view_count"],
                -item["core_superpoint_count"],
            ),
        )
        for candidate in ranked[: args.max_candidate_visuals_per_scene]:
            prefix = f"{scene_name}_candidate{candidate['candidate_id']:04d}_{_slug(candidate['class_name'])}"
            sp_png = osp.join(args.output_visuals, scene_name, f"{prefix}_superpoint_votes_xz.png")
            _plot_candidate_superpoints(sp_png, scene_name, candidate, superpoint_cache)
            visual_index.append(
                {
                    "scene_name": scene_name,
                    "candidate_id": candidate["candidate_id"],
                    "kind": "superpoint_votes_xz",
                    "mask3d_relation": candidate["mask3d_relation"],
                    "path": sp_png,
                }
            )
            montage_png = osp.join(args.output_visuals, scene_name, f"{prefix}_source_overlays.png")
            if _plot_source_montage(montage_png, scene_name, candidate, node_by_obs):
                visual_index.append(
                    {
                        "scene_name": scene_name,
                        "candidate_id": candidate["candidate_id"],
                        "kind": "source_overlays",
                        "mask3d_relation": candidate["mask3d_relation"],
                        "path": montage_png,
                    }
                )

    node_fields = [
        "scene_name",
        "node_id",
        "observation_id",
        "frame_index",
        "class_name",
        "score",
        "sam_score",
        "mask_area",
        "parent_count",
        "child_count",
        "conflict_count",
        "tree_role",
        "candidate_superpoint_count",
        "core_superpoint_count",
        "boundary_superpoint_count",
        "conflict_superpoint_count",
        "clip_feature_status",
        "superpoint_evidence_status",
    ]
    candidate_fields = [
        "scene_name",
        "candidate_id",
        "class_name",
        "candidate_source_kind",
        "score",
        "support_view_count",
        "member_count",
        "dominant_tree_role",
        "class_consistency",
        "core_superpoint_count",
        "boundary_superpoint_count",
        "conflict_superpoint_count",
        "largest_cc_point_count",
        "existing_mask_iou",
        "seed_in_existing_mask_ratio",
        "mask3d_relation",
        "is_new_instance_like",
        "is_local_or_duplicate",
        "is_plane_risk",
        "superpoint_connected_proxy",
        "clip_feature_status",
        "superpoint_evidence_status",
        "same_object_edge_count",
        "containment_edge_count",
        "conflict_edge_count",
        "weak_edge_count",
        "graph_edge_mean_score",
    ]
    _write_csv(osp.join(args.output_diagnostics, "tree_nodes.csv"), all_nodes, node_fields)
    _write_csv(osp.join(args.output_diagnostics, "candidate_instances.csv"), all_candidates, candidate_fields)
    _write_csv(
        osp.join(args.output_diagnostics, "scene_summary.csv"),
        scene_summaries,
        [
            "scene_name",
            "tree_node_count",
            "undersegmented_parent_nodes",
            "oversegmented_child_nodes",
            "same_frame_conflict_nodes",
            "candidate_count",
            "new_instance_like_count",
            "more_complete_like_count",
            "local_or_duplicate_count",
            "plane_risk_candidate_count",
            "clip_feature_status",
        ],
    )
    summary = {
        "prototype": "evidence_tree_v0",
        "input_candidates_dir": args.candidates_dir,
        "scenes": scenes,
        "scene_count": len(scenes),
        "tree_node_count": len(all_nodes),
        "candidate_count": len(all_candidates),
        "new_instance_like_count": sum(1 for item in all_candidates if item["mask3d_relation"] == "new_instance_like"),
        "more_complete_like_count": sum(1 for item in all_candidates if item["mask3d_relation"] == "more_complete_than_mask3d_like"),
        "local_or_duplicate_count": sum(1 for item in all_candidates if item["mask3d_relation"] == "local_or_duplicate_of_mask3d"),
        "clip_feature_status": "missing_in_source_trace",
        "notes": [
            "v0 uses same-frame containment/conflict edges from mask_graph_trace as the 2D mask tree approximation.",
            "CLIP feature vectors were not serialized in the source trace, so v0 records a semantic proxy instead of vector similarity.",
            "When superpoint observation evidence is missing, v0 records missing_superpoint_evidence instead of inferring fake superpoint votes.",
            "Outputs are diagnostics only and are not AP predictions.",
        ],
    }
    _write_json(osp.join(args.output_diagnostics, "summary.json"), summary)
    _write_json(osp.join(args.output_visuals, "visual_review_index.json"), visual_index)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype hierarchical evidence-tree diagnostics.")
    parser.add_argument(
        "--candidates_dir",
        default="output/mask_graph_proposals_scannet200_even48_phase1_relation_fix_gpu",
        help="Existing export_only mask-graph proposal directory.",
    )
    parser.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    parser.add_argument("--output_diagnostics", default="docs/diagnostics/evidence_tree_v0_3scenes")
    parser.add_argument("--output_visuals", default="docs/visual_checks/evidence_tree_v0_3scenes")
    parser.add_argument("--max_candidate_visuals_per_scene", type=int, default=2)
    parser.add_argument("--max_trace_component_candidates_per_scene", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

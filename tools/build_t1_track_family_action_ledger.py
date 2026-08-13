#!/usr/bin/env python3
"""Build T1a's frozen no-GT track-family/action ledger.

This tool starts from hierarchy-safe D1 observations and the frozen Details
frame-sIoU tracks.  It only enumerates broad, auditable alternative association
relations and the plan-only actions ``keep``, ``attach``, ``reassign`` and
``merge``.  It never changes a track, a proposal, a score, a class, or AP.

The DINO interface is deliberately explicit.  ``--dino-state deferred`` is
valid for a CUDA-unavailable environment, but writes missing values rather than
silently substituting an unrelated appearance signal.  RGB/texture/normals and
all geometry remain independently recorded.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_automatic_sam_track_growth_ledger import _raw_superpoint_context
from tools.build_details_frame_siou_tracks import framewise_siou


RECALL_KNN = 12
RECALL_MAX_SHARED_NODES = 32
ADJACENCY_KNN = 12
ADJACENCY_MAX_DISTANCE = 0.05
MIN_CONTACT_POINTS = 3
MIN_CONTACT_RATIO = 0.02
RELATIVE_DEPTH_CONTRACT = "mv3dis_relative_depth_rle_reference"
ACTION_TYPES = ("keep", "attach", "reassign", "merge")
DECISION_CONSTRAINT = (
    "No GT, class, semantic score, threshold, proposal mutation, score mutation, "
    "or AP is read or produced. Relations are broad recall evidence only; no row "
    "is accepted or rejected as an inference action."
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_points(path):
    with np.load(path) as payload:
        return np.unique(np.asarray(payload["point_indices"], dtype=np.int64))


def _load_relative_points(path):
    with np.load(path) as payload:
        points = np.asarray(payload["point_indices"], dtype=np.int64)
        weights = np.asarray(payload["depth_weights"], dtype=np.float32)
    if points.shape != weights.shape or np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("relative-depth point cache is invalid")
    order = np.argsort(points)
    return points[order], weights[order]


def _point_overlap(left, right):
    common = np.intersect1d(left, right, assume_unique=True)
    return int(len(common)), float(len(common) / max(1, len(left) + len(right) - len(common)))


def _weighted_overlap(left_points, left_weights, right_points, right_weights):
    common, left_index, right_index = np.intersect1d(
        left_points, right_points, assume_unique=True, return_indices=True
    )
    if not len(common):
        return 0.0, 0.0, 0.0
    shared = np.minimum(left_weights[left_index], right_weights[right_index])
    left_total, right_total = float(left_weights.sum()), float(right_weights.sum())
    shared_total = float(shared.sum())
    return (
        float(shared_total / max(1e-12, left_total + right_total - shared_total)),
        float(shared_total / max(1e-12, left_total)),
        float(shared_total / max(1e-12, right_total)),
    )


def _track_map(path):
    payload = json.loads(path.read_text())
    tracks = payload.get("tracks", [])
    if int(payload.get("track_count", len(tracks))) != len(tracks):
        raise ValueError("track count is inconsistent")
    result, owner = {}, {}
    for track in tracks:
        track_id = int(track["track_id"])
        observation_ids = tuple(map(int, track.get("observation_ids", [])))
        frame_ids = tuple(map(str, track.get("frame_ids", [])))
        if track_id in result or not observation_ids or len(observation_ids) != len(set(observation_ids)):
            raise ValueError("track identity or observation membership is invalid")
        for observation_id in observation_ids:
            if observation_id in owner:
                raise ValueError("an observation belongs to multiple frozen Details tracks")
            owner[observation_id] = track_id
        result[track_id] = {
            "track_id": track_id,
            "observation_ids": observation_ids,
            "frame_ids": frame_ids,
            "support_view_count": int(track["support_view_count"]),
            "point_count": int(track["point_count"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "mean_edge_score": float(track["mean_edge_score"]),
        }
    return result, owner


def _visibility_superpoints(scene_name, dataset_root, config_path, superpoints, frames):
    """Reproduce Details' common-visible superpoint domain without using GT."""
    import yaml
    from utils import WORLD_2_CAM

    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    world = WORLD_2_CAM(
        str(dataset_root / scene_name), float(config["openyolo3d"]["depth_scale"]), config
    )
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    # WORLD_2_CAM stores the sampled camera-to-world poses in precisely the
    # frame-index coordinate system used by the Details observations.  Keeping
    # this separate from object-centroid distance avoids mislabelling one as
    # the other in the T1 evidence ledger.
    camera_centers = {
        frame: np.asarray(np.loadtxt(world.poses[frame]), dtype=np.float32)[:3, 3]
        for frame in sorted(set(map(int, frames)))
    }
    ids, total_counts = np.unique(superpoints, return_counts=True)
    totals = {int(item): int(count) for item, count in zip(ids, total_counts)}
    result = {}
    for frame in sorted(set(map(int, frames))):
        visible_ids, counts = np.unique(superpoints[np.flatnonzero(visibility[frame])], return_counts=True)
        result[frame] = np.asarray([
            int(item) for item, count in zip(visible_ids, counts)
            if int(count) / max(1, totals[int(item)]) >= .10
        ], dtype=np.int64)
    return result, camera_centers


def _lifted_superpoints(points, superpoints, visible_superpoints):
    ids = np.unique(superpoints[points])
    return np.intersect1d(ids, visible_superpoints, assume_unique=True)


def _observation_nodes(scene_name, d1_root, relative_root, processed, visible_by_frame, camera_centers, dino_state):
    relative_rows = {
        int(row["observation_id"]): row
        for row in _read_jsonl(relative_root / scene_name / "automatic_observations.jsonl")
    }
    ids = list(relative_rows)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate relative observation identity")
    nodes = {}
    for raw in _read_jsonl(d1_root / scene_name / "automatic_observations.jsonl"):
        observation_id = int(raw["observation_id"])
        relative = relative_rows.get(observation_id)
        if relative is None or relative.get("projection_contract") != RELATIVE_DEPTH_CONTRACT:
            raise ValueError(f"{scene_name}/{observation_id} lacks matching relative-depth observation")
        points = _load_points(raw["point_indices_path"])
        points = points[(points >= 0) & (points < len(processed))]
        if not len(points):
            continue
        relative_points, relative_weights = _load_relative_points(relative["point_indices_path"])
        frame = int(raw["frame_index"])
        xyz = np.asarray(processed[points, :3], dtype=np.float32)
        colors = np.asarray(processed[points, 3:6], dtype=np.float32)
        if colors.size and float(colors.max()) > 1.5:
            colors /= 255.0
        normals = np.asarray(processed[points, 6:9], dtype=np.float32)
        normal = normals.mean(axis=0)
        normal /= max(1e-6, float(np.linalg.norm(normal)))
        nodes[observation_id] = {
            "observation_id": observation_id,
            "frame_id": str(raw["frame_id"]),
            "frame_index": frame,
            "point_count": int(len(points)),
            "points": points,
            "relative_points": relative_points,
            "relative_weights": relative_weights,
            "lifted_superpoints": _lifted_superpoints(points, np.asarray(processed[:, 9], dtype=np.int64), visible_by_frame[frame]),
            "centroid": xyz.mean(axis=0),
            "camera_center": camera_centers[frame],
            "mean_rgb": colors.mean(axis=0),
            "rgb_std": colors.std(axis=0),
            "mean_normal": normal,
            "area": int(raw.get("area", 0)),
            "predicted_iou": float(raw["predicted_iou"]),
            "stability_score": float(raw["stability_score"]),
            "point_indices_path": str(raw["point_indices_path"]),
            "relative_point_indices_path": str(relative["point_indices_path"]),
            "mask_rle": raw["mask_rle"],
            "bbox_xywh": [int(value) for value in raw["bbox_xywh"]],
            "crop_box_xywh": [int(value) for value in raw["crop_box_xywh"]],
            "dino_vits14_embedding_state": dino_state,
            "dino_vits14_embedding": None,
        }
    return nodes


def _same_frame_map(path):
    result = {}
    for row in _read_jsonl(path):
        key = tuple(sorted((int(row["left_observation_id"]), int(row["right_observation_id"]))))
        if key in result:
            raise ValueError("duplicate same-frame relation")
        result[key] = row
    return result


def _candidate_observation_pairs(nodes, knn, max_shared_nodes):
    """Broad deterministic recall: centroid KNN plus shared lifted-superpoint buckets."""
    values = sorted(nodes)
    if len(values) < 2:
        return {}
    positions = np.stack([nodes[item]["centroid"] for item in values])
    tree = cKDTree(positions)
    neighbor_count = min(len(values), max(2, int(knn) + 1))
    _, neighbors = tree.query(positions, k=neighbor_count)
    pairs = defaultdict(set)
    for index, row in enumerate(np.atleast_2d(neighbors)):
        for other_index in row:
            left, right = values[index], values[int(other_index)]
            if left != right and nodes[left]["frame_index"] != nodes[right]["frame_index"]:
                pairs[tuple(sorted((left, right)))].add("centroid_knn")
    buckets = defaultdict(list)
    for observation_id in values:
        for superpoint_id in nodes[observation_id]["lifted_superpoints"]:
            bucket = buckets[int(superpoint_id)]
            if len(bucket) < int(max_shared_nodes):
                bucket.append(observation_id)
    for observation_ids in buckets.values():
        for index, left in enumerate(observation_ids):
            for right in observation_ids[index + 1:]:
                if nodes[left]["frame_index"] != nodes[right]["frame_index"]:
                    pairs[tuple(sorted((left, right)))].add("shared_lifted_superpoint")
    return {key: sorted(value) for key, value in sorted(pairs.items())}


def _edge_features(left, right, recall_reasons, visible_by_frame):
    common_visible = np.intersect1d(
        visible_by_frame[left["frame_index"]], visible_by_frame[right["frame_index"]], assume_unique=True
    )
    siou = framewise_siou(left["lifted_superpoints"], right["lifted_superpoints"], common_visible)
    shared_points, point_iou = _point_overlap(left["points"], right["points"])
    relative_iou, left_relative_coverage, right_relative_coverage = _weighted_overlap(
        left["relative_points"], left["relative_weights"], right["relative_points"], right["relative_weights"]
    )
    normal_dot = float(np.clip(np.dot(left["mean_normal"], right["mean_normal"]), -1.0, 1.0))
    return {
        "recall_reasons": recall_reasons,
        "common_visible_superpoint_count": int(len(common_visible)),
        "common_visible_superpoint_siou": float(siou),
        "shared_point_count": shared_points,
        "point_iou": point_iou,
        "left_to_right_reprojection_support_ratio": float(shared_points / max(1, len(left["points"]))),
        "right_to_left_reprojection_support_ratio": float(shared_points / max(1, len(right["points"]))),
        "relative_depth_weighted_iou": relative_iou,
        "left_relative_depth_support_ratio": left_relative_coverage,
        "right_relative_depth_support_ratio": right_relative_coverage,
        "centroid_distance": float(np.linalg.norm(left["centroid"] - right["centroid"])),
        "camera_baseline": float(np.linalg.norm(left["camera_center"] - right["camera_center"])),
        "rgb_mean_l2_difference": float(np.linalg.norm(left["mean_rgb"] - right["mean_rgb"]) / np.sqrt(3.0)),
        "texture_rgb_std_l2_difference": float(np.linalg.norm(left["rgb_std"] - right["rgb_std"]) / np.sqrt(3.0)),
        "normal_difference": float(1.0 - abs(normal_dot)),
        "dino_vits14_masked_crop_cosine": None,
        "dino_vits14_state": left["dino_vits14_embedding_state"],
    }


def _superpoint_set(values):
    """Accept ndarray-backed production nodes and list-backed test fixtures."""
    return {int(item) for item in np.asarray(values, dtype=np.int64).reshape(-1)}


def _track_superpoints(track, nodes):
    return set().union(*(_superpoint_set(nodes[item]["lifted_superpoints"]) for item in track["observation_ids"]))


def _reliable_core(track, nodes):
    frames_by_superpoint = defaultdict(set)
    for observation_id in track["observation_ids"]:
        node = nodes[observation_id]
        for superpoint_id in node["lifted_superpoints"]:
            frames_by_superpoint[int(superpoint_id)].add(node["frame_index"])
    return {item for item, frames in frames_by_superpoint.items() if len(frames) >= 2}


def _contact_summary(left_superpoints, right_superpoints, contact_map):
    edges = []
    for left in left_superpoints:
        for right in right_superpoints:
            if left == right:
                continue
            key = tuple(sorted((int(left), int(right))))
            if key in contact_map:
                edges.append(contact_map[key])
    count = sum(int(edge["boundary_contact_count"]) for edge in edges)
    return {
        "spatial_contact_edge_count": len(edges),
        "spatial_contact_count_sum": int(count),
        "spatial_contact_ratio_max": float(max((edge["boundary_contact_ratio"] for edge in edges), default=0.0)),
    }


def _same_frame_summary(source_observations, target_observations, nodes, relation_map):
    source_by_frame = defaultdict(list)
    target_by_frame = defaultdict(list)
    for item in source_observations:
        source_by_frame[nodes[item]["frame_index"]].append(item)
    for item in target_observations:
        target_by_frame[nodes[item]["frame_index"]].append(item)
    collisions, containment, partial, separation, duplicate = 0, 0, 0, 0, 0
    for frame in sorted(set(source_by_frame) & set(target_by_frame)):
        for left in source_by_frame[frame]:
            for right in target_by_frame[frame]:
                if left == right:
                    continue
                row = relation_map.get(tuple(sorted((left, right))))
                if row is None:
                    continue
                collisions += 1
                kind = str(row.get("relation_kind", ""))
                if "contain" in kind:
                    containment += 1
                elif kind in {"partial", "partial_overlap"}:
                    partial += 1
                elif kind == "disjoint":
                    separation += 1
                elif kind == "duplicate":
                    duplicate += 1
    return {
        "same_frame_collision_count": int(collisions),
        "same_frame_containment_count": int(containment),
        "same_frame_partial_overlap_count": int(partial),
        "same_frame_separation_counterevidence_count": int(separation),
        "same_frame_duplicate_count": int(duplicate),
        "would_create_same_frame_collision": bool(collisions),
    }


def _family_relation(scene_name, source_kind, source_ids, target_track_id, target_track, nodes, evidence, relation_map, contact_map, tracks):
    target_ids = list(target_track["observation_ids"])
    edge_rows = [item["features"] for item in evidence]
    best = max(edge_rows, key=lambda item: (
        item["common_visible_superpoint_siou"], item["relative_depth_weighted_iou"],
        item["left_to_right_reprojection_support_ratio"] + item["right_to_left_reprojection_support_ratio"],
    ))
    source_track = tracks.get(int(source_ids[0])) if source_kind == "track" else None
    source_obs = list(source_track["observation_ids"]) if source_track is not None else list(source_ids)
    source_sp = _track_superpoints(source_track, nodes) if source_track else _superpoint_set(nodes[source_ids[0]]["lifted_superpoints"])
    target_sp = _track_superpoints(target_track, nodes)
    source_core = _reliable_core(source_track, nodes) if source_track else set()
    target_core = _reliable_core(target_track, nodes)
    core_union = source_core | target_core
    return {
        "scene_name": scene_name,
        "source_kind": source_kind,
        "source_track_id": int(source_ids[0]) if source_kind == "track" else None,
        "source_observation_ids": sorted(map(int, source_obs)),
        "target_track_id": int(target_track_id),
        "target_observation_ids": sorted(map(int, target_ids)),
        "evidence_edge_count": len(edge_rows),
        "evidence_recall_reasons": sorted(set(reason for item in evidence for reason in item["features"]["recall_reasons"])),
        "best_common_visible_superpoint_siou": float(best["common_visible_superpoint_siou"]),
        "mean_common_visible_superpoint_siou": float(np.mean([item["common_visible_superpoint_siou"] for item in edge_rows])),
        "best_bidirectional_reprojection_support": float(max(
            min(item["left_to_right_reprojection_support_ratio"], item["right_to_left_reprojection_support_ratio"])
            for item in edge_rows
        )),
        "best_left_to_right_reprojection_support_ratio": float(max(
            item["left_to_right_reprojection_support_ratio"] for item in edge_rows
        )),
        "best_right_to_left_reprojection_support_ratio": float(max(
            item["right_to_left_reprojection_support_ratio"] for item in edge_rows
        )),
        "mean_relative_depth_weighted_iou": float(np.mean([item["relative_depth_weighted_iou"] for item in edge_rows])),
        "best_relative_depth_weighted_iou": float(max(item["relative_depth_weighted_iou"] for item in edge_rows)),
        "mean_centroid_distance": float(np.mean([item["centroid_distance"] for item in edge_rows])),
        "mean_camera_baseline": float(np.mean([item.get("camera_baseline", 0.0) for item in edge_rows])),
        "mean_rgb_difference": float(np.mean([item["rgb_mean_l2_difference"] for item in edge_rows])),
        "mean_texture_difference": float(np.mean([item["texture_rgb_std_l2_difference"] for item in edge_rows])),
        "mean_normal_difference": float(np.mean([item["normal_difference"] for item in edge_rows])),
        "dino_vits14_masked_crop_cosine": None,
        "dino_vits14_state": best["dino_vits14_state"],
        "source_reliable_core_superpoint_count": len(source_core),
        "target_reliable_core_superpoint_count": len(target_core),
        "reliable_core_jaccard": float(len(source_core & target_core) / max(1, len(core_union))),
        "source_view_count": len({nodes[item]["frame_index"] for item in source_obs}),
        "target_view_count": len({nodes[item]["frame_index"] for item in target_ids}),
        **_contact_summary(source_sp, target_sp, contact_map),
        **_same_frame_summary(source_obs, target_ids, nodes, relation_map),
        "ground_truth_usage": "none",
        "proposal_materialization_applied": False,
        "score_used_for_decision": False,
        "decision_constraint": DECISION_CONSTRAINT,
    }


def build_scene_ledger(scene_name, nodes, tracks, owner_by_observation, pair_features, relation_map, contact_map):
    """Construct keep/no-op and broad alternative family actions without mutation."""
    families = defaultdict(list)
    for (left_id, right_id), features in pair_features.items():
        left_owner, right_owner = owner_by_observation.get(left_id), owner_by_observation.get(right_id)
        if left_owner == right_owner:
            continue
        if left_owner is None and right_owner is not None:
            families[("orphan", left_id, right_owner)].append({"features": features})
        elif right_owner is None and left_owner is not None:
            families[("orphan", right_id, left_owner)].append({"features": features})
        elif left_owner is not None and right_owner is not None:
            left_owner, right_owner = sorted((left_owner, right_owner))
            families[("track_pair", left_owner, right_owner)].append({"features": features})
    relations, actions = [], []
    for track_id, track in sorted(tracks.items()):
        actions.append({
            "scene_name": scene_name, "action_type": "keep", "action_name": f"keep_track_{track_id}",
            "source_track_id": track_id, "target_track_id": track_id,
            "source_observation_ids": list(track["observation_ids"]), "target_observation_ids": list(track["observation_ids"]),
            "affected_observation_count": len(track["observation_ids"]), "affected_tracklet_count": 1,
            "changes_association": False, "would_empty_source_track": False,
            "would_leave_source_below_two_views": False, "proposal_materialization_applied": False,
            "score_used_for_decision": False, "ground_truth_usage": "none", "decision_constraint": DECISION_CONSTRAINT,
        })
    for key, evidence in sorted(families.items()):
        kind, left, right = key
        if kind == "orphan":
            relation = _family_relation(scene_name, "orphan_observation", [left], right, tracks[right], nodes, evidence, relation_map, contact_map, tracks)
            relation["family_relation_kind"] = "orphan_to_track"
            relations.append(relation)
            actions.append({
                **{key: relation[key] for key in ("scene_name", "source_observation_ids", "target_track_id", "target_observation_ids")},
                "action_type": "attach", "action_name": f"attach_observation_{left}_to_track_{right}",
                "source_track_id": None, "changes_association": True,
                "affected_observation_count": 1, "affected_tracklet_count": 2,
                "same_frame_conflict_risk": relation["would_create_same_frame_collision"],
                "would_empty_source_track": False, "would_leave_source_below_two_views": False,
                "family_relation_index": len(relations) - 1, "proposal_materialization_applied": False,
                "score_used_for_decision": False, "ground_truth_usage": "none", "decision_constraint": DECISION_CONSTRAINT,
            })
        else:
            relation = _family_relation(scene_name, "track", [left], right, tracks[right], nodes, evidence, relation_map, contact_map, tracks)
            relation["family_relation_kind"] = "track_to_track"
            relation["track_pair_ids"] = [left, right]
            relations.append(relation)
            reverse_relation = _family_relation(scene_name, "track", [right], left, tracks[left], nodes, evidence, relation_map, contact_map, tracks)
            reverse_relation["family_relation_kind"] = "track_to_track"
            reverse_relation["track_pair_ids"] = [left, right]
            relations.append(reverse_relation)
            common = {
                "scene_name": scene_name, "source_track_id": left, "target_track_id": right,
                "source_observation_ids": list(tracks[left]["observation_ids"]), "target_observation_ids": list(tracks[right]["observation_ids"]),
                "changes_association": True, "family_relation_index": len(relations) - 2,
                "affected_observation_count": len(tracks[left]["observation_ids"]) + len(tracks[right]["observation_ids"]),
                "affected_tracklet_count": 2, "same_frame_conflict_risk": relation["would_create_same_frame_collision"],
                "proposal_materialization_applied": False, "score_used_for_decision": False,
                "ground_truth_usage": "none", "decision_constraint": DECISION_CONSTRAINT,
            }
            actions.append({**common, "action_type": "merge", "action_name": f"merge_tracks_{left}_{right}", "would_empty_source_track": False, "would_leave_source_below_two_views": False})
            for source, target in ((left, right), (right, left)):
                for observation_id in tracks[source]["observation_ids"]:
                    remaining_frames = {nodes[item]["frame_index"] for item in tracks[source]["observation_ids"] if item != observation_id}
                    actions.append({
                        **common, "action_type": "reassign", "action_name": f"reassign_observation_{observation_id}_track_{source}_to_{target}",
                        "source_track_id": source, "target_track_id": target,
                        "source_observation_ids": [observation_id], "target_observation_ids": list(tracks[target]["observation_ids"]),
                        "family_relation_index": len(relations) - 2 if source == left else len(relations) - 1,
                        "affected_observation_count": 1, "affected_tracklet_count": 2,
                        "would_empty_source_track": len(remaining_frames) == 0,
                        "would_leave_source_below_two_views": len(remaining_frames) < 2,
                    })
    return relations, actions


def _build_scene(scene_name, args):
    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    observations = _read_jsonl(args.d1_root / scene_name / "automatic_observations.jsonl")
    visible_by_frame, camera_centers = _visibility_superpoints(
        scene_name, args.dataset_root, args.config_path, np.asarray(processed[:, 9], dtype=np.int64),
        [row["frame_index"] for row in observations],
    )
    nodes = _observation_nodes(
        scene_name, args.d1_root, args.relative_observation_root, processed,
        visible_by_frame, camera_centers, args.dino_state,
    )
    tracks, owner = _track_map(args.track_root / scene_name / "automatic_tracks.json")
    if set(owner) - set(nodes):
        raise ValueError(f"{scene_name} frozen tracks reference unavailable D1 observations")
    relation_map = _same_frame_map(args.d1_root / scene_name / "same_frame_hierarchy_relations.jsonl")
    context = _raw_superpoint_context(processed, ADJACENCY_KNN, ADJACENCY_MAX_DISTANCE, MIN_CONTACT_POINTS, MIN_CONTACT_RATIO)
    contact_map = {}
    for left, edges in context["neighbors"].items():
        for edge in edges:
            contact_map[tuple(sorted((int(left), int(edge["neighbor_superpoint_id"]))))] = edge
    pair_features = {
        pair: _edge_features(nodes[pair[0]], nodes[pair[1]], reasons, visible_by_frame)
        for pair, reasons in _candidate_observation_pairs(nodes, args.recall_knn, args.max_shared_superpoint_nodes).items()
    }
    relations, actions = build_scene_ledger(scene_name, nodes, tracks, owner, pair_features, relation_map, contact_map)
    node_rows = [{
        key: value for key, value in node.items()
        if key not in {
            "points", "relative_points", "relative_weights", "lifted_superpoints",
            "centroid", "camera_center", "mean_rgb", "rgb_std", "mean_normal",
        }
    } | {
        "lifted_superpoint_ids": node["lifted_superpoints"].tolist(),
        "centroid": node["centroid"].tolist(), "camera_center": node["camera_center"].tolist(),
        "mean_rgb": node["mean_rgb"].tolist(),
        "rgb_std": node["rgb_std"].tolist(), "mean_normal": node["mean_normal"].tolist(),
        "current_track_id": owner.get(observation_id), "rle_available_in_d1": True,
        "ground_truth_usage": "none", "decision_constraint": DECISION_CONSTRAINT,
    } for observation_id, node in sorted(nodes.items())]
    summary = {
        "scene_name": scene_name, "d1_observation_count": len(nodes), "frozen_details_track_count": len(tracks),
        "orphan_observation_count": len(set(nodes) - set(owner)), "broad_cross_view_relation_count": len(pair_features),
        "track_family_relation_count": len(relations), "action_counts": dict(Counter(row["action_type"] for row in actions)),
        "proposal_materialization_applied_count": 0, "ap_computed": False, "ground_truth_usage": "none",
        "dino_vits14_state": args.dino_state, "decision_constraint": DECISION_CONSTRAINT,
    }
    published, staging = args.output_root / scene_name, args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "t1_observation_nodes.jsonl", node_rows)
        _write_jsonl(staging / "track_family_relations.jsonl", relations)
        _write_jsonl(staging / "track_family_actions.jsonl", actions)
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--relative-observation-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recall-knn", type=int, default=RECALL_KNN)
    parser.add_argument("--max-shared-superpoint-nodes", type=int, default=RECALL_MAX_SHARED_NODES)
    parser.add_argument("--dino-state", choices=("deferred_cuda_unavailable",), default="deferred_cuda_unavailable")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-resume", action="store_true", help="续跑时不逐场景打印已完成条目。")
    return parser


def main():
    args = build_parser().parse_args()
    for name in ("scene_list", "d1_root", "track_root", "relative_observation_root", "processed_scene_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    if args.recall_knn <= 0 or args.max_shared_superpoint_nodes <= 1:
        raise SystemExit("recall parameters are invalid")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("decision_constraint") != DECISION_CONSTRAINT:
                raise SystemExit(f"resume contract mismatch: {scene_name}")
            if not args.quiet_resume:
                print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(f"[done] {index}/{len(scenes)} {scene_name}: tracks {summary['frozen_details_track_count']}, actions {sum(summary['action_counts'].values())}", flush=True)
        summaries.append(summary)
    totals = Counter()
    for summary in summaries:
        totals.update(summary["action_counts"])
    payload = {
        "ledger_type": "T1a frozen no-GT track-family/action ledger",
        "scene_count": len(summaries), "d1_observation_count": sum(row["d1_observation_count"] for row in summaries),
        "frozen_details_track_count": sum(row["frozen_details_track_count"] for row in summaries),
        "orphan_observation_count": sum(row["orphan_observation_count"] for row in summaries),
        "track_family_relation_count": sum(row["track_family_relation_count"] for row in summaries),
        "action_counts": dict(sorted(totals.items())), "proposal_materialization_applied": False,
        "ap_computed": False, "ground_truth_usage": "none", "dino_vits14_state": args.dino_state,
        "dino_note": "DINO embeddings/cosines are explicitly unavailable in this CUDA-unavailable run; RGB/texture/normal fields are not substitutes.",
        "decision_constraint": DECISION_CONSTRAINT, "scene_summaries": summaries,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("scene_count", "d1_observation_count", "frozen_details_track_count", "track_family_relation_count", "action_counts", "dino_vits14_state")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

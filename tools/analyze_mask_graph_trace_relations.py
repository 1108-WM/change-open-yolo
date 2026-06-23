#!/usr/bin/env python3
"""Diagnose mask-graph relation quality with ScanNet200 GT instance ids."""

import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL


def _iter_trace_paths(path):
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        if "mask_graph_trace.json" in files:
            yield osp.join(root, "mask_graph_trace.json")


def _load_gt_masks(path):
    payload = torch.load(path, map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    return masks.astype(bool, copy=False)


def _load_gt_ids(gt_instance_dir, scene_name, num_points):
    path = osp.join(gt_instance_dir, f"{scene_name}.txt")
    if not osp.exists(path):
        return np.zeros((num_points,), dtype=np.int64)
    ids = np.loadtxt(path, dtype=np.int64)
    if ids.shape[0] != num_points:
        return np.zeros((num_points,), dtype=np.int64)
    return ids


def _load_indices(path):
    if not path or not osp.exists(path):
        return np.asarray([], dtype=np.int64)
    return np.load(path)["point_indices"].astype(np.int64)


def _best_gt_for_indices(indices, gt_masks, gt_ids):
    if len(indices) == 0 or gt_masks.size == 0:
        return {
            "gt_index": -1,
            "gt_instance_id": -1,
            "gt_semantic_id": -1,
            "gt_class_name": "",
            "best_iou": 0.0,
            "precision": 0.0,
            "coverage": 0.0,
        }
    mask = np.zeros((gt_masks.shape[0],), dtype=bool)
    valid = (indices >= 0) & (indices < gt_masks.shape[0])
    mask[indices[valid]] = True
    intersections = np.logical_and(mask[:, None], gt_masks).sum(axis=0)
    unions = int(mask.sum()) + gt_masks.sum(axis=0) - intersections
    ious = intersections / np.maximum(unions, 1)
    best = int(np.argmax(ious)) if len(ious) else -1
    if best < 0:
        return {
            "gt_index": -1,
            "gt_instance_id": -1,
            "gt_semantic_id": -1,
            "gt_class_name": "",
            "best_iou": 0.0,
            "precision": 0.0,
            "coverage": 0.0,
        }
    gt_instance_ids, counts = np.unique(gt_ids[gt_masks[:, best]], return_counts=True)
    valid_ids = gt_instance_ids > 0
    instance_id = -1
    if valid_ids.any():
        gt_instance_ids = gt_instance_ids[valid_ids]
        counts = counts[valid_ids]
        instance_id = int(gt_instance_ids[int(np.argmax(counts))])
    semantic_id = int(instance_id // 1000) if instance_id > 0 else -1
    return {
        "gt_index": best,
        "gt_instance_id": instance_id,
        "gt_semantic_id": semantic_id,
        "gt_class_name": ID_TO_LABEL.get(semantic_id, str(semantic_id)) if semantic_id > 0 else "",
        "best_iou": float(ious[best]),
        "precision": float(intersections[best] / max(1, int(mask.sum()))),
        "coverage": float(intersections[best] / max(1, int(gt_masks[:, best].sum()))),
    }


def _edge_truth_label(left_match, right_match):
    if left_match["gt_instance_id"] <= 0 or right_match["gt_instance_id"] <= 0:
        return "unknown"
    if left_match["gt_instance_id"] == right_match["gt_instance_id"]:
        return "same_instance"
    if left_match["gt_semantic_id"] == right_match["gt_semantic_id"]:
        return "different_same_class_instance"
    return "different_class_instance"


def analyze(args):
    rows = []
    for trace_path in _iter_trace_paths(args.traces):
        with open(trace_path) as f:
            trace = json.load(f)
        scene_name = trace.get("scene_name") or osp.basename(osp.dirname(trace_path))
        gt_path = osp.join(args.gt_masks, f"{scene_name}.pt")
        if not osp.exists(gt_path):
            continue
        gt_masks = _load_gt_masks(gt_path)
        gt_ids = _load_gt_ids(args.gt_instance_dir, scene_name, gt_masks.shape[0])
        observation_matches = {}
        for obs in trace.get("observations", []):
            obs_id = int(obs.get("graph_observation_id", -1))
            indices = _load_indices(obs.get("observation_seed_points_path"))
            observation_matches[obs_id] = {
                **_best_gt_for_indices(indices, gt_masks, gt_ids),
                "class_name": obs.get("class_name", ""),
                "class_id": int(obs.get("class_id", -1)),
                "undersegmentation_bridge_risk": bool(obs.get("undersegmentation_bridge_risk", False)),
            }
        for edge in trace.get("relation_edges", []):
            left_id = int(edge.get("left", -1))
            right_id = int(edge.get("right", -1))
            left_match = observation_matches.get(left_id)
            right_match = observation_matches.get(right_id)
            if left_match is None or right_match is None:
                continue
            truth = _edge_truth_label(left_match, right_match)
            relation_type = str(edge.get("relation_type", ""))
            if relation_type == "same_object":
                correct = truth == "same_instance"
            elif "conflict" in relation_type or relation_type == "same_frame_mutex":
                correct = truth in {"different_same_class_instance", "different_class_instance"}
            elif "containment" in relation_type:
                correct = truth == "same_instance"
            else:
                correct = None
            rows.append(
                {
                    "scene_name": scene_name,
                    "relation_type": relation_type,
                    "truth_label": truth,
                    "relation_correct": "" if correct is None else bool(correct),
                    "left_observation": left_id,
                    "right_observation": right_id,
                    "left_gt_instance_id": left_match["gt_instance_id"],
                    "right_gt_instance_id": right_match["gt_instance_id"],
                    "left_gt_class": left_match["gt_class_name"],
                    "right_gt_class": right_match["gt_class_name"],
                    "left_pred_class": left_match["class_name"],
                    "right_pred_class": right_match["class_name"],
                    "edge_score": float(edge.get("edge_score", 0.0) or 0.0),
                    "same_object_score": float(edge.get("same_object_score", 0.0) or 0.0),
                    "conflict_score": float(edge.get("conflict_score", 0.0) or 0.0),
                    "containment_score": float(edge.get("containment_score", 0.0) or 0.0),
                    "seed_iou": float(edge.get("seed_iou", 0.0) or 0.0),
                    "seed_containment": float(edge.get("seed_containment", 0.0) or 0.0),
                    "depth_consistency": float(edge.get("depth_consistency", 0.0) or 0.0),
                    "mask_consistency": float(edge.get("mask_consistency", 0.0) or 0.0),
                    "left_undersegmentation_bridge_risk": left_match["undersegmentation_bridge_risk"],
                    "right_undersegmentation_bridge_risk": right_match["undersegmentation_bridge_risk"],
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = osp.join(args.output_dir, "mask_graph_relation_diagnostics.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation_type"]].append(row)
    summary = {
        "relation_count": len(rows),
        "relation_type_counts": dict(Counter(row["relation_type"] for row in rows)),
        "truth_label_counts": dict(Counter(row["truth_label"] for row in rows)),
        "by_relation": {},
    }
    for relation_type, items in by_relation.items():
        judged = [item for item in items if item["relation_correct"] != ""]
        summary["by_relation"][relation_type] = {
            "count": len(items),
            "judged_count": len(judged),
            "correct_rate": (
                float(sum(str(item["relation_correct"]) == "True" for item in judged) / len(judged))
                if judged
                else None
            ),
            "truth_label_counts": dict(Counter(item["truth_label"] for item in items)),
        }
    json_path = osp.join(args.output_dir, "mask_graph_relation_diagnostics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote {len(rows)} relation rows to {csv_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, help="A mask_graph_trace.json file or a directory containing traces.")
    parser.add_argument("--gt_masks", default="./output/scannet200/scannet200_ground_truth_masks")
    parser.add_argument("--gt_instance_dir", default="./data/scannet200/ground_truth")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

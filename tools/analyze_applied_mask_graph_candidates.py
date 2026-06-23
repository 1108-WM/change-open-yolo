#!/usr/bin/env python3
"""Analyze exported or applied mask-graph candidates against ScanNet200 GT."""

import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import Counter

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.backprojection_fusion import _load_seed_indices
from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL, PRED_ID_TO_ID


def _load_mask_tensor(path):
    payload = torch.load(path, map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    return masks.astype(bool, copy=False)


def _iter_candidate_json_paths(path):
    if path is None:
        return
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        if "backprojection_candidates.json" in files:
            yield osp.join(root, "backprojection_candidates.json")


def _load_exported_candidates(candidate_roots, source_prefix=None, include_existing_support_diagnostics=False):
    items = []
    for root in candidate_roots:
        for json_path in _iter_candidate_json_paths(root):
            with open(json_path) as f:
                payload = json.load(f)
            scene_name = payload.get("scene_name")
            for candidate in payload.get("candidates", []):
                source_kind = str(candidate.get("source_kind") or payload.get("source_kind") or "")
                if source_prefix and not source_kind.startswith(source_prefix):
                    continue
                item = dict(candidate)
                item["scene_name"] = scene_name
                item["source_json"] = json_path
                item.setdefault("source_kind", source_kind)
                items.append(item)
            if include_existing_support_diagnostics:
                for diagnostic in payload.get("existing_support_diagnostics", []):
                    source_kind = str(diagnostic.get("source_kind") or payload.get("source_kind") or "")
                    if source_prefix and not source_kind.startswith(source_prefix):
                        continue
                    item = dict(diagnostic)
                    item["scene_name"] = scene_name
                    item["source_json"] = json_path
                    item.setdefault("source_kind", source_kind)
                    item.setdefault("candidate_id", item.get("diagnostic_id", -1))
                    item.setdefault("seed_points_path", item.get("output_seed_points_path"))
                    item["is_existing_support_diagnostic"] = True
                    items.append(item)
    return items


def _load_gt_instance_ids(gt_instance_dir, scene_name, gt_masks):
    if not gt_instance_dir:
        return [-1] * gt_masks.shape[1], [-1] * gt_masks.shape[1], [""] * gt_masks.shape[1]
    txt_path = osp.join(gt_instance_dir, f"{scene_name}.txt")
    if not osp.exists(txt_path):
        return [-1] * gt_masks.shape[1], [-1] * gt_masks.shape[1], [""] * gt_masks.shape[1]
    gt_ids = np.loadtxt(txt_path, dtype=np.int64)
    instance_ids = []
    semantic_ids = []
    class_names = []
    for gt_index in range(gt_masks.shape[1]):
        ids, counts = np.unique(gt_ids[gt_masks[:, gt_index]], return_counts=True)
        valid = ids > 0
        if not valid.any():
            instance_id = -1
        else:
            ids = ids[valid]
            counts = counts[valid]
            instance_id = int(ids[int(np.argmax(counts))])
        semantic_id = int(instance_id // 1000) if instance_id > 0 else -1
        instance_ids.append(instance_id)
        semantic_ids.append(semantic_id)
        class_names.append(ID_TO_LABEL.get(semantic_id, str(semantic_id)) if semantic_id > 0 else "")
    return instance_ids, semantic_ids, class_names


def _load_scene_points(dataset_root, scene_name):
    if not dataset_root:
        return None
    scene_id = scene_name.replace("scene", "")
    path = osp.join(dataset_root, scene_name, f"{scene_id}.npy")
    if not osp.exists(path):
        return None
    return np.load(path, mmap_mode="r")[:, :3].astype(np.float32)


def _connected_component_count(points, mask, radius, max_points):
    indices = np.flatnonzero(mask)
    if points is None or len(indices) == 0:
        return 0, 0.0
    if len(indices) > int(max_points):
        return -1, 0.0
    local = points[indices]
    tree = cKDTree(local)
    visited = np.zeros((len(local),), dtype=bool)
    component_sizes = []
    for start in range(len(local)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in tree.query_ball_point(local[current], r=float(radius)):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        component_sizes.append(size)
    return len(component_sizes), float(max(component_sizes) / max(1, len(indices))) if component_sizes else 0.0


def _mask_iou_to_many(mask, masks):
    if masks is None or masks.size == 0:
        return np.zeros((0,), dtype=np.float32)
    intersections = np.logical_and(mask[:, None], masks).sum(axis=0)
    unions = int(mask.sum()) + masks.sum(axis=0) - intersections
    return intersections / np.maximum(unions, 1)


def _mask_overlap_to_many(mask, masks):
    if masks is None or masks.size == 0 or int(mask.sum()) == 0:
        return np.zeros((0,), dtype=np.float32)
    intersections = np.logical_and(mask[:, None], masks).sum(axis=0)
    return intersections / max(1, int(mask.sum()))


def _load_candidate_index(report_path, source_prefix=None):
    if report_path is None:
        return []
    with open(report_path) as f:
        payload = json.load(f)
    applied = []
    for scene_name, scene_report in payload.get("scene_reports", {}).items():
        for item in scene_report.get("applied", []):
            source_kind = str(item.get("source_kind") or "")
            if not source_kind.startswith("mask_graph"):
                continue
            if source_prefix and not source_kind.startswith(source_prefix):
                continue
            if item.get("source_json") is None or item.get("candidate_id") is None:
                continue
            record = dict(item)
            record["scene_name"] = scene_name
            applied.append(record)
    return applied


def _read_candidate(source_json, candidate_id):
    json_path = source_json
    if not osp.isabs(json_path):
        json_path = osp.abspath(json_path)
    with open(json_path) as f:
        payload = json.load(f)
    for candidate in payload.get("candidates", []):
        if int(candidate.get("candidate_id", -1)) == int(candidate_id):
            return candidate
    raise KeyError(f"candidate_id={candidate_id} not found in {source_json}")


def _candidate_mask_from_path(candidate, key, num_points):
    path = candidate.get(key)
    if not path:
        return None
    if not osp.isabs(path):
        path = osp.abspath(path)
    if not osp.exists(path):
        return None
    payload = np.load(path)
    indices = payload["point_indices"].astype(np.int64)
    mask = np.zeros((num_points,), dtype=bool)
    valid = (indices >= 0) & (indices < num_points)
    mask[indices[valid]] = True
    return mask


def _candidate_masks(candidate, num_points):
    output_mask = np.zeros((num_points,), dtype=bool)
    indices = _load_seed_indices(candidate, num_points)
    if indices is not None:
        output_mask[np.asarray(indices, dtype=np.int64)] = True
    masks = {"输出点集": output_mask}
    full_core = _candidate_mask_from_path(candidate, "full_core_seed_points_path", num_points)
    gap_core = _candidate_mask_from_path(candidate, "gap_core_seed_points_path", num_points)
    if full_core is not None:
        masks["完整核心"] = full_core
    if gap_core is not None:
        masks["缺口核心"] = gap_core
    return masks


def _existing_revision_masks(candidate, baseline_masks, num_points):
    if baseline_masks is None or baseline_masks.size == 0:
        return {}
    if not bool(candidate.get("is_existing_support_diagnostic", False)):
        return {}
    existing_id = candidate.get("best_existing_mask_id")
    if existing_id is None or str(existing_id) == "None":
        return {}
    try:
        existing_id = int(existing_id)
    except (TypeError, ValueError):
        return {}
    if existing_id < 0 or existing_id >= baseline_masks.shape[1]:
        return {}
    original = np.asarray(baseline_masks[:, existing_id], dtype=bool)
    full_core = _candidate_mask_from_path(candidate, "full_core_seed_points_path", num_points)
    if full_core is None or int(full_core.sum()) == 0:
        return {"原始已有候选": original}
    return {
        "原始已有候选": original,
        "二维证据修剪核心": full_core,
        "原始加核心补全": np.logical_or(original, full_core),
    }


def _label_row(row):
    if row["overlap_gt_count_10"] >= 2 and row["best_gt_iou"] < 0.50:
        return "多物体错误合并"
    if row["best_gt_iou"] >= 0.50 and row["best_gt_best_baseline_iou"] < 0.50:
        return "完整漏检物体"
    if row["best_gt_iou"] >= 0.50:
        return "重复候选"
    if row["best_gt_iou"] >= 0.25 and row["best_gt_best_baseline_iou"] < 0.50:
        return "可用于补全"
    if row["best_gt_iou"] >= 0.25:
        return "物体残片或重复局部"
    if row["candidate_best_baseline_overlap"] >= 0.50:
        return "重复候选但真实重叠低"
    return "背景污染或几何错误"


def _analyze_one_mask(mask_name, mask, row_base, gt_masks, gt_instance_ids, gt_semantic_ids, gt_class_names, baseline_masks, points, args):
    gt_ious = _mask_iou_to_many(mask, gt_masks)
    gt_candidate_overlaps = _mask_overlap_to_many(mask, gt_masks)
    gt_sizes = gt_masks.sum(axis=0) if gt_masks.size else np.zeros((0,), dtype=np.int64)
    intersections = gt_candidate_overlaps * max(1, int(mask.sum()))
    gt_coverages = intersections / np.maximum(gt_sizes, 1)
    best_gt_index = int(np.argmax(gt_ious)) if len(gt_ious) else -1
    best_gt_iou = float(gt_ious[best_gt_index]) if best_gt_index >= 0 else 0.0
    best_gt_candidate_overlap = float(gt_candidate_overlaps[best_gt_index]) if best_gt_index >= 0 else 0.0
    best_gt_coverage = float(gt_coverages[best_gt_index]) if best_gt_index >= 0 else 0.0
    baseline_ious = _mask_iou_to_many(mask, baseline_masks)
    baseline_overlaps = _mask_overlap_to_many(mask, baseline_masks)
    best_baseline_iou = float(baseline_ious.max(initial=0.0))
    best_baseline_overlap = float(baseline_overlaps.max(initial=0.0))
    if best_gt_index >= 0:
        best_gt_mask = gt_masks[:, best_gt_index]
        gt_to_baseline = _mask_iou_to_many(best_gt_mask, baseline_masks)
        best_gt_best_baseline_iou = float(gt_to_baseline.max(initial=0.0))
        best_gt_instance_id = int(gt_instance_ids[best_gt_index])
        best_gt_semantic_id = int(gt_semantic_ids[best_gt_index])
        best_gt_class_name = gt_class_names[best_gt_index]
    else:
        best_gt_best_baseline_iou = 0.0
        best_gt_instance_id = -1
        best_gt_semantic_id = -1
        best_gt_class_name = ""
    predicted_semantic_id = PRED_ID_TO_ID.get(int(row_base.get("class_id", -1)), -1)
    cc_count, largest_cc_ratio = _connected_component_count(
        points,
        mask,
        radius=args.cc_radius,
        max_points=args.cc_max_points,
    )
    row = {
        **row_base,
        "点集类型": mask_name,
        "candidate_points": int(mask.sum()),
        "candidate_precision": best_gt_candidate_overlap,
        "best_gt_coverage": best_gt_coverage,
        "best_gt_index": best_gt_index,
        "best_gt_instance_id": best_gt_instance_id,
        "best_gt_semantic_id": best_gt_semantic_id,
        "best_gt_class_name": best_gt_class_name,
        "predicted_semantic_id": int(predicted_semantic_id) if predicted_semantic_id is not None else -1,
        "predicted_class_correct": bool(predicted_semantic_id == best_gt_semantic_id and best_gt_semantic_id > 0),
        "best_gt_iou": best_gt_iou,
        "best_gt_candidate_overlap": best_gt_candidate_overlap,
        "best_gt_best_baseline_iou": best_gt_best_baseline_iou,
        "candidate_best_baseline_iou": best_baseline_iou,
        "candidate_best_baseline_overlap": best_baseline_overlap,
        "overlap_gt_count_10": int((gt_candidate_overlaps >= 0.10).sum()),
        "overlap_gt_count_25": int((gt_candidate_overlaps >= 0.25).sum()),
        "connected_component_count": int(cc_count),
        "largest_component_ratio": float(largest_cc_ratio),
    }
    row["诊断类别"] = _label_row(row)
    return row


def analyze(args):
    applied = _load_candidate_index(args.backprojection_report, args.source_prefix)
    exported = _load_exported_candidates(
        [item.strip() for item in str(args.candidates or "").split(",") if item.strip()],
        source_prefix=args.source_prefix,
        include_existing_support_diagnostics=args.include_existing_support_diagnostics,
    )
    if applied:
        raw_items = []
        for item in applied:
            candidate = _read_candidate(item["source_json"], item["candidate_id"])
            merged = dict(candidate)
            merged.update(item)
            merged["source_json"] = item["source_json"]
            raw_items.append(merged)
    else:
        raw_items = exported
    if not raw_items:
        raise ValueError("No candidates to analyze. Provide --backprojection_report or --candidates.")
    rows = []
    for item in raw_items:
        candidate = item
        scene_name = item["scene_name"]
        gt_path = osp.join(args.gt_masks, f"{scene_name}.pt")
        baseline_path = osp.join(args.baseline_masks, f"{scene_name}.pt")
        if not osp.exists(gt_path):
            raise FileNotFoundError(gt_path)
        gt_masks = _load_mask_tensor(gt_path)
        baseline_masks = _load_mask_tensor(baseline_path) if osp.exists(baseline_path) else np.zeros((gt_masks.shape[0], 0), dtype=bool)
        gt_instance_ids, gt_semantic_ids, gt_class_names = _load_gt_instance_ids(args.gt_instance_dir, scene_name, gt_masks)
        points = _load_scene_points(args.dataset_root, scene_name)
        row_base = {
            "scene_name": scene_name,
            "candidate_id": int(item.get("candidate_id", -1)),
            "diagnostic_id": int(item.get("diagnostic_id", -1)),
            "is_existing_support_diagnostic": bool(item.get("is_existing_support_diagnostic", False)),
            "candidate_action": item.get("candidate_action", candidate.get("candidate_action", "")),
            "source_kind": item.get("source_kind", ""),
            "class_name": item.get("class_name", candidate.get("class_name", "")),
            "class_id": int(item.get("class_id", -1)),
            "support_view_count": int(candidate.get("support_view_count", item.get("support_view_count", 0)) or 0),
            "selected_view_count": int(candidate.get("selected_view_count", item.get("selected_view_count", 0)) or 0),
            "same_object_edge_count": int(candidate.get("same_object_edge_count", item.get("same_object_edge_count", 0)) or 0),
            "conflict_edge_count": int(candidate.get("conflict_edge_count", item.get("conflict_edge_count", 0)) or 0),
            "uncertain_edge_count": int(candidate.get("uncertain_edge_count", item.get("uncertain_edge_count", 0)) or 0),
            "external_conflict_edge_count": int(candidate.get("external_conflict_edge_count", item.get("external_conflict_edge_count", 0)) or 0),
            "external_uncertain_edge_count": int(candidate.get("external_uncertain_edge_count", item.get("external_uncertain_edge_count", 0)) or 0),
            "graph_consensus_score": float(candidate.get("graph_consensus_score", item.get("graph_consensus_score", 0.0)) or 0.0),
            "depth_consistency_score": float(candidate.get("depth_consistency_score", item.get("depth_consistency_score", 0.0)) or 0.0),
            "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", item.get("seed_in_existing_mask_ratio", 0.0)) or 0.0),
            "best_existing_iou": float(candidate.get("best_existing_iou", item.get("best_existing_iou", 0.0)) or 0.0),
            "hypothesis_formation_policy": candidate.get("hypothesis_formation_policy", ""),
            "full_core_seed_point_count": int(candidate.get("full_core_seed_point_count", 0) or 0),
            "gap_core_seed_point_count": int(candidate.get("gap_core_seed_point_count", 0) or 0),
            "source_json": item.get("source_json", ""),
        }
        masks_to_analyze = _candidate_masks(candidate, gt_masks.shape[0])
        if args.include_revision_variants:
            masks_to_analyze.update(_existing_revision_masks(candidate, baseline_masks, gt_masks.shape[0]))
        for mask_name, mask in masks_to_analyze.items():
            if int(mask.sum()) == 0:
                continue
            rows.append(
                _analyze_one_mask(
                    mask_name,
                    mask,
                    row_base,
                    gt_masks,
                    gt_instance_ids,
                    gt_semantic_ids,
                    gt_class_names,
                    baseline_masks,
                    points,
                    args,
                )
            )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = osp.join(args.output_dir, "applied_mask_graph_candidates.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "candidate_count": len(rows),
        "label_counts": dict(Counter(row["诊断类别"] for row in rows)),
        "mask_type_counts": dict(Counter(row["点集类型"] for row in rows)),
        "mean_best_gt_iou": float(np.mean([row["best_gt_iou"] for row in rows])) if rows else 0.0,
        "mean_candidate_precision": float(np.mean([row["candidate_precision"] for row in rows])) if rows else 0.0,
        "mean_best_gt_coverage": float(np.mean([row["best_gt_coverage"] for row in rows])) if rows else 0.0,
        "mean_best_gt_best_baseline_iou": float(np.mean([row["best_gt_best_baseline_iou"] for row in rows])) if rows else 0.0,
        "mean_candidate_best_baseline_overlap": float(np.mean([row["candidate_best_baseline_overlap"] for row in rows])) if rows else 0.0,
    }
    json_path = osp.join(args.output_dir, "applied_mask_graph_candidates_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote {len(rows)} rows to {csv_path}")
    print(f"[INFO] Wrote summary to {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backprojection_report", default=None)
    parser.add_argument("--candidates", default=None, help="Candidate directory or JSON. Used when --backprojection_report is omitted.")
    parser.add_argument("--include_existing_support_diagnostics", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include_revision_variants", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gt_masks", default="./output/scannet200/scannet200_ground_truth_masks")
    parser.add_argument("--gt_instance_dir", default="./data/scannet200/ground_truth")
    parser.add_argument("--baseline_masks", default="./output/scannet200/scannet200_masks")
    parser.add_argument("--dataset_root", default="./data/scannet200")
    parser.add_argument("--source_prefix", default="mask_graph")
    parser.add_argument("--cc_radius", default=0.03, type=float)
    parser.add_argument("--cc_max_points", default=50000, type=int)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

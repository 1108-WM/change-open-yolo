#!/usr/bin/env python3
"""Analyze mask-graph candidates that were actually applied in an evaluation report."""

import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import Counter

import numpy as np
import torch

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.backprojection_fusion import _load_seed_indices


def _load_mask_tensor(path):
    payload = torch.load(path, map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    return masks.astype(bool, copy=False)


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


def analyze(args):
    applied = _load_candidate_index(args.backprojection_report, args.source_prefix)
    rows = []
    for item in applied:
        scene_name = item["scene_name"]
        gt_path = osp.join(args.gt_masks, f"{scene_name}.pt")
        baseline_path = osp.join(args.baseline_masks, f"{scene_name}.pt")
        if not osp.exists(gt_path):
            raise FileNotFoundError(gt_path)
        gt_masks = _load_mask_tensor(gt_path)
        baseline_masks = _load_mask_tensor(baseline_path) if osp.exists(baseline_path) else np.zeros((gt_masks.shape[0], 0), dtype=bool)
        candidate = _read_candidate(item["source_json"], item["candidate_id"])
        indices = _load_seed_indices(candidate, gt_masks.shape[0])
        if indices is None or len(indices) == 0:
            continue
        mask = np.zeros((gt_masks.shape[0],), dtype=bool)
        mask[indices] = True
        gt_ious = _mask_iou_to_many(mask, gt_masks)
        gt_overlaps = _mask_overlap_to_many(mask, gt_masks)
        best_gt_index = int(np.argmax(gt_ious)) if len(gt_ious) else -1
        best_gt_iou = float(gt_ious[best_gt_index]) if best_gt_index >= 0 else 0.0
        best_gt_overlap = float(gt_overlaps[best_gt_index]) if best_gt_index >= 0 else 0.0
        baseline_ious = _mask_iou_to_many(mask, baseline_masks)
        baseline_overlaps = _mask_overlap_to_many(mask, baseline_masks)
        best_baseline_iou = float(baseline_ious.max(initial=0.0))
        best_baseline_overlap = float(baseline_overlaps.max(initial=0.0))
        if best_gt_index >= 0:
            best_gt_mask = gt_masks[:, best_gt_index]
            gt_to_baseline = _mask_iou_to_many(best_gt_mask, baseline_masks)
            best_gt_best_baseline_iou = float(gt_to_baseline.max(initial=0.0))
        else:
            best_gt_best_baseline_iou = 0.0
        row = {
            "scene_name": scene_name,
            "candidate_id": int(item.get("candidate_id", -1)),
            "source_kind": item.get("source_kind", ""),
            "class_name": item.get("class_name", candidate.get("class_name", "")),
            "candidate_points": int(mask.sum()),
            "best_gt_index": best_gt_index,
            "best_gt_iou": best_gt_iou,
            "best_gt_candidate_overlap": best_gt_overlap,
            "best_gt_best_baseline_iou": best_gt_best_baseline_iou,
            "candidate_best_baseline_iou": best_baseline_iou,
            "candidate_best_baseline_overlap": best_baseline_overlap,
            "overlap_gt_count_10": int((gt_overlaps >= 0.10).sum()),
            "overlap_gt_count_25": int((gt_overlaps >= 0.25).sum()),
            "support_view_count": int(candidate.get("support_view_count", item.get("support_view_count", 0)) or 0),
            "selected_view_count": int(candidate.get("selected_view_count", item.get("selected_view_count", 0)) or 0),
            "same_object_edge_count": int(candidate.get("same_object_edge_count", item.get("same_object_edge_count", 0)) or 0),
            "conflict_edge_count": int(candidate.get("conflict_edge_count", item.get("conflict_edge_count", 0)) or 0),
            "graph_consensus_score": float(candidate.get("graph_consensus_score", item.get("graph_consensus_score", 0.0)) or 0.0),
            "depth_consistency_score": float(candidate.get("depth_consistency_score", item.get("depth_consistency_score", 0.0)) or 0.0),
            "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", item.get("seed_in_existing_mask_ratio", 0.0)) or 0.0),
            "best_existing_iou": float(candidate.get("best_existing_iou", item.get("best_existing_iou", 0.0)) or 0.0),
            "source_json": item.get("source_json", ""),
        }
        row["诊断类别"] = _label_row(row)
        rows.append(row)

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
        "mean_best_gt_iou": float(np.mean([row["best_gt_iou"] for row in rows])) if rows else 0.0,
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
    parser.add_argument("--backprojection_report", required=True)
    parser.add_argument("--gt_masks", default="./output/scannet200/scannet200_ground_truth_masks")
    parser.add_argument("--baseline_masks", default="./output/scannet200/scannet200_masks")
    parser.add_argument("--source_prefix", default="mask_graph")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

#!/usr/bin/env python3
"""仅离线真值诊断：为 SAM2 MVPDist 候选建立几何、语义和融合门槛账本。"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = set(int(value) for value in VALID_CLASS_IDS_200_INST)
    instances = {}
    for instance_id in np.unique(ids):
        semantic_id = int(instance_id) // 1000
        indices = np.flatnonzero(ids == instance_id).astype(np.int32)
        if int(instance_id) <= 0 or semantic_id not in valid_classes or len(indices) < min_region_size:
            continue
        instances[int(instance_id)] = {
            "indices": indices,
            "point_count": int(len(indices)),
            "class_name": str(ID_TO_LABEL.get(semantic_id, semantic_id)),
        }
    return ids, instances


def _match(indices, gt_ids, gt_instances):
    if not len(indices):
        return {"gt_instance_id": -1, "best_iou": 0.0, "precision": 0.0, "coverage": 0.0, "overlap_gt_count_10pct": 0}
    instance_ids, counts = np.unique(gt_ids[indices], return_counts=True)
    best = {"gt_instance_id": -1, "best_iou": 0.0, "precision": 0.0, "coverage": 0.0}
    overlap_count = 0
    for instance_id, intersection in zip(instance_ids, counts):
        instance_id, intersection = int(instance_id), int(intersection)
        if instance_id not in gt_instances:
            continue
        if intersection / len(indices) >= 0.10:
            overlap_count += 1
        size = gt_instances[instance_id]["point_count"]
        iou = intersection / max(1, len(indices) + size - intersection)
        if iou > best["best_iou"]:
            best = {
                "gt_instance_id": instance_id,
                "best_iou": float(iou),
                "precision": float(intersection / len(indices)),
                "coverage": float(intersection / size),
            }
    best["overlap_gt_count_10pct"] = overlap_count
    return best


def _geometry_error(match):
    if match["gt_instance_id"] < 0:
        return "background_or_non_instance"
    if match["overlap_gt_count_10pct"] >= 2 and match["precision"] < 0.70:
        return "mixed_multiple_instances"
    if match["best_iou"] >= 0.50:
        return "good_geometry"
    if match["coverage"] >= 0.50 and match["precision"] < 0.70:
        return "overmerged_or_boundary_leakage"
    if match["precision"] >= 0.70 and match["coverage"] < 0.50:
        return "fragment_or_undercoverage"
    if match["best_iou"] >= 0.25:
        return "partial_overlap"
    return "weak_or_wrong_geometry"


def _fusion_decisions(report_path):
    if report_path is None:
        return {}
    report = json.loads(Path(report_path).read_text())
    decisions = {}
    for scene_name, item in report.get("scene_reports", {}).items():
        fusion = item.get("backprojection") if isinstance(item, dict) else None
        if fusion is None and isinstance(item, dict) and "applied" in item and "skipped" in item:
            fusion = item
        if not fusion:
            continue
        for decision in fusion.get("applied", []):
            if decision.get("source_kind") == "sam2_details_mvpdist":
                decisions[(scene_name, int(decision["candidate_id"]))] = "applied"
        for decision in fusion.get("skipped", []):
            if decision.get("source_kind") == "sam2_details_mvpdist":
                decisions[(scene_name, int(decision["candidate_id"]))] = str(decision.get("reason", "skipped"))
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fusion_report", type=Path, default=None)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--final_score_threshold", type=float, default=0.20)
    parser.add_argument("--max_existing_iou", type=float, default=0.30)
    parser.add_argument("--max_seed_in_existing_ratio", type=float, default=0.70)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("Pass --allow_gt_diagnostics: GT must never enter inference, scoring, merging, or threshold selection.")

    decisions = _fusion_decisions(args.fusion_report)
    rows = []
    gt_coverage = defaultdict(list)
    for scene_name in _read_scenes(args.scene_list):
        gt_ids, gt_instances = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
        for gt_instance_id in gt_instances:
            gt_coverage[(scene_name, gt_instance_id)]
        candidate_path = args.candidate_root / scene_name / "backprojection_candidates.json"
        payload = json.loads(candidate_path.read_text())
        for candidate in payload.get("candidates", []):
            indices = np.unique(np.load(_resolve(candidate["seed_points_path"]))["point_indices"].astype(np.int64))
            indices = indices[(indices >= 0) & (indices < len(gt_ids))]
            match = _match(indices, gt_ids, gt_instances)
            gt_info = gt_instances.get(match["gt_instance_id"], {})
            candidate_id = int(candidate["candidate_id"])
            row = {
                "scene_name": scene_name,
                "candidate_id": candidate_id,
                "source_track_ids": json.dumps(candidate.get("source_track_ids", [])),
                "point_count": int(len(indices)),
                "mvpdist_class": str(candidate["class_name"]),
                "mvpdist_score": float(candidate["score"]),
                "fusion_score": float(candidate.get("fusion_score", candidate["score"])),
                "support_score": float(candidate.get("support_score", 0.0)),
                "best_existing_iou": float(candidate.get("best_existing_iou", 0.0)),
                "seed_in_existing_mask_ratio": float(candidate.get("seed_in_existing_mask_ratio", 0.0)),
                "gt_instance_id": int(match["gt_instance_id"]),
                "gt_class": str(gt_info.get("class_name", "")),
                "best_iou": float(match["best_iou"]),
                "precision": float(match["precision"]),
                "coverage": float(match["coverage"]),
                "geometry_error": _geometry_error(match),
                "semantic_exact": bool(gt_info and candidate["class_name"] == gt_info["class_name"]),
                "passes_final_score": bool(float(candidate["score"]) >= args.final_score_threshold),
                "passes_export_overlap": bool(
                    float(candidate.get("best_existing_iou", 0.0)) <= args.max_existing_iou
                    and float(candidate.get("seed_in_existing_mask_ratio", 0.0)) <= args.max_seed_in_existing_ratio
                ),
                "fusion_decision": decisions.get((scene_name, candidate_id), "unavailable"),
            }
            rows.append(row)
            if row["gt_instance_id"] >= 0:
                gt_coverage[(scene_name, row["gt_instance_id"])].append(row)

    def _rate(predicate, values):
        return float(sum(bool(predicate(row)) for row in values) / max(1, len(values)))

    groups = {
        "all": rows,
        "iou25": [row for row in rows if row["best_iou"] >= 0.25],
        "iou50": [row for row in rows if row["best_iou"] >= 0.50],
        "passes_final_score": [row for row in rows if row["passes_final_score"]],
        "passes_export_overlap": [row for row in rows if row["passes_export_overlap"]],
        "passes_final_score_and_overlap": [
            row for row in rows if row["passes_final_score"] and row["passes_export_overlap"]
        ],
    }
    summary = {
        "gt_usage": "OFFLINE DIAGNOSTICS ONLY; never inference, scoring, merging, or threshold selection.",
        "num_candidates": len(rows),
        "candidate_geometry_error_counts": dict(sorted(Counter(row["geometry_error"] for row in rows).items())),
        "groups": {
            name: {
                "count": len(group),
                "semantic_exact_accuracy": _rate(lambda row: row["semantic_exact"], group),
                "mean_best_iou": float(np.mean([row["best_iou"] for row in group])) if group else 0.0,
            }
            for name, group in groups.items()
        },
        "gt_recall_at_iou25": _rate(
            lambda values: max((row["best_iou"] for row in values), default=0.0) >= 0.25,
            gt_coverage.values(),
        ),
        "gt_recall_at_iou50": _rate(
            lambda values: max((row["best_iou"] for row in values), default=0.0) >= 0.50,
            gt_coverage.values(),
        ),
        "fusion_decision_counts": dict(sorted(Counter(row["fusion_decision"] for row in rows).items())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidate_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

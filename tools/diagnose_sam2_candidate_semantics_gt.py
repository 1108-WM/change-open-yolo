#!/usr/bin/env python3
"""Measure semantic quality of SAM2 candidates after GT-only geometry matching.

This is an offline diagnostic only. Ground truth is used to identify which
candidates have adequate geometry, never for candidate generation, scoring,
selection, or test-time inference.
"""

import argparse
import csv
import json
from pathlib import Path


def _scenes(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _top_class(record):
    topk = record.get("clip_topk") or []
    return str(topk[0].get("class_name", "")) if topk else ""


def _top_score(record):
    topk = record.get("clip_topk") or []
    return float(topk[0].get("prob", 0.0)) if topk else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_diagnostics", type=Path, required=True)
    parser.add_argument("--clip_features", type=Path, required=True)
    parser.add_argument("--mvpdist_candidates", type=Path, default=None)
    parser.add_argument("--scene_names", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_gt_diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise ValueError("Pass --allow_gt_diagnostics to acknowledge that GT is diagnostic-only.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_name in _scenes(args.scene_names):
        feature_path = args.clip_features / scene_name / "multiview_object_clip_features.json"
        payload = json.loads(feature_path.read_text())
        features = {
            int(item["candidate_id"]): item
            for item in payload.get("features", [])
            if item.get("source_kind") == "sam2_details" and item.get("candidate_id") is not None
        }
        mvpdist = {}
        if args.mvpdist_candidates is not None:
            candidate_path = args.mvpdist_candidates / scene_name / "backprojection_candidates.json"
            mvpdist = {
                int(item["candidate_id"]): item
                for item in json.loads(candidate_path.read_text()).get("candidates", [])
            }
        diagnostic_path = args.geometry_diagnostics / scene_name / "candidate_diagnostics.csv"
        for candidate in _read_csv(diagnostic_path):
            candidate_id = int(candidate["candidate_id"])
            feature = features.get(candidate_id)
            gt_class = str(candidate["best_gt_class"])
            yolo_class = "" if feature is None else str(feature.get("pred_class_name", ""))
            alpha_class = "" if feature is None else _top_class(feature)
            mvpdist_record = mvpdist.get(candidate_id)
            mvpdist_class = "" if mvpdist_record is None else str(mvpdist_record.get("class_name", ""))
            rows.append(
                {
                    "scene_name": scene_name,
                    "candidate_id": candidate_id,
                    "best_iou": float(candidate["best_iou"]),
                    "error_type": str(candidate["error_type"]),
                    "gt_class": gt_class,
                    "feature_exported": feature is not None,
                    "yolo_class": yolo_class,
                    "yolo_exact": bool(gt_class and yolo_class == gt_class),
                    "alpha_class": alpha_class,
                    "alpha_score": _top_score(feature) if feature else 0.0,
                    "alpha_exact": bool(gt_class and alpha_class == gt_class),
                    "mvpdist_class": mvpdist_class,
                    "mvpdist_score": 0.0 if mvpdist_record is None else float(mvpdist_record.get("score", 0.0)),
                    "mvpdist_exact": bool(gt_class and mvpdist_class == gt_class),
                }
            )

    fields = list(rows[0]) if rows else []
    with (args.output_dir / "candidate_semantic_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"gt_usage": "OFFLINE DIAGNOSTICS ONLY", "num_candidates": len(rows), "groups": {}}
    for name, predicate in {
        "all": lambda row: True,
        "iou25": lambda row: row["best_iou"] >= 0.25,
        "iou50": lambda row: row["best_iou"] >= 0.50,
        "good_geometry": lambda row: row["error_type"] == "good_geometry",
    }.items():
        group = [row for row in rows if predicate(row)]
        exported = [row for row in group if row["feature_exported"]]
        summary["groups"][name] = {
            "count": len(group),
            "feature_export_recall": float(len(exported) / max(1, len(group))),
            "yolo_exact_accuracy": float(sum(row["yolo_exact"] for row in exported) / max(1, len(exported))),
            "alphaclip_exact_accuracy": float(sum(row["alpha_exact"] for row in exported) / max(1, len(exported))),
            "alphaclip_mean_top_score": float(sum(row["alpha_score"] for row in exported) / max(1, len(exported))),
            "mvpdist_exact_accuracy": float(sum(row["mvpdist_exact"] for row in group) / max(1, len(group))),
            "mvpdist_mean_score": float(sum(row["mvpdist_score"] for row in group) / max(1, len(group))),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline GT-only diagnosis for category-agnostic SAM2 refined instances.

This tool must never be used by candidate generation, scoring, merging, or
test-time inference.  It measures geometry-oracle coverage on labelled scenes
to identify whether failures come from missed instances, fragments, mixed
instances, or duplicates.  It writes diagnostics only.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.eval_semantic_instance import ID_TO_LABEL
from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scene_array_path(dataset_root, scene_name):
    return dataset_root / scene_name / f"{scene_name.removeprefix('scene')}.npy"


def _load_gt_instances(gt_path, min_region_size):
    gt_ids = np.loadtxt(gt_path, dtype=np.int64)
    valid_class_ids = set(int(item) for item in VALID_CLASS_IDS_200_INST)
    instance_indices = {}
    instance_meta = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        semantic_id = instance_id // 1000
        if instance_id <= 0 or semantic_id not in valid_class_ids:
            continue
        indices = np.flatnonzero(gt_ids == instance_id).astype(np.int32)
        if len(indices) < int(min_region_size):
            continue
        instance_indices[instance_id] = indices
        instance_meta[instance_id] = {
            "gt_instance_id": instance_id,
            "semantic_id": semantic_id,
            "class_name": ID_TO_LABEL.get(semantic_id, str(semantic_id)),
            "point_count": int(len(indices)),
        }
    return gt_ids, instance_indices, instance_meta


def _candidate_match(indices, gt_ids, gt_meta):
    point_count = int(len(indices))
    if not point_count:
        return {
            "best_gt_instance_id": -1,
            "best_iou": 0.0,
            "precision": 0.0,
            "coverage": 0.0,
            "matched_points": 0,
            "valid_gt_point_fraction": 0.0,
            "overlap_gt_count_10pct": 0,
        }
    hit_ids, hit_counts = np.unique(gt_ids[indices], return_counts=True)
    best = None
    valid_points = 0
    overlaps = 0
    for instance_id, intersection in zip(hit_ids, hit_counts):
        instance_id = int(instance_id)
        intersection = int(intersection)
        if instance_id not in gt_meta:
            continue
        valid_points += intersection
        if intersection / point_count >= 0.10:
            overlaps += 1
        gt_size = int(gt_meta[instance_id]["point_count"])
        iou = intersection / max(1, point_count + gt_size - intersection)
        if best is None or iou > best["best_iou"]:
            best = {
                "best_gt_instance_id": instance_id,
                "best_iou": float(iou),
                "precision": float(intersection / point_count),
                "coverage": float(intersection / gt_size),
                "matched_points": intersection,
            }
    if best is None:
        best = {
            "best_gt_instance_id": -1,
            "best_iou": 0.0,
            "precision": 0.0,
            "coverage": 0.0,
            "matched_points": 0,
        }
    best["valid_gt_point_fraction"] = float(valid_points / point_count)
    best["overlap_gt_count_10pct"] = overlaps
    return best


def _candidate_error_type(match):
    if match["best_gt_instance_id"] < 0:
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


def _write_visual(points, candidate_indices, gt_indices, title, path, seed):
    rng = np.random.default_rng(seed)
    canvas = np.arange(len(points))
    if len(canvas) > 25000:
        canvas = rng.choice(canvas, size=25000, replace=False)
    gt_set = np.zeros((len(points),), dtype=bool)
    gt_set[gt_indices] = True
    candidate_set = np.zeros((len(points),), dtype=bool)
    candidate_set[candidate_indices] = True
    union = gt_set | candidate_set
    if union.any():
        selected = np.flatnonzero(union)
        selected_xz = points[selected][:, [0, 2]]
        lo = selected_xz.min(axis=0)
        hi = selected_xz.max(axis=0)
        padding = np.maximum((hi - lo) * 0.12, 0.05)
    else:
        lo = points[:, [0, 2]].min(axis=0)
        hi = points[:, [0, 2]].max(axis=0)
        padding = np.zeros(2, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, zoom in zip(axes, (False, True)):
        axis.scatter(points[canvas, 0], points[canvas, 2], s=0.25, c="#9aa0a6", alpha=0.25, linewidths=0)
        axis.scatter(points[gt_set, 0], points[gt_set, 2], s=0.45, c="#e45756", alpha=0.75, linewidths=0)
        axis.scatter(points[candidate_set, 0], points[candidate_set, 2], s=0.45, c="#00a6a6", alpha=0.75, linewidths=0)
        overlap = gt_set & candidate_set
        axis.scatter(points[overlap, 0], points[overlap, 2], s=0.55, c="#6c5ce7", alpha=0.9, linewidths=0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("z")
        if zoom:
            axis.set_xlim(lo[0] - padding[0], hi[0] + padding[0])
            axis.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
            axis.set_title("candidate / best GT zoom")
        else:
            axis.set_title("scene context")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _analyze_scene(scene_name, args):
    candidate_scene = args.candidate_root / scene_name
    records_path = candidate_scene / "instances.jsonl"
    gt_path = args.gt_instance_dir / f"{scene_name}.txt"
    points = np.load(_scene_array_path(args.dataset_root, scene_name), mmap_mode="r")[:, :3].astype(np.float32)
    gt_ids, gt_indices, gt_meta = _load_gt_instances(gt_path, args.min_region_size)
    if len(gt_ids) != len(points):
        raise ValueError(f"GT point count mismatch for {scene_name}: {len(gt_ids)} vs {len(points)}")

    output_scene = args.output_dir / scene_name
    visual_dir = output_scene / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    gt_to_candidates = defaultdict(list)
    for record in _read_jsonl(records_path):
        indices_path = Path(record["point_indices_path"])
        if not indices_path.is_absolute():
            indices_path = PROJECT_ROOT / indices_path
        indices = np.load(indices_path)["point_indices"].astype(np.int32)
        indices = np.unique(indices[(indices >= 0) & (indices < len(points))])
        match = _candidate_match(indices, gt_ids, gt_meta)
        gt_id = int(match["best_gt_instance_id"])
        gt_info = gt_meta.get(gt_id, {})
        candidate_id = int(record.get("instance_id", record.get("track_id", -1)))
        source_track_ids = record.get("source_track_ids")
        if source_track_ids is None:
            source_track_ids = [int(record.get("source_track_id", record.get("track_id", -1)))]
        row = {
            "scene_name": scene_name,
            "candidate_id": candidate_id,
            "source_track_ids": list(source_track_ids),
            "point_count": int(len(indices)),
            "superpoint_count": int(record.get("superpoint_count", 0)),
            "support_score": float(record.get("support_score", 0.0)),
            **match,
            "best_gt_class": gt_info.get("class_name", ""),
            "best_gt_point_count": int(gt_info.get("point_count", 0)),
            "error_type": _candidate_error_type(match),
        }
        if gt_id in gt_meta:
            gt_to_candidates[gt_id].append(row)
            visual_name = f"candidate{row['candidate_id']:04d}_{row['error_type']}.png"
            _write_visual(
                points,
                indices,
                gt_indices[gt_id],
                f"{scene_name} candidate {row['candidate_id']} | {row['best_gt_class']} | "
                f"IoU={row['best_iou']:.3f}, P={row['precision']:.3f}, R={row['coverage']:.3f}",
                visual_dir / visual_name,
                seed=int(row["candidate_id"]),
            )
            row["visual_path"] = str(visual_dir / visual_name)
        candidate_rows.append(row)

    gt_rows = []
    for gt_id, meta in gt_meta.items():
        matched = sorted(gt_to_candidates.get(gt_id, []), key=lambda item: item["best_iou"], reverse=True)
        best = matched[0] if matched else None
        at_25 = [item for item in matched if item["best_iou"] >= 0.25]
        at_50 = [item for item in matched if item["best_iou"] >= 0.50]
        status = "missed" if not best or best["best_iou"] < 0.25 else "covered"
        if len(at_25) >= 2:
            status = "fragmented_or_duplicate"
        gt_rows.append(
            {
                **meta,
                "scene_name": scene_name,
                "best_candidate_id": int(best["candidate_id"]) if best else -1,
                "best_iou": float(best["best_iou"]) if best else 0.0,
                "best_precision": float(best["precision"]) if best else 0.0,
                "best_coverage": float(best["coverage"]) if best else 0.0,
                "candidate_count_iou25": len(at_25),
                "candidate_count_iou50": len(at_50),
                "status": status,
            }
        )

    def rate(predicate, rows):
        return float(sum(bool(predicate(row)) for row in rows) / max(1, len(rows)))

    summary = {
        "scene_name": scene_name,
        "num_gt_instances": len(gt_rows),
        "num_candidates": len(candidate_rows),
        "geometry_oracle": {
            "candidate_precision_at_iou25": rate(lambda row: row["best_iou"] >= 0.25, candidate_rows),
            "candidate_precision_at_iou50": rate(lambda row: row["best_iou"] >= 0.50, candidate_rows),
            "gt_recall_at_iou25": rate(lambda row: row["best_iou"] >= 0.25, gt_rows),
            "gt_recall_at_iou50": rate(lambda row: row["best_iou"] >= 0.50, gt_rows),
            "mean_best_candidate_iou": float(np.mean([row["best_iou"] for row in candidate_rows])) if candidate_rows else 0.0,
        },
        "candidate_error_counts": dict(sorted(Counter(row["error_type"] for row in candidate_rows).items())),
        "gt_status_counts": dict(sorted(Counter(row["status"] for row in gt_rows).items())),
    }
    with (output_scene / "candidate_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(candidate_rows[0]) if candidate_rows else [])
        if candidate_rows:
            writer.writeheader()
            writer.writerows(candidate_rows)
    with (output_scene / "gt_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(gt_rows[0]) if gt_rows else [])
        if gt_rows:
            writer.writeheader()
            writer.writerows(gt_rows)
    (output_scene / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--scene_names", required=True, help="Comma-separated labelled scenes.")
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt_instance_dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_region_size", type=int, default=100)
    parser.add_argument("--allow_gt_diagnostics", action="store_true", help="Required acknowledgement: outputs are diagnostics, never inference inputs.")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("Refusing to run without --allow_gt_diagnostics; GT must remain offline diagnostics only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    reports = [_analyze_scene(scene, args) for scene in scenes]
    overall = {
        "gt_usage": "OFFLINE DIAGNOSTICS ONLY; never candidate generation, scoring, merging, or inference.",
        "scene_summaries": reports,
        "overall": {
            "num_gt_instances": sum(item["num_gt_instances"] for item in reports),
            "num_candidates": sum(item["num_candidates"] for item in reports),
            "gt_recall_at_iou25": float(np.average([item["geometry_oracle"]["gt_recall_at_iou25"] for item in reports], weights=[max(1, item["num_gt_instances"]) for item in reports])),
            "gt_recall_at_iou50": float(np.average([item["geometry_oracle"]["gt_recall_at_iou50"] for item in reports], weights=[max(1, item["num_gt_instances"]) for item in reports])),
        },
    }
    (args.output_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n")
    print(json.dumps(overall["overall"], indent=2))


if __name__ == "__main__":
    main()

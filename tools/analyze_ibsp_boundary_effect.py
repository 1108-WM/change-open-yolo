import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_SCENES = ("scene0011_00", "scene0077_00", "scene0608_01")


def _scene_file(root, scene_name):
    return Path(root) / scene_name / f"{scene_name.replace('scene', '')}.npy"


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _partition_stats(reference, compared):
    pairs = np.stack((reference, compared), axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    reference_ids, reference_child_counts = np.unique(unique_pairs[:, 0], return_counts=True)
    compared_ids, compared_parent_counts = np.unique(unique_pairs[:, 1], return_counts=True)
    reference_sizes = np.bincount(reference, minlength=int(reference.max()) + 1)
    compared_sizes = np.bincount(compared, minlength=int(compared.max()) + 1)
    split_ids = reference_ids[reference_child_counts > 1]
    merged_ids = compared_ids[compared_parent_counts > 1]
    return {
        "reference_segments": int(len(reference_ids)),
        "compared_segments": int(len(compared_ids)),
        "split_reference_segments": int(len(split_ids)),
        "split_reference_points": int(reference_sizes[split_ids].sum()) if len(split_ids) else 0,
        "largest_reference_child_count": int(reference_child_counts.max(initial=0)),
        "merged_compared_segments": int(len(merged_ids)),
        "merged_compared_points": int(compared_sizes[merged_ids].sum()) if len(merged_ids) else 0,
        "largest_compared_parent_count": int(compared_parent_counts.max(initial=0)),
        "pair_count": int(len(unique_pairs)),
        "pair_support_points": int(pair_counts.sum()),
    }


def analyze(args):
    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    ibsp_summary = json.loads(Path(args.ibsp_summary).read_text())
    ibsp_by_scene = {item["scene_name"]: item for item in ibsp_summary["scenes"]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_name in args.scenes:
        baseline = np.load(_scene_file(args.baseline_root, scene_name), mmap_mode="r")[:, 9].astype(np.int64)
        ibsp = np.load(_scene_file(args.ibsp_root, scene_name), mmap_mode="r")[:, 9].astype(np.int64)
        if baseline.shape != ibsp.shape:
            raise ValueError(f"Point count mismatch for {scene_name}: {baseline.shape} vs {ibsp.shape}")
        boundary = ibsp_by_scene[scene_name]["boundary"]
        row = {
            "scene_name": scene_name,
            "points": int(len(baseline)),
            "boundary_used_frames": int(boundary.get("used_frames", 0)),
            "boundary_observed_edges": int(boundary.get("edges_with_boundary_observation", 0)),
            "boundary_conflict_edges": int(boundary.get("edges_with_conflict", 0)),
            "boundary_pruned_edges": int(boundary.get("pruned_edges", 0)),
            "boundary_pruned_edge_ratio": float(boundary.get("pruned_edge_ratio", 0.0)),
        }
        row.update(_partition_stats(baseline, ibsp))
        rows.append(row)

    fields = list(rows[0]) if rows else []
    _write_csv(output_dir / "scene_summary.csv", rows, fields)
    summary = {
        "scope": "diagnostic_only_no_ap_no_fusion_change",
        "baseline_root": args.baseline_root,
        "ibsp_root": args.ibsp_root,
        "baseline_summary": args.baseline_summary,
        "ibsp_summary": args.ibsp_summary,
        "scenes": rows,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose the structural effect of IBSp boundary pruning.")
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--ibsp_root", required=True)
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--ibsp_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

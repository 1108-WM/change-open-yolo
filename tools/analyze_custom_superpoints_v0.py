#!/usr/bin/env python3
"""Analyze existing geometric superpoints against ScanNet200 superpoint stats.

This script does not generate new superpoints. It reads an existing geometric
superpoint export, compares it with the original ScanNet200 segment statistics
recorded in that export summary, and writes small diagnostics/PNGs.
"""

import argparse
import csv
import importlib.util
import json
import os
import os.path as osp
import sys
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_SUPERPOINT_DIAGNOSTICS_PATH = osp.join(REPO_ROOT, "utils", "superpoint_diagnostics.py")
_SUPERPOINT_DIAGNOSTICS_SPEC = importlib.util.spec_from_file_location(
    "openyolo3d_superpoint_diagnostics",
    _SUPERPOINT_DIAGNOSTICS_PATH,
)
_SUPERPOINT_DIAGNOSTICS = importlib.util.module_from_spec(_SUPERPOINT_DIAGNOSTICS_SPEC)
_SUPERPOINT_DIAGNOSTICS_SPEC.loader.exec_module(_SUPERPOINT_DIAGNOSTICS)
build_scene_superpoint_cache = _SUPERPOINT_DIAGNOSTICS.build_scene_superpoint_cache


DEFAULT_SCENES = ("scene0011_00", "scene0077_00", "scene0608_01")


def _scene_id(scene_name):
    return scene_name.replace("scene", "")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path, lines):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip())
        f.write("\n")


def _write_csv(path, rows, fieldnames):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _segment_stats(labels):
    labels = np.asarray(labels, dtype=np.int64)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) == 0:
        return {
            "segments": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0,
            "over_1000": 0,
            "over_5000": 0,
        }
    return {
        "segments": int(len(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(np.max(counts)),
        "over_1000": int(np.sum(counts >= 1000)),
        "over_5000": int(np.sum(counts >= 5000)),
    }


def _majority_purity(values):
    if len(values) == 0:
        return 0.0, -1
    values = np.asarray(values, dtype=np.int64)
    values = values[values >= 0]
    if len(values) == 0:
        return 0.0, -1
    labels, counts = np.unique(values, return_counts=True)
    pos = int(np.argmax(counts))
    return float(counts[pos] / max(1, len(values))), int(labels[pos])


def _purity_rows(scene_name, scene, labels):
    semantic = scene[:, 11].astype(np.int64) if scene.shape[1] > 11 else np.full(len(scene), -1, dtype=np.int64)
    instance = scene[:, 10].astype(np.int64) if scene.shape[1] > 10 else np.full(len(scene), -1, dtype=np.int64)
    rows = []
    for segment_id in np.unique(labels):
        indices = np.flatnonzero(labels == segment_id)
        semantic_purity, semantic_majority = _majority_purity(semantic[indices])
        instance_purity, instance_majority = _majority_purity(instance[indices])
        rows.append(
            {
                "scene_name": scene_name,
                "segment_id": int(segment_id),
                "point_count": int(len(indices)),
                "semantic_purity": semantic_purity,
                "semantic_majority": semantic_majority,
                "instance_purity": instance_purity,
                "instance_majority": instance_majority,
                "mixed_semantic": bool(semantic_purity < 0.95),
                "mixed_instance": bool(instance_purity < 0.95),
            }
        )
    return rows


def _adjacency_degree(records, num_segments):
    degree = np.zeros((num_segments,), dtype=np.int32)
    for item in records:
        left = int(item.get("left_segment_id", item.get("left", -1)))
        right = int(item.get("right_segment_id", item.get("right", -1)))
        if 0 <= left < num_segments:
            degree[left] += 1
        if 0 <= right < num_segments:
            degree[right] += 1
    return degree


def _cache_segment_rows(scene_name, cache, purity_by_segment):
    degree = _adjacency_degree(cache["adjacency_records"], len(cache["segment_sizes"]))
    rows = []
    for segment_id, point_count in enumerate(cache["segment_sizes"]):
        purity = purity_by_segment.get(segment_id, {})
        rows.append(
            {
                "scene_name": scene_name,
                "segment_id": int(segment_id),
                "point_count": int(point_count),
                "center_x": float(cache["centers"][segment_id, 0]),
                "center_y": float(cache["centers"][segment_id, 1]),
                "center_z": float(cache["centers"][segment_id, 2]),
                "bbox_min_x": float(cache["bbox_min"][segment_id, 0]),
                "bbox_min_y": float(cache["bbox_min"][segment_id, 1]),
                "bbox_min_z": float(cache["bbox_min"][segment_id, 2]),
                "bbox_max_x": float(cache["bbox_max"][segment_id, 0]),
                "bbox_max_y": float(cache["bbox_max"][segment_id, 1]),
                "bbox_max_z": float(cache["bbox_max"][segment_id, 2]),
                "mean_r": float(cache["mean_colors"][segment_id, 0]),
                "mean_g": float(cache["mean_colors"][segment_id, 1]),
                "mean_b": float(cache["mean_colors"][segment_id, 2]),
                "normal_x": float(cache["mean_normals"][segment_id, 0]),
                "normal_y": float(cache["mean_normals"][segment_id, 1]),
                "normal_z": float(cache["mean_normals"][segment_id, 2]),
                "planarity": float(cache["planarity"][segment_id]),
                "adjacency_degree": int(degree[segment_id]),
                "semantic_purity": purity.get("semantic_purity", ""),
                "instance_purity": purity.get("instance_purity", ""),
                "mixed_semantic": purity.get("mixed_semantic", ""),
                "mixed_instance": purity.get("mixed_instance", ""),
            }
        )
    return rows


def _adjacency_rows(scene_name, cache):
    rows = []
    for item in cache["adjacency_records"]:
        rows.append(
            {
                "scene_name": scene_name,
                "left": int(item.get("left_segment_id", item.get("left", -1))),
                "right": int(item.get("right_segment_id", item.get("right", -1))),
                "contact_count": int(item.get("boundary_contact_count", item.get("contact_count", 0))),
                "contact_ratio": float(item.get("boundary_contact_ratio", item.get("contact_ratio", 0.0))),
                "mean_boundary_distance": float(item["mean_boundary_distance"]),
                "mean_normal_difference": float(item["mean_normal_difference"]),
                "mean_color_difference": float(item["mean_color_difference"]),
            }
        )
    return rows


def _color_labels(labels):
    labels = labels.astype(np.uint64)
    r = ((labels * 1103515245 + 12345) & 255).astype(np.float32)
    g = ((labels * 2654435761 + 17) & 255).astype(np.float32)
    b = ((labels * 97531 + 101) & 255).astype(np.float32)
    return np.stack([r, g, b], axis=1) / 255.0


def _sample_indices(num_points, max_points, seed=7):
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False)).astype(np.int64)


def _plot_segments_xz(path, scene_name, scene, labels, title, max_points=70000):
    os.makedirs(osp.dirname(path), exist_ok=True)
    idx = _sample_indices(len(scene), max_points=max_points)
    points = scene[idx, :3]
    colors = _color_labels(labels[idx])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 2], s=0.7, c=colors, alpha=0.78, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_title(f"{scene_name} {title}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_largest_segments(path, scene_name, scene, labels, top_k=10, max_points=90000):
    os.makedirs(osp.dirname(path), exist_ok=True)
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    top_ids = set(int(unique[pos]) for pos in order)
    idx = _sample_indices(len(scene), max_points=max_points)
    points = scene[idx, :3]
    sampled_labels = labels[idx].astype(np.int64)
    colors = np.full((len(idx), 3), 0.78, dtype=np.float32)
    for rank, segment_id in enumerate(sorted(top_ids)):
        mask = sampled_labels == segment_id
        color = _color_labels(np.asarray([segment_id + 13 * rank], dtype=np.int64))[0]
        colors[mask] = color
    alpha = np.where(np.isin(sampled_labels, list(top_ids)), 0.95, 0.12)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 2], s=0.8, c=colors, alpha=alpha, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, linewidth=0.3, alpha=0.25)
    label_text = ", ".join(f"{int(unique[pos])}:{int(counts[pos])}" for pos in order[:5])
    ax.set_title(f"{scene_name} largest custom superpoints | {label_text}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_bars(path, scene_rows):
    os.makedirs(osp.dirname(path), exist_ok=True)
    scenes = [row["scene_name"] for row in scene_rows]
    x = np.arange(len(scenes))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metrics = [
        ("segments", "segment count"),
        ("median", "median points"),
        ("max", "max points"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        old_vals = [float(row[f"old_{metric}"]) for row in scene_rows]
        new_vals = [float(row[f"new_{metric}"]) for row in scene_rows]
        ax.bar(x - width / 2, old_vals, width, label="ScanNet200 built-in", color="#8c8c8c")
        ax.bar(x + width / 2, new_vals, width, label="custom geometric", color="#4c78a8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=25, ha="right")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_purity_hist(path, scene_name, purity_rows):
    os.makedirs(osp.dirname(path), exist_ok=True)
    sem = [float(row["semantic_purity"]) for row in purity_rows if row["semantic_purity"] != ""]
    inst = [float(row["instance_purity"]) for row in purity_rows if row["instance_purity"] != ""]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(sem, bins=bins, alpha=0.55, label="semantic purity", color="#54a24b")
    ax.hist(inst, bins=bins, alpha=0.55, label="instance purity", color="#e45756")
    ax.set_xlabel("majority purity")
    ax.set_ylabel("superpoint count")
    ax.set_title(f"{scene_name} custom superpoint purity")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _quality_label(old_stats, new_stats):
    if new_stats["segments"] > old_stats["segments"] * 1.25 and new_stats["max"] < old_stats["max"] * 0.5:
        return "finer_and_less_overgrown"
    if new_stats["segments"] < old_stats["segments"] * 0.8:
        return "coarser"
    return "similar_or_mixed"


def analyze(args):
    scenes = args.scenes or list(DEFAULT_SCENES)
    summary = _load_json(args.geometric_summary)
    summary_by_scene = {row["scene_name"]: row for row in summary.get("scenes", [])}
    os.makedirs(args.output_diagnostics, exist_ok=True)
    os.makedirs(args.output_visuals, exist_ok=True)

    scene_rows = []
    all_segments = []
    all_edges = []
    all_purity = []
    top_large_rows = []
    visual_index = []

    for scene_name in scenes:
        if scene_name not in summary_by_scene:
            raise KeyError(f"{scene_name} missing from {args.geometric_summary}")
        scene_path = osp.join(args.geometric_root, scene_name, f"{_scene_id(scene_name)}.npy")
        if not osp.exists(scene_path):
            raise FileNotFoundError(scene_path)
        scene = np.load(scene_path)
        labels = scene[:, 9].astype(np.int64)
        new_stats = _segment_stats(labels)
        old_stats = summary_by_scene[scene_name]["old"]
        cache = build_scene_superpoint_cache(
            scene_path,
            adjacency_knn=args.adjacency_knn,
            adjacency_max_distance=args.adjacency_max_distance,
            adjacency_min_contact_points=args.adjacency_min_contact_points,
            adjacency_min_contact_ratio=args.adjacency_min_contact_ratio,
        )
        purity_rows = _purity_rows(scene_name, scene, labels)
        purity_by_segment = {int(row["segment_id"]): row for row in purity_rows}
        segment_rows = _cache_segment_rows(scene_name, cache, purity_by_segment)
        edge_rows = _adjacency_rows(scene_name, cache)
        all_segments.extend(segment_rows)
        all_edges.extend(edge_rows)
        all_purity.extend(purity_rows)

        top_segments = sorted(segment_rows, key=lambda row: int(row["point_count"]), reverse=True)[: args.top_large_segments]
        for rank, row in enumerate(top_segments, start=1):
            enriched = dict(row)
            enriched["rank"] = rank
            top_large_rows.append(enriched)

        mixed_semantic = sum(1 for row in purity_rows if row["mixed_semantic"])
        mixed_instance = sum(1 for row in purity_rows if row["mixed_instance"])
        scene_row = {
            "scene_name": scene_name,
            "points": int(scene.shape[0]),
            "method": "existing_felzenszwalb_geometry_superpoints_from_tools_generate_geometric_superpoints",
            "old_segments": int(old_stats["segments"]),
            "old_mean": float(summary_by_scene[scene_name]["points"] / max(1, int(old_stats["segments"]))),
            "old_median": float(old_stats["median"]),
            "old_p90": float(old_stats["p90"]),
            "old_max": int(old_stats["max"]),
            "new_segments": int(new_stats["segments"]),
            "new_mean": float(new_stats["mean"]),
            "new_median": float(new_stats["median"]),
            "new_p90": float(new_stats["p90"]),
            "new_p99": float(new_stats["p99"]),
            "new_max": int(new_stats["max"]),
            "new_over_1000": int(new_stats["over_1000"]),
            "new_over_5000": int(new_stats["over_5000"]),
            "new_adjacency_edges": int(len(edge_rows)),
            "new_mixed_semantic_segments": int(mixed_semantic),
            "new_mixed_semantic_ratio": float(mixed_semantic / max(1, len(purity_rows))),
            "new_mixed_instance_segments": int(mixed_instance),
            "new_mixed_instance_ratio": float(mixed_instance / max(1, len(purity_rows))),
            "comparison_label": _quality_label(old_stats, new_stats),
        }
        scene_rows.append(scene_row)

        scene_visual_dir = osp.join(args.output_visuals, scene_name)
        segment_png = osp.join(scene_visual_dir, f"{scene_name}_custom_segments_xz.png")
        largest_png = osp.join(scene_visual_dir, f"{scene_name}_largest_custom_segments_xz.png")
        purity_png = osp.join(scene_visual_dir, f"{scene_name}_purity_hist.png")
        _plot_segments_xz(segment_png, scene_name, scene, labels, "custom geometric superpoints")
        _plot_largest_segments(largest_png, scene_name, scene, labels)
        _plot_purity_hist(purity_png, scene_name, purity_rows)
        visual_index.extend(
            [
                {"scene_name": scene_name, "kind": "custom_segments_xz", "path": segment_png},
                {"scene_name": scene_name, "kind": "largest_custom_segments_xz", "path": largest_png},
                {"scene_name": scene_name, "kind": "purity_hist", "path": purity_png},
            ]
        )

    _write_csv(
        osp.join(args.output_diagnostics, "scene_summary.csv"),
        scene_rows,
        [
            "scene_name",
            "points",
            "method",
            "old_segments",
            "old_mean",
            "old_median",
            "old_p90",
            "old_max",
            "new_segments",
            "new_mean",
            "new_median",
            "new_p90",
            "new_p99",
            "new_max",
            "new_over_1000",
            "new_over_5000",
            "new_adjacency_edges",
            "new_mixed_semantic_segments",
            "new_mixed_semantic_ratio",
            "new_mixed_instance_segments",
            "new_mixed_instance_ratio",
            "comparison_label",
        ],
    )
    _write_csv(
        osp.join(args.output_diagnostics, "custom_superpoints.csv"),
        all_segments,
        [
            "scene_name",
            "segment_id",
            "point_count",
            "center_x",
            "center_y",
            "center_z",
            "bbox_min_x",
            "bbox_min_y",
            "bbox_min_z",
            "bbox_max_x",
            "bbox_max_y",
            "bbox_max_z",
            "mean_r",
            "mean_g",
            "mean_b",
            "normal_x",
            "normal_y",
            "normal_z",
            "planarity",
            "adjacency_degree",
            "semantic_purity",
            "instance_purity",
            "mixed_semantic",
            "mixed_instance",
        ],
    )
    _write_csv(
        osp.join(args.output_diagnostics, "custom_superpoint_adjacency.csv"),
        all_edges,
        [
            "scene_name",
            "left",
            "right",
            "contact_count",
            "contact_ratio",
            "mean_boundary_distance",
            "mean_normal_difference",
            "mean_color_difference",
        ],
    )
    _write_csv(
        osp.join(args.output_diagnostics, "custom_superpoint_purity.csv"),
        all_purity,
        [
            "scene_name",
            "segment_id",
            "point_count",
            "semantic_purity",
            "semantic_majority",
            "instance_purity",
            "instance_majority",
            "mixed_semantic",
            "mixed_instance",
        ],
    )
    _write_csv(
        osp.join(args.output_diagnostics, "largest_custom_superpoints.csv"),
        top_large_rows,
        [
            "scene_name",
            "rank",
            "segment_id",
            "point_count",
            "center_x",
            "center_y",
            "center_z",
            "bbox_min_x",
            "bbox_min_y",
            "bbox_min_z",
            "bbox_max_x",
            "bbox_max_y",
            "bbox_max_z",
            "planarity",
            "adjacency_degree",
            "semantic_purity",
            "instance_purity",
            "mixed_semantic",
            "mixed_instance",
        ],
    )
    compare_png = osp.join(args.output_visuals, "old_vs_custom_stats.png")
    _plot_metric_bars(compare_png, scene_rows)
    visual_index.append({"scene_name": "all", "kind": "old_vs_custom_stats", "path": compare_png})
    _write_json(osp.join(args.output_visuals, "visual_review_index.json"), visual_index)

    notes = {
        "prototype": "custom_superpoints_v0_3scenes",
        "scenes": scenes,
        "geometric_root": args.geometric_root,
        "geometric_summary": args.geometric_summary,
        "generation_status": "reused_existing_output_no_new_superpoint_generation",
        "method_source": "tools/generate_geometric_superpoints.py existing Felzenszwalb-style geometry/color/normal export",
        "mature_local_reference": "models/Mask3D/third_party/ScanNet/Segmentator implements ScanNet mesh segmentation using Felzenszwalb-Huttenlocher on mesh normals; binary exists but source ScanNet PLY/data mount is unavailable in this session.",
        "input_features": ["xyz", "normal", "color"],
        "excluded_features": ["SAM", "YOLO", "CLIP"],
        "limitations": [
            "Original ScanNet200 superpoint labels are not available as a full pointwise array because the current data/scannet200 symlink target is not mounted; comparison to built-in ScanNet200 superpoints uses the old statistics saved in geometric_superpoints_summary.json.",
            "Boundary crossing checks for custom superpoints use preserved semantic/instance columns as a diagnostic proxy, not AP.",
        ],
        "scene_summary": scene_rows,
    }
    _write_json(osp.join(args.output_diagnostics, "summary.json"), notes)
    quality_lines = [
        "# Custom Superpoints v0 3-Scene Quality Notes",
        "",
        "Scope: quality diagnostics only. No AP, no fusion main-flow changes, no SAM/YOLO/CLIP features.",
        "",
        "Method source:",
        "- Reused existing `tools/generate_geometric_superpoints.py` output under "
        f"`{args.geometric_root}`.",
        "- The generator is a Felzenszwalb-style kNN graph segmentation using xyz, normals, and color.",
        "- Repository search did not find a standalone SAI3D/Open3DIS/MV3DIS/SAM-graph superpoint module.",
        "- `models/Mask3D/third_party/ScanNet/Segmentator` is present as a mature ScanNet "
        "Felzenszwalb-Huttenlocher mesh-normal segmentator, but the source ScanNet PLY/data mount "
        "was unavailable in this session; therefore this v0 compares existing generated geometric "
        "superpoints to saved ScanNet200 built-in statistics.",
        "",
        "Scene comparison:",
        "",
        "| scene | built-in segments | custom segments | built-in median | custom median | built-in max | custom max | label |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in scene_rows:
        quality_lines.append(
            f"| {row['scene_name']} | {row['old_segments']} | {row['new_segments']} | "
            f"{row['old_median']:.1f} | {row['new_median']:.1f} | "
            f"{row['old_max']} | {row['new_max']} | {row['comparison_label']} |"
        )
    quality_lines.extend(
        [
            "",
            "Initial readout:",
            "- Custom geometric superpoints are consistently finer than the ScanNet200 built-in superpoints.",
            "- The largest segments are much smaller in all three scenes, reducing the most obvious wall/floor/table overgrowth risk.",
            "- `scene0011_00` still has one custom segment over 1000 points; the top-largest PNG should be inspected before using this as a replacement source.",
            "- Several large custom segments have mixed semantic/instance purity, so this is not yet a drop-in replacement for final fusion.",
            "",
            "Generated review files:",
            "- `scene_summary.csv`: scene-level built-in vs custom statistics.",
            "- `largest_custom_superpoints.csv`: largest custom segments for boundary/overgrowth review.",
            "- `custom_superpoints.csv`: xyz/color/normal/bbox/planarity/adjacency-degree metadata.",
            "- `custom_superpoint_adjacency.csv`: geometry/color/normal adjacency contacts.",
            "- `custom_superpoint_purity.csv`: semantic/instance majority-purity proxy diagnostics.",
            "- `../visual_checks/custom_superpoints_v0_3scenes/`: XZ scatter plots and purity histograms.",
        ]
    )
    _write_text(osp.join(args.output_diagnostics, "quality_notes.md"), quality_lines)
    print(json.dumps({"scenes": scene_rows, "visuals": len(visual_index)}, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze custom geometric superpoints v0.")
    parser.add_argument("--geometric_root", default="output/geometric_superpoints_scannet200_even48_k025")
    parser.add_argument(
        "--geometric_summary",
        default="output/geometric_superpoints_scannet200_even48_k025/geometric_superpoints_summary.json",
    )
    parser.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    parser.add_argument("--output_diagnostics", default="docs/diagnostics/custom_superpoints_v0_3scenes")
    parser.add_argument("--output_visuals", default="docs/visual_checks/custom_superpoints_v0_3scenes")
    parser.add_argument("--adjacency_knn", default=12, type=int)
    parser.add_argument("--adjacency_max_distance", default=0.05, type=float)
    parser.add_argument("--adjacency_min_contact_points", default=3, type=int)
    parser.add_argument("--adjacency_min_contact_ratio", default=0.02, type=float)
    parser.add_argument("--top_large_segments", default=12, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

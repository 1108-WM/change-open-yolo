import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SCENES = ("scene0011_00", "scene0077_00", "scene0608_01")
DEFAULT_METHODS = (
    "knn_geometry=output/geometric_superpoints_ibsp_v1_geometry_smoke",
    "mesh_normal=output/mesh_normal_python_default_3scenes",
    "mesh_normal_ibsp=output/mesh_normal_ibsp_sam_k070_3scenes",
)


def _scene_id(scene_name):
    return scene_name.replace("scene", "")


def _scene_file(root, scene_name):
    return Path(root) / scene_name / f"{_scene_id(scene_name)}.npy"


def _load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_text(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\n".join(lines).rstrip())
        f.write("\n")


def _parse_methods(items):
    methods = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Method must be name=root, got: {item}")
        name, root = item.split("=", 1)
        name = name.strip()
        root = root.strip()
        if not name or not root:
            raise ValueError(f"Method must be name=root, got: {item}")
        methods.append((name, root))
    return methods


def _segment_stats(labels):
    _, counts = np.unique(labels.astype(np.int64), return_counts=True)
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


def _majority(values):
    values = np.asarray(values, dtype=np.int64)
    values = values[values >= 0]
    if len(values) == 0:
        return 0, -1
    labels, counts = np.unique(values, return_counts=True)
    pos = int(np.argmax(counts))
    return int(counts[pos]), int(labels[pos])


def _purity_stats(labels, target_labels, threshold=0.95):
    rows = []
    total_majority = 0
    mixed_points = 0
    for segment_id in np.unique(labels):
        indices = np.flatnonzero(labels == segment_id)
        majority_count, majority_label = _majority(target_labels[indices])
        purity = float(majority_count / max(1, len(indices)))
        mixed = bool(purity < float(threshold))
        if mixed:
            mixed_points += int(len(indices))
        total_majority += int(majority_count)
        rows.append(
            {
                "segment_id": int(segment_id),
                "point_count": int(len(indices)),
                "majority_label": int(majority_label),
                "purity": purity,
                "mixed": mixed,
            }
        )
    return {
        "rows": rows,
        "mixed_segments": int(sum(1 for row in rows if row["mixed"])),
        "mixed_segment_ratio": float(sum(1 for row in rows if row["mixed"]) / max(1, len(rows))),
        "mixed_points": int(mixed_points),
        "mixed_point_ratio": float(mixed_points / max(1, len(labels))),
        "weighted_purity": float(total_majority / max(1, len(labels))),
        "mean_segment_purity": float(np.mean([row["purity"] for row in rows])) if rows else 0.0,
    }


def _partition_overlap(reference, compared):
    pairs = np.stack((reference.astype(np.int64), compared.astype(np.int64)), axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    reference_ids, reference_child_counts = np.unique(unique_pairs[:, 0], return_counts=True)
    compared_ids, compared_parent_counts = np.unique(unique_pairs[:, 1], return_counts=True)
    reference_sizes = np.bincount(reference.astype(np.int64), minlength=int(reference.max()) + 1)
    compared_sizes = np.bincount(compared.astype(np.int64), minlength=int(compared.max()) + 1)
    split_ids = reference_ids[reference_child_counts > 1]
    merged_ids = compared_ids[compared_parent_counts > 1]
    return {
        "reference_segments": int(len(reference_ids)),
        "compared_segments": int(len(compared_ids)),
        "split_reference_segments": int(len(split_ids)),
        "split_reference_points": int(reference_sizes[split_ids].sum()) if len(split_ids) else 0,
        "merged_compared_segments": int(len(merged_ids)),
        "merged_compared_points": int(compared_sizes[merged_ids].sum()) if len(merged_ids) else 0,
        "pair_count": int(len(unique_pairs)),
        "point_count": int(pair_counts.sum()),
    }


def _color_labels(labels):
    labels = labels.astype(np.uint64)
    red = ((labels * 1103515245 + 12345) & 255).astype(np.float32)
    green = ((labels * 2654435761 + 17) & 255).astype(np.float32)
    blue = ((labels * 97531 + 101) & 255).astype(np.float32)
    return np.stack([red, green, blue], axis=1) / 255.0


def _sample_indices(num_points, max_points, seed=11):
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False)).astype(np.int64)


def _plot_scene_methods(path, scene_name, method_arrays, max_points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(method_arrays.keys())
    base_scene = next(iter(method_arrays.values()))["scene"]
    idx = _sample_indices(len(base_scene), max_points=max_points)
    points = base_scene[idx, :3]
    fig, axes = plt.subplots(1, len(method_names), figsize=(5 * len(method_names), 4.5), constrained_layout=True)
    if len(method_names) == 1:
        axes = [axes]
    for ax, method_name in zip(axes, method_names):
        labels = method_arrays[method_name]["labels"][idx]
        ax.scatter(points[:, 0], points[:, 2], s=0.7, c=_color_labels(labels), alpha=0.78, linewidths=0)
        ax.set_title(f"{scene_name} {method_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_bars(path, scene_rows, metric, ylabel):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scenes = sorted({row["scene_name"] for row in scene_rows})
    methods = list(dict.fromkeys(row["method"] for row in scene_rows))
    x = np.arange(len(scenes))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for index, method in enumerate(methods):
        values = [
            float(next(row[metric] for row in scene_rows if row["scene_name"] == scene and row["method"] == method))
            for scene in scenes
        ]
        ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_method_summary(root):
    path = Path(root) / "geometric_superpoints_summary.json"
    if not path.is_file():
        return {}
    return {row["scene_name"]: row for row in _load_json(path).get("scenes", [])}


def _boundary_fields(summary_scene):
    boundary = summary_scene.get("boundary", {}) if summary_scene else {}
    return {
        "boundary_mode": boundary.get("mode", ""),
        "boundary_graph_type": boundary.get("graph_type", ""),
        "boundary_edges": int(boundary.get("edges", 0) or 0),
        "boundary_observed_edges": int(boundary.get("edges_with_boundary_observation", 0) or 0),
        "boundary_conflict_edges": int(boundary.get("edges_with_conflict", 0) or 0),
        "boundary_pruned_edges": int(boundary.get("pruned_edges", 0) or 0),
        "boundary_pruned_edge_ratio": float(boundary.get("pruned_edge_ratio", 0.0) or 0.0),
        "boundary_mean_conflict_ratio": float(boundary.get("mean_conflict_ratio_for_observed_edges", 0.0) or 0.0),
    }


def analyze(args):
    methods = _parse_methods(args.methods)
    scenes = args.scenes or list(DEFAULT_SCENES)
    output_diagnostics = Path(args.output_diagnostics)
    output_visuals = Path(args.output_visuals)
    output_diagnostics.mkdir(parents=True, exist_ok=True)
    output_visuals.mkdir(parents=True, exist_ok=True)

    summaries = {name: _load_method_summary(root) for name, root in methods}
    scene_rows = []
    purity_rows = []
    top_rows = []
    partition_rows = []
    visual_index = []

    for scene_name in scenes:
        method_arrays = {}
        for method_name, root in methods:
            scene_path = _scene_file(root, scene_name)
            if not scene_path.is_file():
                raise FileNotFoundError(scene_path)
            scene = np.load(scene_path)
            labels = scene[:, 9].astype(np.int64)
            semantic = scene[:, 11].astype(np.int64) if scene.shape[1] > 11 else np.full(len(scene), -1, dtype=np.int64)
            instance = scene[:, 10].astype(np.int64) if scene.shape[1] > 10 else np.full(len(scene), -1, dtype=np.int64)
            method_arrays[method_name] = {"scene": scene, "labels": labels}

            stats = _segment_stats(labels)
            semantic_stats = _purity_stats(labels, semantic, threshold=args.purity_threshold)
            instance_stats = _purity_stats(labels, instance, threshold=args.purity_threshold)
            row = {
                "scene_name": scene_name,
                "method": method_name,
                "root": root,
                "points": int(len(scene)),
                **stats,
                "semantic_weighted_purity": semantic_stats["weighted_purity"],
                "semantic_mean_segment_purity": semantic_stats["mean_segment_purity"],
                "semantic_mixed_segments": semantic_stats["mixed_segments"],
                "semantic_mixed_segment_ratio": semantic_stats["mixed_segment_ratio"],
                "semantic_mixed_point_ratio": semantic_stats["mixed_point_ratio"],
                "instance_weighted_purity": instance_stats["weighted_purity"],
                "instance_mean_segment_purity": instance_stats["mean_segment_purity"],
                "instance_mixed_segments": instance_stats["mixed_segments"],
                "instance_mixed_segment_ratio": instance_stats["mixed_segment_ratio"],
                "instance_mixed_point_ratio": instance_stats["mixed_point_ratio"],
                **_boundary_fields(summaries[method_name].get(scene_name, {})),
            }
            scene_rows.append(row)

            semantic_by_segment = {item["segment_id"]: item for item in semantic_stats["rows"]}
            instance_by_segment = {item["segment_id"]: item for item in instance_stats["rows"]}
            for item in semantic_stats["rows"]:
                inst_item = instance_by_segment[item["segment_id"]]
                purity_rows.append(
                    {
                        "scene_name": scene_name,
                        "method": method_name,
                        "segment_id": item["segment_id"],
                        "point_count": item["point_count"],
                        "semantic_purity": item["purity"],
                        "semantic_majority": item["majority_label"],
                        "semantic_mixed": item["mixed"],
                        "instance_purity": inst_item["purity"],
                        "instance_majority": inst_item["majority_label"],
                        "instance_mixed": inst_item["mixed"],
                    }
                )

            unique, counts = np.unique(labels, return_counts=True)
            order = np.argsort(counts)[::-1][: args.top_segments]
            for rank, pos in enumerate(order, start=1):
                segment_id = int(unique[pos])
                sem_item = semantic_by_segment[segment_id]
                inst_item = instance_by_segment[segment_id]
                top_rows.append(
                    {
                        "scene_name": scene_name,
                        "method": method_name,
                        "rank": rank,
                        "segment_id": segment_id,
                        "point_count": int(counts[pos]),
                        "semantic_purity": sem_item["purity"],
                        "semantic_majority": sem_item["majority_label"],
                        "instance_purity": inst_item["purity"],
                        "instance_majority": inst_item["majority_label"],
                    }
                )

        scene_png = output_visuals / scene_name / f"{scene_name}_methods_xz.png"
        _plot_scene_methods(scene_png, scene_name, method_arrays, args.max_visual_points)
        visual_index.append({"scene_name": scene_name, "kind": "methods_xz", "path": str(scene_png)})

        baseline_name = methods[0][0]
        baseline_labels = method_arrays[baseline_name]["labels"]
        for method_name, _ in methods[1:]:
            compared_labels = method_arrays[method_name]["labels"]
            if baseline_labels.shape != compared_labels.shape:
                raise ValueError(f"Point count mismatch for {scene_name}: {baseline_name} vs {method_name}")
            partition_rows.append(
                {
                    "scene_name": scene_name,
                    "reference_method": baseline_name,
                    "compared_method": method_name,
                    **_partition_overlap(baseline_labels, compared_labels),
                }
            )

    _write_csv(
        output_diagnostics / "scene_method_summary.csv",
        scene_rows,
        [
            "scene_name",
            "method",
            "root",
            "points",
            "segments",
            "mean",
            "median",
            "p90",
            "p99",
            "max",
            "over_1000",
            "over_5000",
            "semantic_weighted_purity",
            "semantic_mean_segment_purity",
            "semantic_mixed_segments",
            "semantic_mixed_segment_ratio",
            "semantic_mixed_point_ratio",
            "instance_weighted_purity",
            "instance_mean_segment_purity",
            "instance_mixed_segments",
            "instance_mixed_segment_ratio",
            "instance_mixed_point_ratio",
            "boundary_mode",
            "boundary_graph_type",
            "boundary_edges",
            "boundary_observed_edges",
            "boundary_conflict_edges",
            "boundary_pruned_edges",
            "boundary_pruned_edge_ratio",
            "boundary_mean_conflict_ratio",
        ],
    )
    _write_csv(
        output_diagnostics / "segment_purity.csv",
        purity_rows,
        [
            "scene_name",
            "method",
            "segment_id",
            "point_count",
            "semantic_purity",
            "semantic_majority",
            "semantic_mixed",
            "instance_purity",
            "instance_majority",
            "instance_mixed",
        ],
    )
    _write_csv(
        output_diagnostics / "largest_segments.csv",
        top_rows,
        [
            "scene_name",
            "method",
            "rank",
            "segment_id",
            "point_count",
            "semantic_purity",
            "semantic_majority",
            "instance_purity",
            "instance_majority",
        ],
    )
    _write_csv(
        output_diagnostics / "partition_overlap_vs_first_method.csv",
        partition_rows,
        [
            "scene_name",
            "reference_method",
            "compared_method",
            "reference_segments",
            "compared_segments",
            "split_reference_segments",
            "split_reference_points",
            "merged_compared_segments",
            "merged_compared_points",
            "pair_count",
            "point_count",
        ],
    )

    for metric, label in [
        ("segments", "superpoint count"),
        ("max", "largest superpoint points"),
        ("instance_weighted_purity", "point-weighted instance purity"),
        ("semantic_weighted_purity", "point-weighted semantic purity"),
    ]:
        png = output_visuals / f"{metric}_by_scene.png"
        _plot_metric_bars(png, scene_rows, metric, label)
        visual_index.append({"scene_name": "all", "kind": metric, "path": str(png)})
    _write_json(output_visuals / "visual_review_index.json", visual_index)

    summary = {
        "scope": "diagnostic_only_no_ap_no_fusion_change",
        "scenes": scenes,
        "methods": [{"name": name, "root": root} for name, root in methods],
        "purity_threshold": args.purity_threshold,
        "scene_method_summary": scene_rows,
        "partition_overlap_vs_first_method": partition_rows,
    }
    _write_json(output_diagnostics / "summary.json", summary)
    _write_quality_notes(output_diagnostics / "quality_notes.md", scene_rows, partition_rows, methods)
    print(json.dumps({"scene_method_rows": len(scene_rows), "visuals": len(visual_index)}, indent=2))


def _format_float(value, digits=4):
    return f"{float(value):.{digits}f}"


def _write_quality_notes(path, scene_rows, partition_rows, methods):
    lines = [
        "# Superpoint Method Quality Comparison",
        "",
        "Scope: diagnostic only. No AP, no fusion main-flow change, no GT-derived inference input.",
        "",
        "Compared methods:",
    ]
    for name, root in methods:
        lines.append(f"- `{name}`: `{root}`")
    lines.extend(
        [
            "",
            "Scene summary:",
            "",
            "| scene | method | segments | median | max | sem weighted purity | inst weighted purity | sem mixed seg ratio | inst mixed seg ratio | pruned edges |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scene_rows:
        lines.append(
            f"| `{row['scene_name']}` | `{row['method']}` | {row['segments']} | "
            f"{_format_float(row['median'], 1)} | {row['max']} | "
            f"{_format_float(row['semantic_weighted_purity'])} | "
            f"{_format_float(row['instance_weighted_purity'])} | "
            f"{_format_float(row['semantic_mixed_segment_ratio'])} | "
            f"{_format_float(row['instance_mixed_segment_ratio'])} | "
            f"{row['boundary_pruned_edges']} |"
        )
    lines.extend(
        [
            "",
            "Partition change relative to the first method:",
            "",
            "| scene | reference | compared | split ref segments | split ref points | merged compared segments | merged compared points |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in partition_rows:
        lines.append(
            f"| `{row['scene_name']}` | `{row['reference_method']}` | `{row['compared_method']}` | "
            f"{row['split_reference_segments']} | {row['split_reference_points']} | "
            f"{row['merged_compared_segments']} | {row['merged_compared_points']} |"
        )
    lines.extend(
        [
            "",
            "Readout:",
            "- `mesh_normal` closely matches ScanNet Segmentator-style built-in superpoints in segment count and largest segment size; it is a mature geometry baseline rather than a finer replacement.",
            "- The existing `mesh_normal_ibsp` run prunes very few graph edges, so its quality proxy stays almost identical to `mesh_normal` on these three scenes.",
            "- The old kNN geometry run creates many more, smaller superpoints and removes the huge built-in regions, but its mixed semantic/instance segment ratios remain non-trivial.",
            "- This comparison still uses GT semantic/instance columns only as offline purity proxies; it should not be interpreted as an AP result or an argument to connect IBSp to the main fusion path.",
            "",
            "Generated files:",
            "- `scene_method_summary.csv`",
            "- `segment_purity.csv`",
            "- `largest_segments.csv`",
            "- `partition_overlap_vs_first_method.csv`",
            "- `../visual_checks/superpoint_method_quality_3scenes/`",
        ]
    )
    _write_text(path, lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare existing superpoint method exports with purity proxies.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    parser.add_argument("--output_diagnostics", default="docs/diagnostics/superpoint_method_quality_3scenes")
    parser.add_argument("--output_visuals", default="docs/visual_checks/superpoint_method_quality_3scenes")
    parser.add_argument("--purity_threshold", default=0.95, type=float)
    parser.add_argument("--top_segments", default=12, type=int)
    parser.add_argument("--max_visual_points", default=70000, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

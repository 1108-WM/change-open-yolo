import argparse
import json
import os
import os.path as osp
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POINT_SET_SPECS = [
    ("point", "seed_points_path", "#4c78a8"),
    ("core_only", "superpoint_core_only_seed_points_path", "#54a24b"),
    ("core_boundary", "superpoint_candidate_seed_points_path", "#f58518"),
    ("largest_cc", "superpoint_candidate_largest_cc_seed_points_path", "#b279a2"),
]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _slug(text):
    keep = []
    for char in str(text):
        if char.isalnum():
            keep.append(char)
        elif char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_point_indices(path):
    if not path or not osp.exists(path):
        return np.zeros((0,), dtype=np.int64)
    data = np.load(path)
    if "point_indices" not in data:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(data["point_indices"], dtype=np.int64)


def _load_scene_payload(candidates_dir, scene_name):
    path = osp.join(candidates_dir, scene_name, "backprojection_candidates.json")
    payload = _load_json(path)
    candidates = {
        _safe_int(candidate.get("candidate_id", -1), -1): candidate
        for candidate in payload.get("candidates", [])
    }
    return payload, candidates


def _load_scene_points(scene_payload):
    scene_diag = scene_payload.get("superpoint_diagnostics", {})
    processed_scene_path = scene_diag.get("processed_scene_path")
    if not processed_scene_path or not osp.exists(processed_scene_path):
        raise FileNotFoundError(f"processed_scene_path not found: {processed_scene_path}")
    arr = np.load(processed_scene_path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"processed scene must be NxC with xyz columns: {processed_scene_path}")
    return np.asarray(arr[:, :3], dtype=np.float32), processed_scene_path


def _candidate_point_sets(candidate):
    point_sets = {}
    for name, path_key, _ in POINT_SET_SPECS:
        point_sets[name] = _load_point_indices(candidate.get(path_key))
    return point_sets


def _points_for_indices(points_xyz, indices):
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    valid = indices[(indices >= 0) & (indices < len(points_xyz))]
    return points_xyz[valid]


def _axis_limits(point_arrays, dims):
    non_empty = [points[:, dims] for points in point_arrays if len(points) > 0]
    if not non_empty:
        return (0.0, 1.0), (0.0, 1.0)
    stacked = np.concatenate(non_empty, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = span * 0.08
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def _scatter(ax, points, dims, color, label, size=2.0, alpha=0.85):
    if len(points) == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{label} (0)")
        return
    ax.scatter(points[:, dims[0]], points[:, dims[1]], s=size, c=color, alpha=alpha, linewidths=0)
    ax.set_title(f"{label} ({len(points)})")


def _format_metrics(row):
    return (
        f"ratio={_safe_float(row.get('largest_cc_to_point_ratio')):.2f}, "
        f"conflict={_safe_float(row.get('conflict_overlap')):.2f}, "
        f"iou={_safe_float(row.get('existing_mask_iou')):.2f}"
    )


def _plot_four_sets(output_path, row, candidate, points_xyz, point_sets):
    arrays = {
        name: _points_for_indices(points_xyz, indices)
        for name, indices in point_sets.items()
    }
    dims = (0, 2)
    xlim, ylim = _axis_limits(arrays.values(), dims)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (name, _, color) in zip(axes.ravel(), POINT_SET_SPECS):
        _scatter(ax, arrays[name], dims, color, name)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.suptitle(
        f"{row.get('scene_name')} candidate{_safe_int(row.get('candidate_id')):04d} "
        f"{row.get('class_name')} | {row.get('recommended_action')} | {_format_metrics(row)}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_overlay(output_path, row, points_xyz, point_sets):
    core_boundary = set(int(item) for item in point_sets.get("core_boundary", []))
    largest_cc = set(int(item) for item in point_sets.get("largest_cc", []))
    removed = np.asarray(sorted(core_boundary - largest_cc), dtype=np.int64)
    kept = np.asarray(sorted(largest_cc), dtype=np.int64)
    kept_points = _points_for_indices(points_xyz, kept)
    removed_points = _points_for_indices(points_xyz, removed)
    views = [
        ("xy", (0, 1)),
        ("xz", (0, 2)),
        ("yz", (1, 2)),
    ]
    xlim_ylim = {
        name: _axis_limits([kept_points, removed_points], dims)
        for name, dims in views
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (name, dims) in zip(axes, views):
        if len(kept_points) > 0:
            ax.scatter(
                kept_points[:, dims[0]],
                kept_points[:, dims[1]],
                s=1.8,
                c="#8c8c8c",
                alpha=0.75,
                linewidths=0,
                label="kept largest_cc",
            )
        if len(removed_points) > 0:
            ax.scatter(
                removed_points[:, dims[0]],
                removed_points[:, dims[1]],
                s=3.0,
                c="#d62728",
                alpha=0.9,
                linewidths=0,
                label="removed",
            )
        xlim, ylim = xlim_ylim[name]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{name} kept={len(kept_points)} removed={len(removed_points)}")
        ax.grid(True, linewidth=0.3, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=8)
    fig.suptitle(
        f"{row.get('scene_name')} candidate{_safe_int(row.get('candidate_id')):04d} "
        f"{row.get('class_name')} largest_cc cleanup | {_format_metrics(row)}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _iter_review_items(review_lists):
    for group_name, rows in review_lists.items():
        for row in rows:
            yield group_name, row


def visualize(args):
    review_lists = _load_json(args.review_lists_json)
    os.makedirs(args.output_dir, exist_ok=True)
    scene_cache = {}
    visual_index = []
    group_counts = defaultdict(int)

    for group_name, row in _iter_review_items(review_lists):
        scene_name = row.get("scene_name")
        candidate_id = _safe_int(row.get("candidate_id", -1), -1)
        if not scene_name or candidate_id < 0:
            continue
        if scene_name not in scene_cache:
            payload, candidates = _load_scene_payload(args.candidates_dir, scene_name)
            points_xyz, processed_scene_path = _load_scene_points(payload)
            scene_cache[scene_name] = (payload, candidates, points_xyz, processed_scene_path)
        _, candidates, points_xyz, processed_scene_path = scene_cache[scene_name]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            visual_index.append(
                {
                    "group": group_name,
                    "scene_name": scene_name,
                    "candidate_id": candidate_id,
                    "missing": True,
                }
            )
            continue

        group_dir = osp.join(args.output_dir, _slug(group_name))
        os.makedirs(group_dir, exist_ok=True)
        group_counts[group_name] += 1
        prefix = (
            f"{group_counts[group_name]:02d}_{scene_name}_candidate{candidate_id:04d}_"
            f"{_slug(row.get('class_name'))}"
        )
        point_sets = _candidate_point_sets(candidate)
        four_sets_path = osp.join(group_dir, f"{prefix}_four_sets_xz.png")
        overlay_path = osp.join(group_dir, f"{prefix}_largest_cc_overlay.png")
        _plot_four_sets(four_sets_path, row, candidate, points_xyz, point_sets)
        _plot_overlay(overlay_path, row, points_xyz, point_sets)
        visual_index.append(
            {
                "group": group_name,
                "scene_name": scene_name,
                "candidate_id": candidate_id,
                "class_name": row.get("class_name"),
                "recommended_action": row.get("recommended_action"),
                "action_reason": row.get("action_reason"),
                "metrics": {
                    "largest_cc_to_point_ratio": _safe_float(row.get("largest_cc_to_point_ratio")),
                    "largest_cc_covered_by_point_ratio": _safe_float(
                        row.get("largest_cc_covered_by_point_ratio")
                    ),
                    "point_covered_by_largest_cc_ratio": _safe_float(
                        row.get("point_covered_by_largest_cc_ratio")
                    ),
                    "existing_mask_iou": _safe_float(row.get("existing_mask_iou")),
                    "existing_mask_seed_coverage": _safe_float(
                        row.get("existing_mask_seed_coverage")
                    ),
                    "conflict_overlap": _safe_float(row.get("conflict_overlap")),
                },
                "processed_scene_path": processed_scene_path,
                "four_sets_xz_path": four_sets_path,
                "largest_cc_overlay_path": overlay_path,
                "point_counts": {name: int(len(indices)) for name, indices in point_sets.items()},
                "removed_by_largest_cc_count": int(
                    len(set(point_sets["core_boundary"].tolist()) - set(point_sets["largest_cc"].tolist()))
                ),
            }
        )

    index_path = osp.join(args.output_dir, "visual_review_index.json")
    with open(index_path, "w") as f:
        json.dump(
            {
                "candidates_dir": args.candidates_dir,
                "review_lists_json": args.review_lists_json,
                "group_counts": dict(group_counts),
                "visuals": visual_index,
            },
            f,
            indent=2,
        )
    print(
        "[SUPERPOINT_VIS] "
        f"visuals={len(visual_index)} output_dir={args.output_dir} index={index_path}"
    )
    print(f"[SUPERPOINT_VIS] group_counts={json.dumps(dict(group_counts), sort_keys=True)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render visual review images for superpoint action diagnostics.")
    parser.add_argument("--candidates_dir", required=True)
    parser.add_argument("--review_lists_json", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())

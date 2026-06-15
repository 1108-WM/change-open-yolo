import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def _scene_processed_path(root, scene_name):
    scene_id = scene_name.replace("scene", "")
    return osp.join(root, scene_name, f"{scene_id}.npy")


def _read_scene_names(args):
    if args.scene_names:
        return [item.strip() for item in args.scene_names.split(",") if item.strip()]
    if args.scene_split:
        with open(args.scene_split) as f:
            return [line.strip() for line in f if line.strip()]
    root = Path(args.input_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("scene"))


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.uint8)
        self.component_size = np.ones(size, dtype=np.int32)
        self.internal = np.zeros(size, dtype=np.float32)

    def find(self, index):
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(self, left, right, edge_weight):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]
        self.internal[left_root] = edge_weight
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


def _normalize_rgb(rgb):
    rgb = rgb.astype(np.float32)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _normalize_normals(normals):
    normals = normals.astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-6)


def _build_knn_edges(points, colors, normals, knn, spatial_weight, normal_weight, color_weight):
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=int(knn) + 1, workers=-1)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    left = np.repeat(np.arange(points.shape[0], dtype=np.int32), int(knn))
    right = neighbors.reshape(-1).astype(np.int32)
    dist = distances.reshape(-1).astype(np.float32)
    valid = (right >= 0) & (right < points.shape[0]) & (left < right)
    left = left[valid]
    right = right[valid]
    dist = dist[valid]

    scale = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    spatial_term = dist / max(scale, 1e-6)
    normal_dot = np.sum(normals[left] * normals[right], axis=1)
    normal_term = 1.0 - np.abs(np.clip(normal_dot, -1.0, 1.0))
    color_term = np.linalg.norm(colors[left] - colors[right], axis=1) / np.sqrt(3.0)
    weights = (
        float(spatial_weight) * spatial_term
        + float(normal_weight) * normal_term
        + float(color_weight) * color_term
    ).astype(np.float32)
    return left, right, weights


def _felzenszwalb_segments(num_points, left, right, weights, merge_k):
    order = np.argsort(weights, kind="mergesort")
    uf = UnionFind(num_points)
    merge_k = float(merge_k)
    for edge_index in order:
        lidx = int(left[edge_index])
        ridx = int(right[edge_index])
        weight = float(weights[edge_index])
        lroot = uf.find(lidx)
        rroot = uf.find(ridx)
        if lroot == rroot:
            continue
        left_threshold = float(uf.internal[lroot]) + merge_k / float(uf.component_size[lroot])
        right_threshold = float(uf.internal[rroot]) + merge_k / float(uf.component_size[rroot])
        if weight <= min(left_threshold, right_threshold):
            uf.union(lroot, rroot, weight)
    return uf


def _merge_small_components(uf, left, right, weights, min_size):
    min_size = int(max(1, min_size))
    if min_size <= 1:
        return
    order = np.argsort(weights, kind="mergesort")
    for edge_index in order:
        lroot = uf.find(int(left[edge_index]))
        rroot = uf.find(int(right[edge_index]))
        if lroot == rroot:
            continue
        if uf.component_size[lroot] < min_size or uf.component_size[rroot] < min_size:
            uf.union(lroot, rroot, float(weights[edge_index]))


def _contiguous_labels(uf, num_points):
    roots = np.asarray([uf.find(index) for index in range(num_points)], dtype=np.int32)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32), unique_roots


def generate_superpoints(scene_array, args):
    points = scene_array[:, :3].astype(np.float32)
    colors = _normalize_rgb(scene_array[:, 3:6])
    normals = _normalize_normals(scene_array[:, 6:9])
    left, right, weights = _build_knn_edges(
        points,
        colors,
        normals,
        args.knn,
        args.spatial_weight,
        args.normal_weight,
        args.color_weight,
    )
    uf = _felzenszwalb_segments(points.shape[0], left, right, weights, args.merge_k)
    _merge_small_components(uf, left, right, weights, args.min_size)
    labels, _ = _contiguous_labels(uf, points.shape[0])
    return labels


def _stats(labels):
    _, counts = np.unique(labels, return_counts=True)
    return {
        "segments": int(len(counts)),
        "min": int(counts.min(initial=0)),
        "p10": float(np.percentile(counts, 10)) if len(counts) else 0.0,
        "median": float(np.median(counts)) if len(counts) else 0.0,
        "p90": float(np.percentile(counts, 90)) if len(counts) else 0.0,
        "max": int(counts.max(initial=0)),
    }


def process_scene(scene_name, args):
    input_path = _scene_processed_path(args.input_root, scene_name)
    if not osp.exists(input_path):
        raise FileNotFoundError(input_path)
    scene = np.load(input_path)
    labels = generate_superpoints(scene, args)
    output_scene = scene.copy()
    output_scene[:, 9] = labels.astype(np.float32)

    output_dir = osp.join(args.output_root, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = _scene_processed_path(args.output_root, scene_name)
    if osp.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    np.save(output_path, output_scene)
    return {
        "scene_name": scene_name,
        "input_path": input_path,
        "output_path": output_path,
        "points": int(scene.shape[0]),
        "old": _stats(scene[:, 9].astype(np.int64)),
        "new": _stats(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="./data/scannet200")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)
    parser.add_argument("--knn", default=10, type=int)
    parser.add_argument("--merge_k", default=0.25, type=float)
    parser.add_argument("--min_size", default=20, type=int)
    parser.add_argument("--spatial_weight", default=0.15, type=float)
    parser.add_argument("--normal_weight", default=1.0, type=float)
    parser.add_argument("--color_weight", default=0.25, type=float)
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    scene_names = _read_scene_names(args)
    if args.max_scenes is not None:
        scene_names = scene_names[: int(args.max_scenes)]
    os.makedirs(args.output_root, exist_ok=True)

    summaries = []
    for scene_name in tqdm(scene_names):
        summary = process_scene(scene_name, args)
        summaries.append(summary)
        old_stats = summary["old"]
        new_stats = summary["new"]
        print(
            f"{scene_name}: points={summary['points']} "
            f"old_segments={old_stats['segments']} old_median={old_stats['median']:.1f} "
            f"new_segments={new_stats['segments']} new_median={new_stats['median']:.1f} "
            f"new_p90={new_stats['p90']:.1f} new_max={new_stats['max']}"
        )

    summary_path = osp.join(args.output_root, "geometric_superpoints_summary.json")
    import json

    with open(summary_path, "w") as f:
        json.dump({"params": vars(args), "scenes": summaries}, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import os.path as osp
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from plyfile import PlyData
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


def _mesh_path(scene_dir, scene_name, args):
    return Path(scene_dir) / args.mesh_name_template.format(scene_name=scene_name)


def _build_mesh_normal_edges(points, scene_dir, scene_name, args):
    """Match ScanNet Segmentator's mesh-normal graph before graph segmentation."""
    mesh_path = _mesh_path(scene_dir, scene_name, args)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing mesh required by mesh_normal graph: {mesh_path}")
    ply = PlyData.read(str(mesh_path))
    if "vertex" not in ply or "face" not in ply:
        raise ValueError(f"Mesh must contain vertex and face elements: {mesh_path}")
    vertices = ply["vertex"].data
    mesh_points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    if mesh_points.shape != points.shape or not np.allclose(
        mesh_points, points, atol=float(args.mesh_alignment_tolerance), rtol=0.0
    ):
        raise ValueError(
            f"Mesh vertices do not match processed points for {scene_name}: "
            f"mesh={mesh_points.shape}, points={points.shape}"
        )
    faces = ply["face"].data
    if "vertex_indices" not in faces.dtype.names:
        raise ValueError(f"Mesh face vertex_indices missing: {mesh_path}")
    triangles = np.asarray(faces["vertex_indices"].tolist(), dtype=np.int32)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Only triangular meshes are supported: {mesh_path}")

    # This reproduces Segmentator: normalized face normals, then running mean at vertices.
    first = points[triangles[:, 0]]
    second = points[triangles[:, 1]]
    third = points[triangles[:, 2]]
    face_normals = np.cross(second - first, third - first)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    vertex_normals = np.zeros_like(points, dtype=np.float32)
    counts = np.zeros(points.shape[0], dtype=np.int32)
    # Preserve the C++ loop order: each face contributes its three vertices before
    # the next face. This matters for deterministic ordering of equal-weight edges.
    flat_indices = triangles.reshape(-1)
    np.add.at(vertex_normals, flat_indices, np.repeat(face_normals, 3, axis=0))
    np.add.at(counts, flat_indices, 1)
    vertex_normals /= np.maximum(counts[:, None], 1)

    left = np.empty(triangles.shape[0] * 3, dtype=np.int32)
    right = np.empty_like(left)
    left[0::3], right[0::3] = triangles[:, 0], triangles[:, 1]
    left[1::3], right[1::3] = triangles[:, 0], triangles[:, 2]
    left[2::3], right[2::3] = triangles[:, 2], triangles[:, 1]
    edge_direction = points[right] - points[left]
    edge_direction /= np.maximum(np.linalg.norm(edge_direction, axis=1, keepdims=True), 1e-12)
    normal_dot = np.sum(vertex_normals[left] * vertex_normals[right], axis=1)
    directional_dot = np.sum(vertex_normals[right] * edge_direction, axis=1)
    weights = 1.0 - normal_dot
    convex = directional_dot > 0.0
    weights[convex] *= weights[convex]
    return left, right, weights.astype(np.float32)


def _build_graph(points, colors, normals, scene_dir, scene_name, args):
    if args.graph_type == "knn":
        return _build_knn_edges(
            points,
            colors,
            normals,
            args.knn,
            args.spatial_weight,
            args.normal_weight,
            args.color_weight,
        )
    if args.graph_type == "mesh_normal":
        return _build_mesh_normal_edges(points, scene_dir, scene_name, args)
    raise ValueError(f"Unsupported graph type: {args.graph_type}")


def _read_intrinsics(scene_dir):
    matrix = np.loadtxt(osp.join(scene_dir, "intrinsics.txt"), dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 intrinsic matrix in {scene_dir}")
    return matrix[:3, :3]


def _frame_names(scene_dir):
    color_dir = Path(scene_dir) / "color"
    if not color_dir.is_dir():
        raise FileNotFoundError(f"Missing color directory: {color_dir}")
    names = [path.stem for path in color_dir.glob("*.*") if path.is_file()]
    return sorted(names, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def _sample_frame_names(frame_names, frame_stride, max_frames):
    sampled = frame_names[:: max(1, int(frame_stride))]
    if max_frames is not None:
        sampled = sampled[: int(max_frames)]
    return sampled


def _boundary_frame_names(scene_dir, scene_name, args):
    """Prefer the actually exported label-map frames when a subdirectory is used."""
    subdir = getattr(args, "boundary_mask_subdir", None)
    if subdir:
        label_dir = Path(args.boundary_mask_root) / scene_name / subdir
        extension = str(args.boundary_mask_extension)
        if label_dir.is_dir():
            names = [path.stem for path in label_dir.glob(f"*{extension}") if path.is_file()]
            if names:
                names.sort(key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
                return _sample_frame_names(names, args.boundary_frame_stride, args.boundary_max_frames)
    return _sample_frame_names(
        _frame_names(scene_dir), args.boundary_frame_stride, args.boundary_max_frames
    )


def _mask_path(mask_root, scene_name, frame_name, extension, subdir=None):
    path = Path(mask_root) / scene_name
    if subdir:
        path = path / subdir
    return path / f"{frame_name}{extension}"


def _project_visible_labels(
    points,
    camera_to_world,
    intrinsics,
    label_image,
    visibility_tolerance,
    unknown_label=0,
    native_shape=None,
):
    """Project points into one label image and retain only front-most points."""
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera_points[:, 2]
    valid_depth = np.isfinite(depth) & (depth > 1e-4)

    pixels = camera_points @ intrinsics.T
    x = pixels[:, 0] / np.maximum(pixels[:, 2], 1e-6)
    y = pixels[:, 1] / np.maximum(pixels[:, 2], 1e-6)

    height, width = label_image.shape[:2]
    if native_shape is None:
        native_height = max(1.0, 2.0 * float(intrinsics[1, 2]))
        native_width = max(1.0, 2.0 * float(intrinsics[0, 2]))
    else:
        native_height, native_width = native_shape[:2]
    x = np.rint(x * (width / native_width)).astype(np.int64)
    y = np.rint(y * (height / native_height)).astype(np.int64)
    in_image = valid_depth & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    flat_pixels = y[in_image] * width + x[in_image]
    visible_depth = depth[in_image]
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, flat_pixels, visible_depth)
    frontmost = visible_depth <= z_buffer[flat_pixels] + float(visibility_tolerance)

    labels = np.full(points.shape[0], unknown_label, dtype=label_image.dtype)
    selected_points = np.flatnonzero(in_image)[frontmost]
    labels[selected_points] = label_image[y[selected_points], x[selected_points]]
    return labels


def _boundary_keep_mask(
    points,
    left,
    right,
    scene_dir,
    scene_name,
    args,
):
    """Return the geometry-graph edges retained after IBSp boundary pruning."""
    frame_names = _boundary_frame_names(scene_dir, scene_name, args)
    intrinsics = _read_intrinsics(scene_dir)
    observed = np.zeros(left.shape[0], dtype=np.uint16)
    conflicts = np.zeros(left.shape[0], dtype=np.uint16)
    used_frames = 0
    missing_masks = []
    unknown = int(args.boundary_unknown_label)

    for frame_name in frame_names:
        path = _mask_path(
            args.boundary_mask_root,
            scene_name,
            frame_name,
            args.boundary_mask_extension,
            getattr(args, "boundary_mask_subdir", None),
        )
        if not path.is_file():
            missing_masks.append(str(path))
            continue
        pose_path = Path(scene_dir) / "poses" / f"{frame_name}.txt"
        if not pose_path.is_file():
            continue
        label_image = imageio.imread(path)
        if label_image.ndim != 2:
            raise ValueError(
                f"Boundary label map must be single-channel, got {label_image.shape}: {path}"
            )
        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            continue
        color_path = next((Path(scene_dir) / "color").glob(f"{frame_name}.*"), None)
        try:
            native_shape = imageio.imread(color_path).shape[:2] if color_path is not None else None
        except Exception:
            native_shape = None
        labels = _project_visible_labels(
            points,
            pose,
            intrinsics,
            label_image,
            args.boundary_visibility_tolerance,
            unknown_label=unknown,
            native_shape=native_shape,
        )
        left_labels = labels[left]
        right_labels = labels[right]
        if args.boundary_cut_against_background:
            known = (left_labels != unknown) | (right_labels != unknown)
        else:
            known = (left_labels != unknown) & (right_labels != unknown)
        if not np.any(known):
            used_frames += 1
            continue
        observed[known] += 1
        conflicts[known & (left_labels != right_labels)] += 1
        used_frames += 1

    if not used_frames:
        preview = "; ".join(missing_masks[:3])
        raise FileNotFoundError(
            "No usable boundary label maps were found for "
            f"{scene_name}. Expected {args.boundary_mask_root}/<scene>/"
            f"{getattr(args, 'boundary_mask_subdir', '') + '/' if getattr(args, 'boundary_mask_subdir', None) else ''}<frame>"
            f"{args.boundary_mask_extension}. Examples: {preview}"
        )

    agreement = conflicts.astype(np.float32) / np.maximum(observed, 1)
    cut = (observed >= int(args.boundary_min_observations)) & (
        agreement >= float(args.boundary_min_conflict_ratio)
    )
    stats = {
        "mode": "ibsp",
        "available_frames": len(frame_names),
        "used_frames": used_frames,
        "missing_mask_frames": len(missing_masks),
        "edges": int(left.shape[0]),
        "edges_with_boundary_observation": int((observed > 0).sum()),
        "edges_with_conflict": int((conflicts > 0).sum()),
        "pruned_edges": int(cut.sum()),
        "pruned_edge_ratio": float(cut.mean()) if len(cut) else 0.0,
        "mean_conflict_ratio_for_observed_edges": float(agreement[observed > 0].mean())
        if np.any(observed > 0)
        else 0.0,
    }
    return ~cut, stats


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
    if getattr(args, "graph_type", "knn") != "knn":
        raise RuntimeError("mesh_normal needs scene context; call generate_scene_superpoints.")
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
    if getattr(args, "boundary_mask_root", None):
        raise RuntimeError("Boundary constraints need scene context; call generate_scene_superpoints.")
    uf = _felzenszwalb_segments(points.shape[0], left, right, weights, args.merge_k)
    _merge_small_components(uf, left, right, weights, args.min_size)
    labels, _ = _contiguous_labels(uf, points.shape[0])
    return labels


def generate_scene_superpoints(scene_array, scene_dir, scene_name, args):
    points = scene_array[:, :3].astype(np.float32)
    colors = _normalize_rgb(scene_array[:, 3:6])
    normals = _normalize_normals(scene_array[:, 6:9])
    left, right, weights = _build_graph(
        points, colors, normals, scene_dir, scene_name, args
    )
    if args.boundary_mask_root:
        keep, boundary_stats = _boundary_keep_mask(
            points, left, right, scene_dir, scene_name, args
        )
        left, right, weights = left[keep], right[keep], weights[keep]
    else:
        boundary_stats = {
            "mode": "geometry",
            "edges": int(left.shape[0]),
            "pruned_edges": 0,
        }
    boundary_stats["graph_type"] = args.graph_type
    uf = _felzenszwalb_segments(points.shape[0], left, right, weights, args.merge_k)
    _merge_small_components(uf, left, right, weights, args.min_size)
    labels, _ = _contiguous_labels(uf, points.shape[0])
    return labels, boundary_stats


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
    labels, boundary_stats = generate_scene_superpoints(scene, osp.join(args.input_root, scene_name), scene_name, args)
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
        "boundary": boundary_stats,
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
    parser.add_argument(
        "--graph_type",
        choices=("knn", "mesh_normal"),
        default="knn",
        help="knn reproduces the existing xyz/color/normal graph; mesh_normal uses ScanNet mesh topology and Segmentator-style normal weights.",
    )
    parser.add_argument(
        "--mesh_name_template",
        default="{scene_name}_vh_clean_2.ply",
        help="Mesh filename within each scene when --graph_type mesh_normal is selected.",
    )
    parser.add_argument("--mesh_alignment_tolerance", default=1e-6, type=float)
    parser.add_argument(
        "--boundary_mask_root",
        default=None,
        help=(
            "Optional root of per-view uint label maps: <root>/<scene>/<frame>.png. "
            "When set, use OVSeg3R-style 2D instance-boundary graph pruning."
        ),
    )
    parser.add_argument("--boundary_mask_extension", default=".png")
    parser.add_argument(
        "--boundary_mask_subdir",
        default=None,
        help="Optional per-scene subdirectory containing frame label maps, e.g. frame_label_maps.",
    )
    parser.add_argument("--boundary_frame_stride", default=50, type=int)
    parser.add_argument("--boundary_max_frames", default=30, type=int)
    parser.add_argument("--boundary_unknown_label", default=0, type=int)
    parser.add_argument("--boundary_min_observations", default=1, type=int)
    parser.add_argument("--boundary_min_conflict_ratio", default=1.0, type=float)
    parser.add_argument("--boundary_visibility_tolerance", default=0.08, type=float)
    parser.add_argument(
        "--boundary_cut_against_background",
        default=False,
        action="store_true",
        help="Treat a known instance label next to label 0 as a boundary. Disabled by default.",
    )
    parser.add_argument("--overwrite", default=False, action="store_true")
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
    with open(summary_path, "w") as f:
        json.dump({"params": vars(args), "scenes": summaries}, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

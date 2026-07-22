import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement


MODULE_PATH = Path(__file__).parents[1] / "tools" / "generate_geometric_superpoints.py"
SPEC = importlib.util.spec_from_file_location("generate_geometric_superpoints", MODULE_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_ibsp_prunes_only_edges_with_consistent_2d_boundary_evidence(tmp_path):
    scene_dir = tmp_path / "scene0000_00"
    (scene_dir / "color").mkdir(parents=True)
    (scene_dir / "poses").mkdir()
    (scene_dir / "color" / "0.jpg").touch()
    np.savetxt(scene_dir / "poses" / "0.txt", np.eye(4), fmt="%.6f")
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[0, 2] = 1.5
    np.savetxt(scene_dir / "intrinsics.txt", intrinsics, fmt="%.6f")

    mask_root = tmp_path / "masks"
    (mask_root / "scene0000_00").mkdir(parents=True)
    imageio = __import__("imageio.v2", fromlist=["imwrite"])
    imageio.imwrite(mask_root / "scene0000_00" / "0.png", np.array([[1, 2, 2]], dtype=np.uint16))

    points = np.array([[-1.5, 0.0, 1.0], [-0.5, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=np.float32)
    left = np.array([0, 1], dtype=np.int32)
    right = np.array([1, 2], dtype=np.int32)
    args = SimpleNamespace(
        boundary_frame_stride=1,
        boundary_max_frames=None,
        boundary_mask_root=str(mask_root),
        boundary_mask_extension=".png",
        boundary_visibility_tolerance=0.0,
        boundary_unknown_label=0,
        boundary_cut_against_background=False,
        boundary_min_observations=1,
        boundary_min_conflict_ratio=1.0,
    )

    keep, stats = GENERATOR._boundary_keep_mask(
        points, left, right, str(scene_dir), "scene0000_00", args
    )

    assert keep.tolist() == [False, True]
    assert stats["pruned_edges"] == 1


def test_boundary_mask_subdir_is_supported(tmp_path):
    root = tmp_path / "masks"
    expected = root / "scene0000_00" / "frame_label_maps" / "12.png"

    path = GENERATOR._mask_path(root, "scene0000_00", "12", ".png", "frame_label_maps")

    assert path == expected


def test_boundary_frame_names_use_available_label_maps_in_subdir(tmp_path):
    scene_dir = tmp_path / "scene0000_00"
    (scene_dir / "color").mkdir(parents=True)
    for frame in ("0", "1", "2"):
        (scene_dir / "color" / f"{frame}.jpg").touch()
    mask_root = tmp_path / "masks"
    label_dir = mask_root / "scene0000_00" / "frame_label_maps"
    label_dir.mkdir(parents=True)
    for frame in ("0", "10", "20"):
        (label_dir / f"{frame}.png").touch()
    args = SimpleNamespace(
        boundary_mask_root=str(mask_root),
        boundary_mask_subdir="frame_label_maps",
        boundary_mask_extension=".png",
        boundary_frame_stride=1,
        boundary_max_frames=None,
    )

    names = GENERATOR._boundary_frame_names(str(scene_dir), "scene0000_00", args)

    assert names == ["0", "10", "20"]


def test_mesh_normal_graph_uses_triangle_connectivity_and_aligns_vertices(tmp_path):
    scene_dir = tmp_path / "scene0000_00"
    scene_dir.mkdir()
    vertices = np.array(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    faces = np.array([([0, 1, 2],)], dtype=[("vertex_indices", "i4", (3,))])
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")], text=False).write(
        scene_dir / "scene0000_00_vh_clean_2.ply"
    )
    points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype=np.float32)
    args = SimpleNamespace(
        mesh_name_template="{scene_name}_vh_clean_2.ply",
        mesh_alignment_tolerance=1e-6,
    )

    left, right, weights = GENERATOR._build_mesh_normal_edges(
        points, str(scene_dir), "scene0000_00", args
    )

    assert list(zip(left.tolist(), right.tolist())) == [(0, 1), (0, 2), (2, 1)]
    assert np.allclose(weights, 0.0)

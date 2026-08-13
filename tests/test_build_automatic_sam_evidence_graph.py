import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_evidence_graph.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_evidence_graph", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(node_id, observation_id, frame_index, points, superpoint_ids, area=10):
    points = np.asarray(points, dtype=np.int64)
    return {
        "node_id": node_id,
        "observation_id": observation_id,
        "frame_id": str(frame_index),
        "frame_index": frame_index,
        "point_count": len(points),
        "area": area,
        "bbox_xywh": [0, 0, 10, 10],
        "points": points,
        "superpoint_ids": np.asarray(superpoint_ids, dtype=np.int64),
        "centroid": np.asarray([float(node_id), 0.0, 0.0], dtype=np.float32),
    }


def test_same_frame_relation_preserves_continuous_containment_features():
    graph = _module()
    parent = _node(0, 10, 0, [0, 1, 2, 3], [4, 5], area=100)
    child = _node(1, 11, 0, [0, 1], [4], area=20)
    relation = graph._same_frame_relation(parent, child)
    assert relation["shared_point_count"] == 2
    assert relation["left_point_coverage"] == 0.5
    assert relation["right_point_coverage"] == 1.0
    assert relation["larger_area_node_id"] == 0
    assert relation["smaller_area_node_id"] == 1


def test_cross_view_edge_reports_depth_visibility_without_acceptance_flag():
    graph = _module()
    left = _node(0, 10, 0, [0, 1, 2], [4, 5])
    right = _node(1, 11, 1, [1, 2, 3], [5, 6])
    visibility = np.asarray([[True, True, True, False], [False, True, True, True]], dtype=bool)
    frame_visible = graph._frame_visible_superpoints(
        visibility, np.asarray([4, 5, 5, 6], dtype=np.int64), [0, 1]
    )
    edge = graph._cross_view_edge(left, right, ["shared_point"], visibility, frame_visible)
    assert edge["shared_point_count"] == 2
    assert edge["left_visible_in_right_frame_count"] == 2
    assert edge["right_visible_in_left_frame_count"] == 2
    assert edge["left_to_right_reprojection_support_ratio"] == 1.0
    assert edge["jointly_visible_shared_superpoint_count"] == 1
    assert "accepted" not in edge


def test_scene_build_writes_graph_records_and_does_not_write_candidates(tmp_path):
    graph = _module()
    scene_name = "scene0000_00"
    automatic_root = tmp_path / "automatic"
    scene_root = automatic_root / scene_name
    points_dir = scene_root / "points"
    points_dir.mkdir(parents=True)
    np.savez_compressed(points_dir / "first.npz", point_indices=np.asarray([0, 1, 2], dtype=np.int64))
    np.savez_compressed(points_dir / "second.npz", point_indices=np.asarray([1, 2, 3], dtype=np.int64))
    records = [
        {"observation_id": 7, "scene_name": scene_name, "frame_id": "0", "frame_index": 0,
         "point_indices_path": "points/first.npz", "area": 30, "predicted_iou": 0.9,
         "stability_score": 0.95, "bbox_xywh": [0, 0, 10, 10], "crop_box_xywh": [0, 0, 10, 10]},
        {"observation_id": 8, "scene_name": scene_name, "frame_id": "1", "frame_index": 1,
         "point_indices_path": "points/second.npz", "area": 20, "predicted_iou": 0.9,
         "stability_score": 0.95, "bbox_xywh": [1, 1, 8, 8], "crop_box_xywh": [0, 0, 10, 10]},
    ]
    (scene_root / "automatic_observations.jsonl").write_text("\n".join(graph.json.dumps(item) for item in records) + "\n")
    processed_root = tmp_path / "processed"
    processed_scene = processed_root / scene_name
    processed_scene.mkdir(parents=True)
    processed = np.zeros((4, 10), dtype=np.float32)
    processed[:, :3] = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    processed[:, 9] = np.asarray([2, 2, 3, 3], dtype=np.float32)
    np.save(processed_scene / "0000_00.npy", processed)
    output_root = tmp_path / "graph"
    args = SimpleNamespace(
        automatic_root=automatic_root,
        processed_scene_root=processed_root,
        dataset_root=tmp_path,
        config_path=tmp_path / "unused.yaml",
        track_root=None,
        output_root=output_root,
        knn=2,
        max_centroid_distance=5.0,
        max_nodes_per_entity=8,
        with_visibility=False,
    )
    summary = graph._build_scene(scene_name, args)
    assert summary["node_count"] == 2
    assert summary["cross_view_edge_count"] == 1
    scene_output = output_root / scene_name
    assert (scene_output / "nodes.jsonl").is_file()
    assert (scene_output / "cross_view_edges.jsonl").is_file()
    assert not list(scene_output.glob("*candidate*"))

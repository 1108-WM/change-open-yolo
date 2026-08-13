import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_track_growth_ledger.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_track_growth_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_superpoint_context_keeps_original_ids_and_contact_features():
    ledger = _module()
    processed = np.zeros((6, 10), dtype=np.float32)
    processed[:, :3] = np.asarray([
        [0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0],
        [0.03, 0.0, 0.0], [0.04, 0.0, 0.0], [0.05, 0.0, 0.0],
    ])
    processed[:, 3:6] = 128
    processed[:, 6:9] = [0.0, 0.0, 1.0]
    processed[:, 9] = [10, 10, 10, 30, 30, 30]
    context = ledger._raw_superpoint_context(processed, 2, 0.03, 1, 0.0)
    assert context["raw_ids"].tolist() == [10, 30]
    assert context["neighbors"][10][0]["neighbor_superpoint_id"] == 30
    assert context["neighbors"][10][0]["boundary_contact_count"] > 0


def test_growth_frontier_requires_cross_view_graph_evidence():
    ledger = _module()
    support = [{"superpoint_id": 10, "point_count": 3, "support_view_count": 2, "support_node_count": 2,
                "max_observation_occupancy": 1.0, "mean_observation_occupancy": 1.0,
                "quality_weighted_occupancy_sum": 1.5, "track_id": 0}]
    neighbors = {10: [{"neighbor_superpoint_id": 30, "boundary_contact_count": 3, "boundary_contact_ratio": 1.0,
                       "mean_boundary_distance": 0.01, "mean_normal_difference": 0.0,
                       "mean_color_difference": 0.0}]}
    assert ledger._growth_frontier(0, support, neighbors, {}) == []
    links = {(0, 30): {"edge_count": 2, "reprojection_support": [0.7, 0.8], "source_node_ids": {1, 2}}}
    frontier = ledger._growth_frontier(0, support, neighbors, links)
    assert frontier[0]["superpoint_id"] == 30
    assert frontier[0]["cross_view_link_count"] == 2
    assert "candidate" not in frontier[0]


def test_scene_build_writes_only_growth_ledger(tmp_path):
    ledger = _module()
    scene_name = "scene0000_00"
    graph_scene = tmp_path / "graph" / scene_name
    graph_scene.mkdir(parents=True)
    nodes = [
        {"node_id": 0, "observation_id": 1, "frame_index": 0, "quality": 0.9,
         "superpoint_ids": [10], "superpoint_point_counts": [3]},
        {"node_id": 1, "observation_id": 2, "frame_index": 1, "quality": 0.9,
         "superpoint_ids": [10], "superpoint_point_counts": [3]},
        {"node_id": 2, "observation_id": 3, "frame_index": 2, "quality": 0.8,
         "superpoint_ids": [30], "superpoint_point_counts": [3]},
    ]
    (graph_scene / "nodes.jsonl").write_text("\n".join(json.dumps(item) for item in nodes) + "\n")
    edge = {"left_node_id": 0, "right_node_id": 2, "left_to_right_reprojection_support_ratio": 0.8,
            "right_to_left_reprojection_support_ratio": 0.6}
    (graph_scene / "cross_view_edges.jsonl").write_text(json.dumps(edge) + "\n")
    tracks_scene = tmp_path / "tracks" / scene_name
    tracks_scene.mkdir(parents=True)
    (tracks_scene / "automatic_tracks.json").write_text(json.dumps({"tracks": [{"track_id": 0, "observation_ids": [1, 2]}]}))
    processed_root = tmp_path / "processed"
    processed_scene = processed_root / scene_name
    processed_scene.mkdir(parents=True)
    processed = np.zeros((6, 10), dtype=np.float32)
    processed[:, :3] = np.asarray([[index * 0.01, 0, 0] for index in range(6)], dtype=np.float32)
    processed[:, 3:6] = 128
    processed[:, 6:9] = [0, 0, 1]
    processed[:, 9] = [10, 10, 10, 30, 30, 30]
    np.save(processed_scene / "0000_00.npy", processed)
    args = SimpleNamespace(
        evidence_graph_root=tmp_path / "graph", track_root=tmp_path / "tracks",
        processed_scene_root=processed_root, output_root=tmp_path / "out", adjacency_knn=2,
        adjacency_max_distance=0.03, min_contact_points=1, min_contact_ratio=0.0,
    )
    summary = ledger._build_scene(scene_name, args)
    assert summary["track_count"] == 1
    record = json.loads((args.output_root / scene_name / "automatic_sam_track_growth_ledger.json").read_text())[0]
    assert record["growth_frontier_superpoints"][0]["superpoint_id"] == 30
    assert not list((args.output_root / scene_name).glob("*candidate*"))

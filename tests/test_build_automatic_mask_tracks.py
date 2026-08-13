import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_mask_tracks.py"
    spec = importlib.util.spec_from_file_location("automatic_tracks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(node_id, frame_index, quality):
    return {"node_id": node_id, "frame_index": frame_index, "quality": quality}


def test_point_iou_uses_unique_points():
    module = _module()
    value, intersection = module._point_iou(np.asarray([1, 2, 3]), np.asarray([3, 4]))
    assert intersection == 1
    assert value == 0.25


def test_tracks_reject_weak_chain_after_two_nodes():
    module = _module()
    nodes = [_node(0, 0, 1.0), _node(1, 1, 0.9), _node(2, 2, 0.8)]
    edges = [
        {"left_node_id": 0, "right_node_id": 1, "edge_score": 0.9, "accepted": True},
        {"left_node_id": 1, "right_node_id": 2, "edge_score": 0.9, "accepted": True},
    ]
    tracks = module._build_tracks(nodes, edges, min_track_views=2, min_track_support_edges=2)
    assert len(tracks) == 1
    assert {item["node_id"] for item in tracks[0]["nodes"]} == {0, 1}

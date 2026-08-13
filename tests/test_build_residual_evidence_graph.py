import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "build_residual_evidence_graph.py"
SPEC = importlib.util.spec_from_file_location("build_residual_evidence_graph", MODULE_PATH)
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


def _node(node_id, frame_index, quality=0.8):
    return {
        "node_id": node_id,
        "frame_index": frame_index,
        "quality": quality,
        "points": np.asarray([node_id, node_id + 10], dtype=np.int64),
    }


def test_point_iou_uses_unique_point_sets():
    iou, intersection = GRAPH._point_iou(
        np.asarray([1, 3, 5], dtype=np.int64),
        np.asarray([3, 5, 7], dtype=np.int64),
    )
    assert intersection == 2
    assert np.isclose(iou, 0.5)


def test_tracks_do_not_accept_a_weak_chain_with_two_support_requirement():
    nodes = [_node(0, 0), _node(1, 1), _node(2, 2)]
    edges = [
        {"left_node_id": 0, "right_node_id": 1, "accepted": True, "edge_score": 0.9},
        {"left_node_id": 1, "right_node_id": 2, "accepted": True, "edge_score": 0.8},
    ]
    tracks = GRAPH._build_tracks(nodes, edges, min_track_views=2, min_track_support_edges=2)
    assert len(tracks) == 1
    assert {node["node_id"] for node in tracks[0]["nodes"]} == {0, 1}

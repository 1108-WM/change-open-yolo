import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_details_frame_siou_tracks.py"
    spec = importlib.util.spec_from_file_location("details_frame_siou_tracks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(node_id, observation_id, frame_index, superpoints):
    return {
        "node_id": node_id,
        "observation_id": observation_id,
        "frame_index": frame_index,
        "quality": 1.0,
        "lifted_superpoints": np.asarray(superpoints, dtype=np.int64),
    }


def test_framewise_siou_uses_only_jointly_visible_superpoints():
    module = _module()
    value = module.framewise_siou([1, 2, 3], [2, 3, 4], [2, 3, 5])
    assert value == 1.0


def test_sequential_association_does_not_join_only_by_nonvisible_overlap():
    module = _module()
    nodes = [_node(0, 0, 0, [1, 2]), _node(1, 1, 1, [2, 3])]
    visible = {0: np.asarray([1, 2]), 1: np.asarray([1, 3])}
    assert module.associate_sequentially(nodes, visible, 0.30, 2) == []


def test_sequential_association_is_one_to_one_per_frame():
    module = _module()
    nodes = [
        _node(0, 0, 0, [1, 2]),
        _node(1, 1, 1, [1, 2]),
        _node(2, 2, 1, [1, 2]),
    ]
    visible = {0: np.asarray([1, 2]), 1: np.asarray([1, 2])}
    tracks = module.associate_sequentially(nodes, visible, 0.30, 2)
    assert len(tracks) == 1
    assert [node["observation_id"] for node in tracks[0]["nodes"]] == [0, 1]

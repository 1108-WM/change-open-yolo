import numpy as np

from tools.train_ncs_fi1_stage_c_member_refinement_oof import (
    connected_refinement_indexes,
)


def _atom(raw_id, role, count=1):
    return {
        "raw_superpoint_id": raw_id,
        "role": role,
        "points": np.arange(count),
    }


def test_connectivity_keeps_shared_and_only_connected_selected_exclusive_atoms():
    atoms = [
        _atom(1, "shared"),
        _atom(2, "track_only"),
        _atom(3, "native_only"),
        _atom(9, "track_only"),
    ]
    neighbors = {
        1: [{"neighbor_superpoint_id": 2}],
        2: [{"neighbor_superpoint_id": 1}, {"neighbor_superpoint_id": 3}],
        3: [{"neighbor_superpoint_id": 2}],
    }
    assert connected_refinement_indexes(atoms, [0.0, 0.8, 0.7, 0.9], neighbors) == {0, 1, 2}


def test_no_shared_fallback_uses_largest_selected_component():
    atoms = [_atom(1, "track_only", 2), _atom(2, "track_only", 3), _atom(9, "native_only", 6)]
    neighbors = {1: [{"neighbor_superpoint_id": 2}], 2: [{"neighbor_superpoint_id": 1}]}
    assert connected_refinement_indexes(atoms, [0.9, 0.9, 0.9], neighbors) == {2}


def test_same_raw_superpoint_role_fragments_are_connected():
    atoms = [_atom(4, "shared"), _atom(4, "native_only")]
    assert connected_refinement_indexes(atoms, [0.0, 0.6], {}) == {0, 1}

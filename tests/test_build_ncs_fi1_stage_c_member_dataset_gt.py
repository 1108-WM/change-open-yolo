import numpy as np
import pytest

from tools.build_ncs_fi1_stage_c_member_dataset_gt import (
    _geometry_sha256,
    decompose_union_atoms,
)


def test_member_atoms_are_disjoint_and_conserve_union_by_role():
    raw = np.asarray([10, 10, 10, 20, 20, 30])
    track = np.asarray([0, 1, 3, 5])
    native = np.asarray([1, 2, 3, 4])
    union = np.union1d(track, native)
    atoms = decompose_union_atoms(union, track, native, raw)
    by_role = {
        role: sorted(np.concatenate([row["points"] for row in atoms if row["role"] == role]).tolist())
        for role in ("shared", "track_only", "native_only")
    }
    assert by_role == {
        "shared": [1, 3],
        "track_only": [0, 5],
        "native_only": [2, 4],
    }
    assert sum(len(row["points"]) for row in atoms) == len(union)


def test_member_decomposition_rejects_non_parent_union():
    with pytest.raises(ValueError, match="differs"):
        decompose_union_atoms(
            np.asarray([0, 1, 2]), np.asarray([0]), np.asarray([1]), np.asarray([1, 1, 2])
        )


def test_geometry_hash_is_order_and_duplicate_invariant():
    assert _geometry_sha256(np.asarray([3, 1, 1])) == _geometry_sha256(np.asarray([1, 3]))

import numpy as np
import pytest

from tools.audit_dm_sms1_unique_geometry_ledger import GeometryResolver


def test_geometry_resolver_reads_native_mask_column(tmp_path):
    path = tmp_path / "masks.npy"
    masks = np.asarray([
        [False, True],
        [True, False],
        [False, True],
    ], dtype=bool)
    np.save(path, masks, allow_pickle=False)
    points = GeometryResolver().points({
        "kind": "native_mask_column",
        "masks_path": str(path),
        "column_index": 1,
    })
    assert points.tolist() == [0, 2]


def test_geometry_resolver_reads_unique_npz_points(tmp_path):
    path = tmp_path / "points.npz"
    np.savez(path, point_indices=np.asarray([3, 1, 3], dtype=np.int64))
    points = GeometryResolver().points({
        "kind": "point_indices_npz",
        "points_path": str(path),
    })
    assert points.tolist() == [1, 3]


def test_geometry_resolver_rejects_unknown_locator_kind():
    with pytest.raises(ValueError, match="unsupported geometry locator kind"):
        GeometryResolver().points({"kind": "unknown"})

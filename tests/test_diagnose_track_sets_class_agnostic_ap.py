import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_track_sets_class_agnostic_ap.py"
    spec = importlib.util.spec_from_file_location("track_class_agnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_track_points_filters_invalid_indices(tmp_path):
    module = _module()
    points_path = tmp_path / "points.npz"
    np.savez_compressed(points_path, point_indices=np.asarray([-1, 1, 1, 4, 7]))
    assert module._track_points({"points_path": str(points_path)}, 5).tolist() == [1, 4]


def test_class_agnostic_mapping_keeps_instances_distinct():
    module = _module()
    valid = next(iter(module.VALID_GT_CLASSES))
    ids = np.asarray([0, valid * 1000 + 1, valid * 1000 + 1, valid * 1000 + 2])
    mapped = module._class_agnostic_gt_ids(ids)
    assert mapped[0] == 0
    assert mapped[1] == mapped[2]
    assert mapped[1] != mapped[3]

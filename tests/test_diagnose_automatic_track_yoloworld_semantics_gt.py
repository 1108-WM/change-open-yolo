import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_automatic_track_yoloworld_semantics_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_semantics_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_gt_returns_no_match_for_ignored_points():
    module = _module()
    instance_id, iou = module._best_gt(np.asarray([0, 1]), np.asarray([0, 0, 1001]), {1001: 1})
    assert instance_id == -1
    assert iou == 0.0

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_track_gvc_features_gt.py"
    spec = importlib.util.spec_from_file_location("gvc_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_gt_uses_iou_for_fixed_track_points():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 1001, 2001, 2001, 2001, 2001], dtype=np.int64)
    instance_id, iou = module._best_gt(np.asarray([0, 1, 3, 4, 5]), gt_ids, {1001: 3, 2001: 4})
    assert instance_id == 2001
    assert iou == 0.5

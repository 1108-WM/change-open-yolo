import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_automatic_mask_tracks_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_tracks_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_gt_uses_iou_not_only_point_precision():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 1001, 2001, 2001, 2001, 2001], dtype=np.int64)
    match = module._best_gt(np.asarray([0, 1, 3, 4, 5]), gt_ids, {1001: 3, 2001: 4})
    assert match["instance_id"] == 2001
    assert match["coverage"] == 0.75

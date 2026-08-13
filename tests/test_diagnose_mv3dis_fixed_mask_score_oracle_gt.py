import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_mv3dis_fixed_mask_score_oracle_gt.py"
    spec = importlib.util.spec_from_file_location("fixed_mask_score_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ideal_scores_use_best_valid_instance_iou_without_changing_masks():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 2001, 2001, 0], dtype=np.int64)
    masks = np.asarray([
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=bool)
    before = masks.copy()
    scores = module.ideal_class_agnostic_scores(masks, gt_ids, {1, 2}, min_region_size=1)
    assert np.allclose(scores, [1.0, 1.0 / 3.0, 2.0 / 3.0])
    assert np.array_equal(masks, before)

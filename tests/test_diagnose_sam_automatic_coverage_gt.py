import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_sam_automatic_coverage_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_support_keeps_the_best_mask_per_gt_and_frame():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 1001, 1001, 2001, 2001], dtype=np.int64)
    observations = [
        {"frame_id": "0", "points": np.asarray([0, 4]), "predicted_iou": 0.9, "stability_score": 0.9},
        {"frame_id": "0", "points": np.asarray([0, 1, 2]), "predicted_iou": 0.8, "stability_score": 0.8},
        {"frame_id": "1", "points": np.asarray([2, 3]), "predicted_iou": 0.7, "stability_score": 0.7},
    ]
    support = module._collect_support(
        observations,
        gt_ids,
        eligible_ids={1001},
        min_points=2,
        min_precision=0.5,
    )
    assert set(support[1001]) == {"0", "1"}
    assert support[1001]["0"]["intersection"] == 3
    assert np.array_equal(support[1001]["0"]["points"], np.asarray([0, 1, 2]))

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_counterevidence_superpoint_variants_gt.py"
    spec = importlib.util.spec_from_file_location("counterevidence_variant_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iou_to_fixed_gt_does_not_rematch_to_a_different_instance():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 2001, 2001])
    value = module._iou_to_fixed_gt(np.asarray([0, 2, 3]), gt_ids, 1001, {1001: 2, 2001: 2})
    assert value == 0.25


def test_summary_keeps_baseline_gap_subset_separate():
    module = _module()
    rows = [{
        "original_best_gt_iou": 0.3,
        "fixed_gt_instance_id": 1001,
        "variant_iou_to_fixed_gt": 0.4,
        "iou_delta": 0.1,
        "removed_point_count": 3,
        "matched_gt_residual_type": "无合格三维候选",
    }]
    assert module._summary(rows)["原轨迹几何合格且对应基线缺口"]["IoU 提升轨迹数"] == 1

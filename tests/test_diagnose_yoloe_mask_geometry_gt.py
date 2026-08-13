import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_yoloe_mask_geometry_gt.py"
    spec = importlib.util.spec_from_file_location("diagnose_yoloe_mask_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_support_keeps_best_same_class_mask_per_frame():
    module = _module()
    gt_ids = np.asarray([1001, 1001, 1001, 2001, 2001])
    observations = [
        {"frame_id": "0", "label_id": 4, "points": np.asarray([0, 3]), "score": 0.9},
        {"frame_id": "0", "label_id": 4, "points": np.asarray([0, 1]), "score": 0.8},
        {"frame_id": "1", "label_id": 4, "points": np.asarray([1, 2]), "score": 0.7},
        {"frame_id": "1", "label_id": 5, "points": np.asarray([0, 1, 2]), "score": 0.99},
    ]
    support = module._collect_support(observations, gt_ids, 1001, 4, min_points=2, min_precision=0.5)
    assert set(support) == {"0", "1"}
    assert np.array_equal(support["0"]["points"], np.asarray([0, 1]))


def test_summary_counts_only_multiview_iou_qualified_rows():
    module = _module()
    rows = [
        {"residual_type": "无合格三维候选", "multi_view_supported": True, "mask_union_iou": 0.3},
        {"residual_type": "边界不足", "multi_view_supported": False, "mask_union_iou": 0.8},
    ]
    summary = module._summary(rows)
    assert summary["YOLOE 独有严格残差实例数"] == 2
    assert summary["指标"]["多视角 YOLOE mask 并集 IoU 不低于 25%"]["实例数"] == 1

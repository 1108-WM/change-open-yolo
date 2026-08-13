import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_details_consensus_failure_modes_gt.py"
    spec = importlib.util.spec_from_file_location("details_consensus_failures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_superpoint_subset_chooses_best_precision_prefix():
    module = _module()
    superpoints = np.asarray([1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    gt_ids = np.asarray([1001, 1001, 1001, 0, 0, 1001, 0, 0], dtype=np.int64)
    result = module.oracle_superpoint_subset({1, 2, 3}, superpoints, gt_ids, 1001, 4)
    assert result["superpoint_count"] == 1
    assert result["point_count"] == 2
    assert result["iou"] == 0.5


def test_classify_instance_separates_fragmentation_and_selection_loss():
    module = _module()
    fragmented = module.classify_instance(0.20, 0.22, 0.18, 0.40, 0.30, 2)
    selection = module.classify_instance(0.20, 0.22, 0.18, 0.40, 0.20, 1)
    assert fragmented == "共识轨迹碎裂且并集可恢复"
    assert selection == "当前共识选择损失"


def test_classify_instance_reports_superpoint_closure_contamination():
    module = _module()
    result = module.classify_instance(0.30, 0.20, 0.10, 0.40, 0.10, 1)
    assert result == "原始superpoint闭包污染"

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_anchor_guided_track_expansions_gt.py"
    spec = importlib.util.spec_from_file_location("anchor_guided_expansion_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_new_target_coverage_only_after_expansion():
    module = _module()
    rows = [
        {"scene_name": "s", "anchor_gt_instance_id": 1001, "meets_minimum_support": True,
         "matched_gt_residual_type": "无合格三维候选", "anchor_iou": 0.2,
         "expanded_same_instance_iou": 0.3, "same_instance_iou_change": 0.1, "same_instance_precision_change": -0.1},
        {"scene_name": "s", "anchor_gt_instance_id": 1002, "meets_minimum_support": True,
         "matched_gt_residual_type": "已有严格三维候选", "anchor_iou": 0.6,
         "expanded_same_instance_iou": 0.7, "same_instance_iou_change": 0.1, "same_instance_precision_change": 0.1},
    ]
    summary = module._summary(rows)
    assert summary["目标残差几何合格独立实例数"]["新增"] == 1
    assert summary["目标残差中由不足 IoU 25% 提升到不低于 25% 的扩展区域数"] == 1

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_anchor_guided_selector_features_gt.py"
    spec = importlib.util.spec_from_file_location("anchor_selector_features_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_separates_consistent_and_changed_semantics():
    module = _module()
    rows = [
        {"anchor_iou": 0.1, "expanded_same_instance_iou": 0.3, "matched_gt_residual_type": "无合格三维候选",
         "expanded_matches_original_alpha": True, "expanded_matches_yoloworld": False, "original_alphaclip_correct": False,
         "yoloworld_correct": False, "expanded_alpha_correct": "True"},
        {"anchor_iou": 0.1, "expanded_same_instance_iou": 0.3, "matched_gt_residual_type": "边界不足",
         "expanded_matches_original_alpha": False, "expanded_matches_yoloworld": True, "original_alphaclip_correct": False,
         "yoloworld_correct": True, "expanded_alpha_correct": "False"},
    ]
    summary = module._summary(rows)
    assert summary["首次几何合格的目标扩展区域数"] == 2
    assert summary["扩展语义与原 Alpha-CLIP 一致"]["扩展 Alpha-CLIP 正确区域数"] == 1

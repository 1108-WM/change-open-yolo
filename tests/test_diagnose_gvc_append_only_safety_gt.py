import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_gvc_append_only_safety_gt.py"
    spec = importlib.util.spec_from_file_location("gvc_append_only_safety", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safety_summary_separates_target_and_same_class_native_duplicates():
    module = _module()
    rows = [
        {"best_gt_iou": 0.6, "semantic_correct": True, "matched_gt_residual_type": "无合格三维候选", "same_class_native_max_iou": 0.1, "candidate_score": 0.8},
        {"best_gt_iou": 0.4, "semantic_correct": False, "matched_gt_residual_type": "边界不足", "same_class_native_max_iou": 0.7, "candidate_score": 0.3},
        {"best_gt_iou": 0.1, "semantic_correct": False, "matched_gt_residual_type": "无有效 GT", "same_class_native_max_iou": 0.0, "candidate_score": 0.1},
    ]
    summary = module._summary(rows)
    assert summary["目标基线缺口候选数"] == 2
    assert summary["目标基线缺口中几何且语义正确候选数"] == 1
    assert summary["与同类 native 候选 IoU≥50% 的候选数"] == 1

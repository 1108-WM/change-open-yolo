import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_automatic_track_alphaclip_semantics_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_track_alpha_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_semantic_complementarity():
    module = _module()
    rows = [
        {"scene_name": "s", "best_gt_instance_id": 1, "best_gt_iou": 0.4, "matched_gt_residual_type": "无合格三维候选",
         "alphaclip_correct": True, "yoloworld_correct": False},
        {"scene_name": "s", "best_gt_instance_id": 2, "best_gt_iou": 0.4, "matched_gt_residual_type": "边界不足",
         "alphaclip_correct": False, "yoloworld_correct": True},
    ]
    summary = module._summary(rows)
    assert summary["几何 IoU 不低于 25% 的 Alpha-CLIP 语义正确率"] == 0.5
    assert summary["几何 IoU 不低于 25% 的互补情况"]["仅 Alpha-CLIP 正确"] == 1
    assert summary["几何 IoU 不低于 25% 的互补情况"]["仅 YOLO-World 正确"] == 1

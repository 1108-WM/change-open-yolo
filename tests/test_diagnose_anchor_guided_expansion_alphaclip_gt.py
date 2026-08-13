import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_anchor_guided_expansion_alphaclip_gt.py"
    spec = importlib.util.spec_from_file_location("expansion_alphaclip_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_requires_geometry_and_semantics_together():
    module = _module()
    rows = [
        {"scene_name": "s", "track_id": 1, "anchor_gt_instance_id": 1001, "matched_gt_residual_type": "无合格三维候选",
         "anchor_iou": 0.1, "expanded_same_instance_iou": 0.3, "expanded_alpha_correct": True},
        {"scene_name": "s", "track_id": 2, "anchor_gt_instance_id": 1002, "matched_gt_residual_type": "无合格三维候选",
         "anchor_iou": 0.1, "expanded_same_instance_iou": 0.3, "expanded_alpha_correct": False},
    ]
    summary = module._summary(rows)
    assert summary["无合格三维候选"]["首次几何合格区域数"] == 2
    assert summary["无合格三维候选"]["Alpha-CLIP 类别正确区域数"] == 1

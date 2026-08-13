import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_visibility_counterevidence_gt.py"
    spec = importlib.util.spec_from_file_location("diagnose_counterevidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_join_rows_attaches_gt_only_after_evidence_is_fixed():
    module = _module()
    joined = module._join_rows(
        [{"scene_name": "scene0000_00", "track_id": 1, "positive_support_frame_count": 2}],
        [{"scene_name": "scene0000_00", "track_id": "1", "best_gt_instance_id": "1001", "best_gt_iou": "0.3", "best_gt_precision": "0.8", "matched_gt_residual_type": "无合格三维候选"}],
    )
    assert joined[0]["best_gt_iou"] == 0.3
    assert joined[0]["matched_gt_residual_type"] == "无合格三维候选"

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_automatic_track_multiview_relations_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_multiview_relation_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_counts_target_tracks_and_identity_groups():
    module = _module()
    rows = [
        {"best_gt_iou": 0.4, "matched_gt_residual_type": "无合格三维候选", "scene_name": "s", "best_gt_instance_id": 1,
         "top_candidate_support_view_count": 3, "top_candidate_support_view_ratio": 1.0,
         "candidate_identity_margin": 1.0, "candidate_with_support_count": 1},
        {"best_gt_iou": 0.1, "matched_gt_residual_type": "无合格三维候选", "scene_name": "s", "best_gt_instance_id": 2,
         "top_candidate_support_view_count": 0, "top_candidate_support_view_ratio": 0.0,
         "candidate_identity_margin": 0.0, "candidate_with_support_count": 0},
    ]
    summary = module._summary(rows)
    assert summary["目标几何合格轨迹数"] == 1
    assert summary["与同一候选的跨帧支持比例"]["全部支持视角"]["目标几何合格轨迹数"] == 1
    assert summary["候选身份间隔"]["无候选共现"]["目标几何合格轨迹数"] == 0

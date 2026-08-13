import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_automatic_track_geometry_relations_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_relation_gt_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_route_summary_counts_only_geometrically_valid_tracks():
    module = _module()
    rows = [
        {"scene_name": "scene", "track_id": 1, "route": "新增实例", "best_gt_iou": 0.40,
         "best_gt_instance_id": 1001, "matched_gt_residual_type": "无合格三维候选"},
        {"scene_name": "scene", "track_id": 2, "route": "新增实例", "best_gt_iou": 0.10,
         "best_gt_instance_id": 1002, "matched_gt_residual_type": "无合格三维候选"},
        {"scene_name": "scene", "track_id": 3, "route": "边界竞争", "best_gt_iou": 0.50,
         "best_gt_instance_id": 2001, "matched_gt_residual_type": "边界不足"},
    ]
    summary = module._route_summary(rows)
    assert summary["新增实例"]["轨迹数"] == 2
    assert summary["新增实例"]["几何合格轨迹数"] == 1
    assert summary["新增实例"]["几何合格轨迹对应的独立 GT 实例数"]["无合格三维候选"] == 1
    assert summary["边界竞争"]["几何合格轨迹对应的独立 GT 实例数"]["边界不足"] == 1

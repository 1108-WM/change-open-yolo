import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_counterevidence_reliability_gt.py"
    spec = importlib.util.spec_from_file_location("reliability_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iou_decreases_when_removing_true_object_points():
    module = _module()
    before = module._iou(point_count=10, intersection=8, target_size=10)
    after = module._iou(point_count=6, intersection=4, target_size=10)
    assert after < before

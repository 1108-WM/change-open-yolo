import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_automatic_track_fragmentation_gt.py"
    spec = importlib.util.spec_from_file_location("automatic_track_fragmentation_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_distinguishes_track_fragmentation_from_missing_tracks():
    module = _module()
    assert module._classify(0.4, 0.3, 1, 0.3) == "已有几何合格轨迹"
    assert module._classify(0.4, 0.1, 0, 0.0) == "三维关联未形成主属轨迹"
    assert module._classify(0.4, 0.1, 2, 0.3) == "同目标轨迹碎裂且并集可恢复"
    assert module._classify(0.4, 0.1, 2, 0.2) == "主属轨迹存在但并集仍不足"

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_counterevidence_score_gt.py"
    spec = importlib.util.spec_from_file_location("counter_score_gt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_split_keeps_one_scene_in_one_fixed_fold():
    module = _module()
    scenes = ["scene0000_00", "scene0001_00"]
    assert module._scene_split("scene0000_00", scenes) == "A"
    assert module._scene_split("scene0001_00", scenes) == "B"

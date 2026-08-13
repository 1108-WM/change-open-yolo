import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_z6c_candidate_selector_dataset_gt.py"
    spec = importlib.util.spec_from_file_location("z6c_selector_dataset", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_candidate_options_add_current_fallback_once():
    module = _module()
    node = {"candidate_classes": [{"class_index": 1, "class_name": "b"}]}
    options = module._candidate_options(node, 2, ["a", "b", "c"])
    assert [row["class_index"] for row in options] == [1, 2]
    assert options[1]["in_yolo_top5"] is False
    assert len(module._candidate_options(node, 1, ["a", "b", "c"])) == 1
    assert len(module._candidate_options(node, 3, ["a", "b", "c"])) == 1

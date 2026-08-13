import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_groundingdino_independent_coverage_gt.py"
    spec = importlib.util.spec_from_file_location("diagnose_groundingdino_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_marks_groundingdino_only_evidence():
    module = _module()
    assert module._comparison_label({"reliable": False}, {"reliable": True}) == "仅 GroundingDINO 有可靠二维证据"


def test_rename_row_keeps_yoloworld_and_renames_second_source():
    module = _module()
    row = module._rename_row({"yoloe_reliable": True, "yoloworld_reliable": False})
    assert row == {"groundingdino_reliable": True, "yoloworld_reliable": False}

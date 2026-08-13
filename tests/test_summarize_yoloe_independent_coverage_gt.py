import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_yoloe_independent_coverage_gt.py"
    spec = importlib.util.spec_from_file_location("summarize_yoloe_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_scene_coverage_rejects_missing_scene():
    module = _module()
    with pytest.raises(ValueError, match="缺少"):
        module._validate_scene_coverage(["a", "b"], [{"scenes": ["a"]}])

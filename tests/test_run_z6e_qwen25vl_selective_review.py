import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "run_z6e_qwen25vl_selective_review.py"
    spec = importlib.util.spec_from_file_location("z6e_qwen", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_parse_has_strict_abstain_fallback():
    module = _module()
    assert module._parse("PROPOSED") == ("PROPOSED", True)
    assert module._parse("I choose PROPOSED") == ("ABSTAIN", False)

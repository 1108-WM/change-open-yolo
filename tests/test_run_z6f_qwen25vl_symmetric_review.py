import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "run_z6f_qwen25vl_symmetric_review.py"
    spec = importlib.util.spec_from_file_location("z6f_qwen", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_symmetric_semantic_mapping():
    module = _module()
    assert module._semantic_choice("A", proposed_is_a=False) == "CURRENT"
    assert module._semantic_choice("B", proposed_is_a=False) == "PROPOSED"
    assert module._semantic_choice("A", proposed_is_a=True) == "PROPOSED"
    assert module._semantic_choice("B", proposed_is_a=True) == "CURRENT"

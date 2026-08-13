import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_details_core_prompt_consensus_variants.py"
    spec = importlib.util.spec_from_file_location("details_core_prompt_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conservative_variant_only_adds_new_stable_superpoints():
    module = _module()
    variant, added = module.conservative_variant_superpoints(
        base_superpoints=[3, 1, 2],
        stable_added_superpoints=[4, 2, 4, 5],
    )
    assert variant == [1, 2, 3, 4, 5]
    assert added == [4, 5]


def test_conservative_variant_is_exact_fallback_without_stable_addition():
    module = _module()
    variant, added = module.conservative_variant_superpoints([3, 1, 2], [2, 1])
    assert variant == [1, 2, 3]
    assert added == []

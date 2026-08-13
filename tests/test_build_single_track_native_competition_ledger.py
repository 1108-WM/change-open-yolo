import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_single_track_native_competition_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("single_competition_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_has_no_gt_ap_semantic_or_action_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "ap", "class", "semantic", "suppress", "score-field")
    assert not any(any(token in option for token in forbidden) for option in options)
    assert "--geometry-variant" in options

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "build_mv3dis_relative_depth_guide_mask_matching_ledger.py"
    spec = importlib.util.spec_from_file_location("mv3dis_relative_matching", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_superpoint_support_preserves_counts_and_weight_sums():
    module = _load_module()
    counts, weight_sums = module.aggregate_weighted_superpoint_support(
        points=np.asarray([0, 1, 3]),
        weights=np.asarray([1.0, 0.5, 0.25]),
        superpoints=np.asarray([4, 4, 5, 5]),
    )
    assert counts == {4: 2, 5: 1}
    assert weight_sums == {4: 1.5, 5: 0.25}


def test_weighted_support_rejects_out_of_range_weights():
    module = _load_module()
    for weights in ([0.0], [1.1]):
        try:
            module.aggregate_weighted_superpoint_support(
                np.asarray([0]), np.asarray(weights), np.asarray([1])
            )
        except ValueError as error:
            assert "weights" in str(error)
        else:
            raise AssertionError("invalid weights must be rejected")


def test_cli_does_not_expose_paper_threshold_or_action_options():
    module = _load_module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("threshold", "alpha", "--gt", "--ap", "assign", "nms", "semantic")
    assert not any(any(token in option for token in forbidden) for option in options)

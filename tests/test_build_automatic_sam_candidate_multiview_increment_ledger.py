import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_candidate_multiview_increment_ledger.py"
    spec = importlib.util.spec_from_file_location("candidate_increment_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_increment_separates_any_and_same_class_native_explanation():
    module = _module()
    result = module.observation_increment(
        np.asarray([1, 2, 3, 4]), np.asarray([2, 3, 4, 5]),
        np.asarray([False, False, True, True, False, False]),
        np.asarray([False, False, False, True, False, False]),
    )
    assert result["candidate_observation_point_count"] == 3
    assert result["candidate_independent_point_count"] == 1
    assert result["candidate_any_native_explained_point_count"] == 2
    assert result["candidate_same_class_native_explained_point_count"] == 1
    assert result["candidate_independent_ratio"] == 1 / 3

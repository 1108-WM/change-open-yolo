import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "annotate_automatic_track_multiview_relations.py"
    spec = importlib.util.spec_from_file_location("automatic_multiview_relations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiview_support_prefers_consistently_supported_candidate():
    module = _module()
    masks = np.zeros((12, 2), dtype=bool)
    masks[[0, 1, 2, 3, 4, 5], 0] = True
    masks[[0, 1, 6, 7], 1] = True
    result = module.summarize_observation_candidate_support(
        [np.asarray([0, 1, 2]), np.asarray([2, 3, 4]), np.asarray([4, 5, 8])], masks, min_shared_points=2
    )
    assert result["top_candidate_id"] == 0
    assert result["top_candidate_support_view_count"] == 3
    assert result["second_candidate_support_view_count"] == 1
    assert result["candidate_identity_margin"] > 0

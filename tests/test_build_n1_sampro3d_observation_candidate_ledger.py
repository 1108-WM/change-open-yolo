import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_n1_sampro3d_observation_candidate_ledger.py"
    spec = importlib.util.spec_from_file_location("n1_candidate_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lift_keeps_only_visible_and_supported_original_superpoints():
    module = _module()
    superpoints = np.array([1, 1, 1, 2, 2, 3])
    lifted, evidence = module.lift_visible_observation_to_superpoints(
        point_indices=np.array([0, 1, 3, 5]),
        superpoints=superpoints,
        superpoint_sizes={1: 3, 2: 2, 3: 1},
        visible_counts={1: 2, 2: 2, 3: 1},
        min_visible_ratio=0.5,
        min_mask_support=0.5,
    )
    assert lifted == [1, 2, 3]
    assert evidence[0]["mask_support_ratio"] == 1.0


def test_lift_rejects_low_visibility_and_low_mask_support_without_dropping_evidence():
    module = _module()
    superpoints = np.array([1, 1, 1, 1, 2, 2])
    lifted, evidence = module.lift_visible_observation_to_superpoints(
        point_indices=np.array([0, 4]),
        superpoints=superpoints,
        superpoint_sizes={1: 4, 2: 2},
        visible_counts={1: 4, 2: 1},
        min_visible_ratio=0.75,
        min_mask_support=0.5,
    )
    assert lifted == []
    assert [row["accepted"] for row in evidence] == [False, False]

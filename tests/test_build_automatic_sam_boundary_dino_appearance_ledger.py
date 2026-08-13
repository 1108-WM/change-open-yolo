import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_boundary_dino_appearance_ledger.py"
    spec = importlib.util.spec_from_file_location("boundary_dino_appearance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rle_round_trip_and_cosine_are_deterministic():
    module = _module()
    payload = {"size": [2, 3], "counts": [1, 2, 1, 2]}
    assert np.array_equal(module.decode_binary_mask_rle(payload), np.asarray([[False, True, True], [True, False, True]]))
    assert module.cosine(np.asarray([1., 0.]), np.asarray([1., 0.])) == 1.0


def test_only_planned_boundary_track_pairs_are_retained():
    module = _module()
    plans = [{"left_source_track_id": 2, "right_source_track_id": 5, "variant_plan_state": "boundary_competition_variant_hypothesis_keep_originals"}]
    nodes = [{"observation_id": 10, "existing_track_ids": [2]}, {"observation_id": 12, "existing_track_ids": [5]}]
    relations = [{"left_observation_id": 10, "right_observation_id": 12}]
    retained, pairs = module.matching_observation_pairs(plans, nodes, relations)
    assert list(retained) == [(2, 5)]
    assert pairs[(2, 5)] == relations

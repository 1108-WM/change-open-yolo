import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "refine_details_consensus_all_view_reobservation.py"
    spec = importlib.util.spec_from_file_location("details_all_view", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_match_metrics_uses_only_visible_superpoints():
    module = _module()
    metrics = module.frame_match_metrics(
        np.asarray([1, 2, 9]),
        np.asarray([1, 3, 9]),
        {1: 10, 2: 10, 3: 10, 9: 1},
        min_visible_points=3,
    )
    assert metrics == {
        "siou": 1 / 3,
        "core_coverage": 0.5,
        "observation_purity": 0.5,
    }


def test_reciprocal_matching_prevents_one_observation_serving_two_tracks():
    module = _module()
    tracks = {
        1: {"frame_index": 4, "member_frame_indices": set(), "core_superpoints": np.asarray([1, 2])},
        2: {"frame_index": 4, "member_frame_indices": set(), "core_superpoints": np.asarray([1, 3])},
    }
    observations = [{
        "observation_id": 8,
        "lifted_superpoints": np.asarray([1, 2]),
        "visible_counts": {1: 10, 2: 10, 3: 10},
        "quality": 0.9,
    }]
    matches = module.reciprocal_frame_matches(tracks, observations, set(), 0.30, 3)
    assert [(item["track_id"], item["observation_id"]) for item in matches] == [(1, 8)]


def test_reciprocal_matching_skips_owned_and_existing_track_frames():
    module = _module()
    tracks = {
        1: {"frame_index": 4, "member_frame_indices": {4}, "core_superpoints": np.asarray([1])},
        2: {"frame_index": 4, "member_frame_indices": set(), "core_superpoints": np.asarray([2])},
    }
    observations = [
        {"observation_id": 8, "lifted_superpoints": np.asarray([2]), "visible_counts": {2: 10}, "quality": 0.9},
        {"observation_id": 9, "lifted_superpoints": np.asarray([2]), "visible_counts": {2: 10}, "quality": 0.8},
    ]
    matches = module.reciprocal_frame_matches(tracks, observations, {8}, 0.30, 3)
    assert [(item["track_id"], item["observation_id"]) for item in matches] == [(2, 9)]


def test_all_view_consistency_keeps_unmatched_visible_frame_as_zero():
    module = _module()
    observations = {
        0: [{
            "lifted_superpoints": np.asarray([1]),
            "visible_counts": {1: 10},
        }],
        1: [{
            "lifted_superpoints": np.asarray([2]),
            "visible_counts": {1: 10, 2: 10},
        }],
        2: [{
            "lifted_superpoints": np.asarray([1]),
            "visible_counts": {1: 1},
        }],
    }
    result = module.all_view_consistency({1}, observations, min_visible_points=3, support_siou=0.30)
    assert result["visible_frame_count"] == 2
    assert result["mean_best_siou"] == 0.5
    assert result["support_frame_rate"] == 0.5


def test_proposal_must_pareto_dominate_base_consistency():
    module = _module()
    base = {"mean_best_siou": 0.4, "support_frame_rate": 0.5}
    assert module.proposal_dominates_base(
        base,
        {"mean_best_siou": 0.45, "support_frame_rate": 0.5},
    )
    assert not module.proposal_dominates_base(
        base,
        {"mean_best_siou": 0.45, "support_frame_rate": 0.4},
    )
    assert not module.proposal_dominates_base(base, dict(base))

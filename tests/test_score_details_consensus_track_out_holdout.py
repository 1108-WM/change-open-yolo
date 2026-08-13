import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "score_details_consensus_track_out_holdout.py"
    spec = importlib.util.spec_from_file_location("details_track_out", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(lifted, visible):
    return {
        "lifted_superpoints": np.asarray(lifted, dtype=np.int64),
        "visible_counts": visible,
    }


def test_paired_holdout_excludes_track_frames_and_keeps_zero_observation_frame():
    module = _module()
    result = module.paired_track_out_consistency(
        base_superpoints=[1],
        candidate_superpoints=[1, 2],
        selected_frame_indices=[0, 1, 2],
        excluded_frame_indices=[0],
        observations_by_frame={
            0: [_observation([1], {1: 10, 2: 10})],
            1: [_observation([1, 2], {1: 10, 2: 10})],
        },
        visible_counts_by_frame={
            0: {1: 10, 2: 10},
            1: {1: 10, 2: 10},
            2: {1: 10, 2: 10},
        },
        min_visible_points=3,
        support_siou=0.30,
    )
    assert [row["frame_index"] for row in result["frames"]] == [1, 2]
    assert result["zero_observation_frame_count"] == 1
    assert result["base"] == {"mean_best_siou": 0.25, "support_frame_rate": 0.5}
    assert result["candidate"] == {"mean_best_siou": 0.5, "support_frame_rate": 0.5}


def test_paired_holdout_uses_only_common_visible_domain():
    module = _module()
    result = module.paired_track_out_consistency(
        base_superpoints=[1],
        candidate_superpoints=[2],
        selected_frame_indices=[0, 1, 2],
        excluded_frame_indices=[],
        observations_by_frame={},
        visible_counts_by_frame={
            0: {1: 10},
            1: {2: 10},
            2: {1: 10, 2: 10},
        },
        min_visible_points=3,
        support_siou=0.30,
    )
    assert result["eligible_frame_count"] == 1
    assert result["frames"][0]["frame_index"] == 2


def test_holdout_result_is_invariant_to_input_order():
    module = _module()
    kwargs = {
        "base_superpoints": [2, 1],
        "candidate_superpoints": [3, 1],
        "selected_frame_indices": [2, 0, 1],
        "excluded_frame_indices": [1],
        "observations_by_frame": {
            0: [_observation([1, 3], {1: 10, 2: 10, 3: 10})],
            2: [_observation([1, 2], {1: 10, 2: 10, 3: 10})],
        },
        "visible_counts_by_frame": {
            0: {1: 10, 2: 10, 3: 10},
            1: {1: 10, 2: 10, 3: 10},
            2: {1: 10, 2: 10, 3: 10},
        },
        "min_visible_points": 3,
        "support_siou": 0.30,
    }
    forward = module.paired_track_out_consistency(**kwargs)
    reversed_inputs = dict(kwargs)
    reversed_inputs["base_superpoints"] = list(reversed(kwargs["base_superpoints"]))
    reversed_inputs["candidate_superpoints"] = list(reversed(kwargs["candidate_superpoints"]))
    reversed_inputs["selected_frame_indices"] = list(reversed(kwargs["selected_frame_indices"]))
    reverse = module.paired_track_out_consistency(**reversed_inputs)
    assert forward == reverse


def test_quality_falls_back_when_independent_views_are_insufficient():
    module = _module()
    insufficient = {
        "eligible_frame_count": 1,
        "candidate": {"mean_best_siou": 0.25, "support_frame_rate": 1.0},
    }
    quality, mode = module.track_out_holdout_quality(0.81, insufficient, 2)
    assert quality == 0.81
    assert mode == "mean_node_quality_fallback"

    sufficient = {
        "eligible_frame_count": 2,
        "candidate": {"mean_best_siou": 0.25, "support_frame_rate": 0.5},
    }
    quality, mode = module.track_out_holdout_quality(0.81, sufficient, 2)
    assert quality == 0.45
    assert mode == "track_out_geometric_mean"

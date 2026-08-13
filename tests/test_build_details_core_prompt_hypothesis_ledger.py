import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_details_core_prompt_hypothesis_ledger.py"
    spec = importlib.util.spec_from_file_location("details_core_prompt_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lift_prompt_points_respects_visibility_and_mask_support():
    module = _module()
    superpoints = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    lifted = module.lift_prompt_points_to_superpoints(
        point_indices=[0, 1, 4],
        superpoints=superpoints,
        superpoint_sizes={1: 4, 2: 4},
        visible_counts={1: 4, 2: 4},
        min_visible_ratio=0.1,
        min_mask_support=0.3,
    )
    assert lifted == [1]


def test_track_aggregation_counts_distinct_frames_and_three_hypothesis_intersection():
    module = _module()
    rows = [
        {"frame_index": 0, "lifted_superpoints": [1, 2, 3]},
        {"frame_index": 0, "lifted_superpoints": [1, 2]},
        {"frame_index": 0, "lifted_superpoints": [1, 2, 4]},
        {"frame_index": 1, "lifted_superpoints": [1, 2, 5]},
        {"frame_index": 1, "lifted_superpoints": [1, 2]},
        {"frame_index": 1, "lifted_superpoints": [1, 2, 6]},
    ]
    result = module.aggregate_track_hypotheses(rows, base_superpoints=[1], min_support_frames=2)
    assert result["stable_new_superpoints_any_hypothesis"] == [2]
    assert result["stable_new_superpoints_all_hypotheses"] == [2]
    any_support = {row["superpoint_id"]: row["support_frame_count"] for row in result["new_superpoint_support_any_hypothesis"]}
    assert any_support == {2: 2, 3: 1, 4: 1, 5: 1, 6: 1}


def test_track_aggregation_does_not_count_three_hypotheses_as_three_frames():
    module = _module()
    rows = [
        {"frame_index": 0, "lifted_superpoints": [1, 2]},
        {"frame_index": 0, "lifted_superpoints": [1, 2]},
        {"frame_index": 0, "lifted_superpoints": [1, 2]},
    ]
    result = module.aggregate_track_hypotheses(rows, base_superpoints=[1], min_support_frames=2)
    assert result["stable_new_superpoints_any_hypothesis"] == []
    assert result["stable_new_superpoints_all_hypotheses"] == []

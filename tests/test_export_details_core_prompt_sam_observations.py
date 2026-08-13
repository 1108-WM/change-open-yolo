import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_details_core_prompt_sam_observations.py"
    spec = importlib.util.spec_from_file_location("details_core_prompt_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_prompt_requests_is_deterministic_and_rejects_duplicates():
    module = _module()
    plans = [
        {
            "track_id": 2,
            "prompt_point_index": 20,
            "prompt_superpoint_id": 5,
            "common_core_superpoint_ids": [5],
            "prompt_frames": [{
                "frame_id": "10", "frame_index": 1, "prompt_xy": [2, 3],
                "visible_common_core_point_count": 4,
            }],
        },
        {
            "track_id": 1,
            "prompt_point_index": 10,
            "prompt_superpoint_id": 4,
            "common_core_superpoint_ids": [4],
            "prompt_frames": [{
                "frame_id": "10", "frame_index": 1, "prompt_xy": [1, 3],
                "visible_common_core_point_count": 6,
            }],
        },
    ]
    grouped = module.group_prompt_requests(plans)
    assert list(grouped) == [1]
    assert [row["track_id"] for row in grouped[1]] == [1, 2]

    duplicate = [plans[0], dict(plans[0])]
    with pytest.raises(ValueError, match="重复提示请求"):
        module.group_prompt_requests(duplicate)


def test_hypothesis_core_metrics_preserves_support_purity_and_prompt_membership():
    module = _module()
    metrics = module.hypothesis_core_metrics(
        point_indices=[2, 3, 4, 8],
        visible_core_points=[1, 2, 3],
        prompt_point_index=2,
    )
    assert metrics == {
        "visible_core_point_count": 3,
        "visible_core_covered_point_count": 2,
        "visible_core_support_ratio": 2 / 3,
        "observation_core_purity_ratio": 0.5,
        "prompt_point_backprojected": True,
    }


def test_hypothesis_core_metrics_handles_empty_observation():
    module = _module()
    metrics = module.hypothesis_core_metrics([], [1, 2], 1)
    assert metrics["visible_core_support_ratio"] == 0.0
    assert metrics["observation_core_purity_ratio"] == 0.0
    assert not metrics["prompt_point_backprojected"]

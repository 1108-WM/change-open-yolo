import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_multiview_fragment_family_quality_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("fragment_family_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(observation_id, frame_index, lifted, visible):
    return {
        "observation_id": observation_id,
        "frame_id": str(frame_index * 10),
        "frame_index": frame_index,
        "lifted_superpoints": np.asarray(lifted, dtype=np.int64),
        "visible_counts": visible,
    }


def test_joint_dominance_uses_common_nonbridge_frames_only():
    module = _module()
    visible = {1: 5, 2: 5}
    observations = {
        1: [_observation(10, 1, [1, 2], visible)],
        2: [_observation(20, 2, [1, 2], visible)],
        3: [_observation(30, 3, [1], visible)],
    }
    result = module.compare_family_on_nonbridge_frames(
        {"anchor": [1], "absorbed": [2], "merged": [1, 2]},
        observations,
        {1: visible, 2: visible, 3: visible},
        {"10": 1, "20": 2, "30": 3},
        ["30"],
    )
    assert result["common_nonbridge_visible_frame_count"] == 2
    assert result["quality_evidence_reliable"] is True
    assert result["candidate_metrics"]["merged"]["mean_best_siou"] == 1.0
    assert result["candidate_metrics"]["anchor"]["mean_best_siou"] == 0.5
    assert result["candidate_metrics"]["absorbed"]["mean_best_siou"] == 0.5
    assert result["merged_jointly_dominates"] is True


def test_insufficient_nonbridge_frames_never_produce_dominance():
    module = _module()
    visible = {1: 5, 2: 5}
    result = module.compare_family_on_nonbridge_frames(
        {"anchor": [1], "absorbed": [2], "merged": [1, 2]},
        {1: [_observation(10, 1, [1, 2], visible)]},
        {1: visible},
        {"10": 1},
        [],
    )
    assert result["common_nonbridge_visible_frame_count"] == 1
    assert result["quality_evidence_reliable"] is False
    assert result["merged_dominates_anchor"] is False
    assert result["merged_dominates_absorbed"] is False


def test_relative_observation_lifting_reuses_frozen_d1_contract():
    module = _module()
    observations = [{
        "observation_id": 1,
        "frame_id": "20",
        "frame_index": 2,
        "inside_counts": {1: 3, 2: 2, 3: 9},
    }]
    lifted, frame_map = module.lift_relative_observations(
        observations,
        {2: {1: 10, 2: 10, 3: 10}},
        {1: 100, 2: 100, 3: 200},
    )
    # 1 passes both exact boundaries; 2 fails mask support; 3 fails visibility.
    assert lifted[2][0]["lifted_superpoints"].tolist() == [1]
    assert frame_map == {"20": 2}


def test_cli_has_no_gt_native_semantic_score_or_threshold_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "native", "semantic", "class", "score", "threshold")
    assert not any(any(token in option for token in forbidden) for option in options)

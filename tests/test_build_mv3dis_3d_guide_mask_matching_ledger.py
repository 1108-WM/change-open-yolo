import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_mv3dis_3d_guide_mask_matching_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("mv3dis_matching", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proposal(proposal_id, superpoint_id, point_count=10):
    return {
        "proposal_id": proposal_id,
        "superpoint_ids": [superpoint_id],
        "point_count": point_count,
        "lineage_proposal_ids": [proposal_id],
    }


def test_paper_thresholds_are_strictly_greater_than():
    module = _load_module()
    proposals = [_proposal(10, 1), _proposal(20, 2)]
    observations = [
        {"observation_id": 1, "frame_index": 0, "inside_counts": {1: 3}},
        {"observation_id": 2, "frame_index": 1, "inside_counts": {2: 9}},
        {"observation_id": 3, "frame_index": 2, "inside_counts": {1: 4}},
    ]
    visible = {
        0: {1: 3, 2: 10},
        1: {1: 10, 2: 10},
        2: {1: 4, 2: 10},
    }
    ledger = module.build_guide_mask_matching(
        proposals, observations, visible, {1: 10, 2: 10}
    )
    matches = [
        (row["guide_proposal_id"], row["observation_id"])
        for row in ledger["match_rows"]
    ]
    assert matches == [(10, 3)]
    assert ledger["match_rows"][0]["frame_visibility"] > 0.3
    assert ledger["match_rows"][0]["mask_visibility"] > 0.9


def test_coverage_vectors_and_pairwise_cosine_are_exact():
    module = _load_module()
    proposals = [_proposal(10, 1), _proposal(20, 2)]
    observations = [
        {
            "observation_id": 1,
            "frame_id": "a",
            "frame_index": 0,
            "inside_counts": {1: 10},
        },
        {
            "observation_id": 2,
            "frame_id": "b",
            "frame_index": 1,
            "inside_counts": {1: 10, 2: 10},
        },
    ]
    visible = {0: {1: 10, 2: 10}, 1: {1: 10, 2: 10}}
    ledger = module.build_guide_mask_matching(
        proposals, observations, visible, {1: 10, 2: 10}
    )
    rows = {row["observation_id"]: row for row in ledger["mask_rows"]}
    assert rows[1]["coverage_vector_sparse"] == [
        {"proposal_id": 10, "value": 1.0}
    ]
    assert rows[2]["coverage_vector_sparse"] == [
        {"proposal_id": 10, "value": 1.0},
        {"proposal_id": 20, "value": 1.0},
    ]
    expected = 1.0 / np.sqrt(2.0)
    assert np.isclose(rows[1]["final_consistency_score"], expected)
    guide10 = [
        row for row in ledger["match_rows"] if row["guide_proposal_id"] == 10
    ]
    assert all(np.isclose(row["guide_consistency_score"], expected) for row in guide10)


def test_continuous_depth_weights_scale_coverage_without_changing_visibility_match():
    module = _load_module()
    proposals = [_proposal(10, 1)]
    observations = [{
        "observation_id": 1,
        "frame_index": 0,
        "inside_counts": {1: 10},
        "inside_weight_sums": {1: 6.0},
    }]
    ledger = module.build_guide_mask_matching(
        proposals, observations, {0: {1: 10}}, {1: 10}
    )
    assert len(ledger["match_rows"]) == 1
    assert ledger["mask_rows"][0]["coverage_vector_sparse"] == [
        {"proposal_id": 10, "value": 0.6}
    ]
    match = ledger["match_rows"][0]
    assert match["mask_visibility"] == 1.0
    assert match["guide_mean_inside_mask_depth_weight"] == 0.6
    assert match["guide_coverage_value"] == 0.6


def test_mask_score_averages_defined_scores_across_multiple_guides():
    module = _load_module()
    proposals = [_proposal(10, 1), _proposal(20, 2)]
    observations = [
        {"observation_id": 1, "frame_index": 0, "inside_counts": {1: 10, 2: 10}},
        {"observation_id": 2, "frame_index": 1, "inside_counts": {1: 10}},
        {"observation_id": 3, "frame_index": 2, "inside_counts": {2: 10}},
    ]
    visible = {
        0: {1: 10, 2: 10},
        1: {1: 10, 2: 10},
        2: {1: 10, 2: 10},
    }
    ledger = module.build_guide_mask_matching(
        proposals, observations, visible, {1: 10, 2: 10}
    )
    row = next(item for item in ledger["mask_rows"] if item["observation_id"] == 1)
    expected = 1.0 / np.sqrt(2.0)
    assert row["matched_guide_proposal_ids"] == [10, 20]
    assert row["defined_guide_score_count"] == 2
    assert np.isclose(row["final_consistency_score"], expected)


def test_empty_and_single_candidate_guides_have_explicit_fallback_states():
    module = _load_module()
    proposals = [_proposal(10, 1), _proposal(20, 2), _proposal(30, 3)]
    observations = [
        {"observation_id": 5, "frame_index": 0, "inside_counts": {1: 10}}
    ]
    visible = {0: {1: 10, 2: 10, 3: 10}}
    ledger = module.build_guide_mask_matching(
        proposals, observations, visible, {1: 10, 2: 10, 3: 10}
    )
    states = {
        row["proposal_id"]: row["consistency_state"]
        for row in ledger["guide_rows"]
    }
    assert states == {
        10: "single_candidate_consistency_undefined",
        20: "no_candidate",
        30: "no_candidate",
    }
    assert ledger["mask_rows"][0]["final_consistency_score"] is None
    assert (
        ledger["mask_rows"][0]["final_consistency_state"]
        == "undefined_no_guide_with_multiple_candidates"
    )


def test_result_is_independent_of_proposal_and_observation_input_order():
    module = _load_module()
    proposals = [_proposal(10, 1), _proposal(20, 2)]
    observations = [
        {"observation_id": 1, "frame_index": 0, "inside_counts": {1: 10}},
        {"observation_id": 2, "frame_index": 1, "inside_counts": {1: 10, 2: 10}},
    ]
    visible = {0: {1: 10, 2: 10}, 1: {1: 10, 2: 10}}
    forward = module.build_guide_mask_matching(
        proposals, observations, visible, {1: 10, 2: 10}
    )
    reverse = module.build_guide_mask_matching(
        list(reversed(proposals)),
        list(reversed(observations)),
        visible,
        {2: 10, 1: 10},
    )
    assert forward == reverse


def test_invalid_lineage_or_nonvisible_mask_support_is_rejected():
    module = _load_module()
    proposals = [
        _proposal(10, 1),
        {
            **_proposal(20, 2),
            "lineage_proposal_ids": [10],
        },
    ]
    try:
        module.build_guide_mask_matching(
            proposals, [], {}, {1: 10, 2: 10}
        )
    except ValueError as error:
        assert "lineage" in str(error)
    else:
        raise AssertionError("duplicate lineage must be rejected")

    try:
        module.build_guide_mask_matching(
            [_proposal(10, 1)],
            [{"observation_id": 1, "frame_index": 0, "inside_counts": {1: 6}}],
            {0: {1: 5}},
            {1: 10},
        )
    except ValueError as error:
        assert "non-visible" in str(error)
    else:
        raise AssertionError("non-visible support must be rejected")


def test_cli_exposes_no_gt_ap_semantic_or_matching_threshold_options():
    module = _load_module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("ground-truth", "--gt", "--ap", "class", "semantic", "threshold")
    assert not any(any(token in option for token in forbidden) for option in options)

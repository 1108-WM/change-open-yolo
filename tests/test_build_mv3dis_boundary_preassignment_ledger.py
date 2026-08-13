import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "build_mv3dis_boundary_preassignment_ledger.py"
    spec = importlib.util.spec_from_file_location("mv3dis_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_point_resolution_requires_defined_unique_highest_score():
    module = _load_module()
    masks = [
        {"observation_id": 1, "frame_index": 0, "final_consistency_score": 0.8},
        {"observation_id": 2, "frame_index": 0, "final_consistency_score": 0.9},
        {"observation_id": 3, "frame_index": 0, "final_consistency_score": None},
        {"observation_id": 4, "frame_index": 0, "final_consistency_score": 0.9},
    ]
    points = {1: [0, 1], 2: [0, 2, 3], 3: [1], 4: [3]}
    resolved, rows = module.resolve_pre_nms_point_labels(masks, points)
    assert resolved == {0: {0: 2, 2: 2}}
    assert rows[0]["unknown_undefined_score_point_count"] == 1
    assert rows[0]["unknown_exact_top_score_tie_point_count"] == 1


def test_histogram_cosine_matches_mv3dis_equation_six():
    module = _load_module()
    assert np.isclose(module.histogram_cosine({1: 1}, {1: 1, 2: 1}), 1 / np.sqrt(2))
    assert module.histogram_cosine({}, {1: 1}) is None


def test_pair_affinity_uses_visibility_weighted_frame_mean():
    module = _load_module()
    contact = {(1, 2): {"boundary_contact_count": 3, "boundary_contact_ratio": 0.1}}
    histograms = {
        (0, 1): {7: 1}, (0, 2): {7: 1},
        (1, 1): {7: 1}, (1, 2): {8: 1},
    }
    visible = {0: {1: 10, 2: 10}, 1: {1: 5, 2: 10}}
    rows = module.build_pair_affinity_rows(contact, histograms, visible, {1: 10, 2: 10})
    assert len(rows) == 1
    assert np.isclose(rows[0]["affinity"], 2 / 3)
    assert rows[0]["defined_frame_count"] == 2


def test_pair_affinity_includes_continuous_depth_weight_product():
    module = _load_module()
    contact = {(1, 2): {"boundary_contact_count": 3, "boundary_contact_ratio": 0.1}}
    histograms = {
        (0, 1): {7: 1}, (0, 2): {7: 1},
        (1, 1): {7: 1}, (1, 2): {8: 1},
    }
    visible = {0: {1: 10, 2: 10}, 1: {1: 5, 2: 10}}
    depth_means = {0: {1: 0.5, 2: 0.5}, 1: {1: 1.0, 2: 1.0}}
    rows = module.build_pair_affinity_rows(
        contact,
        histograms,
        visible,
        {1: 10, 2: 10},
        frame_depth_weight_means=depth_means,
        depth_weight_adapter="relative",
    )
    assert np.isclose(rows[0]["affinity"], 1 / 3)
    assert rows[0]["depth_weight_adapter"] == "relative"
    assert rows[0]["frame_evidence"][0]["depth_weight_product"] == 0.25


def test_boundary_competition_records_owners_without_assignment():
    module = _load_module()
    proposals = [
        {"proposal_id": 10, "superpoint_ids": [1, 2]},
        {"proposal_id": 20, "superpoint_ids": [2, 3]},
    ]
    contact = {
        (1, 2): {"boundary_contact_count": 3, "boundary_contact_ratio": 0.1},
        (2, 3): {"boundary_contact_count": 3, "boundary_contact_ratio": 0.1},
    }
    affinities = [
        {"left_superpoint_id": 1, "right_superpoint_id": 2, "affinity": 0.8},
        {"left_superpoint_id": 2, "right_superpoint_id": 3, "affinity": 0.6},
    ]
    rows = module.build_boundary_competition_rows(proposals, contact, affinities)
    by_id = {row["superpoint_id"]: row for row in rows}
    assert by_id[1]["competition_state"] == "owned_boundary_competition"
    assert by_id[2]["competition_state"] == "current_multi_owner_conflict"
    assert by_id[3]["competition_state"] == "owned_boundary_competition"
    assert all(row["assignment_action"] == "none_preassignment_ledger_only" for row in rows)
    assert all(row["unknown_allowed"] for row in rows)


def test_cli_has_no_threshold_gt_ap_semantic_or_assignment_switch():
    module = _load_module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("threshold", "--gt", "--ap", "class", "semantic", "assign", "nms")
    assert not any(any(token in option for token in forbidden) for option in options)

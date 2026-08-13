import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "diagnose_mv3dis_boundary_action_oracle_gt.py"
    )
    spec = importlib.util.spec_from_file_location("boundary_action_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_is_exact_noop_and_assignment_records_empty_proposal_risk():
    module = _module()
    tracks = {
        10: {"proposal_id": 10, "superpoint_ids": frozenset({1}), "track_score": 0.4},
        20: {"proposal_id": 20, "superpoint_ids": frozenset({2}), "track_score": 0.7},
    }
    points_by_superpoint = {
        1: np.asarray([0, 1], dtype=np.int64),
        2: np.asarray([2, 3], dtype=np.int64),
    }
    gt_ids = np.asarray([1001, 1001, 2001, 2001], dtype=np.int64)
    gt_instances = {1001: 2, 2001: 2}
    native_masks = np.asarray([[1], [1], [0], [0]], dtype=bool)
    native_scores = np.asarray([0.9], dtype=np.float64)
    plan = {
        "superpoint_id": 1,
        "current_owner_proposal_ids": [10],
        "adjacent_candidate_proposal_ids": [10, 20],
        "planned_action": module.ASSIGN_ACTION,
        "planned_owner_proposal_id": 20,
        "competition_state": "owned_boundary_competition",
        "evidence_state": "unique_complete_mean_min_pareto_owner",
        "assignment_applied": False,
        "proposal_mutation_applied": False,
        "ground_truth_usage": "none",
    }
    rows = module.diagnose_scene(
        "scene_unit", [plan], tracks, points_by_superpoint, gt_ids, gt_instances,
        native_masks, native_scores,
    )
    assert len(rows) == 1
    actions = {row["action_name"]: row for row in rows[0]["action_results"]}
    unknown = actions[module.UNKNOWN_NOOP_ACTION]
    assert unknown["action_owner_proposal_ids"] == [10]
    assert unknown["changes_ownership"] is False
    assert all(
        outcome["fixed_gt_iou_delta"] == 0.0
        for outcome in unknown["proposal_outcomes"]
    )
    assigned = actions["assign_to_proposal_20"]
    assert assigned["action_owner_proposal_ids"] == [20]
    assert assigned["locally_feasible_no_empty_proposal"] is False
    proposal_10 = next(
        row for row in assigned["proposal_outcomes"] if row["proposal_id"] == 10
    )
    assert proposal_10["would_be_empty"] is True
    assert proposal_10["fixed_gt_iou_delta"] == -1.0


def test_multi_candidate_boundary_keeps_every_candidate_counterfactual():
    module = _module()
    tracks = {
        proposal_id: {
            "proposal_id": proposal_id,
            "superpoint_ids": frozenset({proposal_id}),
            "track_score": 0.5,
        }
        for proposal_id in (1, 2, 3)
    }
    points_by_superpoint = {
        1: np.asarray([0], dtype=np.int64),
        2: np.asarray([1], dtype=np.int64),
        3: np.asarray([2], dtype=np.int64),
    }
    gt_ids = np.asarray([1001, 1001, 1001], dtype=np.int64)
    plan = {
        "superpoint_id": 1,
        "current_owner_proposal_ids": [1],
        "adjacent_candidate_proposal_ids": [1, 2, 3],
        "planned_action": module.ASSIGN_ACTION,
        "planned_owner_proposal_id": 2,
        "competition_state": "owned_boundary_competition",
        "evidence_state": "unique_complete_mean_min_pareto_owner",
        "assignment_applied": False,
        "proposal_mutation_applied": False,
        "ground_truth_usage": "none",
    }
    rows = module.diagnose_scene(
        "scene_unit", [plan], tracks, points_by_superpoint, gt_ids, {1001: 3},
        np.ones((3, 1), dtype=bool), np.asarray([0.2]),
    )
    assert rows[0]["action_result_count"] == 4
    assert {
        row["action_name"] for row in rows[0]["action_results"]
    } == {
        module.UNKNOWN_NOOP_ACTION,
        "assign_to_proposal_1",
        "assign_to_proposal_2",
        "assign_to_proposal_3",
    }


def test_oracle_without_native_cache_marks_annotation_unavailable():
    module = _module()
    tracks = {
        1: {"proposal_id": 1, "superpoint_ids": frozenset({1}), "track_score": .5},
        2: {"proposal_id": 2, "superpoint_ids": frozenset({2}), "track_score": .5},
    }
    plan = {
        "superpoint_id": 1, "current_owner_proposal_ids": [1],
        "adjacent_candidate_proposal_ids": [1, 2], "planned_action": module.ASSIGN_ACTION,
        "planned_owner_proposal_id": 2, "competition_state": "owned_boundary_competition",
        "evidence_state": "unique_complete_mean_min_pareto_owner", "assignment_applied": False,
        "proposal_mutation_applied": False, "ground_truth_usage": "none",
    }
    rows = module.diagnose_scene(
        "scene_unit", [plan], tracks, {1: np.asarray([0]), 2: np.asarray([1])},
        np.asarray([1001, 1001]), {1001: 2}, None, None,
    )
    assigned = next(item for item in rows[0]["action_results"] if item["action_kind"] == "assign_owner")
    outcome = next(item for item in assigned["proposal_outcomes"] if item["fixed_gt_available"])
    assert outcome["native_coverage_annotation_available"] is False
    assert outcome["best_native_iou"] is None

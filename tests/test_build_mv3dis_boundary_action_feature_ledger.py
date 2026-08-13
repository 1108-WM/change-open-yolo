import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_mv3dis_boundary_action_feature_ledger.py"
    spec = importlib.util.spec_from_file_location("boundary_action_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_rows_are_no_gt_and_keep_unknown_as_noop():
    module = _module()
    plan = [{
        "superpoint_id": 3,
        "planned_action": module.ASSIGN_ACTION,
        "planned_owner_proposal_id": 20,
        "adjacent_candidate_proposal_ids": [10, 20],
        "current_owner_proposal_ids": [10],
        "competition_state": "owned_boundary_competition",
        "evidence_state": "unique_complete_mean_min_pareto_owner",
        "candidate_affinity_summaries": [
            {"proposal_id": 10, "complete_edge_evidence": True},
            {"proposal_id": 20, "complete_edge_evidence": True},
        ],
        "ground_truth_usage": "none",
        "assignment_applied": False,
        "proposal_mutation_applied": False,
    }]
    pre = [{
        "superpoint_id": 3,
        "adjacent_candidate_proposal_ids": [10, 20],
        "current_owner_proposal_ids": [10],
        "gt_usage": "none",
        "candidate_region_evidence": [
            {"proposal_id": 10, "neighbor_pair_count": 1, "defined_pair_affinity_count": 1,
             "observed_pair_affinity_mean": 0.4, "pair_evidence": [{"pair_affinity": 0.4}]},
            {"proposal_id": 20, "neighbor_pair_count": 1, "defined_pair_affinity_count": 1,
             "observed_pair_affinity_mean": 0.8, "pair_evidence": [{"pair_affinity": 0.8}]},
        ],
    }]
    tracks = {
        10: {"proposal_id": 10, "superpoint_ids": frozenset({1, 3}), "point_count": 5,
             "support_view_count": 2, "observation_count": 2, "node_count": 2,
             "mean_node_quality": .5, "mean_consensus_rate": .6, "mean_supported_coverage": .7},
        20: {"proposal_id": 20, "superpoint_ids": frozenset({2}), "point_count": 4,
             "support_view_count": 3, "observation_count": 3, "node_count": 3,
             "mean_node_quality": .8, "mean_consensus_rate": .9, "mean_supported_coverage": 1.0},
    }
    context = {
        "raw_ids": np.asarray([1, 2, 3]), "sizes": np.asarray([3, 4, 2]),
        "neighbors": {3: [
            {"neighbor_superpoint_id": 1, "boundary_contact_count": 5, "boundary_contact_ratio": .5,
             "mean_boundary_distance": .01, "mean_normal_difference": .1, "mean_color_difference": .2},
            {"neighbor_superpoint_id": 2, "boundary_contact_count": 3, "boundary_contact_ratio": .3,
             "mean_boundary_distance": .02, "mean_normal_difference": .3, "mean_color_difference": .4},
        ]},
    }
    boundaries, actions = module.build_feature_rows("scene_unit", plan, pre, tracks, context)
    assert boundaries[0]["unknown_action_name"] == module.UNKNOWN_NOOP_ACTION
    assert boundaries[0]["ground_truth_usage"] == "none"
    assert len(actions) == 2
    by_owner = {row["candidate_owner_proposal_id"]: row for row in actions}
    assert by_owner[20]["is_frozen_planned_owner"] is True
    assert by_owner[20]["would_add_boundary_to_candidate"] is True
    assert by_owner[10]["would_remove_boundary_from_other_current_owner_count"] == 0
    assert by_owner[20]["direct_core_contact_count_sum"] == 3
    assert by_owner[20]["affinity_mean_minus_best_other"] == 0.4
    assert all(row["proposal_materialization_applied"] is False for row in actions)

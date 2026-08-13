import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_mv3dis_global_boundary_assignment_plan.py"
    )
    spec = importlib.util.spec_from_file_location("global_boundary_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(proposal_id, affinities):
    return {
        "proposal_id": proposal_id,
        "pair_evidence": [
            {"neighbor_superpoint_id": index, "pair_affinity": value}
            for index, value in enumerate(affinities)
        ],
    }


def _row(owners, candidates, state="owned_boundary_competition"):
    return {
        "superpoint_id": 7,
        "competition_state": state,
        "current_owner_proposal_ids": owners,
        "adjacent_candidate_proposal_ids": [row["proposal_id"] for row in candidates],
        "candidate_region_evidence": candidates,
        "unknown_allowed": True,
        "assignment_action": "none_preassignment_ledger_only",
        "gt_usage": "none",
    }


def test_unique_owner_requires_complete_mean_and_minimum_pareto_dominance():
    module = _module()
    winner, state, summaries = module.unique_complete_pareto_owner([
        _candidate(10, [0.8, 0.9]),
        _candidate(20, [0.4, 0.7]),
    ])
    assert winner == 10
    assert state == "unique_complete_mean_min_pareto_owner"
    assert summaries[0]["minimum_pair_affinity"] == 0.8

    winner, state, _ = module.unique_complete_pareto_owner([
        _candidate(10, [0.5, 0.9]),
        _candidate(20, [0.6, 0.7]),
    ])
    assert winner is None
    assert state == "no_unique_complete_mean_min_pareto_owner"

    winner, state, _ = module.unique_complete_pareto_owner([
        _candidate(10, [0.9]),
        _candidate(20, [None]),
    ])
    assert winner is None
    assert state == "incomplete_candidate_edge_evidence"


def test_winner_is_planned_as_one_owner_across_resolve_move_and_grow_states():
    module = _module()
    candidates = [_candidate(10, [0.9]), _candidate(20, [0.2])]
    cases = (
        ([10, 20], "current_multi_owner_conflict", [20], []),
        ([20], "owned_boundary_competition", [20], [10]),
        ([], "unowned_between_regions", [], [10]),
    )
    for owners, state, removed, added in cases:
        plan = module.plan_boundary_superpoint(_row(owners, candidates, state))
        assert plan["planned_action"] == module.ASSIGN_ACTION
        assert plan["planned_owner_proposal_id"] == 10
        assert plan["remove_from_proposal_ids"] == removed
        assert plan["add_to_proposal_ids"] == added
        assert plan["assignment_applied"] is False


def test_existing_unique_winner_is_keep_and_mixed_evidence_is_unknown():
    module = _module()
    dominant = [_candidate(10, [0.9]), _candidate(20, [0.2])]
    keep = module.plan_boundary_superpoint(_row([10], dominant))
    assert keep["planned_action"] == module.KEEP_ACTION
    assert keep["planned_target_state"] == "unique_candidate_owner"

    mixed = [_candidate(10, [0.5, 0.9]), _candidate(20, [0.6, 0.7])]
    unknown = module.plan_boundary_superpoint(_row([10], mixed))
    assert unknown["planned_action"] == module.UNKNOWN_ACTION
    assert unknown["planned_target_state"] == "unknown"
    assert unknown["planned_owner_proposal_id"] is None
    assert unknown["remove_from_proposal_ids"] == []
    assert unknown["add_to_proposal_ids"] == []


def test_applied_or_gt_bearing_preassignment_is_rejected():
    module = _module()
    candidates = [_candidate(10, [0.9]), _candidate(20, [0.2])]
    for key, value in (
        ("assignment_action", "already_applied"),
        ("gt_usage", "used"),
        ("unknown_allowed", False),
    ):
        row = _row([10], candidates)
        row[key] = value
        try:
            module.plan_boundary_superpoint(row)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid preassignment field was accepted: {key}")


def test_cli_has_no_gt_native_semantic_score_threshold_or_action_family_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = (
        "gt", "native", "semantic", "class", "score", "threshold", "margin",
        "resolve", "move", "grow",
    )
    assert not any(any(token in option for token in forbidden) for option in options)

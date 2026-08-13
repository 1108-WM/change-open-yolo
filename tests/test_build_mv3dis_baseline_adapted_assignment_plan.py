import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "build_mv3dis_baseline_adapted_assignment_plan.py"
    spec = importlib.util.spec_from_file_location("mv3dis_adapted_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(proposal_id, values):
    return {
        "proposal_id": proposal_id,
        "pair_evidence": [
            {"neighbor_superpoint_id": index, "pair_affinity": value}
            for index, value in enumerate(values)
        ],
    }


def _row(state, owners, candidates):
    return {
        "superpoint_id": 7,
        "competition_state": state,
        "current_owner_proposal_ids": owners,
        "adjacent_candidate_proposal_ids": [row["proposal_id"] for row in candidates],
        "candidate_region_evidence": candidates,
    }


def test_strict_dominance_requires_complete_range_separation():
    module = _load_module()
    winner, state, ranges = module.strict_all_edge_dominant_candidate([
        _candidate(10, [0.8, 0.9]),
        _candidate(20, [0.2, 0.7]),
    ])
    assert winner == 10
    assert state == "unique_strict_all_edge_dominance"
    assert ranges[0]["pair_affinity_min"] == 0.8

    winner, state, _ = module.strict_all_edge_dominant_candidate([
        _candidate(10, [0.6, 0.9]),
        _candidate(20, [0.5, 0.7]),
    ])
    assert winner is None
    assert state == "no_unique_strict_all_edge_dominance"

    incomplete = _candidate(20, [None])
    winner, state, _ = module.strict_all_edge_dominant_candidate([
        _candidate(10, [0.9]), incomplete
    ])
    assert winner is None
    assert state == "incomplete_candidate_pair_evidence"


def test_resolve_move_and_grow_are_separate_unapplied_families():
    module = _load_module()
    candidates = [_candidate(10, [0.9]), _candidate(20, [0.2])]
    resolve = module.plan_boundary_row(
        _row("current_multi_owner_conflict", [10, 20], candidates)
    )
    assert resolve["ablation_family"] == "resolve"
    assert resolve["remove_from_proposal_ids"] == [20]
    assert resolve["add_to_proposal_ids"] == []

    move = module.plan_boundary_row(
        _row("owned_boundary_competition", [20], candidates)
    )
    assert move["ablation_family"] == "move"
    assert move["remove_from_proposal_ids"] == [20]
    assert move["add_to_proposal_ids"] == [10]

    grow = module.plan_boundary_row(
        _row("unowned_between_regions", [], candidates)
    )
    assert grow["ablation_family"] == "grow"
    assert grow["remove_from_proposal_ids"] == []
    assert grow["add_to_proposal_ids"] == [10]
    assert all(not row["proposal_mutation_applied"] for row in (resolve, move, grow))


def test_current_dominant_owned_boundary_is_keep_not_move():
    module = _load_module()
    row = module.plan_boundary_row(_row(
        "owned_boundary_competition",
        [10],
        [_candidate(10, [0.9]), _candidate(20, [0.2])],
    ))
    assert row["ablation_family"] == "keep"
    assert row["planned_action"] == "keep_dominant_current_owner"


def test_cli_exposes_no_threshold_or_mutation_family_switch():
    module = _load_module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("threshold", "margin", "resolve", "move", "grow", "--gt", "--ap")
    assert not any(any(token in option for token in forbidden) for option in options)

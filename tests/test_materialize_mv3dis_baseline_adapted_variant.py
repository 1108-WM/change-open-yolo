import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "materialize_mv3dis_baseline_adapted_variant.py"
    )
    spec = importlib.util.spec_from_file_location("mv3dis_materialize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track(proposal_id, superpoints, score):
    return {
        "proposal_id": proposal_id,
        "track_id": proposal_id,
        "lineage_proposal_ids": [proposal_id, proposal_id + 100],
        "superpoint_ids": sorted(superpoints),
        "superpoint_count": len(superpoints),
        "point_count": 10 * len(superpoints),
        "points_path": f"source{proposal_id}.npz",
        "support_score": score,
        "mean_node_quality": score / 10,
        "decision_state": "D2b source",
    }


def _action(family, superpoint_id, owners, remove_from, add_to):
    return {
        "ablation_family": family,
        "superpoint_id": superpoint_id,
        "current_owner_proposal_ids": owners,
        "remove_from_proposal_ids": remove_from,
        "add_to_proposal_ids": add_to,
        "planned_action": f"test_{family}",
        "assignment_applied": False,
        "proposal_mutation_applied": False,
    }


def test_families_are_isolated_and_identity_scores_lineage_are_conserved():
    module = _load_module()
    tracks = [_track(1, [10, 11], 0.8), _track(2, [12], 0.7)]
    plans = [
        _action("move", 11, [1], [1], [2]),
        _action("grow", 13, [], [], [1]),
    ]
    final, actions, ledger = module.materialize_family(
        tracks, plans, {10: 10, 11: 10, 12: 10, 13: 10}, "move"
    )
    assert [row["proposal_id"] for row in final] == [1, 2]
    assert final[0]["superpoint_ids"] == [10]
    assert final[1]["superpoint_ids"] == [11, 12]
    assert final[0]["support_score"] == tracks[0]["support_score"]
    assert final[1]["lineage_proposal_ids"] == tracks[1]["lineage_proposal_ids"]
    assert len(actions) == 1 and actions[0]["ablation_family"] == "move"
    assert actions[0]["assignment_applied"] is True
    assert sum(row["geometry_changed"] for row in ledger) == 2


def test_actions_that_collectively_empty_a_proposal_fall_back_atomically():
    module = _load_module()
    tracks = [
        _track(1, [10, 11], 0.8),
        _track(2, [10, 12], 0.7),
        _track(3, [11, 13], 0.6),
    ]
    plans = [
        _action("resolve", 10, [1, 2], [1], []),
        _action("resolve", 11, [1, 3], [1], []),
    ]
    final, actions, ledger = module.materialize_family(
        tracks, plans, {item: 10 for item in (10, 11, 12, 13)}, "resolve"
    )
    assert final[0]["superpoint_ids"] == [10, 11]
    assert all(row["materialization_status"] == "fallback_would_empty" for row in actions)
    assert all(not row["assignment_applied"] for row in actions)
    assert not any(row["geometry_changed"] for row in ledger)


def test_fallback_is_local_and_other_actions_remain_applied():
    module = _load_module()
    tracks = [_track(1, [10], 0.8), _track(2, [10, 11], 0.7), _track(3, [12], 0.6)]
    plans = [
        _action("resolve", 10, [1, 2], [1], []),
        _action("resolve", 11, [2], [2], []),
    ]
    # The second synthetic row is ownership-valid but not a real planner state;
    # it checks that empty-proposal fallback does not disable unrelated actions.
    final, actions, _ = module.materialize_family(
        tracks, plans, {10: 10, 11: 10, 12: 10}, "resolve"
    )
    assert final[0]["superpoint_ids"] == [10]
    assert final[1]["superpoint_ids"] == [10]
    assert actions[0]["assignment_applied"] is False
    assert actions[1]["assignment_applied"] is True


def test_invalid_plan_ownership_is_rejected():
    module = _load_module()
    tracks = [_track(1, [10], 0.8), _track(2, [11], 0.7)]
    plans = [_action("move", 10, [2], [2], [1])]
    try:
        module.materialize_family(tracks, plans, {10: 10, 11: 10}, "move")
    except ValueError as error:
        assert "ownership differs" in str(error)
    else:
        raise AssertionError("invalid ownership was accepted")


def test_cli_requires_exactly_one_family_and_exposes_no_gt_or_ap_inputs():
    module = _load_module()
    parser = module.build_parser()
    family = next(action for action in parser._actions if "--family" in action.option_strings)
    assert tuple(family.choices) == module.FAMILIES
    options = {option for action in parser._actions for option in action.option_strings}
    assert not any("gt" in option or "ap" in option for option in options)

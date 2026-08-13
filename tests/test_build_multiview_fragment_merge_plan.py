import importlib.util
from itertools import combinations
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_multiview_fragment_merge_plan.py"
    )
    spec = importlib.util.spec_from_file_location("fragment_merge_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(left, right, bridge=0, separation=0, contact=False, containment="none"):
    return {
        "left_proposal_id": left,
        "right_proposal_id": right,
        "bridge_frame_count": bridge,
        "bridge_observation_count": bridge,
        "separation_frame_count": separation,
        "has_spatial_contact": contact,
        "containment_direction": containment,
    }


def _complete_rows(overrides=None):
    overrides = overrides or {}
    return [
        overrides.get((left, right), _row(left, right))
        for left, right in combinations((10, 20, 30, 40), 2)
    ]


def _nodes():
    return [{"proposal_id": item} for item in (10, 20, 30, 40)]


def test_eligibility_contract_is_strict_and_conjunctive():
    module = _module()
    assert module.eligibility_reasons(_row(1, 2, 2, 0, True, "none")) == []
    assert module.eligibility_reasons(_row(1, 2, 1, 1, False, "left_in_right")) == [
        "insufficient_bridge_frames",
        "separation_counterevidence",
        "no_raw_superpoint_contact",
        "strict_inclusion_observed",
    ]


def test_mutual_unique_best_pair_is_planned_without_applying_action():
    module = _module()
    strong = _row(10, 20, bridge=3, contact=True)
    weaker = _row(10, 30, bridge=2, contact=True)
    decisions, actions, proposal_states = module.build_fragment_merge_plan(
        "scene0000_00", _nodes(), _complete_rows({(10, 20): strong, (10, 30): weaker})
    )
    assert len(actions) == 1
    assert actions[0]["anchor_proposal_id"] == 10
    assert actions[0]["absorbed_proposal_id"] == 20
    assert actions[0]["merge_action_applied"] is False
    assert actions[0]["score_used_for_decision"] is False
    assert sum(row["merge_plan_state"] == "planned_mutual_unique_best" for row in decisions) == 1
    assert {row["proposal_id"] for row in proposal_states if row["planned_merge_partner_id"] is not None} == {10, 20}


def test_tied_best_evidence_falls_back_instead_of_using_id_tiebreak():
    module = _module()
    rows = _complete_rows({
        (10, 20): _row(10, 20, bridge=3, contact=True),
        (10, 30): _row(10, 30, bridge=3, contact=True),
    })
    _, actions, proposal_states = module.build_fragment_merge_plan(
        "scene0000_00", _nodes(), rows
    )
    assert actions == []
    state = {row["proposal_id"]: row for row in proposal_states}[10]
    assert state["unique_best_partner_id"] is None
    assert state["best_partner_tie_count"] == 2


def test_actions_are_disjoint_when_two_mutual_pairs_exist():
    module = _module()
    rows = _complete_rows({
        (10, 20): _row(10, 20, bridge=3, contact=True),
        (30, 40): _row(30, 40, bridge=2, contact=True),
    })
    _, actions, _ = module.build_fragment_merge_plan(
        "scene0000_00", _nodes(), rows
    )
    assert [(row["anchor_proposal_id"], row["absorbed_proposal_id"]) for row in actions] == [
        (10, 20), (30, 40)
    ]


def test_missing_pair_is_rejected():
    module = _module()
    try:
        module.build_fragment_merge_plan(
            "scene0000_00", _nodes(), _complete_rows()[:-1]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete relation ledgers must be rejected")


def test_cli_exposes_no_tunable_threshold_or_score_gt_inputs():
    module = _module()
    options = {
        item
        for action in module.build_parser()._actions
        for item in action.option_strings
    }
    assert {"--scene-list", "--fragment-ledger-root", "--output-root"} <= options
    assert not {
        "--minimum-bridge-frames", "--score-field", "--gt-instance-dir",
        "--native-prediction-cache", "--semantic-root",
    } & options

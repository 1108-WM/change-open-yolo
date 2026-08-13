import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_multiview_fragment_family_competition_plan.py"
    )
    spec = importlib.util.spec_from_file_location("fragment_family_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(reliable=True, anchor=True, absorbed=True):
    return {
        "scene_name": "scene0000_00",
        "action_index": 0,
        "anchor_proposal_id": 1,
        "absorbed_proposal_id": 2,
        "common_nonbridge_visible_frame_count": 3 if reliable else 1,
        "quality_evidence_reliable": reliable,
        "merged_dominates_anchor": anchor,
        "merged_dominates_absorbed": absorbed,
        "merged_jointly_dominates": anchor and absorbed and reliable,
        "candidate_action": "none_ledger_only",
        "score_used_for_decision": False,
        "ground_truth_usage": "none",
    }


def test_only_reliable_joint_dominance_uses_merged_candidate():
    module = _module()
    plan = module.plan_family(_row())
    assert plan["planned_action"] == module.USE_MERGED_ACTION
    assert plan["candidate_action_applied"] is False
    assert plan["score_used_for_decision"] is False


def test_insufficient_or_mixed_evidence_falls_back_to_original_pair():
    module = _module()
    insufficient = module.plan_family(_row(reliable=False, anchor=False, absorbed=False))
    mixed = module.plan_family(_row(anchor=True, absorbed=False))
    assert insufficient["planning_state"] == "fallback_insufficient_nonbridge_frames"
    assert mixed["planning_state"] == "fallback_no_joint_pareto_dominance"
    assert insufficient["planned_action"] == module.KEEP_ORIGINAL_PAIR_ACTION
    assert mixed["planned_action"] == module.KEEP_ORIGINAL_PAIR_ACTION


def test_score_or_action_bearing_ledger_is_rejected():
    module = _module()
    for key, value in (
        ("score_used_for_decision", True),
        ("candidate_action", "already_selected"),
    ):
        row = _row()
        row[key] = value
        try:
            module.plan_family(row)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ledger field was accepted: {key}")


def test_cli_has_no_gt_native_semantic_score_or_threshold_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "native", "semantic", "class", "score", "threshold")
    assert not any(any(token in option for token in forbidden) for option in options)

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_track_native_mutual_duplicate_plan.py"
    )
    spec = importlib.util.spec_from_file_location("mutual_duplicate_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relation(native_id, track_coverage, native_coverage, iou, **extra):
    return {
        "proposal_id": 7,
        "native_candidate_id": native_id,
        "track_inside_native_ratio": track_coverage,
        "native_inside_track_ratio": native_coverage,
        "point_iou": iou,
        **extra,
    }


def _summary():
    return {"proposal_id": 7, "overlap_native_candidate_count": 2}


def test_plan_suppresses_only_strict_bidirectional_duplicate_without_scores():
    module = _module()
    relations = [
        _relation(4, 0.999, 0.995, 0.994, native_score=0.01, track_score=0.99),
        _relation(2, 1.0, 0.991, 0.9905, native_score=1.0, track_score=0.01),
    ]
    plan = module.plan_proposal(_summary(), relations, module.SOURCE_VARIANT)
    assert plan["action_family"] == "mutual_duplicate_suppression"
    assert plan["selected_native_candidate_id"] == 4
    assert plan["score_used_for_decision"] is False
    assert plan["track_suppression_applied"] is False


def test_single_direction_inclusion_and_exact_boundary_are_kept():
    module = _module()
    relations = [
        _relation(1, 1.0, 0.8, 0.8),
        _relation(2, 0.99, 1.0, 0.99),
    ]
    plan = module.plan_proposal(_summary(), relations, module.GROW_VARIANT)
    assert plan["planned_action"] == "keep_appended_track"
    assert plan["selected_native_candidate_id"] is None


def test_cross_variant_comparison_records_enter_and_exit_without_actions():
    module = _module()
    base = {
        "proposal_id": 1,
        "planned_action": "keep_appended_track",
        "action_family": "keep",
    }
    suppress = {
        "proposal_id": 1,
        "planned_action": "suppress_appended_track_as_native_mutual_duplicate",
        "action_family": "mutual_duplicate_suppression",
    }
    row = module.compare_variant_plans([base], [suppress])[0]
    assert row["cross_variant_action_state"] == "grow_only_suppression"
    assert row["action_applied"] is False


def test_parser_has_no_gt_score_semantic_or_threshold_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "ap", "score", "class", "semantic", "threshold")
    assert not any(any(token in option for token in forbidden) for option in options)

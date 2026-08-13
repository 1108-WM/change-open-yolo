import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "materialize_track_native_mutual_duplicate_filter.py"
    )
    spec = importlib.util.spec_from_file_location("mutual_filter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track(proposal_id):
    return {
        "proposal_id": proposal_id,
        "track_id": proposal_id,
        "lineage_proposal_ids": [proposal_id, proposal_id + 10],
        "point_count": 100 + proposal_id,
        "points_path": f"track{proposal_id}.npz",
        "mean_node_quality": 0.7 + proposal_id / 100,
        "superpoint_ids": [proposal_id],
    }


def _plan(module, proposal_id, action):
    suppress = action == module.SUPPRESS_ACTION
    return {
        "proposal_id": proposal_id,
        "geometry_variant": module.SOURCE_VARIANT
        if hasattr(module, "SOURCE_VARIANT") else module.EXPECTED_VARIANTS["source"],
        "decision_contract": module.DECISION_CONTRACT,
        "ground_truth_usage": "none",
        "score_used_for_decision": False,
        "track_suppression_applied": False,
        "planned_action": action,
        "action_family": "mutual_duplicate_suppression" if suppress else "keep",
        "strict_mutual_duplicate_relation_count": 1 if suppress else 0,
        "selected_native_candidate_id": 8 if suppress else None,
        "selected_point_iou": 0.999 if suppress else None,
        "selected_track_inside_native_ratio": 1.0 if suppress else None,
        "selected_native_inside_track_ratio": 0.999 if suppress else None,
    }


def test_filter_removes_only_planned_track_and_retains_records_exactly():
    module = _module()
    source = [_track(1), _track(2), _track(3)]
    plans = [
        _plan(module, 1, module.KEEP_ACTION),
        _plan(module, 2, module.SUPPRESS_ACTION),
        _plan(module, 3, module.KEEP_ACTION),
    ]
    retained, actions, ledger = module.filter_tracks(source, plans, "source")
    assert retained == [source[0], source[2]]
    assert [row["proposal_id"] for row in actions] == [2]
    assert len(ledger) == len(source)
    assert sum(row["retained"] for row in ledger) == 2
    assert actions[0]["score_used_for_decision"] is False
    assert actions[0]["native_candidate_mutation_applied"] is False


def test_filter_rejects_score_decision_and_plan_source_mismatch():
    module = _module()
    source = [_track(1)]
    scored = _plan(module, 1, module.SUPPRESS_ACTION)
    scored["score_used_for_decision"] = True
    try:
        module.filter_tracks(source, [scored], "source")
    except ValueError as error:
        assert "used a score" in str(error)
    else:
        raise AssertionError("score-driven plan was accepted")

    wrong = _plan(module, 2, module.KEEP_ACTION)
    try:
        module.filter_tracks(source, [wrong], "source")
    except ValueError as error:
        assert "count, order, or IDs differ" in str(error)
    else:
        raise AssertionError("mismatched plan was accepted")


def test_filter_accepts_explicit_single_geometry_variant():
    module = _module()
    source = [_track(1), _track(2)]
    plans = [
        _plan(module, 1, module.KEEP_ACTION),
        _plan(module, 2, module.SUPPRESS_ACTION),
    ]
    for row in plans:
        row["geometry_variant"] = "multiview_fragment_merge_f1"
    retained, actions, _ = module.filter_tracks(
        source,
        plans,
        geometry_variant="multiview_fragment_merge_f1",
    )
    assert retained == [source[0]]
    assert [row["proposal_id"] for row in actions] == [2]

    plans[0]["geometry_variant"] = "wrong_variant"
    try:
        module.filter_tracks(
            source,
            plans,
            geometry_variant="multiview_fragment_merge_f1",
        )
    except ValueError as error:
        assert "geometry variant differs" in str(error)
    else:
        raise AssertionError("mismatched explicit geometry variant was accepted")


def test_parser_has_no_gt_ap_score_threshold_or_native_input():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "ap", "score", "threshold", "native-prediction")
    assert not any(any(token in option for token in forbidden) for option in options)

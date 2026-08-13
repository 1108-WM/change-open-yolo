import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "diagnose_mv3dis_global_feasible_action_oracle_gt.py"
    )
    spec = importlib.util.spec_from_file_location("global_feasible_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _action(name, kind, target, outcomes, utility):
    result = {
        "action_name": name,
        "action_kind": kind,
        "target_owner_proposal_id": target,
        "proposal_outcomes": outcomes,
    }
    if utility:
        result["proposal_outcomes"][0].update({
            "fixed_gt_available": True,
            "fixed_gt_iou_delta": utility,
        })
    return result


def _outcome(proposal_id, removed=(), added=()):
    return {
        "proposal_id": proposal_id,
        "removed_superpoint_ids": list(removed),
        "added_superpoint_ids": list(added),
        "fixed_gt_available": False,
    }


def test_tie_prefers_unknown_noop():
    module = _module()
    unknown = _action(module.UNKNOWN_NOOP_ACTION, "unknown_noop", None, [], 0.0)
    assigned = _action("assign_to_proposal_2", "assign_owner", 2, [], 0.0)
    selected, utility = module.choose_local_oracle_action([assigned, unknown])
    assert selected["action_name"] == module.UNKNOWN_NOOP_ACTION
    assert utility == 0.0


def test_global_empty_proposal_falls_back_to_unknown():
    module = _module()
    source = {1: {10, 11}, 2: {20}}
    row_a = {
        "scene_name": "scene_unit", "superpoint_id": 10,
        "frozen_planned_owner_proposal_id": 2,
        "action_results": [
            _action(module.UNKNOWN_NOOP_ACTION, "unknown_noop", None, [], 0.0),
            _action("assign_to_proposal_2", "assign_owner", 2, [
                _outcome(1, removed=[10]), _outcome(2, added=[10])
            ], 1.0),
        ],
    }
    row_b = {
        "scene_name": "scene_unit", "superpoint_id": 11,
        "frozen_planned_owner_proposal_id": 2,
        "action_results": [
            _action(module.UNKNOWN_NOOP_ACTION, "unknown_noop", None, [], 0.0),
            _action("assign_to_proposal_2", "assign_owner", 2, [
                _outcome(1, removed=[11]), _outcome(2, added=[11])
            ], 2.0),
        ],
    }
    geometry, selections = module._prepare_selections(source, [row_a, row_b])
    assert geometry[1]
    assert sum(row["global_status"] == "fallback_would_empty" for row in selections) == 1
    assert any(row["selected_action_kind"] == "unknown_noop" for row in selections)

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_t1_global_feasible_action_oracle_gt.py"
    spec = importlib.util.spec_from_file_location("t1_global_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _action(kind, source, target, observations):
    return {"action_type": kind, "source_track_id": source, "target_track_id": target, "source_observation_ids": observations}


def test_live_selection_enforces_owner_and_same_frame_conflicts():
    module = _module()
    association = {1: [10], 2: [20]}
    frames = {10: "0", 20: "1", 30: "2", 40: "2"}
    owners = module._observation_owners(association)
    by_track = {track: module._track_frames(items, frames) for track, items in association.items()}
    ok, reason = module._try_apply_live_action(association, owners, by_track, _action("attach", None, 1, [40]), frames)
    assert ok and reason is None and owners[40] == 1
    ok, reason = module._try_apply_live_action(association, owners, by_track, _action("attach", None, 2, [40]), frames)
    assert not ok and reason == "observation_already_owned"
    ok, reason = module._try_apply_live_action(association, owners, by_track, _action("attach", None, 1, [30]), frames)
    assert not ok and reason == "same_frame_mutual_exclusion"


def test_live_selection_prevents_cumulative_empty_source_track():
    module = _module()
    association = {1: [10], 2: [20]}
    frames = {10: "0", 20: "1"}
    owners = module._observation_owners(association)
    by_track = {track: module._track_frames(items, frames) for track, items in association.items()}
    ok, reason = module._try_apply_live_action(association, owners, by_track, _action("reassign", 1, 2, [10]), frames)
    assert not ok and reason == "cumulative_empty_track"

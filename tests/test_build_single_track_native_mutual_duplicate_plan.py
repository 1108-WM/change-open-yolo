import importlib.util
import json
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_single_track_native_mutual_duplicate_plan.py"
    )
    spec = importlib.util.spec_from_file_location("single_duplicate_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_requires_frozen_gt_free_actionless_ledger(tmp_path):
    module = _module()
    path = tmp_path / "single_track_native_competition_ledger_summary.json"
    payload = {
        "scene_count": 3,
        "geometry_variant": "fragment_f1",
        "ground_truth_usage": "none",
        "candidate_action_count": 0,
    }
    path.write_text(json.dumps(payload))
    manifest_path, loaded = module._load_manifest(tmp_path, 3)
    assert manifest_path == path
    assert loaded["geometry_variant"] == "fragment_f1"

    payload["candidate_action_count"] = 1
    path.write_text(json.dumps(payload))
    try:
        module._load_manifest(tmp_path, 3)
    except ValueError as error:
        assert "candidate actions" in str(error)
    else:
        raise AssertionError("action-bearing ledger was accepted")


def test_parser_has_no_gt_score_semantic_or_threshold_inputs():
    module = _module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("gt", "ap", "score", "class", "semantic", "threshold")
    assert not any(any(token in option for token in forbidden) for option in options)

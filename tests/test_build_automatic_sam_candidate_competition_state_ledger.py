import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_candidate_competition_state_ledger.py"
    spec = importlib.util.spec_from_file_location("candidate_competition_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_role_only_marks_a_redundancy_direction_when_dominant_candidate_is_container():
    module = _module()
    relation = {
        "left_candidate_id": 1, "right_candidate_id": 2,
        "pareto_dominance_direction": "right_dominates_left",
        "containment": {"state": "left_strictly_contained_by_right"},
    }
    assert module._role_for_candidate(relation, 2) == "dominant_geometric_container"
    assert module._role_for_candidate(relation, 1) == "contained_by_dominant_container"


def test_increment_features_does_not_turn_multiple_views_into_decision():
    module = _module()
    row = {
        "track_observation_count": 2, "candidate_supported_observation_count": 2,
        "candidate_independent_observation_count": 2,
        "view_records": [
            {"candidate_observation_point_count": 2, "candidate_independent_point_count": 1, "candidate_same_class_native_explained_point_count": 1, "candidate_any_native_explained_point_count": 1},
            {"candidate_observation_point_count": 4, "candidate_independent_point_count": 2, "candidate_same_class_native_explained_point_count": 2, "candidate_any_native_explained_point_count": 2},
        ],
    }
    result = module.increment_features(row)
    assert result["all_track_observations_have_independent_points"]
    assert result["weighted_independent_ratio"] == .5

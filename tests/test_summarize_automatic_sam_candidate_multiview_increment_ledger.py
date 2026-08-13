import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_automatic_sam_candidate_multiview_increment_ledger.py"
    spec = importlib.util.spec_from_file_location("increment_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_increment_features_are_weighted_by_observed_candidate_points():
    module = _module()
    row = {
        "track_observation_count": 2,
        "candidate_supported_observation_count": 2,
        "candidate_independent_observation_count": 1,
        "view_records": [
            {"candidate_observation_point_count": 10, "candidate_independent_point_count": 4, "candidate_same_class_native_explained_point_count": 3, "candidate_any_native_explained_point_count": 6},
            {"candidate_observation_point_count": 2, "candidate_independent_point_count": 0, "candidate_same_class_native_explained_point_count": 1, "candidate_any_native_explained_point_count": 2},
        ],
    }
    result = module.candidate_increment_features(row)
    assert result["weighted_candidate_independent_ratio"] == 4 / 12
    assert result["weighted_candidate_same_class_native_explained_ratio"] == 4 / 12
    assert result["weighted_candidate_any_native_explained_ratio"] == 8 / 12

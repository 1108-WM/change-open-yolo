import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_z6a_class_candidate_space_oracle_gt.py"
    spec = importlib.util.spec_from_file_location("z6a_class_candidate_space", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top_positive_indices_are_stable_and_ignore_zero_values():
    module = _module()
    values = np.asarray([0.0, 0.5, 0.5, -1.0, 0.2], dtype=np.float32)
    assert module._top_positive_indices(values, 3) == (1, 2, 4)


def test_candidate_sets_use_own_inherited_max_and_model_union():
    module = _module()
    distributions = {
        "geometry_yolo": np.asarray([[0.8, 0.0, 0.2, 0.0]], dtype=np.float32),
        "inherited_yolo": np.asarray([[0.0, 0.7, 0.0, 0.0]], dtype=np.float32),
        "geometry_alpha": np.asarray([[0.0, 0.0, 0.9, 0.1]], dtype=np.float32),
        "inherited_alpha": np.asarray([[0.6, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    sets = module._candidate_sets(0, distributions, 2)
    assert sets["yolo_top5_oracle"] == {0, 1}
    assert sets["alpha_top5_oracle"] == {0, 2}
    assert sets["yolo_alpha_union_top5_oracle"] == {0, 1, 2}
    assert sets["full_class_oracle"] == {0, 1, 2, 3}


def test_coverage_aggregation_reports_repairable_wrong_fraction():
    module = _module()
    row = {
        "scene_name": "scene",
        "threshold": 0.5,
        "eligible_prediction_count": 10,
        "current_correct_prediction_count": 2,
        "incorrect_prediction_count": 8,
        "eligible_node_target_count": 5,
        "yolo_only_target_count": 1,
        "alpha_only_target_count": 3,
    }
    for variant, contains, repair, attainable, node_contains in (
        ("yolo_top5_oracle", 5, 4, 6, 3),
        ("alpha_top5_oracle", 8, 6, 8, 4),
        ("yolo_alpha_union_top5_oracle", 9, 7, 9, 5),
        ("full_class_oracle", 10, 8, 10, 5),
    ):
        row[f"{variant}_contains_target_count"] = contains
        row[f"{variant}_repairable_wrong_count"] = repair
        row[f"{variant}_attainable_correct_count"] = attainable
        row[f"{variant}_node_target_contains_count"] = node_contains
    result = module._aggregate_coverage([row])["0.5"]
    assert result["variants"]["yolo_alpha_union_top5_oracle"]["repairable_fraction_of_current_wrong"] == 7 / 8
    assert result["variants"]["alpha_top5_oracle"]["node_target_contains_fraction"] == 4 / 5

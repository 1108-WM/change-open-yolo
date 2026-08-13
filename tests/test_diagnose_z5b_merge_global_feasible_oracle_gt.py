import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "diagnose_z5b_merge_global_feasible_oracle_gt.py"
SPEC = importlib.util.spec_from_file_location("diagnose_z5b_merge_global_feasible_oracle_gt", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_crossings_use_official_strict_overlap_contract():
    result = MODULE.threshold_crossings(0.49, 0.61)
    assert result["50"] == 1
    assert result["55"] == 1
    assert result["60"] == 1
    assert result["65"] == 0


def test_targetwise_oracle_keeps_one_action_per_gt_and_requires_official_gain():
    base = {
        "scene_name": "scene", "semantic_correct": True,
        "action_best_same_class_gt_instance_id": 6001,
        "noop_best_same_class_iou": 0.40,
        "threshold_crossings": {"25": 0, "50": 1, "55": 0, "60": 0, "65": 0, "70": 0, "75": 0, "80": 0, "85": 0, "90": 0},
        "predicted_semantic_class_id": 6,
    }
    rows = [
        {**base, "action_id": "a", "child_candidate_id": 1, "action_best_same_class_iou": 0.52},
        {**base, "action_id": "b", "child_candidate_id": 2, "action_best_same_class_iou": 0.54},
        {**base, "action_id": "c", "child_candidate_id": 3,
         "action_best_same_class_gt_instance_id": 7001, "semantic_correct": False,
         "action_best_same_class_iou": 0.90},
    ]
    selected = MODULE.choose_targetwise_actions(rows)
    assert len(selected) == 1
    assert selected[0]["action_id"] == "b"
    assert selected[0]["action_best_same_class_iou"] == pytest.approx(0.54)

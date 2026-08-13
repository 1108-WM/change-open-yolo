import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_n1_sampro3d_candidate_space_oracle_gt.py"
    spec = importlib.util.spec_from_file_location("n1_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maximum_matching_uses_one_seed_family_once_and_maximizes_recovered_gt_count():
    module = _module()
    edges = {"seed_a": {1: .9, 2: .8}, "seed_b": {1: .7}}
    matching = module.maximum_cardinality_matching(edges, .5)
    assert len(matching) == 2
    assert set(matching.values()) == {1, 2}


def test_family_summary_marks_multiview_need_without_selecting_a_member():
    module = _module()
    members = [
        {"candidate_family_key": "seed_1", "seed_superpoint_id": 1, "candidate_superpoint_ids": [1], "iou_by_gt": {5: .2}, "frame_index": 0, "candidate_id": "a", "sam_predicted_iou": .9},
        {"candidate_family_key": "seed_1", "seed_superpoint_id": 1, "candidate_superpoint_ids": [2], "iou_by_gt": {5: .2}, "frame_index": 1, "candidate_id": "b", "sam_predicted_iou": .8},
    ]
    row = module.family_summary(members, {1: 5, 2: 5}, {1: {5: 3}, 2: {5: 3}}, {5: 10})
    assert row["best_single_observation_iou"] == .2
    assert row["multiview_aggregation_needed_iou25"] is True

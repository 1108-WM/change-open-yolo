import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_n2_sampro3d_candidate_family_quality_ledger.py"
    spec = importlib.util.spec_from_file_location("n2_family", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_keeps_core_and_unknown_boundary_without_selecting_hypothesis():
    module = _module()
    members = [
        {"candidate_id": "a", "candidate_family_key": "seed_1", "seed_superpoint_id": 1, "candidate_superpoint_ids": [1, 2], "frame_index": 0, "hypothesis_index": 0, "sam_predicted_iou": .7},
        {"candidate_id": "b", "candidate_family_key": "seed_1", "seed_superpoint_id": 1, "candidate_superpoint_ids": [1, 3], "frame_index": 1, "hypothesis_index": 0, "sam_predicted_iou": .9},
    ]
    row = module.summarize_family(members)
    assert row["reliable_core_superpoint_ids"] == [1]
    assert row["unknown_boundary_superpoint_ids"] == [2, 3]
    assert row["sam_top_member_candidate_id"] == "b"
    assert row["member_count"] == 2


def test_d2b_relation_marks_only_exact_coverage_as_duplicate():
    module = _module()
    family = {"family_union_superpoint_ids": [1, 2]}
    exact = module.d2b_relation(family, [{"track_id": 3, "superpoint_ids": [1, 2]}])
    partial = module.d2b_relation(family, [{"track_id": 3, "superpoint_ids": [1, 2, 3]}])
    assert exact["exact_mutual_duplicate"] is True
    assert partial["exact_mutual_duplicate"] is False

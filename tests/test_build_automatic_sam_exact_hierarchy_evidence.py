import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_exact_hierarchy_evidence.py"
    spec = importlib.util.spec_from_file_location("exact_hierarchy_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rle_area_checks_size_and_foreground_runs():
    module = _module()
    assert module.rle_area({"size": [2, 3], "counts": [1, 2, 1, 1, 1]}) == 3


def test_exact_mask_relations_are_lifted_to_track_pairs_without_decision():
    module = _module()
    nodes = [
        {"observation_id": 10, "existing_track_ids": [4]},
        {"observation_id": 20, "existing_track_ids": [9]},
    ]
    relations = [{
        "left_observation_id": 10, "right_observation_id": 20, "frame_index": 3,
        "intersection_pixel_count": 8, "iou": .5, "left_coverage": .8, "right_coverage": .6,
    }]
    records = module.build_track_pair_evidence(nodes, relations)
    assert records[0]["left_source_track_id"] == 4
    assert records[0]["right_source_track_id"] == 9
    assert records[0]["mean_left_coverage"] == .8
    assert "不定义同实例" in records[0]["relation_state"]

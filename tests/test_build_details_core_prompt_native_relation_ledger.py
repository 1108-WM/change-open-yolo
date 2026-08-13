import importlib.util
import json
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_core_prompt_native_relation_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("prompt_native_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_relation_records_addition_removal_and_bidirectional_containment():
    module = _module()
    relation = module.set_relation([0, 1, 2], [1, 2, 3, 4])
    assert relation == {
        "intersection_point_count": 2,
        "union_point_count": 5,
        "iou": 0.4,
        "left_inside_right_ratio": 2 / 3,
        "right_inside_left_ratio": 0.5,
    }


def test_native_relation_orders_positive_top2_and_keeps_both_containments():
    module = _module()
    masks = np.asarray([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=bool)
    vectors = module.native_overlap_vectors([0, 1, 2], masks)
    relation = module.summarize_native_relation(vectors)
    assert relation["touching_native_count"] == 2
    assert [item["native_candidate_id"] for item in relation["top_matches"]] == [0, 1]
    assert relation["top_matches"][0]["iou"] == 2 / 3
    assert relation["top_matches"][0]["source_inside_native_ratio"] == 2 / 3
    assert relation["top_matches"][0]["native_inside_source_ratio"] == 1.0


def test_greedy_native_cover_counts_masks_needed_for_core_coverage():
    module = _module()
    masks = np.asarray([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=bool)
    cover = module.greedy_native_cover([0, 1, 2, 3, 4], masks)
    assert cover["native_union_coverage_ratio"] == 1.0
    assert cover["greedy_native_candidate_ids"] == [0, 1, 2]
    assert cover["greedy_count_to_50pct"] == 2
    assert cover["greedy_count_to_80pct"] == 2
    assert cover["greedy_count_to_90pct"] == 3


def test_added_point_coverage_separates_best_other_and_any_native():
    module = _module()
    masks = np.asarray([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ], dtype=bool)
    coverage = module.added_point_native_coverage([0, 1, 2, 3], masks, 0)
    assert coverage["inside_best_native_count"] == 2
    assert coverage["inside_other_native_count"] == 2
    assert coverage["inside_any_native_count"] == 3
    assert coverage["inside_any_native_ratio"] == 0.75


def test_added_superpoint_connectivity_distinguishes_direct_and_transitive_links():
    module = _module()
    neighbors = {
        1: [{"neighbor_superpoint_id": 2, "boundary_contact_count": 5}],
        2: [
            {"neighbor_superpoint_id": 1, "boundary_contact_count": 5},
            {"neighbor_superpoint_id": 3, "boundary_contact_count": 4},
        ],
        3: [{"neighbor_superpoint_id": 2, "boundary_contact_count": 4}],
        4: [],
    }
    result = module.added_superpoint_connectivity([1], [2, 3, 4], neighbors)
    by_id = {item["superpoint_id"]: item for item in result["added_superpoints"]}
    assert result["directly_adjacent_to_core_count"] == 1
    assert result["reachable_from_core_count"] == 2
    assert by_id[2]["directly_adjacent_to_core"] is True
    assert by_id[3]["reachable_from_core_via_added"] is True
    assert by_id[4]["reachable_from_core_via_added"] is False


def test_empty_added_points_use_none_ratios_instead_of_false_zero_evidence():
    module = _module()
    masks = np.zeros((3, 2), dtype=bool)
    result = module.added_point_native_coverage([], masks, -1)
    assert result["added_point_count"] == 0
    assert result["inside_any_native_ratio"] is None


def test_native_cache_contract_rejects_strong_cache_for_mask3d_run(tmp_path):
    module = _module()
    (tmp_path / "native_cache_no_gt_manifest.json").write_text(json.dumps({
        "candidate_inputs": {"loaded": 12},
        "decision_state": "strong",
        "scene_count": 1,
    }))
    try:
        module._native_cache_contract(tmp_path, "mask3d_yoloworld_only")
    except ValueError as error:
        assert "期望 mask3d_yoloworld_only" in str(error)
    else:
        raise AssertionError("strong/native 缓存必须被 pure Mask3D 入口拒绝")

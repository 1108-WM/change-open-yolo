import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "build_details_added_superpoint_boundary_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("added_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(observation_id, lifted, inside, visible, quality=0.8):
    return {
        "observation_id": observation_id,
        "lifted_superpoints": np.asarray(lifted, dtype=np.int64),
        "inside_counts": inside,
        "visible_counts": visible,
        "quality": quality,
    }


def test_core_anchor_prefers_coverage_before_quality():
    module = _module()
    visible = {1: 10, 2: 10, 3: 10}
    low_coverage = _observation(1, [1], {1: 10}, visible, quality=1.0)
    full_core = _observation(2, [1, 2, 3], {1: 10, 2: 10, 3: 10}, visible, quality=0.5)
    selected, metrics = module.best_core_anchored_observation(
        [1, 2], [low_coverage, full_core], 3
    )
    assert selected["observation_id"] == 2
    assert metrics["core_coverage"] == 1.0


def test_atomic_holdout_excludes_construction_frames_and_records_delta_gvc():
    module = _module()
    visible = {
        0: {1: 10, 2: 10},
        1: {1: 10, 2: 10},
        2: {1: 10, 2: 10},
    }
    observations = {
        0: [_observation(0, [1, 2], {1: 10, 2: 10}, visible[0])],
        1: [_observation(1, [1, 2], {1: 10, 2: 10}, visible[1])],
        2: [_observation(2, [1], {1: 10}, visible[2])],
    }
    result = module.atomic_holdout_evidence(
        [1], [1], 2, [0, 1, 2], [0], observations, visible, 3, 0.3, 0.3
    )
    assert result["joint_visible_independent_frame_count"] == 2
    assert result["positive_anchor_frame_count"] == 1
    assert result["exclusion_anchor_frame_count"] == 1
    assert result["grown_mean_best_siou"] == 0.75
    assert result["base_mean_best_siou"] == 0.75
    assert result["delta_mean_best_siou"] == 0.0
    assert [row["frame_index"] for row in result["frames"]] == [1, 2]


def test_graph_hops_distinguishes_direct_transitive_and_unreachable_added_atoms():
    module = _module()
    neighbors = {
        1: [{"neighbor_superpoint_id": 2}],
        2: [{"neighbor_superpoint_id": 1}, {"neighbor_superpoint_id": 3}],
        3: [{"neighbor_superpoint_id": 2}],
        4: [],
    }
    assert module.graph_hops_from_core([1], [2, 3, 4], neighbors) == {
        2: 1,
        3: 2,
        4: None,
    }


def test_boundary_relation_uses_contact_weighted_geometry():
    module = _module()
    neighbors = {3: [
        {
            "neighbor_superpoint_id": 1,
            "boundary_contact_count": 3,
            "boundary_contact_ratio": 0.1,
            "mean_boundary_distance": 0.01,
            "mean_normal_difference": 0.2,
            "mean_color_difference": 0.4,
        },
        {
            "neighbor_superpoint_id": 2,
            "boundary_contact_count": 1,
            "boundary_contact_ratio": 0.3,
            "mean_boundary_distance": 0.03,
            "mean_normal_difference": 0.6,
            "mean_color_difference": 0.8,
        },
    ]}
    result = module.boundary_relation(3, [1, 2], neighbors)
    assert result["direct_core_contact_count_sum"] == 4
    assert result["max_direct_core_contact_density"] == 0.3
    assert result["contact_weighted_boundary_distance"] == 0.015
    assert np.isclose(result["contact_weighted_normal_difference"], 0.3)
    assert np.isclose(result["contact_weighted_color_difference"], 0.5)


def test_native_ownership_separates_core_owner_from_other_score_one_candidate():
    module = _module()
    masks = np.asarray([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ], dtype=bool)
    result = module.native_ownership(
        [2, 3], [0, 1], masks, np.asarray([1.0, 1.0, 0.5])
    )
    assert result["core_best_native_candidate_id"] == 0
    assert result["added_inside_core_best_native_ratio"] == 0.0
    assert result["best_other_score_one_native_candidate_id"] == 1
    assert result["best_other_score_one_native_coverage_ratio"] == 1.0

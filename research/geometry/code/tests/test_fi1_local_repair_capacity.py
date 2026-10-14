import itertools
import numpy as np
from tools.fi1_local_repair_capacity import (ideal_repair, unit_ceiling,
    reliable_visibility, add_group_observation, physical_groups)


def test_oracle_preserves_unknown_and_outside_and_can_add_missing():
    parent = np.arange(100); target = np.r_[np.arange(90), 101, 102]
    editable = np.r_[np.arange(80, 100), 101, 102]
    seen = np.ones(110, bool); seen[95] = False
    actual = ideal_repair(parent, editable, target, seen)
    assert set(range(80)).issubset(actual)
    assert 95 in actual and 101 in actual and 102 in actual
    assert len(actual) == 100


def test_unit_ceiling_matches_exhaustive_subsets_with_fixed_outside():
    rng = np.random.default_rng(84)
    for _ in range(40):
        labels = rng.integers(0, 2, 30); labels[0] = 1
        units = rng.integers(0, 6, 30); editable = np.arange(8, 30)
        parent = np.arange(20); result = unit_ceiling(parent, editable, labels, 1, units)
        fixed = np.setdiff1d(parent, editable); n = np.sum(labels == 1); best = 0
        unique = np.unique(units[editable])
        for bits in itertools.product((0, 1), repeat=len(unique)):
            selected = unique[np.array(bits, bool)]
            p = np.union1d(fixed, editable[np.isin(units[editable], selected)])
            hit = np.sum(labels[p] == 1); best = max(best, hit/(n+len(p)-hit))
        assert abs(result['iou']-best) < 1e-12


def test_visibility_does_not_treat_missing_occluded_behind_as_negative():
    sensor = np.full((5, 5), 2.); sensor[2, 3] = 0
    xyz = np.array([[4, 4, 2], [6, 4, 2], [2, 2, 1], [2, 2, -1], [0, 0, 2]])
    result = reliable_visibility(xyz, np.eye(4), np.eye(4), sensor)
    assert result.tolist() == [True, False, False, False, False]


def test_repeated_frames_cannot_manufacture_independent_observations():
    first = np.full(3, -1); counts = np.zeros(3, np.uint8)
    add_group_observation(first, counts, np.array([True, True, False]), 4)
    add_group_observation(first, counts, np.array([True, True, False]), 4)
    assert counts.tolist() == [1, 1, 0]
    add_group_observation(first, counts, np.array([True, False, True]), 7)
    assert counts.tolist() == [2, 1, 1]


def test_vector_groups_match_original_physical_group_definition():
    from tools.fi1_split_grow_core import pose_groups
    rng = np.random.default_rng(12); poses = {}
    for i in range(60):
        p = np.eye(4); theta = rng.uniform(-.7, .7)
        p[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        p[:3, 3] = rng.uniform(-.2, .2, 3); poses[i] = p
    assert physical_groups(poses) == pose_groups(poses)

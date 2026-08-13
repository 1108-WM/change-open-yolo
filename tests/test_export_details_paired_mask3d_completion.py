import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "export_details_paired_mask3d_completion.py"
    )
    spec = importlib.util.spec_from_file_location("paired_completion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mutual_best_positive_pairs_are_one_to_one_and_ignore_zero_overlap():
    module = _module()
    matrix = np.asarray([
        [0.8, 0.1, 0.0],
        [0.5, 0.6, 0.0],
        [0.0, 0.5, 0.0],
    ])
    assert module.mutual_best_positive_pairs(matrix, [10, 20, 30]) == [
        (10, 0, 0.8),
        (20, 1, 0.6),
    ]


def test_mutual_best_ties_are_deterministic():
    module = _module()
    matrix = np.asarray([[0.5, 0.5], [0.5, 0.4]])
    assert module.mutual_best_positive_pairs(matrix, [4, 8]) == [(4, 0, 0.5)]


def test_completion_only_grows_paired_candidate_and_reports_actual_increment():
    module = _module()
    masks = np.asarray([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=bool)
    completed, rows = module.complete_native_masks(
        masks,
        [(11, 0, 0.75)],
        {11: np.asarray([1, 2, 4])},
    )
    assert np.array_equal(completed[:, 0], [1, 1, 1, 0, 1])
    assert np.array_equal(completed[:, 1:], masks[:, 1:])
    assert rows[0]["declared_added_point_count"] == 3
    assert rows[0]["already_inside_native_count"] == 1
    assert rows[0]["actual_added_point_count"] == 2


def test_completion_rejects_duplicate_candidate_pairing():
    module = _module()
    try:
        module.complete_native_masks(
            np.zeros((3, 2), dtype=bool),
            [(1, 0, 0.5), (2, 0, 0.4)],
            {1: [0], 2: [1]},
        )
    except ValueError as error:
        assert "重复配对" in str(error)
    else:
        raise AssertionError("同一 native candidate 不能被多个轨迹补全")

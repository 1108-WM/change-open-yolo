import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_superpoint_methods.py"
SPEC = importlib.util.spec_from_file_location("compare_superpoint_methods", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_purity_stats_reports_weighted_majority_and_mixed_segments():
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    targets = np.array([3, 3, 4, 7, 7], dtype=np.int64)

    stats = MODULE._purity_stats(labels, targets, threshold=0.95)

    assert stats["mixed_segments"] == 1
    assert stats["mixed_points"] == 3
    assert stats["weighted_purity"] == 0.8
    assert stats["rows"][0]["majority_label"] == 3


def test_partition_overlap_counts_split_and_merge():
    reference = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    compared = np.array([0, 1, 1, 2, 2], dtype=np.int64)

    stats = MODULE._partition_overlap(reference, compared)

    assert stats["split_reference_segments"] == 2
    assert stats["merged_compared_segments"] == 2
    assert stats["point_count"] == 5

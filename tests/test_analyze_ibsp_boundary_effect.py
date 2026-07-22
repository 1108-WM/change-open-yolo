import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "analyze_ibsp_boundary_effect.py"
SPEC = importlib.util.spec_from_file_location("analyze_ibsp_boundary_effect", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_partition_stats_reports_a_split_without_false_merge():
    baseline = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    ibsp = np.array([0, 0, 1, 2, 2], dtype=np.int64)

    stats = ANALYZER._partition_stats(baseline, ibsp)

    assert stats["reference_segments"] == 2
    assert stats["compared_segments"] == 3
    assert stats["split_reference_segments"] == 1
    assert stats["split_reference_points"] == 3
    assert stats["merged_compared_segments"] == 0

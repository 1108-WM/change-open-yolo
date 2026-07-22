import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "lift_sam2_tracks_to_superpoints.py"
SPEC = importlib.util.spec_from_file_location("lift_sam2_tracks_to_superpoints", MODULE_PATH)
LIFTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFTER)


def test_same_frame_overlap_cleanup_removes_ambiguous_pixels_from_all_tracks():
    first = np.zeros((2, 4, 4), dtype=bool)
    second = np.zeros((2, 4, 4), dtype=bool)
    first[0, 0, :3] = True
    second[0, 0, 1:4] = True
    first[1, 1, 1] = True
    second[1, 1, 1] = True

    cleaned, summary = LIFTER._remove_same_frame_track_overlaps([first, second], min_overlap_pixels=2)

    assert cleaned[0][0].sum() == 1
    assert cleaned[1][0].sum() == 1
    assert cleaned[0][0, 0, 1] == 0
    assert cleaned[1][0, 0, 1] == 0
    # 单像素交叠低于门槛，保留原始证据。
    assert cleaned[0][1, 1, 1] == 1
    assert cleaned[1][1, 1, 1] == 1
    assert summary["affected_frame_count"] == 1
    assert summary["total_removed_pixels"] == 2

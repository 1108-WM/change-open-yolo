import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_sam_automatic_observations.py"
    spec = importlib.util.spec_from_file_location("automatic_observations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_to_visible_points_uses_only_visible_in_bounds_pixels():
    module = _module()
    mask = np.asarray([[True, False], [False, True]], dtype=bool)
    projections = np.asarray([[[0, 0], [1, 1], [3, 3], [0, 1]]], dtype=np.int64)
    visibility = np.asarray([[True, True, True, False]], dtype=bool)
    points = module.mask_to_visible_points(mask, 0, projections, visibility, (1.0, 1.0))
    assert np.array_equal(points, np.asarray([0, 1], dtype=np.int64))


def test_uniform_frame_selection_spans_loaded_sequence():
    module = _module()
    assert module.select_frame_indices(100, 30, 1, "first") == list(range(30))
    selected = module.select_frame_indices(100, 30, 1, "uniform")
    assert len(selected) == 30
    assert selected[0] == 0
    assert selected[-1] == 99


def test_rle_roundtrip_preserves_non_square_mask_and_exact_relation():
    module = _module()
    mask = np.asarray([[True, False, True], [False, True, False]], dtype=bool)
    restored = module.decode_binary_mask_rle(module.encode_binary_mask_rle(mask))
    relation = module.exact_mask_relation(mask, restored)
    assert np.array_equal(restored, mask)
    assert relation == {"intersection_pixel_count": 3, "iou": 1.0, "left_coverage": 1.0, "right_coverage": 1.0}

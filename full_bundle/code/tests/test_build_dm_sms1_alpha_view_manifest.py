import numpy as np

from tools.build_dm_sms1_alpha_view_manifest import (
    select_top_visible_views,
    square_crop_box,
)


def test_select_top_visible_views_uses_count_then_frame_index():
    counts = np.asarray([3, 8, 8, 0, 5])
    assert select_top_visible_views(counts, ["9", "2", "1", "0", "4"], 3) == [1, 2, 4]


def test_square_crop_box_expands_per_side_then_squares_about_center():
    square, integer = square_crop_box([20.0, 30.0, 40.0, 40.0], 0.2, 100, 80)
    assert np.allclose(square, [16.0, 21.0, 44.0, 49.0])
    assert integer == [16, 21, 45, 50]


def test_square_crop_box_clips_at_image_boundary():
    square, integer = square_crop_box([0.0, 0.0, 10.0, 20.0], 0.0, 30, 40)
    assert square == [0.0, 0.0, 15.0, 20.0]
    assert integer == [0, 0, 16, 21]

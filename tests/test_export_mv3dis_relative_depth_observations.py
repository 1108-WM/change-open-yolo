import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "export_mv3dis_relative_depth_observations.py"
    spec = importlib.util.spec_from_file_location("mv3dis_relative_depth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_depth_visibility_is_strict_and_weight_is_continuous():
    module = _load_module()
    visible, weights = module.relative_depth_weights(
        projected_depth=np.asarray([100.0, 104.0, 105.0, 94.0]),
        measured_depth=np.asarray([100.0, 100.0, 100.0, 100.0]),
        in_image=np.asarray([True, True, True, False]),
    )
    assert visible.tolist() == [True, True, False, False]
    assert np.allclose(weights, [1.0, 0.2, 0.0, 0.0])


def test_projection_uses_camera_pose_intrinsic_and_relative_depth():
    module = _load_module()
    points = np.asarray([
        [0.0, 0.0, 2.0, 1.0],
        [2.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, -1.0, 1.0],
    ])
    intrinsic = np.eye(4)
    depth = np.zeros((2, 2), dtype=np.float64)
    depth[0, 0] = 2.0
    depth[0, 1] = 2.2
    pixels, visible, weights = module.project_relative_depth_frame(
        points, np.eye(4), intrinsic, depth
    )
    assert pixels[:2].tolist() == [[0, 0], [1, 0]]
    assert visible.tolist() == [True, False, False]
    assert weights.tolist() == [1.0, 0.0, 0.0]


def test_rle_resolution_mapping_returns_aligned_points_and_weights():
    module = _load_module()
    mask = np.asarray([
        [True, False, False, False],
        [False, False, False, False],
        [False, False, True, False],
        [False, False, False, False],
    ])
    pixels = np.asarray([[0, 0], [1, 1], [1, 0]], dtype=np.int64)
    visible = np.asarray([True, True, True])
    weights = np.asarray([1.0, 0.7, 0.4], dtype=np.float32)
    points, selected_weights = module.mask_points_from_projection(
        mask, pixels, visible, weights, depth_shape=(2, 2)
    )
    assert points.tolist() == [0, 1]
    assert np.allclose(selected_weights, [1.0, 0.7])


def test_cli_exposes_no_depth_threshold_scan_or_gt_ap_options():
    module = _load_module()
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = ("alpha", "depth-threshold", "--gt", "--ap", "semantic", "class")
    assert not any(any(token in option for token in forbidden) for option in options)

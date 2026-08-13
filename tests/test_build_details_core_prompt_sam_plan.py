import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_details_core_prompt_sam_plan.py"
    spec = importlib.util.spec_from_file_location("details_core_prompt_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_point_prefers_cross_view_visibility_then_center():
    module = _module()
    visibility = np.asarray([
        [True, True, False, False],
        [True, False, True, False],
        [False, True, True, False],
    ])
    xyz = np.asarray([
        [10.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    # 0/1/2 都可见两次，几何中央的点 1 胜出。
    selected = module.select_consistent_prompt_point([2, 0, 1], [2, 0, 1], visibility, xyz)
    assert selected == 1


def test_prompt_point_returns_none_without_new_visible_frame():
    module = _module()
    visibility = np.zeros((2, 3), dtype=bool)
    xyz = np.zeros((3, 3), dtype=np.float64)
    assert module.select_consistent_prompt_point([0, 1], [0, 1], visibility, xyz) is None


def test_prompt_frames_are_ranked_and_clipped_to_rgb_bounds():
    module = _module()
    visibility = np.asarray([
        [True, True, False],
        [True, True, True],
        [True, False, False],
        [True, True, True],
    ])
    projections = np.zeros((4, 3, 2), dtype=np.float64)
    projections[:, 0] = np.asarray([[20, 10], [30, 20], [40, 30], [300, 20]])
    rows = module.select_prompt_frames(
        prompt_point=0,
        common_core_points=[0, 1, 2],
        available_frames=[3, 2, 0, 1],
        visibility=visibility,
        projections=projections,
        scaling_params=(2.0, 2.0),
        image_shape=(100, 100),
        max_prompt_frames=2,
    )
    # 帧 3 越界；帧 1 的共同核心可见点最多，其次按可见数与帧号取帧 0。
    assert [row["frame_index"] for row in rows] == [1, 0]
    assert rows[0]["prompt_xy"] == [15.0, 10.0]

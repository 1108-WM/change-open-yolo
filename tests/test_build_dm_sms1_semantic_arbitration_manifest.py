import numpy as np

from tools.build_dm_sms1_semantic_arbitration_manifest import (
    finite_candidate_classes,
    select_complementary_views,
)


def _view(frame_id, frame_index, ratio):
    return {
        "frame_id": str(frame_id), "frame_index": frame_index,
        "visible_ratio": ratio, "sam_mask_valid": True,
    }


def test_finite_candidates_preserve_sources_without_selecting_a_class():
    result = finite_candidate_classes(4, 9)
    assert result == [
        {"class_index": 4, "sources": ["frozen_control"]},
        {"class_index": 9, "sources": ["alpha_main"]},
    ]
    assert finite_candidate_classes(4, 4) == [
        {"class_index": 4, "sources": ["frozen_control", "alpha_main"]}
    ]


def test_complementary_views_choose_high_coverage_then_far_camera():
    views = [_view("0", 0, 0.90), _view("1", 1, 0.80), _view("2", 2, 0.70)]
    centers = {"0": np.array([0.0, 0.0, 0.0]), "1": np.array([0.1, 0.0, 0.0]),
               "2": np.array([4.0, 0.0, 0.0])}
    assert select_complementary_views(views, target_count=3, centers=centers) == [0, 2, 1]


def test_complementary_views_is_deterministic_without_pose_centres():
    views = [_view("10", 10, 0.6), _view("1", 1, 0.6), _view("2", 2, 0.4)]
    assert select_complementary_views(views, target_count=2, centers=None) == [1, 0]

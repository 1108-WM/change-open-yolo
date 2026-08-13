import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_track_gvc_feature_ledger.py"
    spec = importlib.util.spec_from_file_location("track_gvc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gvc_from_views_uses_most_visible_views_and_multiplies_box_and_mask_support():
    module = _module()
    features = module.gvc_from_views([
        {"frame_index": 2, "visible_point_count": 3, "matched_observation_id": 1, "box_iou": 0.5, "mask_point_support": 0.8, "gvc_frame_score": 0.4},
        {"frame_index": 1, "visible_point_count": 8, "matched_observation_id": 2, "box_iou": 0.6, "mask_point_support": 0.5, "gvc_frame_score": 0.3},
        {"frame_index": 3, "visible_point_count": 6, "matched_observation_id": -1, "box_iou": 0.0, "mask_point_support": 0.0, "gvc_frame_score": 0.0},
    ], max_views=2)
    assert features["gvc_selected_view_count"] == 2
    assert features["gvc_matched_view_count"] == 1
    assert features["gvc_selected_match_ratio"] == 0.5
    assert features["gvc_score"] == 0.15
    assert [row["frame_index"] for row in features["gvc_selected_frames"]] == [1, 3]


def test_gvc_retains_unmatched_visible_frames_as_zero_evidence():
    module = _module()
    features = module.gvc_from_views([
        {"frame_index": 1, "visible_point_count": 9, "matched_observation_id": -1, "box_iou": 0.0, "mask_point_support": 0.0, "gvc_frame_score": 0.0},
        {"frame_index": 2, "visible_point_count": 4, "matched_observation_id": 3, "box_iou": 0.8, "mask_point_support": 0.5, "gvc_frame_score": 0.4},
    ], max_views=2)
    assert features["gvc_eligible_view_count"] == 2
    assert features["gvc_score"] == 0.2


def test_native_relation_uses_frozen_masks_without_candidate_mutation():
    module = _module()
    masks = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    relation = module._native_relation(np.asarray([0, 1, 2]), masks)
    assert relation["native_top_candidate_id"] == 0
    assert relation["native_top_iou"] == 2 / 3

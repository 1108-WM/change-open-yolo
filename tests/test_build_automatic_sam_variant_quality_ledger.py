import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_variant_quality_ledger.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_variant_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_agnostic_gvc_uses_visible_views_without_semantic_input():
    ledger = _module()
    result = ledger.class_agnostic_gvc_from_views([
        {"frame_index": 1, "visible_point_count": 8, "matched_observation_id": 4, "box_iou": 0.5,
         "mask_point_support": 0.8, "gvc_frame_score": 0.4},
        {"frame_index": 2, "visible_point_count": 5, "matched_observation_id": -1, "box_iou": 0.0,
         "mask_point_support": 0.0, "gvc_frame_score": 0.0},
    ], max_views=2)
    assert result["gvc_score"] == 0.2
    assert result["gvc_matched_view_count"] == 1
    assert result["gvc_selected_frames"][0]["frame_index"] == 1


def test_variant_points_materialize_only_original_superpoint_atoms():
    ledger = _module()
    variant = {"base_superpoint_ids": [10], "added_superpoint_ids": [30]}
    points = ledger._variant_points(variant, {
        10: np.asarray([0, 1]), 20: np.asarray([2]), 30: np.asarray([3, 4]),
    })
    assert points.tolist() == [0, 1, 3, 4]


def test_native_relation_is_category_agnostic_and_does_not_mutate_masks():
    ledger = _module()
    masks = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    relation = ledger._native_relation(np.asarray([0, 1, 2]), masks)
    assert relation["native_top_candidate_id"] == 0
    assert relation["native_top_iou"] == 2 / 3
    assert "class_id" not in relation

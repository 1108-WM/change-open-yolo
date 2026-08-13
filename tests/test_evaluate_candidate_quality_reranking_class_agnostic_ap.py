import numpy as np
import pytest

from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (
    EXPECTED_NATIVE,
    _canonical_mask_audit,
    assert_baseline,
    safety_feature_row,
)


def _ledger_row(source):
    return {
        "scene_name": "scene0000_00",
        "candidate_source": source,
        "candidate_id": 0,
        "original_source_score": 0.7,
        "point_count": 2,
        "point_fraction_of_scene": 0.2,
        "source_frame_count": 99,
        "gvc_source_frame_excluded": {
            "gvc": {"mean": 0.1, "max": 0.2, "variance": 0.01},
            "selected_view_count": 3,
            "matched_selected_view_count": 2,
            "zero_support_selected_view_fraction": 1 / 3,
        },
    }


def _track():
    return {
        "track_id": 0,
        "frame_ids": ["0", "5", "10", "15"],
        "support_view_count": 4,
        "superpoint_count": 2,
        "merge_action_count": 1,
        "mean_consensus_rate": 0.8,
        "mean_edge_score": 0.9,
        "mean_node_quality": 0.7,
    }


def test_safety_feature_mapping_keeps_native_track_fields_missing():
    row = safety_feature_row(_ledger_row("native_mask3d_yoloworld"), None)
    assert row["source_frame_count"] is None
    assert row["mean_node_quality"] is None
    assert row["gvc_excluded_mean"] == 0.1


def test_safety_feature_mapping_supplies_track_structure_by_exact_id():
    ledger = _ledger_row("d2b_track")
    ledger["source_frame_count"] = 4
    row = safety_feature_row(ledger, _track())
    assert row["source_frame_count"] == 4
    assert row["superpoint_count"] == 2
    assert row["gvc_excluded_matched_view_count"] == 2


def test_safety_feature_mapping_accepts_preserved_noncontiguous_track_id():
    ledger = _ledger_row("d2b_track")
    ledger["candidate_id"] = 7
    ledger["source_frame_count"] = 4
    track = _track()
    track["track_id"] = 7
    row = safety_feature_row(ledger, track)
    assert row["candidate_id"] == 7


def test_mask_audit_checks_every_candidate_and_changes_digest():
    masks = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=bool)
    first = _canonical_mask_audit(masks, [2, 2])
    changed = masks.copy()
    changed[0, 0] = False
    changed[2, 0] = True
    second = _canonical_mask_audit(changed, [2, 2])
    assert first["canonical_element_sha256"] != second["canonical_element_sha256"]
    with pytest.raises(ValueError, match="点数"):
        _canonical_mask_audit(masks, [1, 2])


def test_frozen_baseline_gate_rejects_any_material_change():
    assert_baseline(dict(EXPECTED_NATIVE), EXPECTED_NATIVE, 1e-12)
    changed = dict(EXPECTED_NATIVE)
    changed["ap"] += 1e-6
    with pytest.raises(ValueError, match="基线复现失败"):
        assert_baseline(changed, EXPECTED_NATIVE, 1e-12)

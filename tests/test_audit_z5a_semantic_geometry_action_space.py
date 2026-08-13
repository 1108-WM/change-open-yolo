import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_z5a_semantic_geometry_action_space.py"
SPEC = importlib.util.spec_from_file_location("audit_z5a_semantic_geometry_action_space", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

frozen_oof_summary = MODULE.frozen_oof_summary
pair_geometry_contract = MODULE.pair_geometry_contract
semantic_summary = MODULE.semantic_summary


def test_pair_geometry_contract_requires_exact_frozen_union():
    track = np.asarray([1, 2, 3], dtype=np.int64)
    native = np.asarray([3, 4], dtype=np.int64)
    child = np.asarray([1, 2, 3, 4], dtype=np.int64)
    result = pair_geometry_contract(track, native, child)
    assert result["child_exactly_equals_parent_union"] is True
    assert result["overlap_type"] == "partial_overlap"
    assert result["shared_point_count"] == 1
    assert result["point_iou"] == pytest.approx(0.25)


def test_frozen_oof_summary_uses_original_score_only_for_union():
    rows = [{
        "candidate_id": 7, "class_index": 4, "original_score": 0.02,
        "oof_predictions": {"C_joint_yolo_alpha": 0.8},
        "label_ap_quality": 1.0,
    }]
    union = frozen_oof_summary(rows, "pair_union")
    track = frozen_oof_summary(rows, "track")
    assert union["top_score"] == pytest.approx(0.02)
    assert union["current_hybrid_score_source"] == "frozen_original_score"
    assert track["top_score"] == pytest.approx(0.8)
    assert "label_ap_quality" not in union


def test_semantic_summary_reports_probability_margin_entropy_and_agreement():
    node = {
        "node_index": 0, "semantic_evidence_node_key": "scene:0",
        "candidate_source": "track", "geometry_hash": "abc", "point_count": 3,
        "geometry_yolo_available": True, "inherited_yolo_available": True,
        "geometry_alpha_available": True, "inherited_alpha_available": True,
        "geometry_yolo_alpha_js": 0.2, "geometry_yolo_alpha_top1_agreement": 1,
        "inherited_yolo_alpha_js": 0.3, "inherited_yolo_alpha_top1_agreement": 0,
        "geometry_inherited_alpha_js": 0.1,
        "geometry_alpha_view_count": 2, "inherited_alpha_view_count": 2,
    }
    arrays = {
        "geometry_yolo": np.asarray([[0.7, 0.3]], dtype=np.float32),
        "inherited_yolo": np.asarray([[0.6, 0.4]], dtype=np.float32),
        "geometry_alpha": np.asarray([[0.8, 0.2]], dtype=np.float32),
        "inherited_alpha": np.asarray([[0.4, 0.6]], dtype=np.float32),
    }
    result = semantic_summary(node, arrays)
    assert result["geometry_yolo"]["top1_probability"] == pytest.approx(0.7)
    assert result["geometry_yolo"]["top1_top2_margin"] == pytest.approx(0.4)
    assert result["geometry_yolo"]["normalized_entropy"] > 0
    assert result["geometry_yolo_alpha_top1_agreement"] == 1

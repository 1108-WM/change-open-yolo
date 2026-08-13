import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_variant_competition_ledger.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_variant_competition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic(top_class, probabilities, margin, entropy, ratio):
    return {
        "semantic_evidence_top_class_index": top_class,
        "semantic_vote_margin": margin,
        "semantic_normalized_entropy": entropy,
        "semantic_top_class_view_ratio": ratio,
        "semantic_class_distribution": [
            {"class_index": index, "probability": probability} for index, probability in probabilities.items()
        ],
    }


def test_pair_record_reports_deltas_but_never_selects_winner():
    ledger = _module()
    base_quality = {
        "scene_name": "scene0000_00", "source_track_id": 1, "variant_id": "base",
        "variant_point_count": 10, "gvc_score": 0.3, "native_top_iou": 0.4,
        "variant_inside_top_native_ratio": 0.6, "gvc_selected_match_ratio": 0.7, "native_top_candidate_id": 4,
    }
    variant_quality = {**base_quality, "variant_id": "expanded", "variant_type": "one_hop_positive_evidence_addition",
                       "variant_point_count": 14, "gvc_score": 0.5, "native_top_iou": 0.3,
                       "variant_inside_top_native_ratio": 0.4, "gvc_selected_match_ratio": 0.8}
    pair = ledger._pair_record(
        base_quality, variant_quality, _semantic(1, {1: 0.8, 2: 0.2}, 0.7, 0.2, 0.8),
        _semantic(2, {1: 0.3, 2: 0.7}, 0.4, 0.5, 0.6),
    )
    assert pair["geometry_delta"]["point_count"] == 4
    assert pair["geometry_delta"]["gvc_score"] == 0.2
    assert pair["semantic_delta"]["top_class_changed"] is True
    assert pair["semantic_delta"]["distribution_js_divergence"] > 0.0
    assert "winner" not in pair

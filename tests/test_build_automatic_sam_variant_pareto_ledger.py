import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_variant_pareto_ledger.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_variant_pareto", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality(variant_id, kind, track_id, gvc):
    return {
        "variant_id": variant_id, "variant_type": kind, "source_track_id": track_id,
        "gvc_score": gvc, "native_top_candidate_id": 1, "native_top_iou": 0.2,
        "variant_inside_top_native_ratio": 0.3, "variant_point_count": 10,
    }


def _semantic(variant_id, top_class, margin, entropy):
    return {
        "variant_id": variant_id, "semantic_evidence_top_class_index": top_class,
        "semantic_vote_margin": margin, "semantic_normalized_entropy": entropy,
    }


def test_pareto_ledger_drops_only_strictly_dominated_same_semantic_variants():
    ledger = _module()
    quality = [
        _quality("base", "seed_superpoint_closure", 4, 0.4),
        _quality("better", "one_hop_positive_evidence_addition", 4, 0.5),
        _quality("conflict", "one_hop_positive_evidence_addition", 4, 0.9),
    ]
    semantic = [
        _semantic("base", 1, 0.4, 0.5),
        _semantic("better", 1, 0.6, 0.3),
        _semantic("conflict", 2, 0.8, 0.2),
    ]
    rows, skipped = ledger.build_pareto_records(quality, semantic)
    by_id = {row["variant_id"]: row for row in rows}
    assert skipped == []
    assert by_id["base"]["pareto_non_dominated"] is False
    assert by_id["base"]["dominating_variant_ids"] == ["better"]
    assert by_id["better"]["pareto_non_dominated"] is True
    assert by_id["conflict"]["pareto_non_dominated"] is True
    assert by_id["conflict"]["semantic_class_changed_from_seed"] is True

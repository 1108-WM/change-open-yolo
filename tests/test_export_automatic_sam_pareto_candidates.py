import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "export_automatic_sam_pareto_candidates.py"
    spec = importlib.util.spec_from_file_location("export_automatic_sam_pareto", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant(variant_id, track_id, base, added=(), kind="one_hop_positive_evidence_addition"):
    return {"variant_id": variant_id, "source_track_id": track_id, "variant_type": kind, "base_superpoint_ids": base, "added_superpoint_ids": added}


def _quality(variant_id, track_id, gvc):
    return {"variant_id": variant_id, "source_track_id": track_id, "gvc_score": gvc, "gvc_selected_view_count": 2, "gvc_matched_view_count": 2}


def _semantic(variant_id, track_id, cls, margin, entropy):
    return {"variant_id": variant_id, "source_track_id": track_id, "semantic_evidence_top_class_index": cls, "semantic_evidence_frame_count": 3, "semantic_vote_margin": margin, "semantic_normalized_entropy": entropy, "semantic_top_class_view_ratio": 1.0}


def _pareto(variant_id, track_id, non_dominated=True):
    return {"variant_id": variant_id, "source_track_id": track_id, "pareto_non_dominated": non_dominated}


def test_selects_one_best_non_dominated_variant_per_track():
    module = _module()
    variants = [_variant("a", 1, [10]), _variant("b", 1, [10], [11]), _variant("c", 1, [10], [12])]
    qualities = [_quality("a", 1, 0.6), _quality("b", 1, 0.8), _quality("c", 1, 0.99)]
    semantics = [_semantic("a", 1, 0, 0.5, 0.5), _semantic("b", 1, 1, 0.4, 0.4), _semantic("c", 1, 0, 1.0, 0.0)]
    pareto = [_pareto("a", 1), _pareto("b", 1), _pareto("c", 1, False)]
    selected, skipped = module.select_track_variants(variants, qualities, semantics, pareto, ["chair", "table"])
    assert [record["variant_id"] for record in selected] == ["b"]
    assert selected[0]["class_id"] == 1
    assert {item["reason"] for item in skipped} == {"pareto_dominated"}


def test_export_reconstructs_raw_superpoint_atoms_and_preserves_append_only_contract(tmp_path):
    module = _module()
    scene = "scene0000_00"
    for root in ("plan", "quality", "semantic", "pareto"):
        (tmp_path / root / scene).mkdir(parents=True)
    variant = _variant("track0001_add_sp8", 1, [7], [8])
    (tmp_path / "plan" / scene / "automatic_sam_growth_variant_plan.jsonl").write_text(json.dumps(variant) + "\n")
    (tmp_path / "quality" / scene / "automatic_sam_variant_quality_ledger.json").write_text(json.dumps([_quality(variant["variant_id"], 1, 0.75)]))
    (tmp_path / "semantic" / scene / "automatic_sam_variant_semantic_ledger.json").write_text(json.dumps([_semantic(variant["variant_id"], 1, 0, 0.8, 0.1)]))
    (tmp_path / "pareto" / scene / "automatic_sam_variant_pareto_ledger.json").write_text(json.dumps([_pareto(variant["variant_id"], 1)]))
    processed_dir = tmp_path / "processed" / scene
    processed_dir.mkdir(parents=True)
    processed = np.zeros((6, 10), dtype=np.float32)
    processed[:, 9] = np.asarray([7, 7, 8, 9, 8, 9])
    np.save(processed_dir / "0000_00.npy", processed)
    args = SimpleNamespace(variant_plan_root=tmp_path / "plan", quality_ledger_root=tmp_path / "quality", semantic_ledger_root=tmp_path / "semantic", pareto_ledger_root=tmp_path / "pareto", processed_scene_root=tmp_path / "processed", output_root=tmp_path / "output", labels=["chair"])
    args.output_root.mkdir()
    audit = module._export_scene(scene, args)
    payload = json.loads((args.output_root / scene / "backprojection_candidates.json").read_text())
    candidate = payload["candidates"][0]
    points = np.load(args.output_root / scene / candidate["seed_points_path"])["point_indices"]
    assert audit["exported_candidate_count"] == 1
    assert points.tolist() == [0, 1, 2, 4]
    assert candidate["score"] == candidate["gvc_score"] == 0.75
    assert payload["append_only_contract"] == {"native_candidates_mutated": False, "native_overlap_filtering": False}

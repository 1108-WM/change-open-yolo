import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_pareto_candidate_competition_ledger.py"
    spec = importlib.util.spec_from_file_location("pareto_candidate_competition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_same_class_and_cross_class_relations_without_decision(tmp_path):
    module = _module()
    candidates = []
    for candidate_id, class_id, points in ((1, 0, [0, 1, 2, 3]), (2, 0, [0, 1, 2, 3]), (3, 1, [0, 1, 2])):
        relative = f"points_{candidate_id}.npz"
        np.savez_compressed(tmp_path / relative, point_indices=np.asarray(points))
        candidates.append({
            "scene_name": "scene0000_00", "candidate_id": candidate_id, "source_track_id": candidate_id,
            "class_id": class_id, "class_name": f"class_{class_id}", "selected_variant_id": f"v{candidate_id}",
            "gvc_score": .5, "semantic_vote_margin": .5, "semantic_normalized_entropy": .2,
            "num_seed_points": len(points), "seed_points_path": relative,
            "native_relation_diagnostic": {"native_top_iou": .1, "variant_inside_top_native_ratio": .2},
        })
    records = {item["candidate_id"]: item for item in module.build_scene_records(candidates, tmp_path)}
    assert records[1]["same_class_best_candidate_id"] == 2
    assert records[1]["same_class_best_iou"] == 1.0
    assert records[1]["cross_class_best_candidate_id"] == 3
    assert records[1]["cross_class_best_iou"] == 0.75
    assert "不按阈值保留" in records[1]["decision_state"]

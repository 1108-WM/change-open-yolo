import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_automatic_sam_variant_semantic_ledger.py"
    spec = importlib.util.spec_from_file_location("automatic_sam_variant_semantic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_votes_keep_only_best_mask_evidence_per_class():
    ledger = _module()
    votes = ledger.frame_class_mask_votes(np.asarray([0, 1, 2, 3]), [
        {"observation_id": 3, "class_id": 1, "quality": 0.4, "points": np.asarray([0, 1, 2])},
        {"observation_id": 4, "class_id": 1, "quality": 0.9, "points": np.asarray([0, 1])},
        {"observation_id": 5, "class_id": 2, "quality": 0.8, "points": np.asarray([2, 3])},
    ])
    assert votes[1][0] == 0.9 * 0.5
    assert votes[1][1] == 4
    assert votes[2][0] == 0.8 * 0.5


def test_semantic_summary_reports_distribution_and_uncertainty_not_final_class():
    ledger = _module()
    result = ledger.summarize_semantic_votes({1: 0.8, 2: 0.4}, {1: 2, 2: 1}, 2, ["zero", "one", "two"])
    assert result["semantic_evidence_top_class_index"] == 1
    assert result["semantic_vote_margin"] == 0.5
    assert result["semantic_top_class_view_ratio"] == 1.0
    assert result["semantic_class_distribution"][0]["class_name"] == "one"
    assert "final_class" not in result

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "tools" / "build_visibility_counterevidence_ledger.py"
    spec = importlib.util.spec_from_file_location("visibility_counterevidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_best_observation_prefers_track_coverage_over_detector_score():
    module = _module()
    best = module._best_observation(
        np.asarray([1, 2, 3, 4]),
        [
            {"points": np.asarray([1]), "quality": 0.99, "observation_id": 1, "score": 1.0, "sam_score": 1.0},
            {"points": np.asarray([1, 2, 3]), "quality": 0.20, "observation_id": 2, "score": 1.0, "sam_score": 1.0},
        ],
    )
    assert best["observation_id"] == 2
    assert best["coverage"] == 0.75


def test_track_evidence_records_positive_and_visible_counterevidence_per_superpoint():
    module = _module()
    evidence = module._track_evidence(
        track_points=np.asarray([0, 1, 2, 3]),
        class_id=1,
        superpoints=np.asarray([10, 10, 20, 20]),
        visibility=np.asarray([[True, True, True, True], [True, True, True, True]]),
        observations_by_frame_class={
            (0, 1): [{"points": np.asarray([0, 1, 2]), "quality": 1.0, "observation_id": 4, "score": 1.0, "sam_score": 1.0}],
            (1, 1): [{"points": np.asarray([0, 1]), "quality": 1.0, "observation_id": 5, "score": 1.0, "sam_score": 1.0}],
        },
        min_visible_points=1,
        min_counterevidence_anchor_coverage=0.30,
    )
    assert evidence["positive_support_frame_count"] == 2
    assert evidence["counterevidence_eligible_frame_count"] == 2
    assert evidence["positive_weights"].tolist() == [2.0, 0.5]
    assert evidence["negative_weights"].tolist() == [0.0, 1.5]
    assert evidence["negative_margin_superpoint_count"] == 1


def test_candidate_relation_exposes_overlap_without_assigning_a_route():
    module = _module()
    masks = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8)
    relation = module._candidate_relation(np.asarray([0, 1, 2]), masks)
    assert relation["top_native_candidate_id"] == 0
    assert "route" not in relation

import numpy as np

from tools.audit_train_candidate_quality_dataset import GVC_FEATURES, TRACK_FEATURES
from tools.train_candidate_relative_quality_nested_q_oof import (
    NATIVE_SOURCE,
    TRACK_SOURCE,
    nested_candidate_q_predictions,
    relation_nested_quality_evidence,
    relation_matrix_with_nested_q,
)


def test_relation_nested_q_uses_native_geometry_median():
    rows = [{
        "scene_name": "scene0001_00",
        "track_id": 3,
        "native_member_candidate_ids": [0, 1],
        "features": {"point_iou": 0.5},
    }]
    lookup = {
        ("scene0001_00", TRACK_SOURCE, 3): 0.8,
        ("scene0001_00", NATIVE_SOURCE, 0): 0.2,
        ("scene0001_00", NATIVE_SOURCE, 1): 0.6,
    }
    matrix, evidence = relation_matrix_with_nested_q(rows, ("point_iou",), lookup)
    assert matrix.shape == (1, 4)
    assert np.allclose(matrix[0], [0.5, 0.8, 0.4, 0.4])
    assert np.isclose(evidence[0]["nested_native_q_range"], 0.4)


def test_relation_nested_quality_exposes_reliability_pair_features():
    rows = [{
        "scene_name": "scene0001_00",
        "track_id": 3,
        "native_member_candidate_ids": [0, 1],
    }]
    lookup = {
        ("scene0001_00", TRACK_SOURCE, 3): {"q": 0.8, "valid25": 0.9, "valid50": 0.4},
        ("scene0001_00", NATIVE_SOURCE, 0): {"q": 0.2, "valid25": 0.6, "valid50": 0.1},
        ("scene0001_00", NATIVE_SOURCE, 1): {"q": 0.6, "valid25": 0.8, "valid50": 0.3},
    }
    evidence = relation_nested_quality_evidence(rows, lookup)[0]
    assert np.isclose(evidence["nested_native_valid25_median"], 0.7)
    assert np.isclose(evidence["nested_valid25_pair_min"], 0.7)
    assert np.isclose(evidence["nested_valid25_pair_product"], 0.63)
    assert np.isclose(evidence["nested_q_delta_track_minus_native"], 0.4)


def _candidate(scene, source, candidate_id, q):
    row = {
        "scene_name": scene,
        "candidate_source": source,
        "candidate_id": candidate_id,
        "native_exact_geometry_group_size": 1,
        "original_source_score": q,
        "point_count": 100 + candidate_id,
        "point_fraction_of_scene": 0.1,
        "label_best_gt_iou": q,
    }
    for name in TRACK_FEATURES:
        if source == TRACK_SOURCE:
            row[name] = 1.0
    for name in GVC_FEATURES:
        row[name] = q
    return row


def test_nested_candidate_q_predicts_every_candidate_once():
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    rows = []
    for index, scene in enumerate(scenes):
        value = 0.2 + 0.6 * (index % 10) / 9
        rows.append(_candidate(scene, NATIVE_SOURCE, 0, value))
        rows.append(_candidate(scene, TRACK_SOURCE, 1, 1.0 - value))
    scene_to_fold = {scene: index % 5 for index, scene in enumerate(scenes)}
    lookup, diagnostics = nested_candidate_q_predictions(
        rows, outer_fold_index=0, scene_to_fold=scene_to_fold,
        protocol_name="unit_test",
    )
    assert len(lookup) == len(rows)
    assert all(0.0 <= value <= 1.0 for value in lookup.values())
    assert diagnostics["outer_train_scene_count"] == 80
    assert diagnostics["outer_validation_scene_count"] == 20
    assert len(diagnostics["inner_folds"]) == 4

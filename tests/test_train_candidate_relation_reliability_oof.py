import numpy as np
import pytest

from tools.train_candidate_relation_reliability_oof import (
    NESTED_FEATURES,
    R1_RAW_FEATURES,
    nested_reliability_fit_predict,
    reliability_labels,
    reliability_matrix,
)


def _row(scene, track_id, reliable):
    state = "same_target" if reliable else "unknown"
    return {
        "scene_name": scene,
        "track_id": track_id,
        "labels": {"reliable_pair": reliable, "target_state": state},
        "features": {name: 0.1 for name in R1_RAW_FEATURES},
    }


def test_reliability_label_contract_uses_unknown_only_as_gate_negative():
    rows = [_row("a", 0, True), _row("b", 1, False)]
    assert reliability_labels(rows).tolist() == [1, 0]
    rows[1]["labels"]["target_state"] = "same_target"
    with pytest.raises(ValueError, match="differs"):
        reliability_labels(rows)


def test_candidate_reliability_matrix_uses_nested_and_gt_free_features():
    rows = [_row("a", 0, True)]
    evidence = [{name: 0.2 for name in NESTED_FEATURES}]
    matrix, names = reliability_matrix(rows, evidence, "R1_candidate_reliability")
    assert matrix.shape == (1, len(names))
    assert set(NESTED_FEATURES) <= set(names)
    assert not any("gt_" in name or name.startswith("label_") for name in names)


def test_nested_reliability_fit_predict_preserves_natural_class_prior():
    rows, labels, matrix = [], [], []
    for scene_index in range(10):
        for relation_index in range(4):
            label = int(relation_index == 0)
            rows.append(_row(f"scene{scene_index}", relation_index, bool(label)))
            labels.append(label)
            matrix.append([label + scene_index * 0.01, relation_index])
    labels = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(matrix, dtype=np.float64)
    train = np.arange(32, dtype=np.int64)
    validation = np.arange(32, 40, dtype=np.int64)
    predictions, diagnostics = nested_reliability_fit_predict(
        rows, matrix, labels, train, validation, outer_fold_index=0
    )
    assert len(predictions) == 8
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert diagnostics["fit_class_balanced"] is False
    assert len(diagnostics["inner_folds"]) == 4

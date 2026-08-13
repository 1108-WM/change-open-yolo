from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE
from tools.evaluate_candidate_component_union_residual_applicability_mixture_oof import (
    compose_predictions,
)


def _row(source, candidate_id, keep):
    return {
        "scene_name": "s",
        "relation_component_id": 0,
        "candidate_source": source,
        "candidate_id": candidate_id,
        "keep_probability": keep,
        "label_component_unique_winner": 0,
        "label_calibrated_quality": 0.0,
        "label_best_gt_iou": 0.1,
        "label_harm_kind": "invalid_or_low_quality",
    }


def test_compose_predictions_uses_residual_head_only_when_residual_is_nonempty():
    incumbent = [
        _row(NATIVE_SOURCE, 0, 0.1),
        _row(TRACK_SOURCE, 1, 0.2),
        _row(TRACK_SOURCE, 2, 0.3),
    ]
    residual = [
        _row(NATIVE_SOURCE, 0, 0.8),
        _row(TRACK_SOURCE, 1, 0.7),
        _row(TRACK_SOURCE, 2, 0.6),
    ]
    output, counts = compose_predictions(
        incumbent, residual, {("s", 1): 1.0, ("s", 2): 0.0}
    )
    by_id = {(row["candidate_source"], row["candidate_id"]): row for row in output}
    assert by_id[(NATIVE_SOURCE, 0)]["keep_probability"] == 0.1
    assert by_id[(TRACK_SOURCE, 1)]["keep_probability"] == 0.2
    assert by_id[(TRACK_SOURCE, 2)]["keep_probability"] == 0.6
    assert counts == {
        "incumbent_native": 1,
        "incumbent_empty_track": 1,
        "residual_nonempty_track": 1,
    }

import pytest

from tools.build_train_candidate_track_marginal_harm_ledger import (
    DEMOTED_TRACK_SCORE,
    _controlled_tracks,
    _records_with_one_track_score,
    marginal_decision,
)


def test_marginal_decision_describes_suppression_effect():
    assert marginal_decision(1e-5) == "suppress_beneficial"
    assert marginal_decision(-1e-5) == "suppress_harmful"
    assert marginal_decision(0.0) == "neutral"
    assert DEMOTED_TRACK_SCORE == 0.0


def test_controlled_tracks_are_deterministic_and_unique():
    cache = {
        "components": [
            {"relation_component_id": 2, "track_ids": [8, 3]},
            {"relation_component_id": 5, "track_ids": [9]},
        ],
        "controlled_track_ids": {3, 8, 9},
    }
    assert _controlled_tracks(cache) == [(2, 3), (2, 8), (5, 9)]

    cache["components"][1]["track_ids"] = [8]
    with pytest.raises(ValueError, match="multiple components"):
        _controlled_tracks(cache)


def test_one_track_score_rewrite_is_local_and_does_not_mutate_cache(monkeypatch):
    observed = []

    def fake_record(gt, pred, threshold):
        observed.append((
            threshold,
            [row["confidence"] for row in pred["chair"]],
            [match["confidence"] for match in gt["chair"][0]["matched_pred"]],
        ))
        return ([], [], 0, True, True)

    monkeypatch.setattr(
        "tools.construct_c1c_global_feasible_ap_oracle_gt._record_from_matches",
        fake_record,
    )
    cache = {
        "uuid_by_candidate": {("track", 7): "target"},
        "pred": {"chair": [
            {"uuid": "native", "confidence": 1.0},
            {"uuid": "target", "confidence": 0.4},
            {"uuid": "other", "confidence": 0.3},
        ]},
        "gt": {"chair": [{"matched_pred": [
            {"uuid": "target", "confidence": 0.4},
            {"uuid": "other", "confidence": 0.3},
        ]}]},
    }
    _, audit = _records_with_one_track_score(cache, 7)
    assert audit["top_level_prediction_confidence_change_count"] == 1
    assert all(scores == [1.0, 0.0, 0.3] for _, scores, _ in observed)
    assert all(matches == [0.0, 0.3] for _, _, matches in observed)
    assert cache["pred"]["chair"][1]["confidence"] == 0.4
    assert cache["gt"]["chair"][0]["matched_pred"][0]["confidence"] == 0.4

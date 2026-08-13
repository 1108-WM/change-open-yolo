from tools.train_candidate_relative_quality_oof import (
    LEARNED_QUALITY_MARKERS,
    MODEL_FEATURES,
    _labels,
    _quality_rows,
)


def _row(target_state, quality_state, reliable=True):
    return {
        "labels": {
            "target_state": target_state,
            "relative_quality_state": quality_state,
            "reliable_pair": reliable,
        }
    }


def test_relative_quality_contract_excludes_learned_quality_features():
    for feature_names in MODEL_FEATURES.values():
        assert not any(
            marker in name for name in feature_names for marker in LEARNED_QUALITY_MARKERS
        )
        assert not any(name.startswith("label_") or "gt_" in name for name in feature_names)


def test_relative_quality_rows_only_keep_reliable_strict_preferences():
    rows = [
        _row("same_target", "prefer_track"),
        _row("same_target", "prefer_native"),
        _row("same_target", "equivalent_abstain"),
        _row("different_target_coexist", "coexist"),
        _row("unknown", "unknown", reliable=False),
    ]
    selected = _quality_rows(rows)
    assert len(selected) == 2
    assert _labels(selected).tolist() == [1, 0]

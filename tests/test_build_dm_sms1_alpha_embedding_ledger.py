import numpy as np

from tools.build_dm_sms1_alpha_embedding_ledger import finalize_scene_semantics


def _records(count=3):
    rows = []
    feature_index = 0
    for geometry_index in range(count):
        scales = []
        for scale_index in range(3):
            scales.append({
                "scale_index": scale_index,
                "feature_index": feature_index,
                "feature_valid": True,
            })
            feature_index += 1
        rows.append({
            "geometry_index": geometry_index,
            "views": [{
                "visible_ratio": 0.5 + 0.1 * geometry_index,
                "sam_mask_valid": True,
                "scales": scales,
            }],
        })
    return rows


def test_finalize_scene_semantics_uses_complete_population_and_tau_zero():
    records = _records(5)
    features = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [0.71, 0.704], [0.71, 0.704], [0.71, 0.704],
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
    ], dtype=np.float32)
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    aggregate, similarities, stats, summary = finalize_scene_semantics(
        records, features, text, threshold=0.0
    )
    assert aggregate.shape == (5, 2)
    assert similarities.shape == (5, 2)
    assert stats.shape == (2, 2)
    assert summary["population_complete"] is True
    assert summary["alpha_valid_count"] == 5
    assert [row["alpha_class_index"] for row in records] == [0, 0, 0, 0, 1]
    assert [row["sms_keep"] for row in records] == [True, True, True, False, True]


def test_finalize_scene_semantics_conservatively_keeps_incomplete_population():
    records = _records()
    records[1]["views"][0]["sam_mask_valid"] = False
    for scale in records[1]["views"][0]["scales"]:
        scale["feature_valid"] = False
    features = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (9, 1))
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _, similarities, stats, summary = finalize_scene_semantics(records, features, text)
    assert summary["population_complete"] is False
    assert summary["alpha_invalid_count"] == 1
    assert np.isnan(similarities[1]).all()
    assert np.isnan(stats).all()
    assert all(row["sms_keep"] is True for row in records)
    assert all(row["sms_valid"] is False for row in records)
    assert all(row["sms_reason"] == "incomplete_scene_population" for row in records)


def test_finalize_scene_semantics_ignores_explicitly_missing_view_when_other_view_is_complete():
    records = _records(1)
    missing_scales = []
    for scale_index in range(3):
        missing_scales.append({
            "scale_index": scale_index,
            "feature_index": 3 + scale_index,
            "feature_valid": False,
        })
    records[0]["views"].append({
        "visible_ratio": 0.2,
        "sam_mask_valid": False,
        "scales": missing_scales,
    })
    features = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
    ], dtype=np.float32)
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _, similarities, _, summary = finalize_scene_semantics(records, features, text)
    assert summary["population_complete"] is True
    assert summary["alpha_valid_count"] == 1
    assert np.allclose(similarities[0], [1.0, 0.0])
    assert records[0]["sms_keep"] is True

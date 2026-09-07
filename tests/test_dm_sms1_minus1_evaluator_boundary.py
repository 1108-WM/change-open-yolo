from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.dm_sms1_minus1_evaluator_boundary import (
    EXPECTED_FOREGROUND_COUNT,
    EXPECTED_MINUS1_COUNT,
    EXPECTED_MINUS1_PLAN_INDICES,
    EXPECTED_NATIVE_BACKGROUND_198_COUNT,
    EXPECTED_TOTAL_CANDIDATE_COUNT,
    audit_boundary_inputs,
    frozen_minus1_identities,
)
from tools.evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt import FrozenPredictionMapping


def _decision(index: int, class_index: int = -1) -> dict:
    minus1 = class_index == -1
    return {
        "scene_name": "scene_smoke",
        "plan_index": index,
        "plan_key": f"plan-{index}",
        "geometry_hash": f"geometry-{index}",
        "candidate_source": "track" if minus1 else "native",
        "canonical_frozen_class_index": class_index,
        "arbitrated_class_index": class_index,
        "class_changed": False,
        "decision_path": (
            "single_candidate_deterministic_keep" if minus1
            else "two_candidate_vlm_arbitration"
        ),
    }


def _minus1_decisions() -> list[dict]:
    return [_decision(index) for index in sorted(EXPECTED_MINUS1_PLAN_INDICES)]


def _write_cache(root: Path, rows: list[dict]) -> None:
    scene = root / "prediction_cache/scene_smoke"
    scene.mkdir(parents=True)
    count = len(rows)
    np.save(scene / "masks.npy", np.ones((2, count), dtype=bool), allow_pickle=False)
    np.save(
        scene / "frozen_classes.npy",
        np.asarray([row["canonical_frozen_class_index"] for row in rows], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(scene / "frozen_scores.npy", np.ones(count, dtype=np.float32), allow_pickle=False)
    for name, values in {
        "geometry_hashes.json": [row["geometry_hash"] for row in rows],
        "plan_keys.json": [row["plan_key"] for row in rows],
        "plan_indices.json": [row["plan_index"] for row in rows],
        "sources.json": [row["candidate_source"] for row in rows],
    }.items():
        (scene / name).write_text(json.dumps(values) + "\n")
    import hashlib
    files = {
        name: hashlib.sha256((scene / name).read_bytes()).hexdigest()
        for name in (
            "masks.npy", "frozen_classes.npy", "frozen_scores.npy",
            "geometry_hashes.json", "plan_keys.json", "plan_indices.json", "sources.json",
        )
    }
    (root / "summary.json").write_text(json.dumps({
        "scene_count": 1,
        "candidate_count": count,
        "scene_summaries": [{"scene_name": "scene_smoke", "file_sha256": files}],
    }) + "\n")


def test_exact_minus1_set_maps_to_background_only_at_evaluator_boundary(tmp_path: Path):
    rows = _minus1_decisions()
    identities = frozen_minus1_identities(rows)
    cache = tmp_path / "cache"
    _write_cache(cache, rows)
    decisions = {(row["scene_name"], row["plan_key"]): row for row in rows}
    control = FrozenPredictionMapping(
        ["scene_smoke"], cache, decisions, challenge=False,
        minus1_boundary_identities=identities,
    )
    challenge = FrozenPredictionMapping(
        ["scene_smoke"], cache, decisions, challenge=True,
        minus1_boundary_identities=identities,
    )
    control_prediction = dict(control.items())["scene_smoke"]
    challenge_prediction = dict(challenge.items())["scene_smoke"]
    assert len(identities) == EXPECTED_MINUS1_COUNT
    assert np.all(control_prediction["pred_classes"] == 198)
    assert np.array_equal(control_prediction["pred_classes"], challenge_prediction["pred_classes"])
    assert np.array_equal(control_prediction["pred_masks"], challenge_prediction["pred_masks"])
    assert np.array_equal(control_prediction["pred_scores"], challenge_prediction["pred_scores"])
    assert control.observed_minus1_boundary_identities == set(identities)
    assert challenge.observed_minus1_boundary_identities == set(identities)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_frozen_class_index", 0),
        ("arbitrated_class_index", 0),
        ("class_changed", True),
        ("decision_path", "two_candidate_vlm_arbitration"),
        ("candidate_source", "native"),
    ],
)
def test_minus1_contract_rejects_frozen_field_tampering(field: str, value):
    rows = _minus1_decisions()
    rows[0][field] = value
    with pytest.raises(ValueError):
        frozen_minus1_identities(rows)


def test_minus1_contract_rejects_omission_extra_and_plan_replacement():
    rows = _minus1_decisions()
    with pytest.raises(ValueError):
        frozen_minus1_identities(rows[:-1])
    extra = rows + [{**rows[0], "plan_key": "extra"}]
    with pytest.raises(ValueError):
        frozen_minus1_identities(extra)
    rows[0]["plan_index"] = 1
    with pytest.raises(ValueError):
        frozen_minus1_identities(rows)


def test_full_synthetic_cache_preserves_all_columns_and_audits_exact_counts(tmp_path: Path):
    minus1 = set(EXPECTED_MINUS1_PLAN_INDICES)
    background = []
    for index in range(EXPECTED_TOTAL_CANDIDATE_COUNT):
        if index not in minus1:
            background.append(index)
        if len(background) == EXPECTED_NATIVE_BACKGROUND_198_COUNT:
            break
    background = set(background)
    rows = [
        _decision(index, -1 if index in minus1 else (198 if index in background else 0))
        for index in range(EXPECTED_TOTAL_CANDIDATE_COUNT)
    ]
    cache = tmp_path / "cache"
    _write_cache(cache, rows)
    result = audit_boundary_inputs(["scene_smoke"], cache, rows)
    assert result["candidate_count"] == EXPECTED_TOTAL_CANDIDATE_COUNT
    assert result["minus1_to_background_count"] == EXPECTED_MINUS1_COUNT
    assert result["native_background_198_count"] == EXPECTED_NATIVE_BACKGROUND_198_COUNT
    assert result["foreground_candidate_count"] == EXPECTED_FOREGROUND_COUNT
    assert result["candidate_deletion_count"] == 0

    # Reordering only one cache field must break the per-plan provenance join.
    scene = cache / "prediction_cache/scene_smoke"
    plan_keys = json.loads((scene / "plan_keys.json").read_text())
    plan_keys[0], plan_keys[-1] = plan_keys[-1], plan_keys[0]
    (scene / "plan_keys.json").write_text(json.dumps(plan_keys) + "\n")
    with pytest.raises(ValueError):
        audit_boundary_inputs(["scene_smoke"], cache, rows)


def test_mapping_rejects_minus1_without_frozen_boundary_authorization(tmp_path: Path):
    rows = _minus1_decisions()
    cache = tmp_path / "cache"
    _write_cache(cache, rows)
    decisions = {(row["scene_name"], row["plan_key"]): row for row in rows}
    mapping = FrozenPredictionMapping(["scene_smoke"], cache, decisions, challenge=False)
    with pytest.raises(ValueError, match="outside evaluator contract"):
        dict(mapping.items())


def test_control_challenge_identity_divergence_is_rejected(tmp_path: Path):
    rows = _minus1_decisions()
    identities = frozen_minus1_identities(rows)
    cache = tmp_path / "cache"
    _write_cache(cache, rows)
    decisions = {(row["scene_name"], row["plan_key"]): row for row in rows}
    missing_one = frozenset(sorted(identities)[1:])
    mapping = FrozenPredictionMapping(
        ["scene_smoke"], cache, decisions, challenge=True,
        minus1_boundary_identities=missing_one,
    )
    with pytest.raises(ValueError, match="unexpected minus-one"):
        dict(mapping.items())

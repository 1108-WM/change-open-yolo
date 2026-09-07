from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.audit_fi1_s2_b0_control import audit
from tools.build_fi1_s2_b0_control import run


SCENE = "scene0307_00"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _semantic(index: int, source: str, candidate_id: int, class_index: int, score, digest: str, locator: dict, origin="legacy_member") -> dict:
    return {
        "hypothesis_index": index,
        "semantic_hypothesis_key": (
            f"{SCENE}:semantic:refined:{candidate_id}" if origin == "fi1_refined_union"
            else f"{SCENE}:semantic:legacy:{source}:{candidate_id}:{digest}"
        ),
        "geometry_evidence_key": f"{SCENE}:geometry_evidence:{digest}",
        "scene_name": SCENE,
        "geometry_hash": digest,
        "origin_kind": origin,
        "candidate_source": source,
        "candidate_id": candidate_id,
        "class_index": class_index,
        "class_valid": 0 <= class_index < 198,
        "legacy_frozen_score": score,
        "fi1_geometry_score": 0.123,
        "geometry_locator_read_only": locator,
        "candidate_retained": True,
        "candidate_deletion": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }


def _fixture(tmp_path: Path) -> argparse.Namespace:
    s1_root = tmp_path / "s1"
    z3_root = tmp_path / "z3"
    stream_root = tmp_path / "stream"
    combined_root = tmp_path / "combined"
    for path in (s1_root, z3_root, stream_root, combined_root):
        path.mkdir()
    native_root = stream_root / SCENE / "native_cache"
    track_root = stream_root / SCENE / "d2b_tracks_filtered" / SCENE
    native_root.mkdir(parents=True)
    track_root.mkdir(parents=True)
    points_root = tmp_path / "points"
    points_root.mkdir()
    track1 = points_root / "track1.npz"
    track40 = points_root / "track40.npz"
    union0 = points_root / "union0.npz"
    refined0 = points_root / "refined0.npz"
    np.savez(track1, point_indices=np.asarray([1, 2], dtype=np.int64))
    np.savez(track40, point_indices=np.asarray([3], dtype=np.int64))
    np.savez(union0, point_indices=np.asarray([2, 3, 4], dtype=np.int64))
    np.savez(refined0, point_indices=np.asarray([0, 4], dtype=np.int64))
    native_masks = np.zeros((6, 2), dtype=bool)
    native_masks[[0, 1], 0] = True
    native_masks[[4, 5], 1] = True
    np.save(native_root / f"{SCENE}_pred_masks.npy", native_masks)
    np.save(native_root / f"{SCENE}_pred_classes.npy", np.asarray([4, 198], dtype=np.int64))
    np.save(native_root / f"{SCENE}_pred_scores.npy", np.asarray([0.9, 0.0], dtype=np.float32))
    (track_root / "automatic_tracks.json").write_text(json.dumps({"tracks": [
        {"track_id": 1, "points_path": str(track1)},
        {"track_id": 40, "points_path": str(track40)},
    ]}))
    rows = [
        _semantic(0, "native", 0, 4, 0.9, "n0", {"kind": "native_mask_column", "column_index": 0, "masks_path": str(native_root / f"{SCENE}_pred_masks.npy")}),
        _semantic(1, "native", 1, 198, 0.0, "n1", {"kind": "native_mask_column", "column_index": 1, "masks_path": str(native_root / f"{SCENE}_pred_masks.npy")}),
        _semantic(2, "track", 1, 5, 0.5, "t1", {"kind": "point_indices_npz", "array_key": "point_indices", "points_path": str(track1)}),
        _semantic(3, "track", 40, -1, 0.2, "t40", {"kind": "point_indices_npz", "array_key": "point_indices", "points_path": str(track40)}),
        _semantic(4, "pair_union", 0, 6, 0.4, "u0", {"kind": "point_indices_npz", "array_key": "point_indices", "points_path": str(union0)}),
        _semantic(5, "refined_union", 0, 9, None, "r0", {"kind": "point_indices_npz", "array_key": "point_indices", "points_path": str(refined0)}, "fi1_refined_union"),
    ]
    _write_jsonl(s1_root / "semantic_hypothesis_ledger.jsonl", rows)
    (s1_root / "summary.json").write_text(json.dumps({"ground_truth_read": False, "ap_computed": False}))
    z3_rows = [
        {"scene_name": SCENE, "candidate_source": "native", "candidate_id": 0, "class_index": 4, "original_score": 0.9, "oof_predictions": {"C_joint_yolo_alpha": 0.7}},
        {"scene_name": SCENE, "candidate_source": "track", "candidate_id": 1, "class_index": 7, "original_score": 0.5, "oof_predictions": {"C_joint_yolo_alpha": 0.6}},
        {"scene_name": SCENE, "candidate_source": "pair_union", "candidate_id": 0, "class_index": 8, "original_score": 0.4, "oof_predictions": {"C_joint_yolo_alpha": 0.4}},
    ]
    _write_jsonl(z3_root / "oof_predictions.jsonl", z3_rows)
    (z3_root / "summary.json").write_text(json.dumps({"ground_truth_usage": "none", "row_count": 3}))
    safe = {"ground_truth_usage": "none"}
    (stream_root / "summary.json").write_text(json.dumps(safe))
    (combined_root / "summary.json").write_text(json.dumps(safe))
    _write_jsonl(combined_root / "pair_union_append_candidates.jsonl", [{
        "scene_name": SCENE, "candidate_id": 0, "points_path": str(union0)
    }])
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(SCENE + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"network2d": {"text_prompts": [f"c{i}" for i in range(198)]}}))
    prereg = tmp_path / "prereg.md"
    prereg.write_text("synthetic frozen preregistration\n")
    result_root = tmp_path / "result"
    common = dict(
        scene_list=scene_list, s1_root=s1_root, z3_root=z3_root,
        stream_records_root=stream_root, combined_plan_root=combined_root,
        config_path=config, preregistration_path=prereg, class_count=198,
        expected_scene_count=1, expected_semantic_hypothesis_count=6,
        expected_b0_in_scope_count=5, expected_b2_only_refined_union_count=1,
        expected_b0_materialized_count=4, expected_b0_native_count=2,
        expected_b0_track_count=1, expected_b0_pair_union_count=1,
        expected_native_background_sentinel_count=1,
        expected_invalid_track_boundary_exclusion_count=1,
        expected_track_class_difference_count=1,
        expected_pair_union_class_difference_count=1, expected_z3_row_count=3,
    )
    run(argparse.Namespace(**common, output_root=result_root))
    return argparse.Namespace(**common, result_root=result_root, output_root=tmp_path / "audit")


def test_clean_b0_control_and_audit(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = audit(args)
    assert result["audit_valid"] is True
    assert result["error_count"] == 0
    assert result["b0_materialized_count"] == 4
    assert result["semantic_hypothesis_count"] == 6


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(0),
        lambda rows: rows.reverse(),
        lambda rows: rows[5].__setitem__("b0_materialized", True),
        lambda rows: rows[3].__setitem__("b0_materialized", True),
        lambda rows: rows[1].__setitem__("b0_materialized", False),
        lambda rows: rows[2].__setitem__("b0_score", rows[2]["fi1_geometry_score"]),
        lambda rows: rows[2].__setitem__("b0_class_index", 5),
        lambda rows: rows[2].__setitem__("s1_class_index", 7),
    ],
)
def test_manifest_tampering_is_rejected(tmp_path: Path, mutate) -> None:
    args = _fixture(tmp_path)
    path = args.result_root / "b0_semantic_control_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mutate(rows)
    _write_jsonl(path, rows)
    args.output_root = tmp_path / "tampered_audit"
    result = audit(args)
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


@pytest.mark.parametrize("kind", ["mask", "class", "score"])
def test_cache_tampering_is_rejected(tmp_path: Path, kind: str) -> None:
    args = _fixture(tmp_path)
    root = args.result_root / "prediction_cache"
    if kind == "mask":
        path = root / f"{SCENE}_pred_masks.npy"
        value = np.load(path)
        value[0, 0] = ~value[0, 0]
    elif kind == "class":
        path = root / f"{SCENE}_pred_classes.npy"
        value = np.load(path)
        value[2] = 5
    else:
        path = root / f"{SCENE}_pred_scores.npy"
        value = np.load(path)
        value[2] = np.float32(0.123)
    np.save(path, value)
    args.output_root = tmp_path / "tampered_audit"
    result = audit(args)
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


def test_builder_rejects_folded_s1_contract(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    rows_path = args.s1_root / "semantic_hypothesis_ledger.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    rows.pop(2)
    for index, row in enumerate(rows):
        row["hypothesis_index"] = index
    _write_jsonl(rows_path, rows)
    with pytest.raises(ValueError):
        run(argparse.Namespace(**{
            key: value for key, value in vars(args).items()
            if key not in {"result_root", "output_root"}
        }, output_root=tmp_path / "folded"))

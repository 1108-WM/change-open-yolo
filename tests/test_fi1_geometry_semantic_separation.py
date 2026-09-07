from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools.audit_fi1_geometry_semantic_separation_ledgers import audit
from tools.build_fi1_geometry_semantic_separation_ledgers import build_records, run


def _flags() -> dict:
    return {
        "candidate_retained": True,
        "candidate_deletion": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
    }


def _legacy_member(scene: str, source: str, candidate_id: int, class_index: int, score: float, digest: str) -> dict:
    return {
        "scene_name": scene,
        "candidate_source": source,
        "original_candidate_source": source,
        "candidate_id": candidate_id,
        "frozen_class_index": class_index,
        "frozen_class_valid": True,
        "frozen_score": score,
        "point_count": 3,
        "geometry_hash": digest,
        "geometry_locator": {"kind": "synthetic", "id": digest},
        "semantic_provenance": {"kind": "synthetic", "id": candidate_id},
    }


def _inputs() -> tuple[list[dict], list[dict]]:
    scene, digest = "scene0000_00", "abc"
    legacy_members = [
        _legacy_member(scene, "native", 0, 4, 0.9, digest),
        _legacy_member(scene, "native", 1, 4, 0.7, digest),
        _legacy_member(scene, "native", 2, 9, 0.6, digest),
    ]
    legacy = [{
        "scene_name": scene,
        "geometry_key": f"{scene}:geometry:{digest}",
        "geometry_hash": digest,
        "point_count": 3,
        "member_count": 3,
        "members": legacy_members,
        "canonical_candidate_source": "native",
        "canonical_candidate_id": 0,
        "canonical_frozen_class_index": 4,
        "ground_truth_read": False,
        "ap_computed": False,
    }]
    original = {
        "plan_index": 0,
        "plan_key": f"{scene}:geometry:{digest}",
        "candidate_source": "native",
        "candidate_id": 0,
        "frozen_class_index": 4,
        "frozen_class_valid": True,
        "challenger_score": 0.55,
        "point_count": 3,
        "geometry_hash": digest,
        "geometry_locator_read_only": {"kind": "synthetic", "id": digest},
        "append_only": False,
        **_flags(),
    }
    refined = {
        "plan_index": 1,
        "plan_key": f"{scene}:union:0:refined",
        "candidate_source": "refined_union",
        "candidate_id": 0,
        "frozen_class_index": 4,
        "frozen_class_valid": True,
        "challenger_score": 0.2,
        "point_count": 3,
        "geometry_hash": digest,
        "geometry_locator_read_only": {"kind": "synthetic", "id": "refined"},
        "append_only": True,
        **_flags(),
    }
    fi1 = [{
        "scene_name": scene,
        "geometry_key": f"{scene}:visual_geometry:{digest}",
        "geometry_hash": digest,
        "point_count": 3,
        "canonical_geometry_locator": {"kind": "synthetic", "id": digest},
        "member_count": 2,
        "members": [original, refined],
        "ground_truth_read": False,
        "ap_computed": False,
    }]
    return legacy, fi1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _fixture(tmp_path: Path):
    legacy_rows, fi1_rows = _inputs()
    legacy_root, fi1_root = tmp_path / "legacy", tmp_path / "fi1"
    legacy_root.mkdir(); fi1_root.mkdir()
    _write_jsonl(legacy_root / "unique_geometry_ledger.jsonl", legacy_rows)
    _write_jsonl(fi1_root / "unique_geometry_ledger.jsonl", fi1_rows)
    safe = {
        "ground_truth_read": False, "ap_computed": False,
        "candidate_mutation": False, "geometry_mutation": False,
        "class_mutation": False, "score_mutation": False,
    }
    (legacy_root / "summary.json").write_text(json.dumps(safe))
    (fi1_root / "summary.json").write_text(json.dumps(safe))
    scene_list = tmp_path / "scenes.txt"; scene_list.write_text("scene0000_00\n")
    prereg = tmp_path / "prereg.md"; prereg.write_text("frozen synthetic preregistration\n")
    result_root, audit_root = tmp_path / "result", tmp_path / "audit"
    run(argparse.Namespace(
        scene_list=scene_list, legacy_unique_geometry_root=legacy_root,
        fi1_unique_geometry_root=fi1_root, preregistration_path=prereg,
        output_root=result_root, class_count=198, expected_scene_count=None,
        expected_legacy_geometry_count=None, expected_legacy_member_count=None,
        expected_native_member_count=None, expected_track_member_count=None,
        expected_pair_union_member_count=None, expected_fi1_candidate_count=None,
        expected_fi1_unique_geometry_count=None, expected_fi1_refined_union_count=None,
        expected_semantic_hypothesis_count=None,
    ))
    return argparse.Namespace(
        scene_list=scene_list, legacy_unique_geometry_root=legacy_root,
        fi1_unique_geometry_root=fi1_root, preregistration_path=prereg,
        result_root=result_root, output_root=audit_root, class_count=198,
    )


def test_build_preserves_same_and_different_class_members() -> None:
    legacy, fi1 = _inputs()
    geometry, semantic, stats = build_records(legacy, fi1, 198)
    assert stats == {
        "legacy_geometry_count": 1,
        "legacy_member_count": 3,
        "legacy_source_member_counts": {"native": 3},
        "fi1_candidate_count": 2,
        "fi1_unique_geometry_count": 1,
        "fi1_refined_union_count": 1,
        "semantic_hypothesis_count": 4,
    }
    assert [row["class_index"] for row in semantic] == [4, 4, 9, 4]
    assert [row["legacy_frozen_score"] for row in semantic] == [0.9, 0.7, 0.6, None]
    assert [row["fi1_geometry_score"] for row in semantic] == [0.55, 0.55, 0.55, 0.2]
    assert geometry[0]["semantic_hypothesis_count"] == 4
    assert geometry[0]["expensive_visual_evidence_execution_count"] == 1


def test_clean_synthetic_audit_passes(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = audit(args)
    assert result["audit_valid"] is True
    assert result["error_count"] == 0


def test_audit_rejects_frozen_count_contract_mismatch(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.expected_semantic_hypothesis_count = 5
    result = audit(args)
    assert result["audit_valid"] is False
    assert result["errors"]["frozen_count_contract::expected_semantic_hypothesis_count"] == 1


@pytest.mark.parametrize(
    ("ledger", "mutate"),
    [
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows.pop(1)),
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows[0].__setitem__("class_index", 17)),
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows[0].__setitem__("legacy_frozen_score", rows[0]["fi1_geometry_score"])),
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows[0].__setitem__("fi1_geometry_score", 0.99)),
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows[0].__setitem__("candidate_source", "track")),
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows[0].__setitem__("fi1_plan_key", "tampered")),
        ("semantic_hypothesis_ledger.jsonl", lambda rows: rows.reverse()),
        ("geometry_evidence_ledger.jsonl", lambda rows: rows[0].__setitem__("semantic_hypothesis_keys", rows[0]["semantic_hypothesis_keys"][:-1])),
        ("geometry_evidence_ledger.jsonl", lambda rows: rows[0].__setitem__("expensive_visual_evidence_execution_count", 2)),
        ("geometry_evidence_ledger.jsonl", lambda rows: rows[0].__setitem__("ground_truth_read", True)),
    ],
)
def test_tampering_is_rejected(tmp_path: Path, ledger: str, mutate) -> None:
    args = _fixture(tmp_path)
    path = args.result_root / ledger
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    mutate(rows)
    _write_jsonl(path, rows)
    args.output_root = tmp_path / "tampered_audit"
    result = audit(args)
    assert result["audit_valid"] is False
    assert result["error_count"] > 0


def test_upstream_candidate_fold_is_rejected_by_frozen_count_contract(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    legacy_path = args.legacy_unique_geometry_root / "unique_geometry_ledger.jsonl"
    damaged = [json.loads(line) for line in legacy_path.read_text().splitlines() if line.strip()]
    damaged[0]["members"].pop()
    damaged[0]["member_count"] = 2
    _write_jsonl(legacy_path, damaged)
    with pytest.raises(ValueError, match="legacy_member_count differs"):
        run(argparse.Namespace(
            scene_list=args.scene_list,
            legacy_unique_geometry_root=args.legacy_unique_geometry_root,
            fi1_unique_geometry_root=args.fi1_unique_geometry_root,
            preregistration_path=args.preregistration_path,
            output_root=tmp_path / "folded_result",
            class_count=198,
            expected_scene_count=1,
            expected_legacy_geometry_count=1,
            expected_legacy_member_count=3,
            expected_native_member_count=3,
            expected_track_member_count=0,
            expected_pair_union_member_count=0,
            expected_fi1_candidate_count=2,
            expected_fi1_unique_geometry_count=1,
            expected_fi1_refined_union_count=1,
            expected_semantic_hypothesis_count=4,
        ))

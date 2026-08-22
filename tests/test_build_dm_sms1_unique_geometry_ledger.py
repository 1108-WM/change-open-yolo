import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.build_dm_sms1_unique_geometry_ledger import run
from tools.audit_dm_sms1_unique_geometry_ledger import run as audit_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path, omit_track_score: bool = False) -> argparse.Namespace:
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen before GT\n")
    records = tmp_path / "records"
    native = records / scene / "native_cache"
    native.mkdir(parents=True)
    masks = np.asarray([
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=bool)
    np.save(native / f"{scene}_pred_masks.npy", masks)
    np.save(native / f"{scene}_pred_classes.npy", np.asarray([3, 4, 5]))
    np.save(native / f"{scene}_pred_scores.npy", np.asarray([0.8, 0.9, 0.7]))

    track_root = records / scene / "d2b_tracks_filtered" / scene
    points_root = track_root / "points"
    points_root.mkdir(parents=True)
    track0 = points_root / "track0.npz"
    track1 = points_root / "track1.npz"
    np.savez_compressed(track0, point_indices=np.asarray([0, 2]))
    np.savez_compressed(track1, point_indices=np.asarray([1, 3]))
    (track_root / "automatic_tracks.json").write_text(json.dumps({"tracks": [
        {"track_id": 0, "points_path": str(track0), "point_count": 2},
        {"track_id": 1, "points_path": str(track1), "point_count": 2},
    ]}))

    semantic_root = tmp_path / "semantics"
    semantic_path = semantic_root / scene / "automatic_track_yoloworld_semantics.json"
    semantic_path.parent.mkdir(parents=True)
    semantic_path.write_text(json.dumps([
        {"scene_name": scene, "track_id": 0, "voted_class_index": 7},
        {"scene_name": scene, "track_id": 1, "voted_class_index": 8},
    ]))

    plan = tmp_path / "plan"
    scores = [
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 0, "planned_score": 0.8, "ground_truth_usage": "none", "ap_evaluation_run": False},
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 1, "planned_score": 0.9, "ground_truth_usage": "none", "ap_evaluation_run": False},
        {"scene_name": scene, "candidate_source": "native_mask3d_yoloworld", "candidate_id": 2, "planned_score": 0.7, "ground_truth_usage": "none", "ap_evaluation_run": False},
        {"scene_name": scene, "candidate_source": "d2b_track", "candidate_id": 0, "planned_score": 0.9, "ground_truth_usage": "none", "ap_evaluation_run": False},
        {"scene_name": scene, "candidate_source": "d2b_track", "candidate_id": 1, "planned_score": 0.6, "ground_truth_usage": "none", "ap_evaluation_run": False},
    ]
    if omit_track_score:
        scores.pop()
    _write_jsonl(plan / "frozen_score_plan.jsonl", scores)
    union_path = plan / scene / "union0.npz"
    union_path.parent.mkdir(parents=True, exist_ok=True)
    union_points = np.asarray([1, 3], dtype=np.int64)
    np.savez_compressed(union_path, point_indices=union_points)
    union_digest = hashlib.sha256()
    union_digest.update(len(union_points).to_bytes(8, "little"))
    union_digest.update(union_points.tobytes())
    _write_jsonl(plan / "pair_union_append_candidates.jsonl", [{
        "scene_name": scene,
        "candidate_source": "pair_union_append",
        "candidate_id": 0,
        "track_id": 1,
        "new_score": 0.95,
        "points_path": str(union_path),
        "proposal_point_count": 2,
        "proposal_geometry_sha256": union_digest.hexdigest(),
        "ground_truth_usage": "none",
        "ap_evaluation_run": False,
    }])
    return argparse.Namespace(
        scene_list=scene_list,
        records_root=records,
        native_cache=None,
        track_root=None,
        plan_root=plan,
        semantic_root=semantic_root,
        output_dir=tmp_path / "output",
        preregistration_path=prereg,
        expected_scene_count=1,
        class_count=198,
        max_scenes=None,
    )


def test_builds_cross_source_unique_geometry_ledger_with_frozen_canonical_order(tmp_path):
    args = _fixture(tmp_path)
    summary = run(args)
    rows = [json.loads(line) for line in (
        args.output_dir / "unique_geometry_ledger.jsonl"
    ).read_text().splitlines()]

    assert summary["member_count"] == 6
    assert summary["unique_geometry_count"] == 2
    assert summary["duplicate_member_count"] == 4
    assert summary["duplicate_geometry_output_count"] == 0
    assert summary["ground_truth_read"] is False
    assert summary["ap_computed"] is False
    by_points = {row["point_count"]: [] for row in rows}
    assert len(rows) == 2
    first = next(row for row in rows if row["canonical_frozen_class_index"] == 4)
    assert first["member_count"] == 3
    assert first["canonical_candidate_source"] == "native"
    assert first["canonical_candidate_id"] == 1
    second = next(row for row in rows if row["canonical_frozen_class_index"] == 8)
    assert second["member_count"] == 3
    assert second["canonical_candidate_source"] == "pair_union"
    assert second["canonical_frozen_score"] == 0.95
    assert second["canonical_geometry_locator"]["kind"] == "point_indices_npz"

    audit = audit_run(argparse.Namespace(
        ledger_root=args.output_dir,
        scene_list=args.scene_list,
        output_dir=tmp_path / "audit",
        expected_scene_count=1,
        class_count=198,
    ))
    assert audit["audit_valid"] is True
    assert audit["locator_reconstruction_error_count"] == 0
    assert audit["canonical_rule_error_count"] == 0


def test_rejects_incomplete_frozen_score_coverage(tmp_path):
    args = _fixture(tmp_path, omit_track_score=True)
    with pytest.raises(ValueError, match="frozen score coverage mismatch"):
        run(args)

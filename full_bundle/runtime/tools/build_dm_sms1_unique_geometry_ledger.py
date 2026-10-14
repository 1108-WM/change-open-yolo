#!/usr/bin/env python3
"""Build the GT-free DM-SMS-1 unique-geometry proposal ledger.

The builder folds exact geometry across the frozen native, filtered-track,
and append-only pair-union sources.  It records lightweight locators rather
than copying point arrays and chooses one canonical frozen prediction using
the preregistered score/source/id order.  It never opens ground truth and does
not run an evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dm_sms_core import canonical_member, exact_geometry_groups, geometry_hash


SOURCE_ORDER = ("native", "track", "pair_union")
SOURCE_RANK = {source: index for index, source in enumerate(SOURCE_ORDER)}
SOURCE_ALIASES = {
    "native": "native",
    "native_mask3d_yoloworld": "native",
    "track": "track",
    "d2b_track": "track",
    "pair_union": "pair_union",
    "pair_union_append": "pair_union",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {path}:{line_number}") from error
    return rows


def _scene_plan_rows(plan_root: Path, scene: str, filename: str) -> list[dict]:
    per_scene = plan_root / scene / filename
    source = per_scene if per_scene.is_file() else plan_root / filename
    rows = [row for row in _read_jsonl(source) if str(row.get("scene_name")) == scene]
    if not rows and filename == "frozen_score_plan.jsonl":
        raise ValueError(f"{scene}: frozen score plan has no rows")
    return rows


def _normalized_source(value: object) -> str:
    source = SOURCE_ALIASES.get(str(value))
    if source is None:
        raise ValueError(f"unexpected candidate source: {value!r}")
    return source


def _score_index(rows: list[dict], scene: str) -> dict[tuple[str, int], dict]:
    result = {}
    for row in rows:
        if str(row.get("scene_name")) != scene:
            raise ValueError(f"{scene}: score plan contains a different scene")
        if row.get("ground_truth_usage") != "none" or row.get("ap_evaluation_run") is not False:
            raise ValueError(f"{scene}: score row violates no-GT/no-AP contract")
        source = _normalized_source(row.get("candidate_source"))
        if source == "pair_union":
            raise ValueError(f"{scene}: pair-union must come from the append-only plan")
        candidate_id = int(row["candidate_id"])
        key = (source, candidate_id)
        if key in result:
            raise ValueError(f"{scene}: duplicate score identity {key}")
        score = float(row["planned_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{scene}: invalid frozen score for {key}")
        result[key] = row
    return result


def _load_points(path: Path, point_count: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        if "point_indices" not in payload:
            raise ValueError(f"point_indices is missing from {path}")
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points) or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid or empty point geometry in {path}")
    return points


def _track_semantics(semantic_root: Path, scene: str) -> dict[int, dict]:
    path = semantic_root / scene / "automatic_track_yoloworld_semantics.json"
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{scene}: track semantic payload is not a list")
    result = {}
    for row in rows:
        if str(row.get("scene_name", scene)) != scene:
            raise ValueError(f"{scene}: semantic row scene mismatch")
        track_id = int(row["track_id"])
        if track_id in result:
            raise ValueError(f"{scene}: duplicate semantic track id {track_id}")
        result[track_id] = row
    return result


def _member(
    *, scene: str, source: str, original_source: str, candidate_id: int,
    class_index: int, class_valid: bool, score: float, point_count: int,
    digest: str, locator: dict, semantic_provenance: dict,
) -> dict:
    return {
        "scene_name": scene,
        "candidate_source": source,
        "original_candidate_source": original_source,
        "candidate_id": int(candidate_id),
        "frozen_class_index": int(class_index),
        "frozen_class_valid": bool(class_valid),
        "frozen_score": float(score),
        "point_count": int(point_count),
        "geometry_hash": digest,
        "geometry_locator": locator,
        "semantic_provenance": semantic_provenance,
    }


def _register_geometry(
    groups: dict[str, dict], points: np.ndarray, members: list[dict], scene: str,
) -> None:
    digest = geometry_hash(points)
    if any(str(member["geometry_hash"]) != digest for member in members):
        raise AssertionError("member geometry digest was not finalized")
    existing = groups.get(digest)
    if existing is None:
        groups[digest] = {"points": points, "members": list(members)}
        return
    if not np.array_equal(existing["points"], points):
        raise ValueError(f"{scene}: SHA-1 geometry collision detected for {digest}")
    existing["members"].extend(members)


def _scene_ledger(scene: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    scene_root = args.records_root / scene
    prefix = (
        args.native_cache / f"{scene}_pred_"
        if args.native_cache is not None
        else scene_root / "native_cache" / f"{scene}_pred_"
    )
    masks_path = Path(str(prefix) + "masks.npy")
    classes_path = Path(str(prefix) + "classes.npy")
    scores_path = Path(str(prefix) + "scores.npy")
    masks = np.load(masks_path, mmap_mode="r")
    classes = np.asarray(np.load(classes_path), dtype=np.int64)
    native_scores = np.asarray(np.load(scores_path), dtype=np.float32)
    if masks.ndim != 2 or masks.shape[1] != len(classes) or len(classes) != len(native_scores):
        raise ValueError(f"{scene}: native cache dimensions disagree")
    if np.any(classes < 0) or np.any(classes > args.class_count):
        raise ValueError(f"{scene}: native class index is outside the frozen class space")
    point_count = int(masks.shape[0])

    track_path = (
        args.track_root / scene / "automatic_tracks.json"
        if args.track_root is not None
        else scene_root / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    )
    track_payload = json.loads(track_path.read_text())
    tracks = list(track_payload.get("tracks", []))
    track_by_id = {}
    for track in tracks:
        track_id = int(track["track_id"])
        if track_id in track_by_id:
            raise ValueError(f"{scene}: duplicate filtered track id {track_id}")
        track_by_id[track_id] = track
    semantics = _track_semantics(args.semantic_root, scene)
    if set(semantics) != set(track_by_id):
        missing = sorted(set(track_by_id) - set(semantics))
        extra = sorted(set(semantics) - set(track_by_id))
        raise ValueError(f"{scene}: track semantic coverage mismatch; missing={missing}, extra={extra}")

    score_rows = _scene_plan_rows(args.plan_root, scene, "frozen_score_plan.jsonl")
    scores = _score_index(score_rows, scene)
    expected_score_keys = {
        *(("native", candidate_id) for candidate_id in range(masks.shape[1])),
        *(("track", track_id) for track_id in track_by_id),
    }
    if set(scores) != expected_score_keys:
        missing = sorted(expected_score_keys - set(scores))
        extra = sorted(set(scores) - expected_score_keys)
        raise ValueError(f"{scene}: frozen score coverage mismatch; missing={missing}, extra={extra}")

    groups: dict[str, dict] = {}
    native_group_count = 0
    for native_ids in exact_geometry_groups(masks):
        points = np.flatnonzero(np.asarray(masks[:, native_ids[0]], dtype=bool)).astype(np.int64)
        if not len(points):
            raise ValueError(f"{scene}: empty native geometry group")
        digest = geometry_hash(points)
        members = []
        for candidate_id in native_ids:
            score_row = scores[("native", int(candidate_id))]
            members.append(_member(
                scene=scene,
                source="native",
                original_source=str(score_row["candidate_source"]),
                candidate_id=int(candidate_id),
                class_index=int(classes[candidate_id]),
                class_valid=bool(int(classes[candidate_id]) < args.class_count),
                score=float(score_row["planned_score"]),
                point_count=len(points),
                digest=digest,
                locator={
                    "kind": "native_mask_column",
                    "masks_path": str(masks_path),
                    "column_index": int(candidate_id),
                },
                semantic_provenance={
                    "kind": "native_cache_class",
                    "classes_path": str(classes_path),
                    "class_array_index": int(candidate_id),
                },
            ))
        _register_geometry(groups, points, members, scene)
        native_group_count += 1

    for track_id in sorted(track_by_id):
        track = track_by_id[track_id]
        path = _resolve(Path(track["points_path"]))
        points = _load_points(path, point_count)
        if "point_count" in track and int(track["point_count"]) != len(points):
            raise ValueError(f"{scene}: track {track_id} point-count mismatch")
        digest = geometry_hash(points)
        score_row = scores[("track", track_id)]
        semantic = semantics[track_id]
        semantic_class = int(semantic.get("voted_class_index", -1))
        if semantic_class > args.class_count:
            raise ValueError(f"{scene}: track {track_id} class is outside the frozen class space")
        member = _member(
            scene=scene,
            source="track",
            original_source=str(score_row["candidate_source"]),
            candidate_id=track_id,
            class_index=semantic_class,
            class_valid=bool(0 <= semantic_class < args.class_count),
            score=float(score_row["planned_score"]),
            point_count=len(points),
            digest=digest,
            locator={
                "kind": "point_indices_npz",
                "points_path": str(path),
                "array_key": "point_indices",
            },
            semantic_provenance={
                "kind": "track_yoloworld_vote",
                "semantic_path": str(
                    args.semantic_root / scene / "automatic_track_yoloworld_semantics.json"
                ),
                "track_id": track_id,
            },
        )
        _register_geometry(groups, points, [member], scene)

    union_rows = _scene_plan_rows(args.plan_root, scene, "pair_union_append_candidates.jsonl")
    seen_union_ids = set()
    for row in sorted(union_rows, key=lambda value: int(value["candidate_id"])):
        if row.get("ground_truth_usage") != "none" or row.get("ap_evaluation_run") is not False:
            raise ValueError(f"{scene}: pair-union row violates no-GT/no-AP contract")
        source = _normalized_source(row.get("candidate_source"))
        if source != "pair_union":
            raise ValueError(f"{scene}: invalid pair-union source")
        candidate_id = int(row["candidate_id"])
        if candidate_id in seen_union_ids:
            raise ValueError(f"{scene}: duplicate pair-union id {candidate_id}")
        seen_union_ids.add(candidate_id)
        track_id = int(row.get("selected_track_id", row.get("track_id", -1)))
        if track_id not in track_by_id:
            raise ValueError(f"{scene}: pair-union {candidate_id} refers to missing track {track_id}")
        path = _resolve(Path(row["points_path"]))
        points = _load_points(path, point_count)
        if "proposal_point_count" in row and int(row["proposal_point_count"]) != len(points):
            raise ValueError(f"{scene}: pair-union {candidate_id} point-count mismatch")
        union_digest = hashlib.sha256()
        union_digest.update(len(points).to_bytes(8, "little"))
        union_digest.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
        sha256_digest = union_digest.hexdigest()
        if row.get("proposal_geometry_sha256") not in (None, sha256_digest):
            raise ValueError(f"{scene}: pair-union {candidate_id} SHA-256 mismatch")
        score = float(row["new_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{scene}: invalid pair-union score {candidate_id}")
        digest = geometry_hash(points)
        inherited_class = int(semantics[track_id].get("voted_class_index", -1))
        if inherited_class > args.class_count:
            raise ValueError(
                f"{scene}: pair-union {candidate_id} class is outside the frozen class space"
            )
        member = _member(
            scene=scene,
            source="pair_union",
            original_source=str(row["candidate_source"]),
            candidate_id=candidate_id,
            class_index=inherited_class,
            class_valid=bool(0 <= inherited_class < args.class_count),
            score=score,
            point_count=len(points),
            digest=digest,
            locator={
                "kind": "point_indices_npz",
                "points_path": str(path),
                "array_key": "point_indices",
            },
            semantic_provenance={
                "kind": "selected_track_yoloworld_vote",
                "semantic_path": str(
                    args.semantic_root / scene / "automatic_track_yoloworld_semantics.json"
                ),
                "selected_track_id": track_id,
            },
        )
        _register_geometry(groups, points, [member], scene)

    records = []
    source_counts = Counter()
    canonical_source_counts = Counter()
    cross_source_duplicate_count = 0
    invalid_class_member_count = 0
    for digest in sorted(groups):
        members = sorted(
            groups[digest]["members"],
            key=lambda row: (SOURCE_RANK[row["candidate_source"]], int(row["candidate_id"])),
        )
        values = np.asarray([float(row["frozen_score"]) for row in members], dtype=np.float64)
        ranks = np.asarray([SOURCE_RANK[row["candidate_source"]] for row in members], dtype=np.int64)
        candidate_ids = np.asarray([int(row["candidate_id"]) for row in members], dtype=np.int64)
        canonical_index = canonical_member(
            list(range(len(members))), values, ranks, candidate_ids
        )
        canonical = members[canonical_index]
        sources = sorted({str(row["candidate_source"]) for row in members}, key=SOURCE_RANK.get)
        source_counts.update(str(row["candidate_source"]) for row in members)
        canonical_source_counts[str(canonical["candidate_source"])] += 1
        cross_source_duplicate_count += int(len(sources) > 1)
        invalid_class_member_count += sum(not row["frozen_class_valid"] for row in members)
        records.append({
            "scene_name": scene,
            "geometry_hash": digest,
            "geometry_key": f"{scene}:geometry:{digest}",
            "point_count": int(len(groups[digest]["points"])),
            "member_count": len(members),
            "member_sources": sources,
            "members": members,
            "canonical_member_index": int(canonical_index),
            "canonical_candidate_source": str(canonical["candidate_source"]),
            "canonical_candidate_id": int(canonical["candidate_id"]),
            "canonical_frozen_class_index": int(canonical["frozen_class_index"]),
            "canonical_frozen_class_valid": bool(canonical["frozen_class_valid"]),
            "canonical_frozen_score": float(canonical["frozen_score"]),
            "canonical_geometry_locator": canonical["geometry_locator"],
            "canonical_rule": "max frozen score; tie native < track < pair_union; tie candidate id",
            "ground_truth_usage": "none",
            "ap_computed": False,
        })

    member_count = sum(len(row["members"]) for row in records)
    summary = {
        "scene_name": scene,
        "point_count": point_count,
        "native_prediction_count": int(masks.shape[1]),
        "native_exact_geometry_group_count": native_group_count,
        "filtered_track_count": len(tracks),
        "pair_union_append_candidate_count": len(union_rows),
        "source_member_counts": dict(sorted(source_counts.items())),
        "canonical_source_counts": dict(sorted(canonical_source_counts.items())),
        "member_count": member_count,
        "unique_geometry_count": len(records),
        "duplicate_member_count": member_count - len(records),
        "cross_source_duplicate_geometry_count": cross_source_duplicate_count,
        "invalid_frozen_class_member_count": invalid_class_member_count,
        "invalid_canonical_frozen_class_count": sum(
            not row["canonical_frozen_class_valid"] for row in records
        ),
    }
    return records, summary


def run(args: argparse.Namespace) -> dict:
    args.scene_list = _resolve(args.scene_list)
    args.records_root = _resolve(args.records_root)
    args.native_cache = (
        None if args.native_cache is None else _resolve(args.native_cache)
    )
    args.track_root = None if args.track_root is None else _resolve(args.track_root)
    args.plan_root = _resolve(args.plan_root)
    args.semantic_root = _resolve(args.semantic_root)
    args.output_dir = _resolve(args.output_dir)
    args.preregistration_path = _resolve(args.preregistration_path)
    scenes = sorted(_read_scenes(args.scene_list))
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"scene count {len(scenes)} differs from expected {args.expected_scene_count}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scene_summaries = []
    all_records = []
    try:
        with (args.output_dir / "unique_geometry_ledger.jsonl").open("w") as output:
            for index, scene in enumerate(scenes, 1):
                records, summary = _scene_ledger(scene, args)
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                all_records.extend(records)
                scene_summaries.append(summary)
                print(
                    f"[DM-SMS-1 ledger] {index}/{len(scenes)} {scene}: "
                    f"members={summary['member_count']} unique={summary['unique_geometry_count']}",
                    flush=True,
                )
        identities = [(row["scene_name"], row["geometry_hash"]) for row in all_records]
        source_members = Counter()
        canonical_sources = Counter()
        for summary in scene_summaries:
            source_members.update(summary["source_member_counts"])
            canonical_sources.update(summary["canonical_source_counts"])
        summary = {
            "version": "dm_sms1_unique_geometry_ledger_v1",
            "experiment": "DM-SMS-1A",
            "scene_count": len(scenes),
            "member_count": sum(row["member_count"] for row in scene_summaries),
            "unique_geometry_count": len(all_records),
            "duplicate_geometry_output_count": len(identities) - len(set(identities)),
            "duplicate_member_count": sum(row["duplicate_member_count"] for row in scene_summaries),
            "cross_source_duplicate_geometry_count": sum(
                row["cross_source_duplicate_geometry_count"] for row in scene_summaries
            ),
            "source_member_counts": dict(sorted(source_members.items())),
            "canonical_source_counts": dict(sorted(canonical_sources.items())),
            "invalid_frozen_class_member_count": sum(
                row["invalid_frozen_class_member_count"] for row in scene_summaries
            ),
            "invalid_canonical_frozen_class_count": sum(
                row["invalid_canonical_frozen_class_count"] for row in scene_summaries
            ),
            "canonical_rule": "max frozen score; exact tie native < track < pair_union; then candidate id",
            "geometry_hash_contract": "SHA-1 over sorted unique int64 scene point indices",
            "geometry_order_contract": "scene_name ascending, then geometry_hash ascending",
            "locator_contract": "native mask column or point_indices NPZ; point arrays are not copied",
            "contract_valid": len(identities) == len(set(identities)),
            "ground_truth_usage": "none",
            "ground_truth_read": False,
            "ap_computed": False,
            "embedding_computed": False,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "input_provenance": {
                "scene_list": str(args.scene_list),
                "scene_list_sha256": _sha256(args.scene_list),
                "records_root": str(args.records_root),
                "native_cache": None if args.native_cache is None else str(args.native_cache),
                "track_root": None if args.track_root is None else str(args.track_root),
                "plan_root": str(args.plan_root),
                "semantic_root": str(args.semantic_root),
                "preregistration_path": str(args.preregistration_path),
                "preregistration_sha256": _sha256(args.preregistration_path),
            },
            "scene_summaries": scene_summaries,
        }
        if not summary["contract_valid"]:
            raise AssertionError("duplicate unique-geometry output identities")
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return summary
    except Exception:
        # Keep a failed partial ledger visible for diagnosis; never write a
        # success summary unless all requested scenes pass every contract.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"),
    )
    parser.add_argument(
        "--records-root", type=Path,
        default=Path("/media/jia/软件1/scannet_train_stream/records_ncs"),
    )
    parser.add_argument(
        "--native-cache", type=Path,
        help="Optional flat native cache; defaults to records-root/<scene>/native_cache.",
    )
    parser.add_argument(
        "--track-root", type=Path,
        help="Optional shared filtered-track root containing <scene>/automatic_tracks.json.",
    )
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("docs/diagnostics/dm_sms1_unique_geometry_ncs_train100_20260817"),
    )
    parser.add_argument(
        "--preregistration-path", type=Path,
        default=Path("docs/DM_SMS1A_PREREGISTRATION_20260817.md"),
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

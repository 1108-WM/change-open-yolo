#!/usr/bin/env python3
"""Build GT-free geometry-evidence and semantic-hypothesis separation ledgers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fi1_geometry_semantic_separation_core import (  # noqa: E402
    GEOMETRY_LEDGER_NAME,
    SEMANTIC_LEDGER_NAME,
    VERSION,
    frozen_flags,
    geometry_evidence_key,
    legacy_semantic_hypothesis_key,
    plan_member_projection,
    read_jsonl,
    refined_semantic_hypothesis_key,
    sha256_file,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path: Path, expected_count: int | None) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    if expected_count is not None and len(scenes) != expected_count:
        raise ValueError("scene count differs from the preregistered contract")
    return scenes


def _safe_summary(path: Path, name: str) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("ground_truth_read") is not False or payload.get("ap_computed") is not False:
        raise ValueError(f"{name} is not a no-GT/no-AP input")
    for field in ("candidate_mutation", "geometry_mutation", "class_mutation", "score_mutation"):
        if field in payload and payload[field] is not False:
            raise ValueError(f"{name} reports forbidden {field}")
    return payload


def _validate_expected(actual: int, expected: int | None, name: str) -> None:
    if expected is not None and int(actual) != int(expected):
        raise ValueError(f"{name} differs: {actual} != {expected}")


def build_records(legacy_rows: list[dict], fi1_rows: list[dict], class_count: int) -> tuple[list[dict], list[dict], dict]:
    legacy_by_identity: dict[tuple[str, str], dict] = {}
    legacy_by_key: dict[str, dict] = {}
    legacy_member_counts = Counter()
    for row in legacy_rows:
        scene = str(row["scene_name"])
        digest = str(row["geometry_hash"])
        identity = (scene, digest)
        key = str(row["geometry_key"])
        members = list(row.get("members", []))
        if identity in legacy_by_identity or key in legacy_by_key:
            raise ValueError("legacy ledger contains duplicate geometry identity")
        if not members or int(row.get("member_count", -1)) != len(members):
            raise ValueError(f"{key}: invalid legacy members")
        for member in members:
            if str(member.get("scene_name")) != scene or str(member.get("geometry_hash")) != digest:
                raise ValueError(f"{key}: legacy member geometry identity differs")
            legacy_member_counts[str(member["candidate_source"])] += 1
        legacy_by_identity[identity] = row
        legacy_by_key[key] = row

    fi1_by_identity: dict[tuple[str, str], dict] = {}
    original_plan_by_legacy_key: dict[str, dict] = {}
    refined_members: list[tuple[str, dict]] = []
    plan_keys = set()
    plan_member_count = 0
    for row in fi1_rows:
        scene = str(row["scene_name"])
        digest = str(row["geometry_hash"])
        identity = (scene, digest)
        members = list(row.get("members", []))
        if identity in fi1_by_identity:
            raise ValueError("FI1 ledger contains duplicate geometry identity")
        if not members or int(row.get("member_count", -1)) != len(members):
            raise ValueError(f"{identity}: invalid FI1 plan members")
        for member in members:
            plan_member_count += 1
            plan_key = str(member.get("plan_key", ""))
            if not plan_key or plan_key in plan_keys:
                raise ValueError("FI1 ledger contains empty or duplicate plan_key")
            plan_keys.add(plan_key)
            if (
                str(member.get("geometry_hash")) != digest
                or int(member.get("point_count", -1)) != int(row["point_count"])
                or member.get("candidate_retained") is not True
                or member.get("candidate_deletion") is not False
                or member.get("geometry_mutation") is not False
                or member.get("class_mutation") is not False
                or member.get("score_mutation") is not False
            ):
                raise ValueError(f"{plan_key}: FI1 plan-member contract differs")
            if str(member["candidate_source"]) == "refined_union":
                if member.get("append_only") is not True:
                    raise ValueError(f"{plan_key}: refined union is not append-only")
                refined_members.append((scene, member))
            else:
                if member.get("append_only") is not False or plan_key in original_plan_by_legacy_key:
                    raise ValueError(f"{plan_key}: original FI1 plan-member contract differs")
                original_plan_by_legacy_key[plan_key] = member
        fi1_by_identity[identity] = row

    if not set(legacy_by_identity).issubset(fi1_by_identity):
        raise ValueError("FI1 geometry ledger does not cover every legacy geometry")
    if set(original_plan_by_legacy_key) != set(legacy_by_key):
        raise ValueError("original FI1 plan keys do not exactly cover legacy geometry keys")

    semantic_rows: list[dict] = []
    semantic_keys_by_geometry: dict[tuple[str, str], list[str]] = defaultdict(list)
    legacy_semantic_count_by_geometry = Counter()
    refined_semantic_count_by_geometry = Counter()

    for legacy in legacy_rows:
        scene = str(legacy["scene_name"])
        digest = str(legacy["geometry_hash"])
        identity = (scene, digest)
        plan_member = original_plan_by_legacy_key[str(legacy["geometry_key"])]
        if (
            str(plan_member["geometry_hash"]) != digest
            or str(plan_member["candidate_source"]) != str(legacy["canonical_candidate_source"])
            or int(plan_member["candidate_id"]) != int(legacy["canonical_candidate_id"])
            or int(plan_member["frozen_class_index"]) != int(legacy["canonical_frozen_class_index"])
        ):
            raise ValueError(f"{legacy['geometry_key']}: legacy/FI1 canonical join differs")
        for member in legacy["members"]:
            key = legacy_semantic_hypothesis_key(member)
            semantic_keys_by_geometry[identity].append(key)
            legacy_semantic_count_by_geometry[identity] += 1
            class_index = int(member["frozen_class_index"])
            semantic_rows.append({
                "hypothesis_index": len(semantic_rows),
                "semantic_hypothesis_key": key,
                "geometry_evidence_key": geometry_evidence_key(scene, digest),
                "scene_name": scene,
                "geometry_hash": digest,
                "origin_kind": "legacy_member",
                "candidate_source": str(member["candidate_source"]),
                "original_candidate_source": str(member["original_candidate_source"]),
                "candidate_id": int(member["candidate_id"]),
                "class_index": class_index,
                "class_valid": 0 <= class_index < class_count,
                "legacy_frozen_score": float(member["frozen_score"]),
                "fi1_geometry_score": float(plan_member["challenger_score"]),
                "score_fields_separated": True,
                "semantic_provenance": dict(member["semantic_provenance"]),
                "point_count": int(member["point_count"]),
                "geometry_locator_read_only": dict(member["geometry_locator"]),
                "fi1_plan_index": int(plan_member["plan_index"]),
                "fi1_plan_key": str(plan_member["plan_key"]),
                "fi1_candidate_source": str(plan_member["candidate_source"]),
                "append_only": False,
                **frozen_flags(),
            })

    for scene, member in sorted(refined_members, key=lambda item: int(item[1]["plan_index"])):
        digest = str(member["geometry_hash"])
        identity = (scene, digest)
        key = refined_semantic_hypothesis_key(member, scene)
        semantic_keys_by_geometry[identity].append(key)
        refined_semantic_count_by_geometry[identity] += 1
        class_index = int(member["frozen_class_index"])
        semantic_rows.append({
            "hypothesis_index": len(semantic_rows),
            "semantic_hypothesis_key": key,
            "geometry_evidence_key": geometry_evidence_key(scene, digest),
            "scene_name": scene,
            "geometry_hash": digest,
            "origin_kind": "fi1_refined_union",
            "candidate_source": "refined_union",
            "original_candidate_source": "fi1_d_v3_refined_union",
            "candidate_id": int(member["candidate_id"]),
            "class_index": class_index,
            "class_valid": 0 <= class_index < class_count,
            "legacy_frozen_score": None,
            "fi1_geometry_score": float(member["challenger_score"]),
            "score_fields_separated": True,
            "semantic_provenance": {
                "kind": "fi1_d_v3_refined_union_frozen_class",
                "fi1_plan_key": str(member["plan_key"]),
            },
            "point_count": int(member["point_count"]),
            "geometry_locator_read_only": dict(member["geometry_locator_read_only"]),
            "fi1_plan_index": int(member["plan_index"]),
            "fi1_plan_key": str(member["plan_key"]),
            "fi1_candidate_source": "refined_union",
            "append_only": True,
            **frozen_flags(),
        })

    semantic_keys = [str(row["semantic_hypothesis_key"]) for row in semantic_rows]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError("semantic hypothesis keys are not unique")

    geometry_rows = []
    for row in fi1_rows:
        scene = str(row["scene_name"])
        digest = str(row["geometry_hash"])
        identity = (scene, digest)
        keys = semantic_keys_by_geometry[identity]
        if not keys:
            raise ValueError(f"{identity}: geometry has no semantic hypotheses")
        geometry_rows.append({
            "geometry_index": len(geometry_rows),
            "geometry_evidence_key": geometry_evidence_key(scene, digest),
            "scene_name": scene,
            "geometry_hash": digest,
            "point_count": int(row["point_count"]),
            "geometry_locator_read_only": dict(row["canonical_geometry_locator"]),
            "fi1_plan_member_count": len(row["members"]),
            "fi1_plan_members": [plan_member_projection(member) for member in row["members"]],
            "legacy_semantic_member_count": int(legacy_semantic_count_by_geometry[identity]),
            "refined_semantic_member_count": int(refined_semantic_count_by_geometry[identity]),
            "semantic_hypothesis_count": len(keys),
            "semantic_hypothesis_keys": keys,
            "expensive_visual_evidence_execution_count": 1,
            **frozen_flags(),
        })

    stats = {
        "legacy_geometry_count": len(legacy_rows),
        "legacy_member_count": sum(legacy_member_counts.values()),
        "legacy_source_member_counts": dict(sorted(legacy_member_counts.items())),
        "fi1_candidate_count": plan_member_count,
        "fi1_unique_geometry_count": len(fi1_rows),
        "fi1_refined_union_count": len(refined_members),
        "semantic_hypothesis_count": len(semantic_rows),
    }
    return geometry_rows, semantic_rows, stats


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "legacy_unique_geometry_root", "fi1_unique_geometry_root",
        "preregistration_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list, args.expected_scene_count)
    legacy_summary_path = args.legacy_unique_geometry_root / "summary.json"
    legacy_ledger_path = args.legacy_unique_geometry_root / "unique_geometry_ledger.jsonl"
    fi1_summary_path = args.fi1_unique_geometry_root / "summary.json"
    fi1_ledger_path = args.fi1_unique_geometry_root / "unique_geometry_ledger.jsonl"
    for path in (
        legacy_summary_path, legacy_ledger_path, fi1_summary_path, fi1_ledger_path,
        args.preregistration_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    _safe_summary(legacy_summary_path, "legacy summary")
    _safe_summary(fi1_summary_path, "FI1 summary")
    legacy_rows = read_jsonl(legacy_ledger_path)
    fi1_rows = read_jsonl(fi1_ledger_path)
    if {str(row["scene_name"]) for row in legacy_rows} != set(scenes):
        raise ValueError("legacy scene coverage differs")
    if {str(row["scene_name"]) for row in fi1_rows} != set(scenes):
        raise ValueError("FI1 scene coverage differs")
    geometry_rows, semantic_rows, stats = build_records(legacy_rows, fi1_rows, args.class_count)
    for field in (
        "legacy_geometry_count", "legacy_member_count", "fi1_candidate_count",
        "fi1_unique_geometry_count", "fi1_refined_union_count", "semantic_hypothesis_count",
    ):
        _validate_expected(stats[field], getattr(args, f"expected_{field}"), field)
    if args.expected_native_member_count is not None:
        _validate_expected(
            stats["legacy_source_member_counts"].get("native", 0),
            args.expected_native_member_count, "legacy native member count",
        )
    if args.expected_track_member_count is not None:
        _validate_expected(
            stats["legacy_source_member_counts"].get("track", 0),
            args.expected_track_member_count, "legacy track member count",
        )
    if args.expected_pair_union_member_count is not None:
        _validate_expected(
            stats["legacy_source_member_counts"].get("pair_union", 0),
            args.expected_pair_union_member_count, "legacy pair-union member count",
        )

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging path exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        geometry_path = staging / GEOMETRY_LEDGER_NAME
        semantic_path = staging / SEMANTIC_LEDGER_NAME
        geometry_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in geometry_rows))
        semantic_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic_rows))
        summary = {
            "version": VERSION,
            "contract": "geometry evidence is shared; semantic hypotheses are never implicitly folded",
            "scene_count": len(scenes),
            **stats,
            "candidate_deletion_count": 0,
            "geometry_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "ground_truth_read": False,
            "ap_computed": False,
            "files": {
                "geometry_evidence_ledger": GEOMETRY_LEDGER_NAME,
                "semantic_hypothesis_ledger": SEMANTIC_LEDGER_NAME,
            },
            "hashes": {
                "geometry_evidence_ledger": sha256_file(geometry_path),
                "semantic_hypothesis_ledger": sha256_file(semantic_path),
            },
            "input_provenance": {
                "scene_list": sha256_file(args.scene_list),
                "legacy_summary": sha256_file(legacy_summary_path),
                "legacy_ledger": sha256_file(legacy_ledger_path),
                "fi1_summary": sha256_file(fi1_summary_path),
                "fi1_ledger": sha256_file(fi1_ledger_path),
                "preregistration": sha256_file(args.preregistration_path),
            },
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _optional_int(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--legacy-unique-geometry-root", type=Path, required=True)
    parser.add_argument("--fi1-unique-geometry-root", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--expected-scene-count", type=_optional_int, default=312)
    parser.add_argument("--expected-legacy-geometry-count", type=_optional_int, default=39198)
    parser.add_argument("--expected-legacy-member-count", type=_optional_int, default=213233)
    parser.add_argument("--expected-native-member-count", type=_optional_int, default=187200)
    parser.add_argument("--expected-track-member-count", type=_optional_int, default=18141)
    parser.add_argument("--expected-pair-union-member-count", type=_optional_int, default=7892)
    parser.add_argument("--expected-fi1-candidate-count", type=_optional_int, default=39304)
    parser.add_argument("--expected-fi1-unique-geometry-count", type=_optional_int, default=39250)
    parser.add_argument("--expected-fi1-refined-union-count", type=_optional_int, default=106)
    parser.add_argument("--expected-semantic-hypothesis-count", type=_optional_int, default=213339)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

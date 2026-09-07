#!/usr/bin/env python3
"""Independently audit FI1 geometry-evidence/semantic-hypothesis ledgers."""

from __future__ import annotations

import argparse
import json
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


def _error(errors: Counter, name: str, condition: bool) -> None:
    if condition:
        errors[name] += 1


def _expected(args: argparse.Namespace, name: str) -> int | None:
    value = getattr(args, name, None)
    return None if value is None else int(value)


def _independently_reconstruct(
    legacy_rows: list[dict], fi1_rows: list[dict], class_count: int,
) -> tuple[list[dict], list[dict], dict]:
    """Reconstruct expected rows without invoking the production builder."""
    legacy_by_identity: dict[tuple[str, str], dict] = {}
    legacy_by_key: dict[str, dict] = {}
    source_counts = Counter()
    for row in legacy_rows:
        scene = str(row["scene_name"])
        digest = str(row["geometry_hash"])
        identity = (scene, digest)
        key = str(row["geometry_key"])
        members = list(row.get("members", []))
        if identity in legacy_by_identity or key in legacy_by_key:
            raise ValueError("legacy audit input contains duplicate geometry identity")
        if not members or int(row.get("member_count", -1)) != len(members):
            raise ValueError(f"{key}: invalid legacy audit members")
        for member in members:
            if str(member.get("scene_name")) != scene or str(member.get("geometry_hash")) != digest:
                raise ValueError(f"{key}: legacy audit member geometry differs")
            source_counts[str(member["candidate_source"])] += 1
        legacy_by_identity[identity] = row
        legacy_by_key[key] = row

    fi1_by_identity: dict[tuple[str, str], dict] = {}
    original_by_legacy_key: dict[str, dict] = {}
    refined_members: list[tuple[str, dict]] = []
    plan_keys = set()
    candidate_count = 0
    for row in fi1_rows:
        scene = str(row["scene_name"])
        digest = str(row["geometry_hash"])
        identity = (scene, digest)
        members = list(row.get("members", []))
        if identity in fi1_by_identity:
            raise ValueError("FI1 audit input contains duplicate geometry identity")
        if not members or int(row.get("member_count", -1)) != len(members):
            raise ValueError(f"{identity}: invalid FI1 audit members")
        for member in members:
            candidate_count += 1
            plan_key = str(member.get("plan_key", ""))
            if not plan_key or plan_key in plan_keys:
                raise ValueError("FI1 audit input contains empty or duplicate plan_key")
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
                raise ValueError(f"{plan_key}: FI1 audit member contract differs")
            if str(member["candidate_source"]) == "refined_union":
                if member.get("append_only") is not True:
                    raise ValueError(f"{plan_key}: refined audit member is not append-only")
                refined_members.append((scene, member))
            else:
                if member.get("append_only") is not False or plan_key in original_by_legacy_key:
                    raise ValueError(f"{plan_key}: original audit member contract differs")
                original_by_legacy_key[plan_key] = member
        fi1_by_identity[identity] = row

    if not set(legacy_by_identity).issubset(fi1_by_identity):
        raise ValueError("FI1 audit input does not cover every legacy geometry")
    if set(original_by_legacy_key) != set(legacy_by_key):
        raise ValueError("FI1 audit original plan keys do not cover legacy geometry keys")

    semantic_rows: list[dict] = []
    semantic_keys_by_geometry: dict[tuple[str, str], list[str]] = defaultdict(list)
    legacy_counts = Counter()
    refined_counts = Counter()
    for legacy in legacy_rows:
        scene = str(legacy["scene_name"])
        digest = str(legacy["geometry_hash"])
        identity = (scene, digest)
        plan_member = original_by_legacy_key[str(legacy["geometry_key"])]
        if (
            str(plan_member["geometry_hash"]) != digest
            or str(plan_member["candidate_source"]) != str(legacy["canonical_candidate_source"])
            or int(plan_member["candidate_id"]) != int(legacy["canonical_candidate_id"])
            or int(plan_member["frozen_class_index"]) != int(legacy["canonical_frozen_class_index"])
        ):
            raise ValueError(f"{legacy['geometry_key']}: independent canonical join differs")
        for member in legacy["members"]:
            key = legacy_semantic_hypothesis_key(member)
            semantic_keys_by_geometry[identity].append(key)
            legacy_counts[identity] += 1
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
        refined_counts[identity] += 1
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

    semantic_keys = [row["semantic_hypothesis_key"] for row in semantic_rows]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError("independently reconstructed semantic keys are not unique")

    geometry_rows: list[dict] = []
    for row in fi1_rows:
        scene = str(row["scene_name"])
        digest = str(row["geometry_hash"])
        identity = (scene, digest)
        keys = semantic_keys_by_geometry[identity]
        if not keys:
            raise ValueError(f"{identity}: independently reconstructed geometry has no semantics")
        geometry_rows.append({
            "geometry_index": len(geometry_rows),
            "geometry_evidence_key": geometry_evidence_key(scene, digest),
            "scene_name": scene,
            "geometry_hash": digest,
            "point_count": int(row["point_count"]),
            "geometry_locator_read_only": dict(row["canonical_geometry_locator"]),
            "fi1_plan_member_count": len(row["members"]),
            "fi1_plan_members": [plan_member_projection(member) for member in row["members"]],
            "legacy_semantic_member_count": int(legacy_counts[identity]),
            "refined_semantic_member_count": int(refined_counts[identity]),
            "semantic_hypothesis_count": len(keys),
            "semantic_hypothesis_keys": keys,
            "expensive_visual_evidence_execution_count": 1,
            **frozen_flags(),
        })

    return geometry_rows, semantic_rows, {
        "legacy_geometry_count": len(legacy_rows),
        "legacy_member_count": sum(source_counts.values()),
        "legacy_source_member_counts": dict(sorted(source_counts.items())),
        "fi1_candidate_count": candidate_count,
        "fi1_unique_geometry_count": len(fi1_rows),
        "fi1_refined_union_count": len(refined_members),
        "semantic_hypothesis_count": len(semantic_rows),
    }


def audit(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "legacy_unique_geometry_root", "fi1_unique_geometry_root",
        "preregistration_path", "result_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    paths = {
        "legacy_summary": args.legacy_unique_geometry_root / "summary.json",
        "legacy_ledger": args.legacy_unique_geometry_root / "unique_geometry_ledger.jsonl",
        "fi1_summary": args.fi1_unique_geometry_root / "summary.json",
        "fi1_ledger": args.fi1_unique_geometry_root / "unique_geometry_ledger.jsonl",
        "result_summary": args.result_root / "summary.json",
        "geometry_ledger": args.result_root / GEOMETRY_LEDGER_NAME,
        "semantic_ledger": args.result_root / SEMANTIC_LEDGER_NAME,
        "scene_list": args.scene_list,
        "preregistration": args.preregistration_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing audit inputs: " + ", ".join(missing))

    result_summary = json.loads(paths["result_summary"].read_text())
    legacy_rows = read_jsonl(paths["legacy_ledger"])
    fi1_rows = read_jsonl(paths["fi1_ledger"])
    observed_geometry = read_jsonl(paths["geometry_ledger"])
    observed_semantic = read_jsonl(paths["semantic_ledger"])
    expected_geometry, expected_semantic, stats = _independently_reconstruct(
        legacy_rows, fi1_rows, args.class_count
    )
    errors = Counter()

    _error(errors, "geometry_record_count", len(observed_geometry) != len(expected_geometry))
    _error(errors, "semantic_record_count", len(observed_semantic) != len(expected_semantic))
    for index in range(min(len(observed_geometry), len(expected_geometry))):
        _error(
            errors, "geometry_record_mismatch",
            observed_geometry[index] != expected_geometry[index],
        )
    for index in range(min(len(observed_semantic), len(expected_semantic))):
        _error(
            errors, "semantic_record_mismatch",
            observed_semantic[index] != expected_semantic[index],
        )

    observed_semantic_keys = [str(row.get("semantic_hypothesis_key", "")) for row in observed_semantic]
    expected_semantic_keys = [str(row["semantic_hypothesis_key"]) for row in expected_semantic]
    observed_geometry_keys = [str(row.get("geometry_evidence_key", "")) for row in observed_geometry]
    expected_geometry_keys = [str(row["geometry_evidence_key"]) for row in expected_geometry]
    _error(errors, "semantic_key_coverage", observed_semantic_keys != expected_semantic_keys)
    _error(errors, "semantic_key_unique", len(observed_semantic_keys) != len(set(observed_semantic_keys)))
    _error(errors, "geometry_key_coverage", observed_geometry_keys != expected_geometry_keys)
    _error(errors, "geometry_key_unique", len(observed_geometry_keys) != len(set(observed_geometry_keys)))

    linked_keys = []
    for row in observed_geometry:
        keys = row.get("semantic_hypothesis_keys", [])
        if not isinstance(keys, list):
            errors["invalid_geometry_semantic_keys"] += 1
            continue
        linked_keys.extend(str(value) for value in keys)
        _error(
            errors, "geometry_semantic_count",
            int(row.get("semantic_hypothesis_count", -1)) != len(keys),
        )
        _error(
            errors, "visual_execution_count",
            int(row.get("expensive_visual_evidence_execution_count", -1)) != 1,
        )
    linked_key_counts = Counter(linked_keys)
    observed_semantic_key_counts = Counter(observed_semantic_keys)
    _error(
        errors, "bidirectional_semantic_coverage",
        linked_key_counts != observed_semantic_key_counts,
    )
    _error(
        errors, "geometry_linked_semantic_key_not_unique",
        any(count != 1 for count in linked_key_counts.values()),
    )

    geometry_key_set = set(observed_geometry_keys)
    for row in observed_semantic:
        _error(
            errors, "missing_geometry_reference",
            str(row.get("geometry_evidence_key", "")) not in geometry_key_set,
        )
        _error(errors, "score_fields_not_separated", row.get("score_fields_separated") is not True)
        if row.get("origin_kind") == "legacy_member":
            _error(errors, "missing_legacy_score", row.get("legacy_frozen_score") is None)
        elif row.get("origin_kind") == "fi1_refined_union":
            _error(errors, "refined_has_legacy_score", row.get("legacy_frozen_score") is not None)
        else:
            errors["invalid_origin_kind"] += 1
        for field in ("candidate_deletion", "candidate_mutation", "geometry_mutation", "class_mutation", "score_mutation", "ground_truth_read", "ap_computed"):
            _error(errors, f"forbidden_semantic_flag::{field}", row.get(field) is not False)
        _error(errors, "candidate_not_retained", row.get("candidate_retained") is not True)
    for row in observed_geometry:
        for field in ("candidate_deletion", "candidate_mutation", "geometry_mutation", "class_mutation", "score_mutation", "ground_truth_read", "ap_computed"):
            _error(errors, f"forbidden_geometry_flag::{field}", row.get(field) is not False)
        _error(errors, "geometry_candidate_not_retained", row.get("candidate_retained") is not True)

    expected_provenance = {
        "scene_list": sha256_file(paths["scene_list"]),
        "legacy_summary": sha256_file(paths["legacy_summary"]),
        "legacy_ledger": sha256_file(paths["legacy_ledger"]),
        "fi1_summary": sha256_file(paths["fi1_summary"]),
        "fi1_ledger": sha256_file(paths["fi1_ledger"]),
        "preregistration": sha256_file(paths["preregistration"]),
    }
    expected_hashes = {
        "geometry_evidence_ledger": sha256_file(paths["geometry_ledger"]),
        "semantic_hypothesis_ledger": sha256_file(paths["semantic_ledger"]),
    }
    _error(errors, "summary_version", result_summary.get("version") != VERSION)
    _error(errors, "summary_provenance", result_summary.get("input_provenance") != expected_provenance)
    _error(errors, "summary_hashes", result_summary.get("hashes") != expected_hashes)
    for field, expected in stats.items():
        _error(errors, f"summary_stat::{field}", result_summary.get(field) != expected)
    actual_scene_count = len({row["scene_name"] for row in expected_geometry})
    _error(errors, "summary_scene_count", int(result_summary.get("scene_count", -1)) != actual_scene_count)
    frozen_count_contract = {
        "expected_scene_count": actual_scene_count,
        "expected_legacy_geometry_count": stats["legacy_geometry_count"],
        "expected_legacy_member_count": stats["legacy_member_count"],
        "expected_native_member_count": stats["legacy_source_member_counts"].get("native", 0),
        "expected_track_member_count": stats["legacy_source_member_counts"].get("track", 0),
        "expected_pair_union_member_count": stats["legacy_source_member_counts"].get("pair_union", 0),
        "expected_fi1_candidate_count": stats["fi1_candidate_count"],
        "expected_fi1_unique_geometry_count": stats["fi1_unique_geometry_count"],
        "expected_fi1_refined_union_count": stats["fi1_refined_union_count"],
        "expected_semantic_hypothesis_count": stats["semantic_hypothesis_count"],
    }
    for argument_name, actual in frozen_count_contract.items():
        expected = _expected(args, argument_name)
        _error(
            errors, f"frozen_count_contract::{argument_name}",
            expected is not None and actual != expected,
        )
    for field in ("geometry_mutation", "class_mutation", "score_mutation", "ground_truth_read", "ap_computed"):
        _error(errors, f"forbidden_summary_flag::{field}", result_summary.get(field) is not False)
    _error(errors, "summary_candidate_deletion", int(result_summary.get("candidate_deletion_count", -1)) != 0)

    output = {
        "version": VERSION + "_audit",
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        "scene_count": len({row["scene_name"] for row in expected_geometry}),
        **stats,
        "candidate_deletion_count": 0,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            **expected_provenance,
            "result_summary": sha256_file(paths["result_summary"]),
            **expected_hashes,
        },
    }
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"audit output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--legacy-unique-geometry-root", type=Path, required=True)
    parser.add_argument("--fi1-unique-geometry-root", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=198)
    parser.add_argument("--expected-scene-count", type=int, default=312)
    parser.add_argument("--expected-legacy-geometry-count", type=int, default=39198)
    parser.add_argument("--expected-legacy-member-count", type=int, default=213233)
    parser.add_argument("--expected-native-member-count", type=int, default=187200)
    parser.add_argument("--expected-track-member-count", type=int, default=18141)
    parser.add_argument("--expected-pair-union-member-count", type=int, default=7892)
    parser.add_argument("--expected-fi1-candidate-count", type=int, default=39304)
    parser.add_argument("--expected-fi1-unique-geometry-count", type=int, default=39250)
    parser.add_argument("--expected-fi1-refined-union-count", type=int, default=106)
    parser.add_argument("--expected-semantic-hypothesis-count", type=int, default=213339)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

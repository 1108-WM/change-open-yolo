#!/usr/bin/env python3
"""Independently audit the FI1-D-v3-to-DM-SMS-1 geometry adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_dm_sms1_unique_geometry_ledger import GeometryResolver  # noqa: E402
from tools.dm_sms_core import geometry_hash  # noqa: E402


SOURCES = ("native", "track", "pair_union", "refined_union")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict:
    for name in ("ledger_root", "inference_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    summary = json.loads((args.ledger_root / "summary.json").read_text())
    ledger_path = args.ledger_root / "unique_geometry_ledger.jsonl"
    rows = _rows(ledger_path)
    plan_summary = json.loads((args.inference_root / "complete_plan" / "summary.json").read_text())
    plan_path = args.inference_root / "complete_plan" / str(plan_summary["files"]["plan"])
    plan = _rows(plan_path)
    plan_by_key = {str(row["plan_key"]): row for row in plan}
    plan_index_by_key = {str(row["plan_key"]): index for index, row in enumerate(plan)}
    errors = Counter()
    if len(plan_by_key) != len(plan):
        errors["duplicate_plan_key"] += 1
    if _sha256(plan_path) != str(plan_summary.get("hashes", {}).get("plan", "")):
        errors["plan_sha256_mismatch"] += 1
    order = [(str(row.get("scene_name", "")), str(row.get("geometry_hash", ""))) for row in rows]
    if order != sorted(order) or len(order) != len(set(order)):
        errors["ledger_identity_or_order"] += 1
    resolver = GeometryResolver()
    sources = Counter()
    observed_plan_keys = []
    duplicate_groups = 0
    duplicate_scenes = set()
    duplicate_different_class = 0
    duplicate_different_score = 0
    for row in rows:
        members = row.get("members")
        if (
            not isinstance(members, list) or not members
            or int(row.get("member_count", -1)) != len(members)
        ):
            errors["member_contract"] += 1
            members = []
        if len(members) > 1:
            duplicate_groups += 1
            duplicate_scenes.add(str(row.get("scene_name", "")))
            duplicate_different_class += int(
                len({int(member.get("frozen_class_index", -999)) for member in members}) > 1
            )
            duplicate_different_score += int(
                len({float(member.get("frozen_score", float("nan"))) for member in members}) > 1
            )
        try:
            points = resolver.points(row["canonical_geometry_locator"])
            digest = geometry_hash(points)
            if (
                len(points) != int(row["point_count"])
                or digest != str(row["geometry_hash"])
            ):
                errors["geometry_mismatch"] += 1
        except (FileNotFoundError, KeyError, ValueError, OSError):
            errors["geometry_resolution_error"] += 1
            points = None
            digest = str(row.get("geometry_hash", ""))
        for member_index, member in enumerate(members):
            if not isinstance(member, dict):
                errors["member_contract"] += 1
                continue
            key = str(member.get("plan_key") or member.get("fi1_d_v3_plan_key") or "")
            observed_plan_keys.append(key)
            plan_row = plan_by_key.get(key)
            source = str(member.get("candidate_source", ""))
            sources[source] += 1
            if plan_row is None:
                errors["plan_join_missing"] += 1
                continue
            if (
                member.get("fi1_d_v3_plan_key") != key
                or int(member.get("plan_index", -1)) != plan_index_by_key[key]
            ):
                errors["member_plan_identity_mismatch"] += 1
            if source not in SOURCES or source != str(plan_row.get("candidate_source")):
                errors["source_mismatch"] += 1
            try:
                member_points = resolver.points(member["geometry_locator_read_only"])
                if points is None or not np.array_equal(member_points, points):
                    errors["member_geometry_mismatch"] += 1
                plan_algorithm = str(plan_row.get("geometry_digest_algorithm", ""))
                if plan_algorithm == "sha1_point_indices":
                    plan_digest = geometry_hash(member_points)
                elif plan_algorithm == "sha256_point_indices":
                    plan_digest = hashlib.sha256(
                        np.unique(np.asarray(member_points, dtype=np.int64)).tobytes()
                    ).hexdigest()
                else:
                    plan_digest = ""
                    errors["geometry_digest_algorithm"] += 1
                if (
                    plan_digest != str(plan_row.get("geometry_digest", ""))
                    or str(member.get("geometry_hash", "")) != digest
                    or member.get("geometry_locator_read_only") != plan_row.get("geometry_locator_read_only")
                    or member.get("geometry_locator") != member.get("geometry_locator_read_only")
                ):
                    errors["member_geometry_mismatch"] += 1
            except (FileNotFoundError, KeyError, ValueError, OSError):
                errors["member_geometry_resolution_error"] += 1
            if (
                int(member.get("frozen_class_index", -999)) != int(plan_row.get("frozen_class_index", -998))
                or float(member.get("challenger_score", -1.0)) != float(plan_row.get("challenger_score", -2.0))
                or float(member.get("frozen_score", -1.0)) != float(member.get("challenger_score", -2.0))
            ):
                errors["class_or_score_mismatch"] += 1
            if (
                member.get("candidate_retained") is not True
                or member.get("candidate_deletion") is not False
                or any(member.get(flag) is not False for flag in (
                    "geometry_mutation", "class_mutation", "score_mutation",
                ))
            ):
                errors["member_mutation_contract"] += 1
            expected_append = source == "refined_union"
            if (
                bool(member.get("append_only")) != expected_append
                or bool(member.get("fi1_d_v3_append_only")) != expected_append
                or plan_row.get("append_only") is not expected_append
            ):
                errors["append_only_mismatch"] += 1
        if members:
            canonical = members[0]
            if (
                int(row.get("canonical_member_index", -1)) != 0
                or row.get("canonical_candidate_source") != canonical.get("candidate_source")
                or int(row.get("canonical_frozen_class_index", -999)) != int(canonical.get("frozen_class_index", -998))
                or float(row.get("canonical_frozen_score", -1.0)) != float(canonical.get("challenger_score", -2.0))
                or row.get("canonical_geometry_locator") != canonical.get("geometry_locator_read_only")
            ):
                errors["canonical_member_mismatch"] += 1
        for flag in (
            "candidate_mutation", "geometry_mutation", "class_mutation", "score_mutation",
        ):
            if row.get(flag) is not False:
                errors[f"mutation::{flag}"] += 1
        if (
            row.get("ground_truth_usage") != "none"
            or row.get("ground_truth_read") is not False
            or row.get("ap_computed") is not False
        ):
            errors["no_gt_contract"] += 1
    if (
        len(observed_plan_keys) != len(set(observed_plan_keys))
        or set(plan_by_key) != set(observed_plan_keys)
    ):
        errors["plan_coverage_mismatch"] += 1
    expected_counts = {source: int(sources[source]) for source in SOURCES if sources[source]}
    expected_canonical_counts = dict(sorted(Counter(
        str(row.get("canonical_candidate_source", "")) for row in rows
    ).items()))
    checks = {
        "row_count": len(rows),
        "member_count": len(observed_plan_keys),
        "source_counts": expected_counts,
        "invalid_class_count": sum(
            not bool(member.get("frozen_class_valid"))
            for row in rows for member in row.get("members", []) if isinstance(member, dict)
        ),
        "invalid_canonical_class_count": sum(
            not bool(row.get("canonical_frozen_class_valid")) for row in rows
        ),
    }
    if (
        summary.get("contract_valid") is not True
        or int(summary.get("unique_geometry_count", -1)) != len(rows)
        or int(summary.get("fi1_d_v3_challenger_candidate_count", -1)) != len(plan)
        or int(summary.get("member_count", -1)) != len(observed_plan_keys)
        or int(summary.get("candidate_deletion_count", -1)) != 0
        or summary.get("source_member_counts") != expected_counts
        or summary.get("canonical_source_counts") != expected_canonical_counts
        or int(summary.get("invalid_frozen_class_member_count", -1)) != checks["invalid_class_count"]
        or int(summary.get("invalid_canonical_frozen_class_count", -1)) != checks["invalid_canonical_class_count"]
        or int(summary.get("duplicate_geometry_group_count", -1)) != duplicate_groups
        or int(summary.get("duplicate_geometry_scene_count", -1)) != len(duplicate_scenes)
        or int(summary.get("duplicate_group_different_class_count", -1)) != duplicate_different_class
        or int(summary.get("duplicate_group_different_score_count", -1)) != duplicate_different_score
    ):
        errors["summary_mismatch"] += 1
    provenance = summary.get("input_provenance", {})
    if str(provenance.get("fi1_d_v3_complete_plan_sha256", "")) != _sha256(plan_path):
        errors["summary_plan_provenance_mismatch"] += 1
    expected_contract = {
        "candidate_count": getattr(args, "expected_candidate_count", None),
        "unique_geometry_count": getattr(args, "expected_unique_geometry_count", None),
        "duplicate_geometry_group_count": getattr(args, "expected_duplicate_group_count", None),
        "duplicate_geometry_scene_count": getattr(args, "expected_duplicate_scene_count", None),
        "duplicate_group_different_class_count": getattr(args, "expected_different_class_group_count", None),
        "duplicate_group_different_score_count": getattr(args, "expected_different_score_group_count", None),
    }
    observed_contract = {
        "candidate_count": len(plan),
        "unique_geometry_count": len(rows),
        "duplicate_geometry_group_count": duplicate_groups,
        "duplicate_geometry_scene_count": len(duplicate_scenes),
        "duplicate_group_different_class_count": duplicate_different_class,
        "duplicate_group_different_score_count": duplicate_different_score,
    }
    for name, expected in expected_contract.items():
        if expected is not None and int(expected) != int(observed_contract[name]):
            errors[f"frozen_contract::{name}"] += 1
    observed_plan_key_set = {key for key in observed_plan_keys if key}
    audit_valid = sum(errors.values()) == 0
    audit = {
        "version": "dm_sms1_fi1_d_v3_unique_geometry_audit_v1",
        "audit_valid": audit_valid,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        "scene_count": len({str(row["scene_name"]) for row in rows}),
        "unique_geometry_count": len(rows),
        "candidate_count": len(plan),
        "member_count": len(observed_plan_keys),
        "candidate_deletion_count": len(set(plan_by_key) - observed_plan_key_set),
        "plan_key_coverage_complete": set(plan_by_key) == observed_plan_key_set,
        "duplicate_geometry_output_count": len(order) - len(set(order)),
        "duplicate_geometry_group_count": duplicate_groups,
        "duplicate_geometry_scene_count": len(duplicate_scenes),
        "duplicate_group_different_class_count": duplicate_different_class,
        "duplicate_group_different_score_count": duplicate_different_score,
        "source_counts": expected_counts,
        "candidate_mutation": False if audit_valid else None,
        "geometry_mutation": False if audit_valid else None,
        "class_mutation": False if audit_valid else None,
        "score_mutation": False if audit_valid else None,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            "ledger_sha256": _sha256(ledger_path),
            "ledger_summary_sha256": _sha256(args.ledger_root / "summary.json"),
            "fi1_d_v3_complete_plan_sha256": _sha256(plan_path),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int)
    parser.add_argument("--expected-unique-geometry-count", type=int)
    parser.add_argument("--expected-duplicate-group-count", type=int)
    parser.add_argument("--expected-duplicate-scene-count", type=int)
    parser.add_argument("--expected-different-class-group-count", type=int)
    parser.add_argument("--expected-different-score-group-count", type=int)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

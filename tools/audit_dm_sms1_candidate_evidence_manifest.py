#!/usr/bin/env python3
"""Audit candidate evidence inputs against attribute and semantic manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    root: Path,
    attribute_root: Path,
    semantic_root: Path,
    expected_candidate_count: int = 39304,
    expected_unique_geometry_count: int = 39250,
) -> dict:
    records_path = root / "candidate_evidence_manifest.jsonl"
    attribute_path = attribute_root / "attribute_extraction_manifest.jsonl"
    semantic_path = semantic_root / "semantic_arbitration_manifest.jsonl"
    rows = _rows(records_path)
    attributes = _rows(attribute_path)
    semantics = _rows(semantic_path)
    summary = json.loads((root / "summary.json").read_text())
    attribute_summary = json.loads((attribute_root / "summary.json").read_text())
    semantic_summary = json.loads((semantic_root / "summary.json").read_text())
    errors: list[str] = []

    attribute_ids = [
        (str(row.get("scene_name", "")), str(row.get("plan_key", ""))) for row in attributes
    ]
    semantic_ids = [
        (str(row.get("scene_name", "")), str(row.get("plan_key", ""))) for row in semantics
    ]
    attribute_by_id = dict(zip(attribute_ids, attributes))
    semantic_by_id = dict(zip(semantic_ids, semantics))
    if len(attribute_ids) != len(set(attribute_ids)) or any(not scene or not key for scene, key in attribute_ids):
        errors.append("attribute manifest has empty or duplicate scene/plan identities")
    if len(semantic_ids) != len(set(semantic_ids)) or any(not scene or not key for scene, key in semantic_ids):
        errors.append("semantic manifest has empty or duplicate scene/plan identities")
    if attribute_ids != semantic_ids:
        errors.append("attribute/semantic plan_key coverage or order mismatch")

    identities = []
    pairs = singles = 0
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        identity = (str(row.get("scene_name", "")), str(row.get("plan_key", "")))
        identities.append(identity)
        attribute = attribute_by_id.get(identity)
        semantic = semantic_by_id.get(identity)
        if attribute is None or semantic is None:
            errors.append(f"{prefix}: plan_key is absent from an upstream manifest")
        else:
            if (
                row.get("task_id") != attribute.get("task_id")
                or row.get("attribute_task_id") != attribute.get("task_id")
                or row.get("plan_index") != attribute.get("plan_index")
                or row.get("geometry_hash") != attribute.get("geometry_hash")
                or row.get("visual_geometry_key") != attribute.get("visual_geometry_key")
            ):
                errors.append(f"{prefix}: attribute join differs")
            frozen_checks = {
                "plan_index": row.get("plan_index") == semantic.get("plan_index"),
                "geometry_hash": row.get("geometry_hash") == semantic.get("geometry_hash"),
                "geometry_locator_read_only": (
                    row.get("geometry_locator_read_only")
                    == semantic.get("geometry_locator_read_only")
                ),
                "frozen_class_index": row.get("frozen_class_index") == semantic.get("frozen_class_index"),
                "challenger_score": row.get("challenger_score") == semantic.get("challenger_score"),
                "candidate_source": row.get("candidate_source") == semantic.get("candidate_source"),
                "append_only": row.get("append_only") == semantic.get("append_only"),
                "canonical_frozen_class_index": (
                    row.get("canonical_frozen_class_index")
                    == semantic.get("canonical_frozen_class_index")
                ),
                "canonical_frozen_score": (
                    row.get("canonical_frozen_score") == semantic.get("canonical_frozen_score")
                ),
            }
            for name, valid in frozen_checks.items():
                if not valid:
                    errors.append(f"{prefix}: frozen {name} differs from semantic manifest")
            expected_hypotheses = [
                {
                    "class_index": int(candidate["class_index"]),
                    "sources": list(candidate.get("sources", [])),
                }
                for candidate in semantic.get("finite_class_hypotheses", [])
            ]
            observed_hypotheses = [
                {
                    "class_index": int(candidate.get("class_index", -1)),
                    "sources": list(candidate.get("sources", [])),
                }
                for candidate in row.get("candidate_hypotheses", [])
            ]
            if observed_hypotheses != expected_hypotheses:
                errors.append(f"{prefix}: finite candidate hypotheses differ from semantic manifest")
        if row.get("fi1_d_v3_plan_key") != row.get("plan_key") or not row.get("plan_key"):
            errors.append(f"{prefix}: invalid plan_key identity")
        if str(row.get("plan_key", "")) not in str(row.get("task_id", "")):
            errors.append(f"{prefix}: task_id does not contain plan_key")
        candidates = row.get("candidate_hypotheses", [])
        if len(candidates) not in (1, 2):
            errors.append(f"{prefix}: candidate count is not one or two")
        if len(candidates) == 2:
            pairs += 1
            if row.get("candidate_order_ba") != list(reversed(row.get("candidate_order_ab", []))):
                errors.append(f"{prefix}: swapped order is not exact reverse")
        else:
            singles += 1
        if row.get("class_decision_made") is not False or row.get("selected_class_index") is not None:
            errors.append(f"{prefix}: class decision already made")
        for key in ("candidate_mutation", "geometry_mutation", "score_mutation"):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is true")
        rule = row.get("decision_rule", {})
        for key in (
            "only_alternative_supported_in_both_orders_may_be_considered",
            "otherwise_keep_frozen_control_class", "all_geometry_nodes_decided_simultaneously",
            "no_proposal_deletion", "no_score_change",
        ):
            if rule.get(key) is not True:
                errors.append(f"{prefix}: decision rule {key} is not true")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")

    unique_geometries = len({(row.get("scene_name"), row.get("geometry_hash")) for row in rows})
    if identities != attribute_ids or identities != semantic_ids:
        errors.append("candidate plan_key coverage or order differs from upstream manifests")
    if len(identities) != len(set(identities)):
        errors.append("duplicate scene/plan identities")
    if len(rows) != expected_candidate_count:
        errors.append("frozen candidate_count mismatch")
    if unique_geometries != expected_unique_geometry_count:
        errors.append("frozen unique_geometry_count mismatch")
    if int(summary.get("candidate_count", -1)) != expected_candidate_count:
        errors.append("summary candidate_count mismatch")
    if int(summary.get("unique_geometry_count", -1)) != expected_unique_geometry_count:
        errors.append("summary unique_geometry_count mismatch")
    if int(summary.get("candidate_deletion_count", -1)) != 0:
        errors.append("summary candidate_deletion_count mismatch")
    for name, upstream_summary in (
        ("attribute", attribute_summary), ("semantic", semantic_summary),
    ):
        if (
            int(upstream_summary.get("candidate_count", -1)) != expected_candidate_count
            or int(upstream_summary.get("unique_geometry_count", -1))
            != expected_unique_geometry_count
            or int(upstream_summary.get("candidate_deletion_count", -1)) != 0
        ):
            errors.append(f"{name} summary frozen counts mismatch")
    if int(summary.get("candidate_pair_count", -1)) != pairs or int(summary.get("single_candidate_count", -1)) != singles:
        errors.append("summary finite candidate count mismatch")
    if summary.get("class_decision_made") is not False or summary.get("selected_class_count") != 0:
        errors.append("summary class decision contract mismatch")
    result = {
        "version": "dm_sms1_candidate_evidence_manifest_audit_v2",
        "candidate_count": len(rows),
        "unique_geometry_count": unique_geometries,
        "candidate_deletion_count": len(set(semantic_ids) - set(identities)),
        "plan_key_coverage_complete": identities == attribute_ids == semantic_ids,
        "frozen_field_error_count": sum("frozen" in error for error in errors),
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "class_decision_made": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            "candidate_manifest_sha256": _sha256(records_path),
            "attribute_manifest_sha256": _sha256(attribute_path),
            "semantic_manifest_sha256": _sha256(semantic_path),
            "attribute_summary_sha256": _sha256(attribute_root / "summary.json"),
            "semantic_summary_sha256": _sha256(semantic_root / "summary.json"),
        },
    }
    (root / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("--attribute-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=39304)
    parser.add_argument("--expected-unique-geometry-count", type=int, default=39250)
    args = parser.parse_args()
    result = audit(
        args.manifest_root, args.attribute_root, args.semantic_root,
        args.expected_candidate_count, args.expected_unique_geometry_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

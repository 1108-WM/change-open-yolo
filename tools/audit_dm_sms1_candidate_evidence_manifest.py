#!/usr/bin/env python3
"""Audit the finite-candidate evidence input ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path) -> dict:
    rows = [json.loads(line) for line in (root / "candidate_evidence_manifest.jsonl").read_text().splitlines() if line.strip()]
    summary = json.loads((root / "summary.json").read_text())
    errors: list[str] = []
    identities = []
    pairs = singles = 0
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        identities.append((row.get("scene_name"), row.get("geometry_hash")))
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
        for key in ("only_alternative_supported_in_both_orders_may_be_considered", "otherwise_keep_frozen_control_class", "all_geometry_nodes_decided_simultaneously", "no_proposal_deletion", "no_score_change"):
            if rule.get(key) is not True:
                errors.append(f"{prefix}: decision rule {key} is not true")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")
    if len(identities) != len(set(identities)):
        errors.append("duplicate geometry identities")
    if int(summary.get("geometry_count", -1)) != len(rows):
        errors.append("summary geometry count mismatch")
    if int(summary.get("candidate_pair_count", -1)) != pairs or int(summary.get("single_candidate_count", -1)) != singles:
        errors.append("summary candidate count mismatch")
    if summary.get("class_decision_made") is not False or summary.get("selected_class_count") != 0:
        errors.append("summary class decision contract mismatch")
    result = {
        "version": "dm_sms1_candidate_evidence_manifest_audit_v1",
        "row_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "class_decision_made": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    (root / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

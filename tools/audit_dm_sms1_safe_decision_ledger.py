#!/usr/bin/env python3
"""Audit complete safe semantic decisions without GT or AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path, candidate_manifest: Path) -> dict:
    rows = [json.loads(line) for line in (root / "safe_decisions.jsonl").read_text().splitlines() if line.strip()]
    summary = json.loads((root / "summary.json").read_text())
    candidate_rows = [
        json.loads(line) for line in candidate_manifest.read_text().splitlines() if line.strip()
    ]
    candidate_task_ids = [str(row.get("task_id", "")) for row in candidate_rows]
    candidate_identities = [(row.get("scene_name"), row.get("plan_key")) for row in candidate_rows]
    errors: list[str] = []
    if any(not task_id for task_id in candidate_task_ids) or len(candidate_task_ids) != len(set(candidate_task_ids)):
        errors.append("candidate manifest has an empty or duplicate task_id")
    if len(candidate_identities) != len(set(candidate_identities)):
        errors.append("candidate manifest has duplicate scene/plan identities")
    candidates_by_task = dict(zip(candidate_task_ids, candidate_rows))
    identities = []
    source_task_ids = []
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        identity = (row.get("scene_name"), row.get("plan_key"))
        identities.append(identity)
        task_id = str(row.get("decision_source_task_id", ""))
        source_task_ids.append(task_id)
        candidate = candidates_by_task.get(task_id)
        if candidate is None:
            errors.append(f"{prefix}: decision task is absent from candidate manifest")
            continue
        if row.get("fi1_d_v3_plan_key") != row.get("plan_key"):
            errors.append(f"{prefix}: plan_key alias mismatch")
        if identity != (candidate.get("scene_name"), candidate.get("plan_key")):
            errors.append(f"{prefix}: decision identity does not match candidate task")
        if row.get("geometry_hash") != candidate.get("geometry_hash"):
            errors.append(f"{prefix}: visual geometry provenance mismatch")
        if (
            row.get("candidate_source") != candidate.get("candidate_source")
            or row.get("challenger_score") != candidate.get("challenger_score")
            or row.get("append_only") != candidate.get("append_only")
        ):
            errors.append(f"{prefix}: frozen candidate provenance mismatch")
        allowed = {int(item["class_index"]) for item in candidate.get("candidate_hypotheses", [])}
        incumbent = int(candidate.get("canonical_frozen_class_index", -1))
        selected = int(row.get("arbitrated_class_index", -1))
        if selected not in allowed:
            errors.append(f"{prefix}: selected class is outside finite candidates")
        if int(row.get("canonical_frozen_class_index", -1)) != incumbent:
            errors.append(f"{prefix}: frozen class does not match candidate manifest")
        changed = selected != incumbent
        if row.get("class_changed") != changed:
            errors.append(f"{prefix}: class_changed is inconsistent")
        if not isinstance(row.get("model_evidence_valid"), bool):
            errors.append(f"{prefix}: model_evidence_valid is not boolean")
        if not row.get("model_evidence_valid") and selected != incumbent:
            errors.append(f"{prefix}: invalid evidence did not keep frozen control")
        if not row.get("model_evidence_valid"):
            if row.get("change_reason") != "invalid_evidence_keep_frozen_control" or not isinstance(row.get("fallback_reason"), str):
                errors.append(f"{prefix}: invalid-evidence fallback provenance is incomplete")
        elif row.get("fallback_reason") is not None:
            errors.append(f"{prefix}: valid evidence unexpectedly has a fallback reason")
        elif row.get("change_reason") != ("both_order_support" if changed else "keep_frozen_control"):
            errors.append(f"{prefix}: valid-evidence change reason is inconsistent")
        for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is true")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")
        if row.get("ground_truth_usage") != "none" or row.get("class_decision_made") is not True:
            errors.append(f"{prefix}: decision provenance is incomplete")
    if len(identities) != len(set(identities)):
        errors.append("duplicate scene/plan decision identities")
    if len(source_task_ids) != len(set(source_task_ids)):
        errors.append("duplicate decision source task_id")
    if int(summary.get("candidate_count", -1)) != len(rows):
        errors.append("summary candidate count mismatch")
    if int(summary.get("candidate_deletion_count", -1)) != 0:
        errors.append("summary candidate deletion count mismatch")
    expected_counts = {
        "model_evidence_valid_count": sum(row.get("model_evidence_valid") is True for row in rows),
        "fallback_keep_count": sum(row.get("model_evidence_valid") is False for row in rows),
        "class_change_count": sum(row.get("class_changed") is True for row in rows),
        "kept_frozen_control_count": sum(row.get("class_changed") is False for row in rows),
    }
    for key, expected in expected_counts.items():
        if int(summary.get(key, -1)) != expected:
            errors.append(f"summary {key} mismatch")
    if float(summary.get("decision_coverage_fraction", -1.0)) != 1.0:
        errors.append("decision coverage is not complete")
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion", "ground_truth_read", "ap_computed"):
        if summary.get(key) is not False:
            errors.append(f"summary {key} is not false")
    result = {
        "version": "dm_sms1_safe_decision_ledger_audit_v1",
        "row_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    (root / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_root", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.decision_root, args.candidate_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

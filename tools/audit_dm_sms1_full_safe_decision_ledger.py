#!/usr/bin/env python3
"""Audit complete pair-plus-single DM-SMS-1 decisions without GT or AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root: Path, candidate_manifest: Path) -> dict:
    rows = _read_jsonl(root / "safe_decisions.jsonl")
    candidates = _read_jsonl(candidate_manifest)
    summary = json.loads((root / "summary.json").read_text())
    candidate_ids = [str(row.get("task_id", "")) for row in candidates]
    decision_ids = [str(row.get("decision_source_task_id", "")) for row in rows]
    errors = []
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate manifest has empty or duplicate task_id")
    if any(not value for value in decision_ids) or len(decision_ids) != len(set(decision_ids)):
        errors.append("decision ledger has empty or duplicate source task_id")
    if set(candidate_ids) != set(decision_ids):
        errors.append("decision coverage does not exactly match candidate manifest")
    candidates_by_id = dict(zip(candidate_ids, candidates))
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        task_id = str(row.get("decision_source_task_id", ""))
        candidate = candidates_by_id.get(task_id)
        if candidate is None:
            continue
        if row.get("fi1_d_v3_plan_key") != row.get("plan_key"):
            errors.append(f"{prefix}: plan_key alias mismatch")
        hypotheses = candidate.get("candidate_hypotheses", [])
        allowed = {int(item["class_index"]) for item in hypotheses}
        incumbent = int(candidate["canonical_frozen_class_index"])
        selected = int(row.get("arbitrated_class_index", -1))
        if (row.get("scene_name"), row.get("plan_key")) != (candidate.get("scene_name"), candidate.get("plan_key")):
            errors.append(f"{prefix}: identity mismatch")
        if row.get("geometry_hash") != candidate.get("geometry_hash"):
            errors.append(f"{prefix}: visual geometry provenance mismatch")
        if (
            row.get("candidate_source") != candidate.get("candidate_source")
            or row.get("challenger_score") != candidate.get("challenger_score")
            or row.get("append_only") != candidate.get("append_only")
        ):
            errors.append(f"{prefix}: frozen candidate provenance mismatch")
        # Singleton rows never promote their sole alternative. The frozen
        # incumbent may be a foreground class or an explicit background
        # sentinel (-1/198), so deterministic keep is valid whenever the
        # selected value exactly equals that frozen incumbent.
        selected_allowed = selected in allowed or (len(hypotheses) == 1 and selected == incumbent)
        if not selected_allowed or int(row.get("canonical_frozen_class_index", -1)) != incumbent:
            errors.append(f"{prefix}: class contract mismatch")
        if row.get("class_changed") != (selected != incumbent):
            errors.append(f"{prefix}: class_changed mismatch")
        if len(hypotheses) == 1:
            if row.get("decision_path") != "single_candidate_deterministic_keep" or selected != incumbent:
                errors.append(f"{prefix}: invalid single-candidate decision")
            if int(row.get("singleton_candidate_class_index", -999)) != int(hypotheses[0]["class_index"]):
                errors.append(f"{prefix}: singleton candidate provenance mismatch")
            if row.get("model_evidence_valid") is not None or row.get("fallback_reason") is not None:
                errors.append(f"{prefix}: single-candidate evidence provenance mismatch")
        elif len(hypotheses) == 2:
            if row.get("decision_path") != "two_candidate_vlm_arbitration":
                errors.append(f"{prefix}: invalid two-candidate decision path")
            if not isinstance(row.get("model_evidence_valid"), bool):
                errors.append(f"{prefix}: two-candidate evidence validity is not boolean")
            if row.get("model_evidence_valid") is False and selected != incumbent:
                errors.append(f"{prefix}: invalid evidence changed class")
        else:
            errors.append(f"{prefix}: unsupported candidate count")
        for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion", "ground_truth_read", "ap_computed"):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is not false")
        if row.get("ground_truth_usage") != "none" or row.get("class_decision_made") is not True:
            errors.append(f"{prefix}: provenance is incomplete")
    identities = [(row.get("scene_name"), row.get("plan_key")) for row in rows]
    if len(identities) != len(set(identities)):
        errors.append("duplicate scene/plan identities")
    expected = {
        "candidate_count": len(rows),
        "two_candidate_count": sum(len(row.get("candidate_hypotheses", [])) == 2 for row in candidates),
        "single_candidate_count": sum(len(row.get("candidate_hypotheses", [])) == 1 for row in candidates),
        "model_evidence_valid_count": sum(row.get("model_evidence_valid") is True for row in rows),
        "invalid_evidence_fallback_count": sum(row.get("model_evidence_valid") is False for row in rows),
        "single_candidate_keep_count": sum(row.get("decision_path") == "single_candidate_deterministic_keep" for row in rows),
        "class_change_count": sum(row.get("class_changed") is True for row in rows),
        "kept_frozen_control_count": sum(row.get("class_changed") is False for row in rows),
    }
    for key, value in expected.items():
        if int(summary.get(key, -1)) != value:
            errors.append(f"summary {key} mismatch")
    if int(summary.get("candidate_deletion_count", -1)) != 0:
        errors.append("summary candidate deletion count mismatch")
    if float(summary.get("decision_coverage_fraction", -1.0)) != 1.0:
        errors.append("summary coverage is not complete")
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion", "ground_truth_read", "ap_computed"):
        if summary.get(key) is not False:
            errors.append(f"summary {key} is not false")
    result = {
        "version": "dm_sms1_full_safe_decision_ledger_audit_v1",
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

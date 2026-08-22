#!/usr/bin/env python3
"""Independently audit DM-SMS-1 v2 dual-counterevidence decisions without GT/AP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


DECISION_VERSION = "dm_sms1_v2_dual_counterevidence_decision_v1"
SUMMARY_VERSION = "dm_sms1_v2_dual_counterevidence_ledger_v1"
AUDIT_VERSION = "dm_sms1_v2_dual_counterevidence_audit_v1"
RULE_ID = "alternative_dual_support_no_counterevidence_and_incumbent_dual_counterevidence"
FI1_FOUNDATION = "FI1-Legacy"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _by_task(rows: list[dict], key: str, label: str, errors: list[str]) -> dict[str, dict]:
    ids = [str(row.get(key, "")) for row in rows]
    if any(not value for value in ids):
        errors.append(f"{label} contains an empty {key}")
    if len(ids) != len(set(ids)):
        errors.append(f"{label} contains duplicate {key}")
    return dict(zip(ids, rows))


def _candidate_contract(row: dict) -> tuple[int, int, list[int], list[int]]:
    task_id = str(row.get("task_id", ""))
    if not task_id:
        raise ValueError("candidate row contains an empty task_id")
    for key in ("scene_name", "geometry_key", "geometry_hash"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{task_id}: candidate row misses {key}")
    hypotheses = row.get("candidate_hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 2:
        raise ValueError(f"{task_id}: expected exactly two candidates")
    indices = []
    for item in hypotheses:
        if not isinstance(item, dict):
            raise ValueError(f"{task_id}: candidate is not an object")
        index = item.get("class_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{task_id}: candidate class_index is not an integer")
        indices.append(index)
    if len(set(indices)) != 2:
        raise ValueError(f"{task_id}: candidate indices are not unique")
    incumbent = row.get("canonical_frozen_class_index")
    if isinstance(incumbent, bool) or not isinstance(incumbent, int) or incumbent not in indices:
        raise ValueError(f"{task_id}: frozen class is outside candidates")
    alternative = next(index for index in indices if index != incumbent)
    if any(key in row for key in ("candidate_order_ab", "candidate_order_ba")):
        names = [item.get("class_name") for item in hypotheses]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 2:
            raise ValueError(f"{task_id}: named order has invalid class names")
        if row.get("candidate_order_ab") != names or row.get("candidate_order_ba") != list(reversed(names)):
            raise ValueError(f"{task_id}: registered AB/BA order is invalid")
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
        if row.get(key) is True:
            raise ValueError(f"{task_id}: candidate row reports {key}=true")
    if row.get("ground_truth_read") is True or row.get("ap_computed") is True:
        raise ValueError(f"{task_id}: candidate row violates no-GT/no-AP")
    return incumbent, alternative, indices, list(reversed(indices))


def _evidence_item(item: object, task_id: str, order_name: str) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"{task_id}: {order_name} item is not an object")
    index = item.get("class_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"{task_id}: {order_name} class_index is invalid")
    if not isinstance(item.get("supported"), bool):
        raise ValueError(f"{task_id}: {order_name} supported is invalid")
    if not isinstance(item.get("strong_counterevidence"), bool):
        raise ValueError(f"{task_id}: {order_name} strong_counterevidence is invalid")
    if not isinstance(item.get("support_evidence"), str):
        raise ValueError(f"{task_id}: {order_name} support_evidence is invalid")
    if not isinstance(item.get("counterevidence"), str):
        raise ValueError(f"{task_id}: {order_name} counterevidence is invalid")
    confidence = item.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{task_id}: {order_name} confidence is invalid")
    return {
        "class_index": int(index),
        "supported": item["supported"],
        "strong_counterevidence": item["strong_counterevidence"],
        "support_evidence": item["support_evidence"],
        "counterevidence": item["counterevidence"],
        "confidence": float(confidence),
    }


def _evidence_maps(candidate: dict, evidence: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    task_id = str(candidate["task_id"])
    if not isinstance(evidence, dict) or str(evidence.get("task_id", "")) != task_id:
        raise ValueError(f"{task_id}: evidence task join mismatch")
    for key in ("scene_name", "geometry_hash"):
        if key in evidence and evidence.get(key) != candidate.get(key):
            raise ValueError(f"{task_id}: evidence {key} join mismatch")
    if "valid" in evidence and not isinstance(evidence.get("valid"), bool):
        raise ValueError(f"{task_id}: valid flag is not boolean")
    if evidence.get("valid") is False:
        raise ValueError(f"{task_id}: source evidence is invalid")
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
        if evidence.get(key) is True:
            raise ValueError(f"{task_id}: evidence reports {key}=true")
    if evidence.get("ground_truth_read") is True or evidence.get("ap_computed") is True:
        raise ValueError(f"{task_id}: evidence violates no-GT/no-AP")
    _incumbent, _alternative, expected_ab, expected_ba = _candidate_contract(candidate)
    maps = []
    for order_name, expected in (("order_ab", expected_ab), ("order_ba", expected_ba)):
        order = evidence.get(order_name)
        if not isinstance(order, dict):
            raise ValueError(f"{task_id}: evidence misses {order_name}")
        items = order.get("candidate_results")
        if not isinstance(items, list) or len(items) != 2:
            raise ValueError(f"{task_id}: {order_name} count is invalid")
        normalized = [_evidence_item(item, task_id, order_name) for item in items]
        if [item["class_index"] for item in normalized] != expected:
            raise ValueError(f"{task_id}: {order_name} order mismatch")
        maps.append({item["class_index"]: item for item in normalized})
    return maps[0], maps[1]


def _expected_valid_decision(candidate: dict, first: dict[int, dict], second: dict[int, dict]) -> dict:
    incumbent, alternative, _expected_ab, _expected_ba = _candidate_contract(candidate)
    gates = {
        "alternative_supported_ab": first[alternative]["supported"],
        "alternative_supported_ba": second[alternative]["supported"],
        "alternative_no_strong_counterevidence_ab": not first[alternative]["strong_counterevidence"],
        "alternative_no_strong_counterevidence_ba": not second[alternative]["strong_counterevidence"],
        "incumbent_strong_counterevidence_ab": first[incumbent]["strong_counterevidence"],
        "incumbent_strong_counterevidence_ba": second[incumbent]["strong_counterevidence"],
    }
    passed = all(gates.values())
    if passed:
        reason = RULE_ID
        selected = alternative
    elif not (gates["alternative_supported_ab"] and gates["alternative_supported_ba"]):
        reason = "alternative_missing_dual_support_keep_frozen_control"
        selected = incumbent
    elif not (
        gates["alternative_no_strong_counterevidence_ab"]
        and gates["alternative_no_strong_counterevidence_ba"]
    ):
        reason = "alternative_has_strong_counterevidence_keep_frozen_control"
        selected = incumbent
    else:
        reason = "incumbent_missing_dual_strong_counterevidence_keep_frozen_control"
        selected = incumbent
    snapshot = {
        "order_ab": {
            "incumbent": dict(first[incumbent]),
            "alternative": dict(first[alternative]),
        },
        "order_ba": {
            "incumbent": dict(second[incumbent]),
            "alternative": dict(second[alternative]),
        },
    }
    return {
        "selected": selected,
        "changed": selected != incumbent,
        "reason": reason,
        "gates": gates,
        "passed": passed,
        "snapshot": snapshot,
        "incumbent": incumbent,
        "alternative": alternative,
    }


def _audit_row(candidate: dict, evidence: dict, decision: dict, prefix: str, errors: list[str]) -> None:
    try:
        incumbent, alternative, _expected_ab, _expected_ba = _candidate_contract(candidate)
    except Exception as error:
        errors.append(f"{prefix}: invalid candidate contract: {error}")
        return
    if decision.get("version") != DECISION_VERSION or decision.get("rule_id") != RULE_ID:
        errors.append(f"{prefix}: decision version or rule_id mismatch")
    if decision.get("fi1_foundation") != FI1_FOUNDATION:
        errors.append(f"{prefix}: FI1 foundation is not FI1-Legacy")
    if (
        decision.get("scene_name"), decision.get("geometry_key"), decision.get("geometry_hash")
    ) != (
        candidate.get("scene_name"), candidate.get("geometry_key"), candidate.get("geometry_hash")
    ):
        errors.append(f"{prefix}: decision identity mismatch")
    if int(decision.get("canonical_frozen_class_index", -999)) != incumbent:
        errors.append(f"{prefix}: frozen class mismatch")
    if int(decision.get("alternative_class_index", -999)) != alternative:
        errors.append(f"{prefix}: alternative class mismatch")
    try:
        first, second = _evidence_maps(candidate, evidence)
    except Exception:
        if decision.get("model_evidence_valid") is not False:
            errors.append(f"{prefix}: invalid evidence was not marked invalid")
        if int(decision.get("arbitrated_class_index", -999)) != incumbent or decision.get("class_changed") is not False:
            errors.append(f"{prefix}: invalid evidence did not keep frozen control")
        if decision.get("change_reason") != "invalid_evidence_keep_frozen_control":
            errors.append(f"{prefix}: invalid evidence reason mismatch")
        if not isinstance(decision.get("fallback_reason"), str) or not decision["fallback_reason"]:
            errors.append(f"{prefix}: invalid evidence fallback provenance is missing")
        if decision.get("decision_gates") != {
            "alternative_supported_ab": None,
            "alternative_supported_ba": None,
            "alternative_no_strong_counterevidence_ab": None,
            "alternative_no_strong_counterevidence_ba": None,
            "incumbent_strong_counterevidence_ab": None,
            "incumbent_strong_counterevidence_ba": None,
        }:
            errors.append(f"{prefix}: invalid evidence gates are not empty")
        if decision.get("all_change_gates_passed") is not False or decision.get("evidence_snapshot") is not None:
            errors.append(f"{prefix}: invalid evidence retained decision evidence")
    else:
        expected = _expected_valid_decision(candidate, first, second)
        comparisons = {
            "model_evidence_valid": True,
            "arbitrated_class_index": expected["selected"],
            "class_changed": expected["changed"],
            "change_reason": expected["reason"],
            "fallback_reason": None,
            "decision_gates": expected["gates"],
            "all_change_gates_passed": expected["passed"],
            "evidence_snapshot": expected["snapshot"],
        }
        for key, value in comparisons.items():
            if decision.get(key) != value:
                errors.append(f"{prefix}: {key} does not match independent recomputation")
    if str(decision.get("decision_source_task_id", "")) != str(candidate.get("task_id", "")):
        errors.append(f"{prefix}: decision task join mismatch")
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
        if decision.get(key) is not False:
            errors.append(f"{prefix}: {key} is not false")
    if decision.get("class_decision_made") is not True:
        errors.append(f"{prefix}: class_decision_made is not true")
    if decision.get("ground_truth_usage") != "none":
        errors.append(f"{prefix}: ground_truth_usage is not none")
    if decision.get("ground_truth_read") is not False or decision.get("ap_computed") is not False:
        errors.append(f"{prefix}: GT/AP provenance is not false")


def audit(
    decision_root: Path, candidate_manifest: Path, evidence_ledger: Path,
) -> dict:
    errors: list[str] = []
    candidate_rows = _read_jsonl(candidate_manifest)
    pair_rows = [
        row for row in candidate_rows
        if isinstance(row.get("candidate_hypotheses"), list)
        and len(row["candidate_hypotheses"]) == 2
    ]
    evidence_rows = _read_jsonl(evidence_ledger)
    decisions = _read_jsonl(decision_root / "v2_safe_decisions.jsonl")
    summary = json.loads((decision_root / "summary.json").read_text())
    candidates = _by_task(pair_rows, "task_id", "pair candidate manifest", errors)
    evidence = _by_task(evidence_rows, "task_id", "evidence ledger", errors)
    decision_by_task = _by_task(
        decisions, "decision_source_task_id", "v2 decision ledger", errors,
    )
    if set(candidates) != set(evidence):
        errors.append("evidence coverage does not exactly match two-candidate tasks")
    if set(candidates) != set(decision_by_task):
        errors.append("decision coverage does not exactly match two-candidate tasks")
    for index, task_id in enumerate(sorted(set(candidates) & set(evidence) & set(decision_by_task))):
        _audit_row(
            candidates[task_id], evidence[task_id], decision_by_task[task_id],
            f"row[{index}] {task_id}", errors,
        )
    identities = [
        (row.get("scene_name"), row.get("geometry_hash")) for row in decisions
    ]
    if len(identities) != len(set(identities)):
        errors.append("decision ledger contains duplicate geometry identities")
    reason_counts = Counter(row.get("change_reason") for row in decisions)
    expected_summary = {
        "version": SUMMARY_VERSION,
        "rule_id": RULE_ID,
        "fi1_foundation": FI1_FOUNDATION,
        "two_candidate_count": len(decisions),
        "model_evidence_valid_count": sum(row.get("model_evidence_valid") is True for row in decisions),
        "invalid_evidence_fallback_count": sum(row.get("model_evidence_valid") is False for row in decisions),
        "class_change_count": sum(row.get("class_changed") is True for row in decisions),
        "kept_frozen_control_count": sum(row.get("class_changed") is False for row in decisions),
        "change_reason_counts": dict(sorted(reason_counts.items())),
        "decision_coverage_fraction": 1.0 if decisions else 0.0,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            errors.append(f"summary {key} mismatch")
    provenance = summary.get("input_provenance")
    if not isinstance(provenance, dict):
        errors.append("summary input_provenance is missing")
    else:
        if provenance.get("candidate_manifest_sha256") != _sha256(candidate_manifest):
            errors.append("summary candidate manifest hash mismatch")
        if provenance.get("evidence_ledger_sha256") != _sha256(evidence_ledger):
            errors.append("summary evidence ledger hash mismatch")
    result = {
        "version": AUDIT_VERSION,
        "fi1_foundation": FI1_FOUNDATION,
        "row_count": len(decisions),
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    (decision_root / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_root", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evidence-ledger", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.decision_root, args.candidate_manifest, args.evidence_ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the preregistered DM-SMS-1 v2 dual-counterevidence decision rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


DECISION_VERSION = "dm_sms1_v2_dual_counterevidence_decision_v1"
SUMMARY_VERSION = "dm_sms1_v2_dual_counterevidence_ledger_v1"
RULE_ID = "alternative_dual_support_no_counterevidence_and_incumbent_dual_counterevidence"
FI1_FOUNDATION = "FI1-Legacy"


class EvidenceValidationError(ValueError):
    """An evidence row is unsafe to use and must fall back to the incumbent."""


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_by_task(rows: list[dict], label: str) -> dict[str, dict]:
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids):
        raise ValueError(f"{label} contains an empty task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{label} contains duplicate task_id")
    return dict(zip(task_ids, rows))


def _candidate_contract(manifest_row: dict) -> tuple[int, int, set[int], list[int], list[int]]:
    task_id = str(manifest_row.get("task_id", ""))
    if not task_id:
        raise ValueError("candidate row contains an empty task_id")
    for key in ("scene_name", "geometry_key", "geometry_hash"):
        if not isinstance(manifest_row.get(key), str) or not manifest_row[key]:
            raise ValueError(f"{task_id}: candidate row misses {key}")
    candidates = manifest_row.get("candidate_hypotheses")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError(f"{task_id}: v2 requires exactly two candidate hypotheses")
    indices: list[int] = []
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError(f"{task_id}: candidate hypothesis is not an object")
        index = item.get("class_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{task_id}: candidate class_index is not an integer")
        indices.append(index)
    if len(set(indices)) != 2:
        raise ValueError(f"{task_id}: candidate class indices are not unique")
    incumbent = manifest_row.get("canonical_frozen_class_index")
    if isinstance(incumbent, bool) or not isinstance(incumbent, int) or incumbent not in indices:
        raise ValueError(f"{task_id}: frozen class is outside the two candidates")
    alternative = next(index for index in indices if index != incumbent)
    names = [item.get("class_name") for item in candidates]
    has_any_order_field = any(
        key in manifest_row for key in ("candidate_order_ab", "candidate_order_ba")
    )
    if has_any_order_field:
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{task_id}: named AB/BA order requires candidate class names")
        if len(set(names)) != 2:
            raise ValueError(f"{task_id}: candidate class names are not unique")
        expected_names_ab = names
        expected_names_ba = list(reversed(names))
        if manifest_row.get("candidate_order_ab") != expected_names_ab:
            raise ValueError(f"{task_id}: candidate_order_ab is not the registered forward order")
        if manifest_row.get("candidate_order_ba") != expected_names_ba:
            raise ValueError(f"{task_id}: candidate_order_ba is not the exact reverse order")
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
        if manifest_row.get(key) is True:
            raise ValueError(f"{task_id}: candidate manifest reports {key}=true")
    if manifest_row.get("ground_truth_read") is True or manifest_row.get("ap_computed") is True:
        raise ValueError(f"{task_id}: candidate manifest violates the no-GT/no-AP contract")
    return incumbent, alternative, set(indices), indices, list(reversed(indices))


def _validate_item(item: object, task_id: str, order_name: str) -> dict:
    if not isinstance(item, dict):
        raise EvidenceValidationError(f"{task_id}: {order_name} candidate result is not an object")
    index = item.get("class_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise EvidenceValidationError(f"{task_id}: {order_name} class_index is not an integer")
    for key in ("supported", "strong_counterevidence"):
        if not isinstance(item.get(key), bool):
            raise EvidenceValidationError(f"{task_id}: {order_name} {key} is not boolean")
    for key in ("support_evidence", "counterevidence"):
        if not isinstance(item.get(key), str):
            raise EvidenceValidationError(f"{task_id}: {order_name} {key} is not a string")
    confidence = item.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise EvidenceValidationError(f"{task_id}: {order_name} confidence is outside [0,1]")
    return {
        "class_index": int(index),
        "supported": item["supported"],
        "strong_counterevidence": item["strong_counterevidence"],
        "support_evidence": item["support_evidence"],
        "counterevidence": item["counterevidence"],
        "confidence": float(confidence),
    }


def _validated_evidence_maps(
    manifest_row: dict, evidence_row: dict,
) -> tuple[dict[int, dict], dict[int, dict]]:
    task_id = str(manifest_row["task_id"])
    if not isinstance(evidence_row, dict):
        raise EvidenceValidationError(f"{task_id}: evidence row is not an object")
    if str(evidence_row.get("task_id", "")) != task_id:
        raise EvidenceValidationError(f"{task_id}: evidence task_id join mismatch")
    for key in ("scene_name", "geometry_hash"):
        if key in evidence_row and evidence_row.get(key) != manifest_row.get(key):
            raise EvidenceValidationError(f"{task_id}: evidence {key} join mismatch")
    if "valid" in evidence_row and not isinstance(evidence_row.get("valid"), bool):
        raise EvidenceValidationError(f"{task_id}: evidence valid flag is not boolean")
    if evidence_row.get("valid") is False:
        raise EvidenceValidationError(
            f"{task_id}: source reported invalid evidence: {evidence_row.get('error')}"
        )
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
        if evidence_row.get(key) is True:
            raise EvidenceValidationError(f"{task_id}: evidence reports {key}=true")
    if evidence_row.get("ground_truth_read") is True or evidence_row.get("ap_computed") is True:
        raise EvidenceValidationError(f"{task_id}: evidence violates the no-GT/no-AP contract")
    _incumbent, _alternative, allowed, expected_ab, expected_ba = _candidate_contract(manifest_row)
    maps = []
    for order_name, expected_order in (("order_ab", expected_ab), ("order_ba", expected_ba)):
        order = evidence_row.get(order_name)
        if not isinstance(order, dict):
            raise EvidenceValidationError(f"{task_id}: evidence misses {order_name}")
        items = order.get("candidate_results")
        if not isinstance(items, list) or len(items) != 2:
            raise EvidenceValidationError(f"{task_id}: {order_name} does not contain two results")
        normalized = [_validate_item(item, task_id, order_name) for item in items]
        observed_order = [item["class_index"] for item in normalized]
        if observed_order != expected_order or set(observed_order) != allowed:
            raise EvidenceValidationError(f"{task_id}: {order_name} candidate order mismatch")
        maps.append({item["class_index"]: item for item in normalized})
    return maps[0], maps[1]


def _empty_gates() -> dict[str, None]:
    return {
        "alternative_supported_ab": None,
        "alternative_supported_ba": None,
        "alternative_no_strong_counterevidence_ab": None,
        "alternative_no_strong_counterevidence_ba": None,
        "incumbent_strong_counterevidence_ab": None,
        "incumbent_strong_counterevidence_ba": None,
    }


def _base_decision(manifest_row: dict, incumbent: int, alternative: int) -> dict:
    return {
        "version": DECISION_VERSION,
        "rule_id": RULE_ID,
        "fi1_foundation": FI1_FOUNDATION,
        "scene_name": manifest_row["scene_name"],
        "geometry_key": manifest_row["geometry_key"],
        "geometry_hash": manifest_row["geometry_hash"],
        "canonical_frozen_class_index": incumbent,
        "alternative_class_index": alternative,
        "arbitrated_class_index": incumbent,
        "class_changed": False,
        "decision_gates": _empty_gates(),
        "all_change_gates_passed": False,
        "evidence_snapshot": None,
        "decision_source_task_id": manifest_row["task_id"],
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "class_decision_made": True,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
    }


def _snapshot(
    first: dict[int, dict], second: dict[int, dict], incumbent: int, alternative: int,
) -> dict:
    return {
        "order_ab": {
            "incumbent": dict(first[incumbent]),
            "alternative": dict(first[alternative]),
        },
        "order_ba": {
            "incumbent": dict(second[incumbent]),
            "alternative": dict(second[alternative]),
        },
    }


def decide_row(manifest_row: dict, evidence_row: dict) -> dict:
    """Return one safe v2 decision; invalid evidence always keeps the incumbent."""
    incumbent, alternative, _allowed, _expected_ab, _expected_ba = _candidate_contract(manifest_row)
    decision = _base_decision(manifest_row, incumbent, alternative)
    try:
        first, second = _validated_evidence_maps(manifest_row, evidence_row)
    except EvidenceValidationError as error:
        decision.update({
            "change_reason": "invalid_evidence_keep_frozen_control",
            "fallback_reason": f"{type(error).__name__}: {error}",
            "model_evidence_valid": False,
        })
        return decision

    gates = {
        "alternative_supported_ab": first[alternative]["supported"],
        "alternative_supported_ba": second[alternative]["supported"],
        "alternative_no_strong_counterevidence_ab": not first[alternative]["strong_counterevidence"],
        "alternative_no_strong_counterevidence_ba": not second[alternative]["strong_counterevidence"],
        "incumbent_strong_counterevidence_ab": first[incumbent]["strong_counterevidence"],
        "incumbent_strong_counterevidence_ba": second[incumbent]["strong_counterevidence"],
    }
    all_passed = all(gates.values())
    if all_passed:
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
    decision.update({
        "arbitrated_class_index": selected,
        "class_changed": selected != incumbent,
        "change_reason": reason,
        "fallback_reason": None,
        "model_evidence_valid": True,
        "decision_gates": gates,
        "all_change_gates_passed": all_passed,
        "evidence_snapshot": _snapshot(first, second, incumbent, alternative),
    })
    return decision


def run(candidate_manifest: Path, evidence_ledger: Path, output_root: Path) -> dict:
    if not output_root.name.startswith("dm_sms1_v2_"):
        raise ValueError("v2 output directory name must start with dm_sms1_v2_")
    candidates = _unique_by_task(_read_jsonl(candidate_manifest), "candidate manifest")
    pair_candidates = {
        task_id: row
        for task_id, row in candidates.items()
        if isinstance(row.get("candidate_hypotheses"), list)
        and len(row["candidate_hypotheses"]) == 2
    }
    for row in pair_candidates.values():
        _candidate_contract(row)
    evidence = _unique_by_task(_read_jsonl(evidence_ledger), "evidence ledger")
    if set(evidence) != set(pair_candidates):
        missing = sorted(set(pair_candidates) - set(evidence))[:3]
        extra = sorted(set(evidence) - set(pair_candidates))[:3]
        raise ValueError(f"evidence coverage does not match two-candidate tasks: missing={missing}, extra={extra}")
    decisions = [
        decide_row(pair_candidates[task_id], evidence[task_id])
        for task_id in sorted(pair_candidates)
    ]
    output_root.mkdir(parents=True, exist_ok=False)
    decision_path = output_root / "v2_safe_decisions.jsonl"
    with decision_path.open("w") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    reason_counts = Counter(row["change_reason"] for row in decisions)
    summary = {
        "version": SUMMARY_VERSION,
        "rule_id": RULE_ID,
        "fi1_foundation": FI1_FOUNDATION,
        "two_candidate_count": len(decisions),
        "model_evidence_valid_count": sum(row["model_evidence_valid"] for row in decisions),
        "invalid_evidence_fallback_count": sum(not row["model_evidence_valid"] for row in decisions),
        "class_change_count": sum(row["class_changed"] for row in decisions),
        "kept_frozen_control_count": sum(not row["class_changed"] for row in decisions),
        "change_reason_counts": dict(sorted(reason_counts.items())),
        "decision_coverage_fraction": 1.0 if decisions else 0.0,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "input_provenance": {
            "candidate_manifest_sha256": _sha256(candidate_manifest),
            "evidence_ledger_sha256": _sha256(evidence_ledger),
        },
        "scope": "preregistered v2 no-GT pair decisions; not accuracy or AP evidence",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evidence-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(args.candidate_manifest, args.evidence_ledger, args.output_root),
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()

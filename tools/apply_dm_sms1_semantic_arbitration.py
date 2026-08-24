#!/usr/bin/env python3
"""Apply the frozen two-order semantic arbitration rule to model evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _result_map(order: dict) -> dict[int, dict]:
    return {int(item["class_index"]): item for item in order.get("candidate_results", [])}


def _unique_rows_by_task(rows: list[dict], label: str) -> dict[str, dict]:
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids):
        raise ValueError(f"{label} contains an empty task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{label} contains duplicate task_id")
    return dict(zip(task_ids, rows))


def validate_evidence_row(manifest_row: dict, evidence_row: dict) -> None:
    candidates = manifest_row.get("candidate_hypotheses", [])
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError(f"{manifest_row['geometry_key']}: candidate count is outside the frozen contract")
    candidate_indices = [int(item["class_index"]) for item in candidates]
    if len(candidate_indices) != len(set(candidate_indices)):
        raise ValueError(f"{manifest_row['geometry_key']}: duplicate candidate class index")
    allowed = set(candidate_indices)
    incumbent = int(manifest_row["canonical_frozen_class_index"])
    if incumbent not in allowed:
        raise ValueError(f"{manifest_row['geometry_key']}: frozen class is outside finite candidates")
    if str(evidence_row.get("task_id", "")) != str(manifest_row.get("task_id", "")):
        raise ValueError(f"{manifest_row['geometry_key']}: task_id join mismatch")
    for order_name in ("order_ab", "order_ba"):
        order = evidence_row.get(order_name)
        if not isinstance(order, dict):
            raise ValueError(f"{manifest_row['geometry_key']}: missing {order_name}")
        items = order.get("candidate_results")
        if not isinstance(items, list) or len(items) != len(allowed):
            raise ValueError(f"{manifest_row['geometry_key']}: invalid {order_name} candidate set")
        indices = [int(item.get("class_index", -1)) for item in items if isinstance(item, dict)]
        if len(indices) != len(items) or len(indices) != len(set(indices)) or set(indices) != allowed:
            raise ValueError(f"{manifest_row['geometry_key']}: invalid {order_name} candidate set")
        for item in items:
            if not isinstance(item.get("supported"), bool) or not isinstance(item.get("strong_counterevidence"), bool):
                raise ValueError(f"{manifest_row['geometry_key']}: invalid support flags")
            if not isinstance(item.get("support_evidence"), str) or not isinstance(item.get("counterevidence"), str):
                raise ValueError(f"{manifest_row['geometry_key']}: missing evidence text")
            if item["supported"] and not item["support_evidence"].strip():
                raise ValueError(
                    f"{manifest_row['geometry_key']}: support is true without non-empty evidence text"
                )
            if item["strong_counterevidence"] and not item["counterevidence"].strip():
                raise ValueError(
                    f"{manifest_row['geometry_key']}: strong counterevidence is true without non-empty text"
                )
            confidence = item.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0):
                raise ValueError(f"{manifest_row['geometry_key']}: invalid evidence confidence")


def decide_row(manifest_row: dict, evidence_row: dict) -> dict:
    validate_evidence_row(manifest_row, evidence_row)
    candidates = manifest_row.get("candidate_hypotheses", [])
    allowed = {int(item["class_index"]) for item in candidates}
    incumbent = int(manifest_row["canonical_frozen_class_index"])
    first = _result_map(evidence_row["order_ab"])
    second = _result_map(evidence_row["order_ba"])
    if set(first) != allowed or set(second) != allowed:
        raise ValueError(f"{manifest_row['geometry_key']}: evidence candidate set mismatch")
    alternatives = [value for value in allowed if value != incumbent]
    approved = [
        value for value in alternatives
        if bool(first[value].get("supported"))
        and bool(second[value].get("supported"))
        and not bool(first[value].get("strong_counterevidence"))
        and not bool(second[value].get("strong_counterevidence"))
    ]
    # The manifest has at most one alternative.  The explicit length check
    # keeps this rule safe if a future input contract is accidentally widened.
    if len(approved) > 1:
        raise ValueError(f"{manifest_row['geometry_key']}: more than one approved alternative")
    selected = approved[0] if approved else incumbent
    changed = selected != incumbent
    return {
        "scene_name": manifest_row["scene_name"],
        "plan_key": manifest_row["plan_key"],
        "fi1_d_v3_plan_key": manifest_row["plan_key"],
        "geometry_key": manifest_row["geometry_key"],
        "visual_geometry_key": manifest_row["visual_geometry_key"],
        "geometry_hash": manifest_row["geometry_hash"],
        "candidate_source": manifest_row["candidate_source"],
        "challenger_score": manifest_row["challenger_score"],
        "append_only": manifest_row["append_only"],
        "canonical_frozen_class_index": incumbent,
        "arbitrated_class_index": int(selected),
        "class_changed": bool(changed),
        "change_reason": "both_order_support" if changed else "keep_frozen_control",
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "class_decision_made": True,
        "decision_source_task_id": evidence_row["task_id"],
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
    }


def run(manifest_path: Path, evidence_path: Path, output_root: Path) -> dict:
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    evidence_rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
    manifest_by_task = _unique_rows_by_task(manifest_rows, "candidate manifest")
    evidence_by_task = _unique_rows_by_task(evidence_rows, "evidence ledger")
    if set(manifest_by_task) != set(evidence_by_task):
        raise ValueError("evidence does not cover exactly the candidate manifest")
    decisions = [decide_row(manifest_by_task[task_id], evidence_by_task[task_id]) for task_id in sorted(manifest_by_task)]
    if any(row["geometry_mutation"] or row["score_mutation"] or row["candidate_mutation"] for row in decisions):
        raise AssertionError("semantic arbitration mutated a frozen field")
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "semantic_arbitration_decisions.jsonl").open("w") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_semantic_arbitration_decisions_v1",
        "candidate_count": len(decisions),
        "geometry_count": len(decisions),
        "unique_geometry_count": len({(row["scene_name"], row["geometry_hash"]) for row in decisions}),
        "class_change_count": sum(row["class_changed"] for row in decisions),
        "class_decision_made": True,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.evidence, args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

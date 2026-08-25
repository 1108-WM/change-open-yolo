#!/usr/bin/env python3
"""Merge audited pair decisions with deterministic single-candidate keeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.dm_sms1_terminal_safe_keep import (  # noqa: E402
    TERMINAL_KEEP_REASON,
    terminal_identity,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(pair_decisions: Path, candidate_manifest: Path, output_root: Path) -> dict:
    candidates = _read_jsonl(candidate_manifest)
    pair_rows = _read_jsonl(pair_decisions)
    candidate_ids = [str(row.get("task_id", "")) for row in candidates]
    pair_ids = [str(row.get("decision_source_task_id", "")) for row in pair_rows]
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate manifest has empty or duplicate task_id")
    if any(not value for value in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair decisions have empty or duplicate source task_id")
    candidates_by_id = dict(zip(candidate_ids, candidates))
    pairs_by_id = dict(zip(pair_ids, pair_rows))
    expected_pair_ids = {
        str(row["task_id"]) for row in candidates if len(row.get("candidate_hypotheses", [])) == 2
    }
    if set(pair_ids) != expected_pair_ids:
        raise ValueError("pair decision coverage does not exactly match two-candidate tasks")

    decisions = []
    for candidate in candidates:
        task_id = str(candidate["task_id"])
        hypotheses = candidate.get("candidate_hypotheses", [])
        incumbent = int(candidate["canonical_frozen_class_index"])
        if candidate.get("terminal_safe_keep") is True:
            if (
                hypotheses != [] or incumbent != 198
                or not terminal_identity(
                    int(candidate.get("plan_index", -1)), str(candidate.get("plan_key", ""))
                )
            ):
                raise ValueError(f"{task_id}: invalid terminal-safe-keep candidate")
            decision = {
                "scene_name": candidate["scene_name"], "plan_key": candidate["plan_key"],
                "fi1_d_v3_plan_key": candidate["plan_key"], "geometry_key": candidate["geometry_key"],
                "visual_geometry_key": candidate["visual_geometry_key"], "geometry_hash": candidate["geometry_hash"],
                "candidate_source": candidate["candidate_source"], "challenger_score": candidate["challenger_score"],
                "append_only": candidate["append_only"], "frozen_class_index": incumbent,
                "canonical_frozen_class_index": incumbent, "arbitrated_class_index": incumbent,
                "class_changed": False, "change_reason": "terminal_safe_keep",
                "decision_source": "terminal_safe_keep", "decision_source_task_id": task_id,
                "decision_path": "terminal_safe_keep", "model_evidence_used": False,
                "model_evidence_valid": None, "fallback_reason": None,
                "terminal_safe_keep": True, "terminal_keep_reason": TERMINAL_KEEP_REASON,
                "candidate_mutation": False, "geometry_mutation": False, "score_mutation": False,
                "proposal_deletion": False, "class_decision_made": True,
                "ground_truth_usage": "none", "ground_truth_read": False, "ap_computed": False,
            }
        elif len(hypotheses) == 2:
            decision = dict(pairs_by_id[task_id])
            decision["decision_path"] = "two_candidate_vlm_arbitration"
        elif len(hypotheses) == 1:
            only_class = int(hypotheses[0]["class_index"])
            # A singleton may be an alternative-only proposal while the
            # frozen incumbent is background (-1). It is never promoted:
            # retain the frozen incumbent and record the finite candidate.
            decision = {
                "scene_name": candidate["scene_name"],
                "plan_key": candidate["plan_key"],
                "fi1_d_v3_plan_key": candidate["plan_key"],
                "geometry_key": candidate["geometry_key"],
                "visual_geometry_key": candidate["visual_geometry_key"],
                "geometry_hash": candidate["geometry_hash"],
                "candidate_source": candidate["candidate_source"],
                "challenger_score": candidate["challenger_score"],
                "append_only": candidate["append_only"],
                "canonical_frozen_class_index": incumbent,
                "arbitrated_class_index": incumbent,
                "class_changed": False,
                "change_reason": "single_candidate_keep_frozen_control",
                "fallback_reason": None,
                "model_evidence_valid": None,
                "decision_source_task_id": task_id,
                "decision_path": "single_candidate_deterministic_keep",
                "singleton_candidate_class_index": only_class,
                "candidate_mutation": False,
                "geometry_mutation": False,
                "score_mutation": False,
                "proposal_deletion": False,
                "class_decision_made": True,
                "ground_truth_usage": "none",
                "ground_truth_read": False,
                "ap_computed": False,
            }
        else:
            raise ValueError(f"{task_id}: expected one or two candidates")
        decision["plan_index"] = int(candidate["plan_index"])
        if (decision.get("scene_name") != candidate.get("scene_name")
                or decision.get("plan_key") != candidate.get("plan_key")
                or decision.get("geometry_hash") != candidate.get("geometry_hash")):
            raise ValueError(f"{task_id}: decision identity mismatch")
        decisions.append(decision)

    identities = [(row["scene_name"], row["plan_key"]) for row in decisions]
    if len(identities) != len(set(identities)):
        raise ValueError("full ledger has duplicate scene/plan identities")
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "safe_decisions.jsonl").open("w") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_full_safe_decision_ledger_v1",
        "candidate_count": len(decisions),
        "geometry_count": len(decisions),
        "unique_geometry_count": len({(row["scene_name"], row["geometry_hash"]) for row in decisions}),
        "candidate_deletion_count": 0,
        "two_candidate_count": len(expected_pair_ids),
        "single_candidate_count": sum(
            len(row.get("candidate_hypotheses", [])) == 1 for row in candidates
        ),
        "model_evidence_valid_count": sum(row.get("model_evidence_valid") is True for row in decisions),
        "invalid_evidence_fallback_count": sum(row.get("model_evidence_valid") is False for row in decisions),
        "single_candidate_keep_count": sum(row.get("decision_path") == "single_candidate_deterministic_keep" for row in decisions),
        "terminal_safe_keep_count": sum(row.get("decision_path") == "terminal_safe_keep" for row in decisions),
        "class_change_count": sum(row["class_changed"] for row in decisions),
        "kept_frozen_control_count": sum(not row["class_changed"] for row in decisions),
        "decision_coverage_fraction": 1.0 if decisions else 0.0,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "scope": "complete no-GT decisions for all frozen geometries; not accuracy evidence",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-decisions", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pair_decisions, args.candidate_manifest, args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

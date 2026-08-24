#!/usr/bin/env python3
"""Materialize complete safe decisions from valid evidence and fallback rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.apply_dm_sms1_semantic_arbitration import decide_row
from tools.run_dm_sms1_vlm_batch_smoke import (
    validate_completed_model_record,
)


def fallback_decision(manifest_row: dict, task_id: str, reason: str) -> dict:
    incumbent = int(manifest_row["canonical_frozen_class_index"])
    allowed = {int(item["class_index"]) for item in manifest_row.get("candidate_hypotheses", [])}
    if incumbent not in allowed:
        raise ValueError(f"{task_id}: frozen class is outside finite candidates")
    return {
        "scene_name": manifest_row["scene_name"],
        "geometry_key": manifest_row["geometry_key"],
        "geometry_hash": manifest_row["geometry_hash"],
        "canonical_frozen_class_index": incumbent,
        "arbitrated_class_index": incumbent,
        "class_changed": False,
        "change_reason": "invalid_evidence_keep_frozen_control",
        "fallback_reason": reason,
        "model_evidence_valid": False,
        "decision_source_task_id": task_id,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "class_decision_made": True,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
    }


def run(
    batch_outputs: Path, candidate_manifest: Path, output_root: Path,
    config_path: Path = Path("pretrained/config_scannet200.yaml"),
    attribute_manifest: Path | None = None,
) -> dict:
    outputs = [json.loads(line) for line in batch_outputs.read_text().splitlines() if line.strip()]
    candidate_rows = [
        json.loads(line) for line in candidate_manifest.read_text().splitlines() if line.strip()
    ]
    output_ids = [str(row.get("task_id", "")) for row in outputs]
    candidate_ids = [str(row.get("task_id", "")) for row in candidate_rows]
    if any(not task_id for task_id in output_ids) or len(output_ids) != len(set(output_ids)):
        raise ValueError("batch outputs contain an empty or duplicate task_id")
    if any(not task_id for task_id in candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate manifest contains an empty or duplicate task_id")
    candidates = dict(zip(candidate_ids, candidate_rows))
    attribute_rows = {}
    if attribute_manifest is not None:
        attribute_list = [
            json.loads(line) for line in attribute_manifest.read_text().splitlines() if line.strip()
        ]
        attribute_ids = [str(row.get("task_id", "")) for row in attribute_list]
        if any(not task_id for task_id in attribute_ids) or len(attribute_ids) != len(set(attribute_ids)):
            raise ValueError("attribute manifest contains an empty or duplicate task_id")
        attribute_rows = dict(zip(attribute_ids, attribute_list))
        missing_attribute_ids = set(output_ids) - set(attribute_rows)
        if missing_attribute_ids:
            raise ValueError(f"attribute manifest misses batch tasks: {sorted(missing_attribute_ids)[:3]}")
    config = yaml.safe_load(config_path.read_text())
    all_class_names = {
        str(name).strip().lower()
        for name in config["network2d"]["text_prompts"] if str(name).strip()
    }
    missing = set(output_ids) - set(candidates)
    if missing:
        raise ValueError(f"batch outputs contain tasks absent from candidate manifest: {sorted(missing)[:3]}")
    decisions = []
    strict_validation_fallback_count = 0
    for row in outputs:
        task_id = str(row["task_id"])
        manifest_row = candidates[task_id]
        if row.get("scene_name") != manifest_row.get("scene_name") or row.get("geometry_hash") != manifest_row.get("geometry_hash"):
            raise ValueError(f"{task_id}: batch/candidate identity mismatch")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            raise ValueError(f"{task_id}: batch row violates no-GT/no-AP contract")
        if not isinstance(row.get("valid"), bool):
            raise ValueError(f"{task_id}: valid is not boolean")
        if row.get("valid") is True:
            try:
                if not attribute_rows:
                    raise ValueError("strict evidence validation requires the attribute manifest")
                validate_completed_model_record(
                    row, manifest_row, attribute_rows[task_id], all_class_names,
                )
                source_decision = row.get("decision")
                if not isinstance(source_decision, dict):
                    raise ValueError(f"{task_id}: valid row misses decision")
                recomputed_decision = decide_row(manifest_row, row)
                if source_decision != recomputed_decision:
                    raise ValueError(f"{task_id}: stored decision does not match recomputed two-order arbitration")
                decision = dict(source_decision)
                allowed = {int(item["class_index"]) for item in manifest_row.get("candidate_hypotheses", [])}
                incumbent = int(manifest_row["canonical_frozen_class_index"])
                selected = int(decision.get("arbitrated_class_index", -1))
                if decision.get("decision_source_task_id") != task_id:
                    raise ValueError(f"{task_id}: decision task join mismatch")
                if decision.get("scene_name") != manifest_row.get("scene_name") or decision.get("geometry_hash") != manifest_row.get("geometry_hash"):
                    raise ValueError(f"{task_id}: decision identity mismatch")
                if int(decision.get("canonical_frozen_class_index", -1)) != incumbent:
                    raise ValueError(f"{task_id}: decision frozen class mismatch")
                if selected not in allowed or bool(decision.get("class_changed")) != (selected != incumbent):
                    raise ValueError(f"{task_id}: decision class contract mismatch")
                if any(decision.get(key) is not False for key in (
                    "candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion",
                )):
                    raise ValueError(f"{task_id}: valid decision mutated a frozen field")
                decision["model_evidence_valid"] = True
                decision["fallback_reason"] = None
            except Exception as error:
                strict_validation_fallback_count += 1
                decision = fallback_decision(
                    manifest_row, task_id,
                    f"strict_evidence_validation_failed: {type(error).__name__}: {error}",
                )
        else:
            if "decision" in row:
                raise ValueError(f"{task_id}: invalid row unexpectedly contains a decision")
            decision = fallback_decision(manifest_row, task_id, str(row.get("error")))
        decisions.append(decision)
    identities = [(row["scene_name"], row["geometry_hash"]) for row in decisions]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate safe decision identities")
    if any(row["candidate_mutation"] or row["geometry_mutation"] or row["score_mutation"] or row["proposal_deletion"] for row in decisions):
        raise AssertionError("safe decision ledger mutated a frozen field")
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "safe_decisions.jsonl").open("w") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "dm_sms1_safe_decision_ledger_v2",
        "geometry_count": len(decisions),
        "model_evidence_valid_count": sum(row["model_evidence_valid"] for row in decisions),
        "fallback_keep_count": sum(not row["model_evidence_valid"] for row in decisions),
        "strict_validation_fallback_count": strict_validation_fallback_count,
        "class_change_count": sum(row["class_changed"] for row in decisions),
        "kept_frozen_control_count": sum(not row["class_changed"] for row in decisions),
        "decision_coverage_fraction": 1.0 if decisions else 0.0,
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "proposal_deletion": False,
        "ground_truth_read": False,
        "ap_computed": False,
        "scope": "strictly revalidated safe no-GT decisions; not accuracy evidence",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-outputs", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--attribute-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        args.batch_outputs, args.candidate_manifest, args.output_root, args.config_path,
        args.attribute_manifest,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

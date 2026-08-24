#!/usr/bin/env python3
"""Strictly audit a DM-SMS-1 batch from saved raw model outputs without GT/AP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_dm_sms1_vlm_batch_smoke import validate_completed_model_record


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _by_task(rows: list[dict], label: str) -> dict[str, dict]:
    ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains an empty or duplicate task_id")
    return dict(zip(ids, rows))


def audit(
    batch_root: Path, candidate_manifest: Path, attribute_manifest: Path,
    output_root: Path, config_path: Path,
) -> dict:
    selection = _rows(batch_root / "selection.jsonl")
    records = _rows(batch_root / "batch_outputs.jsonl")
    candidates = _by_task(_rows(candidate_manifest), "candidate manifest")
    attributes = _by_task(_rows(attribute_manifest), "attribute manifest")
    config = yaml.safe_load(config_path.read_text())
    all_class_names = {
        str(name).strip().lower()
        for name in config["network2d"]["text_prompts"] if str(name).strip()
    }
    errors: list[str] = []
    strict_valid_count = 0
    strict_invalid_claimed_valid_count = 0
    original_invalid_count = 0
    selection_ids = [str(row.get("task_id", "")) for row in selection]
    record_ids = [str(row.get("task_id", "")) for row in records]
    if any(not task_id for task_id in selection_ids) or len(selection_ids) != len(set(selection_ids)):
        errors.append("selection contains an empty or duplicate task_id")
    if record_ids != selection_ids:
        errors.append("batch outputs do not exactly cover selection order")
    for index, record in enumerate(records):
        task_id = str(record.get("task_id", ""))
        prefix = f"row[{index}] {task_id}"
        candidate = candidates.get(task_id)
        attribute = attributes.get(task_id)
        if candidate is None or attribute is None:
            errors.append(f"{prefix}: task is missing from an input manifest")
            continue
        if record.get("ground_truth_read") is not False or record.get("ap_computed") is not False:
            errors.append(f"{prefix}: no-GT/no-AP provenance is invalid")
        if not isinstance(record.get("valid"), bool):
            errors.append(f"{prefix}: valid is not boolean")
            continue
        if record["valid"]:
            try:
                validate_completed_model_record(
                    record, candidate, attribute, all_class_names,
                )
                strict_valid_count += 1
            except Exception as error:
                strict_invalid_claimed_valid_count += 1
                errors.append(f"{prefix}: {type(error).__name__}: {error}")
        else:
            original_invalid_count += 1
            if "decision" in record:
                errors.append(f"{prefix}: invalid record unexpectedly contains a decision")
            if not isinstance(record.get("error"), str) or not record["error"]:
                errors.append(f"{prefix}: invalid record misses its failure reason")
    result = {
        "version": "dm_sms1_vlm_batch_strict_audit_v1",
        "batch_root": str(batch_root),
        "selected_record_count": len(selection),
        "processed_record_count": len(records),
        "claimed_valid_count": sum(row.get("valid") is True for row in records),
        "strict_valid_count": strict_valid_count,
        "strict_invalid_claimed_valid_count": strict_invalid_claimed_valid_count,
        "original_invalid_count": original_invalid_count,
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "ground_truth_read": False,
        "ap_computed": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--attribute-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    args = parser.parse_args()
    result = audit(
        args.batch_root, args.candidate_manifest, args.attribute_manifest,
        args.output_root, args.config_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

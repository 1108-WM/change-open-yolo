#!/usr/bin/env python3
"""Audit category-blind attribute extraction inputs without GT or AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PROMPT_PARTS = ("不要猜测类别", "外观颜色和纹理", "材质", "形状和结构", "功能线索", "空间关系")


def audit(root: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    records = [json.loads(line) for line in (root / "attribute_extraction_manifest.jsonl").read_text().splitlines() if line.strip()]
    errors: list[str] = []
    task_ids = []
    view_count = 0
    for index, row in enumerate(records):
        prefix = f"row[{index}]"
        task_ids.append(row.get("task_id"))
        if row.get("candidate_labels_hidden") is not True:
            errors.append(f"{prefix}: candidate labels are not hidden")
        if row.get("attribute_extraction_completed") is not False or row.get("class_decision_made") is not False:
            errors.append(f"{prefix}: attribute/class decision already completed")
        for key in ("candidate_mutation", "geometry_mutation", "score_mutation"):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is true")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")
        if any(part not in str(row.get("attribute_prompt", "")) for part in REQUIRED_PROMPT_PARTS):
            errors.append(f"{prefix}: fixed category-blind prompt is incomplete")
        forbidden_keys = {"finite_class_hypotheses", "canonical_frozen_class_index", "alpha_class_index", "class_names", "candidate_labels"}
        if forbidden_keys.intersection(row):
            errors.append(f"{prefix}: candidate label field leaked into model input")
        views = row.get("view_inputs", [])
        if not views or len(views) > 3 or len({view.get("frame_id") for view in views}) != len(views):
            errors.append(f"{prefix}: invalid view input set")
        for view in views:
            for key in ("rgb_path", "depth_path", "pose_path", "intrinsics_path"):
                if not Path(view.get(key, "")).is_file():
                    errors.append(f"{prefix}: missing {key}")
        view_count += len(views)
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate task ids")
    if int(summary.get("task_count", -1)) != len(records):
        errors.append("summary task count mismatch")
    if int(summary.get("view_input_count", -1)) != view_count:
        errors.append("summary view count mismatch")
    if summary.get("candidate_labels_hidden") is not True or summary.get("ground_truth_read") is not False or summary.get("ap_computed") is not False:
        errors.append("summary contract mismatch")
    result = {
        "version": "dm_sms1_attribute_extraction_manifest_audit_v1",
        "row_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "audit_valid": not errors,
        "candidate_labels_hidden": True,
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

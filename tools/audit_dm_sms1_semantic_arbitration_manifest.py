#!/usr/bin/env python3
"""Audit the no-GT semantic-arbitration manifest without reading labels or AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path) -> dict:
    summary_path = root / "summary.json"
    records_path = root / "semantic_arbitration_manifest.jsonl"
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    errors: list[str] = []
    identities = []
    selected_view_count = 0
    hypotheses = 0
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        identities.append((row.get("scene_name"), row.get("plan_key")))
        if row.get("fi1_d_v3_plan_key") != row.get("plan_key") or not row.get("plan_key"):
            errors.append(f"{prefix}: invalid plan_key identity")
        if not row.get("visual_geometry_key") or not row.get("geometry_hash"):
            errors.append(f"{prefix}: visual geometry provenance is incomplete")
        if row.get("ground_truth_read") is not False or row.get("ap_computed") is not False:
            errors.append(f"{prefix}: GT/AP provenance is not false")
        if (
            row.get("candidate_source") != row.get("canonical_candidate_source")
            or row.get("frozen_class_index") != row.get("canonical_frozen_class_index")
            or row.get("challenger_score") != row.get("canonical_frozen_score")
            or row.get("append_only") != row.get("fi1_d_v3_append_only")
            or not isinstance(row.get("geometry_locator_read_only"), dict)
            or row.get("candidate_retained") is not True
            or row.get("candidate_deletion") is not False
        ):
            errors.append(f"{prefix}: frozen candidate fields differ")
        for key in ("candidate_mutation", "geometry_mutation", "class_mutation", "score_mutation", "class_decision_made"):
            if row.get(key) is not False:
                errors.append(f"{prefix}: {key} is true")
        candidates = row.get("finite_class_hypotheses", [])
        if not candidates or len(candidates) > 2:
            errors.append(f"{prefix}: invalid finite class hypothesis count")
        for candidate in candidates:
            value = candidate.get("class_index")
            if not isinstance(value, int) or not 0 <= value < 198:
                errors.append(f"{prefix}: invalid class index")
        views = row.get("selected_views", [])
        if len(views) > 3 or len({view.get("frame_id") for view in views}) != len(views):
            errors.append(f"{prefix}: selected views are not unique or exceed three")
        for view in views:
            for key in ("rgb_path", "depth_path", "pose_path", "intrinsics_path"):
                if not Path(view.get(key, "")).is_file():
                    errors.append(f"{prefix}: missing {key}")
            if not view.get("sam_mask_valid"):
                errors.append(f"{prefix}: selected view has invalid SAM mask")
        selected_view_count += len(views)
        hypotheses += len(candidates)
    if len(identities) != len(set(identities)):
        errors.append("duplicate scene/plan identities")
    unique_geometries = len({(row.get("scene_name"), row.get("geometry_hash")) for row in rows})
    if int(summary.get("candidate_count", -1)) != len(rows):
        errors.append("summary candidate count mismatch")
    if int(summary.get("candidate_deletion_count", -1)) != 0:
        errors.append("summary candidate deletion count mismatch")
    if int(summary.get("unique_geometry_count", -1)) != unique_geometries:
        errors.append("summary unique geometry count mismatch")
    if int(summary.get("selected_view_count", -1)) != selected_view_count:
        errors.append("summary selected view count mismatch")
    if int(summary.get("candidate_hypothesis_count", -1)) != hypotheses:
        errors.append("summary candidate hypothesis count mismatch")
    if summary.get("ground_truth_read") is not False or summary.get("ap_computed") is not False:
        errors.append("summary GT/AP provenance is not false")
    result = {
        "version": "dm_sms1_semantic_arbitration_manifest_audit_v1",
        "manifest_root": str(root),
        "row_count": len(rows),
        "candidate_count": len(rows),
        "unique_geometry_count": unique_geometries,
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
    parser.add_argument("manifest_root", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independently audit a GT-free FI1-D-v3 complete inference plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import _read_jsonl, _resolve, _sha256  # noqa: E402
from tools.build_ncs_fi1_stage_c_member_dataset_gt import _geometry_sha256  # noqa: E402


VERSION = "fi1_d_v3_frozen_inference_plan_audit_v1"


def _points(locator: dict, expected_count: int) -> np.ndarray:
    kind = str(locator.get("kind"))
    if kind == "native_mask_column":
        masks = np.load(_resolve(Path(locator["masks_path"])), mmap_mode="r")
        points = np.flatnonzero(np.asarray(masks[:, int(locator["column_index"])], dtype=bool)).astype(np.int64)
    elif kind == "point_indices_npz":
        with np.load(_resolve(Path(locator["points_path"]))) as payload:
            points = np.unique(np.asarray(payload[str(locator.get("array_key", "point_indices"))], dtype=np.int64))
    else:
        raise ValueError(f"unsupported geometry locator: {kind}")
    if len(points) != expected_count or not len(points) or int(points[0]) < 0:
        raise ValueError("geometry is empty or its point count differs")
    return points


def run(args: argparse.Namespace) -> dict:
    for name in ("inference_root", "unique_geometry_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    inference_summary = json.loads((args.inference_root / "summary.json").read_text())
    plan_root = args.inference_root / "complete_plan"
    plan_summary = json.loads((plan_root / "summary.json").read_text())
    plan_path = plan_root / plan_summary["files"]["plan"]
    plan = _read_jsonl(plan_path)
    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    errors = Counter()
    if _sha256(plan_path) != plan_summary["hashes"]["plan"]:
        errors["plan_sha256_mismatch"] += 1
    if len({str(row["plan_key"]) for row in plan}) != len(plan):
        errors["duplicate_plan_key"] += 1
    seen_frozen = set()
    refined_count = 0
    scenes = set()
    for row in plan:
        scenes.add(str(row["scene_name"]))
        source = str(row["candidate_source"])
        if source == "refined_union":
            refined_count += 1
            if row.get("geometry_key") is not None or row.get("control_score") is not None:
                errors["refined_control_contract_mismatch"] += 1
            original = node_by_key.get(str(row.get("original_union_geometry_key")))
            if original is None or int(row["frozen_class_index"]) != int(original["canonical_frozen_class_index"]):
                errors["refined_class_inheritance_mismatch"] += 1
            if row.get("append_only") is not True:
                errors["refined_not_append_only"] += 1
        else:
            key = str(row.get("geometry_key"))
            node = node_by_key.get(key)
            if node is None:
                errors["frozen_geometry_missing"] += 1
                continue
            seen_frozen.add(key)
            if str(node["canonical_candidate_source"]) != source:
                errors["frozen_source_mismatch"] += 1
            if int(row["frozen_class_index"]) != int(node["canonical_frozen_class_index"]):
                errors["frozen_class_mismatch"] += 1
            if not math.isclose(float(row["control_score"]), float(node["canonical_frozen_score"]), rel_tol=0.0, abs_tol=1e-12):
                errors["control_score_mismatch"] += 1
            if row.get("append_only") is not False:
                errors["frozen_append_contract_mismatch"] += 1
        try:
            points = _points(row["geometry_locator_read_only"], int(row["point_count"]))
            algorithm = str(row["geometry_digest_algorithm"])
            digest = hashlib.sha1(points.tobytes()).hexdigest() if algorithm == "sha1_point_indices" else _geometry_sha256(points)
            if algorithm not in {"sha1_point_indices", "sha256_point_indices"} or digest != str(row["geometry_digest"]):
                errors["geometry_digest_mismatch"] += 1
        except (FileNotFoundError, KeyError, ValueError, OSError):
            errors["geometry_resolution_error"] += 1
        if not math.isfinite(float(row["challenger_score"])) or float(row["challenger_score"]) < 0.0:
            errors["invalid_challenger_score"] += 1
        for flag in ("candidate_deletion", "geometry_mutation", "class_mutation"):
            if row.get(flag) is not False:
                errors[f"contract_{flag}_violation"] += 1
        if row.get("candidate_retained") is not True:
            errors["candidate_not_retained"] += 1
    if seen_frozen != set(node_by_key):
        errors["frozen_geometry_coverage_mismatch"] += 1
    if len(scenes) != int(inference_summary["scene_count"]):
        errors["scene_count_mismatch"] += 1
    checks = {
        "audit_error_count_zero": sum(errors.values()) == 0,
        "control_candidate_count_exact": len(seen_frozen) == len(nodes),
        "challenger_candidate_count_exact": len(plan) == len(nodes) + refined_count,
        "candidate_deletion_count_zero": all(row.get("candidate_deletion") is False for row in plan),
        "geometry_mutation_count_zero": all(row.get("geometry_mutation") is False for row in plan),
        "class_mutation_count_zero": all(row.get("class_mutation") is False for row in plan),
        "ground_truth_usage_none": inference_summary.get("ground_truth_usage") == "none",
        "ap_computed_false": inference_summary.get("ap_computed") is False,
    }
    audit = {
        "version": VERSION, "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())), "errors": dict(sorted(errors.items())),
        "scene_count": len(scenes), "control_candidate_count": len(seen_frozen),
        "challenger_candidate_count": len(plan), "refined_union_candidate_count": refined_count,
        "advancement_gate": {"checks": checks, "advancement_authorized": all(checks.values())},
        "ground_truth_usage": "none", "ap_computed": False,
        "input_provenance": {"inference_summary_sha256": _sha256(args.inference_root / "summary.json"), "plan_sha256": _sha256(plan_path)},
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--unique-geometry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

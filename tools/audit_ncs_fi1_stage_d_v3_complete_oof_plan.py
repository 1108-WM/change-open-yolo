#!/usr/bin/env python3
"""Independently audit the frozen complete D-v3 official100 OOF AP plan."""

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

VERSION = "ncs_fi1_stage_d_v3_complete_oof_plan_audit_v1"


def _close(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _points(locator: dict, expected_count: int) -> np.ndarray:
    kind = str(locator["kind"])
    if kind == "native_mask_column":
        masks = np.load(Path(locator["masks_path"]), mmap_mode="r")
        points = np.flatnonzero(np.asarray(masks[:, int(locator["column_index"])], dtype=bool)).astype(np.int64)
    elif kind == "point_indices_npz":
        with np.load(Path(locator["points_path"])) as payload:
            points = np.unique(np.asarray(payload[str(locator.get("array_key", "point_indices"))], dtype=np.int64))
    else:
        raise ValueError(f"unsupported locator: {kind}")
    if len(points) != expected_count or not len(points) or points[0] < 0:
        raise ValueError("geometry points/count invalid")
    return points


def run(args: argparse.Namespace) -> dict:
    for name in ("plan_root", "unique_geometry_root", "stage_b_root", "stage_d_v3_dataset_root", "stage_d_v3_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    summary = json.loads((args.plan_root / "summary.json").read_text())
    plan_path = args.plan_root / summary["files"]["plan"]
    plan = _read_jsonl(plan_path)
    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    node_by_key = {str(row["geometry_key"]): row for row in nodes}
    stage_b = {str(row["geometry_key"]): row for row in _read_jsonl(args.stage_b_root / "stage_b_rerank_plan.jsonl")}
    dataset = {str(row["candidate_key"]): row for row in _read_jsonl(args.stage_d_v3_dataset_root / "full_rank_marginal_gain_dataset.jsonl")}
    scores = {str(row["candidate_key"]): row for row in _read_jsonl(args.stage_d_v3_root / "stage_d_v3_append_score_plan.jsonl")}
    errors = Counter()
    if _sha256(plan_path) != summary["hashes"]["plan"]:
        errors["plan_sha256_mismatch"] += 1
    if len({str(row["plan_key"]) for row in plan}) != len(plan):
        errors["duplicate_plan_key"] += 1
    original_candidate_by_geometry = {
        str(row["original_union_geometry_key"]): str(row["candidate_key"])
        for row in dataset.values() if row["candidate_variant"] == "original"
    }
    original_count = refined_count = 0
    seen_frozen = set()
    for row in plan:
        source = str(row["candidate_source"])
        if source != "refined_union":
            key = str(row["geometry_key"])
            node = node_by_key.get(key)
            if node is None:
                errors["frozen_geometry_missing"] += 1
                continue
            seen_frozen.add(key)
            expected = float(stage_b[key]["stage_b_score"]) if source in {"native", "track"} else float(scores[original_candidate_by_geometry[key]]["stage_d_v3_append_score"])
            if not _close(row["control_score"], node["canonical_frozen_score"]):
                errors["control_score_mismatch"] += 1
            if not _close(row["challenger_score"], expected):
                errors["challenger_score_mismatch"] += 1
            if int(row["frozen_class_index"]) != int(node["canonical_frozen_class_index"]):
                errors["frozen_class_mismatch"] += 1
            original_count += int(source == "pair_union")
        else:
            refined_count += 1
            source_row = dataset.get(str(row["plan_key"]))
            if source_row is None or source_row["candidate_variant"] != "refined":
                errors["refined_source_missing"] += 1
                continue
            original = node_by_key[str(source_row["original_union_geometry_key"])]
            if int(row["frozen_class_index"]) != int(original["canonical_frozen_class_index"]):
                errors["refined_class_inheritance_mismatch"] += 1
            if not _close(row["challenger_score"], scores[str(row["plan_key"])]["stage_d_v3_append_score"]):
                errors["refined_score_mismatch"] += 1
            if row.get("control_score") is not None or row.get("append_only") is not True:
                errors["refined_append_contract_mismatch"] += 1
        points = _points(row["geometry_locator_read_only"], int(row["point_count"]))
        algorithm = str(row.get("geometry_digest_algorithm"))
        expected_digest = str(row.get("geometry_digest"))
        observed_digest = (
            hashlib.sha1(points.tobytes()).hexdigest()
            if algorithm == "sha1_point_indices" else _geometry_sha256(points)
        )
        if algorithm not in {"sha1_point_indices", "sha256_point_indices"} or observed_digest != expected_digest:
            errors["geometry_sha256_mismatch"] += 1
        if not math.isfinite(float(row["challenger_score"])) or float(row["challenger_score"]) < 0.0:
            errors["invalid_challenger_score"] += 1
        for flag in ("candidate_deletion", "geometry_mutation", "class_mutation"):
            if row.get(flag) is not False:
                errors[f"contract_{flag}_violation"] += 1
        if row.get("candidate_retained") is not True:
            errors["candidate_not_retained"] += 1
    if seen_frozen != set(node_by_key):
        errors["frozen_geometry_coverage_mismatch"] += 1
    counts = Counter(str(row["candidate_source"]) for row in plan)
    expected_unique = len(nodes)
    expected_baseline = sum(str(row["canonical_candidate_source"]) in {"native", "track"} for row in nodes)
    expected_original = sum(row["candidate_variant"] == "original" for row in dataset.values())
    expected_refined = sum(row["candidate_variant"] == "refined" for row in dataset.values())
    checks = {
        "audit_error_count_zero": sum(errors.values()) == 0,
        "control_candidate_count_exact": len(seen_frozen) == expected_unique,
        "challenger_candidate_count_exact": len(plan) == expected_unique + expected_refined,
        "native_track_count_exact": counts["native"] + counts["track"] == expected_baseline,
        "original_union_count_exact": original_count == expected_original,
        "refined_union_count_exact": refined_count == expected_refined,
        "candidate_deletion_count_zero": all(row.get("candidate_deletion") is False for row in plan),
        "geometry_mutation_count_zero": all(row.get("geometry_mutation") is False for row in plan),
        "class_mutation_count_zero": all(row.get("class_mutation") is False for row in plan),
        "ap_computed_false": summary.get("ap_computed") is False,
        "validation60_read_false": summary.get("validation60_read") is False,
        "val312_read_false": summary.get("val312_read") is False,
    }
    audit = {
        "version": VERSION, "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())), "errors": dict(sorted(errors.items())),
        "counts": dict(sorted(counts.items())),
        "advancement_gate": {"checks": checks, "advancement_authorized": all(checks.values())},
        "ap_computed": False, "validation60_read": False, "val312_read": False,
        "input_provenance": {"plan_summary_sha256": _sha256(args.plan_root / "summary.json"), "plan_sha256": _sha256(plan_path)},
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
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--unique-geometry-root", type=Path, default=Path("/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"))
    parser.add_argument("--stage-b-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_b_relation_rerank_plan_train100_20260822"))
    parser.add_argument("--stage-d-v3-dataset-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_d_v3_full_rank_marginal_dataset_train100_20260822"))
    parser.add_argument("--stage-d-v3-root", type=Path, default=Path("docs/diagnostics/ncs_fi1_stage_d_v3_full_rank_marginal_oof_train100_20260822"))
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

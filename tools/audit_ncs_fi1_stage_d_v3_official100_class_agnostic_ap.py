#!/usr/bin/env python3
"""Audit saved AP arrays and result contracts without performing a second AP evaluation."""

from __future__ import annotations

import argparse
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

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import _resolve, _sha256  # noqa: E402
from tools.diagnose_gvc_class_agnostic_ap import _configure_scannet200_instance_eval, instance_eval  # noqa: E402

VERSION = "ncs_fi1_stage_d_v3_official100_class_agnostic_ap_audit_v1"


def _metrics(ap_scores):
    averages = instance_eval.compute_averages(ap_scores)
    chair = averages["classes"]["chair"]
    return {"ap": float(chair["ap"]), "ap50": float(chair["ap50%"]), "ap25": float(chair["ap25%"]) }


def _close(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def run(args: argparse.Namespace) -> dict:
    for name in ("result_root", "plan_root", "plan_audit_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.result_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    arrays_path = args.result_root / summary["files"]["raw_ap_arrays"]
    plan_summary = json.loads((args.plan_root / "summary.json").read_text())
    plan_audit = json.loads((args.plan_audit_root / "summary.json").read_text())
    errors = Counter()
    if _sha256(arrays_path) != summary["hashes"]["raw_ap_arrays"]:
        errors["raw_ap_arrays_sha256_mismatch"] += 1
    if plan_audit.get("audit_valid") is not True or plan_audit.get("advancement_gate", {}).get("advancement_authorized") is not True:
        errors["plan_audit_not_authorized"] += 1
    _configure_scannet200_instance_eval()
    with np.load(arrays_path) as arrays:
        control = _metrics(arrays["control_overall"])
        challenger = _metrics(arrays["challenger_overall"])
        folds = {}
        for fold in range(5):
            folds[str(fold)] = {
                "control": _metrics(arrays[f"control_fold_{fold}"]),
                "challenger": _metrics(arrays[f"challenger_fold_{fold}"]),
            }
    delta = {name: challenger[name] - control[name] for name in ("ap", "ap50", "ap25")}
    for group, observed, expected in (("control", summary["control"], control), ("challenger", summary["challenger"], challenger), ("delta", summary["delta"], delta)):
        for name in ("ap", "ap50", "ap25"):
            if not _close(observed[name], expected[name]):
                errors[f"{group}_{name}_mismatch"] += 1
    for fold in range(5):
        for group in ("control", "challenger"):
            for name in ("ap", "ap50", "ap25"):
                if not _close(summary["folds"][str(fold)][group][name], folds[str(fold)][group][name]):
                    errors[f"fold_{fold}_{group}_{name}_mismatch"] += 1
    checks = {
        "audit_error_count_zero": sum(errors.values()) == 0,
        "single_fixed_challenger_true": summary.get("single_fixed_challenger") is True,
        "threshold_or_weight_scan_count_zero": int(summary.get("threshold_or_weight_scan_count", -1)) == 0,
        "control_candidate_count_exact": int(summary.get("control_candidate_count", -1)) == int(plan_summary["control_candidate_count"]),
        "challenger_candidate_count_exact": int(summary.get("challenger_candidate_count", -1)) == int(plan_summary["challenger_candidate_count"]),
        "candidate_deletion_count_zero": int(summary.get("candidate_deletion_count", -1)) == 0,
        "geometry_mutation_false": summary.get("geometry_mutation") is False,
        "class_mutation_false": summary.get("class_mutation") is False,
        "validation60_read_false": summary.get("validation60_read") is False,
        "val312_read_false": summary.get("val312_read") is False,
    }
    audit = {
        "version": VERSION, "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())), "errors": dict(sorted(errors.items())),
        "control": control, "challenger": challenger, "delta": delta, "folds": folds,
        "primary_class_agnostic_ap_improved": delta["ap"] > 0.0,
        "all_reported_metrics_improved": all(value > 0.0 for value in delta.values()),
        "checks": checks, "contract_valid": all(checks.values()),
        "validation60_read": False, "val312_read": False,
        "input_provenance": {"result_summary_sha256": _sha256(summary_path), "raw_ap_arrays_sha256": _sha256(arrays_path)},
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
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--plan-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

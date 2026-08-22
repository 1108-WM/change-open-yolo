#!/usr/bin/env python3
"""Audit the frozen FI1-D-v3 control/challenger AP result and raw arrays."""

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


VERSION = "fi1_d_v3_frozen_class_agnostic_ap_audit_v1"


def _metrics(scores: np.ndarray) -> dict[str, float]:
    chair = instance_eval.compute_averages(scores)["classes"]["chair"]
    return {"ap": float(chair["ap"]), "ap50": float(chair["ap50%"]), "ap25": float(chair["ap25%"])}


def run(args: argparse.Namespace) -> dict:
    for name in ("result_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.result_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    arrays_path = args.result_root / summary["files"]["raw_ap_arrays"]
    errors = Counter()
    if _sha256(arrays_path) != summary["hashes"]["raw_ap_arrays"]:
        errors["raw_ap_arrays_sha256_mismatch"] += 1
    _configure_scannet200_instance_eval()
    with np.load(arrays_path) as payload:
        control = _metrics(np.asarray(payload["control"]))
        challenger = _metrics(np.asarray(payload["challenger"]))
    for system, observed in (("control", control), ("challenger", challenger)):
        for metric in ("ap", "ap50", "ap25"):
            if not math.isclose(observed[metric], float(summary[system][metric]), rel_tol=0.0, abs_tol=1e-12):
                errors[f"{system}_{metric}_mismatch"] += 1
    for metric in ("ap", "ap50", "ap25"):
        expected = challenger[metric] - control[metric]
        if not math.isclose(expected, float(summary["delta"][metric]), rel_tol=0.0, abs_tol=1e-12):
            errors[f"delta_{metric}_mismatch"] += 1
    if summary.get("single_fixed_challenger") is not True:
        errors["single_fixed_challenger_false"] += 1
    if int(summary.get("threshold_or_weight_scan_count", -1)) != 0:
        errors["threshold_or_weight_scan_nonzero"] += 1
    audit = {
        "version": VERSION, "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())), "errors": dict(sorted(errors.items())),
        "control": control, "challenger": challenger,
        "delta": {metric: challenger[metric] - control[metric] for metric in ("ap", "ap50", "ap25")},
        "primary_metric": "class-agnostic AP",
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
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

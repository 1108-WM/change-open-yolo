#!/usr/bin/env python3
"""Independently audit FI1-D-v3 control/DM-SMS-1 AP CSVs and one-shot state."""

from __future__ import annotations

import argparse
import csv
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

from evaluate.scannet200.scannet_constants import (
    COMMON_CATS_SCANNET_200,
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
)
from tools.dm_sms1_minus1_evaluator_boundary import (
    EXPECTED_FOREGROUND_COUNT,
    EXPECTED_MINUS1_COUNT,
    EXPECTED_MINUS1_PLAN_INDICES,
    EXPECTED_NATIVE_BACKGROUND_198_COUNT,
    EXPECTED_TOTAL_CANDIDATE_COUNT,
    RECOVERY_AUTHORIZATION_ID,
    audit_boundary_inputs,
    validate_frozen_recovery_inputs,
    validate_prior_failure,
)


VERSION = "dm_sms1_fi1_d_v3_open_vocab_ap_audit_v1"
AUTHORIZATION_ID = "DM-SMS-1-FI1-D-v3-val312-one-shot-20260824"
METRICS = ("ap", "ap50", "ap25", "head_ap", "common_ap", "tail_ap")
DUPLICATE_SAFE_PREREGISTRATION = PROJECT_ROOT / "docs/DM_SMS1_FI1_D_V3_VAL312_DUPLICATE_SAFE_PREREGISTRATION_REVISION_20260824.md"
MINUS1_BOUNDARY_PREREGISTRATION = PROJECT_ROOT / "docs/DM_SMS1_FI1_D_V3_VAL312_MINUS1_EVALUATOR_BOUNDARY_PREREGISTRATION_REVISION_20260907.md"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_metrics(path: Path) -> tuple[dict[str, float], int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({str(row.get("class", "")) for row in rows}) != len(rows):
        raise ValueError(f"{path}: AP CSV is empty or has duplicate classes")
    required = {"class", "class id", "ap", "ap50", "ap25"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path}: AP CSV columns differ from the official evaluator")
    groups = {
        "head": set(HEAD_CATS_SCANNET_200),
        "common": set(COMMON_CATS_SCANNET_200),
        "tail": set(TAIL_CATS_SCANNET_200),
    }
    values = {name: [] for name in ("ap", "ap50", "ap25")}
    grouped = {name: [] for name in groups}
    for row in rows:
        class_name = str(row["class"])
        memberships = [name for name, classes in groups.items() if class_name in classes]
        if len(memberships) != 1:
            raise ValueError(f"{path}: class {class_name!r} has invalid frequency-group membership")
        observed = {name: float(row[name]) for name in ("ap", "ap50", "ap25")}
        for name, value in observed.items():
            if not (math.isfinite(value) or math.isnan(value)):
                raise ValueError(f"{path}: invalid {name} for {class_name}")
            values[name].append(value)
        grouped[memberships[0]].append(observed["ap"])
    metrics = {
        "ap": float(np.nanmean(values["ap"])),
        "ap50": float(np.nanmean(values["ap50"])),
        "ap25": float(np.nanmean(values["ap25"])),
        "head_ap": float(np.nanmean(grouped["head"])),
        "common_ap": float(np.nanmean(grouped["common"])),
        "tail_ap": float(np.nanmean(grouped["tail"])),
    }
    return metrics, len(rows)


def run(args: argparse.Namespace) -> dict:
    recovery_mode = bool(getattr(args, "minus1_evaluator_boundary_safe", False))
    expected_authorization = RECOVERY_AUTHORIZATION_ID if recovery_mode else AUTHORIZATION_ID
    if getattr(args, "duplicate_safe_preregistration_path", None) is None:
        args.duplicate_safe_preregistration_path = DUPLICATE_SAFE_PREREGISTRATION
    if recovery_mode and getattr(args, "minus1_boundary_preregistration_path", None) is None:
        args.minus1_boundary_preregistration_path = MINUS1_BOUNDARY_PREREGISTRATION
    for name in (
        "result_root", "scene_list", "cache_root", "cache_audit_root",
        "decision_root", "preregistration_path", "duplicate_safe_preregistration_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if recovery_mode:
        for name in ("minus1_boundary_preregistration_path", "prior_failed_ap_root", "prior_failed_ap_log"):
            setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.result_root / "summary.json"
    started_path = args.result_root / "ap_invocation_started.json"
    completed_path = args.result_root / "ap_invocation_completed.json"
    failed_path = args.result_root / "ap_invocation_failed.json"
    summary = json.loads(summary_path.read_text())
    started = json.loads(started_path.read_text())
    completed = json.loads(completed_path.read_text())
    errors = Counter()
    if failed_path.exists():
        errors["failed_marker_present"] += 1
    if (
        started.get("status") != "started"
        or completed.get("status") != "completed"
        or started.get("authorization_id") != expected_authorization
        or completed.get("authorization_id") != expected_authorization
        or summary.get("authorization_id") != expected_authorization
    ):
        errors["invocation_marker_contract"] += 1
    if (
        int(started.get("ap_invocation_count", -1)) != 1
        or int(completed.get("ap_invocation_count", -1)) != 1
        or int(summary.get("ap_invocation_count", -1)) != 1
        or int(started.get("planned_official_evaluator_call_count", -1)) != 2
        or int(completed.get("official_evaluator_call_count", -1)) != 2
        or int(summary.get("official_evaluator_call_count", -1)) != 2
    ):
        errors["unique_invocation_contract"] += 1
    if completed.get("summary_sha256") != _sha256(summary_path):
        errors["completed_summary_sha256"] += 1

    files = summary.get("files", {})
    hashes = summary.get("hashes", {})
    try:
        control_csv = args.result_root / str(files["control_csv"])
        challenge_csv = args.result_root / str(files["challenge_csv"])
        if _sha256(control_csv) != str(hashes["control_csv"]):
            errors["control_csv_sha256"] += 1
        if _sha256(challenge_csv) != str(hashes["challenge_csv"]):
            errors["challenge_csv_sha256"] += 1
        if _sha256(started_path) != str(hashes["started_marker"]):
            errors["started_marker_sha256"] += 1
        control, control_class_count = _csv_metrics(control_csv)
        challenge, challenge_class_count = _csv_metrics(challenge_csv)
    except (FileNotFoundError, KeyError, ValueError, OSError, csv.Error) as error:
        errors[f"csv_read::{type(error).__name__}"] += 1
        control = challenge = {name: float("nan") for name in METRICS}
        control_class_count = challenge_class_count = -1

    for system, observed in (("control", control), ("challenge", challenge)):
        for metric in METRICS:
            if not math.isclose(
                float(observed[metric]), float(summary.get(system, {}).get(metric, float("inf"))),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                errors[f"{system}_{metric}_mismatch"] += 1
    for metric in METRICS:
        expected = challenge[metric] - control[metric]
        if not math.isclose(
            expected, float(summary.get("delta", {}).get(metric, float("inf"))),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            errors[f"delta_{metric}_mismatch"] += 1
    if control_class_count != challenge_class_count or control_class_count <= 0:
        errors["csv_class_count_mismatch"] += 1

    external = {
        "cache_summary": args.cache_root / "summary.json",
        "cache_audit": args.cache_audit_root / "summary.json",
        "decision_ledger": args.decision_root / "safe_decisions.jsonl",
        "decision_summary": args.decision_root / "summary.json",
        "decision_audit": args.decision_root / "audit_summary.json",
        "preregistration": args.preregistration_path,
        "duplicate_safe_preregistration": args.duplicate_safe_preregistration_path,
    }
    if recovery_mode:
        external.update({
            "minus1_boundary_preregistration": args.minus1_boundary_preregistration_path,
            "prior_ap_started_marker": args.prior_failed_ap_root / "ap_invocation_started.json",
            "prior_ap_failed_marker": args.prior_failed_ap_root / "ap_invocation_failed.json",
            "prior_ap_log": args.prior_failed_ap_log,
        })
    recorded = summary.get("input_provenance", {})
    for name, path in external.items():
        if not path.is_file() or str(recorded.get(name, "")) != _sha256(path):
            errors[f"input_provenance::{name}"] += 1
    if str(summary.get("scene_list_sha256", "")) != _sha256(args.scene_list):
        errors["scene_list_sha256"] += 1
    decision_summary = json.loads((args.decision_root / "summary.json").read_text())
    cache_summary = json.loads((args.cache_root / "summary.json").read_text())
    boundary_input_audit = None
    if recovery_mode:
        try:
            scenes = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip()]
            decision_rows = [
                json.loads(line) for line in
                (args.decision_root / "safe_decisions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            validate_frozen_recovery_inputs(
                args.cache_root, args.cache_audit_root, args.decision_root
            )
            validate_prior_failure(args.prior_failed_ap_root, args.prior_failed_ap_log)
            boundary_input_audit = audit_boundary_inputs(scenes, args.cache_root, decision_rows)
        except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError):
            errors["minus1_boundary_input_audit"] += 1
        expected_boundary = {
            "candidate_count": EXPECTED_TOTAL_CANDIDATE_COUNT,
            "foreground_candidate_count": EXPECTED_FOREGROUND_COUNT,
            "native_background_198_count": EXPECTED_NATIVE_BACKGROUND_198_COUNT,
            "minus1_to_background_count": EXPECTED_MINUS1_COUNT,
            "evaluator_background_or_invalid_count": (
                EXPECTED_MINUS1_COUNT + EXPECTED_NATIVE_BACKGROUND_198_COUNT
            ),
            "candidate_deletion_count": 0,
            "cache_or_decision_write_count": 0,
        }
        for key, value in expected_boundary.items():
            if boundary_input_audit is None or boundary_input_audit.get(key) != value:
                errors[f"minus1_boundary::{key}"] += 1
        for key, value in {
            "minus1_evaluator_boundary_safe": True,
            "minus1_to_background_count_control": EXPECTED_MINUS1_COUNT,
            "minus1_to_background_count_challenge": EXPECTED_MINUS1_COUNT,
            "native_background_198_count_control": EXPECTED_NATIVE_BACKGROUND_198_COUNT,
            "native_background_198_count_challenge": EXPECTED_NATIVE_BACKGROUND_198_COUNT,
            "foreground_candidate_count_control": EXPECTED_FOREGROUND_COUNT,
            "foreground_candidate_count_challenge": EXPECTED_FOREGROUND_COUNT,
            "evaluator_background_or_invalid_count": (
                EXPECTED_MINUS1_COUNT + EXPECTED_NATIVE_BACKGROUND_198_COUNT
            ),
            "frozen_cache_or_decision_write_count": 0,
            "prior_failed_ap_preserved": True,
        }.items():
            if summary.get(key) != value:
                errors[f"minus1_summary::{key}"] += 1
        recorded_preflight = summary.get("minus1_evaluator_boundary_preflight", {})
        if boundary_input_audit is None or recorded_preflight != boundary_input_audit:
            errors["minus1_boundary_preflight_provenance"] += 1
        if summary.get("minus1_boundary_plan_indices") != sorted(
            EXPECTED_MINUS1_PLAN_INDICES
        ):
            errors["minus1_boundary_plan_indices"] += 1
    if (
        int(summary.get("scene_count", -1)) != args.expected_scene_count
        or int(cache_summary.get("scene_count", -1)) != args.expected_scene_count
        or int(summary.get("candidate_count", -1)) != int(cache_summary.get("candidate_count", -2))
        or int(summary.get("candidate_count", -1)) != int(decision_summary.get("candidate_count", -3))
        or int(summary.get("unique_geometry_count", -1)) != int(cache_summary.get("unique_geometry_count", -2))
        or int(summary.get("class_change_count", -1)) != int(decision_summary.get("class_change_count", -2))
    ):
        errors["count_contract"] += 1
    for key in ("candidate_mutation", "geometry_mutation", "score_mutation", "proposal_deletion"):
        if summary.get(key) is not False:
            errors[f"mutation::{key}"] += 1
    if (
        summary.get("single_fixed_challenge") is not True
        or int(summary.get("threshold_or_weight_scan_count", -1)) != 0
        or summary.get("ground_truth_usage") != "evaluation_only"
        or summary.get("ground_truth_read") is not True
        or summary.get("ap_computed") is not True
    ):
        errors["evaluation_contract"] += 1

    audit = {
        "version": VERSION,
        "audit_valid": sum(errors.values()) == 0,
        "error_count": int(sum(errors.values())),
        "errors": dict(sorted(errors.items())),
        "control": control,
        "challenge": challenge,
        "delta": {name: challenge[name] - control[name] for name in METRICS},
        "csv_class_count": control_class_count,
        "scene_count": int(summary.get("scene_count", -1)),
        "candidate_count": int(summary.get("candidate_count", -1)),
        "geometry_count": int(summary.get("candidate_count", -1)),
        "unique_geometry_count": int(summary.get("unique_geometry_count", -1)),
        "class_change_count": int(summary.get("class_change_count", -1)),
        "ap_invocation_count": 1,
        "official_evaluator_call_count": 2,
        "input_provenance": {
            "result_summary_sha256": _sha256(summary_path),
            "started_marker_sha256": _sha256(started_path),
            "completed_marker_sha256": _sha256(completed_path),
        },
    }
    if recovery_mode:
        audit.update({
            "minus1_evaluator_boundary_safe": True,
            "minus1_boundary_input_audit": boundary_input_audit,
            "prior_failed_ap_preserved": "minus1_boundary_input_audit" not in errors,
        })
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return audit
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--preregistration-path", type=Path, required=True)
    parser.add_argument(
        "--duplicate-safe-preregistration-path", type=Path,
        default=DUPLICATE_SAFE_PREREGISTRATION,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=312)
    parser.add_argument("--minus1-evaluator-boundary-safe", action="store_true")
    parser.add_argument(
        "--minus1-boundary-preregistration-path", type=Path,
        default=MINUS1_BOUNDARY_PREREGISTRATION,
    )
    parser.add_argument("--prior-failed-ap-root", type=Path)
    parser.add_argument("--prior-failed-ap-log", type=Path)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

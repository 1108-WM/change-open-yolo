"""Frozen evaluator-boundary contract for the 105 val312 minus-one candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


RECOVERY_AUTHORIZATION_ID = (
    "DM-SMS-1-FI1-D-v3-val312-minus1-evaluator-boundary-recovery-20260907"
)
EXPECTED_MINUS1_PLAN_INDICES = frozenset({
    220, 3496, 3533, 4358, 4910, 6486, 6493, 6594, 7755, 7842,
    7850, 7863, 8049, 8051, 8085, 8232, 8263, 9164, 9196, 9212,
    9237, 9248, 9257, 9313, 9326, 9343, 9374, 10620, 11409, 11472,
    11749, 12134, 12466, 12638, 12723, 13207, 13231, 13253, 13563,
    13712, 13879, 14042, 14136, 14154, 14290, 15699, 16147, 16720,
    16783, 16790, 17216, 17837, 18767, 18965, 18988, 19025, 20227,
    20228, 20238, 21601, 22071, 22402, 23146, 23957, 24076, 24301,
    24375, 24778, 25101, 25461, 25951, 26953, 27120, 27203, 27291,
    27327, 28119, 28144, 28216, 28388, 28807, 28867, 28951, 29361,
    29527, 29609, 30248, 31090, 31113, 31412, 31795, 32441, 33131,
    33477, 34560, 34881, 34959, 34991, 35189, 35616, 35638, 36867,
    36892, 36939, 36999,
})
EXPECTED_MINUS1_COUNT = 105
EXPECTED_NATIVE_BACKGROUND_198_COUNT = 31
EXPECTED_FOREGROUND_COUNT = 39168
EXPECTED_TOTAL_CANDIDATE_COUNT = 39304
EXPECTED_PRIOR_STARTED_SHA256 = "b808c187e90e6f491964bca595b7e66eaec0883181b270c1cd66c4e423102e41"
EXPECTED_PRIOR_FAILED_SHA256 = "8a7cfcd6e99bb23eea3a5e3e9c4ca2ed9c26c7383a4239bf4939803581faf2ea"
EXPECTED_PRIOR_LOG_SHA256 = "471143eec688b15324a95847e00004e22a8cfdd5bfc32d4135b99fcf719fa788"
EXPECTED_DECISION_LEDGER_SHA256 = "60b0077fadceeb2873f91e134c3786e99a04c8ba5de9b7acc3d219a67302858f"
EXPECTED_CACHE_SUMMARY_SHA256 = "8fa827f04fa151793c824545c644d29bc6f1ac92effb0f863da75bf5bd421a8f"
EXPECTED_CACHE_AUDIT_SHA256 = "2997be6b02655815af555cee1c562dd369e62a805a2019b5904ca4e904d86c29"


def minus1_identity(row: dict) -> tuple[int, str]:
    return int(row.get("plan_index", -999)), str(row.get("plan_key", ""))


def validate_minus1_decision(row: dict) -> tuple[int, str]:
    identity = minus1_identity(row)
    if (
        identity[0] not in EXPECTED_MINUS1_PLAN_INDICES
        or not identity[1]
        or int(row.get("canonical_frozen_class_index", -999)) != -1
        or int(row.get("arbitrated_class_index", -999)) != -1
        or row.get("class_changed") is not False
        or row.get("decision_path") != "single_candidate_deterministic_keep"
        or row.get("candidate_source") not in {"track", "pair_union"}
    ):
        raise ValueError(f"{identity}: invalid frozen minus-one evaluator-boundary decision")
    return identity


def frozen_minus1_identities(rows: list[dict]) -> frozenset[tuple[int, str]]:
    selected = [
        row for row in rows
        if int(row.get("canonical_frozen_class_index", -999)) == -1
        or int(row.get("arbitrated_class_index", -999)) == -1
    ]
    identities = [validate_minus1_decision(row) for row in selected]
    if (
        len(identities) != EXPECTED_MINUS1_COUNT
        or len(set(identities)) != EXPECTED_MINUS1_COUNT
        or {index for index, _ in identities} != EXPECTED_MINUS1_PLAN_INDICES
    ):
        raise ValueError("frozen minus-one evaluator-boundary identity set differs")
    return frozenset(identities)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(identities: set[tuple[int, str]] | frozenset[tuple[int, str]]) -> str:
    payload = "".join(f"{index}\t{plan_key}\n" for index, plan_key in sorted(identities))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_prior_failure(prior_root: Path, prior_log: Path) -> dict[str, str]:
    started = prior_root / "ap_invocation_started.json"
    failed = prior_root / "ap_invocation_failed.json"
    observed = {
        "prior_ap_started_marker": sha256(started),
        "prior_ap_failed_marker": sha256(failed),
        "prior_ap_log": sha256(prior_log),
    }
    expected = {
        "prior_ap_started_marker": EXPECTED_PRIOR_STARTED_SHA256,
        "prior_ap_failed_marker": EXPECTED_PRIOR_FAILED_SHA256,
        "prior_ap_log": EXPECTED_PRIOR_LOG_SHA256,
    }
    if observed != expected:
        raise ValueError("the first failed AP provenance differs from the frozen recovery contract")
    started_row = json.loads(started.read_text())
    failed_row = json.loads(failed.read_text())
    if (
        started_row.get("status") != "started"
        or int(started_row.get("ap_invocation_count", -1)) != 1
        or failed_row.get("status") != "failed_no_rerun_allowed"
        or int(failed_row.get("ap_invocation_count", -1)) != 1
    ):
        raise ValueError("the first failed AP markers do not retain the frozen failed state")
    return observed


def validate_frozen_recovery_inputs(
    cache_root: Path, cache_audit_root: Path, decision_root: Path,
) -> dict[str, str]:
    observed = {
        "decision_ledger": sha256(decision_root / "safe_decisions.jsonl"),
        "cache_summary": sha256(cache_root / "summary.json"),
        "cache_audit": sha256(cache_audit_root / "summary.json"),
    }
    expected = {
        "decision_ledger": EXPECTED_DECISION_LEDGER_SHA256,
        "cache_summary": EXPECTED_CACHE_SUMMARY_SHA256,
        "cache_audit": EXPECTED_CACHE_AUDIT_SHA256,
    }
    if observed != expected:
        raise ValueError("frozen no-GT inputs differ from the evaluator-boundary recovery contract")
    return observed


def audit_boundary_inputs(
    scenes: list[str], cache_root: Path, decision_rows: list[dict],
) -> dict:
    cache_summary = json.loads((cache_root / "summary.json").read_text())
    scene_summaries = {
        str(row.get("scene_name", "")): row for row in cache_summary.get("scene_summaries", [])
    }
    if (
        int(cache_summary.get("scene_count", -1)) != len(scenes)
        or int(cache_summary.get("candidate_count", -1)) != EXPECTED_TOTAL_CANDIDATE_COUNT
        or set(scene_summaries) != set(scenes)
    ):
        raise ValueError("evaluator-boundary cache summary coverage differs")
    decisions = {
        (str(row.get("scene_name", "")), str(row.get("plan_key", ""))): row
        for row in decision_rows
    }
    if len(decisions) != len(decision_rows):
        raise ValueError("duplicate complete decision identity")
    expected_minus1 = frozen_minus1_identities(decision_rows)
    observed_minus1: set[tuple[int, str]] = set()
    observed_keys: set[tuple[str, str]] = set()
    candidate_count = foreground_count = native_background_count = 0
    for scene in scenes:
        root = cache_root / "prediction_cache" / scene
        recorded_files = scene_summaries[scene].get("file_sha256", {})
        for name in (
            "masks.npy", "frozen_classes.npy", "frozen_scores.npy",
            "geometry_hashes.json", "plan_keys.json", "plan_indices.json", "sources.json",
        ):
            if sha256(root / name) != str(recorded_files.get(name, "")):
                raise ValueError(f"{scene}/{name}: frozen prediction cache hash differs")
        masks = np.load(root / "masks.npy", mmap_mode="r")
        classes = np.asarray(np.load(root / "frozen_classes.npy"), dtype=np.int64)
        scores = np.asarray(np.load(root / "frozen_scores.npy"), dtype=np.float32)
        hashes = json.loads((root / "geometry_hashes.json").read_text())
        plan_keys = json.loads((root / "plan_keys.json").read_text())
        plan_indices = json.loads((root / "plan_indices.json").read_text())
        sources = json.loads((root / "sources.json").read_text())
        if not (
            masks.ndim == 2
            and masks.shape[1] == len(classes) == len(scores) == len(hashes)
            == len(plan_keys) == len(plan_indices) == len(sources)
        ):
            raise ValueError(f"{scene}: evaluator-boundary cache dimensions differ")
        for frozen, geometry_hash, plan_key, plan_index, source in zip(
            classes, hashes, plan_keys, plan_indices, sources
        ):
            key = (scene, str(plan_key))
            if key in observed_keys:
                raise ValueError(f"{scene}/{plan_key}: duplicate evaluator-boundary cache identity")
            observed_keys.add(key)
            decision = decisions.get(key)
            if decision is None:
                raise ValueError(f"{scene}/{plan_key}: missing evaluator-boundary decision")
            selected = int(decision.get("arbitrated_class_index", -999))
            if (
                int(decision.get("canonical_frozen_class_index", -999)) != int(frozen)
                or str(decision.get("geometry_hash", "")) != str(geometry_hash)
                or int(decision.get("plan_index", -999)) != int(plan_index)
                or str(decision.get("candidate_source", "")) != str(source)
                or bool(decision.get("class_changed")) != (selected != int(frozen))
            ):
                raise ValueError(f"{scene}/{plan_key}: evaluator-boundary provenance differs")
            if int(frozen) == -1 or selected == -1:
                identity = validate_minus1_decision(decision)
                if identity != (int(plan_index), str(plan_key)):
                    raise ValueError(f"{scene}/{plan_key}: evaluator-boundary identity differs")
                observed_minus1.add(identity)
            elif not (0 <= int(frozen) <= 198 and 0 <= selected <= 198):
                raise ValueError(f"{scene}/{plan_key}: class index is outside evaluator contract")
            if int(frozen) == 198:
                native_background_count += 1
            elif 0 <= int(frozen) < 198:
                foreground_count += 1
            candidate_count += 1
    if observed_keys != set(decisions):
        raise ValueError("evaluator-boundary cache and decision coverage differ")
    if (
        candidate_count != EXPECTED_TOTAL_CANDIDATE_COUNT
        or foreground_count != EXPECTED_FOREGROUND_COUNT
        or native_background_count != EXPECTED_NATIVE_BACKGROUND_198_COUNT
        or observed_minus1 != set(expected_minus1)
    ):
        raise ValueError("evaluator-boundary aggregate counts differ")
    return {
        "candidate_count": candidate_count,
        "foreground_candidate_count": foreground_count,
        "native_background_198_count": native_background_count,
        "minus1_to_background_count": len(observed_minus1),
        "evaluator_background_or_invalid_count": len(observed_minus1) + native_background_count,
        "minus1_identity_sha256": identity_sha256(observed_minus1),
        "candidate_deletion_count": 0,
        "cache_or_decision_write_count": 0,
    }

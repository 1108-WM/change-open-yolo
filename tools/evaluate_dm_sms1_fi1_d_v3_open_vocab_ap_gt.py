#!/usr/bin/env python3
"""Run the single authorized FI1-D-v3 control/DM-SMS-1 open-vocabulary AP evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval  # noqa: E402
from tools.dm_sms1_minus1_evaluator_boundary import (  # noqa: E402
    EXPECTED_FOREGROUND_COUNT,
    EXPECTED_MINUS1_COUNT,
    EXPECTED_NATIVE_BACKGROUND_198_COUNT,
    EXPECTED_TOTAL_CANDIDATE_COUNT,
    RECOVERY_AUTHORIZATION_ID,
    audit_boundary_inputs,
    frozen_minus1_identities,
    validate_frozen_recovery_inputs,
    validate_prior_failure,
    validate_minus1_decision,
)


VERSION = "dm_sms1_fi1_d_v3_open_vocab_ap_v1"
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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_scenes(path: Path, expected_count: int) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != expected_count or len(scenes) != len(set(scenes)):
        raise ValueError(f"expected exactly {expected_count} unique scenes")
    return scenes


def _metrics(averages: dict) -> dict[str, float]:
    return {
        "ap": float(averages["all_ap"]),
        "ap50": float(averages["all_ap_50%"]),
        "ap25": float(averages["all_ap_25%"]),
        "head_ap": float(averages["head_ap"]),
        "common_ap": float(averages["common_ap"]),
        "tail_ap": float(averages["tail_ap"]),
    }


class FrozenPredictionMapping(Mapping):
    """Load one frozen cache and expose either control or arbitrated classes."""

    def __init__(
        self,
        scenes: list[str],
        cache_root: Path,
        decisions: dict[tuple[str, str], dict],
        challenge: bool,
        minus1_boundary_identities: frozenset[tuple[int, str]] | None = None,
    ) -> None:
        self.scenes = scenes
        self.cache_root = cache_root
        self.decisions = decisions
        self.challenge = challenge
        self.minus1_boundary_identities = minus1_boundary_identities
        self.observed_geometry_count = 0
        self.observed_class_change_count = 0
        self.observed_minus1_boundary_identities: set[tuple[int, str]] = set()
        self.observed_native_background_198_count = 0
        self.observed_foreground_count = 0

    def __len__(self) -> int:
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, scene: str) -> dict[str, np.ndarray]:
        if scene not in self.scenes:
            raise KeyError(scene)
        return self._prediction(scene)

    def items(self):
        self.observed_geometry_count = 0
        self.observed_class_change_count = 0
        self.observed_minus1_boundary_identities.clear()
        self.observed_native_background_198_count = 0
        self.observed_foreground_count = 0
        for index, scene in enumerate(self.scenes, 1):
            prediction = self._prediction(scene)
            print(
                f"[DM-SMS-1 FI1-D-v3 AP] {index}/{len(self.scenes)} "
                f"{'challenge' if self.challenge else 'control'} {scene}",
                flush=True,
            )
            yield scene, prediction

    def _prediction(self, scene: str) -> dict[str, np.ndarray]:
        root = self.cache_root / "prediction_cache" / scene
        masks = np.load(root / "masks.npy", mmap_mode="r")
        classes = np.asarray(np.load(root / "frozen_classes.npy"), dtype=np.int64).copy()
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
            raise ValueError(f"{scene}: frozen prediction cache dimensions disagree")
        for column, (geometry_hash, plan_key, plan_index, source) in enumerate(
            zip(hashes, plan_keys, plan_indices, sources)
        ):
            decision = self.decisions.get((scene, str(plan_key)))
            if decision is None:
                raise ValueError(f"{scene}/{plan_key}: decision is missing")
            if str(decision.get("geometry_hash", "")) != str(geometry_hash):
                raise ValueError(f"{scene}/{plan_key}: visual geometry provenance differs")
            frozen = int(classes[column])
            if int(decision.get("canonical_frozen_class_index", -999)) != frozen:
                raise ValueError(f"{scene}/{plan_key}: frozen class differs from decision ledger")
            selected = int(decision.get("arbitrated_class_index", -999))
            if frozen == -1 or selected == -1:
                identity = (int(plan_index), str(plan_key))
                if self.minus1_boundary_identities is None:
                    raise ValueError(f"{scene}/{plan_key}: class index is outside evaluator contract")
                if identity not in self.minus1_boundary_identities:
                    raise ValueError(f"{scene}/{plan_key}: unexpected minus-one evaluator-boundary identity")
                if (
                    validate_minus1_decision(decision) != identity
                    or str(decision.get("candidate_source")) != str(source)
                ):
                    raise ValueError(f"{scene}/{plan_key}: minus-one evaluator-boundary provenance differs")
                classes[column] = 198
                self.observed_minus1_boundary_identities.add(identity)
            elif selected < 0 or selected > 198 or frozen < 0 or frozen > 198:
                raise ValueError(f"{scene}/{plan_key}: class index is outside evaluator contract")
            elif self.challenge:
                classes[column] = selected
                self.observed_class_change_count += int(selected != frozen)
            if frozen == 198:
                self.observed_native_background_198_count += 1
            elif 0 <= frozen < 198:
                self.observed_foreground_count += 1
        self.observed_geometry_count += len(hashes)
        return {"pred_masks": masks, "pred_classes": classes, "pred_scores": scores}


def _evaluate(mapping: FrozenPredictionMapping, gt_root: Path, csv_path: Path) -> dict[str, float]:
    averages, _, _, _ = instance_eval.evaluate(
        mapping, str(gt_root), str(csv_path), dataset="scannet200"
    )
    return _metrics(averages)


def run(args: argparse.Namespace) -> dict:
    if not args.allow_gt_evaluation:
        raise PermissionError("pass --allow-gt-evaluation for the one authorized GT-reading step")
    recovery_mode = bool(getattr(args, "minus1_evaluator_boundary_safe", False))
    expected_authorization = RECOVERY_AUTHORIZATION_ID if recovery_mode else AUTHORIZATION_ID
    if args.authorization_id != expected_authorization:
        raise PermissionError("the explicit frozen authorization identifier does not match")
    if getattr(args, "duplicate_safe_preregistration_path", None) is None:
        args.duplicate_safe_preregistration_path = DUPLICATE_SAFE_PREREGISTRATION
    if recovery_mode and getattr(args, "minus1_boundary_preregistration_path", None) is None:
        args.minus1_boundary_preregistration_path = MINUS1_BOUNDARY_PREREGISTRATION
    for name in (
        "scene_list", "ground_truth_root", "cache_root", "cache_audit_root",
        "decision_root", "preregistration_path", "duplicate_safe_preregistration_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if recovery_mode:
        for name in ("minus1_boundary_preregistration_path", "prior_failed_ap_root", "prior_failed_ap_log"):
            setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list, args.expected_scene_count)
    if args.output_root.exists():
        raise FileExistsError(
            f"AP output root already exists; a started, failed, or completed run may not be rerun: {args.output_root}"
        )
    required = {
        "cache_summary": args.cache_root / "summary.json",
        "cache_audit": args.cache_audit_root / "summary.json",
        "decision_ledger": args.decision_root / "safe_decisions.jsonl",
        "decision_summary": args.decision_root / "summary.json",
        "decision_audit": args.decision_root / "audit_summary.json",
        "preregistration": args.preregistration_path,
        "duplicate_safe_preregistration": args.duplicate_safe_preregistration_path,
    }
    if recovery_mode:
        required.update({
            "minus1_boundary_preregistration": args.minus1_boundary_preregistration_path,
            "prior_ap_started_marker": args.prior_failed_ap_root / "ap_invocation_started.json",
            "prior_ap_failed_marker": args.prior_failed_ap_root / "ap_invocation_failed.json",
            "prior_ap_log": args.prior_failed_ap_log,
        })
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if not args.ground_truth_root.is_dir():
        missing.append(f"ground_truth_root: {args.ground_truth_root}")
    if missing:
        raise FileNotFoundError("required AP inputs are missing: " + "; ".join(missing))

    cache_summary = json.loads(required["cache_summary"].read_text())
    cache_audit = json.loads(required["cache_audit"].read_text())
    decision_summary = json.loads(required["decision_summary"].read_text())
    decision_audit = json.loads(required["decision_audit"].read_text())
    if (
        int(cache_summary.get("scene_count", -1)) != len(scenes)
        or cache_summary.get("cache_valid") is not True
        or cache_summary.get("ground_truth_read") is not False
        or cache_summary.get("ap_computed") is not False
        or cache_audit.get("audit_valid") is not True
        or int(cache_audit.get("error_count", -1)) != 0
        or decision_audit.get("audit_valid") is not True
        or int(decision_audit.get("error_count", -1)) != 0
        or decision_summary.get("ground_truth_read") is not False
        or decision_summary.get("ap_computed") is not False
    ):
        raise ValueError("prediction cache or complete decision ledger is not fully audited")

    decision_rows = _read_jsonl(required["decision_ledger"])
    decision_keys = [(str(row.get("scene_name", "")), str(row.get("plan_key", ""))) for row in decision_rows]
    if any(not scene or not plan_key for scene, plan_key in decision_keys) or len(decision_keys) != len(set(decision_keys)):
        raise ValueError("complete decision ledger has empty or duplicate plan identity")
    decisions = dict(zip(decision_keys, decision_rows))
    if {scene for scene, _ in decision_keys} != set(scenes):
        raise ValueError("complete decision ledger scene coverage differs")
    geometry_count = int(cache_summary.get("candidate_count", -1))
    minus1_identities = frozen_minus1_identities(decision_rows) if recovery_mode else None
    boundary_preflight = None
    prior_failure_provenance = None
    frozen_recovery_provenance = None
    if recovery_mode:
        frozen_recovery_provenance = validate_frozen_recovery_inputs(
            args.cache_root, args.cache_audit_root, args.decision_root
        )
        prior_failure_provenance = validate_prior_failure(
            args.prior_failed_ap_root, args.prior_failed_ap_log
        )
        boundary_preflight = audit_boundary_inputs(scenes, args.cache_root, decision_rows)
    class_change_count = sum(bool(row.get("class_changed")) for row in decision_rows)
    if (
        len(decision_rows) != geometry_count
        or int(decision_summary.get("candidate_count", -1)) != geometry_count
        or int(decision_summary.get("class_change_count", -1)) != class_change_count
    ):
        raise ValueError("cache and decision ledger counts differ")

    args.output_root.mkdir(parents=True, exist_ok=False)
    started_path = args.output_root / "ap_invocation_started.json"
    started = {
        "version": "dm_sms1_fi1_d_v3_ap_invocation_marker_v1",
        "status": "started",
        "authorization_id": expected_authorization,
        "process_id": os.getpid(),
        "ap_invocation_count": 1,
        "planned_official_evaluator_call_count": 2,
        "input_provenance": {name: _sha256(path) for name, path in required.items()},
        "scene_list_sha256": _sha256(args.scene_list),
    }
    if recovery_mode:
        started["minus1_evaluator_boundary_preflight"] = boundary_preflight
        started["prior_failed_ap_provenance"] = prior_failure_provenance
        started["frozen_recovery_provenance"] = frozen_recovery_provenance
    started_path.write_text(json.dumps(started, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    try:
        control_csv = args.output_root / "fi1_d_v3_frozen_control.csv"
        challenge_csv = args.output_root / "fi1_d_v3_plus_dm_sms1.csv"
        control_mapping = FrozenPredictionMapping(
            scenes, args.cache_root, decisions, challenge=False,
            minus1_boundary_identities=minus1_identities,
        )
        control = _evaluate(control_mapping, args.ground_truth_root, control_csv)
        gc.collect()
        challenge_mapping = FrozenPredictionMapping(
            scenes, args.cache_root, decisions, challenge=True,
            minus1_boundary_identities=minus1_identities,
        )
        challenge = _evaluate(challenge_mapping, args.ground_truth_root, challenge_csv)
        if control_mapping.observed_geometry_count != geometry_count:
            raise ValueError("control evaluator did not consume every frozen geometry")
        if challenge_mapping.observed_geometry_count != geometry_count:
            raise ValueError("challenge evaluator did not consume every frozen geometry")
        if challenge_mapping.observed_class_change_count != class_change_count:
            raise ValueError("challenge evaluator class-change count differs from decision ledger")
        if recovery_mode and (
            control_mapping.observed_minus1_boundary_identities != set(minus1_identities)
            or challenge_mapping.observed_minus1_boundary_identities != set(minus1_identities)
            or control_mapping.observed_minus1_boundary_identities
            != challenge_mapping.observed_minus1_boundary_identities
            or control_mapping.observed_native_background_198_count
            != EXPECTED_NATIVE_BACKGROUND_198_COUNT
            or challenge_mapping.observed_native_background_198_count
            != EXPECTED_NATIVE_BACKGROUND_198_COUNT
            or control_mapping.observed_foreground_count != EXPECTED_FOREGROUND_COUNT
            or challenge_mapping.observed_foreground_count != EXPECTED_FOREGROUND_COUNT
            or geometry_count != EXPECTED_TOTAL_CANDIDATE_COUNT
        ):
            raise ValueError("minus-one evaluator-boundary aggregate contract differs")
        delta = {name: challenge[name] - control[name] for name in METRICS}
        summary = {
            "version": VERSION,
            "dataset_name": args.dataset_name,
            "evaluation_scope": "official ScanNet200 open-vocabulary instance AP",
            "scene_count": len(scenes),
            "geometry_count": geometry_count,
            "candidate_count": geometry_count,
            "unique_geometry_count": int(cache_summary.get("unique_geometry_count", -1)),
            "two_candidate_count": int(decision_summary.get("two_candidate_count", -1)),
            "single_candidate_count": int(decision_summary.get("single_candidate_count", -1)),
            "model_evidence_valid_count": int(decision_summary.get("model_evidence_valid_count", -1)),
            "invalid_evidence_fallback_count": int(decision_summary.get("invalid_evidence_fallback_count", -1)),
            "class_change_count": class_change_count,
            "control": control,
            "challenge": challenge,
            "delta": delta,
            "single_fixed_challenge": True,
            "threshold_or_weight_scan_count": 0,
            "ground_truth_usage": "evaluation_only",
            "ground_truth_read": True,
            "ap_computed": True,
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation": False,
            "proposal_deletion": False,
            "only_allowed_difference": "predicted class index from the audited complete decision ledger",
            "ap_invocation_count": 1,
            "official_evaluator_call_count": 2,
            "authorization_id": expected_authorization,
            "files": {
                "control_csv": control_csv.name,
                "challenge_csv": challenge_csv.name,
                "started_marker": started_path.name,
            },
            "hashes": {
                "control_csv": _sha256(control_csv),
                "challenge_csv": _sha256(challenge_csv),
                "started_marker": _sha256(started_path),
            },
            "input_provenance": started["input_provenance"],
            "scene_list_sha256": started["scene_list_sha256"],
        }
        summary["authorization_id"] = expected_authorization
        if recovery_mode:
            summary.update({
                "minus1_evaluator_boundary_safe": True,
                "minus1_to_background_count_control": len(control_mapping.observed_minus1_boundary_identities),
                "minus1_to_background_count_challenge": len(challenge_mapping.observed_minus1_boundary_identities),
                "native_background_198_count_control": control_mapping.observed_native_background_198_count,
                "native_background_198_count_challenge": challenge_mapping.observed_native_background_198_count,
                "foreground_candidate_count_control": control_mapping.observed_foreground_count,
                "foreground_candidate_count_challenge": challenge_mapping.observed_foreground_count,
                "evaluator_background_or_invalid_count": (
                    EXPECTED_MINUS1_COUNT + EXPECTED_NATIVE_BACKGROUND_198_COUNT
                ),
                "minus1_boundary_plan_indices": sorted(index for index, _ in minus1_identities),
                "frozen_cache_or_decision_write_count": 0,
                "prior_failed_ap_preserved": True,
                "minus1_evaluator_boundary_preflight": boundary_preflight,
                "prior_failed_ap_provenance": prior_failure_provenance,
                "frozen_recovery_provenance": frozen_recovery_provenance,
            })
        summary_path = args.output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        completed = {
            "version": "dm_sms1_fi1_d_v3_ap_invocation_marker_v1",
            "status": "completed",
            "authorization_id": expected_authorization,
            "ap_invocation_count": 1,
            "official_evaluator_call_count": 2,
            "summary_sha256": _sha256(summary_path),
        }
        (args.output_root / "ap_invocation_completed.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return summary
    except Exception as error:
        failed = {
            "version": "dm_sms1_fi1_d_v3_ap_invocation_marker_v1",
            "status": "failed_no_rerun_allowed",
            "authorization_id": expected_authorization,
            "ap_invocation_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (args.output_root / "ap_invocation_failed.json").write_text(
            json.dumps(failed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
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
    parser.add_argument("--dataset-name", default="ScanNet200-val312")
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--minus1-evaluator-boundary-safe", action="store_true")
    parser.add_argument(
        "--minus1-boundary-preregistration-path", type=Path,
        default=MINUS1_BOUNDARY_PREREGISTRATION,
    )
    parser.add_argument("--prior-failed-ap-root", type=Path)
    parser.add_argument("--prior-failed-ap-log", type=Path)
    parser.add_argument("--allow-gt-evaluation", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({
        "control": result["control"], "challenge": result["challenge"],
        "delta": result["delta"], "class_change_count": result["class_change_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

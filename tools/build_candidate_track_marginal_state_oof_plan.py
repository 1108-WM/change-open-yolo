#!/usr/bin/env python3
"""Export the frozen no-GT OOF score plan for marginal AP-harm state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import TRACK_SOURCE, read_scene_list  # noqa: E402
from tools.evaluate_official100_geometry_group_ranking_oof_ap import (  # noqa: E402
    EXPECTED_OOF_SHA256,
)
from tools.fit_candidate_track_marginal_state_head_full import STATE_GATE_KEYS  # noqa: E402


VERSION = "official100_track_marginal_state_oof_plan_v1"
POLICY = "track_marginal_state_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suppressed_track_score(original_score: float, state_probability: float) -> float:
    original = float(original_score)
    probability = float(state_probability)
    if not 0.0 <= original <= 1.0 or not 0.0 <= probability <= 1.0:
        raise ValueError("score and state probability must lie in [0, 1]")
    return original * (1.0 - (1.0 - probability) ** 3)


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("OOF plan requires official100")
    diagnostic = json.loads(args.oof_diagnostic_summary.read_text())
    failed = [key for key in STATE_GATE_KEYS if not diagnostic["gates"].get(key)]
    if failed:
        raise ValueError(f"pure state OOF gates failed: {failed}")
    if _sha256(args.quality_oof_predictions) != EXPECTED_OOF_SHA256:
        raise ValueError("candidate-quality OOF prediction SHA-256 mismatch")

    quality = {}
    for line in args.quality_oof_predictions.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["candidate_source"] != TRACK_SOURCE:
            continue
        key = (str(row["scene_name"]), int(row["candidate_id"]))
        quality[key] = float(row["predictions"]["D_plus_gvc"]["q"])
    prediction_rows = [
        json.loads(line) for line in args.marginal_oof_predictions.read_text().splitlines()
        if line.strip()
    ]
    plan_rows = []
    seen = set()
    for row in prediction_rows:
        key = (str(row["scene_name"]), int(row["track_id"]))
        if key in seen:
            raise ValueError(f"duplicate marginal OOF prediction: {key}")
        if key not in quality:
            raise ValueError(f"track quality score missing: {key}")
        seen.add(key)
        original = quality[key]
        probability = float(row["state_probability"])
        new_score = suppressed_track_score(original, probability)
        if new_score > original + 1e-12:
            raise AssertionError("frozen suppression increased a track score")
        plan_rows.append({
            "scene_name": key[0],
            "candidate_source": TRACK_SOURCE,
            "candidate_id": key[1],
            "policy": POLICY,
            "original_score": original,
            "state_probability": probability,
            "new_score": new_score,
            "score_multiplier": 1.0 - (1.0 - probability) ** 3,
            "candidate_retained": True,
        })
    if len(plan_rows) != int(diagnostic["track_count"]):
        raise ValueError("OOF plan coverage differs from marginal diagnostic")
    observed_scenes = {row["scene_name"] for row in plan_rows}
    if observed_scenes != set(scenes):
        raise ValueError("OOF plan scene coverage differs from official100")

    per_scene = Counter(row["scene_name"] for row in plan_rows)
    summary = {
        "version": VERSION,
        "policy": POLICY,
        "scene_count": len(scenes),
        "controlled_track_count": len(plan_rows),
        "score_contract": {
            "native_scores": "bit-for-bit unchanged; omitted from delta plan",
            "uncontrolled_track_scores": "unchanged; omitted from delta plan",
            "controlled_track_score": "OOF_quality_q * (1 - (1 - OOF_P(demotion_harms_AP))**3)",
            "candidate_removed": False,
            "threshold": None,
            "weight_or_exponent_scan": False,
        },
        "contracts": {
            "candidate_count_modified": False,
            "candidate_geometry_modified": False,
            "candidate_class_modified": False,
            "native_mask_score_class_modified": False,
            "ground_truth_fields_written_to_plan": False,
        },
        "ground_truth_usage_for_plan_generation": "none; only OOF inference scores are consumed",
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ap_evaluated": False,
        "scene_controlled_track_counts": dict(sorted(per_scene.items())),
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "quality_oof_predictions_sha256": _sha256(args.quality_oof_predictions),
            "marginal_oof_predictions_sha256": _sha256(args.marginal_oof_predictions),
            "marginal_oof_summary_sha256": _sha256(args.oof_diagnostic_summary),
            "full_model_metadata_sha256": _sha256(args.full_model_metadata),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "track_score_plan.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sorted(plan_rows, key=lambda row: (
                row["scene_name"], row["candidate_id"]
            ))
        ))
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--quality-oof-predictions", type=Path, required=True)
    parser.add_argument("--marginal-oof-predictions", type=Path, required=True)
    parser.add_argument("--oof-diagnostic-summary", type=Path, required=True)
    parser.add_argument("--full-model-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "quality_oof_predictions", "marginal_oof_predictions",
        "oof_diagnostic_summary", "full_model_metadata", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "version": summary["version"],
        "policy": summary["policy"],
        "scene_count": summary["scene_count"],
        "controlled_track_count": summary["controlled_track_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

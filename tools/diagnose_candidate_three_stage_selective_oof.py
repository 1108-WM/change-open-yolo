#!/usr/bin/env python3
"""Audit the frozen three-stage selective replacement chain.

The diagnostic joins strict outer-fold predictions for relation reliability,
target consistency, and continuous relative margin.  It reports the fixed
probability product and the preregistered conformal lower-bound precondition.
If the precondition yields no candidates, risk-threshold calibration is
explicitly skipped and the system abstains.  No action is emitted and AP is
never run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.diagnose_candidate_two_stage_cascade_oof import _score_metrics  # noqa: E402
from tools.diagnose_train_candidate_relation_features import (  # noqa: E402
    _load_partition,
    _load_rows,
)
from tools.train_candidate_relative_margin_oof import IOU_MARGIN  # noqa: E402


RELIABILITY_MODEL = "R2_plus_relation"
TARGET_FIELD = "target_same_probability"
MARGIN_PROBABILITY_FIELD = "probability_margin_gt_005"
MARGIN_LOWER_FIELD = "lower_conformal"
RISK_ALPHA = 0.10
RISK_DELTA = 0.05
RISK_THRESHOLDS = tuple([index / 20 for index in range(1, 20)] + [0.97, 0.98, 0.99, 0.995, 0.999])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict) -> tuple[str, int, str]:
    return (
        str(row["scene_name"]), int(row["track_id"]),
        str(row["native_exact_geometry_group_id"]),
    )


def _lookup(path: Path) -> dict[tuple[str, int, str], dict]:
    result = {}
    for row in read_jsonl(path):
        key = _identity(row)
        if key in result:
            raise ValueError(f"duplicate selective OOF identity: {key}")
        result[key] = row
    return result


def clopper_pearson_upper(errors: int, total: int, delta: float) -> float | None:
    if total <= 0:
        return None
    if not 0 < delta < 1 or errors < 0 or errors > total:
        raise ValueError("invalid one-sided binomial-bound arguments")
    if errors == total:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, total - errors))


def learn_then_test_threshold(
    labels: np.ndarray, scores: np.ndarray, eligible: np.ndarray,
    thresholds: tuple[float, ...] = RISK_THRESHOLDS,
    alpha: float = RISK_ALPHA, delta: float = RISK_DELTA,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if labels.shape != scores.shape or labels.shape != eligible.shape:
        raise ValueError("risk-control arrays must have the same shape")
    if not np.any(eligible):
        return {
            "status": "not_run_no_conformal_candidates",
            "selected_threshold": None,
            "selected_count": 0,
            "candidate_results": [],
        }
    corrected_delta = delta / len(thresholds)
    results = []
    for threshold in thresholds:
        selected = eligible & (scores >= threshold)
        total = int(selected.sum())
        errors = int(np.sum(1 - labels[selected])) if total else 0
        upper = clopper_pearson_upper(errors, total, corrected_delta)
        results.append({
            "threshold": float(threshold),
            "selected_count": total,
            "error_count": errors,
            "observed_error_rate": errors / total if total else None,
            "bonferroni_error_upper": upper,
            "passes": bool(total and upper is not None and upper <= alpha),
        })
    passing = [row for row in results if row["passes"]]
    chosen = max(passing, key=lambda row: (row["selected_count"], -row["threshold"])) if passing else None
    return {
        "status": "passed" if chosen else "abstain_no_threshold_passed",
        "selected_threshold": chosen["threshold"] if chosen else None,
        "selected_count": chosen["selected_count"] if chosen else 0,
        "familywise_delta": delta,
        "per_threshold_delta": corrected_delta,
        "target_error_rate": alpha,
        "candidate_results": results,
    }


def join_predictions(
    rows: list[dict], reliability_lookup: dict, target_lookup: dict, margin_lookup: dict,
    scene_to_fold: dict[str, int],
) -> list[dict]:
    expected = {_identity(row) for row in rows}
    for name, lookup in (
        ("reliability", reliability_lookup), ("target", target_lookup), ("margin", margin_lookup)
    ):
        if set(lookup) != expected:
            raise ValueError(f"{name} OOF identities differ from the relation ledger")
    result = []
    for row in rows:
        key = _identity(row)
        reliability = reliability_lookup[key]
        target = target_lookup[key]
        margin = margin_lookup[key]
        fold = int(scene_to_fold[row["scene_name"]])
        if any(int(item["fold_index"]) != fold for item in (reliability, target, margin)):
            raise ValueError(f"OOF fold mismatch for relation: {key}")
        p_reliable = float(reliability["predictions"][RELIABILITY_MODEL])
        p_same = float(target[TARGET_FIELD])
        p_margin = float(margin["predictions"][MARGIN_PROBABILITY_FIELD])
        lower = float(margin["predictions"][MARGIN_LOWER_FIELD])
        probabilities = np.asarray([p_reliable, p_same, p_margin], dtype=np.float64)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"invalid three-stage probabilities: {key}")
        result.append({
            "scene_name": row["scene_name"],
            "fold_index": fold,
            "track_id": int(row["track_id"]),
            "native_exact_geometry_group_id": row["native_exact_geometry_group_id"],
            "labels": row["labels"],
            "label_safe_replacement": int(
                row["labels"]["relative_quality_state"] == "prefer_track"
            ),
            "reliability_probability": p_reliable,
            "target_same_probability": p_same,
            "margin_gt_005_probability": p_margin,
            "three_stage_product": p_reliable * p_same * p_margin,
            "conformal_lower_margin": lower,
            "passes_conformal_margin": bool(lower > IOU_MARGIN),
        })
    return result


def _audit(rows: list[dict], cohorts: dict[str, set[str]]) -> tuple[dict, list[dict]]:
    labels = np.asarray([row["label_safe_replacement"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["three_stage_product"] for row in rows], dtype=np.float64)
    eligible = np.asarray([row["passes_conformal_margin"] for row in rows], dtype=bool)
    overall = _score_metrics(rows, labels, scores)
    preselected = np.flatnonzero(eligible)
    preselection = {
        "selected_count": len(preselected),
        "correct_count": int(labels[preselected].sum()),
        "precision": float(labels[preselected].mean()) if len(preselected) else None,
        "selected_scene_count": len({rows[index]["scene_name"] for index in preselected}),
        "label_state_counts": dict(sorted(Counter(
            rows[index]["labels"]["relative_quality_state"] for index in preselected
        ).items())),
    }
    risk = learn_then_test_threshold(labels, scores, eligible)
    fold_metrics = []
    for fold_index in range(5):
        selected = [row for row in rows if row["fold_index"] == fold_index]
        fold_labels = np.asarray([row["label_safe_replacement"] for row in selected], dtype=np.int64)
        fold_scores = np.asarray([row["three_stage_product"] for row in selected], dtype=np.float64)
        fold_metrics.append({
            "fold_index": fold_index,
            "relation_count": len(selected),
            "prefer_track_count": int(fold_labels.sum()),
            "score_metrics": _score_metrics(selected, fold_labels, fold_scores),
            "conformal_preselection_count": int(sum(
                row["passes_conformal_margin"] for row in selected
            )),
        })
    cohort_metrics = {}
    for cohort_name, scenes in cohorts.items():
        selected = [row for row in rows if row["scene_name"] in scenes]
        cohort_labels = np.asarray([row["label_safe_replacement"] for row in selected], dtype=np.int64)
        cohort_scores = np.asarray([row["three_stage_product"] for row in selected], dtype=np.float64)
        cohort_metrics[cohort_name] = {
            "scene_count": len(scenes),
            "relation_count": len(selected),
            "prefer_track_count": int(cohort_labels.sum()),
            "score_metrics": _score_metrics(selected, cohort_labels, cohort_scores),
            "conformal_preselection_count": int(sum(
                row["passes_conformal_margin"] for row in selected
            )),
        }
    return {
        "relation_count": len(rows),
        "prefer_track_count": int(labels.sum()),
        "three_stage_score_metrics": overall,
        "conformal_margin_preselection": preselection,
        "risk_control": risk,
        "cohort_metrics": cohort_metrics,
    }, fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-ledger-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--existing20-scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--reliability-oof-predictions", type=Path, required=True)
    parser.add_argument("--target-all-relation-predictions", type=Path, required=True)
    parser.add_argument("--margin-oof-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    split_sha = _sha256(args.split_manifest)
    if split_sha != args.expected_split_sha256.lower():
        raise ValueError("frozen split manifest SHA-256 mismatch")
    scenes = read_scene_list(args.scene_list)
    scene_to_fold, cohorts, split_payload = _load_partition(
        args.split_manifest, scenes, args.existing20_scene_list
    )
    rows = _load_rows(args.feature_ledger_root, scenes)
    joined = join_predictions(
        rows,
        _lookup(args.reliability_oof_predictions),
        _lookup(args.target_all_relation_predictions),
        _lookup(args.margin_oof_predictions),
        scene_to_fold,
    )
    summary_core, fold_metrics = _audit(joined, cohorts)
    summary = {
        "version": f"{args.protocol_name}_candidate_three_stage_selective_oof_audit_v1",
        "protocol_name": args.protocol_name,
        "scene_count": len(scene_to_fold),
        **summary_core,
        "selection_contract": {
            "score_formula": "P(reliable pair) * P(same target) * P(IoU margin > 0.05)",
            "conformal_precondition": "scene-block conformal lower IoU margin > 0.05",
            "risk_target_error_rate": RISK_ALPHA,
            "risk_familywise_delta": RISK_DELTA,
            "risk_thresholds": list(RISK_THRESHOLDS),
            "no_conformal_candidate_means_abstain_all": True,
        },
        "safety_contract": {
            "threshold_selected": False,
            "fit_all_model_exported": False,
            "replacement_action_generated": False,
            "candidate_geometry_modified": False,
            "ap_evaluation_run": False,
        },
        "input_provenance": {
            "feature_ledger_summary_sha256": _sha256(args.feature_ledger_root / "summary.json"),
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": split_sha,
            "split_existing_scene_count": split_payload.get("existing_scene_count"),
            "split_new_scene_count": split_payload.get("new_scene_count"),
            "reliability_oof_sha256": _sha256(args.reliability_oof_predictions),
            "target_all_relation_predictions_sha256": _sha256(args.target_all_relation_predictions),
            "margin_oof_sha256": _sha256(args.margin_oof_predictions),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "fold_metrics.json").write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (staging / "scored_relations.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in joined
        ))
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

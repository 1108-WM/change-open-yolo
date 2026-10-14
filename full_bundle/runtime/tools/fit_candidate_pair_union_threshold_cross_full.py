#!/usr/bin/env python3
"""Fit a full official100 pair-proposal threshold-cross classifier."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import read_jsonl  # noqa: E402
from tools.diagnose_candidate_pair_union_threshold_cross_oof import (  # noqa: E402
    RANDOM_SEED,
    VERSIONS as OOF_VERSIONS,
    _balanced_weights,
    _component_weights,
    _matrix,
)
from tools.train_candidate_component_list_calibration_head_oof import MODEL_PARAMS  # noqa: E402


VERSIONS = {
    "pair_union": "official100_pair_union_threshold_cross_full_v2_prior_corrected",
    "pair_intersection": "official100_pair_intersection_threshold_cross_full_v1",
}
VERSION = VERSIONS["pair_union"]
REQUIRED_GATES = (
    "at_least_two_positive_proposals_in_every_validation_fold",
    "overall_pr_not_lower", "overall_roc_not_lower",
    "overall_brier_not_higher", "overall_log_loss_not_higher",
    "pr_non_lower_in_at_least_three_folds", "roc_non_lower_in_at_least_three_folds",
    "brier_non_higher_in_at_least_three_folds",
    "log_loss_non_higher_in_at_least_three_folds",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run(args: argparse.Namespace) -> dict:
    diagnostic = json.loads(args.oof_diagnostic_summary.read_text())
    if diagnostic.get("version") != OOF_VERSIONS[args.proposal_kind]:
        raise ValueError(f"unexpected {args.proposal_kind} OOF diagnostic version")
    gate_results = dict(diagnostic["gates"])
    gate_results.setdefault(
        "at_least_two_positive_proposals_in_every_validation_fold",
        min(int(fold["validation_positive_count"]) for fold in diagnostic["fold_details"]) >= 2,
    )
    failed = [key for key in REQUIRED_GATES if not gate_results.get(key)]
    if failed:
        raise ValueError(f"{args.proposal_kind} OOF gates failed: {failed}")
    rows = [
        row for row in read_jsonl(
            args.utility_ledger_root / f"{args.proposal_kind}_utilities.jsonl"
        )
        if row["model_features"]["proposal__geometry_novel_vs_existing"]
        and row["model_features"]["proposal__accepted_min_region"]
    ]
    matrix, feature_names = _matrix(rows)
    labels = np.asarray([
        int(row["labels"]["crosses_any_official_threshold"]) for row in rows
    ], dtype=np.int64)
    indexes = np.arange(len(rows), dtype=np.int64)
    natural_weights = _component_weights(rows, indexes)
    natural_positive_rate = float(np.average(
        labels, weights=natural_weights[indexes]
    ))
    weights = _balanced_weights(rows, indexes, labels)
    if args.reuse_model_package is not None:
        with args.reuse_model_package.open("rb") as handle:
            reused = pickle.load(handle)
        model = reused["model"]
        if list(reused.get("metadata", {}).get("feature_names", ())) != list(feature_names):
            raise ValueError("reused model feature contract does not match utility ledger")
    else:
        model = HistGradientBoostingClassifier(
            loss="log_loss", **{**MODEL_PARAMS, "random_state": RANDOM_SEED + 5000}
        )
        model.fit(matrix, labels, sample_weight=weights)
    metadata = {
        "version": VERSIONS[args.proposal_kind],
        "proposal_kind": args.proposal_kind,
        "training_proposal_count": len(rows),
        "training_positive_count": int(labels.sum()),
        "training_component_balanced_natural_positive_rate": natural_positive_rate,
        "feature_names": feature_names,
        "target": "crosses at least one official IoU50:0.05:0.90 threshold absent from frozen candidate pool",
        "score_contract": {
            "probability": "50/50 balanced-fit predict_proba corrected to the full official100 component-balanced natural positive rate",
            "base_quality": "minimum of track quality q and native exact-group median quality q",
            "append_score": "base_quality * (1 - (1 - P(threshold_cross))**3)",
            "duplicate_proposal_geometry_aggregation": "highest OOF threshold-cross probability",
            "threshold": None,
            "weight_or_exponent_scan": False,
        },
        "candidate_materialized": False,
        "ap_computed": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "input_provenance": {
            "utility_ledger_summary": str((args.utility_ledger_root / "summary.json").resolve()),
            "oof_diagnostic_summary": str(args.oof_diagnostic_summary.resolve()),
        },
    }
    package = {"metadata": metadata, "model": model}
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with (staging / "model_package.pkl").open("wb") as handle:
            pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-ledger-root", type=Path, required=True)
    parser.add_argument("--oof-diagnostic-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-model-package", type=Path,
        help="Reuse an already completed fit when only the score-contract metadata changes.",
    )
    parser.add_argument(
        "--proposal-kind", choices=tuple(VERSIONS), default="pair_union"
    )
    args = parser.parse_args()
    for name in ("utility_ledger_root", "oof_diagnostic_summary", "output_root", "reuse_model_package"):
        if getattr(args, name) is None:
            continue
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "version": summary["version"],
        "training_proposal_count": summary["training_proposal_count"],
        "training_positive_count": summary["training_positive_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

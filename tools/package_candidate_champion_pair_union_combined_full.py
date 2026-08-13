#!/usr/bin/env python3
"""Package the two fitted heads of the winning combined official100 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VERSION = "official100_champion_pair_union_combined_full_v2_prior_corrected"
POLICY = "champion_track_suppression_plus_pair_union_append"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    champion_metadata = json.loads(args.champion_metadata.read_text())
    union_metadata = json.loads(args.pair_union_metadata.read_text())
    evaluation = json.loads(args.combined_oof_summary.read_text())
    if champion_metadata.get("version") != "official100_component_union_track_harm_full_v1":
        raise ValueError("unexpected champion full package")
    if union_metadata.get("version") != "official100_pair_union_threshold_cross_full_v2_prior_corrected":
        raise ValueError("unexpected pair-union full package")
    prior = union_metadata.get("training_component_balanced_natural_positive_rate")
    if not isinstance(prior, (int, float)) or not 0.0 < float(prior) < 1.0:
        raise ValueError("pair-union full package lacks a valid natural positive rate")
    if evaluation.get("policy") != POLICY:
        raise ValueError("unexpected combined OOF evaluation")
    if evaluation["delta_vs_champion"]["ap"] <= 0.0:
        raise ValueError("combined policy did not exceed champion AP")
    if int(evaluation["main_ap_positive_fold_count_vs_champion"]) != 5:
        raise ValueError("combined policy did not improve main AP in all folds")
    with args.champion_model.open("rb") as handle:
        champion_package = pickle.load(handle)
    with args.pair_union_model.open("rb") as handle:
        union_package = pickle.load(handle)
    metadata = {
        "version": VERSION,
        "frozen_policy": POLICY,
        "component_packages": {
            "track_suppression": {
                "version": champion_metadata["version"],
                "model_sha256": _sha256(args.champion_model),
            },
            "pair_union_append": {
                "version": union_metadata["version"],
                "model_sha256": _sha256(args.pair_union_model),
            },
        },
        "score_contract": {
            "existing_candidates": champion_metadata["score_contract"],
            "pair_union_candidates": union_metadata["score_contract"],
            "cross_module_weight": None,
            "threshold_scan": False,
        },
        "official100_oof_metrics": evaluation["metrics"][POLICY],
        "official100_oof_delta_vs_previous_champion": evaluation["delta_vs_champion"],
        "official100_main_ap_positive_fold_count_vs_previous_champion": 5,
        "candidate_geometry_contract": {
            "existing_candidates_modified": False,
            "pair_union_append_only": True,
        },
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "safety_evaluation_run": False,
        "input_provenance": {
            "champion_metadata_sha256": _sha256(args.champion_metadata),
            "pair_union_metadata_sha256": _sha256(args.pair_union_metadata),
            "combined_oof_summary_sha256": _sha256(args.combined_oof_summary),
        },
    }
    package = {
        "metadata": metadata,
        "track_suppression_package": champion_package,
        "pair_union_append_package": union_package,
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with (staging / "model_package.pkl").open("wb") as handle:
            pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata["model_package_sha256"] = _sha256(staging / "model_package.pkl")
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
    parser.add_argument("--champion-model", type=Path, required=True)
    parser.add_argument("--champion-metadata", type=Path, required=True)
    parser.add_argument("--pair-union-model", type=Path, required=True)
    parser.add_argument("--pair-union-metadata", type=Path, required=True)
    parser.add_argument("--combined-oof-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "champion_model", "champion_metadata", "pair_union_model",
        "pair_union_metadata", "combined_oof_summary", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    metadata = run(args)
    print(json.dumps({
        "version": metadata["version"],
        "frozen_policy": metadata["frozen_policy"],
        "model_package_sha256": metadata["model_package_sha256"],
        "official100_oof_metrics": metadata["official100_oof_metrics"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

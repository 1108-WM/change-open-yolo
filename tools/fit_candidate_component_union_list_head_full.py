#!/usr/bin/env python3
"""Fit the full official100 component-list package with union features.

Candidate-quality and relation heads are fit on all official100 scenes.  The
final list winner is fit on scene-disjoint OOF stacked evidence augmented by
the frozen no-GT component-union feature ledger.  No holdout split is read and
no AP is evaluated.
"""
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

from tools.audit_train_candidate_quality_dataset import read_jsonl, read_scene_list  # noqa: E402
from tools.build_train_candidate_component_action_utility_ledger import _sha256  # noqa: E402
from tools.fit_candidate_component_action_head_structured_full import (  # noqa: E402
    _fit_quality_heads,
    _fit_relation_head,
)
from tools.fit_candidate_component_list_calibration_head_full import _build_oof_meta_rows  # noqa: E402
from tools.train_candidate_component_action_head_oof import EXPECTED_SPLIT_SHA256  # noqa: E402
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    RANDOM_SEED as RELATION_RANDOM_SEED,
    RELATIVE_FEATURES,
    TARGET_FEATURES,
)
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    MODEL_PARAMS,
    _balanced_winner_weights,
    _feature_matrix,
    _load_component_candidates,
)
from tools.train_candidate_component_union_list_head_oof import augment_union_features  # noqa: E402
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_union_ap25_protected_head_oof import _load_union_features  # noqa: E402


VERSION = "official100_component_union_track_harm_full_v1"
POLICY = "track_harm_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("full fit requires frozen official100")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    candidates, component_keys = _load_component_candidates(
        scenes, args.action_ledger_root, candidate_rows
    )
    relation_rows = [
        row for scene in scenes
        for row in read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl")
    ]
    if any(row.get("contracts", {}).get("feature_ground_truth_usage") != "none" for row in relation_rows):
        raise ValueError("relation feature ledger violates no-GT inference contract")
    union_lookup, union_names, union_summary = _load_union_features(
        args.component_union_feature_ledger_root, scenes
    )
    quality_models, quality_names, quality_diagnostics = _fit_quality_heads(candidate_rows)
    target_model, target_diagnostics = _fit_relation_head(
        relation_rows, TARGET_FEATURES,
        lambda row: row["labels"]["target_state"] in ("same_target", "different_target_coexist"),
        lambda row: row["labels"]["target_state"] == "same_target",
        RELATION_RANDOM_SEED + 3000,
    )
    relative_model, relative_diagnostics = _fit_relation_head(
        relation_rows, RELATIVE_FEATURES,
        lambda row: row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native"),
        lambda row: row["labels"]["relative_quality_state"] == "prefer_track",
        RELATION_RANDOM_SEED + 3001,
    )
    meta_rows, _base_feature_names, meta_diagnostics = _build_oof_meta_rows(
        candidates, relation_rows, candidate_rows, manifest,
        args.candidate_quality_protocol_name,
    )
    meta_rows = augment_union_features(meta_rows, union_lookup, union_names)
    matrix, feature_names = _feature_matrix(meta_rows)
    labels = np.asarray([
        int(row["label_component_unique_winner"]) for row in meta_rows
    ], dtype=np.int64)
    indexes = np.arange(len(meta_rows), dtype=np.int64)
    weights = _balanced_winner_weights(meta_rows, indexes, labels)
    model = HistGradientBoostingClassifier(
        loss="log_loss", **{**MODEL_PARAMS, "random_state": 20260813}
    )
    model.fit(matrix, labels, sample_weight=weights)
    metadata = {
        "version": VERSION,
        "frozen_policy": POLICY,
        "training_scene_count": len(scenes),
        "training_candidate_count": len(candidate_rows),
        "training_component_candidate_count": len(candidates),
        "training_relation_count": len(relation_rows),
        "training_component_count": len(component_keys),
        "feature_contract": {
            "candidate_quality_feature_names": quality_names,
            "target_relation_feature_names": list(TARGET_FEATURES),
            "relative_relation_feature_names": list(RELATIVE_FEATURES),
            "list_candidate_feature_names": feature_names,
            "component_union_feature_names": union_names,
            "component_union_native_not_applicable_encoding": "zeros plus union__not_applicable=1",
            "ground_truth_fields_in_inference_features": False,
            "list_head_training_features": "scene-disjoint OOF stacked evidence plus no-GT union geometry",
        },
        "fit_diagnostics": {
            "candidate_quality": quality_diagnostics,
            "target_relation": target_diagnostics,
            "relative_relation": relative_diagnostics,
            "OOF_meta_folds": meta_diagnostics,
            "list_winner_count": int(labels.sum()),
            "list_nonwinner_count": int(len(labels) - labels.sum()),
        },
        "score_contract": {
            "native_score": "bit-for-bit frozen",
            "track_score": "full_quality_q * (1 - (1 - P(keep))**3)",
            "candidate_removed": False,
            "threshold": None,
            "continuous_weight_scan": False,
        },
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ap_evaluation_run": False,
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "relation_feature_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
            "action_ledger_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "component_union_summary_sha256": _sha256(
                args.component_union_feature_ledger_root / "summary.json"
            ),
            "component_union_version": union_summary["version"],
            "OOF_selection_summary_sha256": _sha256(args.oof_selection_summary),
        },
    }
    package = {
        "metadata": metadata,
        "quality_models": quality_models,
        "target_relation_model": target_model,
        "relative_relation_model": relative_model,
        "list_winner_model": model,
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
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--candidate-records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--component-union-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--oof-selection-summary", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "split_manifest", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_feature_ledger_root", "oof_selection_summary", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    metadata = run(args)
    print(json.dumps({
        "version": metadata["version"],
        "frozen_policy": metadata["frozen_policy"],
        "model_package_sha256": metadata["model_package_sha256"],
        "training_component_candidate_count": metadata["training_component_candidate_count"],
        "component_union_feature_count": len(
            metadata["feature_contract"]["component_union_feature_names"]
        ),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

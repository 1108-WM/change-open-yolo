#!/usr/bin/env python3
"""Fit the frozen full-official100 track-risk calibration package.

The deployable package reuses full official100 candidate-quality and relation
heads, then fits the selected component unique-representative classifier on
scene-disjoint OOF meta-features.  The frozen inference policy keeps every
native score bit-for-bit unchanged and continuously suppresses only track
scores with ``1 - (1 - P(keep))**3``.  No safety/even/test input is read.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
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
from tools.train_candidate_component_action_head_oof import EXPECTED_SPLIT_SHA256  # noqa: E402
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    RANDOM_SEED as RELATION_RANDOM_SEED,
    RELATIVE_FEATURES,
    TARGET_FEATURES,
    _relation_evidence_for_outer_fold,
)
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    MODEL_PARAMS,
    _balanced_winner_weights,
    _feature_matrix,
    _load_component_candidates,
    build_candidate_feature_rows,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)


VERSION = "official100_component_track_harm_calibration_full_v1"
POLICY = "track_harm_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_oof_meta_rows(
    candidates: list[dict], relation_rows: list[dict], candidate_rows: list[dict],
    manifest: dict, candidate_protocol: str,
) -> tuple[list[dict], list[str], list[dict]]:
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    relation_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in relation_rows:
        relation_by_component[(str(row["scene_name"]), int(row["relation_component_id"]))].append(row)
    meta_rows = []
    feature_names = None
    diagnostics = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked, evidence_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index, candidate_protocol,
        )
        stacked_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for raw, evidence in zip(relation_rows, stacked):
            stacked_by_component[(str(raw["scene_name"]), int(raw["relation_component_id"]))].append(evidence)
        validation = set(fold["validation_scenes"])
        current = build_candidate_feature_rows(
            [row for row in candidates if row["scene_name"] in validation],
            relation_by_component,
            stacked_by_component,
        )
        _, names = _feature_matrix(current)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("full-fit OOF meta feature schema changed")
        meta_rows.extend(current)
        diagnostics.append({
            "fold_index": fold_index,
            "validation_candidate_count": len(current),
            "stacked_evidence": evidence_diagnostics,
        })
        print(f"[track-risk full fit] OOF meta fold {fold_index} complete", flush=True)
    if len(meta_rows) != len(candidates):
        raise AssertionError("OOF meta rows do not cover every component candidate")
    return meta_rows, feature_names or [], diagnostics


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
    meta_rows, feature_names, meta_diagnostics = _build_oof_meta_rows(
        candidates, relation_rows, candidate_rows, manifest,
        args.candidate_quality_protocol_name,
    )
    matrix, names = _feature_matrix(meta_rows)
    if names != feature_names:
        raise AssertionError("full-fit feature schema differs from OOF schema")
    labels = np.asarray([
        int(row["label_component_unique_winner"]) for row in meta_rows
    ], dtype=np.int64)
    indexes = np.arange(len(meta_rows), dtype=np.int64)
    weights = _balanced_winner_weights(meta_rows, indexes, labels)
    model = HistGradientBoostingClassifier(
        loss="log_loss", **{**MODEL_PARAMS, "random_state": 20260812}
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
            "ground_truth_fields_in_inference_features": False,
            "list_head_training_features": "scene-disjoint OOF stacked evidence",
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
            "track_score": "original_score * (1 - (1 - P(keep))**3)",
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
    parser.add_argument("--oof-selection-summary", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "split_manifest", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "oof_selection_summary", "output_root",
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
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

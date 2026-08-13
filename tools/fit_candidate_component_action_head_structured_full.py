#!/usr/bin/env python3
"""Fit and export the frozen full-data structured component action package.

The deployable package contains three candidate-quality heads, the frozen
same-target and relative-quality relation heads, and two action-kind model
families.  Action heads are fitted on scene-disjoint OOF stacked evidence so
their training feature distribution matches the frozen official100 study.
No safety60 input or ground truth is read by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    candidate_ledger_path,
    read_jsonl,
    read_scene_list,
)
from tools.build_train_candidate_component_action_utility_ledger import EPS, _sha256  # noqa: E402
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import (  # noqa: E402
    FEATURE_GROUP,
    PROTOCOL_NAME,
)
from tools.train_candidate_component_action_head_oof import load_training_rows  # noqa: E402
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    ACTION_MODEL_PARAMS,
    RANDOM_SEED,
    RELATIVE_FEATURES,
    TARGET_FEATURES,
    UTILITY_SCALE,
    _component_weights,
    _feature_matrix,
    _relation_evidence_for_outer_fold,
    class_balanced_state_weights,
    structured_action_feature_row,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    base_sample_weights,
    canonicalize_predictions,
    feature_matrix,
    fit_predict,
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
    make_model,
    source_balanced_weights,
    target_values,
)
from tools.train_candidate_target_consistency_oof import (  # noqa: E402
    _fit_base,
    scene_track_balanced_weights,
)


VERSION = "official100_structured_component_action_full_v2"
QUALITY_TARGETS = ("q", "valid25", "valid50")
QUALITY_RANDOM_SEED = 20260808


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _combined_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fit_quality_heads(candidate_rows: list[dict]) -> tuple[dict, list[str], dict]:
    matrix, names = feature_matrix(candidate_rows, FEATURE_GROUP, PROTOCOL_NAME)
    base = base_sample_weights(candidate_rows)
    weights = source_balanced_weights(candidate_rows, base)
    models = {}
    diagnostics = {}
    for offset, target in enumerate(QUALITY_TARGETS):
        labels = target_values(candidate_rows, target)
        model = make_model(target, QUALITY_RANDOM_SEED + offset, labels, weights, source_only=False)
        check = canonicalize_predictions(
            target, fit_predict(model, matrix, labels, weights, matrix[:1])
        )
        if check.shape != (1,):
            raise AssertionError(f"{target}: full quality fit failed")
        models[target] = model
        diagnostics[target] = {
            "label_positive_count": int(labels.sum()) if target != "q" else None,
            "label_mean": float(np.average(labels, weights=weights)),
            "random_seed": QUALITY_RANDOM_SEED + offset,
            "model": type(model).__name__,
        }
    return models, names, diagnostics


def _fit_relation_head(
    rows: list[dict], feature_names: tuple[str, ...], selector, labeler, seed: int,
):
    selected = [row for row in rows if selector(row)]
    labels = np.asarray([int(labeler(row)) for row in selected], dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("full relation fit lacks both classes")
    matrix = np.asarray([
        [float(row["features"][name]) for name in feature_names] for row in selected
    ], dtype=np.float64)
    weights = scene_track_balanced_weights(selected)
    model = _fit_base(matrix, labels, weights, seed)
    return model, {
        "training_relation_count": len(selected),
        "positive_count": int(labels.sum()),
        "feature_names": list(feature_names),
        "random_seed": seed,
    }


def _build_oof_meta_rows(
    scenes: list[str], manifest: dict, relation_rows: list[dict], candidate_rows: list[dict],
    base_rows: list[dict], all_actions: dict, candidate_protocol: str,
) -> tuple[list[dict], list[str], list[dict]]:
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    relation_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in relation_rows:
        relation_by_component[(str(row["scene_name"]), int(row["relation_component_id"]))].append(row)
    base_by_key = {
        (str(row["scene_name"]), int(row["relation_component_id"]), str(row["action_name"])): row
        for row in base_rows
    }
    meta_rows = []
    fold_diagnostics = []
    feature_names = None
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked, diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index, candidate_protocol,
        )
        stacked_by_component: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for raw, evidence in zip(relation_rows, stacked):
            stacked_by_component[(str(raw["scene_name"]), int(raw["relation_component_id"]))].append(evidence)
        validation = set(fold["validation_scenes"])
        current = []
        for key, base in base_by_key.items():
            if key[0] not in validation:
                continue
            action = next(
                row for row in all_actions[(key[0], key[1])]
                if str(row["action_name"]) == key[2]
            )
            current.append({
                **{field: base[field] for field in (
                    "scene_name", "relation_component_id", "action_name", "action_kind",
                    "selected_track_id", "label_utility", "label_positive",
                )},
                "model_features": structured_action_feature_row(
                    action, relation_by_component[(key[0], key[1])],
                    stacked_by_component[(key[0], key[1])],
                ),
            })
        _, names = _feature_matrix(current)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("OOF meta feature schema changed between folds")
        meta_rows.extend(current)
        fold_diagnostics.append({
            "fold_index": fold_index,
            "validation_meta_action_count": len(current),
            "stacked_evidence": diagnostics,
        })
        print(f"[full fit] OOF meta features fold {fold_index} complete", flush=True)
    if len(meta_rows) != len(base_rows):
        raise AssertionError("OOF meta rows do not cover every non-coexist action exactly once")
    return meta_rows, feature_names or [], fold_diagnostics


def _fit_action_models(rows: list[dict]) -> tuple[dict, dict]:
    matrix, feature_names = _feature_matrix(rows)
    utility = np.asarray([float(row["label_utility"]) for row in rows], dtype=np.float64)
    states = np.asarray([
        2 if value > EPS else (0 if value < -EPS else 1) for value in utility
    ], dtype=np.int64)
    package = {}
    diagnostics = {"feature_names": feature_names, "kinds": {}}
    for kind_offset, kind in enumerate(("baseline_only", "track_only_one")):
        indexes = np.flatnonzero(np.asarray([row["action_kind"] == kind for row in rows]))
        common = {**ACTION_MODEL_PARAMS, "random_state": RANDOM_SEED + 2000 + kind_offset}
        state_weights = class_balanced_state_weights(rows, indexes, states)
        state_model = HistGradientBoostingClassifier(loss="log_loss", **common)
        state_model.fit(matrix[indexes], states[indexes], sample_weight=state_weights[indexes])
        positive = indexes[states[indexes] == 2]
        harmful = indexes[states[indexes] == 0]
        if len(positive) < 20 or len(harmful) < 20:
            raise ValueError(f"{kind}: insufficient positive/harmful full-fit actions")
        magnitude_weights = _component_weights(rows, indexes)
        models = {
            "gain_mean": HistGradientBoostingRegressor(loss="squared_error", **common),
            "gain_lower25": HistGradientBoostingRegressor(loss="quantile", quantile=0.25, **common),
            "cost_mean": HistGradientBoostingRegressor(loss="squared_error", **common),
            "cost_upper75": HistGradientBoostingRegressor(loss="quantile", quantile=0.75, **common),
        }
        models["gain_mean"].fit(
            matrix[positive], utility[positive] * UTILITY_SCALE,
            sample_weight=magnitude_weights[positive],
        )
        models["gain_lower25"].fit(
            matrix[positive], utility[positive] * UTILITY_SCALE,
            sample_weight=magnitude_weights[positive],
        )
        models["cost_mean"].fit(
            matrix[harmful], -utility[harmful] * UTILITY_SCALE,
            sample_weight=magnitude_weights[harmful],
        )
        models["cost_upper75"].fit(
            matrix[harmful], -utility[harmful] * UTILITY_SCALE,
            sample_weight=magnitude_weights[harmful],
        )
        package[kind] = {"state": state_model, **models}
        diagnostics["kinds"][kind] = {
            "training_action_count": len(indexes),
            "state_counts": {
                "harmful": int(np.sum(states[indexes] == 0)),
                "neutral": int(np.sum(states[indexes] == 1)),
                "positive": int(np.sum(states[indexes] == 2)),
            },
            "random_seed": RANDOM_SEED + 2000 + kind_offset,
        }
    return package, diagnostics


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise ValueError("full structured fit requires frozen official100")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    relation_rows = [
        row for scene in scenes
        for row in read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl")
    ]
    if any(row.get("contracts", {}).get("feature_ground_truth_usage") != "none" for row in relation_rows):
        raise ValueError("official relation feature ledger violates no-GT feature contract")
    base_rows, all_actions, base_contract = load_training_rows(
        scenes, args.action_ledger_root, args.relation_feature_ledger_root,
    )
    quality_models, quality_names, quality_diagnostics = _fit_quality_heads(candidate_rows)
    target_model, target_diagnostics = _fit_relation_head(
        relation_rows, TARGET_FEATURES,
        lambda row: row["labels"]["target_state"] in ("same_target", "different_target_coexist"),
        lambda row: row["labels"]["target_state"] == "same_target",
        RANDOM_SEED + 3000,
    )
    relative_model, relative_diagnostics = _fit_relation_head(
        relation_rows, RELATIVE_FEATURES,
        lambda row: row["labels"]["relative_quality_state"] in ("prefer_track", "prefer_native"),
        lambda row: row["labels"]["relative_quality_state"] == "prefer_track",
        RANDOM_SEED + 3001,
    )
    meta_rows, _, fold_diagnostics = _build_oof_meta_rows(
        scenes, manifest, relation_rows, candidate_rows, base_rows, all_actions,
        args.candidate_quality_protocol_name,
    )
    action_models, action_diagnostics = _fit_action_models(meta_rows)

    input_paths = [args.scene_list, args.split_manifest]
    input_paths.extend(candidate_ledger_path(args.candidate_records_root, scene) for scene in scenes)
    input_paths.extend(args.relation_feature_ledger_root / scene / "relation_features.jsonl" for scene in scenes)
    input_paths.extend(args.action_ledger_root / scene / "component_action_utilities.jsonl" for scene in scenes)
    metadata = {
        "version": VERSION,
        "training_scene_count": len(scenes),
        "training_candidate_count": len(candidate_rows),
        "training_relation_count": len(relation_rows),
        "training_noncoexist_action_count": len(meta_rows),
        "training_source_counts": {
            source: sum(row["candidate_source"] == source for row in candidate_rows)
            for source in (NATIVE_SOURCE, TRACK_SOURCE)
        },
        "feature_contract": {
            "candidate_quality_group": FEATURE_GROUP,
            "candidate_quality_protocol": PROTOCOL_NAME,
            "candidate_quality_feature_names": quality_names,
            "target_relation_feature_names": list(TARGET_FEATURES),
            "relative_relation_feature_names": list(RELATIVE_FEATURES),
            "action_feature_names": action_diagnostics["feature_names"],
            "action_base_contract": base_contract,
            "ground_truth_fields_in_inference_features": False,
            "action_meta_training_features": "scene-disjoint OOF stacked evidence",
        },
        "fit_diagnostics": {
            "candidate_quality": quality_diagnostics,
            "target_relation": target_diagnostics,
            "relative_relation": relative_diagnostics,
            "action": action_diagnostics["kinds"],
            "oof_meta_folds": fold_diagnostics,
        },
        "frozen_policy": "structured_lower_relation_veto",
        "threshold_scanning": False,
        "safety60_read": False,
        "safety60_ground_truth_read": False,
        "ap_evaluation_run": False,
        "input_provenance": {
            "scene_list": str(args.scene_list.resolve()),
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "relation_feature_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
            "action_ledger_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "all_training_inputs_combined_sha256": _combined_digest(input_paths),
        },
    }
    package = {
        "metadata": metadata,
        "quality_models": quality_models,
        "target_relation_model": target_model,
        "relative_relation_model": relative_model,
        "action_models": action_models,
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
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "split_manifest", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root exists and is non-empty: {args.output_root}")
    metadata = run(args)
    print(json.dumps({
        "version": metadata["version"],
        "model_package_sha256": metadata["model_package_sha256"],
        "training_candidate_count": metadata["training_candidate_count"],
        "training_relation_count": metadata["training_relation_count"],
        "training_noncoexist_action_count": metadata["training_noncoexist_action_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

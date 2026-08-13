#!/usr/bin/env python3
"""Train a residual-applicable track expert on frozen official100 outer folds.

Only non-empty-residual tracks from each outer training fold fit the expert.
For validation tracks with a non-empty residual, its winner probability is
combined with the incumbent winner probability by a fixed geometric mean.
Native candidates and empty-residual tracks keep incumbent predictions exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (  # noqa: E402
    NATIVE_SOURCE,
    TRACK_SOURCE,
    read_jsonl,
    read_scene_list,
)
from tools.build_train_candidate_component_action_utility_ledger import (  # noqa: E402
    _sha256,
    _write_jsonl,
    configure_track_score_context,
)
from tools.train_candidate_component_action_head_oof import EXPECTED_SPLIT_SHA256  # noqa: E402
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    _relation_evidence_for_outer_fold,
)
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    MODEL_PARAMS,
    RANDOM_SEED,
    _balanced_winner_weights,
    _evaluate_oof_scores,
    _feature_matrix,
    _load_component_candidates,
    _triplet,
    build_candidate_feature_rows,
)
from tools.train_candidate_component_union_list_head_oof import augment_union_features  # noqa: E402
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_union_ap25_protected_head_oof import _load_union_features  # noqa: E402


VERSION = "official100_component_union_residual_expert_head_oof_v1"
PRIMARY_POLICY = "track_harm_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def fixed_geometric_keep(incumbent: float, expert: float) -> float:
    return float(math.sqrt(max(1e-12, float(incumbent)) * max(1e-12, float(expert))))


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("must use frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    scene_to_fold = {
        scene: int(fold["fold_index"])
        for fold in manifest["folds"] for scene in fold["validation_scenes"]
    }
    score_context = configure_track_score_context(args, scenes)
    candidate_rows = load_candidate_rows(args.scene_list, args.candidate_records_root)
    candidates, _ = _load_component_candidates(
        scenes, args.action_ledger_root, candidate_rows
    )
    relation_rows = []
    relation_by_component = defaultdict(list)
    for scene in scenes:
        for row in read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl"):
            relation_rows.append(row)
            relation_by_component[(scene, int(row["relation_component_id"]))].append(row)
    union_lookup, union_names, union_summary = _load_union_features(
        args.component_union_residual_feature_ledger_root, scenes
    )
    incumbent_rows = read_jsonl(args.incumbent_predictions)
    incumbent = {
        (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])): row
        for row in incumbent_rows
    }
    if len(incumbent) != len(incumbent_rows):
        raise ValueError("duplicate incumbent OOF predictions")

    final_predictions = {}
    fold_details = []
    feature_names = None
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked_all, stacked_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index,
            args.candidate_quality_protocol_name,
        )
        stacked_by_component = defaultdict(list)
        for raw, stacked in zip(relation_rows, stacked_all):
            stacked_by_component[(str(raw["scene_name"]), int(raw["relation_component_id"]))].append(stacked)
        model_rows = build_candidate_feature_rows(
            candidates, relation_by_component, stacked_by_component
        )
        model_rows = augment_union_features(model_rows, union_lookup, union_names)
        matrix, current_names = _feature_matrix(model_rows)
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise AssertionError("residual expert feature schema changed across folds")
        row_scenes = np.asarray([row["scene_name"] for row in model_rows], dtype=object)
        labels = np.asarray([int(row["label_component_unique_winner"]) for row in model_rows])
        applicable = np.asarray([
            row["candidate_source"] == TRACK_SOURCE
            and row["model_features"]["union__residual_empty"] == 0.0
            for row in model_rows
        ])
        train = np.flatnonzero(
            applicable & np.isin(row_scenes, fold["train_scenes"])
        )
        validation = np.flatnonzero(
            applicable & np.isin(row_scenes, fold["validation_scenes"])
        )
        if len(set(labels[train])) != 2 or len(set(labels[validation])) != 2:
            raise ValueError(f"fold {fold_index}: residual expert lacks one class")
        weights = _balanced_winner_weights(model_rows, train, labels)
        model = HistGradientBoostingClassifier(
            loss="log_loss", **{**MODEL_PARAMS, "random_state": RANDOM_SEED + 7000 + fold_index * 100}
        )
        model.fit(matrix[train], labels[train], sample_weight=weights[train])
        expert_probability = model.predict_proba(matrix[validation])[:, 1]
        expert_by_key = {
            (
                model_rows[index]["scene_name"], model_rows[index]["candidate_source"],
                int(model_rows[index]["candidate_id"]),
            ): float(probability)
            for index, probability in zip(validation, expert_probability)
        }
        validation_scenes = set(fold["validation_scenes"])
        selected_counts = {"incumbent_native": 0, "incumbent_empty_track": 0, "blended_nonempty_track": 0}
        for row in model_rows:
            if row["scene_name"] not in validation_scenes:
                continue
            key = (row["scene_name"], row["candidate_source"], int(row["candidate_id"]))
            old = incumbent[key]
            if row["candidate_source"] == NATIVE_SOURCE:
                prediction = {**old, "residual_expert_source": "incumbent_native"}
                selected_counts["incumbent_native"] += 1
            elif row["model_features"]["union__residual_empty"] == 1.0:
                prediction = {**old, "residual_expert_source": "incumbent_empty_track"}
                selected_counts["incumbent_empty_track"] += 1
            else:
                expert = expert_by_key[key]
                prediction = {
                    **old,
                    "keep_probability": fixed_geometric_keep(old["keep_probability"], expert),
                    "incumbent_keep_probability": float(old["keep_probability"]),
                    "residual_expert_keep_probability": expert,
                    "residual_expert_source": "blended_nonempty_track",
                }
                selected_counts["blended_nonempty_track"] += 1
            if key in final_predictions:
                raise AssertionError("OOF residual expert prediction repeated")
            final_predictions[key] = prediction
        fold_details.append({
            "fold_index": fold_index,
            "train_applicable_track_count": len(train),
            "validation_applicable_track_count": len(validation),
            "validation_winner_count": int(labels[validation].sum()),
            "expert_roc_auc": float(roc_auc_score(labels[validation], expert_probability)),
            "expert_pr_auc": float(average_precision_score(labels[validation], expert_probability)),
            "selection_counts": selected_counts,
            "stacked_evidence": stacked_diagnostics,
        })
        print(f"[residual expert head] outer fold {fold_index} complete", flush=True)
    if set(final_predictions) != set(incumbent):
        raise ValueError("residual expert OOF coverage differs from incumbent")
    overall, fold_ap, score_rows = _evaluate_oof_scores(
        args, scenes, manifest, final_predictions
    )
    baseline = _triplet(overall["frozen_coexist"])
    triplets = {name: _triplet(metrics) for name, metrics in overall.items()}
    deltas = {
        name: {key: value - baseline[key] for key, value in triplet.items()}
        for name, triplet in triplets.items()
    }
    primary_fold_deltas = []
    for fold in fold_ap["folds"]:
        base = _triplet(fold["policy_metrics"]["frozen_coexist"])
        selected = _triplet(fold["policy_metrics"][PRIMARY_POLICY])
        primary_fold_deltas.append({
            "fold_index": int(fold["fold_index"]),
            "delta": {key: selected[key] - base[key] for key in base},
        })
    summary = {
        "version": VERSION,
        "primary_policy": PRIMARY_POLICY,
        "scene_count": 100,
        "controlled_candidate_count": len(final_predictions),
        "expert_contract": {
            "fit_population": "non-empty-residual tracks from outer training fold only",
            "native_prediction": "incumbent unchanged",
            "empty_residual_track_prediction": "incumbent unchanged",
            "nonempty_residual_track_prediction": "fixed geometric mean of incumbent and expert keep probabilities",
            "blend_weight_scan": False,
            "applicability_threshold_scan": False,
            "GVC_hard_gate": False,
        },
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "component_union_residual_feature_count": len(union_names),
            "ground_truth_fields_in_features": False,
        },
        "fold_details": fold_details,
        "policy_metrics": triplets,
        "policy_delta_vs_frozen_coexist": deltas,
        "primary_fold_deltas": primary_fold_deltas,
        "score_context": score_context,
        "candidate_count_modified": False,
        "candidate_geometry_modified": False,
        "candidate_class_modified": False,
        "candidate_files_modified": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official100 outer-fold supervision and offline AP only",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "incumbent_predictions_sha256": _sha256(args.incumbent_predictions),
            "component_union_residual_summary_sha256": _sha256(
                args.component_union_residual_feature_ledger_root / "summary.json"
            ),
            "component_union_residual_version": union_summary["version"],
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "oof_candidate_predictions.jsonl", [
            final_predictions[key] for key in sorted(final_predictions)
        ])
        _write_jsonl(staging / "oof_candidate_scores.jsonl", score_rows)
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
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--candidate-records-root", type=Path, required=True)
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--action-ledger-root", type=Path, required=True)
    parser.add_argument("--component-union-residual-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--incumbent-predictions", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--track-score-mode", choices=("oof_quality",), default="oof_quality")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_residual_feature_ledger_root", "incumbent_predictions",
        "gt_dir", "oof_predictions", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "primary_metrics": summary["policy_metrics"][PRIMARY_POLICY],
        "primary_delta": summary["policy_delta_vs_frozen_coexist"][PRIMARY_POLICY],
        "primary_fold_deltas": summary["primary_fold_deltas"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

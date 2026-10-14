#!/usr/bin/env python3
"""Train the incumbent component-list head with union geometry features.

This is a single preregistered ablation: the incumbent labels, models, folds,
and ``track_harm_focal_suppression`` score formula are unchanged.  The only
change is appending the no-GT component-union feature ledger to track rows;
native rows receive explicit not-applicable values.  No holdout split is read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


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
    POLICIES,
    RANDOM_SEED,
    _balanced_winner_weights,
    _evaluate_oof_scores,
    _feature_matrix,
    _load_component_candidates,
    _triplet,
    build_candidate_feature_rows,
    fit_fold as fit_fold_incumbent,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)
from tools.train_candidate_union_ap25_protected_head_oof import _load_union_features  # noqa: E402


VERSION = "official100_component_union_list_head_oof_v1"
RESIDUAL_VERSION = "official100_component_union_residual_list_head_oof_v1"
OFFICIAL_RELEVANCE_VERSION = "official100_component_union_official_relevance_head_oof_v1"
MULTITASK_RELATION_VERSION = "official100_component_union_multitask_relation_list_head_oof_v1"
DUAL_RELATION_VERSION = "official100_component_union_dual_relation_list_head_oof_v1"
PRIMARY_POLICY = "track_harm_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def augment_union_features(
    rows: list[dict], union_lookup: dict, union_names: list[str],
) -> list[dict]:
    output = []
    seen_tracks = set()
    for row in rows:
        features = dict(row["model_features"])
        source_track = row["candidate_source"] == TRACK_SOURCE
        if source_track:
            key = (
                str(row["scene_name"]), int(row["relation_component_id"]),
                int(row["candidate_id"]),
            )
            if key not in union_lookup:
                raise ValueError(f"missing component union features: {key}")
            seen_tracks.add(key)
            for name in union_names:
                features[f"union__{name}"] = float(union_lookup[key][name])
            features["union__not_applicable"] = 0.0
        else:
            for name in union_names:
                features[f"union__{name}"] = 0.0
            features["union__not_applicable"] = 1.0
        output.append({**row, "model_features": features})
    if seen_tracks != set(union_lookup):
        missing = sorted(set(union_lookup) - seen_tracks)[:5]
        extra = sorted(seen_tracks - set(union_lookup))[:5]
        raise ValueError(f"union track coverage mismatch; missing={missing}, extra={extra}")
    return output


def strip_multitask_auxiliary_features(rows: list[dict]) -> list[dict]:
    """Remove only the optional multitask relation channel from model rows."""
    output = []
    for row in rows:
        features = {
            name: value for name, value in row["model_features"].items()
            if not name.startswith("relation__multitask_")
        }
        output.append({**row, "model_features": features})
    return output


def collapse_multitask_auxiliary_to_candidate_win_residual(
    rows: list[dict],
) -> list[dict]:
    """Keep the incumbent schema and append one signed auxiliary residual."""
    output = []
    incumbent_name = "relation__candidate_win_relation_score__mean"
    auxiliary_name = "relation__multitask_candidate_win_relation_score__mean"
    residual_name = "relation__multitask_candidate_win_residual__mean"
    for row in rows:
        source = row["model_features"]
        if incumbent_name not in source or auxiliary_name not in source:
            raise ValueError("candidate-win residual requires both relation channels")
        features = {
            name: value for name, value in source.items()
            if not name.startswith("relation__multitask_")
        }
        features[residual_name] = float(source[auxiliary_name] - source[incumbent_name])
        output.append({**row, "model_features": features})
    return output


def official_relevance_target(row: dict) -> float:
    value = float(row["label_official_relevance"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"official relevance is outside [0, 1]: {value}")
    if not int(row["label_component_unique_winner"]) and value != 0.0:
        raise ValueError("non-winner candidate has non-zero official relevance")
    return value


def fit_fold_official_relevance(
    rows: list[dict], matrix: np.ndarray,
    train_scenes: list[str], validation_scenes: list[str], seed: int,
) -> tuple[dict, dict]:
    """Replace only the incumbent keep head with an AP-threshold relevance head."""
    predictions, metrics = fit_fold_incumbent(
        rows, matrix, train_scenes, validation_scenes, seed,
    )
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    train = np.flatnonzero(np.isin(scenes, train_scenes))
    validation = np.flatnonzero(np.isin(scenes, validation_scenes))
    winner = np.asarray([int(row["label_component_unique_winner"]) for row in rows])
    relevance = np.asarray([official_relevance_target(row) for row in rows], dtype=np.float64)
    weights = _balanced_winner_weights(rows, train, winner)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        **{**MODEL_PARAMS, "random_state": seed + 9000},
    )
    model.fit(matrix[train], relevance[train], sample_weight=weights[train])
    relevance_prediction = np.clip(model.predict(matrix[validation]), 1e-6, 1.0 - 1e-6)
    prediction_by_candidate = {
        (
            str(prediction_key[0]), str(prediction["candidate_source"]),
            int(prediction["candidate_id"]),
        ): prediction
        for prediction_key, prediction in predictions.items()
    }
    for index, predicted in zip(validation, relevance_prediction):
        row = rows[int(index)]
        key = (str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"]))
        prediction = prediction_by_candidate[key]
        prediction["incumbent_keep_probability"] = float(prediction["keep_probability"])
        prediction["keep_probability"] = float(predicted)
        prediction["official_relevance_prediction"] = float(predicted)
        prediction["label_official_relevance"] = float(relevance[index])
    validation_target = relevance[validation]
    metrics = {
        **metrics,
        "keep_head": "official_relevance_hist_gradient_boosting_regression",
        "validation_official_relevance_nonzero_count": int((validation_target > 0.0).sum()),
        "validation_official_relevance_mean": float(validation_target.mean()),
        "validation_official_relevance_prediction_mean": float(relevance_prediction.mean()),
        "validation_official_relevance_mae": float(
            mean_absolute_error(validation_target, relevance_prediction)
        ),
    }
    return predictions, metrics


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
    candidates, component_keys = _load_component_candidates(
        scenes, args.action_ledger_root, candidate_rows
    )
    relation_rows = []
    relation_by_component = defaultdict(list)
    for scene in scenes:
        for row in read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl"):
            relation_rows.append(row)
            relation_by_component[(scene, int(row["relation_component_id"]))].append(row)
    union_lookup, union_names, union_summary = _load_union_features(
        args.component_union_feature_ledger_root, scenes
    )
    final_predictions = {}
    fold_details = []
    feature_names = None
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        stacked_all, stacked_diagnostics = _relation_evidence_for_outer_fold(
            relation_rows, candidate_rows, scene_to_fold, fold_index,
            args.candidate_quality_protocol_name,
            relative_target=args.relative_relation_target,
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
            raise AssertionError("union list feature schema changed across folds")
        fit_function = (
            fit_fold_official_relevance
            if args.head_target == "official_relevance" else fit_fold_incumbent
        )
        predictions, metrics = fit_function(
            model_rows, matrix, fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        validation = set(fold["validation_scenes"])
        for key, prediction in predictions.items():
            row_key = (prediction["candidate_source"], int(prediction["candidate_id"]))
            full_key = (key[0], row_key[0], row_key[1])
            if full_key in final_predictions:
                raise AssertionError("OOF candidate prediction repeated")
            final_predictions[full_key] = {
                "scene_name": key[0], "relation_component_id": key[1], **prediction,
            }
        expected_validation = sum(row["scene_name"] in validation for row in candidates)
        if len(predictions) != expected_validation:
            raise AssertionError("outer validation candidate coverage differs")
        fold_details.append({
            "fold_index": fold_index,
            "stacked_evidence": stacked_diagnostics,
            "head_metrics": metrics,
        })
        print(f"[component union list head] outer fold {fold_index} complete", flush=True)
    if len(final_predictions) != len(candidates):
        raise AssertionError("OOF union list predictions do not cover all candidates")
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
        metrics = fold["policy_metrics"]
        base = _triplet(metrics["frozen_coexist"])
        selected = _triplet(metrics[PRIMARY_POLICY])
        primary_fold_deltas.append({
            "fold_index": int(fold["fold_index"]),
            "delta": {key: selected[key] - base[key] for key in base},
        })
    residual_ablation = args.feature_family == "component_union_residual"
    official_relevance_ablation = args.head_target == "official_relevance"
    multitask_relation_ablation = (
        args.relative_relation_target == "multitask_binary_plus_official_gap"
    )
    dual_relation_ablation = (
        args.relative_relation_target == "incumbent_plus_multitask_auxiliary"
    )
    if residual_ablation and official_relevance_ablation:
        raise ValueError("residual features and official relevance target must be separate ablations")
    if multitask_relation_ablation and (residual_ablation or official_relevance_ablation):
        raise ValueError("multitask relation target must be an isolated component-union ablation")
    if dual_relation_ablation and (
        residual_ablation or official_relevance_ablation or multitask_relation_ablation
    ):
        raise ValueError("dual relation evidence must be an isolated component-union ablation")
    summary = {
        "version": (
            DUAL_RELATION_VERSION if dual_relation_ablation else (
                MULTITASK_RELATION_VERSION if multitask_relation_ablation else (
                    OFFICIAL_RELEVANCE_VERSION if official_relevance_ablation else (
                        RESIDUAL_VERSION if residual_ablation else VERSION
                    )
                )
            )
        ),
        "primary_policy": PRIMARY_POLICY,
        "scene_count": 100,
        "component_count": len(component_keys),
        "controlled_candidate_count": len(candidates),
        "controlled_source_counts": dict(sorted(Counter(
            row["candidate_source"] for row in candidates
        ).items())),
        "ablation_contract": {
            "incumbent_labels_unchanged": not official_relevance_ablation,
            "incumbent_model_unchanged": not (
                official_relevance_ablation or multitask_relation_ablation
                or dual_relation_ablation
            ),
            "incumbent_score_formula_unchanged": True,
            "only_change": (
                "replace binary IoU25 unique-winner keep head with continuous official AP50-90 relevance regression"
                if official_relevance_ablation else (
                    "replace only the relative-relation binary loss with the fixed 50/50 binary plus official-gap multitask loss"
                    if multitask_relation_ablation else (
                        "preserve incumbent relative-relation scores and append the fixed multitask scores as independent continuous auxiliary features"
                        if dual_relation_ablation else (
                            "append preregistered no-GT residual-region features to incumbent component-union features"
                            if residual_ablation else "append no-GT component-union features"
                        )
                    )
                )
            ),
        },
        "target_contract": {
            "keep_head_target": (
                "unique representative mean relevance over IoU thresholds 0.50:0.05:0.90"
                if official_relevance_ablation else "IoU25 unique representative binary label"
            ),
            "official_relevance_used": official_relevance_ablation,
            "relative_relation_target": args.relative_relation_target,
            "relative_relation_main_loss_fraction": (
                0.5 if multitask_relation_ablation else 1.0
            ),
            "relative_relation_official_gap_auxiliary_loss_fraction": (
                0.5 if (multitask_relation_ablation or dual_relation_ablation) else 0.0
            ),
            "incumbent_relative_relation_score_preserved": (
                not multitask_relation_ablation
            ),
            "multitask_relation_score_appended_as_separate_features": (
                dual_relation_ablation
            ),
            "AP25_label_added": False,
            "target_weight_scan": False,
        },
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "component_union_feature_count": len(union_names),
            "component_union_feature_names": union_names,
            "ground_truth_fields_in_features": False,
            "GVC_usage": "continuous feature only; never a hard gate",
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
        "ground_truth_usage": "official100 nested OOF supervision and offline AP only",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "relation_summary_sha256": _sha256(args.relation_feature_ledger_root / "summary.json"),
            "action_summary_sha256": _sha256(args.action_ledger_root / "summary.json"),
            "component_union_summary_sha256": _sha256(
                args.component_union_feature_ledger_root / "summary.json"
            ),
            "component_union_version": union_summary["version"],
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
    parser.add_argument("--component-union-feature-ledger-root", type=Path, required=True)
    parser.add_argument(
        "--feature-family",
        choices=("component_union", "component_union_residual"),
        default="component_union",
    )
    parser.add_argument(
        "--head-target",
        choices=("incumbent_unique_winner", "official_relevance"),
        default="incumbent_unique_winner",
    )
    parser.add_argument(
        "--relative-relation-target",
        choices=(
            "incumbent_binary", "multitask_binary_plus_official_gap",
            "incumbent_plus_multitask_auxiliary",
        ),
        default="incumbent_binary",
    )
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--soft-suppression-summary", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--track-score-mode", choices=("oof_quality",), default="oof_quality")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_feature_ledger_root", "gt_dir", "oof_predictions",
        "soft_suppression_summary", "output_root",
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

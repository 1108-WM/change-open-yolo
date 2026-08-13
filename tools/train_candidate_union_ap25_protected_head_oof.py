#!/usr/bin/env python3
"""Train and evaluate an AP25-protected union-risk head on official100 OOF.

The head is track-only.  It combines three scene-disjoint signals:

* OOF candidate-quality ``P(valid25)``;
* a component-union classifier for independent AP25 value;
* a component-union classifier for same-target native domination.

The fixed preservation probability is

``P(valid25) * (1 - P(native_dominated) * (1 - P(unique_valid25)))``.

Native scores remain bit-for-bit unchanged.  Track scores are continuously
suppressed with the same cubic tail as the previous frozen policy.  Only the
official100 train split and its frozen five folds are read.
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
    _scene_inputs,
    _scene_records,
    _sha256,
    configure_track_score_context,
)
from tools.diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    instance_eval,
)
from tools.evaluate_candidate_quality_reranking_class_agnostic_ap import _set_match_scores  # noqa: E402
from tools.train_candidate_component_action_head_oof import (  # noqa: E402
    EXPECTED_SPLIT_SHA256,
    _metrics_from_records,
)
from tools.train_candidate_component_action_head_structured_oof import (  # noqa: E402
    _relation_evidence_for_outer_fold,
)
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    MODEL_PARAMS,
    _load_component_candidates,
    build_candidate_feature_rows,
)
from tools.train_candidate_quality_head_oof import (  # noqa: E402
    load_frozen_split_manifest,
    load_rows as load_candidate_rows,
)


VERSION = "official100_union_ap25_protected_head_oof_v1"
POLICY = "union_ap25_protected_track_suppression"
RANDOM_SEED = 20260813


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def assign_track_protection_labels(candidates: list[dict]) -> None:
    """Attach GT-only OOF labels; never consumed as model features."""
    by_component = defaultdict(list)
    for row in candidates:
        by_component[(str(row["scene_name"]), int(row["relation_component_id"]))].append(row)
    for rows in by_component.values():
        natives = [row for row in rows if row["candidate_source"] == NATIVE_SOURCE]
        for track in (row for row in rows if row["candidate_source"] == TRACK_SOURCE):
            target = track.get("label_best_gt_instance_id")
            track_iou = float(track["label_best_gt_iou"])
            valid25 = target is not None and track_iou > 0.25
            same_target_native = [
                row for row in natives
                if target is not None and row.get("label_best_gt_instance_id") == target
            ]
            native_best = max(
                (float(row["label_best_gt_iou"]) for row in same_target_native),
                default=0.0,
            )
            native_covers25 = native_best > 0.25
            unique_valid25 = bool(valid25 and not native_covers25)
            native_dominated = bool(
                valid25 and native_covers25 and native_best >= track_iou - 1e-12
            )
            safe_keep25 = bool(valid25 and (unique_valid25 or not native_dominated))
            track.update({
                "label_track_valid25": int(valid25),
                "label_unique_valid25": int(unique_valid25),
                "label_native_dominated": int(native_dominated),
                "label_safe_keep25": int(safe_keep25),
                "label_same_target_native_best_iou": float(native_best),
            })


def protected_preserve_probability(
    valid25_probability: float,
    unique25_probability: float,
    native_dominated_probability: float,
) -> float:
    valid = min(1.0, max(0.0, float(valid25_probability)))
    unique = min(1.0, max(0.0, float(unique25_probability)))
    dominated = min(1.0, max(0.0, float(native_dominated_probability)))
    return valid * (1.0 - dominated * (1.0 - unique))


def suppressed_track_score(score: float, preserve_probability: float) -> tuple[float, float]:
    base = min(1.0, max(0.0, float(score)))
    preserve = min(1.0, max(0.0, float(preserve_probability)))
    risk = 1.0 - preserve
    multiplier = 1.0 - risk ** 3
    return multiplier, base * multiplier


def _load_union_features(root: Path, scenes: list[str]) -> tuple[dict, list[str], dict]:
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("feature_ground_truth_usage") != "none":
        raise ValueError("component union feature ledger violates no-GT contract")
    rows = [
        row for scene in scenes
        for row in read_jsonl(root / scene / "component_union_track_features.jsonl")
    ]
    lookup = {
        (str(row["scene_name"]), int(row["relation_component_id"]), int(row["track_id"])):
        row["model_features"] for row in rows
    }
    if len(lookup) != len(rows):
        raise ValueError("component union track feature identities repeat")
    names = sorted(rows[0]["model_features"]) if rows else []
    if names != summary["feature_names"]:
        raise ValueError("component union feature summary schema mismatch")
    return lookup, names, summary


def _track_rows(
    candidates: list[dict], relation_by_component: dict, stacked_by_component: dict,
    union_lookup: dict, union_names: list[str],
) -> tuple[list[dict], np.ndarray, list[str]]:
    rows = build_candidate_feature_rows(
        candidates, relation_by_component, stacked_by_component,
    )
    candidate_lookup = {
        (
            str(value["scene_name"]), int(value["relation_component_id"]),
            str(value["candidate_source"]), int(value["candidate_id"]),
        ): value for value in candidates
    }
    tracks = []
    for row in rows:
        if row["candidate_source"] != TRACK_SOURCE:
            continue
        key = (str(row["scene_name"]), int(row["relation_component_id"]), int(row["candidate_id"]))
        if key not in union_lookup:
            raise ValueError(f"missing union feature row: {key}")
        candidate = candidate_lookup[(key[0], key[1], TRACK_SOURCE, key[2])]
        model_features = dict(row["model_features"])
        model_features.update({
            f"union__{name}": float(union_lookup[key][name]) for name in union_names
        })
        tracks.append({
            **row,
            "model_features": model_features,
            "label_track_valid25": int(candidate["label_track_valid25"]),
            "label_unique_valid25": int(candidate["label_unique_valid25"]),
            "label_native_dominated": int(candidate["label_native_dominated"]),
            "label_safe_keep25": int(candidate["label_safe_keep25"]),
            "label_same_target_native_best_iou": float(
                candidate["label_same_target_native_best_iou"]
            ),
        })
    names = sorted(tracks[0]["model_features"])
    if any(sorted(row["model_features"]) != names for row in tracks):
        raise ValueError("union-risk feature schemas differ")
    forbidden = [name for name in names if "label" in name or "best_gt" in name]
    if forbidden:
        raise AssertionError(f"GT field leaked into features: {forbidden}")
    matrix = np.asarray([
        [float(row["model_features"][name]) for name in names] for row in tracks
    ], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("union-risk matrix contains non-finite values")
    return tracks, matrix, names


def _balanced_binary_weights(
    rows: list[dict], indexes: np.ndarray, labels: np.ndarray,
) -> np.ndarray:
    component_counts = Counter(
        (rows[int(index)]["scene_name"], int(rows[int(index)]["relation_component_id"]))
        for index in indexes
    )
    weights = np.zeros(len(rows), dtype=np.float64)
    for index in indexes:
        row = rows[int(index)]
        key = (row["scene_name"], int(row["relation_component_id"]))
        weights[int(index)] = 1.0 / component_counts[key]
    for label in (0, 1):
        selected = indexes[labels[indexes] == label]
        if not len(selected):
            raise ValueError("binary training subset lacks one class")
        weights[selected] *= 0.5 / weights[selected].sum()
    weights[indexes] /= weights[indexes].mean()
    return weights


def _metric(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) != 2:
        return {"positive_count": int(labels.sum()), "roc_auc": None, "pr_auc": None}
    return {
        "positive_count": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def _fit_fold(
    rows: list[dict], matrix: np.ndarray, train_scenes: list[str],
    validation_scenes: list[str], seed: int,
) -> tuple[dict, dict]:
    scenes = np.asarray([row["scene_name"] for row in rows], dtype=object)
    train = np.flatnonzero(np.isin(scenes, train_scenes))
    validation = np.flatnonzero(np.isin(scenes, validation_scenes))
    unique = np.asarray([int(row["label_unique_valid25"]) for row in rows], dtype=np.int64)
    dominated = np.asarray([int(row["label_native_dominated"]) for row in rows], dtype=np.int64)
    valid = np.asarray([int(row["label_track_valid25"]) for row in rows], dtype=np.int64)
    params = {**MODEL_PARAMS, "random_state": seed}
    unique_model = HistGradientBoostingClassifier(loss="log_loss", **params)
    unique_weights = _balanced_binary_weights(rows, train, unique)
    unique_model.fit(matrix[train], unique[train], sample_weight=unique_weights[train])
    dominated_train = train[valid[train] == 1]
    dominated_model = HistGradientBoostingClassifier(
        loss="log_loss", **{**MODEL_PARAMS, "random_state": seed + 1}
    )
    dominated_weights = _balanced_binary_weights(rows, dominated_train, dominated)
    dominated_model.fit(
        matrix[dominated_train], dominated[dominated_train],
        sample_weight=dominated_weights[dominated_train],
    )
    p_unique = unique_model.predict_proba(matrix[validation])[:, 1]
    p_dominated = dominated_model.predict_proba(matrix[validation])[:, 1]
    predictions = {}
    preserve_scores = []
    for local, index in enumerate(validation):
        row = rows[int(index)]
        p_valid25 = float(
            row["model_features"]["relation__candidate_nested_valid25__mean"]
        )
        preserve = protected_preserve_probability(
            p_valid25, p_unique[local], p_dominated[local]
        )
        predictions[(row["scene_name"], int(row["candidate_id"]))] = {
            "scene_name": row["scene_name"],
            "relation_component_id": int(row["relation_component_id"]),
            "track_id": int(row["candidate_id"]),
            "valid25_probability": p_valid25,
            "unique_valid25_probability": float(p_unique[local]),
            "native_dominated_probability": float(p_dominated[local]),
            "preserve_probability": float(preserve),
            "label_track_valid25": int(valid[index]),
            "label_unique_valid25": int(unique[index]),
            "label_native_dominated": int(dominated[index]),
            "label_safe_keep25": int(row["label_safe_keep25"]),
            "label_best_gt_iou": float(row["label_best_gt_iou"]),
        }
        preserve_scores.append(preserve)
    return predictions, {
        "train_track_count": len(train),
        "validation_track_count": len(validation),
        "dominated_training_valid_track_count": len(dominated_train),
        "unique_head": _metric(unique[validation], p_unique),
        "native_dominated_head": _metric(dominated[validation], p_dominated),
        "preserve_vs_safe_keep25": _metric(
            np.asarray([rows[int(index)]["label_safe_keep25"] for index in validation]),
            np.asarray(preserve_scores),
        ),
    }


def _evaluate_oof(args, scenes: list[str], manifest: dict, predictions: dict) -> tuple[dict, dict, list[dict]]:
    baseline_records = {}
    policy_records = {}
    score_rows = []
    _configure_scannet200_instance_eval()
    original_load_ids = instance_eval.util_3d.load_ids
    instance_eval.util_3d.load_ids = lambda path: _class_agnostic_gt_ids(original_load_ids(path))
    try:
        for scene_index, scene in enumerate(scenes, start=1):
            cache = _scene_inputs(scene, args)
            uuid_to_key = {uuid: key for key, uuid in cache["uuid_by_candidate"].items()}
            base_scores = {
                uuid_to_key[row["uuid"]]: float(row["confidence"])
                for row in cache["pred"]["chair"]
            }
            selected_scores = dict(base_scores)
            for key, original in base_scores.items():
                if key[0] != "track":
                    continue
                prediction = predictions.get((scene, int(key[1])))
                if prediction is None:
                    continue
                multiplier, selected = suppressed_track_score(
                    original, prediction["preserve_probability"]
                )
                selected_scores[key] = selected
                score_rows.append({
                    **prediction,
                    "frozen_coexist_score": float(original),
                    "suppression_multiplier": float(multiplier),
                    "selected_score": float(selected),
                    "candidate_retained": True,
                })
            baseline_uuid_scores = {
                cache["uuid_by_candidate"][key]: score for key, score in base_scores.items()
            }
            _set_match_scores({"scene": {"gt": cache["gt"], "pred": cache["pred"]}}, baseline_uuid_scores)
            baseline_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            policy_uuid_scores = {
                cache["uuid_by_candidate"][key]: score for key, score in selected_scores.items()
            }
            _set_match_scores({"scene": {"gt": cache["gt"], "pred": cache["pred"]}}, policy_uuid_scores)
            policy_records[scene] = _scene_records(
                cache, cache["all_native_representatives"], cache["all_track_ids"]
            )
            print(f"[union AP25 OOF AP] {scene_index}/100 {scene}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    overall = {
        "frozen_coexist": _metrics_from_records(baseline_records),
        POLICY: _metrics_from_records(policy_records),
    }
    folds = []
    for fold in manifest["folds"]:
        validation = set(fold["validation_scenes"])
        folds.append({
            "fold_index": int(fold["fold_index"]),
            "frozen_coexist": _metrics_from_records({
                scene: baseline_records[scene] for scene in validation
            }),
            POLICY: _metrics_from_records({
                scene: policy_records[scene] for scene in validation
            }),
        })
    return overall, {"folds": folds}, score_rows


def _triplet(metrics: dict) -> dict:
    return {
        "ap": float(metrics["official_ap"]),
        "ap50": float(metrics["threshold_metrics"]["50"]["ap"]),
        "ap25": float(metrics["threshold_metrics"]["25"]["ap"]),
    }


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
    assign_track_protection_labels(candidates)
    relation_rows = [
        row for scene in scenes
        for row in read_jsonl(args.relation_feature_ledger_root / scene / "relation_features.jsonl")
    ]
    relation_by_component = defaultdict(list)
    for row in relation_rows:
        relation_by_component[(str(row["scene_name"]), int(row["relation_component_id"]))].append(row)
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
        )
        stacked_by_component = defaultdict(list)
        for raw, stacked in zip(relation_rows, stacked_all):
            stacked_by_component[(str(raw["scene_name"]), int(raw["relation_component_id"]))].append(stacked)
        rows, matrix, names = _track_rows(
            candidates, relation_by_component, stacked_by_component,
            union_lookup, union_names,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("union-risk feature schema changed across folds")
        predictions, metrics = _fit_fold(
            rows, matrix, fold["train_scenes"], fold["validation_scenes"],
            RANDOM_SEED + fold_index * 100,
        )
        overlap = set(final_predictions) & set(predictions)
        if overlap:
            raise AssertionError("OOF union-risk predictions repeated")
        final_predictions.update(predictions)
        fold_details.append({
            "fold_index": fold_index,
            "stacked_evidence": stacked_diagnostics,
            "head_metrics": metrics,
        })
        print(f"[union AP25 head] outer fold {fold_index} complete", flush=True)
    expected_tracks = sum(row["candidate_source"] == TRACK_SOURCE for row in candidates)
    if len(final_predictions) != expected_tracks or len(final_predictions) != len(union_lookup):
        raise AssertionError("OOF union-risk prediction coverage mismatch")
    overall, fold_ap, score_rows = _evaluate_oof(args, scenes, manifest, final_predictions)
    baseline = _triplet(overall["frozen_coexist"])
    selected = _triplet(overall[POLICY])
    delta = {key: selected[key] - baseline[key] for key in baseline}
    fold_deltas = []
    for fold in fold_ap["folds"]:
        base = _triplet(fold["frozen_coexist"])
        current = _triplet(fold[POLICY])
        fold_deltas.append({
            "fold_index": int(fold["fold_index"]),
            "delta": {key: current[key] - base[key] for key in base},
        })
    label_counts = Counter()
    for row in candidates:
        if row["candidate_source"] == TRACK_SOURCE:
            for name in (
                "label_track_valid25", "label_unique_valid25",
                "label_native_dominated", "label_safe_keep25",
            ):
                label_counts[name] += int(row[name])
    summary = {
        "version": VERSION,
        "policy": POLICY,
        "scene_count": 100,
        "component_count": len(component_keys),
        "controlled_track_count": len(final_predictions),
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "component_union_feature_count": len(union_names),
            "component_union_feature_names": union_names,
            "ground_truth_fields_in_features": False,
            "candidate_quality_and_relation_evidence": "scene-disjoint OOF",
            "GVC_usage": "continuous feature only; never a hard gate",
        },
        "label_contract": {
            "valid25": "track best GT IoU > 0.25",
            "unique_valid25": "valid25 target has no native representative above 0.25",
            "native_dominated": "same-target native IoU >= track IoU and native IoU > 0.25",
            "safe_keep25": "valid25 and either unique or not native-dominated",
            "counts": dict(sorted(label_counts.items())),
        },
        "score_contract": {
            "preserve_probability": (
                "P(valid25) * (1 - P(native_dominated) * (1 - P(unique_valid25)))"
            ),
            "native_score": "bit-for-bit frozen",
            "track_score": "frozen_coexist_q * (1 - (1 - preserve_probability)^3)",
            "candidate_removed": False,
            "threshold_scanning": False,
            "continuous_weight_scanning": False,
        },
        "fold_details": fold_details,
        "metrics": {
            "frozen_coexist": baseline,
            POLICY: selected,
            "delta_vs_frozen_coexist": delta,
            "fold_deltas": fold_deltas,
        },
        "score_context": score_context,
        "candidate_count_modified": False,
        "candidate_geometry_modified": False,
        "candidate_class_modified": False,
        "candidate_files_modified": False,
        "safety60_read": False,
        "even48_read": False,
        "test60_read": False,
        "ground_truth_usage": "official100 nested OOF labels and offline AP only",
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
        _write_jsonl(staging / "oof_track_predictions.jsonl", [
            final_predictions[key] for key in sorted(final_predictions)
        ])
        _write_jsonl(staging / "oof_track_scores.jsonl", score_rows)
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
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--candidate-quality-protocol-name", required=True)
    parser.add_argument("--track-score-mode", choices=("oof_quality",), default="oof_quality")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics for official100 OOF evaluation")
    for name in (
        "scene_list", "split_manifest", "records_root", "candidate_records_root",
        "relation_feature_ledger_root", "action_ledger_root",
        "component_union_feature_ledger_root", "gt_dir", "oof_predictions", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

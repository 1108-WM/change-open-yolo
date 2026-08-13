#!/usr/bin/env python3
"""Evaluate a fixed applicability mixture of two frozen official100 OOF heads.

Tracks with no geometry outside their component native union use the incumbent
component-union prediction exactly.  Tracks with a non-empty residual use the
residual-augmented prediction.  Native predictions use the incumbent record.
The applicability decision is exact set emptiness, not a tuned threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


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
from tools.train_candidate_component_list_calibration_head_oof import (  # noqa: E402
    _evaluate_oof_scores,
    _triplet,
)
from tools.train_candidate_quality_head_oof import load_frozen_split_manifest  # noqa: E402


VERSION = "official100_component_union_residual_applicability_mixture_oof_v1"
PRIMARY_POLICY = "track_harm_focal_suppression"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _prediction_key(row: dict) -> tuple[str, str, int]:
    return str(row["scene_name"]), str(row["candidate_source"]), int(row["candidate_id"])


def compose_predictions(
    incumbent_rows: list[dict], residual_rows: list[dict], residual_empty: dict,
) -> tuple[list[dict], dict]:
    incumbent = {_prediction_key(row): row for row in incumbent_rows}
    residual = {_prediction_key(row): row for row in residual_rows}
    if len(incumbent) != len(incumbent_rows) or len(residual) != len(residual_rows):
        raise ValueError("duplicate OOF candidate predictions")
    if set(incumbent) != set(residual):
        raise ValueError("incumbent and residual OOF candidate coverage differs")
    output = []
    counts = {"incumbent_native": 0, "incumbent_empty_track": 0, "residual_nonempty_track": 0}
    for key in sorted(incumbent):
        old, new = incumbent[key], residual[key]
        for field in (
            "relation_component_id", "label_component_unique_winner",
            "label_calibrated_quality", "label_best_gt_iou", "label_harm_kind",
        ):
            if old[field] != new[field]:
                raise ValueError(f"OOF prediction contract differs for {key}: {field}")
        if key[1] == NATIVE_SOURCE:
            selected = old
            source = "incumbent_native"
        elif key[1] == TRACK_SOURCE:
            residual_key = (key[0], key[2])
            if residual_key not in residual_empty:
                raise ValueError(f"missing residual applicability feature: {residual_key}")
            if float(residual_empty[residual_key]) == 1.0:
                selected = old
                source = "incumbent_empty_track"
            elif float(residual_empty[residual_key]) == 0.0:
                selected = new
                source = "residual_nonempty_track"
            else:
                raise ValueError(f"non-binary residual_empty: {residual_key}")
        else:
            raise ValueError(f"unknown candidate source: {key[1]}")
        counts[source] += 1
        output.append({**selected, "applicability_prediction_source": source})
    if set(residual_empty) != {
        (key[0], key[2]) for key in incumbent if key[1] == TRACK_SOURCE
    }:
        raise ValueError("residual applicability coverage differs from OOF tracks")
    return output, counts


def run(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100 or _sha256(args.split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("must use frozen official100 five-fold split")
    manifest = load_frozen_split_manifest(args.split_manifest, scenes)
    score_context = configure_track_score_context(args, scenes)
    incumbent_rows = read_jsonl(args.incumbent_predictions)
    residual_rows = read_jsonl(args.residual_predictions)
    feature_rows = read_jsonl(
        args.component_union_residual_feature_ledger_root
        / "component_union_track_features.jsonl"
    )
    residual_empty = {
        (str(row["scene_name"]), int(row["track_id"])):
        float(row["model_features"]["residual_empty"])
        for row in feature_rows
    }
    composed_rows, selection_counts = compose_predictions(
        incumbent_rows, residual_rows, residual_empty,
    )
    predictions = {
        (row["scene_name"], row["candidate_source"], int(row["candidate_id"])): row
        for row in composed_rows
    }
    overall, fold_ap, score_rows = _evaluate_oof_scores(args, scenes, manifest, predictions)
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
        "controlled_candidate_count": len(composed_rows),
        "applicability_contract": {
            "native_prediction": "incumbent component-union OOF prediction",
            "empty_residual_track_prediction": "incumbent component-union OOF prediction",
            "nonempty_residual_track_prediction": "residual-augmented OOF prediction",
            "applicability_rule": "exact residual point count equals zero",
            "threshold_scanning": False,
            "ground_truth_in_applicability_rule": False,
            "GVC_hard_gate": False,
        },
        "selection_counts": selection_counts,
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
        "ground_truth_usage": "official100 offline AP only; applicability is no-GT",
        "input_provenance": {
            "scene_list_sha256": _sha256(args.scene_list),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "incumbent_predictions_sha256": _sha256(args.incumbent_predictions),
            "residual_predictions_sha256": _sha256(args.residual_predictions),
            "component_union_residual_summary_sha256": _sha256(
                args.component_union_residual_feature_ledger_root / "summary.json"
            ),
        },
    }
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "oof_candidate_predictions.jsonl", composed_rows)
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
    parser.add_argument("--relation-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--component-union-residual-feature-ledger-root", type=Path, required=True)
    parser.add_argument("--incumbent-predictions", type=Path, required=True)
    parser.add_argument("--residual-predictions", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--track-score-mode", choices=("oof_quality",), default="oof_quality")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "split_manifest", "records_root", "relation_feature_ledger_root",
        "component_union_residual_feature_ledger_root", "incumbent_predictions",
        "residual_predictions", "gt_dir", "oof_predictions", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    summary = run(args)
    print(json.dumps({
        "primary_metrics": summary["policy_metrics"][PRIMARY_POLICY],
        "primary_delta": summary["policy_delta_vs_frozen_coexist"][PRIMARY_POLICY],
        "primary_fold_deltas": summary["primary_fold_deltas"],
        "selection_counts": summary["selection_counts"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

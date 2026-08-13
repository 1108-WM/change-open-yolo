#!/usr/bin/env python3
"""Scene-disjoint, GT-only univariate feature feasibility audit for frozen N2.

No classifier, threshold, candidate mutation, or inference score is produced.
For every feature, direction is selected on four fifths of scenes and evaluated
on the held-out fifth only.  This tests whether a future no-GT selector is even
worth studying without silently training one.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    "GT-only scene-disjoint univariate N2 feature feasibility diagnostic; no classifier, "
    "threshold, score, candidate selection, materialization, or AP evaluation is produced."
)
FEATURES = (
    "point_count", "superpoint_count", "candidate_seed_scale_ratio", "eligible_view_count",
    "medoid_cross_view_mean_jaccard", "sam_predicted_iou", "source_backprojected_point_count",
    "source_visible_core_support_ratio", "source_observation_core_purity_ratio",
    "rgb_channel_std_normalized", "normal_channel_std", "normal_mean_resultant_length",
    "centroid_connectivity_proxy_component_count", "centroid_connectivity_proxy_largest_component_point_ratio",
    "near_duplicate_component_size", "best_d2b_jaccard", "best_d2b_candidate_covered_ratio",
    "best_d2b_track_covered_ratio", "best_native_iou", "best_native_candidate_covered_ratio",
    "best_native_covered_ratio",
    "dino_selected_view_count", "dino_medoid_to_view_cosine_mean",
    "dino_medoid_to_view_cosine_min", "dino_medoid_to_view_cosine_std",
)
LABELS = (
    "new_geometry_eligible_iou25", "new_geometry_eligible_iou50",
    "system_increment_oracle_iou25", "system_increment_oracle_iou50",
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def scene_list(path):
    rows = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or duplicated")
    return rows


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--dinov2-ledger-root", type=Path)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in ("scene_list", "quality_ledger_root", "dinov2_ledger_root", "oracle_root", "output_root"):
        if getattr(args, name) is None:
            continue
        setattr(args, name, resolve(getattr(args, name)))
    if args.fold_count < 2:
        raise SystemExit("fold count must be at least two")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("output root is non-empty")
    scenes = scene_list(args.scene_list)
    rows = []
    for scene_index, scene in enumerate(scenes):
        quality = {int(row["candidate_id"]): row for row in (json.loads(line) for line in (args.quality_ledger_root / scene / "candidate_quality_competition_ledger.jsonl").read_text().splitlines() if line)}
        dino = {} if args.dinov2_ledger_root is None else {int(row["candidate_id"]): row for row in (json.loads(line) for line in (args.dinov2_ledger_root / scene / "n2_medoid_dinov2_appearance_ledger.jsonl").read_text().splitlines() if line)}
        oracle = {int(row["candidate_id"]): row for row in (json.loads(line) for line in (args.oracle_root / scene / "candidate_oracle_gt.jsonl").read_text().splitlines() if line)}
        if set(quality) != set(oracle) or (args.dinov2_ledger_root is not None and set(dino) != set(oracle)):
            raise ValueError(f"{scene}: candidate IDs differ between frozen quality and oracle ledgers")
        for candidate_id in sorted(quality):
            quality_row, oracle_row = quality[candidate_id], oracle[candidate_id]
            row = {"scene_name": scene, "scene_fold": scene_index % args.fold_count, "candidate_id": candidate_id}
            for feature in FEATURES:
                value = quality_row.get(feature, dino.get(candidate_id, {}).get(feature))
                row[feature] = None if value is None else float(value)
            row["new_geometry_eligible_iou25"] = oracle_row["classification_iou25"] == "new_geometry_eligible"
            row["new_geometry_eligible_iou50"] = oracle_row["classification_iou50"] == "new_geometry_eligible"
            row["system_increment_oracle_iou25"] = bool(oracle_row["selected_by_component_system_oracle_iou25"])
            row["system_increment_oracle_iou50"] = bool(oracle_row["selected_by_component_system_oracle_iou50"])
            row["ground_truth_usage"] = "offline_diagnostic_only"
            row["proposal_materialization_applied"] = False
            rows.append(row)
    by_fold = defaultdict(list)
    for row in rows:
        by_fold[row["scene_fold"]].append(row)
    details, summaries = [], []
    for label in LABELS:
        for feature in FEATURES:
            held_out = []
            for fold in range(args.fold_count):
                train = [row for other, values in by_fold.items() if other != fold for row in values]
                test = by_fold[fold]
                train_auc = auc([row[feature] for row in train], [row[label] for row in train])
                natural_test_auc = auc([row[feature] for row in test], [row[label] for row in test])
                direction = None if train_auc is None else ("higher_is_positive" if train_auc >= .5 else "lower_is_positive")
                held_out_auc = None if natural_test_auc is None or direction is None else (natural_test_auc if direction == "higher_is_positive" else 1.0 - natural_test_auc)
                held_out.append(held_out_auc)
                details.append({
                    "label": label, "feature": feature, "held_out_fold": fold, "train_scene_count": len(scenes) - len([s for s in scenes if scenes.index(s) % args.fold_count == fold]),
                    "test_scene_count": len([s for s in scenes if scenes.index(s) % args.fold_count == fold]),
                    "train_positive_count": int(sum(row[label] for row in train)), "test_positive_count": int(sum(row[label] for row in test)),
                    "train_auc_natural": train_auc, "held_out_auc_natural": natural_test_auc,
                    "training_only_direction": direction, "held_out_auc_oriented_by_training": held_out_auc,
                    "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
                })
            valid = [value for value in held_out if value is not None]
            summaries.append({
                "label": label, "feature": feature, "fold_count_with_valid_auc": len(valid),
                "held_out_auc_oriented_mean": float(np.mean(valid)) if valid else None,
                "held_out_auc_oriented_min": float(np.min(valid)) if valid else None,
                "held_out_auc_oriented_max": float(np.max(valid)) if valid else None,
                "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False,
            })
    args.output_root.mkdir(parents=True)
    with (args.output_root / "candidate_feature_oracle_join_gt.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_root / "feature_fold_metrics_gt.jsonl").open("w") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_root / "feature_summary_gt.jsonl").open("w") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    root = {
        "diagnostic_type": "GT-only scene-disjoint univariate N2 feature feasibility audit",
        "decision_constraint": CONTRACT,
        "scene_count": len(scenes), "candidate_count": len(rows), "fold_count": args.fold_count,
        "label_positive_counts": {label: int(sum(row[label] for row in rows)) for label in LABELS},
        "proposal_materialization_applied": False, "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a lightweight geometry-quality discriminator for candidate proposals.

The script uses offline candidate diagnostics as supervision. It intentionally
keeps GT-derived columns out of the feature matrix so the learned model can be
used as a practical candidate-quality scorer later.
"""

import argparse
import csv
import json
import math
import os
import os.path as osp
import pickle
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_NUMERIC_FEATURES = [
    "best_existing_iou",
    "seed_in_existing_mask_ratio",
    "score",
    "fusion_score",
    "quality_score",
    "support_view_count",
    "support_mean_iou",
    "support_best_iou",
    "consistency_rate",
    "box_area_ratio",
    "num_seed_points",
    "num_mask_points",
    "superpoint_expansion_ratio",
    "geometry_component_count",
    "geometry_largest_component_ratio",
    "geometry_non_largest_component_ratio",
    "geometry_small_component_ratio",
    "geometry_extent_x",
    "geometry_extent_y",
    "geometry_extent_z",
    "geometry_extent_max",
    "geometry_extent_mid",
    "geometry_extent_min",
    "geometry_aspect_max_mid",
    "geometry_aspect_mid_min",
    "geometry_aspect_max_min",
    "geometry_bbox_volume",
    "geometry_bbox_density",
    "geometry_pca_linearity",
    "geometry_pca_planarity",
    "geometry_pca_scattering",
    "geometry_plane_inlier_ratio",
    "geometry_plane_residual_mean",
    "geometry_plane_residual_p90",
    "depth_valid_projection_ratio",
    "depth_consistency_ratio",
    "depth_front_gap_ratio",
    "depth_behind_surface_ratio",
    "depth_residual_mean",
    "depth_residual_p50",
    "depth_residual_p90",
    "report_mask_support_mean_positive_ratio",
    "report_mask_support_mean_negative_ratio",
    "report_mask_support_filtered_segments",
    "report_mask_support_filtered_ratio",
    "report_mask_support_usable_view_count",
    "report_cc_component_count",
    "report_cc_largest_component_ratio",
    "report_cc_keep_ratio",
    "relation_base_contained_count",
    "relation_base_max_coverage",
    "relation_base_max_area_ratio",
    "relation_candidate_contained_count",
    "relation_candidate_max_coverage",
    "relation_candidate_max_area_ratio",
    "relation_any_contained_count",
    "relation_any_max_coverage",
    "relation_exclusive_point_ratio",
    "relation_exclusive_point_count",
    "relation_contained_point_count",
    "hierarchy_superpoint_count",
    "hierarchy_mean_superpoint_occupancy",
    "hierarchy_min_superpoint_occupancy",
    "hierarchy_max_superpoint_occupancy",
    "hierarchy_low_occupancy_mass_ratio",
    "hierarchy_base_parent_count",
    "hierarchy_candidate_parent_count",
    "hierarchy_parent_count",
    "hierarchy_parent_max_candidate_coverage",
    "hierarchy_parent_max_weighted_jaccard",
    "hierarchy_parent_min_extra_mass_ratio",
    "hierarchy_base_child_count",
    "hierarchy_candidate_child_count",
    "hierarchy_child_count",
    "hierarchy_child_max_coverage",
    "hierarchy_child_max_weighted_jaccard",
    "hierarchy_child_max_area_ratio",
    "hierarchy_child_union_coverage",
    "hierarchy_exclusive_superpoint_ratio",
    "hierarchy_exclusive_superpoint_count",
    "hierarchy_any_related_count",
    "scene_source_quality_z",
    "class_source_quality_z",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "source_kind",
    "superpoint_refined",
    "report_mask_support_enabled",
    "report_mask_support_mode",
]

GT_OR_LABEL_COLUMNS = {
    "best_same_class_gt_iou",
    "best_any_gt_iou",
    "best_gt_class",
    "best_gt_instance_id",
    "best_baseline_gt_iou",
    "diagnostic_label",
    "semantic_id",
}


def _parse_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_diagnostic_csv"] = path
                rows.append(row)
    return rows


def _safe_float(value):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(output) or math.isinf(output):
        return 0.0
    return output


def _build_feature_names(rows, numeric_features, categorical_features):
    feature_names = []
    for name in numeric_features:
        if name in GT_OR_LABEL_COLUMNS:
            raise ValueError(f"Refusing to use GT/label column as a feature: {name}")
        feature_names.append(name)

    categorical_values = {}
    for name in categorical_features:
        if name in GT_OR_LABEL_COLUMNS:
            raise ValueError(f"Refusing to use GT/label column as a feature: {name}")
        values = sorted({str(row.get(name, "")) for row in rows})
        categorical_values[name] = values
        feature_names.extend(f"{name}={value}" for value in values)
    return feature_names, categorical_values


def _vectorize(rows, numeric_features, categorical_values):
    matrix = []
    for row in rows:
        values = [_safe_float(row.get(name)) for name in numeric_features]
        for name, categories in categorical_values.items():
            value = str(row.get(name, ""))
            values.extend(1.0 if value == category else 0.0 for category in categories)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


def _labels(rows, positive_labels, negative_labels):
    kept_rows = []
    labels = []
    positive = set(positive_labels)
    negative = set(negative_labels)
    for row in rows:
        label = row.get("diagnostic_label")
        if label in positive:
            kept_rows.append(row)
            labels.append(1)
        elif label in negative:
            kept_rows.append(row)
            labels.append(0)
    return kept_rows, np.asarray(labels, dtype=np.int64)


def _threshold_table(y_true, scores, min_recalls):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    rows = []
    threshold_items = list(thresholds) + [1.0]
    for min_recall in min_recalls:
        candidates = []
        for index, threshold in enumerate(threshold_items):
            pred = scores >= float(threshold)
            true_positive = int(np.logical_and(pred, y_true == 1).sum())
            false_positive = int(np.logical_and(pred, y_true == 0).sum())
            false_negative = int(np.logical_and(~pred, y_true == 1).sum())
            pred_count = int(pred.sum())
            actual_recall = true_positive / max(1, true_positive + false_negative)
            actual_precision = true_positive / max(1, true_positive + false_positive)
            keep_ratio = pred_count / max(1, len(y_true))
            if actual_recall >= float(min_recall):
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "precision": float(actual_precision),
                        "recall": float(actual_recall),
                        "keep_ratio": float(keep_ratio),
                        "kept": pred_count,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                    }
                )
        if candidates:
            best = max(candidates, key=lambda item: (item["precision"], -item["keep_ratio"]))
            best["target_min_recall"] = float(min_recall)
            rows.append(best)
    return rows


def _feature_importance(model, feature_names):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(-np.asarray(importances))[:25]
    return [
        {"feature": feature_names[int(index)], "importance": float(importances[int(index)])}
        for index in order
    ]


def _make_model(kind, random_state):
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics_csv", required=True, nargs="+")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "gradient_boosting"])
    parser.add_argument(
        "--positive_labels",
        default="true_completion_50,partial_completion_25",
        help="Comma-separated diagnostic labels treated as useful candidates.",
    )
    parser.add_argument(
        "--negative_labels",
        default="background_or_bad_geometry,class_error_or_cross_class_overlap,existing_overlap_low_gt",
        help="Comma-separated diagnostic labels treated as bad candidates.",
    )
    parser.add_argument("--numeric_features", default=",".join(DEFAULT_NUMERIC_FEATURES))
    parser.add_argument("--categorical_features", default=",".join(DEFAULT_CATEGORICAL_FEATURES))
    parser.add_argument("--test_size", default=0.30, type=float)
    parser.add_argument("--random_state", default=13, type=int)
    parser.add_argument("--min_recalls", default="0.95,0.90,0.80,0.70")
    parser.add_argument(
        "--export_fit_all",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fit the saved model on all labeled rows after split-based validation.",
    )
    args = parser.parse_args()

    rows = _read_rows(args.diagnostics_csv)
    positive_labels = _parse_list(args.positive_labels)
    negative_labels = _parse_list(args.negative_labels)
    rows, y = _labels(rows, positive_labels, negative_labels)
    if len(rows) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Need both positive and negative labeled rows after filtering.")

    numeric_features = _parse_list(args.numeric_features)
    categorical_features = _parse_list(args.categorical_features)
    feature_names, categorical_values = _build_feature_names(rows, numeric_features, categorical_features)
    x = _vectorize(rows, numeric_features, categorical_values)
    groups = np.asarray([row.get("scene_name", "") for row in rows])

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    validation_model = _make_model(args.model, args.random_state)
    validation_model.fit(x[train_idx], y[train_idx])
    scores = validation_model.predict_proba(x[test_idx])[:, 1]
    if args.export_fit_all:
        export_model = _make_model(args.model, args.random_state)
        export_model.fit(x, y)
    else:
        export_model = validation_model

    roc_auc = None
    if len(np.unique(y[test_idx])) > 1:
        roc_auc = float(roc_auc_score(y[test_idx], scores))
    avg_precision = float(average_precision_score(y[test_idx], scores))
    min_recalls = [_safe_float(item) for item in _parse_list(args.min_recalls)]
    thresholds = _threshold_table(y[test_idx], scores, min_recalls)

    output = {
        "model": args.model,
        "diagnostics_csv": args.diagnostics_csv,
        "num_rows": int(len(rows)),
        "num_features": int(x.shape[1]),
        "label_counts": dict(Counter(row["diagnostic_label"] for row in rows)),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_scenes": int(len(set(groups[train_idx]))),
        "test_scenes": int(len(set(groups[test_idx]))),
        "export_fit_all": bool(args.export_fit_all),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "thresholds": thresholds,
        "feature_importance": _feature_importance(validation_model, feature_names),
        "feature_names": feature_names,
        "categorical_values": categorical_values,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(output, f, indent=2)
    with open(osp.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(
            {
                "model": export_model,
                "model_kind": args.model,
                "schema_version": 1,
                "export_fit_all": bool(args.export_fit_all),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "categorical_values": categorical_values,
                "feature_names": feature_names,
                "positive_labels": positive_labels,
                "negative_labels": negative_labels,
            },
            f,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

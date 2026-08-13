#!/usr/bin/env python3
"""Scene-disjoint GT-only audit of public-view relative track--native GVC.

Only tracks with an actual overlapping native and public (source-frame-excluded)
common views enter relative-feature AUCs.  Tracks without such a comparison are
reported separately and never receive an invented native value.  Feature
orientation is chosen on four scene folds; both ordinary and |global ΔAP|
weighted ranking consistency are reported on the held-out fold.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-10
FEATURES = (
    "track_public_common_gvc_mean", "max_overlapping_native_public_gvc_mean",
    "track_minus_max_native_gvc", "track_over_max_native_gvc_ratio",
    "track_minus_native_box_iou", "track_minus_native_mask_support",
    "public_common_view_count", "point_iou", "track_inside_native_ratio",
    "native_inside_track_ratio", "component_track_gvc_rank_fraction",
    "component_relative_gvc_rank_fraction",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _scenes(path: Path) -> list[str]:
    result = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("场景列表为空或含重复")
    return result


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _auc(scores, labels, weights=None):
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if weights is None:
        weights = np.ones(len(values), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, target, weights = values[valid], target[valid], weights[valid]
    pos, neg = target, ~target
    if not pos.any() or not neg.any():
        return None
    if weights is None:
        return None
    # Small fixed diagnostic populations; pairwise weighted concordance is exact and transparent.
    numerator = 0.0
    for score, weight in zip(values[pos], weights[pos]):
        numerator += weight * np.sum(weights[neg] * ((score > values[neg]) + .5 * (score == values[neg])))
    return float(numerator / (weights[pos].sum() * weights[neg].sum()))


def _average_precision(scores, labels):
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    valid = np.isfinite(values)
    values, target = values[valid], target[valid]
    if not target.any():
        return None
    order = np.argsort(-values, kind="mergesort")
    ordered = target[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / ordered.sum())


def _coverage(scores, labels, orientation, fraction):
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    if not len(values):
        return None
    count = max(1, int(np.ceil(len(values) * fraction)))
    order = np.argsort(-values if orientation == "higher_is_necessary" else values, kind="mergesort")
    necessary, suppress = labels == "necessary_keep", labels == "beneficial_suppress"
    return {
        "candidate_count": count,
        "top_necessary_keep_coverage": float(necessary[order[:count]].sum() / max(1, necessary.sum())),
        "bottom_beneficial_suppress_coverage": float(suppress[order[-count:]].sum() / max(1, suppress.sum())),
    }


def _metrics(train, test, feature):
    train_auc = _auc([row[feature] for row in train], [row["binary_target"] for row in train])
    direction = None if train_auc is None else ("higher_is_necessary" if train_auc >= .5 else "lower_is_necessary")
    sign = 1.0 if direction == "higher_is_necessary" else -1.0
    values = [
        sign * float(row[feature]) if row[feature] is not None else float("nan")
        for row in test
    ] if direction else []
    labels = [row["binary_target"] for row in test]
    weights = [row["abs_global_ap_margin"] for row in test]
    return {
        "training_only_direction": direction,
        "train_auc_natural": train_auc,
        "held_out_auc_oriented": _auc(values, labels) if direction else None,
        "held_out_weighted_auc_oriented": _auc(values, labels, weights) if direction else None,
        "held_out_pr_auc_oriented": _average_precision(values, labels) if direction else None,
        "coverage_10": _coverage(values, [row["track_label"] for row in test], "higher_is_necessary", .10) if direction else None,
        "coverage_20": _coverage(values, [row["track_label"] for row in test], "higher_is_necessary", .20) if direction else None,
    }


def _rank_fraction(rows, field):
    """One is highest; ties use average rank.  Only comparator-available rows participate."""
    grouped = defaultdict(list)
    for row in rows:
        if row["comparison_state"] == "relative_public_evidence_available":
            grouped[(row["scene_name"], row["component_id"])].append(row)
    for values in grouped.values():
        order = sorted(values, key=lambda row: -row[field])
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and order[end][field] == order[cursor][field]:
                end += 1
            value = 1.0 - ((cursor + 1 + end) / 2.0 - 1.0) / max(1, len(order) - 1)
            for row in order[cursor:end]:
                row[f"component_{field}_rank_fraction"] = value
            cursor = end
    for row in rows:
        row.setdefault(f"component_{field}_rank_fraction", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relative-ledger-root", type=Path, required=True)
    parser.add_argument("--action-margin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in ("scene_list", "relative_ledger_root", "action_margin_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.fold_count < 2:
        raise SystemExit("--fold-count 必须至少为 2")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    rows = []
    for scene_index, scene in enumerate(_scenes(args.scene_list)):
        relative = {int(row["track_id"]): row for row in _jsonl(args.relative_ledger_root / scene / "track_relative_gvc.jsonl")}
        labels = {int(row["track_id"]): row for row in _jsonl(args.action_margin_root / scene / "track_action_labels.jsonl")}
        if set(relative) != set(labels):
            raise ValueError(f"{scene}: relative GVC 与动作标签不一致")
        pair_rows = _jsonl(args.relative_ledger_root / scene / "pair_relative_gvc.jsonl")
        pair_by_track = defaultdict(list)
        for pair in pair_rows:
            pair_by_track[int(pair["track_id"])].append(pair)
        for track_id in sorted(relative):
            item, label = relative[track_id], labels[track_id]
            pair = item["public_comparison_pair"]
            if pair is None:
                values = {feature: None for feature in FEATURES}
            else:
                pairs = [row for row in pair_by_track[track_id] if row["selected_public_common_view_count"] > 0]
                relation = next(row for row in pairs if int(row["native_candidate_id"]) == int(item["public_comparison_native_candidate_id"]))
                values = {
                    "track_public_common_gvc_mean": float(pair["track_gvc_public_common"]["mean"]),
                    "max_overlapping_native_public_gvc_mean": float(pair["native_gvc_public_common"]["mean"]),
                    "track_minus_max_native_gvc": float(pair["track_minus_native_gvc"]),
                    "track_over_max_native_gvc_ratio": pair["track_over_native_gvc_ratio"],
                    "track_minus_native_box_iou": float(pair["track_minus_native_box_iou"]),
                    "track_minus_native_mask_support": float(pair["track_minus_native_mask_support"]),
                    "public_common_view_count": float(pair["selected_public_common_view_count"]),
                    "point_iou": float(relation["point_iou"]),
                    "track_inside_native_ratio": float(relation["track_inside_native_ratio"]),
                    "native_inside_track_ratio": float(relation["native_inside_track_ratio"]),
                    "component_track_gvc_rank_fraction": None,
                    "component_relative_gvc_rank_fraction": None,
                }
            rows.append({
                "scene_name": scene, "scene_fold": scene_index % args.fold_count, "track_id": track_id,
                "component_id": int(item["component_id"]), "comparison_state": item["comparison_state"],
                "track_label": label["label"], "binary_target": label["label"] == "necessary_keep",
                "strict_binary_example": label["label"] in ("necessary_keep", "beneficial_suppress"),
                "abs_global_ap_margin": abs(float(label["counterfactual_best_delta"] or 0.0)),
                **values,
            })
    # Rankings are deliberately computed only within the fixed relation component and public-evidence subset.
    _rank_fraction(rows, "track_public_common_gvc_mean")
    _rank_fraction(rows, "track_minus_max_native_gvc")
    for row in rows:
        row["component_track_gvc_rank_fraction"] = row.pop("component_track_public_common_gvc_mean_rank_fraction")
        row["component_relative_gvc_rank_fraction"] = row.pop("component_track_minus_max_native_gvc_rank_fraction")
    strict = [row for row in rows if row["strict_binary_example"] and row["comparison_state"] == "relative_public_evidence_available"]
    by_fold = defaultdict(list)
    for row in strict:
        by_fold[row["scene_fold"]].append(row)
    metrics = []
    for fold in range(args.fold_count):
        train = [row for other, values in by_fold.items() if other != fold for row in values]
        test = by_fold[fold]
        for feature in FEATURES:
            metrics.append({"feature": feature, "held_out_fold": fold, "train_count": len(train), "test_count": len(test), **_metrics(train, test, feature)})
    summary_rows = []
    for feature in FEATURES:
        values = [row for row in metrics if row["feature"] == feature]
        for metric_name in ("held_out_auc_oriented", "held_out_weighted_auc_oriented", "held_out_pr_auc_oriented"):
            valid = [row[metric_name] for row in values if row[metric_name] is not None]
            summary_rows.append({
                "feature": feature, "metric": metric_name, "valid_fold_count": len(valid),
                "mean": float(np.mean(valid)) if valid else None, "min": float(np.min(valid)) if valid else None,
            })
    state_counts = Counter(row["comparison_state"] for row in rows)
    state_label_counts = {state: dict(sorted(Counter(row["track_label"] for row in rows if row["comparison_state"] == state).items())) for state in state_counts}
    args.output_root.mkdir(parents=True)
    for name, payload in (
        ("track_relative_gvc_action_join_gt.jsonl", rows),
        ("relative_feature_fold_metrics_gt.jsonl", metrics),
        ("relative_feature_summary_gt.jsonl", summary_rows),
    ):
        (args.output_root / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload))
    root = {
        "diagnostic_type": "GT-only scene-disjoint public-view relative track-native GVC separability audit",
        "decision_constraint": "只在训练折选特征方向；不训练、不设阈值、不重排、不做 NMS、物化或 AP。",
        "ground_truth_usage": "offline diagnostic only", "proposal_materialization_applied": False,
        "scene_count": len(_scenes(args.scene_list)), "fold_count": args.fold_count,
        "track_count": len(rows), "strict_public_relative_binary_count": len(strict),
        "comparison_state_counts": dict(sorted(state_counts.items())), "comparison_state_label_counts": state_label_counts,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

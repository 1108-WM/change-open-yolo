#!/usr/bin/env python3
"""Scene-disjoint GT-only feasibility audit of C1 GVC evidence.

This joins the frozen no-GT GVC ledger with C1c single-component global-AP
action margins.  It selects only the orientation of each scalar feature on
four scene folds, then reports ROC-AUC, PR-AUC and rank coverage on the held
out fold.  It neither learns a model nor produces a threshold, score rewrite,
NMS decision, candidate mutation or AP result.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-10
TRACK_FEATURES = (
    "gvc_excluded_mean", "gvc_excluded_min", "gvc_excluded_max", "gvc_excluded_variance",
    "gvc_including_mean", "gvc_including_min", "gvc_including_max", "gvc_including_variance",
    "independent_eligible_view_count", "independent_selected_view_count",
    "independent_matched_selected_view_count", "independent_zero_support_selected_view_fraction",
    "independent_box_iou_mean", "independent_mask_point_support_mean",
    "independent_visible_point_fraction_mean", "missing_independent_evidence",
    "point_fraction_of_scene", "original_source_score",
)
COMPONENT_FEATURES = (
    "max_gvc_excluded_mean", "mean_gvc_excluded_mean",
    "max_gvc_including_mean", "mean_gvc_including_mean",
    "max_original_source_score", "mean_original_source_score",
    "missing_independent_track_fraction",
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复")
    return scenes


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _json_array(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path}: 应为 JSON 数组")
    return rows


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    valid = np.isfinite(values)
    values, target = values[valid], target[valid]
    positive, negative = int(target.sum()), int((~target).sum())
    if not positive or not negative:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2.0
        left = right
    return float((ranks[target].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def _average_precision(scores: list[float], labels: list[bool]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    valid = np.isfinite(values)
    values, target = values[valid], target[valid]
    positive = int(target.sum())
    if not positive:
        return None
    order = np.argsort(-values, kind="mergesort")
    ordered = target[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / positive)


def _coverage(scores: list[float], labels: list[str], orientation: str, fraction: float) -> dict:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    ordered = np.argsort(-np.asarray(scores) if orientation == "higher_is_necessary" else np.asarray(scores), kind="mergesort")
    top = ordered[:count]
    bottom = ordered[-count:]
    labels_np = np.asarray(labels, dtype=object)
    necessary = labels_np == "necessary_keep"
    suppress = labels_np == "beneficial_suppress"
    return {
        "fraction": fraction,
        "candidate_count": count,
        "top_necessary_keep_coverage": float(necessary[top].sum() / max(1, necessary.sum())),
        "bottom_beneficial_suppress_coverage": float(suppress[bottom].sum() / max(1, suppress.sum())),
    }


def _direction(train: list[dict], feature: str, target_key: str = "binary_target") -> tuple[str | None, float | None]:
    natural = _auc([row[feature] for row in train], [row[target_key] for row in train])
    if natural is None:
        return None, None
    return ("higher_is_necessary" if natural >= 0.5 else "lower_is_necessary"), natural


def _metric_row(train: list[dict], test: list[dict], feature: str, target_key: str = "binary_target") -> dict:
    direction, train_auc = _direction(train, feature, target_key)
    natural_auc = _auc([row[feature] for row in test], [row[target_key] for row in test])
    natural_pr = _average_precision([row[feature] for row in test], [row[target_key] for row in test])
    sign = 1.0 if direction == "higher_is_necessary" else -1.0
    oriented = [sign * row[feature] for row in test] if direction else []
    return {
        "training_only_direction": direction,
        "train_auc_natural": train_auc,
        "held_out_auc_natural": natural_auc,
        "held_out_auc_oriented": _auc(oriented, [row[target_key] for row in test]) if direction else None,
        "held_out_pr_auc_oriented": _average_precision(oriented, [row[target_key] for row in test]) if direction else None,
        "coverage": _coverage(oriented, [row["track_label"] for row in test], "higher_is_necessary", .10) if direction else None,
        "coverage_20": _coverage(oriented, [row["track_label"] for row in test], "higher_is_necessary", .20) if direction else None,
    }


def _ecdf(train_values: list[float], values: list[float]) -> np.ndarray:
    reference = np.sort(np.asarray(train_values, dtype=np.float64))
    return np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right") / max(1, len(reference))


def _combined_metric(train: list[dict], test: list[dict]) -> dict:
    first, second = "gvc_excluded_mean", "original_source_score"
    first_direction, first_train_auc = _direction(train, first)
    second_direction, second_train_auc = _direction(train, second)
    if first_direction is None or second_direction is None:
        return {"training_only_direction": None}
    train_first = _ecdf([row[first] for row in train], [row[first] for row in train])
    train_second = _ecdf([row[second] for row in train], [row[second] for row in train])
    test_first = _ecdf([row[first] for row in train], [row[first] for row in test])
    test_second = _ecdf([row[second] for row in train], [row[second] for row in test])
    if first_direction == "lower_is_necessary":
        train_first, test_first = 1.0 - train_first, 1.0 - test_first
    if second_direction == "lower_is_necessary":
        train_second, test_second = 1.0 - train_second, 1.0 - test_second
    del train_first, train_second
    combined = .5 * test_first + .5 * test_second  # pre-registered equal weights; never fitted.
    return {
        "training_only_direction": f"equal_rank_average({first_direction},{second_direction})",
        "train_auc_natural": {first: first_train_auc, second: second_train_auc},
        "held_out_auc_oriented": _auc(combined.tolist(), [row["binary_target"] for row in test]),
        "held_out_pr_auc_oriented": _average_precision(combined.tolist(), [row["binary_target"] for row in test]),
        "coverage": _coverage(combined.tolist(), [row["track_label"] for row in test], "higher_is_necessary", .10),
        "coverage_20": _coverage(combined.tolist(), [row["track_label"] for row in test], "higher_is_necessary", .20),
    }


def _feature_values(gvc: dict) -> dict[str, float]:
    excluded = gvc["gvc_source_frame_excluded"]
    including = gvc["gvc_including_source_frames"]
    return {
        "gvc_excluded_mean": float(excluded["gvc"]["mean"]),
        "gvc_excluded_min": float(excluded["gvc"]["min"]),
        "gvc_excluded_max": float(excluded["gvc"]["max"]),
        "gvc_excluded_variance": float(excluded["gvc"]["variance"]),
        "gvc_including_mean": float(including["gvc"]["mean"]),
        "gvc_including_min": float(including["gvc"]["min"]),
        "gvc_including_max": float(including["gvc"]["max"]),
        "gvc_including_variance": float(including["gvc"]["variance"]),
        "independent_eligible_view_count": float(excluded["eligible_depth_consistent_view_count"]),
        "independent_selected_view_count": float(excluded["selected_view_count"]),
        "independent_matched_selected_view_count": float(excluded["matched_selected_view_count"]),
        "independent_zero_support_selected_view_fraction": float(excluded["zero_support_selected_view_fraction"]),
        "independent_box_iou_mean": float(excluded["projected_box_iou"]["mean"]),
        "independent_mask_point_support_mean": float(excluded["visible_point_mask_support"]["mean"]),
        "independent_visible_point_fraction_mean": float(excluded["selected_visible_point_count"]["mean"] / max(1, gvc["point_count"])),
        "missing_independent_evidence": float(excluded["selected_view_count"] == 0),
        "point_fraction_of_scene": float(gvc["point_fraction_of_scene"]),
        "original_source_score": float(gvc["original_source_score"]),
    }


def _summarize(metrics: list[dict], name: str) -> list[dict]:
    output = []
    for feature in sorted({row["feature"] for row in metrics}):
        rows = [row for row in metrics if row["feature"] == feature]
        aucs = [row["held_out_auc_oriented"] for row in rows if row.get("held_out_auc_oriented") is not None]
        prs = [row["held_out_pr_auc_oriented"] for row in rows if row.get("held_out_pr_auc_oriented") is not None]
        output.append({
            "audit": name, "feature": feature,
            "valid_fold_count": len(aucs),
            "held_out_auc_mean": float(np.mean(aucs)) if aucs else None,
            "held_out_auc_min": float(np.min(aucs)) if aucs else None,
            "held_out_pr_auc_mean": float(np.mean(prs)) if prs else None,
            "held_out_pr_auc_min": float(np.min(prs)) if prs else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--gvc-ledger-root", type=Path, required=True)
    parser.add_argument("--action-margin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required")
    for name in ("scene_list", "gvc_ledger_root", "action_margin_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.fold_count < 2:
        raise SystemExit("--fold-count 必须至少为 2")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    track_rows, component_rows = [], []
    for scene_index, scene in enumerate(scenes):
        fold = scene_index % args.fold_count
        gvc = {
            int(row["candidate_id"]): row for row in _json_array(
                args.gvc_ledger_root / scene / "c1_gvc_quality_ledger.json"
            ) if row["candidate_source"] == "d2b_track"
        }
        labels = {
            int(row["track_id"]): row for row in _jsonl(
                args.action_margin_root / scene / "track_action_labels.jsonl"
            )
        }
        if set(gvc) != set(labels):
            raise ValueError(f"{scene}: GVC track 与动作标签不一致")
        by_component: dict[int, list[dict]] = defaultdict(list)
        for track_id in sorted(gvc):
            label = labels[track_id]["label"]
            row = {
                "scene_name": scene, "scene_fold": fold, "track_id": track_id,
                "component_id": int(labels[track_id]["component_id"]), "track_label": label,
                "binary_target": label == "necessary_keep",
                "strict_binary_example": label in ("necessary_keep", "beneficial_suppress"),
                **_feature_values(gvc[track_id]),
            }
            track_rows.append(row)
            by_component[row["component_id"]].append(row)
        margin_rows = _jsonl(args.action_margin_root / scene / "component_action_margins.jsonl")
        margin_by_component: dict[int, list[dict]] = defaultdict(list)
        for row in margin_rows:
            margin_by_component[int(row["component_id"])].append(row)
        for component_id, rows in margin_by_component.items():
            coexist = next((row for row in rows if row["action_kind"] == "coexist"), None)
            native_only = next((row for row in rows if row["action_kind"] == "native_only"), None)
            if coexist is None or native_only is None:
                continue
            delta = float(coexist["global_official_ap_after_action"] - native_only["global_official_ap_after_action"])
            members = by_component[component_id]
            if not members:
                raise ValueError(f"{scene}/{component_id}: 缺少组件轨迹 GVC")
            component_rows.append({
                "scene_name": scene, "scene_fold": fold, "component_id": component_id,
                "coexist_minus_native_only_global_ap": delta,
                "component_label": "coexist_preferred" if delta > EPS else ("native_only_preferred" if delta < -EPS else "neutral"),
                "binary_target": delta > EPS,
                "strict_binary_example": abs(delta) > EPS,
                "track_label": "necessary_keep" if delta > EPS else "beneficial_suppress",
                "max_gvc_excluded_mean": max(row["gvc_excluded_mean"] for row in members),
                "mean_gvc_excluded_mean": float(np.mean([row["gvc_excluded_mean"] for row in members])),
                "max_gvc_including_mean": max(row["gvc_including_mean"] for row in members),
                "mean_gvc_including_mean": float(np.mean([row["gvc_including_mean"] for row in members])),
                "max_original_source_score": max(row["original_source_score"] for row in members),
                "mean_original_source_score": float(np.mean([row["original_source_score"] for row in members])),
                "missing_independent_track_fraction": float(np.mean([row["missing_independent_evidence"] for row in members])),
            })
    strict_tracks = [row for row in track_rows if row["strict_binary_example"]]
    strict_components = [row for row in component_rows if row["strict_binary_example"]]
    by_fold_tracks, by_fold_components = defaultdict(list), defaultdict(list)
    for row in strict_tracks:
        by_fold_tracks[row["scene_fold"]].append(row)
    for row in strict_components:
        by_fold_components[row["scene_fold"]].append(row)
    track_metrics, component_metrics = [], []
    for fold in range(args.fold_count):
        train_t = [row for other, rows in by_fold_tracks.items() if other != fold for row in rows]
        test_t = by_fold_tracks[fold]
        for feature in TRACK_FEATURES:
            metric = _metric_row(train_t, test_t, feature)
            track_metrics.append({"feature": feature, "held_out_fold": fold, "train_count": len(train_t), "test_count": len(test_t), **metric})
        combined = _combined_metric(train_t, test_t)
        track_metrics.append({"feature": "equal_rank_average_excluded_gvc_and_original_score", "held_out_fold": fold, "train_count": len(train_t), "test_count": len(test_t), **combined})
        train_c = [row for other, rows in by_fold_components.items() if other != fold for row in rows]
        test_c = by_fold_components[fold]
        for feature in COMPONENT_FEATURES:
            metric = _metric_row(train_c, test_c, feature)
            component_metrics.append({"feature": feature, "held_out_fold": fold, "train_count": len(train_c), "test_count": len(test_c), **metric})
    missing = [row for row in track_rows if row["missing_independent_evidence"] > .5]
    missing_counts = Counter(row["track_label"] for row in missing)
    args.output_root.mkdir(parents=True)
    for filename, rows in (
        ("track_gvc_action_join_gt.jsonl", track_rows),
        ("track_feature_fold_metrics_gt.jsonl", track_metrics),
        ("track_feature_summary_gt.jsonl", _summarize(track_metrics, "track")),
        ("component_native_vs_coexist_join_gt.jsonl", component_rows),
        ("component_feature_fold_metrics_gt.jsonl", component_metrics),
        ("component_feature_summary_gt.jsonl", _summarize(component_metrics, "native_only_vs_coexist")),
    ):
        (args.output_root / filename).write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ))
    payload = {
        "diagnostic_type": "GT-only scene-disjoint C1 GVC/action-margin separability audit",
        "decision_constraint": "只选特征方向；不训练、不选阈值、不输出推理分数、不做 NMS、候选修改或 AP。",
        "ground_truth_usage": "offline diagnostic only",
        "proposal_materialization_applied": False,
        "scene_count": len(scenes), "fold_count": args.fold_count,
        "track_count": len(track_rows), "strict_track_binary_count": len(strict_tracks),
        "track_label_counts": dict(sorted(Counter(row["track_label"] for row in track_rows).items())),
        "missing_independent_evidence_count": len(missing),
        "missing_independent_evidence_label_counts": dict(sorted(missing_counts.items())),
        "mixed_component_count": len(component_rows), "strict_mixed_component_binary_count": len(strict_components),
        "component_label_counts": dict(sorted(Counter(row["component_label"] for row in component_rows).items())),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

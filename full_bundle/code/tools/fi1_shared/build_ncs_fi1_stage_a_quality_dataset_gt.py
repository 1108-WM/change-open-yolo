"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

import json

import math

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]

THRESHOLDS = tuple(value / 100.0 for value in range(50, 100, 5))

GVC_FEATURE_NAMES = (
    "log1p_point_count",
    "point_fraction_of_scene",
    "log1p_member_count",
    "frozen_score",
    "original_source_score",
    "gvc_mean",
    "gvc_max",
    "gvc_variance",
    "log1p_selected_view_count",
    "matched_view_fraction",
    "zero_support_fraction",
    "projected_box_iou_mean",
    "projected_box_iou_max",
    "projected_box_iou_variance",
    "visible_mask_support_mean",
    "visible_mask_support_max",
    "visible_mask_support_variance",
    "visible_point_fraction_mean",
    "visible_point_fraction_max",
    "visible_point_fraction_variance",
)

UNION_FEATURE_NAMES = (
    "log1p_point_count",
    "point_fraction_of_scene",
    "log1p_member_count",
    "frozen_score",
    "base_quality",
    "track_quality_q",
    "native_group_median_quality_q",
    "threshold_cross_probability",
    "balanced_fit_raw_probability",
    "log1p_support_relation_count",
    "point_iou",
    "native_inside_track_ratio",
    "track_inside_native_ratio",
    "aabb_iou",
    "centroid_distance_normalized",
    "log_track_over_native_point_count",
    "native_shared_superpoint_fraction",
    "track_shared_superpoint_fraction",
    "log1p_public_common_selected_view_count",
    "public_same_matched_observation_fraction",
    "public_different_matched_observation_fraction",
    "public_projected_box_iou_mean",
    "public_native_gvc_mean",
    "public_track_gvc_mean",
    "public_track_minus_native_gvc",
    "exclusive_boundary_contact_ratio_mean",
    "mean_rgb_distance",
    "mean_normal_difference",
)

FEATURE_NAMES_BY_SOURCE = {
    "native": GVC_FEATURE_NAMES,
    "track": GVC_FEATURE_NAMES,
    "pair_union": UNION_FEATURE_NAMES,
}

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {path}:{line_number}") from error
    return rows

def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes

def _fold_by_scene(path: Path, scenes: list[str]) -> dict[str, int]:
    manifest = json.loads(path.read_text())
    if manifest.get("contract_valid") is False:
        raise ValueError("fold manifest is not contract-valid")
    folds = list(manifest.get("folds", []))
    if len(folds) != 5:
        raise ValueError("fold manifest must contain exactly five folds")
    scene_set = set(scenes)
    result = {}
    seen_fold_indices = set()
    for fold in folds:
        fold_index = int(fold["fold_index"])
        if fold_index in seen_fold_indices:
            raise ValueError(f"duplicate fold index: {fold_index}")
        seen_fold_indices.add(fold_index)
        train = {str(scene) for scene in fold["train_scenes"]}
        validation = {str(scene) for scene in fold["validation_scenes"]}
        if train & validation or train | validation != scene_set:
            raise ValueError(f"fold {fold_index} is not an exact train/validation partition")
        if len(validation) != len(scenes) // 5 or len(train) != len(scenes) - len(validation):
            raise ValueError(f"fold {fold_index} does not have the fixed 80/20 scene split")
        for scene in fold["validation_scenes"]:
            if scene in result:
                raise ValueError(f"scene occurs in multiple validation folds: {scene}")
            result[str(scene)] = fold_index
    if seen_fold_indices != set(range(5)):
        raise ValueError("fold indices must be exactly 0..4")
    if set(result) != set(scenes):
        raise ValueError("fold validation scenes are not an exact scene-list partition")
    return result

class GeometryResolver:
    def __init__(self) -> None:
        self.native_masks: dict[Path, np.ndarray] = {}

    def points(self, locator: dict, expected_count: int, scene_point_count: int) -> np.ndarray:
        kind = str(locator.get("kind"))
        if kind == "native_mask_column":
            path = _resolve(Path(locator["masks_path"]))
            masks = self.native_masks.get(path)
            if masks is None:
                masks = np.load(path, mmap_mode="r")
                if masks.ndim != 2:
                    raise ValueError(f"native mask cache is not 2D: {path}")
                self.native_masks[path] = masks
            column = int(locator["column_index"])
            if column < 0 or column >= masks.shape[1]:
                raise ValueError(f"native mask column is out of range: {path}:{column}")
            points = np.flatnonzero(np.asarray(masks[:, column], dtype=bool)).astype(np.int64)
        elif kind == "point_indices_npz":
            path = _resolve(Path(locator["points_path"]))
            key = str(locator.get("array_key", "point_indices"))
            with np.load(path) as payload:
                if key not in payload:
                    raise ValueError(f"{key} is missing from {path}")
                points = np.unique(np.asarray(payload[key], dtype=np.int64))
        else:
            raise ValueError(f"unsupported geometry locator: {kind!r}")
        if len(points) != expected_count:
            raise ValueError(f"geometry point count differs: {len(points)} != {expected_count}")
        if not len(points) or points[0] < 0 or points[-1] >= scene_point_count:
            raise ValueError("geometry points are empty or outside the scene")
        return points

def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite feature {name}: {value!r}")
    return result

def _stats(mapping: dict | None) -> dict:
    mapping = mapping or {}
    return {
        "mean": _finite(mapping.get("mean", 0.0), "mean"),
        "max": _finite(mapping.get("max", 0.0), "max"),
        "variance": _finite(mapping.get("variance", 0.0), "variance"),
    }

def _gvc_features(row: dict, node: dict, scene_point_count: int) -> dict[str, float]:
    evidence = row.get("gvc_source_frame_excluded", {})
    gvc = _stats(evidence.get("gvc"))
    box = _stats(evidence.get("projected_box_iou"))
    support = _stats(evidence.get("visible_point_mask_support"))
    frames = list(evidence.get("selected_frames", []))
    visible = np.asarray(
        [_finite(frame.get("visible_point_fraction", 0.0), "visible_point_fraction") for frame in frames],
        dtype=np.float64,
    )
    selected = int(evidence.get("selected_view_count", 0))
    matched = int(evidence.get("matched_selected_view_count", 0))
    features = {
        "log1p_point_count": math.log1p(int(node["point_count"])),
        "point_fraction_of_scene": int(node["point_count"]) / max(1, scene_point_count),
        "log1p_member_count": math.log1p(int(node["member_count"])),
        "frozen_score": _finite(node["canonical_frozen_score"], "frozen_score"),
        "original_source_score": _finite(row.get("original_source_score", 0.0), "original_source_score"),
        "gvc_mean": gvc["mean"],
        "gvc_max": gvc["max"],
        "gvc_variance": gvc["variance"],
        "log1p_selected_view_count": math.log1p(selected),
        "matched_view_fraction": matched / max(1, selected),
        "zero_support_fraction": _finite(evidence.get("zero_support_selected_view_fraction", 0.0), "zero_support_fraction"),
        "projected_box_iou_mean": box["mean"],
        "projected_box_iou_max": box["max"],
        "projected_box_iou_variance": box["variance"],
        "visible_mask_support_mean": support["mean"],
        "visible_mask_support_max": support["max"],
        "visible_mask_support_variance": support["variance"],
        "visible_point_fraction_mean": float(visible.mean()) if len(visible) else 0.0,
        "visible_point_fraction_max": float(visible.max()) if len(visible) else 0.0,
        "visible_point_fraction_variance": float(visible.var()) if len(visible) else 0.0,
    }
    return {name: _finite(features[name], name) for name in GVC_FEATURE_NAMES}

def _union_features(
    union: dict, relation: dict, node: dict, scene_point_count: int,
) -> dict[str, float]:
    raw = relation["features"]
    features = {
        "log1p_point_count": math.log1p(int(node["point_count"])),
        "point_fraction_of_scene": int(node["point_count"]) / max(1, scene_point_count),
        "log1p_member_count": math.log1p(int(node["member_count"])),
        "frozen_score": _finite(node["canonical_frozen_score"], "frozen_score"),
        "base_quality": union["base_quality"],
        "track_quality_q": union["track_quality_q"],
        "native_group_median_quality_q": union["native_group_median_quality_q"],
        "threshold_cross_probability": union["threshold_cross_probability"],
        "balanced_fit_raw_probability": union["balanced_fit_raw_probability"],
        "log1p_support_relation_count": math.log1p(int(union["support_relation_count"])),
        "point_iou": raw.get("point_iou", 0.0),
        "native_inside_track_ratio": raw.get("native_inside_track_ratio", 0.0),
        "track_inside_native_ratio": raw.get("track_inside_native_ratio", 0.0),
        "aabb_iou": raw.get("aabb_iou", 0.0),
        "centroid_distance_normalized": raw.get("centroid_distance_normalized", 0.0),
        "log_track_over_native_point_count": raw.get("log_track_over_native_point_count", 0.0),
        "native_shared_superpoint_fraction": raw.get("native_shared_superpoint_fraction", 0.0),
        "track_shared_superpoint_fraction": raw.get("track_shared_superpoint_fraction", 0.0),
        "log1p_public_common_selected_view_count": math.log1p(int(raw.get("public_common_selected_view_count", 0))),
        "public_same_matched_observation_fraction": raw.get("public_same_matched_observation_fraction", 0.0),
        "public_different_matched_observation_fraction": raw.get("public_different_matched_observation_fraction", 0.0),
        "public_projected_box_iou_mean": raw.get("public_projected_box_iou_mean", 0.0),
        "public_native_gvc_mean": raw.get("public_native_gvc_mean", 0.0),
        "public_track_gvc_mean": raw.get("public_track_gvc_mean", 0.0),
        "public_track_minus_native_gvc": raw.get("public_track_minus_native_gvc", 0.0),
        "exclusive_boundary_contact_ratio_mean": raw.get("exclusive_boundary_contact_ratio_mean", 0.0),
        "mean_rgb_distance": raw.get("mean_rgb_distance", 0.0),
        "mean_normal_difference": raw.get("mean_normal_difference", 0.0),
    }
    return {name: _finite(features[name], name) for name in UNION_FEATURE_NAMES}

def _quality_target(best_iou: float) -> float:
    return round(float(np.mean([best_iou >= threshold for threshold in THRESHOLDS])), 1)

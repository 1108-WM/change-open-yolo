#!/usr/bin/env python3
"""Build the preregistered NCS first-innovation stage-A quality dataset.

The tool reads GT only to create label fields.  All model features come from
the already frozen, no-GT first-innovation ledgers.  It never changes a
candidate, writes a score plan, computes AP, or reads validation60/val312.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_train_scene_candidate_quality_ledger import _best_gt, _load_gt  # noqa: E402


VERSION = "ncs_fi1_stage_a_quality_dataset_v1"
THRESHOLDS = tuple(value / 100.0 for value in range(50, 100, 5))
SOURCE_NAMES = ("native", "track", "pair_union")

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


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "fold_manifest", "unique_geometry_root", "unique_geometry_audit_root",
        "gvc_root", "relation_root", "champion_plan_root", "ground_truth_root",
        "preregistration", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _read_scenes(args.scene_list)
    dataset_name = str(args.dataset_name)
    folds = _fold_by_scene(args.fold_manifest, scenes)
    unique_summary_path = args.unique_geometry_root / "summary.json"
    unique_audit_path = args.unique_geometry_audit_root / "summary.json"
    unique_summary = json.loads(unique_summary_path.read_text())
    unique_audit = json.loads(unique_audit_path.read_text())
    if unique_summary.get("contract_valid") is not True or unique_summary.get("ground_truth_read") is not False:
        raise ValueError("unique geometry ledger violates the frozen no-GT contract")
    unique_audit_errors = sum(
        int(value) for key, value in unique_audit.items() if key.endswith("_error_count")
    )
    if unique_audit.get("audit_valid") is not True or unique_audit_errors != 0:
        raise ValueError("unique geometry audit is not valid")
    relation_summary_path = args.relation_root / "summary.json"
    relation_summary = json.loads(relation_summary_path.read_text())
    if relation_summary.get("ground_truth_read") is not False or relation_summary.get("candidate_mutation") is not False:
        raise ValueError("relation ledger violates the no-GT/no-mutation contract")

    nodes = _read_jsonl(args.unique_geometry_root / "unique_geometry_ledger.jsonl")
    if len(nodes) != int(unique_summary["unique_geometry_count"]):
        raise ValueError("unique geometry row count differs from its summary")

    unions = _read_jsonl(args.champion_plan_root / "pair_union_append_candidates.jsonl")
    union_by_key = {
        (str(row["scene_name"]), int(row["candidate_id"])): row for row in unions
    }
    if len(union_by_key) != len(unions):
        raise ValueError("duplicate pair-union candidate key")

    nodes_by_scene: dict[str, list[dict]] = {scene: [] for scene in scenes}
    for node in nodes:
        scene = str(node["scene_name"])
        if scene not in nodes_by_scene:
            raise ValueError(f"unique geometry contains an unexpected scene: {scene}")
        nodes_by_scene[scene].append(node)

    resolver = GeometryResolver()
    output_rows = []
    source_counts = Counter()
    fold_source_counts = Counter()
    scene_summaries = []
    for scene_index, scene in enumerate(scenes, 1):
        gt_path = args.ground_truth_root / f"{scene}.txt"
        gt, gt_meta = _load_gt(gt_path, args.min_gt_points)
        scene_point_count = len(gt)
        gvc_rows = json.loads((args.gvc_root / scene / "c1_gvc_quality_ledger.json").read_text())
        gvc_by_key = {
            (str(row["candidate_source"]), int(row["candidate_id"])): row for row in gvc_rows
        }
        relation_rows = _read_jsonl(args.relation_root / scene / "relation_features_no_gt.jsonl")
        relation_by_key = {}
        for row in relation_rows:
            key = (int(row["track_id"]), str(row["native_exact_geometry_group_id"]))
            if key in relation_by_key:
                raise ValueError(f"{scene}: duplicate relation key {key}")
            relation_by_key[key] = row
        local_counts = Counter()
        for node in nodes_by_scene[scene]:
            source = str(node["canonical_candidate_source"])
            if source not in SOURCE_NAMES:
                raise ValueError(f"unsupported canonical source: {source}")
            candidate_id = int(node["canonical_candidate_id"])
            points = resolver.points(
                node["canonical_geometry_locator"], int(node["point_count"]), scene_point_count
            )
            label = _best_gt(points, gt, gt_meta)
            best_gt = label["best_gt"]
            if source == "native":
                evidence = gvc_by_key.get(("native_mask3d_yoloworld", candidate_id))
                if evidence is None:
                    raise ValueError(f"{scene}: missing native GVC row {candidate_id}")
                features = _gvc_features(evidence, node, scene_point_count)
            elif source == "track":
                evidence = gvc_by_key.get(("d2b_track", candidate_id))
                if evidence is None:
                    raise ValueError(f"{scene}: missing track GVC row {candidate_id}")
                features = _gvc_features(evidence, node, scene_point_count)
            else:
                union = union_by_key.get((scene, candidate_id))
                if union is None:
                    raise ValueError(f"{scene}: missing union plan row {candidate_id}")
                relation_key = (int(union["track_id"]), str(union["native_exact_geometry_group_id"]))
                relation = relation_by_key.get(relation_key)
                if relation is None:
                    raise ValueError(f"{scene}: missing union relation {relation_key}")
                features = _union_features(union, relation, node, scene_point_count)
            expected_names = FEATURE_NAMES_BY_SOURCE[source]
            if tuple(features) != expected_names:
                raise ValueError(f"{scene}/{node['geometry_hash']}: feature order differs")
            best_iou = float(label["best_iou"])
            row = {
                "row_id": f"{scene}:{node['geometry_hash']}",
                "scene_name": scene,
                "fold_index": int(folds[scene]),
                "geometry_key": str(node["geometry_key"]),
                "geometry_hash": str(node["geometry_hash"]),
                "candidate_source": source,
                "canonical_candidate_id_metadata_only": candidate_id,
                "canonical_geometry_locator_read_only": node["canonical_geometry_locator"],
                "point_count": int(node["point_count"]),
                "member_count": int(node["member_count"]),
                "frozen_score_metadata_only": float(node["canonical_frozen_score"]),
                "features": features,
                "label_best_gt_iou": best_iou,
                "label_best_gt_intersection": int(label["best_intersection"]),
                "label_best_gt_encoded_id": int(best_gt["encoded_id"]) if best_gt else None,
                "label_best_gt_semantic_id": int(best_gt["semantic_id"]) if best_gt else None,
                "label_best_gt_instance_id": int(best_gt["instance_id"]) if best_gt else None,
                "label_quality_q": _quality_target(best_iou),
                "ground_truth_usage": f"{dataset_name} label fields only",
                "candidate_mutation": False,
                "geometry_mutation": False,
                "score_mutation": False,
                "ap_computed": False,
            }
            output_rows.append(row)
            source_counts[source] += 1
            fold_source_counts[(int(folds[scene]), source)] += 1
            local_counts[source] += 1
        scene_summaries.append({
            "scene_name": scene,
            "fold_index": int(folds[scene]),
            "row_count": sum(local_counts.values()),
            "source_counts": dict(sorted(local_counts.items())),
            "ground_truth_sha256": _sha256(gt_path),
        })
        print(f"[stage A dataset] {scene_index}/{len(scenes)} {scene}: {dict(local_counts)}", flush=True)

    row_ids = [row["row_id"] for row in output_rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate stage-A dataset row id")
    output_rows.sort(key=lambda row: (row["scene_name"], row["geometry_hash"]))
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if args.output_root.exists() or staging.exists():
        raise FileExistsError(f"output or staging exists: {args.output_root}, {staging}")
    staging.mkdir(parents=True)
    try:
        dataset_path = staging / "quality_dataset.jsonl"
        dataset_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows
        ))
        summary = {
            "version": VERSION,
            "dataset_name": dataset_name,
            "preregistration_status": "frozen_before_first_run",
            "scene_count": len(scenes),
            "geometry_count": len(output_rows),
            "source_counts": dict(sorted(source_counts.items())),
            "fold_source_counts": {
                str(fold): {source: int(fold_source_counts[(fold, source)]) for source in SOURCE_NAMES}
                for fold in range(5)
            },
            "quality_thresholds": list(THRESHOLDS),
            "quality_target": "mean indicator of best class-agnostic IoU at 0.50:0.05:0.95",
            "feature_names_by_source": {
                source: list(names) for source, names in FEATURE_NAMES_BY_SOURCE.items()
            },
            "dataset_file": dataset_path.name,
            "dataset_sha256": _sha256(dataset_path),
            "ground_truth_usage": f"{dataset_name} label fields only",
            "feature_ground_truth_usage": "none",
            "candidate_mutation": False,
            "geometry_mutation": False,
            "score_mutation": False,
            "ap_computed": False,
            "validation60_read": False,
            "val312_read": False,
            "input_provenance": {
                "preregistration_sha256": _sha256(args.preregistration),
                "scene_list_sha256": _sha256(args.scene_list),
                "fold_manifest_sha256": _sha256(args.fold_manifest),
                "unique_geometry_summary_sha256": _sha256(unique_summary_path),
                "unique_geometry_audit_sha256": _sha256(unique_audit_path),
                "gvc_summary_sha256": _sha256(args.gvc_root / "summary.json"),
                "relation_summary_sha256": _sha256(relation_summary_path),
                "champion_plan_summary_sha256": _sha256(args.champion_plan_root / "summary.json"),
            },
            "scene_summaries": scene_summaries,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, args.output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"
    ))
    parser.add_argument("--fold-manifest", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json"
    ))
    parser.add_argument("--unique-geometry-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger"
    ))
    parser.add_argument("--unique-geometry-audit-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_audit"
    ))
    parser.add_argument("--gvc-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/gvc_quality_ledger"
    ))
    parser.add_argument("--relation-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/relation_ledger"
    ))
    parser.add_argument("--champion-plan-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/champion_plan"
    ))
    parser.add_argument("--ground-truth-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth"
    ))
    parser.add_argument("--preregistration", type=Path, default=Path(
        "docs/NCS_FI1_STAGE_A_PREREGISTRATION_20260822.md"
    ))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="NCS-train100")
    parser.add_argument("--min-gt-points", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output_root": str(_resolve(args.output_root)),
        "geometry_count": result["geometry_count"],
        "source_counts": result["source_counts"],
        "ap_computed": result["ap_computed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

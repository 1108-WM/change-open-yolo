#!/usr/bin/env python3
"""Audit the official-train candidate-quality supervision ledgers.

This is a read-only audit of candidate records.  GT-derived fields are
accepted only as labels and are never included in the exported feature
schema.  It intentionally does not build pair actions, modify candidates, or
evaluate AP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = "native_mask3d_yoloworld"
TRACK_SOURCE = "d2b_track"
SOURCES = (NATIVE_SOURCE, TRACK_SOURCE)
LABEL_FIELDS = (
    "label_best_gt_iou",
    "label_best_gt_instance_id",
    "label_valid_iou25",
    "label_valid_iou50",
)
GVC_FEATURES = (
    "gvc_excluded_mean",
    "gvc_excluded_max",
    "gvc_excluded_variance",
    "gvc_excluded_selected_view_count",
    "gvc_excluded_matched_view_count",
    "gvc_excluded_zero_support_fraction",
)
TRACK_FEATURES = (
    "support_view_count",
    "superpoint_count",
    "source_frame_count",
    "merge_action_count",
    "mean_consensus_rate",
    "mean_edge_score",
    "mean_node_quality",
)
BASE_NUMERIC_FEATURES = ("original_source_score", "point_count", "point_fraction_of_scene")
FORBIDDEN_FEATURES = {
    "scene_name", "candidate_id", "native_class_id", "label_best_gt_instance_id",
    "label_best_gt_semantic_id", "native_exact_geometry_group_size",
}


def read_scene_list(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicate scene names")
    return scenes


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
    return rows


def candidate_ledger_path(records_root: Path, scene: str) -> Path:
    return records_root / scene / "candidate_quality_training_ledger" / scene / "candidate_labels.jsonl"


def _finite_number(row: dict, field: str, scene: str) -> float:
    if field not in row:
        raise ValueError(f"{scene}: missing required field {field}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{scene}: {field} is not numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{scene}: {field} is not finite")
    return value


def feature_schema(protocol_name: str = "candidate_quality") -> dict:
    if not protocol_name or not protocol_name.replace("_", "").isalnum():
        raise ValueError("protocol_name must contain only letters, digits, and underscores")
    groups = {
        "A_source_only": {"categorical": ["candidate_source"], "numeric": []},
        "B_source_plus_original_score": {
            "categorical": ["candidate_source"], "numeric": ["original_source_score"],
        },
        "C_plus_geometry_track_structure": {
            "categorical": ["candidate_source"],
            "numeric": list(BASE_NUMERIC_FEATURES + TRACK_FEATURES),
        },
        "D_plus_gvc": {
            "categorical": ["candidate_source"],
            "numeric": list(BASE_NUMERIC_FEATURES + TRACK_FEATURES + GVC_FEATURES),
        },
    }
    all_features = sorted({field for group in groups.values() for field in group["categorical"] + group["numeric"]})
    invalid = [field for field in all_features if field.startswith("label_") or field in FORBIDDEN_FEATURES]
    if invalid:
        raise AssertionError(f"forbidden fields leaked into feature schema: {invalid}")
    return {
        "version": f"{protocol_name}_candidate_quality_feature_schema_v1",
        "protocol_name": protocol_name,
        "groups": groups,
        "track_missing_value": "NaN",
        "track_missing_indicator": True,
        "native_exact_geometry_group_size": "sample_weight_only",
        "excluded_fields": sorted(FORBIDDEN_FEATURES),
    }


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    values = sorted(values)
    def q(fraction: float) -> float:
        index = (len(values) - 1) * fraction
        lower, upper = int(index), min(len(values) - 1, int(index) + 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    return {"count": len(values), "min": values[0], "p25": q(.25), "p50": q(.5), "p75": q(.75), "max": values[-1], "mean": sum(values) / len(values)}


def _evaluation_scenes(paths: Iterable[Path]) -> set[str]:
    scenes: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing required evaluation scene list: {path}")
        scenes.update(read_scene_list(path))
    return scenes


def audit_dataset(
    scene_list: Path,
    records_root: Path,
    evaluation_scene_lists: Iterable[Path],
    *,
    protocol_name: str = "candidate_quality",
    prepared_root: Path | None = None,
) -> tuple[dict, list[dict], dict]:
    scenes = read_scene_list(scene_list)
    evaluation_scene_lists = list(evaluation_scene_lists)
    if not evaluation_scene_lists:
        raise ValueError("at least one evaluation scene list is required")
    overlaps = sorted(set(scenes) & _evaluation_scenes(evaluation_scene_lists))
    if overlaps:
        raise ValueError(f"official train scenes overlap evaluation scenes: {overlaps}")
    schema = feature_schema(protocol_name)
    scene_statistics = []
    aggregate = Counter()
    unique_geometry_estimate = 0.0
    for scene in scenes:
        record_dir = records_root / scene
        if prepared_root is not None:
            prepared_manifest = prepared_root / scene / "stream_prepare_manifest.json"
            if not prepared_manifest.is_file():
                raise ValueError(f"{scene}: missing prepared manifest {prepared_manifest}")
            prepared = json.loads(prepared_manifest.read_text())
            if prepared.get("scene_name") != scene or prepared.get("split") != "official_scannet200_train":
                raise ValueError(f"{scene}: prepared manifest is not an official train scene")
        for manifest in ("native_export_manifest.json", "track_pipeline_manifest.json"):
            if not (record_dir / manifest).is_file():
                raise ValueError(f"{scene}: missing manifest {manifest}")
        rows = read_jsonl(candidate_ledger_path(records_root, scene))
        if not rows:
            raise ValueError(f"{scene}: candidate ledger is empty")
        source_rows = defaultdict(list)
        for row in rows:
            if row.get("scene_name") != scene:
                raise ValueError(f"{scene}: candidate row has mismatched scene_name")
            source = row.get("candidate_source")
            if source not in SOURCES:
                raise ValueError(f"{scene}: unexpected candidate_source {source!r}")
            if row.get("ground_truth_usage") != "label_only":
                raise ValueError(f"{scene}: candidate GT contract is not label_only")
            _finite_number(row, "point_count", scene)
            if int(row["point_count"]) <= 0:
                raise ValueError(f"{scene}: candidate has non-positive point_count")
            for field in BASE_NUMERIC_FEATURES + GVC_FEATURES + LABEL_FIELDS:
                _finite_number(row, field, scene) if field not in ("label_best_gt_instance_id", "label_valid_iou25", "label_valid_iou50") else None
            if not isinstance(row.get("label_valid_iou25"), bool) or not isinstance(row.get("label_valid_iou50"), bool):
                raise ValueError(f"{scene}: validity labels must be booleans")
            best_iou = float(row["label_best_gt_iou"])
            if not 0.0 <= best_iou <= 1.0:
                raise ValueError(f"{scene}: label_best_gt_iou must be in [0, 1]")
            if bool(row["label_valid_iou25"]) != (best_iou >= 0.25):
                raise ValueError(f"{scene}: label_valid_iou25 disagrees with label_best_gt_iou")
            if bool(row["label_valid_iou50"]) != (best_iou >= 0.50):
                raise ValueError(f"{scene}: label_valid_iou50 disagrees with label_best_gt_iou")
            # A candidate with no eligible GT instance has a well-defined zero
            # IoU label.  Its null ID is a valid label state, not missing data.
            if row.get("label_best_gt_instance_id") is None and (
                float(row["label_best_gt_iou"]) != 0.0
                or bool(row["label_valid_iou25"])
                or bool(row["label_valid_iou50"])
            ):
                raise ValueError(f"{scene}: null best-GT ID has inconsistent validity labels")
            if source == NATIVE_SOURCE:
                size = _finite_number(row, "native_exact_geometry_group_size", scene)
                if size < 1 or int(size) != size:
                    raise ValueError(f"{scene}: invalid native_exact_geometry_group_size")
                unique_geometry_estimate += 1.0 / size
            source_rows[source].append(row)
        source_summary = {}
        for source in SOURCES:
            source_values = source_rows[source]
            ious = [float(row["label_best_gt_iou"]) for row in source_values]
            source_summary[source] = {
                "candidate_count": len(source_values),
                "valid_iou25_count": sum(bool(row["label_valid_iou25"]) for row in source_values),
                "valid_iou50_count": sum(bool(row["label_valid_iou50"]) for row in source_values),
                "best_gt_iou": _quantiles(ious),
            }
            aggregate[("candidate", source)] += len(source_values)
            aggregate[("valid25", source)] += source_summary[source]["valid_iou25_count"]
            aggregate[("valid50", source)] += source_summary[source]["valid_iou50_count"]
        scene_statistics.append({"scene_name": scene, "candidate_count": len(rows), "by_source": source_summary})
    summary = {
        "version": f"{protocol_name}_candidate_quality_audit_v1",
        "protocol_name": protocol_name,
        "scene_count": len(scenes),
        "scenes": scenes,
        "evaluation_overlap_count": 0,
        "candidate_count": sum(row["candidate_count"] for row in scene_statistics),
        "native_exact_geometry_group_count_estimate": round(unique_geometry_estimate),
        "by_source": {
            source: {
                "candidate_count": aggregate[("candidate", source)],
                "valid_iou25_count": aggregate[("valid25", source)],
                "valid_iou50_count": aggregate[("valid50", source)],
            } for source in SOURCES
        },
        "ground_truth_usage": "labels_only_for_offline_train_quality_oof",
        "pair_actions_enabled": False,
        "candidate_geometry_modified": False,
        "ap_evaluation_run": False,
    }
    return summary, scene_statistics, schema


def write_audit(output_root: Path, summary: dict, scene_statistics: list[dict], schema: dict) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {output_root}")
    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (staging / "scene_statistics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scene_statistics))
        (staging / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--evaluation-scene-list", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.evaluation_scene_list:
        args.evaluation_scene_list = [
            PROJECT_ROOT / "output/scannet200/scene_splits/gvc_holdout_20260803/gvc_safety60.txt",
            PROJECT_ROOT / "output/scannet200/scene_splits/even48.txt",
            PROJECT_ROOT / "output/scannet200/scene_splits/odd96.txt",
            PROJECT_ROOT / "output/scannet200/scene_splits/gvc_holdout_20260803/gvc_test60.txt",
        ]
    summary, scene_statistics, schema = audit_dataset(
        args.scene_list,
        args.records_root,
        args.evaluation_scene_list,
        protocol_name=args.protocol_name,
        prepared_root=args.prepared_root,
    )
    write_audit(args.output_root, summary, scene_statistics, schema)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

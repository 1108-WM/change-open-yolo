#!/usr/bin/env python3
"""Independently audit the NCS first-innovation stage-A quality dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ncs_fi1_stage_a_quality_dataset_gt import (  # noqa: E402
    FEATURE_NAMES_BY_SOURCE,
    SOURCE_NAMES,
    _fold_by_scene,
    _read_jsonl,
    _read_scenes,
    _resolve,
    _sha256,
)


VERSION = "ncs_fi1_stage_a_quality_dataset_audit_v1"
FORBIDDEN_FEATURE_FRAGMENTS = (
    "ground_truth", "semantic_id", "class_index", "class_id", "scene_name",
    "candidate_id", "geometry_hash", "instance_id", "iou_target", "best_iou",
)


def _forbidden_feature_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("label_")
        or lowered.startswith("gt_")
        or any(fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    )


def run(args: argparse.Namespace) -> dict:
    for name in ("dataset_root", "scene_list", "fold_manifest", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    summary_path = args.dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = _read_jsonl(args.dataset_root / summary["dataset_file"])
    scenes = _read_scenes(args.scene_list)
    folds = _fold_by_scene(args.fold_manifest, scenes)
    errors = Counter()
    row_ids = set()
    source_counts = Counter()
    fold_source_counts = Counter()
    label_counts = Counter()
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id in row_ids:
            errors["duplicate_row_id"] += 1
        row_ids.add(row_id)
        scene = str(row.get("scene_name"))
        source = str(row.get("candidate_source"))
        if scene not in folds:
            errors["unexpected_scene"] += 1
            continue
        if source not in SOURCE_NAMES:
            errors["unexpected_source"] += 1
            continue
        if int(row.get("fold_index", -1)) != int(folds[scene]):
            errors["fold_mismatch"] += 1
        features = row.get("features")
        if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES_BY_SOURCE[source]):
            errors["feature_schema_mismatch"] += 1
        else:
            for name, value in features.items():
                if _forbidden_feature_name(name):
                    errors["forbidden_feature_name"] += 1
                try:
                    if not math.isfinite(float(value)):
                        errors["nonfinite_feature"] += 1
                except (TypeError, ValueError):
                    errors["nonfinite_feature"] += 1
        best_iou = float(row.get("label_best_gt_iou", -1.0))
        quality = float(row.get("label_quality_q", -1.0))
        if not (0.0 <= best_iou <= 1.0):
            errors["invalid_best_iou"] += 1
        if min(abs(quality - index / 10.0) for index in range(11)) > 1e-9:
            errors["invalid_quality_target"] += 1
        if row.get("candidate_mutation") is not False:
            errors["candidate_mutation"] += 1
        if row.get("geometry_mutation") is not False:
            errors["geometry_mutation"] += 1
        if row.get("score_mutation") is not False:
            errors["score_mutation"] += 1
        if row.get("ap_computed") is not False:
            errors["ap_computed"] += 1
        source_counts[source] += 1
        fold_source_counts[(int(folds[scene]), source)] += 1
        label_counts[f"{quality:.1f}"] += 1
    if len(rows) != int(summary.get("geometry_count", -1)):
        errors["geometry_count_mismatch"] += 1
    if dict(sorted(source_counts.items())) != summary.get("source_counts"):
        errors["source_count_mismatch"] += 1
    if _sha256(args.dataset_root / summary["dataset_file"]) != summary.get("dataset_sha256"):
        errors["dataset_sha256_mismatch"] += 1
    if set(row["scene_name"] for row in rows) != set(scenes):
        errors["scene_coverage_mismatch"] += 1
    for fold in range(5):
        for source in SOURCE_NAMES:
            if fold_source_counts[(fold, source)] <= 0:
                errors["empty_fold_source"] += 1
    output = {
        "version": VERSION,
        "audit_valid": not errors,
        "error_count": int(sum(errors.values())),
        "error_counts": dict(sorted(errors.items())),
        "row_count": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "fold_source_counts": {
            str(fold): {source: int(fold_source_counts[(fold, source)]) for source in SOURCE_NAMES}
            for fold in range(5)
        },
        "quality_target_counts": dict(sorted(label_counts.items(), key=lambda item: float(item[0]))),
        "feature_ground_truth_leakage_count": int(errors.get("forbidden_feature_name", 0)),
        "candidate_mutation": False,
        "geometry_mutation": False,
        "score_mutation": False,
        "ap_computed": False,
        "validation60_read": False,
        "val312_read": False,
        "input_provenance": {
            "dataset_summary_sha256": _sha256(summary_path),
            "dataset_sha256": _sha256(args.dataset_root / summary["dataset_file"]),
            "scene_list_sha256": _sha256(args.scene_list),
            "fold_manifest_sha256": _sha256(args.fold_manifest),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"
    ))
    parser.add_argument("--fold-manifest", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json"
    ))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

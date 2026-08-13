#!/usr/bin/env python3
"""Re-audit q Spearman at class-expanded, geometry-group, and track units.

This reads frozen OOF predictions only.  Native exact geometry groups are
rebuilt from the original native masks solely to correct reporting units; they
are never used as candidate features or inference actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import NATIVE_SOURCE, TRACK_SOURCE, read_jsonl


def _spearman(labels: list[float], scores: list[float]) -> float | None:
    if len(labels) < 2 or len(set(labels)) < 2 or len(set(scores)) < 2:
        return None
    result = spearmanr(labels, scores)
    value = getattr(result, "statistic", getattr(result, "correlation", float("nan")))
    return float(value) if math.isfinite(float(value)) else None


def _geometry_groups(mask_path: Path) -> dict[int, str]:
    masks = np.load(mask_path, mmap_mode="r")
    groups = defaultdict(list)
    for candidate_id in range(masks.shape[1]):
        digest = hashlib.sha256(np.packbits(np.asarray(masks[:, candidate_id], dtype=np.uint8)).tobytes()).hexdigest()
        groups[digest].append(candidate_id)
    mapping = {}
    for group_index, members in enumerate(sorted(groups.values(), key=lambda values: values[0])):
        mapping.update({candidate_id: f"geometry:{group_index:04d}" for candidate_id in members})
    return mapping


def audit(rows: list[dict], records_root: Path) -> dict:
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_name"]].append(row)
    reports = {}
    for model in sorted(rows[0]["predictions"]):
        native_rows = [row for row in rows if row["candidate_source"] == NATIVE_SOURCE]
        track_rows = [row for row in rows if row["candidate_source"] == TRACK_SOURCE]
        native_group_rows = []
        for scene, scene_rows in by_scene.items():
            groups = _geometry_groups(records_root / scene / "native_cache" / f"{scene}_pred_masks.npy")
            native_by_id = {int(row["candidate_id"]): row for row in scene_rows if row["candidate_source"] == NATIVE_SOURCE}
            if set(groups) != set(native_by_id):
                raise ValueError(f"{scene}: native OOF rows differ from native mask IDs")
            grouped = defaultdict(list)
            for candidate_id, group_id in groups.items():
                grouped[group_id].append(native_by_id[candidate_id])
            for group_id, members in grouped.items():
                labels = {float(row["label_best_gt_iou"]) for row in members}
                if len(labels) != 1:
                    raise ValueError(f"{scene}/{group_id}: identical native geometry has inconsistent IoU labels")
                native_group_rows.append({
                    "label": labels.pop(),
                    "prediction": float(np.median([row["predictions"][model]["q"] for row in members])),
                })
        reports[model] = {
            "class_expanded_unweighted_spearman": _spearman(
                [float(row["label_best_gt_iou"]) for row in rows], [float(row["predictions"][model]["q"]) for row in rows]
            ),
            "native_class_expanded_unweighted_spearman": _spearman(
                [float(row["label_best_gt_iou"]) for row in native_rows], [float(row["predictions"][model]["q"]) for row in native_rows]
            ),
            "native_geometry_group_spearman": _spearman(
                [row["label"] for row in native_group_rows], [row["prediction"] for row in native_group_rows]
            ),
            "track_raw_spearman": _spearman(
                [float(row["label_best_gt_iou"]) for row in track_rows], [float(row["predictions"][model]["q"]) for row in track_rows]
            ),
            "native_geometry_group_count": len(native_group_rows),
            "track_raw_count": len(track_rows),
        }
    return {
        "version": "candidate_quality_oof_spearman_unit_reaudit_v1",
        "model_q_spearman": reports,
        "ground_truth_usage": "existing official-train labels only",
        "model_retrained": False,
        "candidate_geometry_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    report = audit(read_jsonl(args.oof_predictions), args.records_root)
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

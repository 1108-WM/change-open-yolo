#!/usr/bin/env python3
"""Audit frozen native geometry for possible class-agnostic folding.

This no-GT diagnostic only measures exact and high-overlap native masks.  It
does not fold, delete, score, relabel, materialize, or evaluate candidates.
Per-candidate provenance is not present in the frozen cache, so that absence
is recorded rather than inferred from class IDs or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
NEAR_IOU = 0.99
INCLUSION = 0.99
REPORT_IOU = 0.50


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _scenes(path: Path) -> list[str]:
    result = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("场景列表为空或含重复")
    return result


def _hash_mask(mask: np.ndarray) -> str:
    points = np.flatnonzero(mask).astype(np.int32, copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray([len(points)], dtype=np.int64).tobytes())
    digest.update(points.tobytes())
    return digest.hexdigest()


def _scene(scene: str, args) -> tuple[list[dict], list[dict], dict]:
    masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
    scores = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_scores.npy"), dtype=np.float64)
    classes = np.asarray(np.load(args.native_prediction_cache / f"{scene}_pred_classes.npy"), dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != len(scores) or len(scores) != len(classes):
        raise ValueError(f"{scene}: native 缓存维度不一致")
    binary = np.asarray(masks, dtype=bool)
    sizes = np.count_nonzero(binary, axis=0).astype(np.int64)
    exact = defaultdict(list)
    for candidate_id in range(binary.shape[1]):
        exact[_hash_mask(binary[:, candidate_id])].append(candidate_id)
    exact_rows = []
    for geometry_hash, ids in sorted(exact.items()):
        if len(ids) < 2:
            continue
        exact_rows.append({
            "scene_name": scene, "geometry_hash": geometry_hash, "candidate_ids": ids,
            "candidate_count": len(ids), "point_count": int(sizes[ids[0]]),
            "class_indices": [int(classes[item]) for item in ids],
            "distinct_class_count": len({int(classes[item]) for item in ids}),
            "scores": [float(scores[item]) for item in ids],
            "per_candidate_source_provenance": "unavailable_in_frozen_native_cache",
            "ground_truth_usage": "none", "proposal_materialization_applied": False,
        })
    incidence = sparse.csc_matrix(binary.astype(np.int32, copy=False))
    intersections = (incidence.T @ incidence).tocoo()
    near_rows = []
    for left, right, intersection in zip(intersections.row, intersections.col, intersections.data):
        if left >= right or intersection <= 0:
            continue
        union = int(sizes[left] + sizes[right] - intersection)
        iou = float(intersection / max(1, union))
        left_covered = float(intersection / max(1, sizes[left]))
        right_covered = float(intersection / max(1, sizes[right]))
        if iou < REPORT_IOU and max(left_covered, right_covered) < INCLUSION:
            continue
        near_rows.append({
            "scene_name": scene, "left_candidate_id": int(left), "right_candidate_id": int(right),
            "left_class_index": int(classes[left]), "right_class_index": int(classes[right]),
            "left_score": float(scores[left]), "right_score": float(scores[right]),
            "intersection_point_count": int(intersection), "left_point_count": int(sizes[left]),
            "right_point_count": int(sizes[right]), "point_iou": iou,
            "left_covered_ratio": left_covered, "right_covered_ratio": right_covered,
            "near_duplicate_iou_099": iou >= NEAR_IOU,
            "left_in_right_099": left_covered >= INCLUSION,
            "right_in_left_099": right_covered >= INCLUSION,
            "same_class": bool(classes[left] == classes[right]),
            "per_candidate_source_provenance": "unavailable_in_frozen_native_cache",
            "ground_truth_usage": "none", "proposal_materialization_applied": False,
        })
    summary = {
        "scene_name": scene, "native_candidate_count": int(binary.shape[1]),
        "exact_geometry_group_count": len(exact_rows),
        "exact_geometry_candidate_count": sum(row["candidate_count"] for row in exact_rows),
        "exact_cross_class_group_count": sum(row["distinct_class_count"] > 1 for row in exact_rows),
        "reported_high_overlap_pair_count": len(near_rows),
        "near_duplicate_iou_099_pair_count": sum(row["near_duplicate_iou_099"] for row in near_rows),
        "inclusion_099_pair_count": sum(row["left_in_right_099"] or row["right_in_left_099"] for row in near_rows),
    }
    return exact_rows, near_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "native_prediction_cache", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    scenes = _scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    summaries = []
    for index, scene in enumerate(scenes, start=1):
        scene_root = args.output_root / scene
        if (scene_root / "summary.json").is_file():
            if not args.resume:
                raise SystemExit(f"输出场景已存在：{scene}")
            summaries.append(json.loads((scene_root / "summary.json").read_text()))
            print(f"[跳过已有] {index}/{len(scenes)} {scene}", flush=True)
            continue
        exact, near, summary = _scene(scene, args)
        scene_root.mkdir()
        for name, rows in (("exact_geometry_groups.jsonl", exact), ("high_overlap_pairs.jsonl", near)):
            (scene_root / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene}: exact={summary['exact_geometry_group_count']}, high_overlap={summary['reported_high_overlap_pair_count']}", flush=True)
    payload = {
        "diagnostic_type": "no-GT frozen native exact/near geometry folding audit",
        "ground_truth_usage": "none", "proposal_materialization_applied": False,
        "native_geometry_fold_applied": False,
        "near_duplicate_iou_threshold": NEAR_IOU, "inclusion_ratio_threshold": INCLUSION,
        "report_pair_iou_floor": REPORT_IOU,
        "per_candidate_source_provenance": "unavailable_in_frozen_native_cache",
        "scene_count": len(summaries),
        "native_candidate_count": sum(row["native_candidate_count"] for row in summaries),
        "exact_geometry_group_count": sum(row["exact_geometry_group_count"] for row in summaries),
        "exact_geometry_candidate_count": sum(row["exact_geometry_candidate_count"] for row in summaries),
        "exact_cross_class_group_count": sum(row["exact_cross_class_group_count"] for row in summaries),
        "reported_high_overlap_pair_count": sum(row["reported_high_overlap_pair_count"] for row in summaries),
        "near_duplicate_iou_099_pair_count": sum(row["near_duplicate_iou_099_pair_count"] for row in summaries),
        "inclusion_099_pair_count": sum(row["inclusion_099_pair_count"] for row in summaries),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

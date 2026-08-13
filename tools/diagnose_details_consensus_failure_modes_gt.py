#!/usr/bin/env python3
"""GT-only：归因 Details 共识轨迹在类别无关几何阶段的剩余损失。

工具只读取已经固定的 Details 轨迹、共识轨迹、原始 ScanNet superpoint 和 GT，
比较原始点并集、接触 superpoint 闭包、当前共识以及候选 superpoint 的 oracle
上限。GT 只生成离线账本，不输出推理候选，也不修改任何阈值或 mask。
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复场景")
    return scenes


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(value) for value in VALID_CLASS_IDS_200_INST}
    sizes = {}
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id <= 0 or instance_id // 1000 not in valid_classes:
            continue
        size = int(np.sum(gt_ids == instance_id))
        if size >= int(min_region_size):
            sizes[instance_id] = size
    return gt_ids, sizes


def _valid_points(points, point_count):
    points = np.unique(np.asarray(points, dtype=np.int64))
    return points[(points >= 0) & (points < int(point_count))]


def _load_track_points(record, point_count):
    return _valid_points(np.load(record["points_path"])["point_indices"], point_count)


def _point_metrics(points, gt_ids, instance_id, gt_size):
    points = _valid_points(points, len(gt_ids))
    if len(points) == 0 or instance_id <= 0:
        return {"iou": 0.0, "precision": 0.0, "coverage": 0.0, "point_count": int(len(points))}
    intersection = int(np.sum(gt_ids[points] == int(instance_id)))
    return {
        "iou": float(intersection / max(1, len(points) + int(gt_size) - intersection)),
        "precision": float(intersection / max(1, len(points))),
        "coverage": float(intersection / max(1, int(gt_size))),
        "point_count": int(len(points)),
    }


def _best_gt(points, gt_ids, gt_sizes):
    points = _valid_points(points, len(gt_ids))
    if len(points) == 0:
        return -1, 0.0
    ids, counts = np.unique(gt_ids[points], return_counts=True)
    best_id, best_iou = -1, 0.0
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        iou = float(intersection / max(1, len(points) + gt_sizes[instance_id] - int(intersection)))
        if iou > best_iou:
            best_id, best_iou = instance_id, iou
    return best_id, best_iou


def _points_by_superpoint(superpoints):
    order = np.argsort(superpoints, kind="mergesort")
    sorted_ids = superpoints[order]
    ids, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], len(order))
    return {
        int(superpoint_id): np.asarray(order[start:end], dtype=np.int64)
        for superpoint_id, start, end in zip(ids, starts, ends)
    }


def _closure_points(superpoint_ids, points_by_superpoint):
    chunks = [points_by_superpoint[int(item)] for item in np.unique(superpoint_ids) if int(item) in points_by_superpoint]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def oracle_superpoint_subset(
    superpoint_ids,
    superpoints,
    gt_ids,
    instance_id,
    gt_size,
    points_by_superpoint=None,
):
    """返回接触 superpoint 中对固定 GT 实例最优的精度排序前缀上限。"""
    if points_by_superpoint is None:
        points_by_superpoint = _points_by_superpoint(superpoints)
    if isinstance(superpoint_ids, set):
        superpoint_ids = sorted(superpoint_ids)
    items = []
    for superpoint_id in np.unique(np.asarray(superpoint_ids, dtype=np.int64)):
        points = points_by_superpoint.get(int(superpoint_id), np.empty(0, dtype=np.int64))
        intersection = int(np.sum(gt_ids[points] == int(instance_id)))
        false_positive = int(len(points) - intersection)
        precision = float(intersection / max(1, len(points)))
        items.append((precision, intersection, false_positive, int(superpoint_id), int(len(points))))
    items.sort(key=lambda item: (-item[0], item[3]))
    best = {"iou": 0.0, "superpoint_count": 0, "point_count": 0}
    intersection_sum = false_positive_sum = point_sum = 0
    for index, (_, intersection, false_positive, _, point_count) in enumerate(items, start=1):
        intersection_sum += intersection
        false_positive_sum += false_positive
        point_sum += point_count
        iou = float(intersection_sum / max(1, int(gt_size) + false_positive_sum))
        if iou > best["iou"]:
            best = {"iou": iou, "superpoint_count": index, "point_count": point_sum}
    return best


def classify_instance(best_raw_iou, best_closure_iou, best_consensus_iou, best_oracle_iou, union_consensus_iou, track_count):
    if best_consensus_iou >= 0.25:
        return "当前共识已恢复"
    if track_count == 0:
        return "Details关联未形成主属轨迹"
    if best_oracle_iou < 0.25:
        return "接触superpoint候选上限不足"
    if union_consensus_iou >= 0.25:
        return "共识轨迹碎裂且并集可恢复"
    if best_raw_iou >= 0.25 and best_closure_iou < 0.25:
        return "原始superpoint闭包污染"
    return "当前共识选择损失"


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size)
    processed = np.load(args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy", mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10 or len(processed) != len(gt_ids):
        raise ValueError(f"{scene_name} 的处理点云、superpoint或GT点数不一致")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint = _points_by_superpoint(superpoints)
    source = json.loads((args.details_track_root / scene_name / "automatic_tracks.json").read_text())
    consensus = json.loads((args.consensus_track_root / scene_name / "automatic_tracks.json").read_text())
    consensus_by_source = {int(track.get("source_track_id", track["track_id"])): track for track in consensus.get("tracks", [])}

    track_rows = []
    current_points_by_target = defaultdict(list)
    rows_by_target = defaultdict(list)
    for track in source.get("tracks", []):
        track_id = int(track["track_id"])
        raw_points = _load_track_points(track, len(gt_ids))
        instance_id, raw_best_iou = _best_gt(raw_points, gt_ids, gt_sizes)
        touched_ids = np.unique(superpoints[raw_points]) if len(raw_points) else np.empty(0, dtype=np.int64)
        closure_points = _closure_points(touched_ids, points_by_superpoint)
        current = consensus_by_source.get(track_id)
        current_points = _load_track_points(current, len(gt_ids)) if current is not None else np.empty(0, dtype=np.int64)
        gt_size = int(gt_sizes.get(instance_id, 0))
        raw_metrics = _point_metrics(raw_points, gt_ids, instance_id, gt_size)
        closure_metrics = _point_metrics(closure_points, gt_ids, instance_id, gt_size)
        current_metrics = _point_metrics(current_points, gt_ids, instance_id, gt_size)
        oracle = oracle_superpoint_subset(
            touched_ids,
            superpoints,
            gt_ids,
            instance_id,
            gt_size,
            points_by_superpoint,
        ) if instance_id > 0 else {"iou": 0.0, "superpoint_count": 0, "point_count": 0}
        row = {
            "scene_name": scene_name,
            "source_track_id": track_id,
            "support_view_count": int(track.get("support_view_count", 0)),
            "best_gt_instance_id": int(instance_id),
            "raw_point_iou": float(raw_metrics["iou"]),
            "raw_point_precision": float(raw_metrics["precision"]),
            "raw_point_coverage": float(raw_metrics["coverage"]),
            "touched_superpoint_count": int(len(touched_ids)),
            "closure_iou": float(closure_metrics["iou"]),
            "closure_precision": float(closure_metrics["precision"]),
            "closure_coverage": float(closure_metrics["coverage"]),
            "consensus_nonempty": bool(current is not None),
            "consensus_iou": float(current_metrics["iou"]),
            "consensus_precision": float(current_metrics["precision"]),
            "consensus_coverage": float(current_metrics["coverage"]),
            "oracle_superpoint_iou": float(oracle["iou"]),
            "oracle_superpoint_count": int(oracle["superpoint_count"]),
            "oracle_point_count": int(oracle["point_count"]),
        }
        track_rows.append(row)
        if instance_id > 0:
            rows_by_target[instance_id].append(row)
            if len(current_points):
                current_points_by_target[instance_id].append(current_points)

    instance_rows = []
    for instance_id, gt_size in sorted(gt_sizes.items()):
        matched = rows_by_target.get(instance_id, [])
        current_chunks = current_points_by_target.get(instance_id, [])
        union_points = np.unique(np.concatenate(current_chunks)) if current_chunks else np.empty(0, dtype=np.int64)
        union_iou = _point_metrics(union_points, gt_ids, instance_id, gt_size)["iou"]
        best_raw = max((row["raw_point_iou"] for row in matched), default=0.0)
        best_closure = max((row["closure_iou"] for row in matched), default=0.0)
        best_consensus = max((row["consensus_iou"] for row in matched), default=0.0)
        best_oracle = max((row["oracle_superpoint_iou"] for row in matched), default=0.0)
        attribution = classify_instance(best_raw, best_closure, best_consensus, best_oracle, union_iou, len(matched))
        instance_rows.append({
            "scene_name": scene_name,
            "gt_instance_id": int(instance_id),
            "gt_point_count": int(gt_size),
            "matched_source_track_count": int(len(matched)),
            "best_raw_point_iou": float(best_raw),
            "best_closure_iou": float(best_closure),
            "best_consensus_iou": float(best_consensus),
            "best_oracle_superpoint_iou": float(best_oracle),
            "union_consensus_iou": float(union_iou),
            "loss_attribution": attribution,
        })
    return track_rows, instance_rows


def _summary(track_rows, instance_rows):
    valid_tracks = [row for row in track_rows if row["best_gt_instance_id"] > 0]
    return {
        "source_track_count": len(track_rows),
        "valid_gt_matched_track_count": len(valid_tracks),
        "empty_after_consensus_count": sum(not row["consensus_nonempty"] for row in track_rows),
        "track_threshold_counts": {
            stage: {
                "iou25": sum(row[field] >= 0.25 for row in valid_tracks),
                "iou50": sum(row[field] >= 0.50 for row in valid_tracks),
            }
            for stage, field in {
                "raw_points": "raw_point_iou",
                "touched_superpoint_closure": "closure_iou",
                "current_consensus": "consensus_iou",
                "oracle_superpoint_subset": "oracle_superpoint_iou",
            }.items()
        },
        "instance_loss_attribution": dict(sorted(Counter(row["loss_attribution"] for row in instance_rows).items())),
        "instance_threshold_counts": {
            stage: {
                "iou25": sum(row[field] >= 0.25 for row in instance_rows),
                "iou50": sum(row[field] >= 0.50 for row in instance_rows),
            }
            for stage, field in {
                "raw_points": "best_raw_point_iou",
                "touched_superpoint_closure": "best_closure_iou",
                "current_consensus": "best_consensus_iou",
                "oracle_superpoint_subset": "best_oracle_superpoint_iou",
                "union_current_consensus": "union_consensus_iou",
            }.items()
        },
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--details-track-root", type=Path, required=True)
    parser.add_argument("--consensus-track-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；GT 只能用于离线共识失败归因。")
    for name in (
        "scene_list", "details_track_root", "consensus_track_root",
        "processed_scene_root", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    track_rows, instance_rows = [], []
    for index, scene_name in enumerate(scenes, start=1):
        scene_tracks, scene_instances = _scene_rows(scene_name, args)
        track_rows.extend(scene_tracks)
        instance_rows.extend(scene_instances)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_tracks)} 条轨迹", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "track_stage_failure_gt.csv", track_rows)
    _write_csv(args.output_dir / "instance_stage_failure_gt.csv", instance_rows)
    payload = {
        "diagnostic_type": "GT-only Details共识几何阶段失败归因；不是推理规则或正式AP。",
        "decision_constraint": "只在geometry_dev30提出结构假设；不得以GT选择轨迹、superpoint、阈值、分数或候选。",
        "scene_count": len(scenes),
        "summary": _summary(track_rows, instance_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

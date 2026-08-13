#!/usr/bin/env python3
"""为自动 mask 轨迹标注其相对 native 最终候选的无 GT 几何关系。"""

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def relation_from_overlap(track_size, candidate_sizes, intersections):
    """固定的保守几何分流；不读取 GT，阈值不在本脚本外自适应。"""
    if len(candidate_sizes) == 0 or track_size == 0:
        return {"route": "新增实例", "top_candidate_id": -1, "top_iou": 0.0, "track_inside_ratio": 0.0,
                "candidate_covered_ratio": 0.0, "overlap_candidate_count": 0}
    unions = track_size + candidate_sizes - intersections
    ious = intersections / np.maximum(1, unions)
    top = int(np.argmax(ious))
    inside = float(intersections[top] / max(1, track_size))
    covered = float(intersections[top] / max(1, candidate_sizes[top]))
    overlap_count = int(np.sum(ious >= 0.10))
    if inside <= 0.25:
        route = "新增实例"
    elif inside < 0.75 and overlap_count == 1:
        route = "边界竞争"
    else:
        route = "内部重复或冲突"
    return {
        "route": route,
        "top_candidate_id": top,
        "top_iou": float(ious[top]),
        "track_inside_ratio": inside,
        "candidate_covered_ratio": covered,
        "overlap_candidate_count": overlap_count,
    }


def _load_masks(root, scene_name):
    masks = np.load(f"{root / scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常")
    return masks


def _scene_records(scene_name, args):
    payload = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())
    masks = _load_masks(args.prediction_cache_dir, scene_name)
    if masks.shape[0] < masks.shape[1]:
        masks = masks.T
    sizes = masks.sum(axis=0, dtype=np.int64)
    records = []
    for track in payload["tracks"]:
        points = np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        intersections = masks[points].sum(axis=0, dtype=np.int64) if len(points) else np.zeros(masks.shape[1], dtype=np.int64)
        relation = relation_from_overlap(len(points), sizes, intersections)
        records.append({"scene_name": scene_name, "track_id": int(track["track_id"]), "point_count": int(len(points)),
                        "support_view_count": int(track["support_view_count"]), **relation})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "prediction_cache_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空：{args.output_root}")
    args.output_root.mkdir(parents=True)
    records = []
    for index, scene_name in enumerate(_read_scenes(args.scene_list), start=1):
        scene_records = _scene_records(scene_name, args)
        records.extend(scene_records)
        root = args.output_root / scene_name
        root.mkdir()
        (root / "automatic_track_geometry_relations.json").write_text(json.dumps(scene_records, ensure_ascii=False, indent=2) + "\n")
        print(f"[场景完成] {index}/48 {scene_name}: {len(scene_records)} 条关系", flush=True)
    summary = {"gt_usage": "不读取 GT；不写候选、不融合、不评分、不评测。", "track_count": len(records),
               "route_counts": {name: sum(row["route"] == name for row in records) for name in ("新增实例", "边界竞争", "内部重复或冲突")}}
    (args.output_root / "automatic_track_geometry_relation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

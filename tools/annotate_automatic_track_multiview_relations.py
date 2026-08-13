#!/usr/bin/env python3
"""为自动 mask 轨迹记录其与 native 候选的逐观测多视角关系，不读取 GT。"""

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


def summarize_observation_candidate_support(observation_points, masks, min_shared_points=3):
    """以各帧自动观测的点交集统计候选关系，不使用类别或 GT。"""
    candidate_count = masks.shape[1]
    if candidate_count == 0 or not observation_points:
        return {
            "top_candidate_id": -1,
            "top_candidate_support_view_count": 0,
            "top_candidate_support_view_ratio": 0.0,
            "top_candidate_mean_observation_inside_ratio": 0.0,
            "second_candidate_support_view_count": 0,
            "candidate_identity_margin": 0.0,
            "candidate_with_support_count": 0,
        }
    support_count = np.zeros(candidate_count, dtype=np.int64)
    inside_sum = np.zeros(candidate_count, dtype=np.float64)
    for points in observation_points:
        points = np.unique(np.asarray(points, dtype=np.int64))
        if len(points) == 0:
            continue
        intersections = masks[points].sum(axis=0, dtype=np.int64)
        supported = intersections >= int(min_shared_points)
        support_count += supported
        inside_sum += intersections / max(1, len(points))
    ordering = np.lexsort((np.arange(candidate_count), -inside_sum, -support_count))
    top = int(ordering[0])
    second_count = int(support_count[ordering[1]]) if candidate_count > 1 else 0
    top_count = int(support_count[top])
    return {
        "top_candidate_id": top if top_count else -1,
        "top_candidate_support_view_count": top_count,
        "top_candidate_support_view_ratio": float(top_count / max(1, len(observation_points))),
        "top_candidate_mean_observation_inside_ratio": float(inside_sum[top] / max(1, len(observation_points))),
        "second_candidate_support_view_count": second_count,
        "candidate_identity_margin": float((top_count - second_count) / max(1, top_count)),
        "candidate_with_support_count": int(np.sum(support_count > 0)),
    }


def _load_masks(root, scene_name):
    masks = np.load(root / f"{scene_name}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的最终预测 mask 维度异常：{masks.shape}")
    if masks.shape[0] < masks.shape[1]:
        masks = masks.T
    return masks


def _load_observation_points(scene_root):
    points = {}
    with (scene_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                points[int(record["observation_id"])] = np.asarray(
                    np.load(record["point_indices_path"])["point_indices"], dtype=np.int64
                )
    return points


def _scene_records(scene_name, args):
    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    observations = _load_observation_points(args.automatic_root / scene_name)
    masks = _load_masks(args.prediction_cache_dir, scene_name)
    records = []
    for track in tracks:
        source_points = [observations[observation_id] for observation_id in track["observation_ids"]]
        relation = summarize_observation_candidate_support(source_points, masks, args.min_shared_points)
        records.append({
            "scene_name": scene_name,
            "track_id": int(track["track_id"]),
            "support_view_count": int(track["support_view_count"]),
            "point_count": int(track["point_count"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "mean_edge_score": float(track["mean_edge_score"]),
            **relation,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--min_shared_points", type=int, default=3)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "automatic_root", "prediction_cache_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    args.output_root.mkdir(parents=True)
    records = []
    scenes = _read_scenes(args.scene_list)
    for index, scene_name in enumerate(scenes, start=1):
        scene_records = _scene_records(scene_name, args)
        records.extend(scene_records)
        root = args.output_root / scene_name
        root.mkdir()
        (root / "automatic_track_multiview_relations.json").write_text(
            json.dumps(scene_records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_records)} 条关系", flush=True)
    payload = {
        "gt_usage": "不读取 GT；不写候选、不融合、不评分、不评测。",
        "track_count": len(records),
        "with_candidate_multiview_support_count": sum(row["top_candidate_id"] >= 0 for row in records),
        "params": {"min_shared_points": args.min_shared_points},
    }
    (args.output_root / "automatic_track_multiview_relation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

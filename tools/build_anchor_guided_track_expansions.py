#!/usr/bin/env python3
"""用粗三维自动轨迹引导跨帧自动 mask 选择，输出待诊断的扩展区域。"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def select_mask_for_anchor(anchor_visible_points, observations, min_anchor_coverage, min_shared_points):
    """在单帧观测中选择最完整覆盖锚点可见部分的一个 mask。"""
    anchor_visible_points = np.unique(np.asarray(anchor_visible_points, dtype=np.int64))
    if len(anchor_visible_points) < min_shared_points:
        return None
    best = None
    for observation in observations:
        points = observation["points"]
        shared = int(np.intersect1d(anchor_visible_points, points, assume_unique=True).size)
        coverage = float(shared / len(anchor_visible_points))
        if shared < min_shared_points or coverage < min_anchor_coverage:
            continue
        rank = (coverage, float(observation["predicted_iou"]) * float(observation["stability_score"]), shared)
        if best is None or rank > best[0]:
            best = (rank, observation, coverage, shared)
    if best is None:
        return None
    _, observation, coverage, shared = best
    return {
        "observation_id": int(observation["observation_id"]),
        "frame_id": str(observation["frame_id"]),
        "frame_index": int(observation["frame_index"]),
        "anchor_coverage": coverage,
        "shared_anchor_point_count": shared,
        "points": observation["points"],
    }


def _load_observations(scene_root):
    by_frame = defaultdict(list)
    with (scene_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            by_frame[int(record["frame_index"])].append({
                "observation_id": int(record["observation_id"]),
                "frame_id": str(record["frame_id"]),
                "frame_index": int(record["frame_index"]),
                "predicted_iou": float(record["predicted_iou"]),
                "stability_score": float(record["stability_score"]),
                "points": np.unique(np.asarray(np.load(record["point_indices_path"])["point_indices"], dtype=np.int64)),
            })
    return by_frame


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    tracks = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    observations = _load_observations(args.automatic_root / scene_name)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, visibility = world.get_mesh_projections()
    visibility = visibility.detach().cpu().numpy().astype(bool)
    records = []
    expansion_dir = args.output_root / scene_name / "expanded_points"
    expansion_dir.mkdir(parents=True, exist_ok=False)
    for track in tracks:
        anchor_points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        selected = []
        for frame_index, frame_observations in observations.items():
            visible = anchor_points[visibility[frame_index, anchor_points]]
            picked = select_mask_for_anchor(
                visible, frame_observations, args.min_anchor_coverage, args.min_shared_points,
            )
            if picked is not None:
                selected.append(picked)
        selected = sorted(selected, key=lambda item: item["frame_index"])
        extra_points = [item["points"] for item in selected]
        expanded_points = np.unique(np.concatenate([anchor_points, *extra_points])) if extra_points else anchor_points
        path = expansion_dir / f"track{int(track['track_id']):04d}_expanded_points.npz"
        np.savez_compressed(path, point_indices=expanded_points)
        records.append({
            "scene_name": scene_name,
            "track_id": int(track["track_id"]),
            "anchor_points_path": str(track["points_path"]),
            "expanded_points_path": str(path),
            "anchor_point_count": int(len(anchor_points)),
            "expanded_point_count": int(len(expanded_points)),
            "added_point_count": int(len(expanded_points) - len(anchor_points)),
            "support_view_count": int(track["support_view_count"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "selected_observation_count": len(selected),
            "selected_frame_count": len({item["frame_index"] for item in selected}),
            "meets_minimum_support": bool(len(selected) >= args.min_support_views),
            "selected_observations": [
                {key: value for key, value in item.items() if key != "points"} for item in selected
            ],
        })
    del world, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--automatic_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--min_anchor_coverage", type=float, default=0.60)
    parser.add_argument("--min_shared_points", type=int, default=20)
    parser.add_argument("--min_support_views", type=int, default=2)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.min_anchor_coverage <= 1.0:
        raise SystemExit("--min_anchor_coverage 必须在零到一之间。")
    for name in ("scene_list", "track_root", "automatic_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    args.output_root.mkdir(parents=True)
    all_records = []
    for index, scene_name in enumerate(scenes, start=1):
        root = args.output_root / scene_name
        records_path = root / "anchor_guided_expansions.json"
        if args.skip_existing and records_path.is_file():
            records = json.loads(records_path.read_text())
            all_records.extend(records)
            print(f"[复用场景] {index}/{len(scenes)} {scene_name}: {len(records)} 条扩展区域", flush=True)
            continue
        root.mkdir()
        records = _scene_records(scene_name, args)
        all_records.extend(records)
        (root / "anchor_guided_expansions.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条扩展区域", flush=True)
    payload = {
        "gt_usage": "不读取 GT；不赋类别、不生成候选、不融合、不评分、不评测。",
        "track_count": len(all_records),
        "with_minimum_support_count": sum(record["meets_minimum_support"] for record in all_records),
        "with_added_points_count": sum(record["added_point_count"] > 0 for record in all_records),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "anchor_guided_expansion_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

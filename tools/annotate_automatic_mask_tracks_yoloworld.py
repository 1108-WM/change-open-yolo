#!/usr/bin/env python3
"""为类别无关自动 mask 轨迹附加冻结 YOLO-World 的多视角语义投票。

对象轨迹来自独立的 SAM 自动 mask；本脚本仅把其三维点重投影到原始检测框中累积类别
证据。不读取 GT、不生成候选、不融合、不评测。
"""

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


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def frame_class_votes(xs, ys, boxes, labels, scores):
    """每帧每类只保留最强框证据，避免重复框反复加票。"""
    votes = {}
    if len(xs) == 0:
        return votes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        if not inside.any():
            continue
        value = float(score) * float(inside.mean())
        label = int(label)
        votes[label] = max(votes.get(label, 0.0), value)
    return votes


def _scene_records(scene_name, args):
    from utils import WORLD_2_CAM

    track_payload = json.loads((args.track_root / scene_name / "automatic_tracks.json").read_text())
    predictions = torch.load(args.bboxes_2d_root / f"{scene_name}.pt", map_location="cpu")
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    records = []
    for track in track_payload["tracks"]:
        points = np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
        votes = defaultdict(float)
        support_views = defaultdict(int)
        used_frames = 0
        for frame_id in track["frame_ids"]:
            frame_id = str(frame_id)
            frame_index = frame_lookup.get(frame_id)
            prediction = predictions.get(frame_id)
            if frame_index is None or prediction is None:
                continue
            visible_points = points[visibility[frame_index, points]]
            if len(visible_points) == 0:
                continue
            coords = projections[frame_index, visible_points].astype(np.float32)
            xs = coords[:, 0] / float(scaling[1])
            ys = coords[:, 1] / float(scaling[0])
            boxes = _as_numpy(prediction["bbox"]).astype(np.float32)
            labels = _as_numpy(prediction["labels"]).astype(np.int64)
            scores = _as_numpy(prediction["scores"]).astype(np.float32)
            local_votes = frame_class_votes(xs, ys, boxes, labels, scores)
            if not local_votes:
                continue
            used_frames += 1
            for label, value in local_votes.items():
                votes[label] += value
                support_views[label] += 1
        ordered = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        best_label, best_vote = ordered[0] if ordered else (-1, 0.0)
        second_vote = ordered[1][1] if len(ordered) > 1 else 0.0
        total_vote = float(sum(votes.values()))
        records.append({
            "scene_name": scene_name,
            "track_id": int(track["track_id"]),
            "points_path": str(track["points_path"]),
            "support_view_count": int(track["support_view_count"]),
            "point_count": int(track["point_count"]),
            "mean_node_quality": float(track["mean_node_quality"]),
            "mean_predicted_iou": float(track["mean_predicted_iou"]),
            "mean_stability_score": float(track["mean_stability_score"]),
            "mean_edge_score": float(track["mean_edge_score"]),
            "semantic_frame_count": used_frames,
            "voted_class_index": int(best_label),
            "top_vote": float(best_vote),
            "second_vote": float(second_vote),
            "vote_total": total_vote,
            "vote_margin": float((best_vote - second_vote) / max(best_vote, 1e-6)),
            "voted_class_support_views": int(support_views.get(best_label, 0)),
            "class_votes": [{"class_index": int(label), "vote": float(value), "support_views": int(support_views[label])} for label, value in ordered[:8]],
        })
    del world, projections, visibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--bboxes_2d_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "bboxes_2d_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_records = []
    for index, scene_name in enumerate(scenes, start=1):
        records = _scene_records(scene_name, args)
        all_records.extend(records)
        scene_root = args.output_root / scene_name
        scene_root.mkdir()
        (scene_root / "automatic_track_yoloworld_semantics.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(records)} 条语义轨迹", flush=True)
    payload = {
        "gt_usage": "不读取 GT；输出仅供后续 GT-only 语义账本读取。",
        "decision_state": "不生成候选、不融合、不评分、不评测。",
        "scene_count": len(scenes),
        "track_count": len(all_records),
        "with_semantic_vote_count": sum(record["voted_class_index"] >= 0 for record in all_records),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "automatic_track_yoloworld_semantic_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

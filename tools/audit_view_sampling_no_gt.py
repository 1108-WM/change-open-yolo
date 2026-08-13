#!/usr/bin/env python3
"""审计 f30 首段采样与均匀采样的无 GT 相机覆盖差异。

本工具不加载 mask、GT、检测或候选；只读取已有相机位姿，回答辅助观测分支
使用“已按配置频率抽帧后的前 N 帧”是否会明显缩小时间与相机位置覆盖范围。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def select_loaded_frame_positions(total_frames, count, mode):
    """返回配置频率抽帧后的序列位置，不使用图像或 GT。"""
    if total_frames <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), int(total_frames))
    if mode == "first":
        return np.arange(count, dtype=np.int64)
    if mode == "uniform":
        return np.rint(np.linspace(0, total_frames - 1, count)).astype(np.int64)
    raise ValueError(f"未知采样方式：{mode}")


def _camera_centers(scene_root, raw_frame_ids):
    centers = []
    for frame_id in raw_frame_ids:
        pose_path = scene_root / "poses" / f"{int(frame_id)}.txt"
        try:
            pose = np.loadtxt(pose_path, dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                continue
            centers.append(np.linalg.inv(pose)[:3, 3])
        except (OSError, ValueError, np.linalg.LinAlgError):
            continue
    return np.asarray(centers, dtype=np.float64)


def _coverage_metrics(loaded_positions, total_loaded_frames, centers):
    temporal_span = 0.0
    if total_loaded_frames > 1 and len(loaded_positions):
        temporal_span = float((loaded_positions.max() - loaded_positions.min()) / (total_loaded_frames - 1))
    if len(centers) < 2:
        return {"temporal_span_ratio": temporal_span, "valid_pose_count": int(len(centers)), "camera_diameter_m": 0.0, "camera_path_length_m": 0.0}
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    return {
        "temporal_span_ratio": temporal_span,
        "valid_pose_count": int(len(centers)),
        "camera_diameter_m": float(distances.max()),
        "camera_path_length_m": path_length,
    }


def _mean(rows, key):
    values = [row[key] for row in rows]
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config_path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--view_count", type=int, default=30)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    for name in ("scene_list", "dataset_root", "config_path", "output_path"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_path.exists():
        raise SystemExit(f"输出已存在，为避免覆盖拒绝执行：{args.output_path}")
    config = yaml.safe_load(args.config_path.read_text())
    frequency = int(config["openyolo3d"]["frequency"])
    scenes = _read_scenes(args.scene_list)
    rows = []
    for scene_name in scenes:
        scene_root = args.dataset_root / scene_name
        raw_total = len(list((scene_root / "poses").glob("*.txt")))
        loaded_total = len(range(0, raw_total, frequency))
        item = {"scene_name": scene_name, "raw_frame_count": raw_total, "loaded_frame_count": loaded_total}
        for mode in ("first", "uniform"):
            positions = select_loaded_frame_positions(loaded_total, args.view_count, mode)
            raw_ids = positions * frequency
            item[mode] = {
                "loaded_positions": positions.tolist(),
                "raw_frame_ids": raw_ids.tolist(),
                **_coverage_metrics(positions, loaded_total, _camera_centers(scene_root, raw_ids)),
            }
        rows.append(item)
    payload = {
        "gt_usage": "不读取 GT、mask、检测或候选；仅审计相机位姿采样覆盖。",
        "decision_state": "用于决定是否值得重跑独立二维观测，不能当作候选质量或 AP。",
        "frequency": frequency,
        "view_count": int(args.view_count),
        "scene_count": len(rows),
        "summary": {
            mode: {key: _mean([row[mode] for row in rows], key) for key in ("temporal_span_ratio", "camera_diameter_m", "camera_path_length_m")}
            for mode in ("first", "uniform")
        },
        "scenes": rows,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare one ScanNet200 validation scene for the original Open-YOLO 3D code.

The official Open-YOLO 3D loader uses ``frequency=10`` and therefore only
opens RGB/depth frames ``0, 10, 20, ...`` while it uses the pose directory to
determine the original frame count.  This converter keeps every pose, writes
only those sampled RGB/depth frames, copies the official mesh, and creates the
ScanNet200 evaluator ground-truth file.  It never runs a model or evaluator.

The output is written atomically through a temporary scene directory so an
interrupted conversion can be safely resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


def _read_exact(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"unexpected end of SENS file: wanted {size}, got {len(value)}")
    return value


def _unpack(handle, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(handle, size))


def _read_matrix(handle) -> np.ndarray:
    return np.frombuffer(_read_exact(handle, 16 * 4), dtype="<f4").reshape(4, 4).copy()


def _write_matrix(path: Path, value: np.ndarray) -> None:
    np.savetxt(path, np.asarray(value), fmt="%.9f")


def _read_label_map(path: Path) -> dict[str, int]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        mapping: dict[str, int] = {}
        for row in rows:
            category = row.get("raw_category")
            value = row.get("id")
            if category and value and category not in mapping:
                mapping[category] = int(value)
    return mapping


def _build_ground_truth(raw_scene: Path, output_path: Path, label_map: dict[str, int]) -> dict:
    scene = raw_scene.name
    label_path = raw_scene / f"{scene}_vh_clean_2.labels.ply"
    segment_path = raw_scene / f"{scene}_vh_clean_2.0.010000.segs.json"
    aggregation_path = raw_scene / f"{scene}.aggregation.json"

    labels_vertex = PlyData.read(label_path)["vertex"].data
    semantic_labels = np.asarray(labels_vertex["label"], dtype=np.int64).copy()
    raw_segments = np.asarray(json.loads(segment_path.read_text())["segIndices"], dtype=np.int64)
    if semantic_labels.shape[0] != raw_segments.shape[0]:
        raise ValueError("label PLY and segment index counts differ")

    aggregation = json.loads(aggregation_path.read_text())
    missing_categories: set[str] = set()
    instance_labels = np.full(semantic_labels.shape[0], -1, dtype=np.int64)
    for group in aggregation["segGroups"]:
        occupied = np.isin(raw_segments, np.asarray(group["segments"], dtype=np.int64))
        instance_labels[occupied] = int(group["id"])
        category = str(group["label"])
        if category not in label_map:
            missing_categories.add(category)
        semantic_labels[occupied] = label_map.get(category, 0)

    gt = semantic_labels * 1000 + instance_labels + 1
    np.savetxt(output_path, gt.astype(np.int32), fmt="%d")
    return {
        "point_count": int(gt.shape[0]),
        "instance_count": int(np.unique(instance_labels[instance_labels >= 0]).size),
        "missing_raw_categories": sorted(missing_categories),
    }


def _decode_sens_sampled(sens_path: Path, scene_root: Path, frame_step: int) -> dict:
    color_root = scene_root / "color"
    depth_root = scene_root / "depth"
    pose_root = scene_root / "poses"
    color_root.mkdir(parents=True)
    depth_root.mkdir()
    pose_root.mkdir()

    with sens_path.open("rb") as handle:
        version = _unpack(handle, "<I")[0]
        if version != 4:
            raise ValueError(f"unsupported SENS version {version}")
        sensor_name_length = _unpack(handle, "<Q")[0]
        sensor_name = _read_exact(handle, sensor_name_length).decode("utf-8", errors="replace")
        intrinsic_color = _read_matrix(handle)
        _read_matrix(handle)  # extrinsic color
        intrinsic_depth = _read_matrix(handle)
        _read_matrix(handle)  # extrinsic depth
        color_compression = _unpack(handle, "<i")[0]
        depth_compression = _unpack(handle, "<i")[0]
        color_width, color_height = _unpack(handle, "<II")
        depth_width, depth_height = _unpack(handle, "<II")
        depth_shift = _unpack(handle, "<f")[0]
        frame_count = _unpack(handle, "<Q")[0]

        if color_compression != 2 or depth_compression != 1:
            raise ValueError(
                "only ScanNet JPEG color and zlib ushort depth are supported; "
                f"got color={color_compression}, depth={depth_compression}"
            )

        sampled_count = 0
        for frame_index in range(frame_count):
            pose = _read_matrix(handle)
            _unpack(handle, "<QQ")  # color/depth timestamps
            color_size, depth_size = _unpack(handle, "<QQ")

            # The official loader counts poses and then indexes sampled paths by
            # their original frame number, so every pose is retained.
            _write_matrix(pose_root / f"{frame_index}.txt", pose)
            if frame_index % frame_step == 0:
                color_bytes = _read_exact(handle, color_size)
                depth_bytes = _read_exact(handle, depth_size)
                (color_root / f"{frame_index}.jpg").write_bytes(color_bytes)
                depth = np.frombuffer(zlib.decompress(depth_bytes), dtype="<u2")
                expected = int(depth_width * depth_height)
                if depth.size != expected:
                    raise ValueError(
                        f"frame {frame_index}: depth has {depth.size} values, expected {expected}"
                    )
                Image.fromarray(depth.reshape(depth_height, depth_width)).save(
                    depth_root / f"{frame_index}.png"
                )
                sampled_count += 1
            else:
                handle.seek(color_size + depth_size, os.SEEK_CUR)

    _write_matrix(scene_root / "intrinsics.txt", intrinsic_color)
    return {
        "sensor_name": sensor_name,
        "frame_count": int(frame_count),
        "sampled_frame_count": int(sampled_count),
        "frame_step": int(frame_step),
        "color_resolution": [int(color_width), int(color_height)],
        "depth_resolution": [int(depth_width), int(depth_height)],
        "depth_shift": float(depth_shift),
        "intrinsic_depth": intrinsic_depth.tolist(),
    }


def prepare_scene(raw_scene: Path, output_root: Path, label_map_path: Path, frame_step: int, resume: bool) -> dict:
    scene = raw_scene.name
    if not raw_scene.is_dir() or not scene.startswith("scene"):
        raise ValueError(f"invalid raw scene directory: {raw_scene}")
    required = [
        raw_scene / f"{scene}.sens",
        raw_scene / f"{scene}.aggregation.json",
        raw_scene / f"{scene}_vh_clean_2.0.010000.segs.json",
        raw_scene / f"{scene}_vh_clean_2.labels.ply",
        raw_scene / f"{scene}_vh_clean_2.ply",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required official files: " + ", ".join(missing))

    final_root = output_root / scene
    gt_root = output_root / "ground_truth"
    gt_final = gt_root / f"{scene}.txt"
    manifest_path = final_root / "stream_prepare_manifest.json"
    if final_root.is_dir() and manifest_path.is_file() and gt_final.is_file():
        manifest = json.loads(manifest_path.read_text())
        if resume and manifest.get("frame_step") == frame_step:
            return {"scene_name": scene, "status": "already_prepared", **manifest}
        raise FileExistsError(f"prepared scene already exists: {final_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{scene}.tmp.{os.getpid()}"
    gt_staging = gt_root / f".{scene}.tmp.{os.getpid()}.txt"
    if staging.exists() or gt_staging.exists():
        raise FileExistsError(f"staging path already exists for {scene}; remove only that stale path")
    staging.mkdir()
    try:
        sens_summary = _decode_sens_sampled(raw_scene / f"{scene}.sens", staging, frame_step)
        gt_summary = _build_ground_truth(raw_scene, gt_staging, _read_label_map(label_map_path))
        shutil.copy2(raw_scene / f"{scene}_vh_clean_2.ply", staging / f"{scene}_vh_clean_2.ply")
        scene_txt = raw_scene / f"{scene}.txt"
        if scene_txt.is_file():
            shutil.copy2(scene_txt, staging / scene_txt.name)
        manifest = {
            "scene_name": scene,
            "split": "official_scannet200_validation",
            "frame_step": int(frame_step),
            "source_raw_scene": str(raw_scene.resolve()),
            "sens": sens_summary,
            "ground_truth": gt_summary,
            "operations": {
                "mask3d_inference": False,
                "yoloworld_inference": False,
                "evaluation": False,
            },
        }
        (staging / "stream_prepare_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(final_root)
        gt_staging.rename(gt_final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        gt_staging.unlink(missing_ok=True)
        raise
    return {"scene_name": scene, "status": "prepared", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scene-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be positive")
    result = prepare_scene(
        args.raw_scene_dir.resolve(),
        args.output_root.resolve(),
        args.label_map.resolve(),
        args.frame_step,
        args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

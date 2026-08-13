#!/usr/bin/env python3
"""Prepare one official ScanNet200 train scene for the Open-YOLO pipeline.

The converter is intentionally scene-local. It decodes the official ``.sens``
stream without loading the complete recording into memory, reproduces Mask3D's
12-column ScanNet200 array, and writes instance GT for later supervised label
generation. It does not run Mask3D, YOLO-World, SAM, or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_LIST = (
    PROJECT_ROOT
    / "_external/ESAM/ESAM-main/data/scannet200/meta_data/scannetv2_train.txt"
)
DEFAULT_LABEL_MAP = (
    PROJECT_ROOT
    / "_external/ESAM/ESAM-main/data/scannet200/meta_data/scannetv2-labels.combined.tsv"
)


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


def decode_sens(sens_path: Path, scene_root: Path) -> dict:
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

        for frame_index in range(frame_count):
            pose = _read_matrix(handle)
            _unpack(handle, "<QQ")  # color/depth timestamps
            color_size, depth_size = _unpack(handle, "<QQ")
            color_bytes = _read_exact(handle, color_size)
            depth_bytes = _read_exact(handle, depth_size)

            (color_root / f"{frame_index}.jpg").write_bytes(color_bytes)
            depth = np.frombuffer(zlib.decompress(depth_bytes), dtype="<u2")
            expected_depth_values = int(depth_width * depth_height)
            if depth.size != expected_depth_values:
                raise ValueError(
                    f"frame {frame_index}: depth has {depth.size} values, "
                    f"expected {expected_depth_values}"
                )
            Image.fromarray(depth.reshape(depth_height, depth_width)).save(
                depth_root / f"{frame_index}.png"
            )
            _write_matrix(pose_root / f"{frame_index}.txt", pose)

    _write_matrix(scene_root / "intrinsics.txt", intrinsic_color)
    return {
        "sensor_name": sensor_name,
        "frame_count": int(frame_count),
        "color_resolution": [int(color_width), int(color_height)],
        "depth_resolution": [int(depth_width), int(depth_height)],
        "depth_shift": float(depth_shift),
        "intrinsic_depth": intrinsic_depth.tolist(),
    }


def _ply_arrays(path: Path, *, normals: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertex = PlyData.read(path)["vertex"].data
    coords = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    colors = np.stack([vertex[name] for name in ("red", "green", "blue")], axis=1).astype(np.float32)
    labels = np.asarray(vertex["label"], dtype=np.int64) if "label" in vertex.dtype.names else None
    if not normals:
        return coords, colors, labels

    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("open3d is required; run with the openyolo3d conda environment") from exc
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    mesh_coords = np.asarray(mesh.vertices, dtype=np.float32)
    if mesh_coords.shape != coords.shape or not np.allclose(mesh_coords, coords):
        raise ValueError("PLY reader and Open3D vertex orders differ")
    normal_values = np.asarray(mesh.vertex_normals, dtype=np.float32)
    return coords, np.hstack((colors, normal_values)), labels


def _raw_category_to_id(label_map: Path) -> dict[str, int]:
    with label_map.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        mapping = {}
        for row in rows:
            category = row.get("raw_category")
            value = row.get("id")
            if category and value and category not in mapping:
                mapping[category] = int(value)
        return mapping


def build_processed_scene(raw_scene: Path, scene_root: Path, label_map: Path) -> dict:
    scene = raw_scene.name
    mesh_path = raw_scene / f"{scene}_vh_clean_2.ply"
    label_path = raw_scene / f"{scene}_vh_clean_2.labels.ply"
    segment_path = raw_scene / f"{scene}_vh_clean_2.0.010000.segs.json"
    aggregation_path = raw_scene / f"{scene}.aggregation.json"

    coords, features, _ = _ply_arrays(mesh_path, normals=True)
    label_coords, _, semantic_labels = _ply_arrays(label_path, normals=False)
    if semantic_labels is None or not np.allclose(coords, label_coords):
        raise ValueError("mesh and label PLY files do not have matching vertices")

    segments_payload = json.loads(segment_path.read_text())
    raw_segments = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    if raw_segments.shape[0] != coords.shape[0]:
        raise ValueError("segment index count does not match mesh vertices")
    segment_ids = np.unique(raw_segments, return_inverse=True)[1].astype(np.int64)

    instance_labels = np.full(coords.shape[0], -1, dtype=np.int64)
    semantic_labels = semantic_labels.copy()
    category_ids = _raw_category_to_id(label_map)
    aggregation = json.loads(aggregation_path.read_text())
    missing_categories = set()
    for group in aggregation["segGroups"]:
        occupied = np.isin(raw_segments, np.asarray(group["segments"], dtype=np.int64))
        instance_labels[occupied] = int(group["id"])
        category = str(group["label"])
        if category not in category_ids:
            missing_categories.add(category)
        semantic_labels[occupied] = category_ids.get(category, 0)

    processed = np.hstack(
        (
            coords,
            features,
            segment_ids[:, None],
            semantic_labels[:, None],
            instance_labels[:, None],
        )
    ).astype(np.float32)
    short_scene = scene.removeprefix("scene")
    np.save(scene_root / f"{short_scene}.npy", processed)
    gt = semantic_labels * 1000 + instance_labels + 1
    return {
        "point_count": int(processed.shape[0]),
        "processed_columns": int(processed.shape[1]),
        "superpoint_count": int(np.unique(segment_ids).size),
        "instance_count": int(np.unique(instance_labels[instance_labels >= 0]).size),
        "gt": gt.astype(np.int32),
        "missing_raw_categories": sorted(missing_categories),
    }


def _scene_names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def validate_split(scene: str, train_list: Path, validation_lists: list[Path]) -> None:
    if scene not in _scene_names(train_list):
        raise ValueError(f"{scene} is not in the official ScanNet200 train split")
    overlaps = [str(path) for path in validation_lists if path.is_file() and scene in _scene_names(path)]
    if overlaps:
        raise ValueError(f"{scene} overlaps validation/evaluation lists: {overlaps}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scene-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--validation-list", type=Path, action="append", default=[])
    args = parser.parse_args()

    raw_scene = args.raw_scene_dir.resolve()
    output_root = args.output_root.resolve()
    scene = raw_scene.name
    if not raw_scene.is_dir() or not scene.startswith("scene"):
        raise SystemExit(f"invalid raw scene directory: {raw_scene}")
    validate_split(scene, args.train_list.resolve(), [path.resolve() for path in args.validation_list])

    required = [
        raw_scene / f"{scene}.sens",
        raw_scene / f"{scene}.aggregation.json",
        raw_scene / f"{scene}_vh_clean_2.0.010000.segs.json",
        raw_scene / f"{scene}_vh_clean_2.labels.ply",
        raw_scene / f"{scene}_vh_clean_2.ply",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required official files: {missing}")

    final_root = output_root / scene
    if final_root.exists():
        raise SystemExit(f"output scene already exists: {final_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{scene}.tmp.{os.getpid()}"
    staging.mkdir()
    try:
        sens_summary = decode_sens(raw_scene / f"{scene}.sens", staging)
        processed_summary = build_processed_scene(raw_scene, staging, args.label_map.resolve())
        shutil.copy2(raw_scene / f"{scene}_vh_clean_2.ply", staging / f"{scene}_vh_clean_2.ply")
        scene_txt = raw_scene / f"{scene}.txt"
        if scene_txt.is_file():
            shutil.copy2(scene_txt, staging / scene_txt.name)

        gt_root = output_root / "ground_truth"
        gt_root.mkdir(exist_ok=True)
        gt_stage = gt_root / f".{scene}.tmp.{os.getpid()}.txt"
        np.savetxt(gt_stage, processed_summary.pop("gt"), fmt="%d")
        gt_final = gt_root / f"{scene}.txt"
        if gt_final.exists():
            raise SystemExit(f"ground truth already exists: {gt_final}")

        manifest = {
            "scene_name": scene,
            "split": "official_scannet200_train",
            "source_raw_scene": str(raw_scene),
            "sens": sens_summary,
            "processed": processed_summary,
            "operations": {
                "mask3d_inference": False,
                "yoloworld_inference": False,
                "sam_inference": False,
                "training_label_generation": False,
            },
        }
        (staging / "stream_prepare_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(final_root)
        gt_stage.rename(gt_final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

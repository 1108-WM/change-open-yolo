#!/usr/bin/env python3
"""Audit local ScanNet scene inputs against downloaded official v2 files.

The tool is intentionally read-only with respect to both input trees.  It
checks mesh identity, processed point xyz/RGB correspondence, and a small
set of RGB-D/pose samples decoded directly from each official ``.sens`` file.
"""

import argparse
import hashlib
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


JPEG_MEAN_ABSOLUTE_DIFFERENCE_TOLERANCE = 3.0


def _read_exact(handle, size):
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected EOF: needed {size} bytes, got {len(value)}")
    return value


def _read_u32(handle):
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle):
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_f32_matrix(handle):
    return np.frombuffer(_read_exact(handle, 64), dtype="<f4").reshape(4, 4)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ply_vertices(path):
    with path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            decoded = line.decode("ascii").strip()
            if decoded.startswith("element vertex "):
                vertex_count = int(decoded.rsplit(" ", 1)[1])
            if decoded == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"No vertex count in PLY: {path}")
        dtype = np.dtype(
            [
                ("xyz", "<f4", (3,)),
                ("rgb", "u1", (3,)),
                ("alpha", "u1"),
            ]
        )
        return np.fromfile(handle, dtype=dtype, count=vertex_count)


def _read_sens_index(path):
    """Read the SENS header and record locations of requested frames later."""
    with path.open("rb") as handle:
        version = _read_u32(handle)
        if version != 4:
            raise ValueError(f"Unsupported SENS version {version}: {path}")
        sensor_name_length = _read_u64(handle)
        _read_exact(handle, sensor_name_length)
        intrinsic_color = _read_f32_matrix(handle)
        _read_f32_matrix(handle)  # extrinsic_color
        intrinsic_depth = _read_f32_matrix(handle)
        _read_f32_matrix(handle)  # extrinsic_depth
        color_compression = _read_u32(handle)
        depth_compression = _read_u32(handle)
        color_width = _read_u32(handle)
        color_height = _read_u32(handle)
        depth_width = _read_u32(handle)
        depth_height = _read_u32(handle)
        depth_shift = struct.unpack("<f", _read_exact(handle, 4))[0]
        frame_count = _read_u64(handle)

        if color_compression != 2 or depth_compression != 1:
            raise ValueError(
                "Only ScanNet jpeg color + zlib_ushort depth is supported; "
                f"got color={color_compression}, depth={depth_compression}"
            )

        requested = {0, frame_count // 2, frame_count - 1}
        frames = {}
        for index in range(frame_count):
            pose = _read_f32_matrix(handle)
            _read_u64(handle)  # timestamp_color
            _read_u64(handle)  # timestamp_depth
            color_size = _read_u64(handle)
            depth_size = _read_u64(handle)
            color_offset = handle.tell()
            handle.seek(color_size, 1)
            depth_offset = handle.tell()
            handle.seek(depth_size, 1)
            if index in requested:
                frames[index] = {
                    "pose": pose,
                    "color_offset": color_offset,
                    "color_size": color_size,
                    "depth_offset": depth_offset,
                    "depth_size": depth_size,
                }

    return {
        "frame_count": frame_count,
        "intrinsic_color": intrinsic_color,
        "intrinsic_depth": intrinsic_depth,
        "color_shape": [color_height, color_width, 3],
        "depth_shape": [depth_height, depth_width],
        "depth_shift": depth_shift,
        "samples": frames,
    }


def _load_sens_sample(path, index_info, depth_shape):
    with path.open("rb") as handle:
        handle.seek(index_info["color_offset"])
        color_bytes = _read_exact(handle, index_info["color_size"])
        handle.seek(index_info["depth_offset"])
        depth_bytes = _read_exact(handle, index_info["depth_size"])
    color = np.asarray(Image.open(io.BytesIO(color_bytes)).convert("RGB"))
    depth = np.frombuffer(zlib.decompress(depth_bytes), dtype="<u2").reshape(depth_shape)
    return color, depth


def _image_difference(left, right):
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "shape_match": True,
        "mean_absolute_difference": float(delta.mean()),
        "max_absolute_difference": int(delta.max()),
        "exact_value_fraction": float((delta == 0).mean()),
    }


def _matrix_difference(left, right):
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.abs(left[finite] - right[finite])
    return {
        "shape_match": left.shape == right.shape,
        "max_absolute_difference_finite_entries": float(delta.max()) if delta.size else 0.0,
        "allclose_atol_1e-5": bool(np.allclose(left, right, rtol=0.0, atol=1e-5, equal_nan=True)),
    }


def _audit_scene(local_root, official_root, scene):
    local_scene = local_root / scene
    official_scene = official_root / "scans" / scene
    short_name = scene.removeprefix("scene")
    local_ply = local_scene / f"{scene}_vh_clean_2.ply"
    official_ply = official_scene / f"{scene}_vh_clean_2.ply"
    local_npy = local_scene / f"{short_name}.npy"
    official_sens = official_scene / f"{scene}.sens"

    required = [local_ply, official_ply, local_npy, official_sens]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"scene": scene, "passed": False, "missing": missing}

    local_hash = _sha256(local_ply)
    official_hash = _sha256(official_ply)
    vertices = _read_ply_vertices(official_ply)
    points = np.load(local_npy, mmap_mode="r")
    xyz_match = points.shape[0] == len(vertices) and np.array_equal(points[:, :3], vertices["xyz"])
    rgb_match = points.shape[0] == len(vertices) and np.array_equal(points[:, 3:6].astype(np.uint8), vertices["rgb"])

    sens = _read_sens_index(official_sens)
    local_color_count = len(list((local_scene / "color").glob("*.jpg")))
    local_depth_count = len(list((local_scene / "depth").glob("*.png")))
    local_pose_count = len(list((local_scene / "poses").glob("*.txt")))
    local_intrinsic = np.loadtxt(local_scene / "intrinsics.txt", dtype=np.float32)
    intrinsic_comparison = _matrix_difference(sens["intrinsic_color"], local_intrinsic)

    frame_reports = {}
    for index, index_info in sens["samples"].items():
        local_color = np.asarray(Image.open(local_scene / "color" / f"{index}.jpg").convert("RGB"))
        local_depth = np.asarray(Image.open(local_scene / "depth" / f"{index}.png"))
        local_pose = np.loadtxt(local_scene / "poses" / f"{index}.txt", dtype=np.float32)
        official_color, official_depth = _load_sens_sample(official_sens, index_info, tuple(sens["depth_shape"]))
        frame_reports[str(index)] = {
            "color": _image_difference(official_color, local_color),
            "depth": _image_difference(official_depth, local_depth),
            "pose": _matrix_difference(index_info["pose"], local_pose),
        }

    frames_pass = all(
        report["color"].get("shape_match")
        # The project color JPEGs were re-encoded during extraction.  Their
        # decoded pixels need not be byte-identical to the JPEG payload in
        # the official SENS file, unlike depth which is losslessly exported.
        and report["color"]["mean_absolute_difference"] <= JPEG_MEAN_ABSOLUTE_DIFFERENCE_TOLERANCE
        and report["depth"].get("shape_match")
        and report["depth"]["max_absolute_difference"] == 0
        and report["pose"]["allclose_atol_1e-5"]
        for report in frame_reports.values()
    )
    counts_match = sens["frame_count"] == local_color_count == local_depth_count == local_pose_count
    passed = (
        local_hash == official_hash
        and xyz_match
        and rgb_match
        and counts_match
        and intrinsic_comparison["allclose_atol_1e-5"]
        and frames_pass
    )
    return {
        "scene": scene,
        "passed": passed,
        "mesh": {
            "local_sha256": local_hash,
            "official_sha256": official_hash,
            "byte_identical": local_hash == official_hash,
            "vertex_count": int(len(vertices)),
            "npy_shape": list(points.shape),
            "npy_xyz_exact": bool(xyz_match),
            "npy_rgb_exact": bool(rgb_match),
        },
        "sens": {
            "frame_count": int(sens["frame_count"]),
            "local_color_count": local_color_count,
            "local_depth_count": local_depth_count,
            "local_pose_count": local_pose_count,
            "counts_match": counts_match,
            "color_shape": sens["color_shape"],
            "depth_shape": sens["depth_shape"],
            "depth_shift": sens["depth_shift"],
            "intrinsic_color_vs_local": intrinsic_comparison,
            "samples": frame_reports,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    scene_reports = [_audit_scene(args.local_root, args.official_root, scene) for scene in args.scenes]
    report = {
        "local_root": str(args.local_root),
        "official_root": str(args.official_root),
        "scenes": scene_reports,
        "passed": all(scene["passed"] for scene in scene_reports),
        "scope": (
            "Three-scene source audit only. This verifies local inference inputs against "
            "official samples; it does not prove provenance for every ScanNet200 scene."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.report)}, indent=2))
    for scene in scene_reports:
        print(f"{scene['scene']}: {'PASS' if scene['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()

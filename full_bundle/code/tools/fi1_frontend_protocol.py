"""No-GT contracts for the isolated FI1 frame/frontend factorial experiment."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


ARMS = {
    "A": {"sampling": "uniform30", "frontend": "automatic_sam"},
    "B": {"sampling": "original_stride5", "frontend": "automatic_sam"},
    "C": {"sampling": "uniform30", "frontend": "grounded_sam"},
    "D": {"sampling": "original_stride5", "frontend": "grounded_sam"},
}
PARAMETERS = {
    "historical_frequency": 10, "uniform_max_frames": 30, "dense_original_stride": 5,
    "sam_model_type": "vit_b", "points_per_side": 16,
    "pred_iou_threshold": 0.88, "stability_score_threshold": 0.95,
    "stability_score_offset": 1.0, "crop_n_layers": 0,
    "min_mask_area": 64, "min_visible_points": 20, "max_masks_per_frame": 40,
    "box_threshold": 0.35, "nms_iou": 0.30, "max_detections": 100,
    "gd_multimask_output": False, "grounded_class_usage": "provenance_only",
    "hierarchy_policy": "hierarchy_safe", "duplicate_min_coverage": 0.95,
    "depth_scale": 1000.0, "vis_depth_threshold": 0.05,
    "invalid_pose_policy": "retain_sampled_frame_with_zero_3D_visibility_and_logged_inference_skip",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n")
    temp.replace(path)


def freeze_json(path, value):
    """Identical replay is allowed; a changed contract needs another output root."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing to replace frozen contract: {path}")
    else:
        write_json(path, value)


def choose_scenes(training_scans, excluded_scans, count=20, seed="fi1-frontends-v1"):
    """One lexicographically first training scan per hash-ranked main scene."""
    if count < 1:
        raise ValueError("count must be positive")
    all_scans = set(training_scans) | set(excluded_scans)
    if any(re.fullmatch(r"scene\d{4}_\d{2}", s) is None for s in all_scans):
        raise ValueError("invalid ScanNet scan identity")
    excluded = {s.split("_")[0] for s in excluded_scans}
    representatives = {}
    for scene in sorted(set(training_scans)):
        main = scene.split("_")[0]
        if main not in excluded:
            representatives.setdefault(main, scene)
    ranked = sorted(representatives, key=lambda s: (digest([seed, s]), s))
    if len(ranked) < count:
        raise ValueError("not enough independent development main scenes")
    return [representatives[main] for main in ranked[:count]]


def frame_lists(original_frame_count, historical_frequency=10):
    if original_frame_count < 1 or historical_frequency < 1:
        raise ValueError("frame count and frequency must be positive")
    available = np.arange(0, original_frame_count, historical_frequency, dtype=np.int64)
    if len(available) > 30:
        available = available[np.rint(np.linspace(0, len(available) - 1, 30)).astype(np.int64)]
    uniform = available.tolist()
    dense = list(range(0, original_frame_count, 5))
    return {"A": uniform, "B": dense, "C": list(uniform), "D": list(dense)}


def build_scene_manifest(scene, root, processed_root):
    """Require original numeric pose IDs. Never silently subsample a sparse folder."""
    root = Path(root).resolve() / scene
    pose_root = root / "poses"
    if not pose_root.is_dir():
        raise FileNotFoundError(f"missing original poses: {pose_root}")
    ids = sorted(int(p.stem) for p in pose_root.glob("*.txt") if p.stem.isdigit())
    if not ids or ids != list(range(ids[-1] + 1)):
        raise ValueError(f"poses must retain consecutive ORIGINAL frame IDs: {pose_root}")
    arms = frame_lists(len(ids))
    domain = sorted(set(arms["A"]) | set(arms["B"]))
    mesh_paths = sorted(root.glob("*.ply"))
    if len(mesh_paths) != 1:
        raise ValueError(f"expected one unambiguous mesh at {root}")
    processed = Path(processed_root).resolve() / scene / (scene.replace("scene", "") + ".npy")
    files = {"mesh": mesh_paths[0], "intrinsics": root / "intrinsics.txt", "processed": processed}
    frames = []
    for index, frame_id in enumerate(domain):
        row = {"frame_id": str(frame_id), "frame_index": index}
        for name, subdir, suffix in (("color", "color", ".jpg"), ("depth", "depth", ".png"),
                                      ("pose", "poses", ".txt")):
            path = root / subdir / f"{frame_id}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            row[name] = {"path": str(path), "sha256": sha256_file(path)}
        pose = np.loadtxt(row["pose"]["path"])
        row["pose_valid"] = bool(pose.shape == (4, 4) and np.isfinite(pose).all()
                                 and abs(np.linalg.det(pose)) >= 1e-12)
        frames.append(row)
    receipts = {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in files.items()}
    return {"scene_name": scene, "original_frame_count": len(ids), "arms": arms,
            "frames": frames, "files": receipts, "frame_index_domain": "manifest.frames"}


def verify_scene_inputs(manifest):
    receipts = list(manifest["files"].values())
    for row in manifest["frames"]:
        receipts.extend(row[name] for name in ("color", "depth", "pose"))
    for item in receipts:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"input changed: {item['path']}")


def validate_observation(row, frame, point_count):
    if row["frame_id"] != frame["frame_id"] or row["frame_index"] != frame["frame_index"]:
        raise ValueError("observation frame ID/index mismatch")
    if any(k in row for k in ("class_id", "label", "semantic_score")):
        raise ValueError("frontend labels must remain nested provenance")
    for key in ("predicted_iou", "stability_score"):
        value = float(row[key])
        # SAM's predicted IoU is a regression output; it need not be <= 1.
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid SAM quality: {key}")
    with np.load(row["point_indices_path"], allow_pickle=False) as source:
        points = source["point_indices"]
    if points.ndim != 1 or points.dtype.kind not in "iu" or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError("invalid mesh point indices")
    if len(np.unique(points)) != len(points):
        raise ValueError("duplicate mesh point indices")

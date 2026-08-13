#!/usr/bin/env python3
"""Generate frozen Mask3D, YOLO-World, and native caches for one train scene."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MASK3D_ROOT = PROJECT_ROOT / "models" / "Mask3D"
if str(MASK3D_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK3D_ROOT))
YOLO_WORLD_ROOT = PROJECT_ROOT / "models" / "YOLO-World"
if str(YOLO_WORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_WORLD_ROOT))

def _read_scenes(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _require_empty_or_missing(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")


def _save_native(output_root: Path, scene: str, prediction) -> dict:
    masks, classes, scores = (
        value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        for value in prediction[:3]
    )
    if masks.ndim != 2:
        raise ValueError(f"native masks must be point-by-candidate, got shape {masks.shape}")
    if classes.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            "native classes and scores must be one-dimensional, got "
            f"{classes.shape} and {scores.shape}"
        )
    if masks.shape[1] != classes.shape[0] or classes.shape[0] != scores.shape[0]:
        raise ValueError(
            "native candidate dimensions disagree: "
            f"masks={masks.shape}, classes={classes.shape}, scores={scores.shape}"
        )
    keep = np.asarray(scores >= 0.0, dtype=bool)
    np.save(output_root / f"{scene}_pred_masks.npy", np.asarray(masks[:, keep], dtype=bool))
    np.save(output_root / f"{scene}_pred_classes.npy", np.asarray(classes[keep], dtype=np.int64))
    np.save(output_root / f"{scene}_pred_scores.npy", np.asarray(scores[keep], dtype=np.float32))
    return {
        "point_count": int(masks.shape[0]),
        "native_candidate_count": int(keep.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--train-list",
        type=Path,
        default=PROJECT_ROOT
        / "_external/ESAM/ESAM-main/data/scannet200/meta_data/scannetv2_train.txt",
    )
    parser.add_argument(
        "--config-path", type=Path, default=PROJECT_ROOT / "pretrained/config_scannet200.yaml"
    )
    args = parser.parse_args()

    scene = args.scene
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    train_list = args.train_list.resolve()
    config_path = args.config_path.resolve()
    if scene not in _read_scenes(train_list):
        raise SystemExit(f"{scene} is not an official ScanNet200 train scene")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; Mask3D/YOLO-World cache generation requires a working GPU")

    scene_root = dataset_root / scene
    processed_path = scene_root / f"{scene.removeprefix('scene')}.npy"
    required = [
        processed_path,
        scene_root / f"{scene}_vh_clean_2.ply",
        scene_root / "color",
        scene_root / "depth",
        scene_root / "poses",
        scene_root / "intrinsics.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"prepared scene is incomplete: {missing}")

    from utils import OpenYolo3D

    mask_root = output_root / "mask3d_masks"
    bbox_root = output_root / "yoloworld_bboxes_2d"
    native_root = output_root / "native_cache"
    for path in (mask_root, bbox_root, native_root):
        _require_empty_or_missing(path)
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(output_root / "mplconfig"))
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    depth_scale = float(config["openyolo3d"]["depth_scale"])
    model = OpenYolo3D(str(config_path))
    result = model.predict(
        path_2_scene_data=str(scene_root),
        depth_scale=depth_scale,
        datatype="mesh",
        processed_scene=str(processed_path),
        path_to_3d_masks=None,
        is_gt=False,
        path_to_2d_preds=str(bbox_root),
        save_2d_preds=True,
        reuse_2d_preds=False,
    )

    raw_masks, raw_scores = model.preds_3d
    torch.save(
        (raw_masks.detach().cpu(), raw_scores.detach().cpu()),
        mask_root / f"{scene}.pt",
    )
    native_summary = _save_native(native_root, scene, result[scene])
    bbox_payload = torch.load(bbox_root / f"{scene}.pt", map_location="cpu")
    if not isinstance(bbox_payload, dict) or "metadata" not in bbox_payload:
        raise ValueError("YOLO-World cache was not written with signed metadata")

    manifest = {
        "scene_name": scene,
        "split": "official_scannet200_train",
        "dataset_root": str(dataset_root),
        "config_path": str(config_path),
        "cache_contract": "Mask3D + YOLO-World only",
        "raw_mask3d_candidate_count": int(raw_scores.numel()),
        "yoloworld_frame_count": int(len(bbox_payload["predictions"])),
        **native_summary,
        "ground_truth_usage": "none",
        "sam_inference": False,
        "d2b_inference": False,
        "training_label_generation": False,
    }
    (output_root / "native_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))

    del result, model, raw_masks, raw_scores
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

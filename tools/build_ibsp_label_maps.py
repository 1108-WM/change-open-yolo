import argparse
import json
import os
import os.path as osp
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def _read_scene_names(args):
    if args.scene_names:
        return [item.strip() for item in args.scene_names.split(",") if item.strip()]
    if args.scene_split:
        with open(args.scene_split) as f:
            return [line.strip() for line in f if line.strip()]
    root = Path(args.source_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("scene"))


def _frame_id_from_mask_path(path):
    marker = "_frame"
    suffix = "_det"
    name = path.name
    if marker not in name or suffix not in name:
        raise ValueError(f"Cannot infer frame id from SAM mask filename: {path}")
    return name.split(marker, 1)[1].split(suffix, 1)[0]


def _mask_iou(left, right):
    intersection = int(np.logical_and(left, right).sum())
    if not intersection:
        return 0.0
    union = int(np.logical_or(left, right).sum())
    return float(intersection / max(union, 1))


def _deduplicate_masks(items, iou_threshold):
    kept = []
    for item in items:
        if all(_mask_iou(item["mask"], other["mask"]) < iou_threshold for other in kept):
            kept.append(item)
    return kept


def build_label_map(mask_paths, min_area, duplicate_iou):
    items = []
    for path in mask_paths:
        mask = imageio.imread(path)
        if mask.ndim != 2:
            raise ValueError(f"SAM mask must be single-channel: {path}")
        mask = mask > 0
        area = int(mask.sum())
        if area >= int(min_area):
            items.append({"path": str(path), "mask": mask, "area": area})
    if not items:
        return None, {"input_masks": len(mask_paths), "kept_masks": 0, "labels": 0, "coverage": 0.0}

    reference_shape = items[0]["mask"].shape
    if any(item["mask"].shape != reference_shape for item in items):
        raise ValueError("SAM masks from one frame must have identical image shape")

    items.sort(key=lambda item: (item["area"], item["path"]))
    items = _deduplicate_masks(items, float(duplicate_iou))
    labels = np.zeros(reference_shape, dtype=np.uint16)
    unassigned = np.ones(reference_shape, dtype=bool)
    for label_id, item in enumerate(items, start=1):
        claim = item["mask"] & unassigned
        labels[claim] = label_id
        unassigned[claim] = False
    stats = {
        "input_masks": len(mask_paths),
        "kept_masks": len(items),
        "labels": int(labels.max()),
        "coverage": float((labels > 0).mean()),
        "smallest_area": int(items[0]["area"]),
        "largest_area": int(items[-1]["area"]),
    }
    return labels, stats


def process_scene(scene_name, args):
    mask_dir = Path(args.source_root) / scene_name / args.mask_dir_name
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing SAM mask directory: {mask_dir}")
    by_frame = {}
    for path in sorted(mask_dir.glob(args.mask_glob)):
        by_frame.setdefault(_frame_id_from_mask_path(path), []).append(path)

    output_dir = Path(args.output_root) / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_id, mask_paths in sorted(by_frame.items(), key=lambda item: int(item[0])):
        labels, stats = build_label_map(mask_paths, args.min_mask_area, args.duplicate_iou)
        if labels is None:
            continue
        output_path = output_dir / f"{frame_id}.png"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}")
        imageio.imwrite(output_path, labels)
        frames.append({"frame_id": frame_id, "output_path": str(output_path), **stats})
    return {
        "scene_name": scene_name,
        "mask_dir": str(mask_dir),
        "frames": frames,
        "frame_count": len(frames),
        "input_masks": int(sum(item["input_masks"] for item in frames)),
        "kept_masks": int(sum(item["kept_masks"] for item in frames)),
        "mean_coverage": float(np.mean([item["coverage"] for item in frames])) if frames else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Flatten existing YOLO-World + SAM observations into IBSp per-frame instance label maps."
    )
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)
    parser.add_argument("--mask_dir_name", default="sam_fused_masks")
    parser.add_argument("--mask_glob", default="*_mask.png")
    parser.add_argument("--min_mask_area", default=64, type=int)
    parser.add_argument("--duplicate_iou", default=0.90, type=float)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()

    scene_names = _read_scene_names(args)
    if args.max_scenes is not None:
        scene_names = scene_names[: int(args.max_scenes)]
    os.makedirs(args.output_root, exist_ok=True)
    scenes = [process_scene(scene_name, args) for scene_name in tqdm(scene_names)]
    summary_path = Path(args.output_root) / "ibsp_label_maps_summary.json"
    with summary_path.open("w") as f:
        json.dump({"params": vars(args), "scenes": scenes}, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

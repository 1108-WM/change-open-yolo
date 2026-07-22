#!/usr/bin/env python3
"""Export dense YOLO-World + SAM observations for IBSp boundary evidence.

This exporter intentionally stays outside the AP/fusion path.  It keeps a
larger set of per-frame masks than ``export_sam_fused_proposals.py`` and writes
both individual mask evidence and non-overlapping instance label maps that can
be consumed by ``generate_geometric_superpoints.py --boundary_mask_root``.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


def _read_scene_names(scene_names, scene_split, max_scenes):
    if scene_names:
        names = [item.strip() for item in scene_names.split(",") if item.strip()]
    elif scene_split:
        with open(scene_split) as handle:
            names = [line.strip() for line in handle if line.strip()]
    else:
        from evaluate import SCENE_NAMES_SCANNET200

        names = list(SCENE_NAMES_SCANNET200)
    return names[:max_scenes] if max_scenes is not None else names


def _mask_iou(left, right):
    intersection = int(np.logical_and(left, right).sum())
    if not intersection:
        return 0.0
    return float(intersection / max(1, int(np.logical_or(left, right).sum())))


def _mask_containment(left, right):
    intersection = int(np.logical_and(left, right).sum())
    return float(intersection / max(1, min(int(left.sum()), int(right.sum()))))


def select_frame_observations(items, duplicate_iou=0.92, duplicate_containment=0.985, max_items=20):
    """Suppress duplicate masks while retaining nested small-object evidence."""
    ordered = sorted(
        items,
        key=lambda item: (-float(item["priority"]), int(item["area"]), int(item["detection_id"])),
    )
    kept = []
    for item in ordered:
        is_duplicate = any(
            _mask_iou(item["mask"], other["mask"]) >= float(duplicate_iou)
            or (
                _mask_containment(item["mask"], other["mask"]) >= float(duplicate_containment)
                and min(item["area"], other["area"]) / max(1, max(item["area"], other["area"])) >= 0.70
            )
            for other in kept
        )
        if not is_duplicate:
            kept.append(item)
        if len(kept) >= int(max_items):
            break
    return kept


def build_frame_label_map(items):
    """Assign compact per-frame ids; smaller masks claim overlap first."""
    if not items:
        return None, []
    shape = items[0]["mask"].shape
    if any(item["mask"].shape != shape for item in items):
        raise ValueError("All masks from a frame must have the same shape")
    labels = np.zeros(shape, dtype=np.uint16)
    available = np.ones(shape, dtype=bool)
    assigned = []
    for label_id, item in enumerate(sorted(items, key=lambda item: (item["area"], -item["priority"])), start=1):
        claim = item["mask"] & available
        labels[claim] = label_id
        available[claim] = False
        assigned.append((item, int(label_id), int(claim.sum())))
    return labels, assigned


def _scene_output_paths(output_root, scene_name):
    scene_root = Path(output_root) / scene_name
    return {
        "root": scene_root,
        "masks": scene_root / "masks",
        "label_maps": scene_root / "frame_label_maps",
        "observations": scene_root / "observations.jsonl",
        "summary": scene_root / "summary.json",
    }


def export_scene_dense_observations(openyolo3d, predictor, scene_name, output_root, args):
    paths = _scene_output_paths(output_root, scene_name)
    if paths["root"].exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {paths['root']}; use --overwrite to replace it")
    paths["masks"].mkdir(parents=True, exist_ok=True)
    paths["label_maps"].mkdir(parents=True, exist_ok=True)

    projections = _to_numpy(openyolo3d.mesh_projections[0]).astype(np.int64)
    visible = _to_numpy(openyolo3d.mesh_projections[1]).astype(bool)
    labels = openyolo3d.openyolo3d_config["network2d"]["text_prompts"]
    image_height, image_width = openyolo3d.world2cam.image_resolution
    frame_indices = list(range(0, len(openyolo3d.world2cam.color_paths), max(1, args.frame_stride)))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]

    records = []
    frame_stats = []
    raw_mask_count = 0
    skipped = defaultdict(int)
    for frame_idx in frame_indices:
        image_path = openyolo3d.world2cam.color_paths[frame_idx]
        frame_id = _frame_key(image_path)
        frame_prediction = openyolo3d.preds_2d.get(frame_id)
        if frame_prediction is None:
            skipped["missing_2d_prediction"] += 1
            continue
        boxes = _to_numpy(frame_prediction["bbox"]).astype(np.float32)
        class_ids = _to_numpy(frame_prediction["labels"]).astype(np.int64)
        scores = _to_numpy(frame_prediction["scores"]).astype(np.float32)
        order = np.argsort(-scores)[: args.max_detections_per_frame]
        image = _prepare_image(image_path)
        predictor.set_image(image)
        items = []
        for detection_id in order:
            score = float(scores[detection_id])
            if score < args.detection_score_th:
                skipped["low_detection_score"] += 1
                continue
            box = _clamp_box(boxes[detection_id], image_width, image_height)
            box_area_ratio = float((box[2] - box[0]) * (box[3] - box[1]) / max(1, image_width * image_height))
            if box_area_ratio > args.max_box_area_ratio:
                skipped["large_box"] += 1
                continue
            masks, sam_scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            for sam_rank, mask_id in enumerate(np.argsort(-sam_scores)[: args.sam_multimask_topk]):
                mask = masks[int(mask_id)].astype(bool)
                area = int(mask.sum())
                if area < args.min_mask_area:
                    skipped["small_mask"] += 1
                    continue
                point_indices = _sam_mask_to_indices(openyolo3d, frame_idx, mask, projections, visible)
                if len(point_indices) < args.min_visible_points:
                    skipped["few_visible_points"] += 1
                    continue
                sam_score = float(sam_scores[int(mask_id)])
                items.append(
                    {
                        "mask": mask,
                        "area": area,
                        "point_indices": point_indices,
                        "detection_id": int(detection_id),
                        "class_id": int(class_ids[detection_id]),
                        "class_name": _safe_label(labels, int(class_ids[detection_id])),
                        "score": score,
                        "sam_score": sam_score,
                        "sam_mask_id": int(mask_id),
                        "sam_rank": int(sam_rank),
                        "bbox_xyxy": [float(value) for value in box.tolist()],
                        "box_area_ratio": box_area_ratio,
                        "priority": float(score * max(0.05, sam_score) * np.log1p(len(point_indices))),
                    }
                )
        raw_mask_count += len(items)
        kept = select_frame_observations(
            items,
            duplicate_iou=args.duplicate_iou,
            duplicate_containment=args.duplicate_containment,
            max_items=args.max_masks_per_frame,
        )
        label_map, assigned = build_frame_label_map(kept)
        if label_map is None:
            continue
        label_path = paths["label_maps"] / f"{frame_id}.png"
        imageio.imwrite(label_path, label_map)
        for item, label_id, claimed_pixels in assigned:
            observation_id = len(records)
            mask_path = paths["masks"] / f"obs{observation_id:06d}_frame{frame_id}_det{item['detection_id']:03d}_sam{item['sam_mask_id']}_mask.png"
            points_path = paths["masks"] / f"obs{observation_id:06d}_points.npz"
            imageio.imwrite(mask_path, item.pop("mask").astype(np.uint8) * 255)
            np.savez_compressed(points_path, point_indices=item.pop("point_indices"))
            records.append(
                {
                    "observation_id": observation_id,
                    "scene_name": scene_name,
                    "frame_id": str(frame_id),
                    "frame_index": int(frame_idx),
                    "frame_label_id": label_id,
                    "claimed_pixels": claimed_pixels,
                    "mask_path": str(mask_path),
                    "point_indices_path": str(points_path),
                    **item,
                }
            )
        frame_stats.append(
            {
                "frame_id": str(frame_id),
                "frame_index": int(frame_idx),
                "raw_masks": len(items),
                "kept_masks": len(kept),
                "label_coverage": float((label_map > 0).mean()),
            }
        )

    with paths["observations"].open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "scene_name": scene_name,
        "frame_count": len(frame_stats),
        "observation_count": len(records),
        "raw_mask_count": raw_mask_count,
        "mean_masks_per_frame": float(np.mean([item["kept_masks"] for item in frame_stats])) if frame_stats else 0.0,
        "mean_label_coverage": float(np.mean([item["label_coverage"] for item in frame_stats])) if frame_stats else 0.0,
        "skipped": dict(sorted(skipped.items())),
        "frames": frame_stats,
        "params": {
            key: value
            for key, value in vars(args).items()
            if key not in {"scene_names", "scene_split", "max_scenes", "overwrite"}
        },
    }
    with paths["summary"].open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main():
    # Keep import-time dependencies light so frame-mask utilities are testable
    # without the full OpenYOLO runtime installed.
    global OpenYolo3D, _clamp_box, _frame_key, _load_sam_predictor
    global _prepare_image, _safe_label, _sam_mask_to_indices, _to_numpy
    from run_evaluation import load_yaml
    from utils import OpenYolo3D
    from export_sam_fused_proposals import (
        _clamp_box,
        _frame_key,
        _load_sam_predictor,
        _prepare_image,
        _safe_label,
        _sam_mask_to_indices,
        _to_numpy,
    )

    parser = argparse.ArgumentParser(description="Export dense per-frame YOLO-World + SAM instance evidence.")
    parser.add_argument("--dataset", default="scannet200", choices=("scannet200",))
    parser.add_argument("--path_to_3d_masks", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_source", required=True)
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument(
        "--allow_legacy_2d_cache",
        default=False,
        action="store_true",
        help="Allow the repository's existing pre-metadata YOLO-World cache files.",
    )
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", default=None, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--detection_score_th", default=0.25, type=float)
    parser.add_argument("--max_detections_per_frame", default=20, type=int)
    parser.add_argument("--max_box_area_ratio", default=0.85, type=float)
    parser.add_argument("--sam_multimask_topk", default=2, type=int)
    parser.add_argument("--min_mask_area", default=64, type=int)
    parser.add_argument("--min_visible_points", default=8, type=int)
    parser.add_argument("--duplicate_iou", default=0.92, type=float)
    parser.add_argument("--duplicate_containment", default=0.985, type=float)
    parser.add_argument("--max_masks_per_frame", default=20, type=int)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()

    if args.allow_legacy_2d_cache:
        os.environ["OPENYOLO3D_ALLOW_LEGACY_2D_CACHE"] = "1"

    config = load_yaml("./pretrained/config_scannet200.yaml")
    scene_names = _read_scene_names(args.scene_names, args.scene_split, args.max_scenes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _load_sam_predictor(args.sam_checkpoint, args.sam_model_type, device, args.sam_source)
    openyolo3d = OpenYolo3D("./pretrained/config_scannet200.yaml")
    os.makedirs(args.output_root, exist_ok=True)
    summaries = []
    start_time = time.time()
    for scene_name in tqdm(scene_names):
        scene_id = scene_name.replace("scene", "")
        openyolo3d.predict(
            path_2_scene_data=osp.join("./data/scannet200", scene_name),
            depth_scale=config["openyolo3d"]["depth_scale"],
            datatype="mesh",
            processed_scene=osp.join("./data/scannet200", scene_name, f"{scene_id}.npy"),
            path_to_3d_masks=args.path_to_3d_masks,
            is_gt=False,
            path_to_2d_preds=args.path_to_2d_preds,
            save_2d_preds=False,
            reuse_2d_preds=True,
        )
        summaries.append(export_scene_dense_observations(openyolo3d, predictor, scene_name, args.output_root, args))
    payload = {"elapsed_seconds": time.time() - start_time, "params": vars(args), "scenes": summaries}
    with open(osp.join(args.output_root, "dense_frame_instance_observations_summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

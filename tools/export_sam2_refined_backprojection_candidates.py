#!/usr/bin/env python3
"""Export GT-free SAM2 refined instances in the existing fusion-candidate format.

Semantic labels are inferred only from cached YOLO-World 2D detections that
overlap the SAM2 track masks.  The output is compatible with
``load_backprojection_candidates`` and is deliberately separate from AP
evaluation so geometry/semantic quality can be inspected first.
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


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _unpack_masks(path):
    payload = np.load(path)
    shape = tuple(int(value) for value in payload["mask_shape"])
    return np.unpackbits(payload["packed_masks"], axis=1, count=int(np.prod(shape[1:]))).reshape(shape).astype(bool)


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _box_iou(box, boxes):
    if not len(boxes):
        return np.zeros((0,), dtype=np.float32)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    box_areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area + box_areas - intersection, 1e-6)


def _mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _mask_inside_box_fraction(mask, box):
    height, width = mask.shape
    x1, y1, x2, y2 = np.rint(box).astype(np.int64)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1 or not mask.any():
        return 0.0
    return float(mask[y1:y2, x1:x2].sum() / mask.sum())


def _load_baseline_masks(path, scene_name, num_points):
    payload = torch.load(Path(path) / f"{scene_name}.pt", map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = _as_numpy(masks).astype(bool)
    if masks.shape[0] != num_points and masks.shape[1] == num_points:
        masks = masks.T
    if masks.shape[0] != num_points:
        raise ValueError(f"Baseline mask point count mismatch for {scene_name}: {masks.shape}")
    return masks


def _existing_overlap(indices, baseline_masks, num_points):
    mask = np.zeros((num_points,), dtype=bool)
    mask[indices] = True
    intersections = np.logical_and(baseline_masks, mask[:, None]).sum(axis=0)
    unions = baseline_masks.sum(axis=0) + len(indices) - intersections
    return {
        "best_existing_iou": float((intersections / np.maximum(unions, 1)).max(initial=0.0)),
        "seed_in_existing_mask_ratio": float(np.any(baseline_masks[indices], axis=1).mean()) if len(indices) else 0.0,
    }


def _semantic_evidence(source_track_ids, tracks, frame_ids, preds, min_box_iou, min_mask_box_coverage):
    evidence = defaultdict(float)
    support = defaultdict(list)
    for track_id in source_track_ids:
        track = tracks.get(int(track_id))
        if track is None:
            continue
        masks = _unpack_masks(track["mask_path"])
        if masks.shape[0] != len(frame_ids):
            raise ValueError(f"Frame count mismatch in {track['mask_path']}")
        for frame_index, mask in enumerate(masks):
            if not mask.any():
                continue
            frame_pred = preds.get(str(frame_ids[frame_index]))
            if frame_pred is None:
                continue
            boxes = _as_numpy(frame_pred["bbox"]).astype(np.float32)
            labels = _as_numpy(frame_pred["labels"]).astype(np.int64)
            scores = _as_numpy(frame_pred["scores"]).astype(np.float32)
            box = _mask_box(mask)
            if box is None or not len(boxes):
                continue
            ious = _box_iou(box, boxes)
            for det_id in np.flatnonzero(ious >= float(min_box_iou)):
                coverage = _mask_inside_box_fraction(mask, boxes[det_id])
                if coverage < float(min_mask_box_coverage):
                    continue
                class_id = int(labels[det_id])
                weight = float(scores[det_id]) * float(ious[det_id]) * coverage
                evidence[class_id] += weight
                support[class_id].append(
                    {
                        "frame_id": str(frame_ids[frame_index]),
                        "frame_index": int(frame_index),
                        "track_id": int(track_id),
                        "iou": float(ious[det_id]),
                        "score": float(scores[det_id]),
                        "mask_inside_box_fraction": coverage,
                        "bbox_xyxy": [float(value) for value in boxes[det_id]],
                    }
                )
    if not evidence:
        return None
    top_class = max(evidence, key=evidence.get)
    total = float(sum(evidence.values()))
    top = float(evidence[top_class])
    ranked = sorted(evidence.items(), key=lambda item: item[1], reverse=True)
    second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    selected = support[top_class]
    return {
        "class_id": int(top_class),
        "semantic_confidence": float(top / max(total, 1e-6)),
        "semantic_margin": float((top - second) / max(total, 1e-6)),
        "support_views": selected,
        "support_view_count": len({item["frame_id"] for item in selected}),
        "support_mean_iou": float(np.mean([item["iou"] for item in selected])) if selected else 0.0,
        "support_best_iou": float(np.max([item["iou"] for item in selected])) if selected else 0.0,
        "support_mean_score": float(np.mean([item["score"] for item in selected])) if selected else 0.0,
    }


def _scene_names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _export_scene(scene_name, args, prompts):
    refined_scene = args.refined_root / scene_name
    track_scene = args.track_root / scene_name
    records = _read_jsonl(refined_scene / "instances.jsonl")
    tracks = {int(item["track_id"]): item for item in _read_jsonl(track_scene / "tracks.jsonl")}
    frame_ids = json.loads((track_scene / "summary.json").read_text())["frame_ids"]
    preds = torch.load(args.bboxes_2d_root / f"{scene_name}.pt", map_location="cpu")
    scene_id = scene_name.removeprefix("scene")
    points = np.load(args.dataset_root / scene_name / f"{scene_id}.npy", mmap_mode="r")
    baseline_masks = _load_baseline_masks(args.baseline_masks_root, scene_name, len(points))
    output_scene = args.output_root / scene_name
    seed_dir = output_scene / "seed_points"
    seed_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    skipped = []
    for record in records:
        indices_path = Path(record["point_indices_path"])
        if not indices_path.is_absolute():
            indices_path = PROJECT_ROOT / indices_path
        indices = np.unique(np.load(indices_path)["point_indices"].astype(np.int64))
        indices = indices[(indices >= 0) & (indices < len(points))]
        semantic = _semantic_evidence(
            record.get("source_track_ids", []), tracks, frame_ids, preds,
            args.min_box_iou, args.min_mask_box_coverage,
        )
        if semantic is None:
            skipped.append({"instance_id": int(record["instance_id"]), "reason": "no_2d_semantic_evidence"})
            continue
        class_id = int(semantic["class_id"])
        if class_id < 0 or class_id >= len(prompts):
            skipped.append({"instance_id": int(record["instance_id"]), "reason": "class_out_of_prompt_range", "class_id": class_id})
            continue
        overlap = _existing_overlap(indices, baseline_masks, len(points))
        seed_path = seed_dir / f"sam2_instance{int(record['instance_id']):04d}.npz"
        np.savez_compressed(seed_path, point_indices=indices.astype(np.int32))
        geometry_confidence = min(1.0, float(record.get("support_score", 0.0)) / max(float(args.support_score_scale), 1e-6))
        score = float(semantic["support_mean_score"] * semantic["semantic_confidence"])
        fusion_score = float(score * (0.5 + 0.5 * geometry_confidence))
        candidates.append(
            {
                "scene_name": scene_name,
                "candidate_id": int(record["instance_id"]),
                "source_kind": "sam2_details",
                "class_id": class_id,
                "class_name": str(prompts[class_id]),
                "score": score,
                "fusion_score": fusion_score,
                "proposal_priority": fusion_score,
                "seed_points_path": str(seed_path),
                "num_seed_points": int(len(indices)),
                "support_score": float(record.get("support_score", 0.0)),
                "source_track_ids": [int(item) for item in record.get("source_track_ids", [])],
                "semantic_confidence": semantic["semantic_confidence"],
                "semantic_margin": semantic["semantic_margin"],
                **overlap,
                "support_view_count": semantic["support_view_count"],
                "support_mean_iou": semantic["support_mean_iou"],
                "support_best_iou": semantic["support_best_iou"],
                "support_mean_score": semantic["support_mean_score"],
                "support_views": semantic["support_views"],
            }
        )
    payload = {
        "scene_name": scene_name,
        "source_kind": "sam2_details",
        "gt_usage": "none",
        "candidates": candidates,
        "skipped": skipped,
        "params": {
            "min_box_iou": args.min_box_iou,
            "min_mask_box_coverage": args.min_mask_box_coverage,
            "support_score_scale": args.support_score_scale,
        },
    }
    (output_scene / "backprojection_candidates.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"scene_name": scene_name, "candidates": len(candidates), "skipped": len(skipped)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined_root", type=Path, required=True)
    parser.add_argument("--track_root", type=Path, required=True)
    parser.add_argument("--bboxes_2d_root", type=Path, required=True)
    parser.add_argument("--baseline_masks_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--scene_names", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--min_box_iou", type=float, default=0.20)
    parser.add_argument("--min_mask_box_coverage", type=float, default=0.35)
    parser.add_argument("--support_score_scale", type=float, default=100.0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    prompts = config["network2d"]["text_prompts"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [_export_scene(scene, args, prompts) for scene in _scene_names(args.scene_names)]
    (args.output_root / "summary.json").write_text(json.dumps({"scenes": summaries, "gt_usage": "none"}, indent=2) + "\n")
    print(json.dumps({"scenes": summaries, "gt_usage": "none"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export SAM2 refined instances with Open-YOLO 3D's native MVPDist labels.

The refined masks are category agnostic. This adapter reuses the same
multi-view label-map voting routine that Open-YOLO 3D applies to Mask3D masks,
instead of assigning a class from a single overlapping 2D detection box.
Ground truth is never read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import OpenYolo3D


def _scene_names(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _load_baseline_masks(root, scene_name, num_points):
    payload = torch.load(root / f"{scene_name}.pt", map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    masks = _as_numpy(masks).astype(bool)
    if masks.shape[0] != num_points and masks.shape[1] == num_points:
        masks = masks.T
    if masks.shape[0] != num_points:
        raise ValueError(f"Baseline point count mismatch for {scene_name}: {masks.shape}")
    return masks


def _existing_overlap(indices, baseline_masks, num_points):
    proposal = np.zeros((num_points,), dtype=bool)
    proposal[indices] = True
    intersections = np.logical_and(baseline_masks, proposal[:, None]).sum(axis=0)
    unions = baseline_masks.sum(axis=0) + len(indices) - intersections
    return {
        "best_existing_iou": float((intersections / np.maximum(unions, 1)).max(initial=0.0)),
        "seed_in_existing_mask_ratio": float(np.any(baseline_masks[indices], axis=1).mean()) if len(indices) else 0.0,
    }


def _candidate_indices(record, num_points):
    path = Path(record["point_indices_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    indices = np.unique(np.load(path)["point_indices"].astype(np.int64))
    return indices[(indices >= 0) & (indices < num_points)]


def _label_candidates(openyolo3d, candidate_masks):
    # The baseline config expands each Mask3D proposal to many class entries.
    # Refined instances need exactly one native MVPDist vote per candidate.
    topk_per_image = openyolo3d.openyolo3d_config["openyolo3d"].get("topk_per_image", -1)
    openyolo3d.openyolo3d_config["openyolo3d"]["topk_per_image"] = -1
    try:
        _, classes, scores = openyolo3d.label_3d_masks_from_label_maps(
            torch.from_numpy(candidate_masks),
            openyolo3d.preds_2d,
            openyolo3d.mesh_projections[0],
            openyolo3d.mesh_projections[1],
            is_gt=False,
        )
    finally:
        openyolo3d.openyolo3d_config["openyolo3d"]["topk_per_image"] = topk_per_image
    return _as_numpy(classes).astype(np.int64), _as_numpy(scores).astype(np.float32)


def _export_scene(scene_name, args, openyolo3d, labels):
    scene_id = scene_name.removeprefix("scene")
    points = np.load(args.dataset_root / scene_name / f"{scene_id}.npy", mmap_mode="r")
    records = _read_jsonl(args.refined_root / scene_name / "instances.jsonl")
    candidates_raw = [(record, _candidate_indices(record, len(points))) for record in records]
    candidates_raw = [(record, indices) for record, indices in candidates_raw if len(indices) >= args.min_seed_points]
    if not candidates_raw:
        return {"scene_name": scene_name, "candidates": 0}

    prediction = openyolo3d.predict(
        path_2_scene_data=str(args.dataset_root / scene_name),
        depth_scale=args.depth_scale,
        datatype="mesh",
        processed_scene=str(args.dataset_root / scene_name / f"{scene_id}.npy"),
        path_to_3d_masks=str(args.baseline_masks_root),
        path_to_2d_preds=str(args.bboxes_2d_root),
        reuse_2d_preds=True,
    )
    del prediction
    masks = np.zeros((len(points), len(candidates_raw)), dtype=bool)
    for index, (_, point_indices) in enumerate(candidates_raw):
        masks[point_indices, index] = True
    class_ids, semantic_scores = _label_candidates(openyolo3d, masks)
    if len(class_ids) != len(candidates_raw):
        raise RuntimeError(f"MVPDist candidate count mismatch in {scene_name}: {len(class_ids)} != {len(candidates_raw)}")

    baseline_masks = _load_baseline_masks(args.baseline_masks_root, scene_name, len(points))
    output_scene = args.output_root / scene_name
    seed_dir = output_scene / "seed_points"
    seed_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for index, ((record, point_indices), class_id, semantic_score) in enumerate(zip(candidates_raw, class_ids, semantic_scores)):
        if not 0 <= int(class_id) < len(labels):
            continue
        seed_path = seed_dir / f"sam2_instance{int(record['instance_id']):04d}.npz"
        np.savez_compressed(seed_path, point_indices=point_indices.astype(np.int32))
        overlap = _existing_overlap(point_indices, baseline_masks, len(points))
        geometry_confidence = min(1.0, float(record.get("support_score", 0.0)) / max(args.support_score_scale, 1e-6))
        score = float(semantic_score)
        exported.append(
            {
                "scene_name": scene_name,
                "candidate_id": int(record["instance_id"]),
                "source_kind": "sam2_details_mvpdist",
                "class_id": int(class_id),
                "class_name": str(labels[int(class_id)]),
                "score": score,
                "fusion_score": float(score * (0.5 + 0.5 * geometry_confidence)),
                "proposal_priority": float(score * (0.5 + 0.5 * geometry_confidence)),
                "seed_points_path": str(seed_path),
                "num_seed_points": int(len(point_indices)),
                "support_score": float(record.get("support_score", 0.0)),
                "source_track_ids": [int(item) for item in record.get("source_track_ids", [])],
                "semantic_method": "openyolo3d_mvpdist_label_map_vote",
                **overlap,
            }
        )
    payload = {
        "scene_name": scene_name,
        "source_kind": "sam2_details_mvpdist",
        "gt_usage": "none",
        "candidates": exported,
        "params": {"min_seed_points": args.min_seed_points, "support_score_scale": args.support_score_scale},
    }
    (output_scene / "backprojection_candidates.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"scene_name": scene_name, "candidates": len(exported)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--baseline_masks_root", type=Path, required=True)
    parser.add_argument("--bboxes_2d_root", type=Path, required=True)
    parser.add_argument("--scene_names", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--config", default="pretrained/config_scannet200.yaml")
    parser.add_argument("--depth_scale", default=1000.0, type=float)
    parser.add_argument("--min_seed_points", default=100, type=int)
    parser.add_argument("--support_score_scale", default=100.0, type=float)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    openyolo3d = OpenYolo3D(args.config)
    labels = openyolo3d.openyolo3d_config["network2d"]["text_prompts"]
    summaries = [_export_scene(scene, args, openyolo3d, labels) for scene in _scene_names(args.scene_names)]
    (args.output_root / "summary.json").write_text(json.dumps({"scenes": summaries, "gt_usage": "none"}, indent=2) + "\n")
    print(json.dumps({"scenes": summaries, "gt_usage": "none"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adapt refined dense-instance tracks to the existing fusion candidate format.

The refined tracks already contain 3D point indices and a semantic vote.  This
tool only serializes that evidence for the established candidate append path;
it neither reads ScanNet200 ground truth nor changes the baseline masks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def _read_scene_names(scene_names, scene_split, refined_root):
    if scene_names:
        return [item.strip() for item in scene_names.split(",") if item.strip()]
    if scene_split:
        with open(scene_split) as handle:
            return [line.strip() for line in handle if line.strip()]
    return sorted(path.name for path in Path(refined_root).glob("scene*") if path.is_dir())


def _load_observations(path):
    with open(path) as handle:
        return {
            int(item["observation_id"]): item
            for line in handle
            if line.strip()
            for item in (json.loads(line),)
        }


def _semantic_confidence(instance, observations):
    """Use mean detector x SAM confidence, keeping scores on the native 0--1 scale."""
    values = []
    for observation_id in instance.get("observation_ids", []):
        observation = observations.get(int(observation_id))
        if observation is None:
            continue
        detector = float(observation.get("score", 0.0))
        sam = float(observation.get("sam_score", 0.0))
        values.append(max(0.0, min(1.0, detector * sam)))
    return float(np.mean(values)) if values else 0.0


def build_candidate(scene_name, instance, observations, label_to_prediction_id):
    class_name = str(instance.get("class_name", ""))
    if class_name not in label_to_prediction_id:
        return None
    point_path = instance.get("point_indices_path")
    if not point_path or not Path(point_path).is_file():
        return None
    try:
        point_count = int(instance.get("point_count", 0))
    except (TypeError, ValueError):
        point_count = 0
    score = _semantic_confidence(instance, observations)
    frame_count = int(instance.get("frame_count", 0))
    observation_ids = [int(item) for item in instance.get("observation_ids", [])]
    return {
        "candidate_id": int(instance.get("instance_id", -1)),
        "scene_name": scene_name,
        "source_kind": "dense_ibsp",
        "class_id": int(label_to_prediction_id[class_name]),
        "class_name": class_name,
        "score": score,
        "fusion_score": score,
        "proposal_priority": score * max(1, frame_count),
        "refined_seed_points_path": str(point_path),
        "num_seed_points": point_count,
        "support_view_count": frame_count,
        "cluster_observation_count": len(observation_ids),
        "track_score": float(instance.get("score", 0.0)),
        "semantic_vote_score": float(instance.get("semantic_vote_score", 0.0)),
        "best_existing_iou": 0.0,
        "seed_in_existing_mask_ratio": 0.0,
    }


def export_scene_candidates(scene_name, refined_root, output_root, label_to_prediction_id, min_points, min_support_views):
    source_scene = Path(refined_root) / scene_name
    payload_path = source_scene / "refined_instances.json"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = json.loads(payload_path.read_text())
    observations = _load_observations(payload["source_observations"])
    candidates = []
    skipped = []
    for instance in payload.get("instances", []):
        if int(instance.get("point_count", 0)) < int(min_points):
            skipped.append({"instance_id": instance.get("instance_id"), "reason": "few_points"})
            continue
        if int(instance.get("frame_count", 0)) < int(min_support_views):
            skipped.append({"instance_id": instance.get("instance_id"), "reason": "few_support_views"})
            continue
        candidate = build_candidate(scene_name, instance, observations, label_to_prediction_id)
        if candidate is None:
            skipped.append({"instance_id": instance.get("instance_id"), "reason": "invalid_class_or_point_path"})
            continue
        candidates.append(candidate)
    output_scene = Path(output_root) / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)
    with (output_scene / "backprojection_candidates.json").open("w") as handle:
        json.dump(
            {
                "scene_name": scene_name,
                "source_kind": "dense_ibsp",
                "num_candidates": len(candidates),
                "candidates": candidates,
                "skipped": skipped,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return {"scene_name": scene_name, "candidates": len(candidates), "skipped": len(skipped)}


def main():
    parser = argparse.ArgumentParser(description="Export refined dense instances as fusion candidates.")
    parser.add_argument("--refined_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--labels_config", default="./pretrained/config_scannet200.yaml")
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--min_points", default=100, type=int)
    parser.add_argument("--min_support_views", default=2, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_root}; use --overwrite to replace it")
    with open(args.labels_config) as handle:
        labels = yaml.safe_load(handle)["network2d"]["text_prompts"]
    label_to_prediction_id = {str(label): index for index, label in enumerate(labels)}
    scenes = _read_scene_names(args.scene_names, args.scene_split, args.refined_root)
    summaries = [
        export_scene_candidates(
            scene_name,
            args.refined_root,
            output_root,
            label_to_prediction_id,
            args.min_points,
            args.min_support_views,
        )
        for scene_name in tqdm(scenes)
    ]
    with (output_root / "refined_dense_candidates_summary.json").open("w") as handle:
        json.dump({"params": vars(args), "scenes": summaries}, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare scene-specific next-round Any3DIS seed lists without GT.

The input track export records all reliable IBSp ids.  The post-track instances
record which ids were already claimed after consensus and cleanup.  Their set
difference is the next-round seed pool described by Any3DIS iterative object
sampling.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _scene_names(args):
    if args.scene_names:
        return [item.strip() for item in args.scene_names.split(",") if item.strip()]
    if args.scene_split:
        return [line.strip() for line in Path(args.scene_split).read_text().splitlines() if line.strip()]
    return sorted(path.name for path in Path(args.track_root).glob("scene*") if path.is_dir())


def _baseline_superpoint_coverage(scene_name, superpoint_root, baseline_masks_root):
    """Return the fraction of each superpoint covered by existing 3D masks.

    This is a GT-free novelty check.  It deliberately uses only the baseline
    proposal masks already available at inference, so subsequent SAM2 rounds
    spend their limited seed budget on regions the baseline did not claim.
    """
    scene_id = scene_name.replace("scene", "")
    array_path = Path(superpoint_root) / scene_name / f"{scene_id}.npy"
    mask_path = Path(baseline_masks_root) / f"{scene_name}.pt"
    if not array_path.is_file():
        raise FileNotFoundError(f"Missing superpoint array: {array_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing baseline masks: {mask_path}")
    superpoints = np.load(array_path, mmap_mode="r")[:, 9].astype(np.int64, copy=False)
    payload = torch.load(mask_path, map_location="cpu")
    masks = payload[0] if isinstance(payload, (tuple, list)) else payload
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError(f"Expected NxM baseline masks for {scene_name}, got {masks.shape}")
    if masks.shape[0] != len(superpoints) and masks.shape[1] == len(superpoints):
        masks = masks.T
    if masks.shape[0] != len(superpoints):
        raise ValueError(f"Baseline mask point count mismatch for {scene_name}: {masks.shape}")
    sizes = np.bincount(superpoints, minlength=int(superpoints.max(initial=-1)) + 1)
    covered = np.any(masks, axis=1)
    covered_counts = np.bincount(superpoints[covered], minlength=len(sizes))
    return covered_counts / np.maximum(sizes, 1)


def main():
    parser = argparse.ArgumentParser(description="Select reliable IBSp superpoints not claimed by postprocessed SAM2 instances.")
    parser.add_argument("--track_root", required=True)
    parser.add_argument("--postprocess_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--scene_split", default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--baseline_masks_root", default=None, help="Optional existing Mask3D proposal masks used only for GT-free seed novelty filtering.")
    parser.add_argument("--superpoint_root", default=None, help="Required with --baseline_masks_root; contains the inference-time IBSp arrays.")
    parser.add_argument("--max_baseline_superpoint_coverage", type=float, default=0.70, help="Exclude reliable seeds whose points are mostly covered by baseline masks.")
    args = parser.parse_args()
    scenes = _scene_names(args)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if bool(args.baseline_masks_root) != bool(args.superpoint_root):
        raise ValueError("Use --baseline_masks_root and --superpoint_root together.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene_name in scenes:
        track_summary = json.loads((Path(args.track_root) / scene_name / "summary.json").read_text())
        instances_path = Path(args.postprocess_root) / scene_name / "instances.jsonl"
        instances = _read_jsonl(instances_path) if instances_path.is_file() else []
        reliable = {int(item) for item in track_summary.get("reliable_superpoint_ids", [])}
        claimed = {
            int(segment_id)
            for instance in instances
            for segment_id in instance.get("superpoint_ids", [])
        }
        remaining = sorted(reliable - claimed)
        baseline_coverage = None
        baseline_excluded = set()
        if args.baseline_masks_root:
            baseline_coverage = _baseline_superpoint_coverage(
                scene_name,
                args.superpoint_root,
                args.baseline_masks_root,
            )
            baseline_excluded = {
                segment_id
                for segment_id in remaining
                if segment_id < len(baseline_coverage)
                and float(baseline_coverage[segment_id]) > float(args.max_baseline_superpoint_coverage)
            }
            remaining = [segment_id for segment_id in remaining if segment_id not in baseline_excluded]
        (output_root / f"{scene_name}.txt").write_text("".join(f"{item}\n" for item in remaining))
        summaries.append(
            {
                "scene_name": scene_name,
                "reliable_superpoint_count": len(reliable),
                "claimed_superpoint_count": len(claimed & reliable),
                "remaining_seed_count": len(remaining),
                "baseline_excluded_seed_count": len(baseline_excluded),
                "remaining_seed_mean_baseline_coverage": (
                    float(np.mean([baseline_coverage[item] for item in remaining]))
                    if baseline_coverage is not None and remaining
                    else None
                ),
            }
        )
    (output_root / "summary.json").write_text(json.dumps({"scenes": summaries, "params": vars(args)}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

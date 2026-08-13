#!/usr/bin/env python3
"""Build the N1a no-GT 3D observation-candidate ledger.

N1a deliberately keeps every SAM multimask observation.  This tool only
projects each frozen observation onto the *original* ScanNet superpoints and
writes its evidence/lineage.  It neither selects a mask nor merges views into
a proposal: those operations belong to N1b/N2 and must not be influenced by
GT.  Empty lifted observations are retained in the ledger as explicit failed
geometry evidence.
"""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTRACT = (
    "N1a no-GT observation-candidate ledger: preserve every frozen SAM "
    "hypothesis and its 3D superpoint evidence; do not select hypotheses, "
    "merge views, materialize proposals, read GT/native/semantics, or compute AP."
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _scenes(path):
    scenes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or duplicated")
    return scenes


def lift_visible_observation_to_superpoints(
    point_indices,
    superpoints,
    superpoint_sizes,
    visible_counts,
    min_visible_ratio,
    min_mask_support,
):
    """Return raw-superpoint support for one projected 2D SAM observation."""
    points = np.unique(np.asarray(point_indices, dtype=np.int64))
    points = points[(points >= 0) & (points < len(superpoints))]
    ids, inside_counts = np.unique(superpoints[points], return_counts=True)
    lifted = []
    evidence = []
    for superpoint_id, inside_count in zip(ids, inside_counts):
        superpoint_id = int(superpoint_id)
        visible_count = int(visible_counts.get(superpoint_id, 0))
        total_count = int(superpoint_sizes.get(superpoint_id, 0))
        visible_ratio = float(visible_count / max(1, total_count))
        mask_support = float(int(inside_count) / max(1, visible_count))
        accepted = (
            visible_count > 0
            and visible_ratio >= float(min_visible_ratio)
            and mask_support >= float(min_mask_support)
        )
        evidence.append({
            "superpoint_id": superpoint_id,
            "inside_point_count": int(inside_count),
            "visible_point_count": visible_count,
            "total_point_count": total_count,
            "visible_ratio": visible_ratio,
            "mask_support_ratio": mask_support,
            "accepted": bool(accepted),
        })
        if accepted:
            lifted.append(superpoint_id)
    return sorted(lifted), evidence


def _visible_counts(superpoints, visibility, frame_indices):
    result = {}
    for frame_index in sorted(set(map(int, frame_indices))):
        ids, counts = np.unique(superpoints[np.flatnonzero(visibility[frame_index])], return_counts=True)
        result[frame_index] = {int(i): int(c) for i, c in zip(ids, counts)}
    return result


def _read_scene_observations(prompt_root, scene):
    pattern = f"{scene}_seed*/observations/{scene}/prompt_observations.jsonl"
    paths = sorted((prompt_root / "batches").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"{scene} has no N1a prompt observation batches")
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scene_name") != scene:
                raise ValueError(f"observation scene mismatch in {path}")
            rows.append(row)
    keys = [(int(r["track_id"]), int(r["frame_index"]), int(r["hypothesis_index"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{scene} has duplicate seed/frame/hypothesis observations")
    by_request = defaultdict(list)
    for row in rows:
        by_request[(int(row["track_id"]), int(row["frame_index"]))].append(int(row["hypothesis_index"]))
    for request, hypotheses in by_request.items():
        if sorted(hypotheses) != [0, 1, 2]:
            raise ValueError(f"{scene} request {request} does not contain exactly frozen hypotheses 0/1/2")
    return sorted(rows, key=lambda r: (int(r["track_id"]), int(r["frame_index"]), int(r["hypothesis_index"])))


def _scene(scene, args):
    from utils import WORLD_2_CAM

    seed_rows = [json.loads(line) for line in (args.seed_ledger_root / scene / "seed_view_ledger.jsonl").read_text().splitlines() if line.strip()]
    eligible_seeds = {int(row["seed_superpoint_id"]) for row in seed_rows if row.get("views")}
    observations = _read_scene_observations(args.prompt_root, scene)
    observed_seeds = {int(row["track_id"]) for row in observations}
    if observed_seeds != eligible_seeds:
        raise ValueError(f"{scene} N1a seed coverage differs from frozen eligible seed ledger")

    processed = np.load(args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy", mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene} lacks raw superpoint column")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, sizes = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {int(i): int(n) for i, n in zip(ids, sizes)}
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    _, visible_raw = world.get_mesh_projections()
    visibility = visible_raw.detach().cpu().numpy().astype(bool, copy=False)
    visible_counts = _visible_counts(superpoints, visibility, [row["frame_index"] for row in observations])

    candidates = []
    empty = 0
    seed_not_retained = 0
    for candidate_index, row in enumerate(observations):
        points_path = Path(row["point_indices_path"])
        if not points_path.is_file():
            raise FileNotFoundError(f"missing projected points: {points_path}")
        points = np.asarray(np.load(points_path)["point_indices"], dtype=np.int64)
        lifted, evidence = lift_visible_observation_to_superpoints(
            points, superpoints, superpoint_sizes, visible_counts[int(row["frame_index"])],
            args.min_superpoint_visible_ratio, args.min_mask_support,
        )
        seed_id = int(row["prompt_superpoint_id"])
        seed_retained = seed_id in set(lifted)
        empty += int(not lifted)
        seed_not_retained += int(not seed_retained)
        candidates.append({
            "candidate_id": f"n1a_obs_{scene}_{candidate_index:07d}",
            "candidate_index": candidate_index,
            "scene_name": scene,
            "candidate_kind": "single_view_single_sam_multimask_observation",
            "candidate_family_key": f"seed_{seed_id}",
            "seed_superpoint_id": seed_id,
            "seed_retained_after_lift": seed_retained,
            "frame_id": str(row["frame_id"]),
            "frame_index": int(row["frame_index"]),
            "hypothesis_index": int(row["hypothesis_index"]),
            "sam_predicted_iou": float(row["sam_predicted_iou"]),
            "mask_area": int(row["mask_area"]),
            "backprojected_point_count": int(len(np.unique(points))),
            "candidate_superpoint_ids": lifted,
            "candidate_superpoint_count": len(lifted),
            "candidate_is_empty": not bool(lifted),
            "superpoint_evidence": evidence,
            "source_observation_id": int(row["observation_id"]),
            "source_point_indices_path": str(points_path),
            "source_mask_rle": row["mask_rle"],
            "source_visible_core_support_ratio": float(row["visible_core_support_ratio"]),
            "source_observation_core_purity_ratio": float(row["observation_core_purity_ratio"]),
            "source_prompt_point_backprojected": bool(row["prompt_point_backprojected"]),
            "ground_truth_usage": "none",
            "proposal_materialization_applied": False,
            "ap_computed": False,
            "decision_constraint": CONTRACT,
        })

    staging = args.output_root / f".{scene}.tmp.{os.getpid()}"
    staging.mkdir(parents=True)
    with (staging / "observation_candidate_ledger.jsonl").open("w") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "scene_name": scene,
        "eligible_seed_count": len(eligible_seeds),
        "observation_candidate_count": len(candidates),
        "candidate_family_count": len(observed_seeds),
        "empty_observation_candidate_count": empty,
        "seed_not_retained_after_lift_count": seed_not_retained,
        "ground_truth_usage": "none",
        "proposal_materialization_applied": False,
        "ap_computed": False,
    }
    (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(staging, args.output_root / scene)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--seed-ledger-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-superpoint-visible-ratio", type=float, default=0.10)
    parser.add_argument("--min-mask-support", type=float, default=0.30)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "seed_ledger_root", "prompt_root", "processed_scene_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if not 0 < args.min_superpoint_visible_ratio <= 1 or not 0 < args.min_mask_support <= 1:
        raise SystemExit("support thresholds must lie in (0, 1]")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for index, scene in enumerate(scenes, 1):
            existing = args.output_root / scene / "summary.json"
            summary = json.loads(existing.read_text()) if args.resume and existing.is_file() else _scene(scene, args)
            summaries.append(summary)
            print(f"[candidate-ledger] {index}/{len(scenes)} {scene}: {summary['observation_candidate_count']} observations", flush=True)
    except Exception:
        for temp in args.output_root.glob(".*.tmp.*"):
            shutil.rmtree(temp)
        raise
    payload = {
        "diagnostic_type": "N1a paper-reference no-GT single-view SAM observation candidate ledger",
        "decision_constraint": CONTRACT,
        "scene_count": len(summaries),
        "observation_candidate_count": sum(row["observation_candidate_count"] for row in summaries),
        "empty_observation_candidate_count": sum(row["empty_observation_candidate_count"] for row in summaries),
        "seed_not_retained_after_lift_count": sum(row["seed_not_retained_after_lift_count"] for row in summaries),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

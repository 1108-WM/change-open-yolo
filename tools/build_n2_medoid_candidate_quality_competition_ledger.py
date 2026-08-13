#!/usr/bin/env python3
"""Build a no-GT flood/quality/competition ledger for frozen N2 medoids.

This deliberately diagnoses fixed candidates only.  It never uses GT and it
never deletes, merges, re-scores, or evaluates a candidate.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    "No-GT N2 candidate flood ledger only; frozen candidates are not deleted, "
    "merged, rescored, or evaluated."
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def scenes(path):
    result = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or duplicated")
    return result


def overlap(left, right):
    left, right = set(left), set(right)
    intersection = len(left & right)
    return (
        intersection / max(1, len(left | right)),
        intersection / max(1, len(left)),
        intersection / max(1, len(right)),
        intersection,
    )


def superpoint_index(superpoints):
    """Create point indices per raw superpoint without repeated full scans."""
    order = np.argsort(superpoints, kind="stable")
    sorted_ids = superpoints[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    ends = np.r_[starts[1:], len(order)]
    return {
        int(sorted_ids[start]): order[start:end]
        for start, end in zip(starts, ends)
    }


def centroid_connectivity_proxy(ids, centers, sizes, radius):
    """Return a clearly-labelled centroid connectivity proxy, not a mutation rule."""
    ids = list(ids)
    if not ids:
        return 0, 0.0
    if len(ids) == 1:
        return 1, 1.0
    points = np.asarray([centers[sp] for sp in ids], dtype=np.float32)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    parent = list(range(len(ids)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            if distance[left, right] <= radius:
                join(left, right)
    components = defaultdict(int)
    for index, superpoint_id in enumerate(ids):
        components[root(index)] += int(sizes[superpoint_id])
    component_sizes = list(components.values())
    return len(component_sizes), max(component_sizes) / max(1, sum(component_sizes))


def relation_state(jaccard, left_covered, right_covered, threshold):
    if not jaccard:
        return "nonoverlap"
    if jaccard >= threshold:
        return "near_duplicate_or_containment"
    if left_covered > 0.99 or right_covered > 0.99:
        return "strict_containment_below_near_duplicate_threshold"
    return "partial_overlap"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--n2-cache-root", type=Path, required=True)
    parser.add_argument("--variant-ledger-root", type=Path, required=True)
    parser.add_argument("--candidate-ledger-root", type=Path, required=True)
    parser.add_argument("--d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.5)
    parser.add_argument("--centroid-connectivity-radius", type=float, default=0.15)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "scene_list", "n2_cache_root", "variant_ledger_root", "candidate_ledger_root",
        "d2b_track_root", "native_prediction_cache", "processed_scene_root", "output_root",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if not 0 < args.near_duplicate_jaccard <= 1:
        raise SystemExit("near-duplicate Jaccard must lie in (0, 1]")
    if args.centroid_connectivity_radius <= 0:
        raise SystemExit("centroid connectivity radius must be positive")
    if args.scene_offset < 0:
        raise SystemExit("scene offset must be non-negative")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("output root is non-empty; use --resume only for this exact ledger contract")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    chosen_scenes = scenes(args.scene_list)[args.scene_offset:]
    if args.max_scenes is not None:
        chosen_scenes = chosen_scenes[:args.max_scenes]

    for ordinal, scene in enumerate(chosen_scenes, 1):
        existing_summary = args.output_root / scene / "summary.json"
        if args.resume and existing_summary.is_file():
            summaries.append(json.loads(existing_summary.read_text()))
            print(f"[N2 flood ledger] {ordinal}/{len(chosen_scenes)} {scene}: already complete", flush=True)
            continue
        cache = json.loads((args.n2_cache_root / scene / "n2_medoid_candidates.json").read_text())["candidates"]
        variants = {
            row["candidate_family_key"]: row
            for row in (
                json.loads(line)
                for line in (args.variant_ledger_root / scene / "family_formation_variant_ledger.jsonl").read_text().splitlines()
                if line
            )
            if row["variant_name"] == "cross_view_consistency_medoid"
        }
        observations = {
            row["candidate_id"]: row
            for row in (
                json.loads(line)
                for line in (args.candidate_ledger_root / scene / "observation_candidate_ledger.jsonl").read_text().splitlines()
                if line
            )
        }
        tracks = json.loads((args.d2b_track_root / scene / "automatic_tracks.json").read_text())["tracks"]
        data = np.load(args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy", mmap_mode="r")
        superpoints = np.asarray(data[:, 9], dtype=np.int64)
        points_by_superpoint = superpoint_index(superpoints)
        superpoint_sizes = {sp: len(points) for sp, points in points_by_superpoint.items()}
        centers = {sp: np.asarray(data[points, :3], dtype=np.float32).mean(axis=0) for sp, points in points_by_superpoint.items()}

        native_masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
        native_masks = native_masks if native_masks.shape[0] == len(superpoints) else native_masks.T
        native_sizes = native_masks.sum(axis=0, dtype=np.int64)
        native_count_by_superpoint = {
            sp: np.asarray(native_masks[points].sum(axis=0, dtype=np.int64))
            for sp, points in points_by_superpoint.items()
        }
        candidate_sets = [set(map(int, row["superpoint_ids"])) for row in cache]
        parent = list(range(len(cache)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def join(left, right):
            left, right = root(left), root(right)
            if left != right:
                parent[right] = left

        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        pair_count = 0
        nonzero_pair_count = 0
        near_duplicate_pair_count = 0
        with (stage / "n2_pair_relation_ledger.jsonl").open("w") as handle:
            for left in range(len(cache)):
                for right in range(left + 1, len(cache)):
                    pair_count += 1
                    jaccard, left_covered, right_covered, intersection = overlap(candidate_sets[left], candidate_sets[right])
                    if not intersection:
                        continue
                    nonzero_pair_count += 1
                    state = relation_state(jaccard, left_covered, right_covered, args.near_duplicate_jaccard)
                    if jaccard >= args.near_duplicate_jaccard:
                        join(left, right)
                        near_duplicate_pair_count += 1
                    handle.write(json.dumps({
                        "scene_name": scene,
                        "left_candidate_id": left,
                        "right_candidate_id": right,
                        "intersection_superpoint_count": intersection,
                        "jaccard": jaccard,
                        "left_covered_ratio": left_covered,
                        "right_covered_ratio": right_covered,
                        "relation_state": state,
                        "ground_truth_usage": "none",
                        "proposal_materialization_applied": False,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
        components = defaultdict(list)
        for candidate_id in range(len(cache)):
            components[root(candidate_id)].append(candidate_id)

        rows = []
        for candidate_id, candidate in enumerate(cache):
            candidate_set = candidate_sets[candidate_id]
            variant = variants[candidate["canonical_family_key"]]
            medoid_observation_id = variant["source_candidate_ids"][0]
            observation = observations[medoid_observation_id]
            point_count = sum(superpoint_sizes[sp] for sp in candidate_set)
            native_intersection = np.zeros(len(native_sizes), dtype=np.int64)
            for superpoint_id in candidate_set:
                native_intersection += native_count_by_superpoint[superpoint_id]
            native_iou = native_intersection / np.maximum(1, point_count + native_sizes - native_intersection)
            best_native_id = int(np.argmax(native_iou)) if len(native_iou) else -1
            best_native_intersection = int(native_intersection[best_native_id]) if best_native_id >= 0 else 0

            best_d2b = max(
                (overlap(candidate_set, track["superpoint_ids"]) + (-int(track["track_id"]),) for track in tracks),
                default=(0.0, 0.0, 0.0, 0, 1),
            )
            point_indices = np.concatenate([points_by_superpoint[sp] for sp in candidate_set]) if candidate_set else np.empty(0, dtype=np.int64)
            rgb = np.asarray(data[point_indices, 3:6], dtype=np.float32) if len(point_indices) else np.empty((0, 3), dtype=np.float32)
            if len(rgb) and rgb.max(initial=0.0) > 1.5:
                rgb /= 255.0
            normals = np.asarray(data[point_indices, 6:9], dtype=np.float32) if len(point_indices) else np.empty((0, 3), dtype=np.float32)
            normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(normal_lengths, 1e-6)
            component_count, largest_component_ratio = centroid_connectivity_proxy(
                candidate_set, centers, superpoint_sizes, args.centroid_connectivity_radius
            )
            component_id = min(components[root(candidate_id)])
            seed_id = int(variant["seed_superpoint_id"])
            seed_point_count = superpoint_sizes.get(seed_id, 0)
            rows.append({
                "scene_name": scene,
                "candidate_id": candidate_id,
                "canonical_family_key": candidate["canonical_family_key"],
                "point_count": point_count,
                "superpoint_count": len(candidate_set),
                "seed_point_count": seed_point_count,
                "candidate_seed_scale_ratio": point_count / max(1, seed_point_count),
                "eligible_view_count": variant["eligible_view_count"],
                "medoid_cross_view_mean_jaccard": variant["medoid_cross_view_mean_jaccard"],
                "sam_predicted_iou": observation["sam_predicted_iou"],
                "source_backprojected_point_count": observation["backprojected_point_count"],
                "source_prompt_point_backprojected": observation["source_prompt_point_backprojected"],
                "source_visible_core_support_ratio": observation["source_visible_core_support_ratio"],
                "source_observation_core_purity_ratio": observation["source_observation_core_purity_ratio"],
                "depth_evidence_kind": "frozen_backprojection_success_and_visible_core_support_only",
                "rgb_channel_std_normalized": float(rgb.std()) if len(rgb) else None,
                "normal_channel_std": float(normals.std()) if len(normals) else None,
                "normal_mean_resultant_length": float(np.linalg.norm(normals.mean(axis=0))) if len(normals) else None,
                "centroid_connectivity_proxy_radius": args.centroid_connectivity_radius,
                "centroid_connectivity_proxy_component_count": component_count,
                "centroid_connectivity_proxy_largest_component_point_ratio": largest_component_ratio,
                "near_duplicate_component_id": component_id,
                "near_duplicate_component_size": len(components[root(candidate_id)]),
                "best_d2b_track_id": -best_d2b[4],
                "best_d2b_jaccard": best_d2b[0],
                "best_d2b_candidate_covered_ratio": best_d2b[1],
                "best_d2b_track_covered_ratio": best_d2b[2],
                "best_native_candidate_id": best_native_id,
                "best_native_iou": float(native_iou[best_native_id]) if best_native_id >= 0 else 0.0,
                "best_native_candidate_covered_ratio": best_native_intersection / max(1, point_count),
                "best_native_covered_ratio": best_native_intersection / max(1, int(native_sizes[best_native_id])) if best_native_id >= 0 else 0.0,
                "ground_truth_usage": "none",
                "proposal_materialization_applied": False,
                "ap_computed": False,
                "decision_constraint": CONTRACT,
            })
        with (stage / "candidate_quality_competition_ledger.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "scene_name": scene,
            "candidate_count": len(rows),
            "candidate_pair_count": pair_count,
            "nonzero_overlap_pair_count": nonzero_pair_count,
            "near_duplicate_pair_count": near_duplicate_pair_count,
            "near_duplicate_component_count": len(components),
            "ground_truth_usage": "none",
            "proposal_materialization_applied": False,
            "ap_computed": False,
        }
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        os.replace(stage, args.output_root / scene)
        summaries.append(summary)
        print(f"[N2 flood ledger] {ordinal}/{len(chosen_scenes)} {scene}: {len(rows)} candidates", flush=True)
    complete_summaries = []
    for scene_dir in sorted(args.output_root.glob("scene*")):
        summary_path = scene_dir / "summary.json"
        if summary_path.is_file():
            complete_summaries.append(json.loads(summary_path.read_text()))
    root = {
        "diagnostic_type": "no-GT N2 medoid candidate quality/flood/competition ledger",
        "decision_constraint": CONTRACT,
        "scene_count": len(complete_summaries),
        "candidate_count": sum(item["candidate_count"] for item in complete_summaries),
        "candidate_pair_count": sum(item["candidate_pair_count"] for item in complete_summaries),
        "nonzero_overlap_pair_count": sum(item["nonzero_overlap_pair_count"] for item in complete_summaries),
        "near_duplicate_pair_count": sum(item["near_duplicate_pair_count"] for item in complete_summaries),
        "near_duplicate_component_count": sum(item["near_duplicate_component_count"] for item in complete_summaries),
        "proposal_materialization_applied": False,
        "ap_computed": False,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(root, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

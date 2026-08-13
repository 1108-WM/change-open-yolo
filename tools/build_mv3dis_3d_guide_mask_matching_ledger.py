#!/usr/bin/env python3
"""Build the no-GT MV3DIS 3D-guide mask-matching ledger for frozen D2b.

This M1a stage implements the published mask-matching equations only.  It uses
strict frame visibility > 0.3 and mask visibility > 0.9, constructs a coverage
vector over every frozen D2b proposal, and computes per-guide and cross-guide
mask consistency scores.  The current Open-YOLO projection cache exposes only
binary absolute-depth visibility, not MV3DIS's continuous relative-depth
weight.  Consequently this tool records a ``binary_visibility_adapter`` and
does not claim exact MV3DIS region refinement.

No NMS, boundary reassignment, proposal mutation, GT, class, semantic score,
Mask3D proposal, or AP result is read or produced.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FRAME_VISIBILITY_THRESHOLD = 0.30
MASK_VISIBILITY_THRESHOLD = 0.90
DEPTH_WEIGHT_ADAPTER = "binary_visibility_adapter"


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _point_index_digest(point_indices):
    values = np.asarray(point_indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _sum_for_superpoints(counts, superpoint_ids):
    return int(sum(int(counts.get(int(item), 0)) for item in superpoint_ids))


def _cosine_similarity(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("candidate coverage vectors must have non-zero norms")
    return float(np.dot(left, right) / denominator)


def _normalize_proposals(proposals, superpoint_sizes):
    normalized = []
    proposal_ids = []
    lineage_ids = []
    known_superpoints = set(map(int, superpoint_sizes))
    for raw in sorted(proposals, key=lambda row: int(row["proposal_id"])):
        proposal_id = int(raw["proposal_id"])
        superpoint_ids = list(map(int, raw.get("superpoint_ids", [])))
        if not superpoint_ids or superpoint_ids != sorted(set(superpoint_ids)):
            raise ValueError(f"proposal {proposal_id} has invalid superpoint IDs")
        missing = set(superpoint_ids) - known_superpoints
        if missing:
            raise ValueError(
                f"proposal {proposal_id} references unknown superpoints: {sorted(missing)[:5]}"
            )
        point_count = int(sum(superpoint_sizes[item] for item in superpoint_ids))
        if int(raw.get("point_count", point_count)) != point_count:
            raise ValueError(f"proposal {proposal_id} has an inconsistent point count")
        lineage = list(map(int, raw.get("lineage_proposal_ids", [proposal_id])))
        if not lineage or lineage != sorted(set(lineage)):
            raise ValueError(f"proposal {proposal_id} has invalid lineage")
        proposal_ids.append(proposal_id)
        lineage_ids.extend(lineage)
        row = dict(raw)
        row.update({
            "proposal_id": proposal_id,
            "superpoint_ids": superpoint_ids,
            "point_count": point_count,
            "lineage_proposal_ids": lineage,
        })
        normalized.append(row)
    if not normalized:
        raise ValueError("frozen D2b proposal set is empty")
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("proposal IDs are not unique")
    if len(lineage_ids) != len(set(lineage_ids)):
        raise ValueError("D2b lineage is not conserved exactly once")
    return normalized


def _normalize_observations(observations, frame_visible_counts, superpoint_sizes):
    normalized = []
    observation_ids = []
    known_superpoints = set(map(int, superpoint_sizes))
    for raw in sorted(observations, key=lambda row: int(row["observation_id"])):
        observation_id = int(raw["observation_id"])
        frame_index = int(raw["frame_index"])
        if frame_index not in frame_visible_counts:
            raise ValueError(
                f"observation {observation_id} references missing frame visibility {frame_index}"
            )
        inside_counts = {
            int(key): int(value) for key, value in raw.get("inside_counts", {}).items()
            if int(value) > 0
        }
        raw_weight_sums = raw.get("inside_weight_sums")
        inside_weight_sums = (
            {int(key): float(value) for key, value in raw_weight_sums.items()}
            if raw_weight_sums is not None
            else {key: float(value) for key, value in inside_counts.items()}
        )
        if set(inside_weight_sums) != set(inside_counts):
            raise ValueError(
                f"observation {observation_id} weight support does not match point support"
            )
        for superpoint_id, weight_sum in inside_weight_sums.items():
            if not 0.0 < weight_sum <= float(inside_counts[superpoint_id]):
                raise ValueError(
                    f"observation {observation_id} has invalid depth weights in "
                    f"superpoint {superpoint_id}"
                )
        missing = set(inside_counts) - known_superpoints
        if missing:
            raise ValueError(
                f"observation {observation_id} references unknown superpoints: {sorted(missing)[:5]}"
            )
        visible_counts = frame_visible_counts[frame_index]
        for superpoint_id, inside_count in inside_counts.items():
            if inside_count > int(visible_counts.get(superpoint_id, 0)):
                raise ValueError(
                    f"observation {observation_id} contains non-visible points in "
                    f"superpoint {superpoint_id}"
                )
        observation_ids.append(observation_id)
        normalized.append({
            **raw,
            "observation_id": observation_id,
            "frame_index": frame_index,
            "inside_counts": inside_counts,
            "inside_weight_sums": inside_weight_sums,
        })
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("observation IDs are not unique")
    return normalized


def build_guide_mask_matching(
    proposals,
    observations,
    frame_visible_counts,
    superpoint_sizes,
    depth_weight_adapter=DEPTH_WEIGHT_ADAPTER,
):
    """Implement MV3DIS Eqs. (3)-(5) with binary depth visibility.

    ``inside_counts`` must count depth-visible 3D points whose projections lie
    inside each automatic SAM mask.  With no continuous depth weight available,
    Eq. (4)'s mean depth weight is one for every non-empty component, so each
    coverage entry equals mask visibility for the corresponding proposal.
    """
    proposals = _normalize_proposals(proposals, superpoint_sizes)
    observations = _normalize_observations(
        observations, frame_visible_counts, superpoint_sizes
    )
    proposal_ids = [int(row["proposal_id"]) for row in proposals]
    proposal_by_id = {int(row["proposal_id"]): row for row in proposals}

    coverage_by_observation = {}
    observation_by_id = {
        int(row["observation_id"]): row for row in observations
    }
    candidates_by_guide = {proposal_id: [] for proposal_id in proposal_ids}
    match_geometry = {}

    for observation in observations:
        observation_id = int(observation["observation_id"])
        frame_index = int(observation["frame_index"])
        visible_counts = frame_visible_counts[frame_index]
        inside_counts = observation["inside_counts"]
        inside_weight_sums = observation["inside_weight_sums"]
        vector = []
        for proposal in proposals:
            proposal_id = int(proposal["proposal_id"])
            superpoint_ids = proposal["superpoint_ids"]
            visible_count = _sum_for_superpoints(visible_counts, superpoint_ids)
            inside_count = _sum_for_superpoints(inside_counts, superpoint_ids)
            inside_weight_sum = float(
                sum(float(inside_weight_sums.get(int(item), 0.0)) for item in superpoint_ids)
            )
            total_count = int(proposal["point_count"])
            frame_visibility = float(visible_count / max(1, total_count))
            mask_visibility = float(inside_count / max(1, visible_count))
            mean_depth_weight = float(
                inside_weight_sum / max(1, inside_count)
            )
            coverage_value = float(mean_depth_weight * mask_visibility)
            vector.append(coverage_value if inside_count > 0 else 0.0)
            if (
                frame_visibility > FRAME_VISIBILITY_THRESHOLD
                and mask_visibility > MASK_VISIBILITY_THRESHOLD
            ):
                candidates_by_guide[proposal_id].append(observation_id)
                match_geometry[(proposal_id, observation_id)] = {
                    "guide_total_point_count": total_count,
                    "guide_visible_point_count": visible_count,
                    "guide_inside_mask_point_count": inside_count,
                    "guide_inside_mask_depth_weight_sum": inside_weight_sum,
                    "guide_mean_inside_mask_depth_weight": mean_depth_weight,
                    "guide_coverage_value": coverage_value,
                    "frame_visibility": frame_visibility,
                    "mask_visibility": mask_visibility,
                }
        coverage_by_observation[observation_id] = np.asarray(
            vector, dtype=np.float64
        )

    guide_scores = {}
    guide_rows = []
    for proposal_id in proposal_ids:
        candidates = sorted(candidates_by_guide[proposal_id])
        if not candidates:
            state = "no_candidate"
        elif len(candidates) == 1:
            state = "single_candidate_consistency_undefined"
            guide_scores[(proposal_id, candidates[0])] = None
        else:
            state = "multi_candidate_consistency_defined"
            for observation_id in candidates:
                peers = [item for item in candidates if item != observation_id]
                similarities = [
                    _cosine_similarity(
                        coverage_by_observation[observation_id],
                        coverage_by_observation[peer_id],
                    )
                    for peer_id in peers
                ]
                guide_scores[(proposal_id, observation_id)] = float(
                    np.mean(similarities)
                )
        proposal = proposal_by_id[proposal_id]
        guide_rows.append({
            "proposal_id": proposal_id,
            "candidate_mask_count": len(candidates),
            "candidate_observation_ids": candidates,
            "consistency_state": state,
            "superpoint_count": len(proposal["superpoint_ids"]),
            "point_count": int(proposal["point_count"]),
            "gt_usage": "none",
        })

    matched_guides_by_observation = defaultdict(list)
    for proposal_id, candidates in candidates_by_guide.items():
        for observation_id in candidates:
            matched_guides_by_observation[observation_id].append(proposal_id)

    mask_rows = []
    final_scores = {}
    for observation_id in sorted(matched_guides_by_observation):
        matched_guides = sorted(matched_guides_by_observation[observation_id])
        defined = [
            guide_scores[(proposal_id, observation_id)]
            for proposal_id in matched_guides
            if guide_scores[(proposal_id, observation_id)] is not None
        ]
        final_score = float(np.mean(defined)) if defined else None
        final_scores[observation_id] = final_score
        observation = observation_by_id[observation_id]
        vector = coverage_by_observation[observation_id]
        sparse_vector = [
            {"proposal_id": proposal_id, "value": float(vector[index])}
            for index, proposal_id in enumerate(proposal_ids)
            if float(vector[index]) > 0.0
        ]
        mask_rows.append({
            "observation_id": observation_id,
            "frame_id": str(observation.get("frame_id", observation["frame_index"])),
            "frame_index": int(observation["frame_index"]),
            "matched_guide_proposal_ids": matched_guides,
            "matched_guide_count": len(matched_guides),
            "defined_guide_score_count": len(defined),
            "undefined_guide_score_count": len(matched_guides) - len(defined),
            "final_consistency_score": final_score,
            "final_consistency_state": (
                "defined_mean_over_defined_guide_scores"
                if final_score is not None
                else "undefined_no_guide_with_multiple_candidates"
            ),
            "coverage_vector_dimension": len(proposal_ids),
            "coverage_vector_sparse": sparse_vector,
            "coverage_vector_proposal_order": proposal_ids,
            "depth_weight_adapter": depth_weight_adapter,
            "gt_usage": "none",
        })

    match_rows = []
    for proposal_id in proposal_ids:
        for observation_id in sorted(candidates_by_guide[proposal_id]):
            guide_score = guide_scores[(proposal_id, observation_id)]
            match_rows.append({
                "guide_proposal_id": proposal_id,
                "observation_id": observation_id,
                "frame_id": str(
                    observation_by_id[observation_id].get(
                        "frame_id", observation_by_id[observation_id]["frame_index"]
                    )
                ),
                "frame_index": int(observation_by_id[observation_id]["frame_index"]),
                **match_geometry[(proposal_id, observation_id)],
                "guide_consistency_score": guide_score,
                "guide_consistency_state": (
                    "defined_pairwise_cosine_mean"
                    if guide_score is not None
                    else "undefined_single_candidate"
                ),
                "final_mask_consistency_score": final_scores[observation_id],
                "frame_visibility_contract": "strict > 0.3",
                "mask_visibility_contract": "strict > 0.9",
                "depth_weight_adapter": depth_weight_adapter,
                "gt_usage": "none",
            })

    return {
        "proposal_ids": proposal_ids,
        "guide_rows": guide_rows,
        "mask_rows": mask_rows,
        "match_rows": match_rows,
    }


def _load_scene_observations(scene_name, scene_root, superpoints, visibility):
    observations = []
    seen = set()
    with (scene_root / "automatic_observations.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            observation_id = int(raw["observation_id"])
            frame_index = int(raw["frame_index"])
            if observation_id in seen:
                raise ValueError(f"duplicate observation ID {observation_id}")
            if str(raw["scene_name"]) != scene_name:
                raise ValueError(f"observation {observation_id} has a mismatched scene")
            if frame_index < 0 or frame_index >= len(visibility):
                raise ValueError(f"observation {observation_id} has an invalid frame index")
            point_path = Path(raw["point_indices_path"])
            with np.load(point_path) as payload:
                points = np.unique(
                    np.asarray(payload["point_indices"], dtype=np.int64)
                )
            if np.any(points < 0) or np.any(points >= len(superpoints)):
                raise ValueError(f"observation {observation_id} has invalid point indices")
            if np.any(~visibility[frame_index, points]):
                raise ValueError(
                    f"observation {observation_id} contains points outside frozen visibility"
                )
            ids, counts = np.unique(superpoints[points], return_counts=True)
            observations.append({
                "observation_id": observation_id,
                "frame_id": str(raw["frame_id"]),
                "frame_index": frame_index,
                "inside_counts": {
                    int(item): int(count) for item, count in zip(ids, counts)
                },
            })
            seen.add(observation_id)
    return observations


def _frame_visible_counts(observations, superpoints, visibility):
    result = {}
    for frame_index in sorted({int(row["frame_index"]) for row in observations}):
        ids, counts = np.unique(
            superpoints[np.flatnonzero(visibility[frame_index])], return_counts=True
        )
        result[frame_index] = {
            int(item): int(count) for item, count in zip(ids, counts)
        }
    return result


def _validate_and_build_fallback_rows(proposals, superpoints, guide_rows):
    guide_by_id = {int(row["proposal_id"]): row for row in guide_rows}
    fallback_rows = []
    for proposal in proposals:
        proposal_id = int(proposal["proposal_id"])
        expected = np.flatnonzero(
            np.isin(superpoints, np.asarray(proposal["superpoint_ids"], dtype=np.int64))
        )
        point_path = Path(proposal["points_path"])
        with np.load(point_path) as payload:
            actual = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"proposal {proposal_id} point file is not its frozen superpoint union"
            )
        fallback_rows.append({
            "proposal_id": proposal_id,
            "source_points_path": str(point_path),
            "point_indices_sha256": _point_index_digest(actual),
            "point_count": int(proposal["point_count"]),
            "superpoint_ids": list(map(int, proposal["superpoint_ids"])),
            "lineage_proposal_ids": list(
                map(int, proposal.get("lineage_proposal_ids", [proposal_id]))
            ),
            "candidate_mask_count": int(
                guide_by_id[proposal_id]["candidate_mask_count"]
            ),
            "mask_matching_state": guide_by_id[proposal_id]["consistency_state"],
            "geometry_action": "unchanged_frozen_d2b_fallback",
            "point_indices_changed": False,
            "superpoint_ids_changed": False,
            "lineage_changed": False,
            "gt_usage": "none",
        })
    return fallback_rows


def _build_and_publish_scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    superpoint_sizes = {
        int(item): int(count) for item, count in zip(ids, counts)
    }

    source = json.loads(
        (args.proposal_root / scene_name / "automatic_tracks.json").read_text()
    )
    proposals = _normalize_proposals(source.get("tracks", []), superpoint_sizes)
    world = WORLD_2_CAM(
        str(args.dataset_root / scene_name), args.depth_scale, args.config
    )
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    observations = _load_scene_observations(
        scene_name, args.automatic_root / scene_name, superpoints, visibility
    )
    visible_counts = _frame_visible_counts(observations, superpoints, visibility)
    ledger = build_guide_mask_matching(
        proposals, observations, visible_counts, superpoint_sizes
    )
    fallback_rows = _validate_and_build_fallback_rows(
        proposals, superpoints, ledger["guide_rows"]
    )

    state_counts = Counter(row["consistency_state"] for row in ledger["guide_rows"])
    summary = {
        "scene_name": scene_name,
        "source_proposal_count": len(proposals),
        "fallback_proposal_count": len(fallback_rows),
        "source_observation_count": len(observations),
        "matched_mask_count": len(ledger["mask_rows"]),
        "guide_mask_match_count": len(ledger["match_rows"]),
        "guide_state_counts": dict(state_counts),
        "defined_final_mask_score_count": sum(
            row["final_consistency_score"] is not None for row in ledger["mask_rows"]
        ),
        "undefined_final_mask_score_count": sum(
            row["final_consistency_score"] is None for row in ledger["mask_rows"]
        ),
        "frame_visibility_contract": "strict > 0.3",
        "mask_visibility_contract": "strict > 0.9",
        "coverage_contract": (
            "MV3DIS Eq. (4) with binary absolute-depth visibility; "
            "continuous relative-depth weights unavailable"
        ),
        "depth_weight_adapter": DEPTH_WEIGHT_ADAPTER,
        "proposal_mutation_count": 0,
        "ground_truth_usage": "none",
        "decision_state": (
            "M1a mask-matching ledger only; no NMS, affinity assignment, "
            "proposal mutation, semantics, or AP."
        ),
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "guide_mask_matches.jsonl", ledger["match_rows"])
        _write_jsonl(staging / "mask_coverage_vectors.jsonl", ledger["mask_rows"])
        _write_jsonl(staging / "guide_summaries.jsonl", ledger["guide_rows"])
        _write_jsonl(staging / "proposal_fallback_ledger.jsonl", fallback_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    del world, raw_visibility, visibility, observations, processed
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument(
        "--processed-scene-root", type=Path, default=Path("data/scannet200")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument(
        "--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "scene_list", "proposal_root", "automatic_root", "processed_scene_root",
        "dataset_root", "config_path", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("depth_weight_adapter") != DEPTH_WEIGHT_ADAPTER:
                raise SystemExit(f"resume adapter mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_and_publish_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"guides {summary['source_proposal_count']}, "
                f"matched masks {summary['matched_mask_count']}, "
                f"links {summary['guide_mask_match_count']}",
                flush=True,
            )
        summaries.append(summary)

    totals = Counter()
    guide_state_counts = Counter()
    for row in summaries:
        totals.update({
            "source_proposal_count": row["source_proposal_count"],
            "fallback_proposal_count": row["fallback_proposal_count"],
            "source_observation_count": row["source_observation_count"],
            "matched_mask_count": row["matched_mask_count"],
            "guide_mask_match_count": row["guide_mask_match_count"],
            "defined_final_mask_score_count": row["defined_final_mask_score_count"],
            "undefined_final_mask_score_count": row["undefined_final_mask_score_count"],
            "proposal_mutation_count": row["proposal_mutation_count"],
        })
        guide_state_counts.update(row["guide_state_counts"])
    payload = {
        "scene_count": len(summaries),
        **dict(totals),
        "guide_state_counts": dict(guide_state_counts),
        "frame_visibility_contract": "strict > 0.3",
        "mask_visibility_contract": "strict > 0.9",
        "depth_weight_adapter": DEPTH_WEIGHT_ADAPTER,
        "ground_truth_usage": "none",
        "decision_state": (
            "M1a mask-matching ledger only; no NMS, affinity assignment, "
            "proposal mutation, semantics, or AP."
        ),
        "params": {key: value for key, value in vars(args).items() if key != "config"},
        "scene_summaries": summaries,
    }
    (args.output_root / "mv3dis_guide_mask_matching_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "source_proposal_count": payload.get("source_proposal_count", 0),
        "fallback_proposal_count": payload.get("fallback_proposal_count", 0),
        "matched_mask_count": payload.get("matched_mask_count", 0),
        "guide_mask_match_count": payload.get("guide_mask_match_count", 0),
        "proposal_mutation_count": payload.get("proposal_mutation_count", 0),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

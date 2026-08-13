#!/usr/bin/env python3
"""Build a GT-free non-bridge quality ledger for fragment merge families.

Each frozen F1 action defines a family containing the original D2b anchor A,
the original absorbed proposal B, and the materialized merged proposal M.
Bridge frames that formed the action are excluded. A, B, and M are then
compared on the same remaining uniform30 relative-depth-visible frame domain.

This tool reuses the frozen D1 lifting and holdout sIoU contracts. It records
Pareto-dominance facts but never selects, merges, suppresses, rescales, or
mutates a candidate. It reads no GT, native predictions, classes, semantics,
scores, or AP results.
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_mv3dis_relative_depth_guide_mask_matching_ledger import (
    _load_relative_cache,
)
from tools.export_mv3dis_relative_depth_observations import PROJECTION_CONTRACT
from tools.refine_details_consensus_all_view_reobservation import (
    frame_match_metrics,
    proposal_dominates_base,
)


QUALITY_CONTRACT = {
    "min_superpoint_visible_ratio": 0.10,
    "min_initial_mask_support": 0.30,
    "min_visible_points_per_superpoint": 3,
    "support_siou": 0.30,
    "min_reliable_nonbridge_frames": 2,
}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_tracks(root, scene_name):
    payload = json.loads((root / scene_name / "automatic_tracks.json").read_text())
    tracks = payload.get("tracks", [])
    ids = [int(row["proposal_id"]) for row in tracks]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{scene_name} proposal IDs are invalid at {root}")
    return tracks


def lift_relative_observations(
    observations, frame_visible_counts, superpoint_sizes
):
    """Apply the frozen D1 lifting contract to relative-depth observations."""
    by_frame = defaultdict(list)
    frame_id_to_index = {}
    for raw in observations:
        frame_index = int(raw["frame_index"])
        frame_id = str(raw["frame_id"])
        previous = frame_id_to_index.setdefault(frame_id, frame_index)
        if previous != frame_index:
            raise ValueError(f"frame {frame_id} maps to multiple frame indices")
        visible_counts = frame_visible_counts[frame_index]
        lifted = []
        for superpoint_id, inside_count in raw["inside_counts"].items():
            superpoint_id = int(superpoint_id)
            visible_count = int(visible_counts.get(superpoint_id, 0))
            total_count = int(superpoint_sizes.get(superpoint_id, 0))
            if visible_count <= 0 or total_count <= 0:
                continue
            if (
                visible_count / total_count
                < QUALITY_CONTRACT["min_superpoint_visible_ratio"]
            ):
                continue
            if (
                int(inside_count) / visible_count
                >= QUALITY_CONTRACT["min_initial_mask_support"]
            ):
                lifted.append(superpoint_id)
        by_frame[frame_index].append({
            "observation_id": int(raw["observation_id"]),
            "frame_id": frame_id,
            "frame_index": frame_index,
            "lifted_superpoints": np.asarray(sorted(lifted), dtype=np.int64),
            "visible_counts": visible_counts,
        })
    for rows in by_frame.values():
        rows.sort(key=lambda row: row["observation_id"])
    return by_frame, frame_id_to_index


def _is_visible(superpoint_ids, visible_counts):
    minimum = QUALITY_CONTRACT["min_visible_points_per_superpoint"]
    return any(
        int(visible_counts.get(int(item), 0)) >= minimum
        for item in superpoint_ids
    )


def _best_frame_match(superpoint_ids, observations):
    best = None
    for observation in observations:
        metrics = frame_match_metrics(
            superpoint_ids,
            observation["lifted_superpoints"],
            observation["visible_counts"],
            QUALITY_CONTRACT["min_visible_points_per_superpoint"],
        )
        candidate = (
            float(metrics["siou"]),
            float(metrics["core_coverage"]),
            float(metrics["observation_purity"]),
            -int(observation["observation_id"]),
        )
        if best is None or candidate > best[0]:
            best = (candidate, observation, metrics)
    if best is None:
        return {
            "best_observation_id": None,
            "best_siou": 0.0,
            "best_core_coverage": 0.0,
            "best_observation_purity": 0.0,
        }
    _, observation, metrics = best
    return {
        "best_observation_id": int(observation["observation_id"]),
        "best_siou": float(metrics["siou"]),
        "best_core_coverage": float(metrics["core_coverage"]),
        "best_observation_purity": float(metrics["observation_purity"]),
    }


def compare_family_on_nonbridge_frames(
    candidate_superpoints,
    observations_by_frame,
    frame_visible_counts,
    frame_id_to_index,
    bridge_frame_ids,
):
    """Compare A/B/M on one common non-bridge visibility domain."""
    labels = ("anchor", "absorbed", "merged")
    if set(candidate_superpoints) != set(labels):
        raise ValueError("fragment family must contain anchor, absorbed, and merged")
    normalized = {
        label: tuple(sorted(set(map(int, candidate_superpoints[label]))))
        for label in labels
    }
    if any(not normalized[label] for label in labels):
        raise ValueError("fragment family contains an empty candidate")
    bridge_indices = set()
    for frame_id in bridge_frame_ids:
        if str(frame_id) not in frame_id_to_index:
            raise ValueError(f"bridge frame {frame_id} is absent from relative cache")
        bridge_indices.add(int(frame_id_to_index[str(frame_id)]))

    frame_rows = []
    for frame_index in sorted(frame_visible_counts):
        if frame_index in bridge_indices:
            continue
        visible_counts = frame_visible_counts[frame_index]
        if not all(
            _is_visible(normalized[label], visible_counts) for label in labels
        ):
            continue
        observations = observations_by_frame.get(frame_index, [])
        row = {
            "frame_index": int(frame_index),
            "frame_id": (
                str(observations[0]["frame_id"])
                if observations else str(frame_index)
            ),
            "observation_count": len(observations),
        }
        for label in labels:
            row[label] = _best_frame_match(normalized[label], observations)
        frame_rows.append(row)

    metrics = {}
    for label in labels:
        scores = np.asarray(
            [row[label]["best_siou"] for row in frame_rows], dtype=np.float64
        )
        metrics[label] = {
            "mean_best_siou": float(scores.mean()) if len(scores) else 0.0,
            "support_frame_rate": (
                float(np.mean(scores >= QUALITY_CONTRACT["support_siou"]))
                if len(scores) else 0.0
            ),
        }
    reliable = (
        len(frame_rows) >= QUALITY_CONTRACT["min_reliable_nonbridge_frames"]
    )
    merged_dominates_anchor = bool(
        reliable and proposal_dominates_base(metrics["anchor"], metrics["merged"])
    )
    merged_dominates_absorbed = bool(
        reliable and proposal_dominates_base(metrics["absorbed"], metrics["merged"])
    )
    return {
        "bridge_frame_ids": sorted(map(str, bridge_frame_ids)),
        "excluded_bridge_frame_count": len(bridge_indices),
        "common_nonbridge_visible_frame_count": len(frame_rows),
        "zero_observation_frame_count": sum(
            row["observation_count"] == 0 for row in frame_rows
        ),
        "quality_evidence_reliable": reliable,
        "candidate_metrics": metrics,
        "merged_dominates_anchor": merged_dominates_anchor,
        "merged_dominates_absorbed": merged_dominates_absorbed,
        "merged_jointly_dominates": bool(
            merged_dominates_anchor and merged_dominates_absorbed
        ),
        "frames": frame_rows,
    }


def _pair_evidence_for_actions(rows, actions):
    needed = {
        tuple(sorted((
            int(action["anchor_proposal_id"]),
            int(action["absorbed_proposal_id"]),
        )))
        for action in actions
    }
    result = {}
    for row in rows:
        pair = (
            int(row["left_proposal_id"]), int(row["right_proposal_id"])
        )
        if pair in needed:
            result[pair] = row
    if set(result) != needed:
        raise ValueError("fragment relation ledger lacks a planned family")
    return result


def _validate_family_tracks(action, source_by_id, merged_by_id):
    anchor_id = int(action["anchor_proposal_id"])
    absorbed_id = int(action["absorbed_proposal_id"])
    if anchor_id not in source_by_id or absorbed_id not in source_by_id:
        raise ValueError("fragment action references missing source proposal")
    if anchor_id not in merged_by_id or absorbed_id in merged_by_id:
        raise ValueError("materialized fragment proposal identity is inconsistent")
    anchor = source_by_id[anchor_id]
    absorbed = source_by_id[absorbed_id]
    merged = merged_by_id[anchor_id]
    expected_lineage = sorted(set(
        map(int, anchor["lineage_proposal_ids"])
    ) | set(map(int, absorbed["lineage_proposal_ids"])))
    expected_observations = sorted(set(
        map(int, anchor["observation_ids"])
    ) | set(map(int, absorbed["observation_ids"])))
    if list(map(int, merged["lineage_proposal_ids"])) != expected_lineage:
        raise ValueError("merged family lineage differs from A+B")
    if list(map(int, merged["observation_ids"])) != expected_observations:
        raise ValueError("merged family observations differ from A+B")
    return anchor, absorbed, merged


def _build_scene(scene_name, args):
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
    observations, frame_visible_counts, relative_summary = _load_relative_cache(
        scene_name, args.relative_observation_root / scene_name, superpoints
    )
    observations_by_frame, frame_id_to_index = lift_relative_observations(
        observations, frame_visible_counts, superpoint_sizes
    )
    source_tracks = _load_tracks(args.source_track_root, scene_name)
    merged_tracks = _load_tracks(args.merged_track_root, scene_name)
    source_by_id = {int(row["proposal_id"]): row for row in source_tracks}
    merged_by_id = {int(row["proposal_id"]): row for row in merged_tracks}
    actions = _read_jsonl(
        args.fragment_plan_root / scene_name / "fragment_merge_actions.jsonl"
    )
    pair_evidence = _pair_evidence_for_actions(
        _read_jsonl(
            args.fragment_ledger_root / scene_name / "fragment_pair_evidence.jsonl"
        ),
        actions,
    )

    family_rows = []
    frame_rows = []
    for action in actions:
        anchor, absorbed, merged = _validate_family_tracks(
            action, source_by_id, merged_by_id
        )
        anchor_id = int(action["anchor_proposal_id"])
        absorbed_id = int(action["absorbed_proposal_id"])
        evidence = pair_evidence[(anchor_id, absorbed_id)]
        consistency = compare_family_on_nonbridge_frames(
            {
                "anchor": anchor["superpoint_ids"],
                "absorbed": absorbed["superpoint_ids"],
                "merged": merged["superpoint_ids"],
            },
            observations_by_frame,
            frame_visible_counts,
            frame_id_to_index,
            evidence["bridge_frame_ids"],
        )
        for frame in consistency.pop("frames"):
            frame_rows.append({
                "scene_name": scene_name,
                "action_index": int(action["action_index"]),
                "anchor_proposal_id": anchor_id,
                "absorbed_proposal_id": absorbed_id,
                **frame,
                "ground_truth_usage": "none",
            })
        family_rows.append({
            "scene_name": scene_name,
            "action_index": int(action["action_index"]),
            "anchor_proposal_id": anchor_id,
            "absorbed_proposal_id": absorbed_id,
            "anchor_superpoint_count": len(anchor["superpoint_ids"]),
            "absorbed_superpoint_count": len(absorbed["superpoint_ids"]),
            "merged_superpoint_count": len(merged["superpoint_ids"]),
            **consistency,
            "candidate_action": "none_ledger_only",
            "score_used_for_decision": False,
            "candidate_mutation_count": 0,
            "ground_truth_usage": "none",
        })

    summary = {
        "scene_name": scene_name,
        "source_proposal_count": len(source_tracks),
        "merged_proposal_count": len(merged_tracks),
        "family_count": len(family_rows),
        "family_frame_evidence_count": len(frame_rows),
        "reliable_family_count": sum(
            row["quality_evidence_reliable"] for row in family_rows
        ),
        "merged_dominates_anchor_count": sum(
            row["merged_dominates_anchor"] for row in family_rows
        ),
        "merged_dominates_absorbed_count": sum(
            row["merged_dominates_absorbed"] for row in family_rows
        ),
        "merged_jointly_dominates_count": sum(
            row["merged_jointly_dominates"] for row in family_rows
        ),
        "relative_depth_contract": relative_summary["projection_contract"],
        "quality_contract": QUALITY_CONTRACT,
        "candidate_action_count": 0,
        "candidate_mutation_count": 0,
        "score_used_for_decision_count": 0,
        "ground_truth_usage": "none",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "fragment_family_quality.jsonl", family_rows)
        _write_jsonl(staging / "fragment_family_frame_evidence.jsonl", frame_rows)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    del processed
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--source-track-root", type=Path, required=True)
    parser.add_argument("--merged-track-root", type=Path, required=True)
    parser.add_argument("--fragment-ledger-root", type=Path, required=True)
    parser.add_argument("--fragment-plan-root", type=Path, required=True)
    parser.add_argument("--relative-observation-root", type=Path, required=True)
    parser.add_argument(
        "--processed-scene-root", type=Path, default=Path("data/scannet200")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "source_track_root", "merged_track_root",
        "fragment_ledger_root", "fragment_plan_root",
        "relative_observation_root", "processed_scene_root", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("quality_contract") != QUALITY_CONTRACT:
                raise SystemExit(f"resume quality contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: families "
                f"{summary['family_count']}, reliable "
                f"{summary['reliable_family_count']}, jointly dominant "
                f"{summary['merged_jointly_dominates_count']}",
                flush=True,
            )
        summaries.append(summary)

    additive_keys = (
        "source_proposal_count", "merged_proposal_count", "family_count",
        "family_frame_evidence_count", "reliable_family_count",
        "merged_dominates_anchor_count", "merged_dominates_absorbed_count",
        "merged_jointly_dominates_count", "candidate_action_count",
        "candidate_mutation_count", "score_used_for_decision_count",
    )
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in additive_keys
        },
        "relative_depth_contract": PROJECTION_CONTRACT,
        "quality_contract": QUALITY_CONTRACT,
        "decision_contract": (
            "ledger only; compare A/B/M on common non-bridge frames; "
            "no candidate action"
        ),
        "ground_truth_usage": "none",
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scene_summaries": summaries,
    }
    path = args.output_root / "multiview_fragment_family_quality_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "scene_count", "family_count", "family_frame_evidence_count",
        "reliable_family_count", "merged_dominates_anchor_count",
        "merged_dominates_absorbed_count", "merged_jointly_dominates_count",
        "candidate_action_count", "candidate_mutation_count",
        "score_used_for_decision_count",
    )}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

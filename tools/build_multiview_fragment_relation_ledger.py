#!/usr/bin/env python3
"""Build a no-GT multiview fragment relation ledger over frozen D2b proposals.

The ledger combines the frozen D2b pair geometry with the already materialized
MV3DIS relative-depth guide/mask matches.  A bridge observation is one automatic
SAM mask that satisfies the published guide-matching contract for both
proposals.  A separation counterexample is a same-frame pair of distinct masks
that separately match the two proposals and whose hierarchy relation is
``disjoint`` or ``partial``.

Every unordered proposal pair is retained.  This tool never merges, suppresses,
rescores, relabels, or mutates a proposal, and it does not read GT, native
predictions, semantics, or AP results.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_details_consensus_proposal_relation_graph import (
    build_scene_graph,
    build_superpoint_contact_map,
)
from tools.export_mv3dis_relative_depth_observations import PROJECTION_CONTRACT


SEPARATION_RELATION_KINDS = frozenset({"disjoint", "partial"})
CONTACT_PARAMETERS = {
    "adjacency_knn": 12,
    "adjacency_max_distance": 0.05,
    "min_contact_points": 3,
    "min_contact_ratio": 0.02,
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


def _proposal_frame_observations(match_rows, proposal_ids):
    proposal_ids = set(map(int, proposal_ids))
    seen_links = set()
    observation_frames = {}
    observation_proposals = defaultdict(set)
    proposal_frame_observations = defaultdict(lambda: defaultdict(set))
    for row in match_rows:
        proposal_id = int(row["guide_proposal_id"])
        observation_id = int(row["observation_id"])
        frame_id = str(row["frame_id"])
        link = (proposal_id, observation_id)
        if proposal_id not in proposal_ids:
            raise ValueError(f"matching ledger references unknown proposal {proposal_id}")
        if link in seen_links:
            raise ValueError(f"duplicate guide/mask link {link}")
        if row.get("depth_weight_adapter") != PROJECTION_CONTRACT:
            raise ValueError("matching ledger does not use the relative-depth contract")
        previous_frame = observation_frames.setdefault(observation_id, frame_id)
        if previous_frame != frame_id:
            raise ValueError(f"observation {observation_id} has inconsistent frames")
        seen_links.add(link)
        observation_proposals[observation_id].add(proposal_id)
        proposal_frame_observations[proposal_id][frame_id].add(observation_id)
    return observation_frames, observation_proposals, proposal_frame_observations


def _hierarchy_relation_map(rows, valid_observation_ids):
    valid_observation_ids = set(map(int, valid_observation_ids))
    result = {}
    for row in rows:
        left = int(row["left_observation_id"])
        right = int(row["right_observation_id"])
        if left not in valid_observation_ids or right not in valid_observation_ids:
            continue
        key = tuple(sorted((left, right)))
        if key in result:
            raise ValueError(f"duplicate same-frame hierarchy relation {key}")
        result[key] = {
            "frame_id": str(row["frame_id"]),
            "relation_kind": str(row["relation_kind"]),
        }
    return result


def build_fragment_pair_evidence(nodes, geometry_relations, match_rows, hierarchy_rows):
    """Attach multiview bridge and separation facts to every proposal pair."""
    proposal_ids = [int(row["proposal_id"]) for row in nodes]
    observation_frames, observation_proposals, by_proposal_frame = (
        _proposal_frame_observations(match_rows, proposal_ids)
    )
    hierarchy = _hierarchy_relation_map(
        hierarchy_rows, observation_proposals.keys()
    )
    bridge_observations = defaultdict(list)
    for observation_id, matched_proposals in observation_proposals.items():
        for pair in combinations(sorted(matched_proposals), 2):
            bridge_observations[pair].append(int(observation_id))

    evidence_rows = []
    for geometry in geometry_relations:
        left = int(geometry["left_proposal_id"])
        right = int(geometry["right_proposal_id"])
        pair = (left, right)
        bridge_ids = sorted(bridge_observations.get(pair, []))
        bridge_frames = sorted({observation_frames[item] for item in bridge_ids})

        common_frames = sorted(
            set(by_proposal_frame[left]) & set(by_proposal_frame[right])
        )
        separation_frames = set()
        separation_pairs = []
        same_frame_relation_counts = Counter()
        for frame_id in common_frames:
            left_only = sorted(
                observation_id
                for observation_id in by_proposal_frame[left][frame_id]
                if right not in observation_proposals[observation_id]
            )
            right_only = sorted(
                observation_id
                for observation_id in by_proposal_frame[right][frame_id]
                if left not in observation_proposals[observation_id]
            )
            for left_observation_id in left_only:
                for right_observation_id in right_only:
                    relation = hierarchy.get(
                        tuple(sorted((left_observation_id, right_observation_id)))
                    )
                    if relation is None:
                        continue
                    if relation["frame_id"] != frame_id:
                        raise ValueError("hierarchy relation frame does not match guide links")
                    kind = relation["relation_kind"]
                    same_frame_relation_counts[kind] += 1
                    if kind in SEPARATION_RELATION_KINDS:
                        separation_frames.add(frame_id)
                        separation_pairs.append({
                            "frame_id": frame_id,
                            "left_observation_id": left_observation_id,
                            "right_observation_id": right_observation_id,
                            "relation_kind": kind,
                        })

        if bridge_frames and separation_frames:
            state = "bridge_with_separation_counterevidence"
        elif bridge_frames:
            state = "bridge_without_separation_counterevidence"
        elif separation_frames:
            state = "separation_counterevidence_only"
        else:
            state = "no_multiview_pair_evidence"
        evidence_rows.append({
            **geometry,
            "bridge_observation_ids": bridge_ids,
            "bridge_observation_count": len(bridge_ids),
            "bridge_frame_ids": bridge_frames,
            "bridge_frame_count": len(bridge_frames),
            "common_matched_frame_ids": common_frames,
            "common_matched_frame_count": len(common_frames),
            "separation_frame_ids": sorted(separation_frames),
            "separation_frame_count": len(separation_frames),
            "separation_observation_pairs": separation_pairs,
            "separation_observation_pair_count": len(separation_pairs),
            "same_frame_exclusive_relation_counts": dict(
                sorted(same_frame_relation_counts.items())
            ),
            "multiview_evidence_state": state,
            "fragment_action": "none_ledger_only",
            "proposal_mutation_count": 0,
            "gt_usage": "none",
            "decision_state": (
                "Observed pair evidence only; no merge, suppression, score, or "
                "proposal mutation applied."
            ),
        })
    validate_fragment_ledger(nodes, evidence_rows)
    return evidence_rows


def validate_fragment_ledger(nodes, evidence_rows):
    proposal_ids = [int(row["proposal_id"]) for row in nodes]
    expected_pairs = list(combinations(proposal_ids, 2))
    actual_pairs = [
        (int(row["left_proposal_id"]), int(row["right_proposal_id"]))
        for row in evidence_rows
    ]
    if actual_pairs != expected_pairs:
        raise ValueError("fragment pair ledger does not conserve deterministic pairs")
    valid_states = {
        "bridge_with_separation_counterevidence",
        "bridge_without_separation_counterevidence",
        "separation_counterevidence_only",
        "no_multiview_pair_evidence",
    }
    for row in evidence_rows:
        if row["multiview_evidence_state"] not in valid_states:
            raise ValueError("invalid multiview evidence state")
        if row["bridge_frame_count"] != len(row["bridge_frame_ids"]):
            raise ValueError("bridge frame count is inconsistent")
        if row["separation_frame_count"] != len(row["separation_frame_ids"]):
            raise ValueError("separation frame count is inconsistent")
        if row["proposal_mutation_count"] != 0 or row["fragment_action"] != "none_ledger_only":
            raise ValueError("ledger attempted to apply a fragment action")


def _build_scene(scene_name, args):
    proposal_scene_root = args.proposal_root / scene_name
    source = json.loads((proposal_scene_root / "automatic_tracks.json").read_text())
    tracks = source.get("tracks", [])
    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    contact_map = build_superpoint_contact_map(processed, **CONTACT_PARAMETERS)
    nodes, geometry_relations = build_scene_graph(
        scene_name, tracks, processed, proposal_scene_root, contact_map
    )
    match_rows = _read_jsonl(
        args.matching_root / scene_name / "guide_mask_matches.jsonl"
    )
    hierarchy_rows = _read_jsonl(
        args.automatic_root / scene_name / "same_frame_hierarchy_relations.jsonl"
    )
    evidence_rows = build_fragment_pair_evidence(
        nodes, geometry_relations, match_rows, hierarchy_rows
    )
    states = Counter(row["multiview_evidence_state"] for row in evidence_rows)
    summary = {
        "scene_name": scene_name,
        "proposal_count": len(nodes),
        "pair_count": len(evidence_rows),
        "expected_pair_count": len(nodes) * (len(nodes) - 1) // 2,
        "guide_mask_match_count": len(match_rows),
        "bridge_pair_count": sum(row["bridge_frame_count"] > 0 for row in evidence_rows),
        "multiframe_bridge_pair_count": sum(
            row["bridge_frame_count"] >= 2 for row in evidence_rows
        ),
        "separation_pair_count": sum(
            row["separation_frame_count"] > 0 for row in evidence_rows
        ),
        "contact_pair_count": sum(row["has_spatial_contact"] for row in evidence_rows),
        "evidence_state_counts": dict(sorted(states.items())),
        "contact_parameters": CONTACT_PARAMETERS,
        "relative_depth_contract": PROJECTION_CONTRACT,
        "proposal_mutation_count": 0,
        "fragment_action_count": 0,
        "ground_truth_usage": "none",
        "decision_state": "Pair evidence ledger only; all D2b proposals are frozen.",
    }

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "proposal_nodes.jsonl", nodes)
        _write_jsonl(staging / "fragment_pair_evidence.jsonl", evidence_rows)
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
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
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
        "scene_list", "proposal_root", "matching_root", "automatic_root",
        "processed_scene_root", "output_root",
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
            if summary.get("relative_depth_contract") != PROJECTION_CONTRACT:
                raise SystemExit(f"resume contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"pairs {summary['pair_count']}, bridge "
                f"{summary['bridge_pair_count']}, separation "
                f"{summary['separation_pair_count']}",
                flush=True,
            )
        summaries.append(summary)

    additive_keys = (
        "proposal_count", "pair_count", "expected_pair_count",
        "guide_mask_match_count", "bridge_pair_count",
        "multiframe_bridge_pair_count", "separation_pair_count",
        "contact_pair_count", "proposal_mutation_count", "fragment_action_count",
    )
    states = Counter()
    for row in summaries:
        states.update(row["evidence_state_counts"])
    payload = {
        "scene_count": len(summaries),
        **{
            key: sum(int(row[key]) for row in summaries)
            for key in additive_keys
        },
        "evidence_state_counts": dict(sorted(states.items())),
        "contact_parameters": CONTACT_PARAMETERS,
        "relative_depth_contract": PROJECTION_CONTRACT,
        "proposal_mutation_count": 0,
        "fragment_action_count": 0,
        "ground_truth_usage": "none",
        "decision_state": "Pair evidence ledger only; all D2b proposals are frozen.",
        "params": vars(args),
        "scene_summaries": summaries,
    }
    (args.output_root / "multiview_fragment_relation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "proposal_count": payload["proposal_count"],
        "pair_count": payload["pair_count"],
        "bridge_pair_count": payload["bridge_pair_count"],
        "multiframe_bridge_pair_count": payload["multiframe_bridge_pair_count"],
        "separation_pair_count": payload["separation_pair_count"],
        "proposal_mutation_count": payload["proposal_mutation_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

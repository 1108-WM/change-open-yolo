#!/usr/bin/env python3
"""Build a frozen no-GT feature ledger for F2 boundary owner actions.

The input A/B/unknown plan remains a *plan*: this tool does not change a
proposal, score, class, or mask.  It records one inference-available feature
row for every candidate-owner counterfactual of every frozen planned boundary
action, plus a boundary-level row describing the unknown=no-op fallback.

GT, AP, native predictions, semantic labels, and oracle results are purposely
not accepted as inputs.  A later explicitly GT-only diagnostic may join these
immutable rows to action-outcome labels, but that join must never feed a
decision back into this builder.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_automatic_sam_track_growth_ledger import _raw_superpoint_context


ASSIGN_ACTION = "assign_unique_pareto_owner"
UNKNOWN_NOOP_ACTION = "unknown_noop_keep_current_ownership"
ADJACENCY_KNN = 12
ADJACENCY_MAX_DISTANCE = 0.05
MIN_CONTACT_POINTS = 3
MIN_CONTACT_RATIO = 0.02
DECISION_CONSTRAINT = (
    "No GT, native prediction, class, semantic score, AP, threshold, or selector "
    "is read or produced. Rows are descriptive only and cannot materialize an action."
)


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


def _load_tracks(path, superpoint_sizes):
    payload = json.loads(path.read_text())
    tracks = payload.get("tracks", [])
    if int(payload.get("track_count", len(tracks))) != len(tracks):
        raise ValueError("track count is inconsistent")
    result = {}
    for track in tracks:
        proposal_id = int(track["proposal_id"])
        ids = tuple(map(int, track.get("superpoint_ids", [])))
        if proposal_id in result or int(track.get("track_id", proposal_id)) != proposal_id:
            raise ValueError("track proposal identity is invalid")
        if not ids or list(ids) != sorted(set(ids)):
            raise ValueError(f"proposal {proposal_id} has invalid superpoint IDs")
        if set(ids) - set(superpoint_sizes):
            raise ValueError(f"proposal {proposal_id} references unknown superpoints")
        point_count = sum(superpoint_sizes[item] for item in ids)
        if int(track.get("point_count", -1)) != point_count:
            raise ValueError(f"proposal {proposal_id} point count differs")
        result[proposal_id] = {
            "proposal_id": proposal_id,
            "superpoint_ids": frozenset(ids),
            "point_count": point_count,
            "support_view_count": int(track.get("support_view_count", 0)),
            "observation_count": len(set(map(int, track.get("observation_ids", [])))),
            "node_count": len(set(map(int, track.get("node_ids", [])))),
            "mean_node_quality": float(track.get("mean_node_quality", 0.0)),
            "mean_consensus_rate": float(track.get("mean_consensus_rate", 0.0)),
            "mean_supported_coverage": float(track.get("mean_supported_coverage", 0.0)),
        }
    return result


def _weighted_contact(edges, field):
    total = sum(int(edge["boundary_contact_count"]) for edge in edges)
    if not total:
        return None
    return float(sum(
        float(edge[field]) * int(edge["boundary_contact_count"])
        for edge in edges
    ) / total)


def _candidate_affinity_features(candidate, all_candidates):
    pairs = candidate.get("pair_evidence", [])
    defined = [float(row["pair_affinity"]) for row in pairs if row.get("pair_affinity") is not None]
    observed_mean = candidate.get("observed_pair_affinity_mean")
    if observed_mean is not None:
        observed_mean = float(observed_mean)
    comparable = [
        float(row["observed_pair_affinity_mean"])
        for row in all_candidates
        if row.get("observed_pair_affinity_mean") is not None
    ]
    other = [value for value in comparable if value != observed_mean]
    return {
        "affinity_neighbor_pair_count": int(candidate.get("neighbor_pair_count", 0)),
        "affinity_defined_pair_count": int(candidate.get("defined_pair_affinity_count", 0)),
        "affinity_defined_pair_ratio": float(len(defined) / max(1, len(pairs))),
        "affinity_observed_mean": observed_mean,
        "affinity_observed_min": float(min(defined)) if defined else None,
        "affinity_observed_max": float(max(defined)) if defined else None,
        "affinity_mean_minus_best_other": (
            float(observed_mean - max(other)) if observed_mean is not None and other else None
        ),
        "affinity_mean_minus_mean_other": (
            float(observed_mean - np.mean(other)) if observed_mean is not None and other else None
        ),
    }


def _candidate_contact_features(superpoint_id, candidate_core, neighbors):
    core = set(map(int, candidate_core))
    edges = [
        edge for edge in neighbors.get(int(superpoint_id), [])
        if int(edge["neighbor_superpoint_id"]) in core
    ]
    count = sum(int(edge["boundary_contact_count"]) for edge in edges)
    return {
        "direct_core_neighbor_count": int(len(edges)),
        "direct_core_contact_count_sum": int(count),
        "direct_core_contact_ratio_sum": float(sum(
            float(edge["boundary_contact_ratio"]) for edge in edges
        )),
        "direct_core_contact_ratio_max": float(max(
            (float(edge["boundary_contact_ratio"]) for edge in edges), default=0.0
        )),
        "contact_weighted_boundary_distance": _weighted_contact(edges, "mean_boundary_distance"),
        "contact_weighted_normal_difference": _weighted_contact(edges, "mean_normal_difference"),
        "contact_weighted_color_difference": _weighted_contact(edges, "mean_color_difference"),
    }


def build_feature_rows(scene_name, plan_rows, preassignment_rows, tracks, context):
    """Return immutable boundary and candidate-action feature rows for one scene."""
    pre_by_superpoint = {int(row["superpoint_id"]): row for row in preassignment_rows}
    if len(pre_by_superpoint) != len(preassignment_rows):
        raise ValueError(f"{scene_name} has duplicate preassignment superpoints")
    sizes = {int(raw_id): int(size) for raw_id, size in zip(context["raw_ids"], context["sizes"])}
    boundaries, actions = [], []
    for plan in sorted(plan_rows, key=lambda row: int(row["superpoint_id"])):
        if plan.get("planned_action") != ASSIGN_ACTION:
            continue
        if plan.get("ground_truth_usage") != "none":
            raise ValueError(f"{scene_name} plan uses GT")
        if bool(plan.get("assignment_applied")) or bool(plan.get("proposal_mutation_applied")):
            raise ValueError(f"{scene_name} plan was materialized")
        superpoint_id = int(plan["superpoint_id"])
        source = pre_by_superpoint.get(superpoint_id)
        if source is None or source.get("gt_usage") != "none":
            raise ValueError(f"{scene_name}/{superpoint_id} has no valid no-GT source row")
        candidates = tuple(sorted(map(int, plan["adjacent_candidate_proposal_ids"])))
        current = tuple(sorted(map(int, plan["current_owner_proposal_ids"])))
        source_candidates = tuple(sorted(map(int, source["adjacent_candidate_proposal_ids"])))
        if candidates != source_candidates:
            raise ValueError(f"{scene_name}/{superpoint_id} candidate identities differ")
        if current != tuple(sorted(map(int, source["current_owner_proposal_ids"]))):
            raise ValueError(f"{scene_name}/{superpoint_id} current owners differ")
        if len(candidates) < 2 or set(candidates) - set(tracks):
            raise ValueError(f"{scene_name}/{superpoint_id} has invalid candidates")
        evidence = source.get("candidate_region_evidence", [])
        evidence_by_id = {int(row["proposal_id"]): row for row in evidence}
        if tuple(sorted(evidence_by_id)) != candidates:
            raise ValueError(f"{scene_name}/{superpoint_id} affinity evidence differs")
        summaries = {int(row["proposal_id"]): row for row in plan.get("candidate_affinity_summaries", [])}
        if tuple(sorted(summaries)) != candidates:
            raise ValueError(f"{scene_name}/{superpoint_id} plan affinity summaries differ")
        planned_owner = int(plan["planned_owner_proposal_id"])
        if planned_owner not in candidates:
            raise ValueError(f"{scene_name}/{superpoint_id} planned owner is not a candidate")
        boundary = {
            "scene_name": scene_name,
            "superpoint_id": superpoint_id,
            "boundary_point_count": int(sizes[superpoint_id]),
            "competition_state": str(plan["competition_state"]),
            "candidate_owner_proposal_ids": list(candidates),
            "current_owner_proposal_ids": list(current),
            "candidate_owner_count": len(candidates),
            "current_owner_count": len(current),
            "frozen_planned_owner_proposal_id": planned_owner,
            "frozen_plan_evidence_state": str(plan["evidence_state"]),
            "unknown_action_name": UNKNOWN_NOOP_ACTION,
            "unknown_definition": "no-op; retain exact current owner memberships",
            "proposal_materialization_applied": False,
            "score_used_for_decision": False,
            "ground_truth_usage": "none",
            "decision_constraint": DECISION_CONSTRAINT,
        }
        boundaries.append(boundary)
        candidate_evidence = list(evidence_by_id.values())
        for proposal_id in candidates:
            track = tracks[proposal_id]
            core = set(track["superpoint_ids"]) - {superpoint_id}
            core_point_count = sum(sizes[item] for item in core)
            action = {
                "scene_name": scene_name,
                "superpoint_id": superpoint_id,
                "action_name": f"assign_to_proposal_{proposal_id}",
                "action_kind": "assign_owner",
                "candidate_owner_proposal_id": proposal_id,
                "is_frozen_planned_owner": proposal_id == planned_owner,
                "candidate_is_current_owner": proposal_id in current,
                "would_add_boundary_to_candidate": superpoint_id not in track["superpoint_ids"],
                "would_remove_boundary_from_other_current_owner_count": sum(
                    owner != proposal_id for owner in current
                ),
                "boundary_point_count": int(sizes[superpoint_id]),
                "candidate_owner_count": len(candidates),
                "current_owner_count": len(current),
                "competition_state": str(plan["competition_state"]),
                "candidate_proposal_superpoint_count": len(track["superpoint_ids"]),
                "candidate_proposal_point_count": int(track["point_count"]),
                "candidate_core_superpoint_count": len(core),
                "candidate_core_point_count": int(core_point_count),
                "candidate_boundary_point_fraction_of_core": float(
                    sizes[superpoint_id] / max(1, core_point_count)
                ),
                "candidate_support_view_count": int(track["support_view_count"]),
                "candidate_observation_count": int(track["observation_count"]),
                "candidate_node_count": int(track["node_count"]),
                "candidate_mean_node_quality": float(track["mean_node_quality"]),
                "candidate_mean_consensus_rate": float(track["mean_consensus_rate"]),
                "candidate_mean_supported_coverage": float(track["mean_supported_coverage"]),
                "plan_complete_edge_evidence": bool(summaries[proposal_id]["complete_edge_evidence"]),
                "proposal_materialization_applied": False,
                "score_used_for_decision": False,
                "ground_truth_usage": "none",
                "decision_constraint": DECISION_CONSTRAINT,
            }
            action.update(_candidate_affinity_features(evidence_by_id[proposal_id], candidate_evidence))
            action.update(_candidate_contact_features(superpoint_id, core, context["neighbors"]))
            actions.append(action)
    return boundaries, actions


def _build_scene(scene_name, args):
    processed = np.load(
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy",
        mmap_mode="r",
    )
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    ids, counts = np.unique(np.asarray(processed[:, 9], dtype=np.int64), return_counts=True)
    tracks = _load_tracks(
        args.f2_track_root / scene_name / "automatic_tracks.json",
        {int(item): int(count) for item, count in zip(ids, counts)},
    )
    context = _raw_superpoint_context(
        processed, ADJACENCY_KNN, ADJACENCY_MAX_DISTANCE,
        MIN_CONTACT_POINTS, MIN_CONTACT_RATIO,
    )
    boundaries, actions = build_feature_rows(
        scene_name,
        _read_jsonl(args.assignment_plan_root / scene_name / "global_boundary_assignment_plan.jsonl"),
        _read_jsonl(args.preassignment_root / scene_name / "boundary_competition.jsonl"),
        tracks,
        context,
    )
    summary = {
        "scene_name": scene_name,
        "planned_boundary_action_count": len(boundaries),
        "candidate_assign_action_feature_count": len(actions),
        "unknown_noop_feature_count": len(boundaries),
        "proposal_materialization_applied_count": 0,
        "ground_truth_usage": "none",
        "decision_constraint": DECISION_CONSTRAINT,
    }
    published = args.output_dir / scene_name
    staging = args.output_dir / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        _write_jsonl(staging / "boundary_no_gt_features.jsonl", boundaries)
        _write_jsonl(staging / "candidate_action_no_gt_features.jsonl", actions)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, published)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--f2-track-root", type=Path, required=True)
    parser.add_argument("--assignment-plan-root", type=Path, required=True)
    parser.add_argument("--preassignment-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for name in (
        "scene_list", "f2_track_root", "assignment_plan_root", "preassignment_root",
        "processed_scene_root", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_dir / scene_name / "summary.json"
        if existing.is_file() and args.resume:
            summary = json.loads(existing.read_text())
            if summary.get("decision_constraint") != DECISION_CONSTRAINT:
                raise SystemExit(f"resume contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_dir / scene_name).exists():
                raise SystemExit(f"incomplete or existing output: {scene_name}")
            summary = _build_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"boundaries {summary['planned_boundary_action_count']}, "
                f"candidate actions {summary['candidate_assign_action_feature_count']}",
                flush=True,
            )
        summaries.append(summary)
    result = {
        "ledger_type": "frozen F2 boundary action/candidate no-GT feature ledger",
        "scene_count": len(scenes),
        "planned_boundary_action_count": sum(row["planned_boundary_action_count"] for row in summaries),
        "candidate_assign_action_feature_count": sum(row["candidate_assign_action_feature_count"] for row in summaries),
        "unknown_noop_feature_count": sum(row["unknown_noop_feature_count"] for row in summaries),
        "proposal_materialization_applied": False,
        "ground_truth_usage": "none",
        "decision_constraint": DECISION_CONSTRAINT,
        "feature_groups": {
            "geometry": ["direct core contact", "boundary point/proposal size", "normal and RGB boundary discontinuity"],
            "relative_depth_affinity": ["defined affinity coverage", "mean/min/max affinity", "relative affinity margins"],
            "multiview": ["support views", "observations", "nodes", "track consensus/support coverage"],
        },
        "scene_summaries": summaries,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

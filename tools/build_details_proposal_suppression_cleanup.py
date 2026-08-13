#!/usr/bin/env python3
"""Apply reversible duplicate suppression and Details inclusion cleanup.

This D2c stage consumes converged D2b proposals. Mutual directional coverage
strictly above the paper's 0.99 inclusion threshold forms deterministic duplicate
components; then the paper's asymmetric inclusion cleanup is applied once to the
remaining proposals. Geometry, scores, categories, and source files are not changed.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETAILS_MERGE_IOU = 0.30
DETAILS_INCLUSION_COVERAGE = 0.99


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


def _point_count(superpoint_ids, superpoint_sizes):
    return int(sum(int(superpoint_sizes[item]) for item in superpoint_ids))


def pair_relations(proposals, superpoint_sizes):
    ordered = sorted(proposals, key=lambda row: int(row["proposal_id"]))
    rows = []
    for left, right in combinations(ordered, 2):
        left_ids = set(map(int, left["superpoint_ids"]))
        right_ids = set(map(int, right["superpoint_ids"]))
        left_count = _point_count(left_ids, superpoint_sizes)
        right_count = _point_count(right_ids, superpoint_sizes)
        intersection = _point_count(left_ids & right_ids, superpoint_sizes)
        union = left_count + right_count - intersection
        left_coverage = float(intersection / max(1, left_count))
        right_coverage = float(intersection / max(1, right_count))
        duplicate = (
            left_coverage > DETAILS_INCLUSION_COVERAGE
            and right_coverage > DETAILS_INCLUSION_COVERAGE
        )
        rows.append({
            "left_proposal_id": int(left["proposal_id"]),
            "right_proposal_id": int(right["proposal_id"]),
            "left_point_count": left_count,
            "right_point_count": right_count,
            "point_intersection_count": intersection,
            "point_iou": float(intersection / max(1, union)),
            "left_point_coverage": left_coverage,
            "right_point_coverage": right_coverage,
            "mutual_duplicate_observed": duplicate,
            "left_in_right_observed": bool(
                left_coverage > DETAILS_INCLUSION_COVERAGE and not duplicate
            ),
            "right_in_left_observed": bool(
                right_coverage > DETAILS_INCLUSION_COVERAGE and not duplicate
            ),
            "decision_state": "Fixed D2c relation; strict directional point coverage > 0.99.",
        })
    expected = len(ordered) * (len(ordered) - 1) // 2
    if len(rows) != expected:
        raise ValueError("cleanup relation count is not conserved")
    return rows


def _duplicate_components(proposal_ids, duplicate_pairs):
    parent = {int(item): int(item) for item in proposal_ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for left, right in duplicate_pairs:
        union(int(left), int(right))
    components = defaultdict(list)
    for proposal_id in sorted(parent):
        components[find(proposal_id)].append(proposal_id)
    return [values for values in components.values() if len(values) > 1]


def _ultimate_surviving_containers(proposal_id, containers, suppressed_ids):
    reachable, stack, visited = set(), list(containers.get(proposal_id, [])), set()
    while stack:
        current = int(stack.pop())
        if current in visited:
            continue
        visited.add(current)
        if current not in suppressed_ids:
            reachable.add(current)
        else:
            stack.extend(containers.get(current, []))
    return sorted(reachable)


def cleanup_proposals(source_proposals, superpoint_sizes):
    """Run duplicate suppression, then one-shot asymmetric inclusion cleanup."""
    proposals = [dict(row) for row in source_proposals]
    proposal_by_id = {int(row["proposal_id"]): row for row in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("proposal IDs must be unique")
    relations = pair_relations(proposals, superpoint_sizes)

    duplicate_pairs = [
        (int(row["left_proposal_id"]), int(row["right_proposal_id"]))
        for row in relations
        if row["mutual_duplicate_observed"]
    ]
    duplicate_components = _duplicate_components(proposal_by_id, duplicate_pairs)
    duplicate_representative = {}
    duplicate_actions = []
    for component in sorted(duplicate_components, key=lambda values: values[0]):
        representative = min(component)
        for proposal_id in sorted(set(component) - {representative}):
            duplicate_representative[proposal_id] = representative
            duplicate_actions.append({
                "action": "suppressed_mutual_duplicate",
                "suppressed_proposal_id": proposal_id,
                "replacement_proposal_id": representative,
                "duplicate_component_proposal_ids": sorted(component),
                "threshold_contract": "both directional point coverages strictly > 0.99",
                "decision_state": "Reversible suppression; geometry and score are unchanged.",
            })

    remaining_ids = set(proposal_by_id) - set(duplicate_representative)
    containers = defaultdict(set)
    direct_evidence = defaultdict(list)
    for row in relations:
        left = int(row["left_proposal_id"])
        right = int(row["right_proposal_id"])
        if left not in remaining_ids or right not in remaining_ids:
            continue
        if row["left_in_right_observed"]:
            containers[left].add(right)
            direct_evidence[left].append({
                "container_proposal_id": right,
                "directional_coverage": float(row["left_point_coverage"]),
                "point_iou": float(row["point_iou"]),
            })
        if row["right_in_left_observed"]:
            containers[right].add(left)
            direct_evidence[right].append({
                "container_proposal_id": left,
                "directional_coverage": float(row["right_point_coverage"]),
                "point_iou": float(row["point_iou"]),
            })

    inclusion_suppressed_ids = set(containers)
    all_suppressed_ids = set(duplicate_representative) | inclusion_suppressed_ids
    inclusion_actions = []
    for proposal_id in sorted(inclusion_suppressed_ids):
        ultimate = _ultimate_surviving_containers(
            proposal_id, containers, all_suppressed_ids
        )
        if not ultimate:
            raise ValueError(
                f"included proposal {proposal_id} has no surviving container; "
                "possible unresolved inclusion cycle"
            )
        inclusion_actions.append({
            "action": "suppressed_asymmetrically_included",
            "suppressed_proposal_id": proposal_id,
            "replacement_proposal_id": min(ultimate),
            "direct_container_proposal_ids": sorted(containers[proposal_id]),
            "ultimate_surviving_container_proposal_ids": ultimate,
            "direct_container_evidence": sorted(
                direct_evidence[proposal_id],
                key=lambda row: (-row["directional_coverage"], row["container_proposal_id"]),
            ),
            "threshold_contract": "one directional point coverage strictly > 0.99; applied once",
            "decision_state": "Reversible suppression; geometry and score are unchanged.",
        })

    actions = duplicate_actions + inclusion_actions
    for index, action in enumerate(actions):
        action["action_index"] = index
    final = [
        proposal_by_id[proposal_id]
        for proposal_id in sorted(set(proposal_by_id) - all_suppressed_ids)
    ]
    suppressed = []
    action_by_id = {
        int(row["suppressed_proposal_id"]): row for row in actions
    }
    for proposal_id in sorted(all_suppressed_ids):
        record = dict(proposal_by_id[proposal_id])
        record["suppression_action"] = action_by_id[proposal_id]
        suppressed.append(record)
    validate_cleanup_result(proposals, final, suppressed, actions)
    return final, suppressed, actions, relations


def validate_cleanup_result(source, final, suppressed, actions):
    source_ids = sorted(int(row["proposal_id"]) for row in source)
    final_ids = sorted(int(row["proposal_id"]) for row in final)
    suppressed_ids = sorted(int(row["proposal_id"]) for row in suppressed)
    if sorted(final_ids + suppressed_ids) != source_ids:
        raise ValueError("active and suppressed proposal IDs do not conserve the source")
    if set(final_ids) & set(suppressed_ids):
        raise ValueError("a proposal is both active and suppressed")
    if len(actions) != len(suppressed):
        raise ValueError("each suppressed proposal must have exactly one action")
    replacement_ids = {int(row["replacement_proposal_id"]) for row in actions}
    if not replacement_ids <= set(final_ids):
        raise ValueError("suppression action points to a non-surviving replacement")
    source_lineage = sorted(
        item for row in source for item in row.get("lineage_proposal_ids", [row["proposal_id"]])
    )
    output_lineage = sorted(
        item
        for row in final + suppressed
        for item in row.get("lineage_proposal_ids", [row["proposal_id"]])
    )
    if output_lineage != source_lineage:
        raise ValueError("proposal lineage is not conserved across suppression")


def validate_converged_d2b(relations):
    violating = [row for row in relations if row["point_iou"] > DETAILS_MERGE_IOU]
    if violating:
        first = violating[0]
        raise ValueError(
            "D2c input is not merge-converged: "
            f"{first['left_proposal_id']}, {first['right_proposal_id']} "
            f"have IoU {first['point_iou']}"
        )


def _load_processed(scene_name, processed_scene_root):
    path = processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    processed = np.load(path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} lacks raw superpoint IDs")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    sizes = {int(item): int(count) for item, count in zip(ids, counts)}
    return processed, superpoints, sizes


def _validate_source_points(proposals, superpoints):
    for proposal in proposals:
        with np.load(proposal["points_path"]) as payload:
            actual = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
        expected = np.flatnonzero(
            np.isin(superpoints, np.asarray(proposal["superpoint_ids"], dtype=np.int64))
        )
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"proposal {proposal['proposal_id']} point file is inconsistent"
            )


def _scene_summary(scene_name, source, final, suppressed, actions, relations):
    action_counts = Counter(row["action"] for row in actions)
    return {
        "scene_name": scene_name,
        "source_proposal_count": len(source),
        "final_proposal_count": len(final),
        "suppressed_proposal_count": len(suppressed),
        "duplicate_suppressed_count": int(action_counts["suppressed_mutual_duplicate"]),
        "inclusion_suppressed_count": int(action_counts["suppressed_asymmetrically_included"]),
        "relation_count": len(relations),
        "mutual_duplicate_relation_count": sum(row["mutual_duplicate_observed"] for row in relations),
        "asymmetric_inclusion_relation_count": sum(
            row["left_in_right_observed"] or row["right_in_left_observed"]
            for row in relations
        ),
        "threshold_contract": "strict directional point coverage > 0.99",
        "ground_truth_usage": "none",
        "decision_state": "D2c reversible suppression only; no geometry, score, semantics, or AP changes.",
    }


def _build_and_publish_scene(scene_name, args):
    source_payload = json.loads(
        (args.track_root / scene_name / "automatic_tracks.json").read_text()
    )
    source = source_payload.get("tracks", [])
    processed, superpoints, superpoint_sizes = _load_processed(
        scene_name, args.processed_scene_root
    )
    _validate_source_points(source, superpoints)
    final, suppressed, actions, relations = cleanup_proposals(
        source, superpoint_sizes
    )
    validate_converged_d2b(relations)
    summary = _scene_summary(
        scene_name, source, final, suppressed, actions, relations
    )

    published = args.output_root / scene_name
    staging = args.output_root / f".{scene_name}.tmp.{os.getpid()}"
    if published.exists() or staging.exists():
        raise FileExistsError(f"scene output already exists: {scene_name}")
    staging.mkdir()
    try:
        point_root = staging / "track_points"
        point_root.mkdir()
        public_final = []
        for proposal in final:
            proposal_id = int(proposal["proposal_id"])
            filename = f"track{proposal_id:04d}_points.npz"
            expected = np.flatnonzero(
                np.isin(superpoints, np.asarray(proposal["superpoint_ids"], dtype=np.int64))
            )
            np.savez_compressed(point_root / filename, point_indices=expected)
            record = dict(proposal)
            record["points_path"] = str(
                args.output_root / scene_name / "track_points" / filename
            )
            record["decision_state"] = (
                "Active after D2c reversible duplicate/inclusion suppression."
            )
            public_final.append(record)
        payload = {
            "scene_name": scene_name,
            "source_track_count": len(source),
            "track_count": len(public_final),
            "suppressed_track_count": len(suppressed),
            "tracks": public_final,
        }
        (staging / "automatic_tracks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _write_jsonl(staging / "suppressed_proposals.jsonl", suppressed)
        _write_jsonl(staging / "suppression_actions.jsonl", actions)
        _write_jsonl(staging / "cleanup_relations.jsonl", relations)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "track_root", "processed_scene_root", "output_root"):
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
            if summary.get("threshold_contract") != "strict directional point coverage > 0.99":
                raise SystemExit(f"resume threshold contract mismatch: {scene_name}")
            print(f"[resume] {index}/{len(scenes)} {scene_name}", flush=True)
        else:
            if (args.output_root / scene_name).exists():
                raise SystemExit(f"incomplete or existing scene output: {scene_name}")
            summary = _build_and_publish_scene(scene_name, args)
            print(
                f"[done] {index}/{len(scenes)} {scene_name}: "
                f"{summary['source_proposal_count']} -> {summary['final_proposal_count']}, "
                f"suppressed {summary['suppressed_proposal_count']}",
                flush=True,
            )
        summaries.append(summary)

    totals = Counter()
    for row in summaries:
        for key in (
            "source_proposal_count", "final_proposal_count", "suppressed_proposal_count",
            "duplicate_suppressed_count", "inclusion_suppressed_count", "relation_count",
            "mutual_duplicate_relation_count", "asymmetric_inclusion_relation_count",
        ):
            totals[key] += int(row[key])
    payload = {
        "scene_count": len(summaries),
        **dict(totals),
        "threshold_contract": "strict directional point coverage > 0.99",
        "ground_truth_usage": "none",
        "decision_state": "D2c reversible suppression only; no geometry, score, semantics, or AP changes.",
        "params": vars(args),
        "scene_summaries": summaries,
    }
    (args.output_root / "details_suppression_cleanup_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "source_proposal_count": payload.get("source_proposal_count", 0),
        "final_proposal_count": payload.get("final_proposal_count", 0),
        "duplicate_suppressed_count": payload.get("duplicate_suppressed_count", 0),
        "inclusion_suppressed_count": payload.get("inclusion_suppressed_count", 0),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

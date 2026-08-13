#!/usr/bin/env python3
"""Build GT-label-only native-geometry-collapsed pair ledgers for train scenes.

The input v1 relation ledger enumerates every YOLO-World class expansion of a
native Mask3D geometry.  This tool replaces it with one track--native-geometry
relation, records the original members for audit, and assigns only offline
training labels.  It never emits pair predictions or replacement actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_train_candidate_quality_dataset import (
    NATIVE_SOURCE, TRACK_SOURCE, candidate_ledger_path, read_jsonl, read_scene_list,
)


IOU_MARGIN = 0.05


def _native_groups(mask_path: Path, scene: str) -> tuple[dict[int, dict], list[dict]]:
    masks = np.load(mask_path, mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene}: native masks must be point-by-candidate")
    by_digest: dict[str, list[int]] = defaultdict(list)
    for candidate_id in range(masks.shape[1]):
        packed = np.packbits(np.asarray(masks[:, candidate_id], dtype=np.uint8)).tobytes()
        by_digest[hashlib.sha256(packed).hexdigest()].append(candidate_id)
    groups = []
    member_to_group = {}
    for index, members in enumerate(sorted(by_digest.values(), key=lambda values: values[0])):
        group_id = f"{scene}:native_geometry:{index:04d}"
        row = {"native_exact_geometry_group_id": group_id, "native_member_candidate_ids": members}
        groups.append(row)
        member_to_group.update({member: row for member in members})
    if len(member_to_group) != masks.shape[1]:
        raise AssertionError(f"{scene}: native grouping did not conserve candidates")
    return member_to_group, groups


def _relation_label(track: dict, native: dict) -> tuple[str, bool | None, float | None, bool, str]:
    track_gt, native_gt = track.get("label_best_gt_instance_id"), native.get("label_best_gt_instance_id")
    track_reliable = bool(track.get("label_valid_iou25"))
    native_reliable = bool(native.get("label_valid_iou25"))
    reliable_pair = bool(track_reliable and native_reliable)
    if reliable_pair:
        reliability_state = "both_iou25_reliable"
    elif track_reliable:
        reliability_state = "track_only_iou25_reliable"
    elif native_reliable:
        reliability_state = "native_only_iou25_reliable"
    else:
        reliability_state = "neither_iou25_reliable"
    # The argmax GT ID is arbitrary when a candidate has no IoU25 match.  Such
    # pairs can retain one-sided quality observations, but cannot supervise a
    # same-target gate or replacement action.
    if not reliable_pair or track_gt is None or native_gt is None:
        return "unknown", None, None, reliable_pair, reliability_state
    same_best_gt = int(track_gt) == int(native_gt)
    margin = float(track["label_best_gt_iou"]) - float(native["label_best_gt_iou"])
    if not same_best_gt:
        return "coexist", False, margin, reliable_pair, reliability_state
    if margin > IOU_MARGIN:
        return "prefer_track", True, margin, reliable_pair, reliability_state
    if margin < -IOU_MARGIN:
        return "prefer_native", True, margin, reliable_pair, reliability_state
    return "equivalent_abstain", True, margin, reliable_pair, reliability_state


def build_scene(scene: str, records_root: Path) -> tuple[list[dict], dict]:
    candidate_rows = read_jsonl(candidate_ledger_path(records_root, scene))
    candidates = {(row["candidate_source"], int(row["candidate_id"])): row for row in candidate_rows}
    native_candidates = {key[1]: row for key, row in candidates.items() if key[0] == NATIVE_SOURCE}
    track_candidates = {key[1]: row for key, row in candidates.items() if key[0] == TRACK_SOURCE}
    mask_path = records_root / scene / "native_cache" / f"{scene}_pred_masks.npy"
    member_to_group, groups = _native_groups(mask_path, scene)
    if set(member_to_group) != set(native_candidates):
        raise ValueError(f"{scene}: native mask IDs differ from candidate ledger")
    relation_path = records_root / scene / "candidate_quality_training_ledger" / scene / "pair_labels.jsonl"
    source_relations = read_jsonl(relation_path)
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for relation in source_relations:
        track_id, native_id = int(relation["track_id"]), int(relation["native_candidate_id"])
        if track_id not in track_candidates or native_id not in native_candidates:
            raise ValueError(f"{scene}: relation references unknown candidate")
        grouped[(track_id, member_to_group[native_id]["native_exact_geometry_group_id"])].append(relation)
    pair_rows = []
    folded_relation_count = 0
    for (track_id, group_id), relations in sorted(grouped.items()):
        first = relations[0]
        observed_fields = ("point_iou", "track_inside_native_ratio", "native_inside_track_ratio", "mutual_duplicate_strict_099")
        for relation in relations[1:]:
            if any(relation[field] != first[field] for field in observed_fields):
                raise ValueError(f"{scene}: exact native geometry group has inconsistent relation geometry")
        members = member_to_group[int(first["native_candidate_id"])]["native_member_candidate_ids"]
        native_rows = [native_candidates[member] for member in members]
        native = native_rows[0]
        label_fields = ("label_best_gt_iou", "label_best_gt_instance_id", "label_valid_iou25", "label_valid_iou50")
        if any(any(row.get(field) != native.get(field) for field in label_fields) for row in native_rows[1:]):
            raise ValueError(f"{scene}: exact native geometry group has inconsistent GT labels")
        track = track_candidates[track_id]
        label, same_best_gt, margin, reliable_pair, reliability_state = _relation_label(track, native)
        folded_relation_count += len(relations)
        pair_rows.append({
            "scene_name": scene,
            "track_id": track_id,
            "native_exact_geometry_group_id": group_id,
            "native_member_candidate_ids": members,
            "native_exact_geometry_group_size": len(members),
            "native_anchor_candidate_id": min(members),
            "native_original_score_min": min(float(row["original_source_score"]) for row in native_rows),
            "native_original_score_median": float(np.median([float(row["original_source_score"]) for row in native_rows])),
            "native_original_score_max": max(float(row["original_source_score"]) for row in native_rows),
            "track_original_score": float(track["original_source_score"]),
            "point_iou": float(first["point_iou"]),
            "track_inside_native_ratio": float(first["track_inside_native_ratio"]),
            "native_inside_track_ratio": float(first["native_inside_track_ratio"]),
            "mutual_duplicate_strict_099": bool(first["mutual_duplicate_strict_099"]),
            "source_relation_count": len(relations),
            "track_best_gt_instance_id": track.get("label_best_gt_instance_id"),
            "native_best_gt_instance_id": native.get("label_best_gt_instance_id"),
            "track_gt_reliable_iou25": bool(track["label_valid_iou25"]),
            "native_gt_reliable_iou25": bool(native["label_valid_iou25"]),
            "reliable_pair": reliable_pair,
            "reliability_state": reliability_state,
            "same_best_gt": same_best_gt,
            "track_best_gt_iou": float(track["label_best_gt_iou"]),
            "native_best_gt_iou": float(native["label_best_gt_iou"]),
            "iou_margin": margin,
            "label_pair_preference": label,
            "ground_truth_usage": "label_only",
            "decision_state": "training label only; coexist/no-op at inference until a future validated model",
        })
    if folded_relation_count != len(source_relations):
        raise AssertionError(f"{scene}: exact-geometry relation folding did not conserve v1 relations")
    if sum(len(group["native_member_candidate_ids"]) for group in groups) != len(native_candidates):
        raise AssertionError(f"{scene}: exact-geometry groups did not conserve native candidates")
    summary = {
        "scene_name": scene,
        "native_candidate_count": len(native_candidates),
        "native_exact_geometry_group_count": len(groups),
        "track_candidate_count": len(track_candidates),
        "v1_relation_count": len(source_relations),
        "v2_geometry_relation_count": len(pair_rows),
        "folded_source_relation_count": folded_relation_count,
        "label_counts": dict(sorted(Counter(row["label_pair_preference"] for row in pair_rows).items())),
        "reliability_state_counts": dict(sorted(Counter(row["reliability_state"] for row in pair_rows).items())),
        "ground_truth_usage": "label_only",
        "pair_prediction_trained": False,
        "replacement_materialized": False,
    }
    return pair_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    args = parser.parse_args()
    scenes = read_scene_list(args.scene_list)
    if len(scenes) != args.expected_scene_count:
        raise ValueError(f"{args.protocol_name} requires exactly {args.expected_scene_count} scenes, got {len(scenes)}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"output root is non-empty: {args.output_root}")
    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    summaries = []
    try:
        for scene in scenes:
            pairs, summary = build_scene(scene, args.records_root)
            scene_root = staging / scene
            scene_root.mkdir()
            (scene_root / "pair_labels_v2.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs))
            (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            summaries.append(summary)
        payload = {
            "version": f"{args.protocol_name}_candidate_pair_ledger_v2",
            "protocol_name": args.protocol_name,
            "scene_count": len(summaries),
            "scene_list_path": str(args.scene_list.resolve()),
            "scene_list_sha256": hashlib.sha256(args.scene_list.read_bytes()).hexdigest(),
            "v1_relation_count": sum(row["v1_relation_count"] for row in summaries),
            "v2_geometry_relation_count": sum(row["v2_geometry_relation_count"] for row in summaries),
            "folded_source_relation_count": sum(row["folded_source_relation_count"] for row in summaries),
            "label_counts": dict(sorted(sum((Counter(row["label_counts"]) for row in summaries), Counter()).items())),
            "reliability_state_counts": dict(sorted(sum((Counter(row["reliability_state_counts"]) for row in summaries), Counter()).items())),
            "ground_truth_usage": "label_only",
            "pair_prediction_trained": False,
            "replacement_materialized": False,
            "scene_summaries": summaries,
        }
        (staging / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

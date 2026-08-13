#!/usr/bin/env python3
"""Build the first N2 no-GT SAMPro3D family quality/duplicate ledger.

The N1a observation space is frozen.  This tool only describes each seed
family's stable core, unknown boundary, internal agreement, exact duplicate
fingerprint, and geometry relation to frozen D2b tracks.  It does not choose a
SAM hypothesis, merge views, suppress anything, or read GT/native/semantics.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTRACT = (
    "N2 no-GT family-quality and duplicate ledger only: no hypothesis selection, "
    "view aggregation, proposal materialization, suppression, score/class/native/GT/AP use."
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _scenes(path):
    rows = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or duplicated")
    return rows


def set_metrics(left, right):
    left, right = set(map(int, left)), set(map(int, right))
    intersection = len(left & right)
    union = len(left | right)
    return {
        "intersection_superpoint_count": intersection,
        "union_superpoint_count": union,
        "jaccard": float(intersection / max(1, union)),
        "left_covered_ratio": float(intersection / max(1, len(left))),
        "right_covered_ratio": float(intersection / max(1, len(right))),
    }


def summarize_family(members):
    """Pure, no-GT seed-family core/boundary and agreement summary."""
    if not members:
        raise ValueError("family has no members")
    members = sorted(members, key=lambda row: (int(row["frame_index"]), int(row["hypothesis_index"]), row["candidate_id"]))
    nonempty = [row for row in members if row["candidate_superpoint_ids"]]
    sets = [set(map(int, row["candidate_superpoint_ids"])) for row in nonempty]
    union = set.union(*sets) if sets else set()
    core = set.intersection(*sets) if sets else set()
    pair_jaccard = []
    cross_view_jaccard = []
    exact_pairs = 0
    for index, left in enumerate(nonempty):
        for right in nonempty[index + 1:]:
            metrics = set_metrics(left["candidate_superpoint_ids"], right["candidate_superpoint_ids"])
            pair_jaccard.append(metrics["jaccard"])
            if int(left["frame_index"]) != int(right["frame_index"]):
                cross_view_jaccard.append(metrics["jaccard"])
            exact_pairs += int(metrics["jaccard"] == 1.0)
    top_sam = min(members, key=lambda row: (-float(row["sam_predicted_iou"]), row["candidate_id"]))
    support = Counter()
    support_views = defaultdict(set)
    for row in nonempty:
        frame = int(row["frame_index"])
        for sp in set(map(int, row["candidate_superpoint_ids"])):
            support[sp] += 1
            support_views[sp].add(frame)
    return {
        "candidate_family_key": str(members[0]["candidate_family_key"]),
        "seed_superpoint_id": int(members[0]["seed_superpoint_id"]),
        "member_count": len(members),
        "nonempty_member_count": len(nonempty),
        "empty_member_count": len(members) - len(nonempty),
        "distinct_view_count": len({int(row["frame_index"]) for row in members}),
        "distinct_nonempty_view_count": len({int(row["frame_index"]) for row in nonempty}),
        "reliable_core_superpoint_ids": sorted(core),
        "reliable_core_superpoint_count": len(core),
        "unknown_boundary_superpoint_ids": sorted(union - core),
        "unknown_boundary_superpoint_count": len(union - core),
        "family_union_superpoint_ids": sorted(union),
        "family_union_superpoint_count": len(union),
        "per_superpoint_observation_support_count": {str(sp): int(support[sp]) for sp in sorted(support)},
        "per_superpoint_view_support_count": {str(sp): len(support_views[sp]) for sp in sorted(support_views)},
        "internal_pair_count": len(pair_jaccard),
        "internal_exact_geometry_pair_count": exact_pairs,
        "internal_mean_jaccard": float(np.mean(pair_jaccard)) if pair_jaccard else 0.0,
        "internal_max_jaccard": float(max(pair_jaccard, default=0.0)),
        "cross_view_pair_count": len(cross_view_jaccard),
        "cross_view_mean_jaccard": float(np.mean(cross_view_jaccard)) if cross_view_jaccard else 0.0,
        "cross_view_max_jaccard": float(max(cross_view_jaccard, default=0.0)),
        "sam_top_member_candidate_id": top_sam["candidate_id"],
        "sam_top_member_frame_index": int(top_sam["frame_index"]),
        "sam_top_member_hypothesis_index": int(top_sam["hypothesis_index"]),
        "sam_top_member_predicted_iou": float(top_sam["sam_predicted_iou"]),
        "exact_duplicate_fingerprint": ",".join(map(str, sorted(union))),
    }


def d2b_relation(family, tracks):
    family_ids = family["family_union_superpoint_ids"]
    best = None
    for track in tracks:
        metrics = set_metrics(family_ids, track["superpoint_ids"])
        candidate = (metrics["jaccard"], metrics["left_covered_ratio"], metrics["right_covered_ratio"], -int(track["track_id"]))
        if best is None or candidate > best[0]:
            best = (candidate, track, metrics)
    if best is None:
        return {"best_d2b_track_id": None, "best_d2b_jaccard": 0.0, "best_d2b_family_covered_ratio": 0.0, "best_d2b_track_covered_ratio": 0.0, "exact_mutual_duplicate": False, "family_contained_by_d2b": False, "d2b_contained_by_family": False}
    _, track, metrics = best
    return {
        "best_d2b_track_id": int(track["track_id"]),
        "best_d2b_jaccard": float(metrics["jaccard"]),
        "best_d2b_family_covered_ratio": float(metrics["left_covered_ratio"]),
        "best_d2b_track_covered_ratio": float(metrics["right_covered_ratio"]),
        "exact_mutual_duplicate": bool(metrics["left_covered_ratio"] > .99 and metrics["right_covered_ratio"] > .99),
        "family_contained_by_d2b": bool(metrics["left_covered_ratio"] > .99),
        "d2b_contained_by_family": bool(metrics["right_covered_ratio"] > .99),
    }


def _scene(scene, args):
    rows = [json.loads(line) for line in (args.candidate_ledger_root / scene / "observation_candidate_ledger.jsonl").read_text().splitlines() if line.strip()]
    expected_ids = [row["candidate_id"] for row in rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"{scene} candidate IDs are duplicated")
    groups = defaultdict(list)
    for row in rows:
        if row.get("ground_truth_usage") != "none" or row.get("proposal_materialization_applied"):
            raise ValueError(f"{scene} candidate ledger violates frozen no-GT contract")
        groups[str(row["candidate_family_key"])].append(row)
    source = json.loads((args.d2b_track_root / scene / "automatic_tracks.json").read_text())
    tracks = source.get("tracks", [])
    summaries = []
    for key, members in sorted(groups.items()):
        family = summarize_family(members)
        family.update(d2b_relation(family, tracks))
        family.update({"scene_name": scene, "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False, "decision_constraint": CONTRACT})
        summaries.append(family)
    fingerprints = defaultdict(list)
    for row in summaries:
        if row["family_union_superpoint_ids"]:
            fingerprints[row["exact_duplicate_fingerprint"]].append(row["candidate_family_key"])
    duplicates = []
    for fingerprint, family_keys in sorted(fingerprints.items()):
        if len(family_keys) > 1:
            duplicates.append({"scene_name": scene, "duplicate_kind": "exact_family_union_geometry", "candidate_family_keys": sorted(family_keys), "family_count": len(family_keys), "superpoint_fingerprint": fingerprint, "ground_truth_usage": "none", "proposal_materialization_applied": False})
    summary = {"scene_name": scene, "observation_candidate_count": len(rows), "candidate_family_count": len(summaries), "empty_family_count": sum(not row["family_union_superpoint_ids"] for row in summaries), "exact_cross_family_duplicate_group_count": len(duplicates), "exact_cross_family_duplicate_member_count": sum(row["family_count"] for row in duplicates), "exact_d2b_mutual_duplicate_family_count": sum(row["exact_mutual_duplicate"] for row in summaries), "ground_truth_usage": "none", "proposal_materialization_applied": False, "ap_computed": False}
    return summaries, duplicates, summary


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-ledger-root", type=Path, required=True)
    parser.add_argument("--d2b-track-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in ("scene_list", "candidate_ledger_root", "d2b_track_root", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    scenes = _scenes(args.scene_list)[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for number, scene in enumerate(scenes, 1):
        families, duplicates, summary = _scene(scene, args)
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"; stage.mkdir()
        _write_jsonl(stage / "candidate_family_quality_ledger.jsonl", families)
        _write_jsonl(stage / "exact_duplicate_ledger.jsonl", duplicates)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_root / scene); all_summaries.append(summary)
        print(f"[N2 family ledger] {number}/{len(scenes)} {scene}: families={summary['candidate_family_count']}", flush=True)
    root = {"diagnostic_type": "N2 no-GT SAMPro3D candidate family quality and duplicate ledger", "decision_constraint": CONTRACT, "scene_count": len(all_summaries), "proposal_materialization_applied": False, "ap_computed": False, "scene_summaries": all_summaries, "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    for key in ("observation_candidate_count", "candidate_family_count", "empty_family_count", "exact_cross_family_duplicate_group_count", "exact_cross_family_duplicate_member_count", "exact_d2b_mutual_duplicate_family_count"):
        root[key] = sum(int(row[key]) for row in all_summaries)
    (args.output_root / "summary.json").write_text(json.dumps(root, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in root.items() if key != "scene_summaries"}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

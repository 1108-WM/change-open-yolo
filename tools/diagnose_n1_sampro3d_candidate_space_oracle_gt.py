#!/usr/bin/env python3
"""GT-only N1b oracle over the frozen N1a SAM observation candidate ledger.

This is deliberately a *candidate-space* diagnostic.  A seed's at most three
views times three SAM multimasks are one mutually-exclusive candidate family,
not nine predictions.  The tool writes no masks/proposals and never lets GT
select an inference-time observation.  Its only purpose is to decide whether
N1 has enough geometry headroom to justify N2.
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

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST

CONTRACT = (
    "GT-only offline N1b candidate-space oracle. GT may label frozen candidate "
    "space only; it must not select SAM hypotheses for inference, materialize "
    "proposals, alter scores, or enter N2/GVC/semantics."
)
THRESHOLDS = (0.25, 0.50)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _scenes(path):
    rows = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or duplicated")
    return rows


def _load_gt(path, min_region_size):
    ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = set(map(int, VALID_CLASS_IDS_200_INST))
    values, counts = np.unique(ids, return_counts=True)
    sizes = {
        int(value): int(count)
        for value, count in zip(values, counts)
        if int(value) > 0
        and int(value) // 1000 in valid_classes
        and int(count) >= int(min_region_size)
    }
    return ids, sizes


def _sp_gt_counts(superpoints, gt_ids, gt_sizes):
    """Sparse superpoint -> valid GT intersection counts, computed once/scene."""
    table = defaultdict(dict)
    for sp, gt in zip(superpoints, gt_ids):
        gt = int(gt)
        if gt in gt_sizes:
            bucket = table[int(sp)]
            bucket[gt] = bucket.get(gt, 0) + 1
    return table


def _geometry_for_superpoints(superpoint_ids, sp_sizes, sp_gt):
    ids = sorted(set(map(int, superpoint_ids)))
    size = sum(int(sp_sizes.get(sp, 0)) for sp in ids)
    intersections = Counter()
    for sp in ids:
        intersections.update(sp_gt.get(sp, {}))
    return ids, int(size), {int(k): int(v) for k, v in intersections.items()}


def _ious(candidate_size, intersections, gt_sizes):
    return {
        int(gt): float(inter / max(1, candidate_size + int(gt_sizes[gt]) - inter))
        for gt, inter in intersections.items() if gt in gt_sizes
    }


def _best(iou_by_gt):
    if not iou_by_gt:
        return -1, 0.0
    target = min(iou_by_gt, key=lambda gt: (-iou_by_gt[gt], gt))
    return target, float(iou_by_gt[target])


def _load_track_iou(track_root, scene, gt_ids, gt_sizes):
    payload = json.loads((track_root / scene / "automatic_tracks.json").read_text())
    best = {int(gt): 0.0 for gt in gt_sizes}
    point_count = len(gt_ids)
    for track in payload.get("tracks", []):
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < point_count)]
        if not len(points):
            continue
        values, counts = np.unique(gt_ids[points], return_counts=True)
        for gt, inter in zip(values, counts):
            gt = int(gt)
            if gt not in gt_sizes:
                continue
            iou = float(int(inter) / max(1, len(points) + gt_sizes[gt] - int(inter)))
            best[gt] = max(best[gt], iou)
    return best


def _load_native_iou(native_root, scene, gt_ids, gt_sizes):
    masks = np.load(native_root / f"{scene}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene} native masks must be 2-D")
    if masks.shape[0] != len(gt_ids) and masks.shape[1] == len(gt_ids):
        masks = masks.T
    if masks.shape[0] != len(gt_ids):
        raise ValueError(f"{scene} native point dimension mismatches GT")
    sizes = masks.sum(axis=0, dtype=np.int64)
    best = {int(gt): 0.0 for gt in gt_sizes}
    for gt, gt_size in gt_sizes.items():
        intersections = np.asarray(masks[gt_ids == gt].sum(axis=0, dtype=np.int64)).reshape(-1)
        if len(intersections):
            best[int(gt)] = float(np.max(intersections / np.maximum(1, sizes + gt_size - intersections)))
    return best


def maximum_cardinality_matching(family_edges, threshold):
    """Deterministic Kuhn maximum matching for one threshold.

    Edges are ``family -> {gt: best-family-member IoU}``.  Sorting by IoU only
    breaks ties; cardinality, not local GT utility, is the measured oracle.
    """
    match_gt = {}
    chosen = {}
    for family in sorted(family_edges, key=lambda item: (len([v for v in family_edges[item].values() if v >= threshold]), item)):
        seen = set()
        def visit(current):
            options = sorted(
                ((gt, value) for gt, value in family_edges[current].items() if value >= threshold),
                key=lambda pair: (-pair[1], pair[0]),
            )
            for gt, _ in options:
                if gt in seen:
                    continue
                seen.add(gt)
                previous = match_gt.get(gt)
                if previous is None or visit(previous):
                    match_gt[gt] = current
                    chosen[current] = gt
                    return True
            return False
        visit(family)
    return {family: gt for family, gt in chosen.items() if match_gt.get(gt) == family}


def family_summary(members, sp_sizes, sp_gt, gt_sizes):
    """Pure seed-family aggregation used only by the GT diagnostic."""
    family_key = str(members[0]["candidate_family_key"])
    by_gt = defaultdict(lambda: {"best_iou": 0.0, "best_member": None, "views": set()})
    union_sp = set()
    for member in members:
        union_sp.update(map(int, member["candidate_superpoint_ids"]))
        for gt, iou in member["iou_by_gt"].items():
            gt = int(gt)
            current = by_gt[gt]
            if iou > current["best_iou"] or (iou == current["best_iou"] and member["candidate_id"] < current["best_member"]):
                current["best_iou"] = float(iou)
                current["best_member"] = member["candidate_id"]
            if iou > 0:
                current["views"].add(int(member["frame_index"]))
    union_ids, union_size, union_intersections = _geometry_for_superpoints(union_sp, sp_sizes, sp_gt)
    union_iou = _ious(union_size, union_intersections, gt_sizes)
    best_gt, best_iou = _best({gt: row["best_iou"] for gt, row in by_gt.items()})
    top_sam = min(members, key=lambda row: (-float(row["sam_predicted_iou"]), row["candidate_id"]))
    top_sam_iou = float(top_sam["iou_by_gt"].get(best_gt, 0.0)) if best_gt > 0 else 0.0
    best_member_id = by_gt[best_gt]["best_member"] if best_gt > 0 else None
    return {
        "candidate_family_key": family_key,
        "seed_superpoint_id": int(members[0]["seed_superpoint_id"]),
        "member_count": len(members),
        "view_count": len({int(row["frame_index"]) for row in members}),
        "best_single_observation_gt_instance_id": best_gt,
        "best_single_observation_iou": best_iou,
        "best_single_observation_candidate_id": best_member_id,
        "sam_top_observation_candidate_id": top_sam["candidate_id"],
        "sam_top_observation_iou_on_best_gt": top_sam_iou,
        "sam_predicted_iou_selects_gt_best_observation": bool(top_sam["candidate_id"] == best_member_id),
        "per_gt_best_single_iou": {str(gt): float(row["best_iou"]) for gt, row in sorted(by_gt.items())},
        "per_gt_support_view_count": {str(gt): len(row["views"]) for gt, row in sorted(by_gt.items())},
        "diagnostic_multiview_union_superpoint_count": len(union_ids),
        "diagnostic_multiview_union_iou_by_gt": {str(gt): float(v) for gt, v in sorted(union_iou.items())},
        "multiview_aggregation_needed_iou25": any(
            union_iou.get(gt, 0.0) >= .25 and row["best_iou"] < .25 and len(row["views"]) >= 2
            for gt, row in by_gt.items()
        ),
        "multiview_aggregation_needed_iou50": any(
            union_iou.get(gt, 0.0) >= .50 and row["best_iou"] < .50 and len(row["views"]) >= 2
            for gt, row in by_gt.items()
        ),
    }


def _scene(scene, args):
    gt_ids, gt_sizes = _load_gt(args.gt_instance_dir / f"{scene}.txt", args.min_region_size)
    processed = np.load(args.processed_scene_root / scene / f"{scene.replace('scene', '')}.npy", mmap_mode="r")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    ids, counts = np.unique(superpoints, return_counts=True)
    sp_sizes = {int(sp): int(n) for sp, n in zip(ids, counts)}
    sp_gt = _sp_gt_counts(superpoints, gt_ids, gt_sizes)
    d2b_iou = _load_track_iou(args.d2b_track_root, scene, gt_ids, gt_sizes)
    native_iou = _load_native_iou(args.native_prediction_cache, scene, gt_ids, gt_sizes)
    raw_rows = [json.loads(line) for line in (args.candidate_ledger_root / scene / "observation_candidate_ledger.jsonl").read_text().splitlines() if line.strip()]
    candidates = []
    families = defaultdict(list)
    for raw in raw_rows:
        sp_ids, size, intersections = _geometry_for_superpoints(raw["candidate_superpoint_ids"], sp_sizes, sp_gt)
        iou_by_gt = _ious(size, intersections, gt_sizes)
        best_gt, best_iou = _best(iou_by_gt)
        seed_id = int(raw["seed_superpoint_id"])
        seed_intersections = sp_gt.get(seed_id, {})
        seed_target, _ = _best(_ious(sp_sizes.get(seed_id, 0), seed_intersections, gt_sizes))
        seed_target_ratio = float(seed_intersections.get(best_gt, 0) / max(1, sp_sizes.get(seed_id, 0))) if best_gt > 0 else 0.0
        row = {
            "scene_name": scene, "candidate_id": raw["candidate_id"], "candidate_family_key": raw["candidate_family_key"],
            "seed_superpoint_id": seed_id, "frame_index": int(raw["frame_index"]), "hypothesis_index": int(raw["hypothesis_index"]),
            "sam_predicted_iou": float(raw["sam_predicted_iou"]), "candidate_superpoint_count": len(sp_ids),
            "candidate_superpoint_ids": sp_ids,
            "candidate_point_count": size, "candidate_is_empty": size == 0,
            "best_gt_instance_id": best_gt, "best_gt_iou": best_iou,
            "iou25": best_iou >= .25, "iou50": best_iou >= .50,
            "seed_best_gt_instance_id": seed_target, "seed_belongs_to_best_gt": bool(seed_target == best_gt and best_gt > 0),
            "seed_point_ratio_on_best_gt": seed_target_ratio,
            "d2b_best_iou_on_candidate_target": float(d2b_iou.get(best_gt, 0.0)),
            "native_best_iou_on_candidate_target": float(native_iou.get(best_gt, 0.0)),
            "d2b_already_covers_target_iou25": bool(d2b_iou.get(best_gt, 0.0) >= .25),
            "d2b_already_covers_target_iou50": bool(d2b_iou.get(best_gt, 0.0) >= .50),
            "native_already_covers_target_iou25": bool(native_iou.get(best_gt, 0.0) >= .25),
            "native_already_covers_target_iou50": bool(native_iou.get(best_gt, 0.0) >= .50),
            "iou_by_gt": {str(gt): float(value) for gt, value in sorted(iou_by_gt.items())},
            "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False, "ap_computed": False,
        }
        candidates.append(row); families[row["candidate_family_key"]].append(row)
    family_rows = [family_summary(rows, sp_sizes, sp_gt, gt_sizes) for _, rows in sorted(families.items())]
    family_edges = {row["candidate_family_key"]: {int(gt): float(value) for gt, value in row["per_gt_best_single_iou"].items()} for row in family_rows}
    matches = {threshold: maximum_cardinality_matching(family_edges, threshold) for threshold in THRESHOLDS}
    matched_gt = {threshold: set(matches[threshold].values()) for threshold in THRESHOLDS}
    gt_rows = []
    for gt in sorted(gt_sizes):
        best_family, best_family_iou = _best({family: edges.get(gt, 0.0) for family, edges in family_edges.items()})
        union_best = max((float(row["diagnostic_multiview_union_iou_by_gt"].get(str(gt), 0.0)) for row in family_rows), default=0.0)
        row = {"scene_name": scene, "gt_instance_id": gt, "gt_point_count": gt_sizes[gt], "best_single_family_key": best_family, "best_single_family_iou": best_family_iou, "best_diagnostic_multiview_union_iou": union_best, "d2b_best_iou": float(d2b_iou[gt]), "native_best_iou": float(native_iou[gt])}
        for threshold in THRESHOLDS:
            tag = str(int(threshold * 100))
            row[f"matched_by_n1_iou{tag}"] = gt in matched_gt[threshold]
            row[f"new_vs_d2b_iou{tag}"] = bool(gt in matched_gt[threshold] and d2b_iou[gt] < threshold)
            row[f"new_vs_native_iou{tag}"] = bool(gt in matched_gt[threshold] and native_iou[gt] < threshold)
            row[f"native_duplicate_iou{tag}"] = bool(gt in matched_gt[threshold] and native_iou[gt] >= threshold)
            row[f"boundary_only_no_new_match_iou{tag}"] = bool(best_family_iou > d2b_iou[gt] + 1e-12 and d2b_iou[gt] >= threshold)
        gt_rows.append(row)
    summary = {"scene_name": scene, "valid_gt_instance_count": len(gt_sizes), "observation_candidate_count": len(candidates), "candidate_family_count": len(family_rows), "ground_truth_usage": "offline_diagnostic_only", "proposal_materialization_applied": False, "ap_computed": False}
    for threshold in THRESHOLDS:
        tag = str(int(threshold * 100)); subset = [row for row in gt_rows if row[f"matched_by_n1_iou{tag}"]]
        summary.update({
            f"n1_matched_instance_count_iou{tag}": len(subset),
            f"new_vs_d2b_instance_count_iou{tag}": sum(row[f"new_vs_d2b_iou{tag}"] for row in gt_rows),
            f"new_vs_native_instance_count_iou{tag}": sum(row[f"new_vs_native_iou{tag}"] for row in gt_rows),
            f"native_duplicate_instance_count_iou{tag}": sum(row[f"native_duplicate_iou{tag}"] for row in gt_rows),
            f"boundary_only_no_new_match_instance_count_iou{tag}": sum(row[f"boundary_only_no_new_match_iou{tag}"] for row in gt_rows),
        })
    summary["families_requiring_diagnostic_multiview_iou25"] = sum(row["multiview_aggregation_needed_iou25"] for row in family_rows)
    summary["families_requiring_diagnostic_multiview_iou50"] = sum(row["multiview_aggregation_needed_iou50"] for row in family_rows)
    return candidates, family_rows, gt_rows, summary


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--candidate-ledger-root", type=Path, required=True)
    parser.add_argument("--d2b-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("--allow-gt-diagnostics is required; this is GT-only offline diagnostics")
    for name in ("scene_list", "candidate_ledger_root", "d2b_track_root", "native_prediction_cache", "processed_scene_root", "gt_instance_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"output root is non-empty: {args.output_root}")
    scenes = _scenes(args.scene_list)[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for number, scene in enumerate(scenes, 1):
        candidates, families, gt_rows, summary = _scene(scene, args)
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"; stage.mkdir()
        _write_jsonl(stage / "candidate_oracle_gt.jsonl", candidates)
        _write_jsonl(stage / "seed_family_oracle_gt.jsonl", families)
        _write_jsonl(stage / "gt_instance_oracle_gt.jsonl", gt_rows)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output_root / scene); summaries.append(summary)
        print(f"[N1b oracle] {number}/{len(scenes)} {scene}: families={summary['candidate_family_count']}", flush=True)
    totals = {"scene_count": len(summaries), "diagnostic_type": "N1b GT-only SAMPro3D candidate-space oracle", "decision_constraint": CONTRACT, "proposal_materialization_applied": False, "ap_computed": False, "summaries": summaries, "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    for key in summaries[0] if summaries else ():
        if key.endswith("count_iou25") or key.endswith("count_iou50") or key in {"observation_candidate_count", "candidate_family_count", "valid_gt_instance_count", "families_requiring_diagnostic_multiview_iou25", "families_requiring_diagnostic_multiview_iou50"}:
            totals[key] = sum(int(row.get(key, 0)) for row in summaries)
    (args.output_root / "summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in totals.items() if key != "summaries"}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

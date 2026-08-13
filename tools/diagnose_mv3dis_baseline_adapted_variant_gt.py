#!/usr/bin/env python3
"""GT-only attribution for a frozen MV3DIS baseline-adapted variant.

The D2b proposal's best class-agnostic GT instance is held fixed while its
source and variant IoUs are compared.  Native Mask3D coverage and score are
reported only to explain system dilution.  This tool never changes proposals,
scores, actions, thresholds, or the frozen materialization rule.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_THRESHOLDS = (0.25, 0.50)


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


def _track_points(track, point_count):
    points = np.unique(
        np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64)
    )
    if np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"proposal {track['proposal_id']} has out-of-range points")
    return points


def _valid_gt_instances(gt_ids, valid_class_ids):
    rows = []
    for instance_id, count in zip(*np.unique(gt_ids, return_counts=True)):
        instance_id = int(instance_id)
        if instance_id > 0 and instance_id // 1000 in valid_class_ids:
            rows.append((instance_id, int(count)))
    return rows


def _iou_with_instance(points, gt_ids, instance_id, instance_size):
    intersection = int(np.count_nonzero(gt_ids[points] == instance_id))
    union = len(points) + int(instance_size) - intersection
    return float(intersection / max(1, union)), intersection


def best_gt_instance(points, gt_ids, gt_instances):
    best = None
    for instance_id, instance_size in gt_instances:
        iou, intersection = _iou_with_instance(
            points, gt_ids, instance_id, instance_size
        )
        candidate = (iou, intersection, -instance_id, instance_id, instance_size)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None or best[1] == 0:
        return None
    return {
        "instance_id": int(best[3]),
        "instance_size": int(best[4]),
        "iou": float(best[0]),
        "intersection": int(best[1]),
    }


def native_target_stats(
    native_masks, native_scores, gt_ids, instance_id, instance_size
):
    if native_masks.ndim != 2 or native_masks.shape[0] != len(gt_ids):
        raise ValueError("native mask shape differs from GT")
    if native_masks.shape[1] != len(native_scores):
        raise ValueError("native score count differs from masks")
    counts = np.count_nonzero(native_masks, axis=0)
    intersections = np.count_nonzero(native_masks[gt_ids == instance_id], axis=0)
    unions = counts + int(instance_size) - intersections
    ious = intersections / np.maximum(1, unions)
    result = {
        "best_native_iou": float(np.max(ious)) if len(ious) else 0.0,
    }
    for threshold in EVAL_THRESHOLDS:
        eligible = ious >= threshold
        key = f"iou{int(threshold * 100)}"
        result[f"native_{key}_candidate_count"] = int(np.count_nonzero(eligible))
        result[f"max_native_score_at_{key}"] = (
            float(np.max(native_scores[eligible])) if np.any(eligible) else None
        )
    return result


def _threshold_state(source_iou, variant_iou, threshold):
    source_pass = source_iou >= threshold
    variant_pass = variant_iou >= threshold
    if not source_pass and variant_pass:
        return "upcross"
    if source_pass and not variant_pass:
        return "downcross"
    if source_pass:
        return "both_pass"
    return "both_fail"


def diagnose_scene(
    scene_name,
    source_tracks,
    variant_tracks,
    proposal_ledger,
    gt_ids,
    valid_gt_classes,
    native_masks,
    native_scores,
):
    source_by_id = {int(row["proposal_id"]): row for row in source_tracks}
    variant_by_id = {int(row["proposal_id"]): row for row in variant_tracks}
    if list(source_by_id) != list(variant_by_id):
        raise ValueError(f"{scene_name} proposal IDs or order differ")
    changed_ids = {
        int(row["proposal_id"])
        for row in proposal_ledger
        if row["geometry_changed"]
    }
    actual_changed = {
        proposal_id
        for proposal_id in source_by_id
        if source_by_id[proposal_id]["superpoint_ids"]
        != variant_by_id[proposal_id]["superpoint_ids"]
    }
    if changed_ids != actual_changed:
        raise ValueError(f"{scene_name} proposal ledger changed IDs differ")
    gt_instances = _valid_gt_instances(gt_ids, valid_gt_classes)
    native_cache = {}
    rows = []
    for proposal_id in sorted(changed_ids):
        source = source_by_id[proposal_id]
        variant = variant_by_id[proposal_id]
        score = float(source.get("mean_node_quality", 0.0))
        if float(variant.get("mean_node_quality", 0.0)) != score:
            raise ValueError(f"{scene_name}/{proposal_id} score changed")
        source_points = _track_points(source, len(gt_ids))
        variant_points = _track_points(variant, len(gt_ids))
        fixed = best_gt_instance(source_points, gt_ids, gt_instances)
        row = {
            "scene_name": scene_name,
            "proposal_id": proposal_id,
            "track_score": score,
            "source_point_count": len(source_points),
            "variant_point_count": len(variant_points),
            "added_point_count": int(len(np.setdiff1d(variant_points, source_points))),
            "removed_point_count": int(len(np.setdiff1d(source_points, variant_points))),
            "fixed_gt_available": fixed is not None,
            "ground_truth_usage": "post-hoc attribution only",
        }
        if fixed is None:
            rows.append(row)
            continue
        instance_id = int(fixed["instance_id"])
        instance_size = int(fixed["instance_size"])
        source_iou = float(fixed["iou"])
        variant_iou, variant_intersection = _iou_with_instance(
            variant_points, gt_ids, instance_id, instance_size
        )
        variant_best = best_gt_instance(variant_points, gt_ids, gt_instances)
        if instance_id not in native_cache:
            native_cache[instance_id] = native_target_stats(
                native_masks,
                native_scores,
                gt_ids,
                instance_id,
                instance_size,
            )
        native = native_cache[instance_id]
        row.update({
            "fixed_gt_instance_id": instance_id,
            "fixed_gt_point_count": instance_size,
            "source_fixed_gt_intersection": int(fixed["intersection"]),
            "variant_fixed_gt_intersection": int(variant_intersection),
            "source_fixed_gt_iou": source_iou,
            "variant_fixed_gt_iou": variant_iou,
            "fixed_gt_iou_delta": variant_iou - source_iou,
            "variant_best_gt_instance_id": (
                int(variant_best["instance_id"]) if variant_best else None
            ),
            "best_gt_target_changed": bool(
                variant_best and int(variant_best["instance_id"]) != instance_id
            ),
            **native,
        })
        for threshold in EVAL_THRESHOLDS:
            key = f"iou{int(threshold * 100)}"
            row[f"fixed_gt_{key}_state"] = _threshold_state(
                source_iou, variant_iou, threshold
            )
            max_native_score = native[f"max_native_score_at_{key}"]
            row[f"native_already_covers_fixed_gt_at_{key}"] = bool(
                native[f"native_{key}_candidate_count"] > 0
            )
            row[f"native_higher_score_cover_at_{key}"] = bool(
                max_native_score is not None and max_native_score >= score
            )
        rows.append(row)
    return rows


def summarize(rows):
    valid = [row for row in rows if row["fixed_gt_available"]]
    deltas = [float(row["fixed_gt_iou_delta"]) for row in valid]
    summary = {
        "changed_proposal_count": len(rows),
        "fixed_gt_available_count": len(valid),
        "no_fixed_gt_count": len(rows) - len(valid),
        "fixed_gt_iou_improved_count": sum(delta > 0 for delta in deltas),
        "fixed_gt_iou_equal_count": sum(delta == 0 for delta in deltas),
        "fixed_gt_iou_declined_count": sum(delta < 0 for delta in deltas),
        "mean_fixed_gt_iou_delta": float(np.mean(deltas)) if deltas else None,
        "best_gt_target_changed_count": sum(
            row["best_gt_target_changed"] for row in valid
        ),
    }
    for threshold in EVAL_THRESHOLDS:
        key = f"iou{int(threshold * 100)}"
        states = Counter(row[f"fixed_gt_{key}_state"] for row in valid)
        for state in ("upcross", "downcross", "both_pass", "both_fail"):
            summary[f"fixed_gt_{key}_{state}_count"] = states[state]
        summary[f"native_already_covers_fixed_gt_at_{key}_count"] = sum(
            row[f"native_already_covers_fixed_gt_at_{key}"] for row in valid
        )
        summary[f"native_higher_score_cover_at_{key}_count"] = sum(
            row[f"native_higher_score_cover_at_{key}"] for row in valid
        )
        summary[f"downcross_with_native_cover_at_{key}_count"] = sum(
            row[f"fixed_gt_{key}_state"] == "downcross"
            and row[f"native_already_covers_fixed_gt_at_{key}"]
            for row in valid
        )
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--source-track-root", type=Path, required=True)
    parser.add_argument("--variant-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in (
        "scene_list", "source_track_root", "variant_track_root",
        "native_prediction_cache", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    manifest = json.loads(
        (args.native_prediction_cache / "native_cache_no_gt_manifest.json").read_text()
    )
    if manifest.get("candidate_inputs", {}).get("mode") != "mask3d_yoloworld_only":
        raise ValueError("native cache is not mask3d_yoloworld_only")

    from evaluate.scannet200 import eval_semantic_instance as instance_eval

    valid_gt_classes = frozenset(
        int(value) for value in instance_eval.PRED_ID_TO_ID.values() if int(value) >= 0
    )
    scenes = _read_scenes(args.scene_list)
    args.output_dir.mkdir(parents=True)
    all_rows = []
    for index, scene_name in enumerate(scenes, start=1):
        source_payload = json.loads(
            (args.source_track_root / scene_name / "automatic_tracks.json").read_text()
        )
        variant_payload = json.loads(
            (args.variant_track_root / scene_name / "automatic_tracks.json").read_text()
        )
        proposal_ledger = [
            json.loads(line)
            for line in (
                args.variant_track_root / scene_name / "proposal_geometry_ledger.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        gt_ids = instance_eval.util_3d.load_ids(
            args.gt_instance_dir / f"{scene_name}.txt"
        )
        native_masks = np.load(
            args.native_prediction_cache / f"{scene_name}_pred_masks.npy",
            mmap_mode="r",
        )
        native_scores = np.load(
            args.native_prediction_cache / f"{scene_name}_pred_scores.npy",
            mmap_mode="r",
        )
        rows = diagnose_scene(
            scene_name,
            source_payload.get("tracks", []),
            variant_payload.get("tracks", []),
            proposal_ledger,
            gt_ids,
            valid_gt_classes,
            native_masks,
            native_scores,
        )
        all_rows.extend(rows)
        print(f"[done] {index}/{len(scenes)} {scene_name}: changed {len(rows)}", flush=True)
    result = {
        "diagnostic_type": "GT-only frozen MV3DIS variant failure attribution",
        "decision_constraint": (
            "must not tune actions, thresholds, scores, or combinations from this report"
        ),
        "scene_count": len(scenes),
        **summarize(all_rows),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    _write_jsonl(args.output_dir / "changed_proposal_attribution.jsonl", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

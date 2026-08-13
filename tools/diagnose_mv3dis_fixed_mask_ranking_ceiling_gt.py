#!/usr/bin/env python3
"""GT-only threshold-specific ranking ceiling for fixed global-oracle masks."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_mv3dis_boundary_action_oracle_gt import _read_scenes, _resolve  # noqa: E402
from diagnose_mv3dis_fixed_mask_score_oracle_gt import _read_jsonl  # noqa: E402
from diagnose_mv3dis_global_feasible_action_oracle_gt import _track_prediction, build_scene  # noqa: E402


OVERLAPS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.25)


def maximum_bipartite_matching(edges, prediction_count):
    """Kuhn maximum matching: GT nodes -> prediction nodes."""
    matched_gt_for_prediction = np.full(prediction_count, -1, dtype=np.int64)

    def visit(gt_index, seen):
        for prediction_index in edges[gt_index]:
            if seen[prediction_index]:
                continue
            seen[prediction_index] = True
            owner = matched_gt_for_prediction[prediction_index]
            if owner < 0 or visit(int(owner), seen):
                matched_gt_for_prediction[prediction_index] = gt_index
                return True
        return False

    count = 0
    for gt_index in range(len(edges)):
        seen = np.zeros(prediction_count, dtype=bool)
        count += int(visit(gt_index, seen))
    return count


def scene_ranking_ceiling(masks, gt_ids, valid_class_ids, min_region_size=100):
    masks = np.asarray(masks, dtype=bool)
    mask_sizes = np.count_nonzero(masks, axis=0)
    gt_instances = [
        (int(instance_id), int(size))
        for instance_id, size in zip(*np.unique(gt_ids, return_counts=True))
        if int(instance_id) > 0
        and int(instance_id) // 1000 in valid_class_ids
        and int(size) >= min_region_size
    ]
    intersections = np.zeros((len(gt_instances), masks.shape[1]), dtype=np.int32)
    for index, (instance_id, _) in enumerate(gt_instances):
        intersections[index] = np.count_nonzero(masks[gt_ids == instance_id], axis=0)
    result = {"valid_gt_instance_count": len(gt_instances), "prediction_count": masks.shape[1]}
    for overlap in OVERLAPS:
        edges = []
        for index, (_, size) in enumerate(gt_instances):
            ious = intersections[index] / np.maximum(1, mask_sizes + size - intersections[index])
            edges.append(np.flatnonzero(ious > overlap).tolist())
        result[f"match_count_{overlap}"] = maximum_bipartite_matching(edges, masks.shape[1])
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--action-ledger", type=Path, required=True)
    parser.add_argument("--f2-track-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("must pass --allow-gt-diagnostics")
    for name in ("scene_list", "action_ledger", "f2_track_root", "processed_scene_root", "native_prediction_cache", "gt_instance_dir", "output"):
        setattr(args, name, _resolve(getattr(args, name)))
    from evaluate.scannet200 import eval_semantic_instance as instance_eval
    valid_classes = frozenset(int(value) for value in instance_eval.PRED_ID_TO_ID.values() if int(value) >= 0)
    scenes = _read_scenes(args.scene_list)
    rows_by_scene = {scene: [] for scene in scenes}
    for row in _read_jsonl(args.action_ledger):
        if row["scene_name"] in rows_by_scene:
            rows_by_scene[row["scene_name"]].append(row)
    totals = {"valid_gt_instance_count": 0, "prediction_count": 0}
    for overlap in OVERLAPS:
        totals[f"match_count_{overlap}"] = 0
    for index, scene_name in enumerate(scenes, start=1):
        tracks, _, geometry, points, _, _ = build_scene(scene_name, rows_by_scene[scene_name], args.f2_track_root, args.processed_scene_root)
        native = np.load(args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r")
        track = _track_prediction(tracks, geometry, points, native.shape[0])
        masks = np.concatenate([native, track["pred_masks"]], axis=1)
        gt_ids = instance_eval.util_3d.load_ids(args.gt_instance_dir / f"{scene_name}.txt")
        values = scene_ranking_ceiling(masks, gt_ids, valid_classes)
        for key, value in values.items():
            totals[key] += int(value)
        print(f"[ranking ceiling] {index}/{len(scenes)} {scene_name}", flush=True)
    metrics = {overlap: totals[f"match_count_{overlap}"] / max(1, totals["valid_gt_instance_count"]) for overlap in OVERLAPS}
    result = {
        "diagnostic_type": "GT-only fixed-mask threshold-specific ranking ceiling",
        "decision_constraint": "GT matching only; no score, mask, candidate, class, or selector is written.",
        "definition": "for each IoU threshold, maximum one-to-one GT/prediction matching then rank all matched TPs before all FPs; this is a threshold-specific AP ceiling, not one realizable shared score.",
        "geometry_source": "global-feasible frozen boundary oracle reconstructed in memory",
        "scene_count": len(scenes),
        "totals": totals,
        "ceiling": {
            "ap": float(np.mean([value for overlap, value in metrics.items() if overlap != 0.25])),
            "ap50": float(metrics[0.5]),
            "ap25": float(metrics[0.25]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

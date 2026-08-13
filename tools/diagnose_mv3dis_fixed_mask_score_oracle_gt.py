#!/usr/bin/env python3
"""GT-only ideal ranking oracle with global-feasible boundary geometry fixed.

The frozen F2/native masks are never written.  The global-feasible geometry
oracle is reconstructed in memory from its frozen action ledger, then every
native and track mask receives its best valid GT IoU as an ideal offline score.
This measures ranking headroom only; it is not an inference score.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_gvc_class_agnostic_ap import (  # noqa: E402
    UNIFIED_PREDICTED_CLASS,
    _class_agnostic_gt_ids,
    _configure_scannet200_instance_eval,
    _merge_scan_matches,
    instance_eval,
)
from diagnose_mv3dis_boundary_action_oracle_gt import _read_scenes, _resolve, _write_jsonl  # noqa: E402
from diagnose_mv3dis_global_feasible_action_oracle_gt import (  # noqa: E402
    _scene_ap_records,
    _track_prediction,
    build_scene,
)


DECISION_CONSTRAINT = (
    "GT-derived scores are an offline fixed-mask ranking upper bound only and "
    "must not be exported to inference, thresholds, classes, or a selector."
)


def _read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ideal_class_agnostic_scores(masks, gt_ids, valid_gt_class_ids, min_region_size=100):
    """Return the best valid-GT IoU for each fixed prediction mask."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or masks.shape[0] != len(gt_ids):
        raise ValueError("mask shape differs from GT")
    scores = np.zeros(masks.shape[1], dtype=np.float32)
    mask_sizes = np.count_nonzero(masks, axis=0)
    for instance_id, instance_size in zip(*np.unique(gt_ids, return_counts=True)):
        instance_id = int(instance_id)
        instance_size = int(instance_size)
        if (
            instance_id <= 0
            or instance_id // 1000 not in valid_gt_class_ids
            or instance_size < min_region_size
        ):
            continue
        intersections = np.count_nonzero(masks[gt_ids == instance_id], axis=0)
        ious = intersections / np.maximum(1, mask_sizes + instance_size - intersections)
        scores = np.maximum(scores, ious.astype(np.float32))
    return scores


def _score_oracle_prediction(
    scene_name,
    rows,
    track_root,
    processed_scene_root,
    cache_root,
    gt_ids,
    valid_gt_class_ids,
):
    tracks, _, oracle_geometry, points_by_superpoint, _, scene_summary = build_scene(
        scene_name, rows, track_root, processed_scene_root
    )
    prefix = cache_root / f"{scene_name}_pred_"
    native_masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
    track_prediction = _track_prediction(
        tracks, oracle_geometry, points_by_superpoint, native_masks.shape[0]
    )
    if len(gt_ids) != native_masks.shape[0]:
        raise ValueError(f"{scene_name} GT point count differs")
    native_scores = ideal_class_agnostic_scores(native_masks, gt_ids, valid_gt_class_ids)
    track_scores = ideal_class_agnostic_scores(
        track_prediction["pred_masks"], gt_ids, valid_gt_class_ids
    )
    prediction = {
        "pred_masks": np.concatenate([native_masks, track_prediction["pred_masks"]], axis=1),
        "pred_scores": np.concatenate([native_scores, track_scores]),
        "pred_classes": np.full(
            native_masks.shape[1] + track_prediction["pred_masks"].shape[1],
            UNIFIED_PREDICTED_CLASS,
            dtype=np.int64,
        ),
    }
    score_summary = {
        "scene_name": scene_name,
        "native_prediction_count": int(native_masks.shape[1]),
        "track_prediction_count": int(track_prediction["pred_masks"].shape[1]),
        "native_positive_ideal_score_count": int(np.count_nonzero(native_scores > 0)),
        "track_positive_ideal_score_count": int(np.count_nonzero(track_scores > 0)),
        **scene_summary,
    }
    return prediction, score_summary


def evaluate_chunk(
    scenes,
    rows_by_scene,
    track_root,
    processed_scene_root,
    cache_root,
    gt_dir,
    valid_gt_class_ids,
):
    _configure_scannet200_instance_eval()
    records = {
        float(overlap): {"true": [], "score": [], "fn": 0, "has_gt": False, "has_pred": False}
        for overlap in instance_eval.opt["overlaps"]
    }
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    scene_summaries = []
    try:
        for index, scene_name in enumerate(scenes, start=1):
            raw_gt_ids = original_load_ids(gt_dir / f"{scene_name}.txt")
            prediction, score_summary = _score_oracle_prediction(
                scene_name, rows_by_scene[scene_name], track_root,
                processed_scene_root, cache_root, raw_gt_ids, valid_gt_class_ids,
            )
            gt_file = str(gt_dir / f"{scene_name}.txt")
            gt, pred = instance_eval.assign_instances_for_scan(prediction, gt_file)
            for overlap, result in records.items():
                true, score, fn, has_gt, has_pred = _scene_ap_records(gt, pred, overlap)
                result["true"].append(true)
                result["score"].append(score)
                result["fn"] += fn
                result["has_gt"] = result["has_gt"] or has_gt
                result["has_pred"] = result["has_pred"] or has_pred
            scene_summaries.append(score_summary)
            del prediction, gt, pred, raw_gt_ids
            gc.collect()
            print(f"[score oracle] {index}/{len(scenes)} {scene_name}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
    return records, scene_summaries


def _write_records(path, records):
    payload = {}
    for overlap, result in records.items():
        key = str(overlap).replace(".", "_")
        payload[f"true_{key}"] = np.concatenate(result["true"]) if result["true"] else np.empty(0)
        payload[f"score_{key}"] = np.concatenate(result["score"]) if result["score"] else np.empty(0)
        payload[f"fn_{key}"] = np.asarray([result["fn"]], dtype=np.int64)
        payload[f"has_gt_{key}"] = np.asarray([result["has_gt"]], dtype=bool)
        payload[f"has_pred_{key}"] = np.asarray([result["has_pred"]], dtype=bool)
    np.savez_compressed(path, **payload)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--action-ledger", type=Path, required=True)
    parser.add_argument("--f2-track-root", type=Path, required=True)
    parser.add_argument("--processed-scene-root", type=Path, default=Path("data/scannet200"))
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
        "scene_list", "action_ledger", "f2_track_root", "processed_scene_root",
        "native_prediction_cache", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    from evaluate.scannet200 import eval_semantic_instance as raw_eval
    valid_gt_class_ids = frozenset(
        int(value) for value in raw_eval.PRED_ID_TO_ID.values() if int(value) >= 0
    )
    scenes = _read_scenes(args.scene_list)
    rows_by_scene = {scene: [] for scene in scenes}
    for row in _read_jsonl(args.action_ledger):
        if row["scene_name"] in rows_by_scene:
            rows_by_scene[row["scene_name"]].append(row)
    args.output_dir.mkdir(parents=True)
    records, scene_summaries = evaluate_chunk(
        scenes, rows_by_scene, args.f2_track_root, args.processed_scene_root,
        args.native_prediction_cache, args.gt_instance_dir, valid_gt_class_ids,
    )
    _write_records(args.output_dir / "fixed_mask_score_oracle_ap_records.npz", records)
    summary = {
        "diagnostic_type": "GT-only fixed-mask ideal score/ranking oracle",
        "decision_constraint": DECISION_CONSTRAINT,
        "geometry_source": "global-feasible frozen boundary oracle reconstructed in memory",
        "proposal_materialization_applied": False,
        "native_prediction_mutation_applied": False,
        "scene_count": len(scenes),
        "scene_summaries": scene_summaries,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

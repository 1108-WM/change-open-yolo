#!/usr/bin/env python3
"""GT-only：评估固定预测缓存的类别无关实例 AP。

这不是 ScanNet200 官方开放词汇主结果。脚本仅在当前进程内将有效 GT 实例及
预测统一映射为 ``chair`` 标签，再调用项目现有的实例评测器。它不重新推理、
不修改 mask/分数/候选。可只评一套缓存，也可继续比较既有 native/GVC 缓存。
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200 import eval_semantic_instance as instance_eval


UNIFIED_PREDICTED_CLASS = 0
UNIFIED_GT_CLASS = int(instance_eval.PRED_ID_TO_ID[UNIFIED_PREDICTED_CLASS])
VALID_GT_CLASSES = frozenset(
    int(class_id) for class_id in instance_eval.PRED_ID_TO_ID.values() if int(class_id) >= 0
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复场景")
    return scenes


def _class_agnostic_gt_ids(ids):
    """保留实例边界，移除语义标签；无效 GT 继续作为 evaluator 的 void。"""
    ids = np.asarray(ids, dtype=np.int64)
    mapped = np.zeros_like(ids)
    valid_instance_ids = [
        int(instance_id)
        for instance_id in np.unique(ids)
        if instance_id > 0 and int(instance_id) // 1000 in VALID_GT_CLASSES
    ]
    if len(valid_instance_ids) >= 1000:
        raise ValueError("单场景有效 GT 实例超过类别无关编码上限")
    for new_instance_index, instance_id in enumerate(valid_instance_ids, start=1):
        mapped[ids == instance_id] = UNIFIED_GT_CLASS * 1000 + new_instance_index
    return mapped


def _track_points(record, point_count):
    path = Path(record["points_path"])
    if not path.is_file():
        raise FileNotFoundError(f"轨迹点文件不存在：{path}")
    points = np.unique(np.asarray(np.load(path)["point_indices"], dtype=np.int64))
    return points[(points >= 0) & (points < point_count)]


def append_track_predictions(prediction, tracks, score_field="mean_node_quality"):
    """原样保留缓存预测，并在类别无关诊断中追加固定轨迹。"""
    masks = np.asarray(prediction["pred_masks"])
    if masks.ndim != 2:
        raise ValueError("预测 mask 必须为二维数组")
    track_masks = np.zeros((masks.shape[0], len(tracks)), dtype=bool)
    track_scores = np.zeros(len(tracks), dtype=np.float32)
    for index, track in enumerate(tracks):
        track_masks[_track_points(track, masks.shape[0]), index] = True
        track_scores[index] = max(0.0, float(track.get(score_field, 0.0)))
    return {
        "pred_masks": np.concatenate([masks, track_masks], axis=1),
        "pred_scores": np.concatenate([
            np.asarray(prediction["pred_scores"], dtype=np.float32), track_scores
        ]),
        "pred_classes": np.full(masks.shape[1] + len(tracks), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
    }


def _load_predictions(cache_root, scenes, track_root=None, score_field="mean_node_quality"):
    predictions = {}
    for scene_name in scenes:
        prefix = cache_root / f"{scene_name}_pred_"
        masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
        scores = np.load(str(prefix) + "scores.npy", mmap_mode="r")
        classes = np.load(str(prefix) + "classes.npy", mmap_mode="r")
        if masks.ndim != 2 or masks.shape[1] != len(scores) or len(scores) != len(classes):
            raise ValueError(f"{scene_name} 预测缓存维度不一致")
        prediction = {
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_classes": np.full(len(scores), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
        }
        if track_root is not None:
            payload = json.loads((track_root / scene_name / "automatic_tracks.json").read_text())
            prediction = append_track_predictions(prediction, payload.get("tracks", []), score_field)
        predictions[scene_name] = prediction
    return predictions


def _configure_scannet200_instance_eval():
    valid_ids = tuple(
        int(instance_eval.PRED_ID_TO_ID[index])
        for index in range(len(instance_eval.VALID_CLASS_IDS_200_INST))
    )
    all_labels = {
        int(class_id): label
        for class_id, label in zip(
            instance_eval.VALID_CLASS_IDS_200,
            instance_eval.CLASS_LABELS_200,
        )
    }
    labels = tuple(all_labels[class_id] for class_id in valid_ids)
    instance_eval.DATASET_NAME = "scannet200"
    instance_eval.VALID_CLASS_IDS = np.asarray(valid_ids, dtype=np.int64)
    instance_eval.CLASS_LABELS = labels
    instance_eval.ID_TO_LABEL = dict(zip(valid_ids, labels))
    instance_eval.LABEL_TO_ID = dict(zip(labels, valid_ids))
    instance_eval.HEAD_CATS_SCANNET_200 = set(instance_eval.HEAD_CATS_SCANNET_200)
    instance_eval.COMMON_CATS_SCANNET_200 = set(instance_eval.COMMON_CATS_SCANNET_200)
    instance_eval.TAIL_CATS_SCANNET_200 = set(instance_eval.TAIL_CATS_SCANNET_200)


def _merge_scan_matches(native_gt, native_pred, track_gt, track_pred):
    if set(native_gt) != set(track_gt) or set(native_pred) != set(track_pred):
        raise ValueError("native and track match label domains differ")
    gt_identity_keys = (
        "instance_id", "label_id", "vert_count", "med_dist", "dist_conf"
    )
    for label_name in native_gt:
        native_rows = native_gt[label_name]
        track_rows = track_gt[label_name]
        if len(native_rows) != len(track_rows):
            raise ValueError(f"GT instance count differs for {label_name}")
        for native_row, track_row in zip(native_rows, track_rows):
            if any(native_row[key] != track_row[key] for key in gt_identity_keys):
                raise ValueError(f"GT instance identity differs for {label_name}")
            native_row["matched_pred"].extend(track_row["matched_pred"])
        native_pred[label_name].extend(track_pred[label_name])
    return native_gt, native_pred


def _track_prediction(track_root, scene_name, point_count, score_field):
    payload = json.loads(
        (track_root / scene_name / "automatic_tracks.json").read_text()
    )
    tracks = payload.get("tracks", [])
    masks = np.zeros((point_count, len(tracks)), dtype=bool)
    scores = np.zeros(len(tracks), dtype=np.float32)
    for index, track in enumerate(tracks):
        masks[_track_points(track, point_count), index] = True
        scores[index] = max(0.0, float(track.get(score_field, 0.0)))
    return {
        "pred_masks": masks,
        "pred_scores": scores,
        "pred_classes": np.full(
            len(tracks), UNIFIED_PREDICTED_CLASS, dtype=np.int64
        ),
    }


def _evaluate_native_plus_tracks_streaming(
    name, cache_root, scenes, gt_dir, output_dir, track_root, score_field
):
    """Merge independent scan matches before the unchanged AP computation."""
    _configure_scannet200_instance_eval()
    matches = {}
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        print(f"evaluating {len(scenes)} scans with streaming native/track matches...")
        for index, scene_name in enumerate(scenes, start=1):
            prefix = cache_root / f"{scene_name}_pred_"
            native_masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
            native_scores = np.load(str(prefix) + "scores.npy", mmap_mode="r")
            native_classes = np.load(str(prefix) + "classes.npy", mmap_mode="r")
            if (
                native_masks.ndim != 2
                or native_masks.shape[1] != len(native_scores)
                or len(native_scores) != len(native_classes)
            ):
                raise ValueError(f"{scene_name} 预测缓存维度不一致")
            native_prediction = {
                "pred_masks": native_masks,
                "pred_scores": np.asarray(native_scores, dtype=np.float32),
                "pred_classes": np.full(
                    len(native_scores), UNIFIED_PREDICTED_CLASS, dtype=np.int64
                ),
            }
            track_prediction = _track_prediction(
                track_root, scene_name, native_masks.shape[0], score_field
            )
            gt_file = str(gt_dir / f"{scene_name}.txt")
            native_gt, native_pred = instance_eval.assign_instances_for_scan(
                native_prediction, gt_file
            )
            track_gt, track_pred = instance_eval.assign_instances_for_scan(
                track_prediction, gt_file
            )
            merged_gt, merged_pred = _merge_scan_matches(
                native_gt, native_pred, track_gt, track_pred
            )
            matches[os.path.abspath(gt_file)] = {
                "gt": merged_gt,
                "pred": merged_pred,
            }
            del native_prediction, track_prediction, native_masks
            print(f"[stream] {index}/{len(scenes)} {scene_name}", flush=True)
    finally:
        instance_eval.util_3d.load_ids = original_load_ids

    ap_scores, _, _, _, _, _ = instance_eval.evaluate_matches(matches)
    averages = instance_eval.compute_averages(ap_scores)
    instance_eval.write_result_file(averages, str(output_dir / f"{name}.csv"))
    chair = averages["classes"]["chair"]
    return {
        "ap": float(chair["ap"]),
        "ap50": float(chair["ap50%"]),
        "ap25": float(chair["ap25%"]),
    }


def _evaluate_variant(name, cache_root, scenes, gt_dir, output_dir, track_root=None, score_field="mean_node_quality"):
    predictions = (
        LazyTrackPredictions(cache_root, scenes, track_root, score_field)
        if track_root is not None
        else _load_predictions(cache_root, scenes)
    )
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        averages, _, _, _ = instance_eval.evaluate(
            predictions,
            str(gt_dir),
            str(output_dir / f"{name}.csv"),
            dataset="scannet200",
        )
    finally:
        instance_eval.util_3d.load_ids = original_load_ids
        del predictions
        gc.collect()
    chair = averages["classes"]["chair"]
    return {
        "ap": float(chair["ap"]),
        "ap50": float(chair["ap50%"]),
        "ap25": float(chair["ap25%"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--gvc-prediction-cache", type=Path)
    parser.add_argument("--track-root", type=Path)
    parser.add_argument("--track-score-field", default="mean_node_quality")
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；本工具只能用于事后诊断。")
    for name in ("scene_list", "native_prediction_cache", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.gvc_prediction_cache is not None:
        args.gvc_prediction_cache = _resolve(args.gvc_prediction_cache)
    if args.track_root is not None:
        args.track_root = _resolve(args.track_root)
    if args.gvc_prediction_cache is not None and args.track_root is not None:
        raise SystemExit("--gvc-prediction-cache 与 --track-root 只能选择一个比较来源")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _read_scenes(args.scene_list)
    roots = [args.native_prediction_cache, args.gt_instance_dir]
    if args.gvc_prediction_cache is not None:
        roots.append(args.gvc_prediction_cache)
    if args.track_root is not None:
        roots.append(args.track_root)
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"缺少输入目录：{root}")

    combined = None
    if args.track_root is not None:
        combined = _evaluate_native_plus_tracks_streaming(
            "native_plus_tracks_class_agnostic",
            args.native_prediction_cache,
            scenes,
            args.gt_instance_dir,
            args.output_dir,
            args.track_root,
            args.track_score_field,
        )
    native = _evaluate_variant(
        "native_class_agnostic",
        args.native_prediction_cache,
        scenes,
        args.gt_instance_dir,
        args.output_dir,
    )
    payload = {
        "diagnostic_type": "GT-only class-agnostic instance AP；不是 ScanNet200 官方开放词汇主结果。",
        "decision_constraint": "只评估固定缓存；不得让 GT 回流到分数、阈值、类别、候选或融合。",
        "scene_count": len(scenes),
        "unified_predicted_class_index": UNIFIED_PREDICTED_CLASS,
        "unified_gt_class_id": UNIFIED_GT_CLASS,
        "native": native,
    }
    if args.gvc_prediction_cache is not None:
        gvc = _evaluate_variant(
            "native_plus_gvc_class_agnostic",
            args.gvc_prediction_cache,
            scenes,
            args.gt_instance_dir,
            args.output_dir,
        )
        payload["native_plus_gvc_append_only"] = gvc
        payload["delta_gvc_minus_native"] = {key: gvc[key] - native[key] for key in native}
    elif args.track_root is not None:
        payload["track_score_field"] = args.track_score_field
        payload["native_plus_tracks"] = combined
        payload["delta_tracks_minus_native"] = {
            key: combined[key] - native[key] for key in native
        }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

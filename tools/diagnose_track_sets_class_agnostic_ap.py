#!/usr/bin/env python3
"""GT-only：比较两套固定类别无关轨迹的实例 AP。

两套轨迹的 mask 全部映射到同一个合法 ScanNet200 类别，且只用轨迹自身的固定
``mean_node_quality`` 排序。该工具专门隔离跨帧关联的几何影响：不读取、也不使用
YOLO-World、GVC、语义类别或 native 候选；绝不能用于回调轨迹参数或正式 AP。
"""

import argparse
import gc
import json
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
    """保留实例边界，移除 ScanNet200 语义类；void 保持 void。"""
    ids = np.asarray(ids, dtype=np.int64)
    mapped = np.zeros_like(ids)
    valid_instance_ids = [
        int(instance_id) for instance_id in np.unique(ids)
        if instance_id > 0 and int(instance_id) // 1000 in VALID_GT_CLASSES
    ]
    if len(valid_instance_ids) >= 1000:
        raise ValueError("单场景有效 GT 实例超过类别无关编码上限")
    for new_index, instance_id in enumerate(valid_instance_ids, start=1):
        mapped[ids == instance_id] = UNIFIED_GT_CLASS * 1000 + new_index
    return mapped


def _track_points(record, point_count):
    path = Path(record["points_path"])
    if not path.is_file():
        raise FileNotFoundError(f"轨迹点文件不存在：{path}")
    points = np.unique(np.asarray(np.load(path)["point_indices"], dtype=np.int64))
    return points[(points >= 0) & (points < point_count)]


def build_track_predictions(track_root, scenes, gt_instance_dir, score_field="mean_node_quality"):
    """仅把固定轨迹转换成统一类别预测，不改变任何轨迹输入。"""
    predictions = {}
    for scene_name in scenes:
        gt_ids = instance_eval.util_3d.load_ids(gt_instance_dir / f"{scene_name}.txt")
        payload = json.loads((track_root / scene_name / "automatic_tracks.json").read_text())
        tracks = payload.get("tracks", [])
        masks = np.zeros((len(gt_ids), len(tracks)), dtype=bool)
        scores = np.zeros(len(tracks), dtype=np.float32)
        for index, track in enumerate(tracks):
            masks[_track_points(track, len(gt_ids)), index] = True
            scores[index] = max(0.0, float(track.get(score_field, 0.0)))
        predictions[scene_name] = {
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_classes": np.full(len(tracks), UNIFIED_PREDICTED_CLASS, dtype=np.int64),
        }
    return predictions


def _evaluate(name, track_root, scenes, gt_instance_dir, output_dir, score_field):
    predictions = build_track_predictions(track_root, scenes, gt_instance_dir, score_field)
    original_load_ids = instance_eval.util_3d.load_ids

    def load_class_agnostic_ids(filename):
        return _class_agnostic_gt_ids(original_load_ids(filename))

    instance_eval.util_3d.load_ids = load_class_agnostic_ids
    try:
        averages, _, _, _ = instance_eval.evaluate(
            predictions, str(gt_instance_dir), str(output_dir / f"{name}.csv"), dataset="scannet200"
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
    parser.add_argument("--track-root-a", type=Path, required=True)
    parser.add_argument("--track-root-b", type=Path, required=True)
    parser.add_argument("--name-a", default="baseline_tracks")
    parser.add_argument("--name-b", default="comparison_tracks")
    parser.add_argument("--score-field", default="mean_node_quality")
    parser.add_argument("--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；该工具只能用于事后诊断。")
    for name in ("scene_list", "track_root_a", "track_root_b", "gt_instance_dir", "output_dir"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    for root in (args.track_root_a, args.track_root_b, args.gt_instance_dir):
        if not root.is_dir():
            raise ValueError(f"缺少输入目录：{root}")
    args.output_dir.mkdir(parents=True)
    first = _evaluate(args.name_a, args.track_root_a, scenes, args.gt_instance_dir, args.output_dir, args.score_field)
    second = _evaluate(args.name_b, args.track_root_b, scenes, args.gt_instance_dir, args.output_dir, args.score_field)
    payload = {
        "diagnostic_type": "GT-only class-agnostic track AP；不是 ScanNet200 官方开放词汇主结果。",
        "decision_constraint": "固定既有轨迹和 score_field；不得用于回调关联阈值、候选、语义、融合或正式 AP。",
        "scene_count": len(scenes),
        "score_field": args.score_field,
        args.name_a: first,
        args.name_b: second,
        f"delta_{args.name_b}_minus_{args.name_a}": {key: second[key] - first[key] for key in first},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

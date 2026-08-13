#!/usr/bin/env python3
"""为帧级 YOLO-World + SAM 观测标记相对当前三维预测的残差。

输入是 ``export_dense_frame_instance_observations.py`` 导出的二值 SAM mask 与
其可见三维点。脚本只比较这些点与 native 最终预测 mask 的几何覆盖，分别
记录“任意类别候选”和“同类别候选”的覆盖情况，并保存未被覆盖的三维点。

它不读取 GT、不生成最终候选、不执行阈值筛选或融合；输出仅是后续跨帧关联、
三维提升和 GT-only 特征账本所需的推理期证据缓存。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _load_prediction(root, scene_name):
    prefix = root / scene_name
    masks = np.load(f"{prefix}_pred_masks.npy", mmap_mode="r")
    classes = np.load(f"{prefix}_pred_classes.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的预测 mask 维度异常：{masks.shape}")
    if masks.shape[1] != len(classes) and masks.shape[0] == len(classes):
        masks = masks.T
    if masks.shape[1] != len(classes):
        raise ValueError(f"{scene_name} 的预测 mask 与类别数量不一致")
    return np.asarray(masks, dtype=bool), np.asarray(classes, dtype=np.int64)


def _load_indices(path, point_count):
    payload = np.load(path)
    indices = np.asarray(payload["point_indices"], dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < point_count)]
    return np.unique(indices)


def _overlap_metrics(masks, classes, point_indices, class_id):
    """返回观测点相对任意/同类最终候选的覆盖统计及未覆盖点。"""
    point_indices = np.asarray(point_indices, dtype=np.int64)
    point_count = int(len(point_indices))
    candidate_count = int(masks.shape[1])
    result = {
        "point_count": point_count,
        "candidate_count": candidate_count,
        "same_class_candidate_count": int(np.sum(classes == int(class_id))),
        "best_any_candidate_id": -1,
        "best_any_iou": 0.0,
        "best_any_seed_coverage": 0.0,
        "best_any_candidate_coverage": 0.0,
        "best_same_class_candidate_id": -1,
        "best_same_class_iou": 0.0,
        "best_same_class_seed_coverage": 0.0,
        "best_same_class_candidate_coverage": 0.0,
        "any_candidate_seed_coverage": 0.0,
        "same_class_candidate_seed_coverage": 0.0,
    }
    if point_count == 0 or candidate_count == 0:
        return result, point_indices.copy(), point_indices.copy()

    candidate_sizes = masks.sum(axis=0, dtype=np.int64)
    intersections = masks[point_indices].sum(axis=0, dtype=np.int64)
    unions = candidate_sizes + point_count - intersections
    ious = intersections / np.maximum(1, unions)
    seed_coverage = intersections / max(1, point_count)
    candidate_coverage = intersections / np.maximum(1, candidate_sizes)

    best_any = int(np.argmax(ious))
    result.update(
        {
            "best_any_candidate_id": best_any,
            "best_any_iou": float(ious[best_any]),
            "best_any_seed_coverage": float(seed_coverage[best_any]),
            "best_any_candidate_coverage": float(candidate_coverage[best_any]),
        }
    )
    same_class = np.flatnonzero(classes == int(class_id))
    if len(same_class):
        best_same = int(same_class[np.argmax(ious[same_class])])
        result.update(
            {
                "best_same_class_candidate_id": best_same,
                "best_same_class_iou": float(ious[best_same]),
                "best_same_class_seed_coverage": float(seed_coverage[best_same]),
                "best_same_class_candidate_coverage": float(candidate_coverage[best_same]),
            }
        )

    point_rows = masks[point_indices]
    covered_by_any = point_rows.any(axis=1)
    covered_by_same = point_rows[:, same_class].any(axis=1) if len(same_class) else np.zeros(point_count, dtype=bool)
    result["any_candidate_seed_coverage"] = float(covered_by_any.mean())
    result["same_class_candidate_seed_coverage"] = float(covered_by_same.mean())
    return result, point_indices[~covered_by_any], point_indices[~covered_by_same]


def _load_observations(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_indices(path, indices):
    np.savez_compressed(path, point_indices=np.asarray(indices, dtype=np.int64))


def annotate_scene(scene_name, args):
    source_scene = args.observation_root / scene_name
    observation_path = source_scene / "observations.jsonl"
    if not observation_path.is_file():
        raise FileNotFoundError(f"缺少帧级 SAM 观测：{observation_path}")
    masks, classes = _load_prediction(args.prediction_cache_dir, scene_name)
    output_scene = args.output_root / scene_name
    output_scene.mkdir(parents=True, exist_ok=False)
    points_dir = output_scene / "residual_points"
    points_dir.mkdir()

    rows = []
    totals = {
        "observation_count": 0,
        "source_point_count": 0,
        "residual_after_any_candidate_point_count": 0,
        "residual_after_same_class_candidate_point_count": 0,
    }
    for observation in _load_observations(observation_path):
        source_points = _load_indices(observation["point_indices_path"], masks.shape[0])
        metrics, residual_any, residual_same = _overlap_metrics(
            masks,
            classes,
            source_points,
            int(observation["class_id"]),
        )
        observation_id = int(observation["observation_id"])
        any_path = None
        same_path = None
        if len(residual_any) >= args.min_saved_residual_points:
            any_path = points_dir / f"obs{observation_id:06d}_after_any.npz"
            _write_indices(any_path, residual_any)
        if len(residual_same) >= args.min_saved_residual_points:
            same_path = points_dir / f"obs{observation_id:06d}_after_same_class.npz"
            _write_indices(same_path, residual_same)
        row = {
            "scene_name": scene_name,
            "observation_id": observation_id,
            "frame_id": str(observation["frame_id"]),
            "frame_index": int(observation["frame_index"]),
            "class_id": int(observation["class_id"]),
            "class_name": str(observation["class_name"]),
            "detection_score": float(observation["score"]),
            "sam_score": float(observation["sam_score"]),
            "source_mask_path": str(observation["mask_path"]),
            "source_point_indices_path": str(observation["point_indices_path"]),
            "residual_after_any_points_path": str(any_path) if any_path else None,
            "residual_after_same_class_points_path": str(same_path) if same_path else None,
            "residual_after_any_point_count": int(len(residual_any)),
            "residual_after_same_class_point_count": int(len(residual_same)),
            **metrics,
        }
        rows.append(row)
        totals["observation_count"] += 1
        totals["source_point_count"] += int(len(source_points))
        totals["residual_after_any_candidate_point_count"] += int(len(residual_any))
        totals["residual_after_same_class_candidate_point_count"] += int(len(residual_same))

    output_path = output_scene / "residual_observations.jsonl"
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "scene_name": scene_name,
        "source_observations": str(observation_path),
        "prediction_cache": str(args.prediction_cache_dir / scene_name),
        "min_saved_residual_points": int(args.min_saved_residual_points),
        **totals,
    }
    summary["residual_after_any_candidate_ratio"] = float(
        totals["residual_after_any_candidate_point_count"] / max(1, totals["source_point_count"])
    )
    summary["residual_after_same_class_candidate_ratio"] = float(
        totals["residual_after_same_class_candidate_point_count"] / max(1, totals["source_point_count"])
    )
    (output_scene / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", type=Path, required=True)
    parser.add_argument("--observation_root", type=Path, required=True)
    parser.add_argument("--prediction_cache_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_scenes", type=int)
    parser.add_argument("--min_saved_residual_points", type=int, default=20)
    args = parser.parse_args()
    if args.min_saved_residual_points < 1:
        raise SystemExit("--min_saved_residual_points 必须不小于 1。")
    for name in ("scene_list", "observation_root", "prediction_cache_dir", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")

    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        summary = annotate_scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['observation_count']} 条观测",
            flush=True,
        )
    total_source = sum(item["source_point_count"] for item in summaries)
    total_any = sum(item["residual_after_any_candidate_point_count"] for item in summaries)
    total_same = sum(item["residual_after_same_class_candidate_point_count"] for item in summaries)
    payload = {
        "gt_usage": "不读取 GT；输出只用于后续无 GT 残差关联、候选形成和离线特征诊断。",
        "decision_state": "未执行候选筛选、跨帧合并、融合或最终评分。",
        "scene_count": len(summaries),
        "min_saved_residual_points": int(args.min_saved_residual_points),
        "observation_count": sum(item["observation_count"] for item in summaries),
        "source_point_count": total_source,
        "residual_after_any_candidate_point_count": total_any,
        "residual_after_same_class_candidate_point_count": total_same,
        "residual_after_any_candidate_ratio": float(total_any / max(1, total_source)),
        "residual_after_same_class_candidate_ratio": float(total_same / max(1, total_source)),
        "scenes": summaries,
    }
    (args.output_root / "residual_observations_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

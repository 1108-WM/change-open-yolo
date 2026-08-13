#!/usr/bin/env python3
"""用核心点提示增量对互为最佳的 Mask3D 候选做只增不减局部补全。

匹配只使用 v0/prompt 共同核心与纯 Mask3D mask 的点级 IoU。仅处理
``core_prompt_variant_changed=true`` 的轨迹；互为最佳且正重叠时，把该轨迹
声明的新增 superpoint 并入对应 Mask3D mask。候选类别、分数、数量和全部未配对
mask 保持不变。本工具不读取 GT，不追加或修改共识 v0 候选。
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_details_core_prompt_native_relation_ledger import (
    _load_track_points,
    _load_tracks,
    _native_cache_contract,
    _read_scenes,
    _validate_track_contract,
    native_overlap_vectors,
)
from refine_details_automatic_tracks_consensus import _superpoint_points


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mutual_best_positive_pairs(iou_matrix, track_ids):
    """返回确定性的正 IoU 互为最佳 (track_id, candidate_id, iou) 对。"""
    iou_matrix = np.asarray(iou_matrix, dtype=np.float64)
    track_ids = [int(item) for item in track_ids]
    if iou_matrix.ndim != 2 or iou_matrix.shape[0] != len(track_ids):
        raise ValueError("IoU 矩阵与 track_ids 维度不一致")
    if not len(track_ids) or not iou_matrix.shape[1]:
        return []
    best_candidate_by_track = np.argmax(iou_matrix, axis=1)
    best_track_by_candidate = np.argmax(iou_matrix, axis=0)
    pairs = []
    for track_index, candidate_id in enumerate(best_candidate_by_track.tolist()):
        overlap = float(iou_matrix[track_index, candidate_id])
        if overlap <= 0.0 or int(best_track_by_candidate[candidate_id]) != track_index:
            continue
        pairs.append((track_ids[track_index], int(candidate_id), overlap))
    return pairs


def complete_native_masks(native_masks, pairs, added_points_by_track):
    """物化只增不减补全，并返回逐对实际增量统计。"""
    source = np.asarray(native_masks, dtype=bool)
    if source.ndim != 2:
        raise ValueError("native mask 必须为 [points, candidates] 二维数组")
    completed = source.copy()
    rows = []
    used_candidates = set()
    for track_id, candidate_id, core_iou in pairs:
        if candidate_id in used_candidates:
            raise ValueError(f"candidate {candidate_id} 被重复配对")
        if not 0 <= int(candidate_id) < completed.shape[1]:
            raise ValueError(f"candidate {candidate_id} 越界")
        points = np.unique(
            np.asarray(added_points_by_track[int(track_id)], dtype=np.int64)
        )
        if len(points) and (int(points[0]) < 0 or int(points[-1]) >= completed.shape[0]):
            raise ValueError(f"track {track_id} 的新增点越界")
        before = int(completed[:, candidate_id].sum())
        already_inside = int(completed[points, candidate_id].sum()) if len(points) else 0
        completed[points, candidate_id] = True
        after = int(completed[:, candidate_id].sum())
        rows.append({
            "track_id": int(track_id),
            "native_candidate_id": int(candidate_id),
            "common_core_native_iou": float(core_iou),
            "declared_added_point_count": int(len(points)),
            "already_inside_native_count": already_inside,
            "actual_added_point_count": after - before,
            "native_point_count_before": before,
            "native_point_count_after": after,
        })
        used_candidates.add(candidate_id)
    if np.any(source & ~completed):
        raise AssertionError("局部补全违反只增不减合同")
    return completed, rows


def _scene_inputs(scene_name, args):
    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint = _superpoint_points(superpoints)
    valid_superpoints = set(points_by_superpoint)
    base_tracks = _load_tracks(args.base_track_root, scene_name)
    prompt_tracks = _load_tracks(args.prompt_track_root, scene_name)
    _validate_track_contract(scene_name, base_tracks, prompt_tracks, valid_superpoints)

    prefix = args.native_prediction_cache / f"{scene_name}_pred_"
    masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
    scores = np.load(str(prefix) + "scores.npy", mmap_mode="r")
    classes = np.load(str(prefix) + "classes.npy", mmap_mode="r")
    if masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 native mask 维度异常：{masks.shape}")
    if masks.shape[0] != len(processed) and masks.shape[1] == len(processed):
        masks = masks.T
    if masks.shape != (len(processed), len(scores)) or len(scores) != len(classes):
        raise ValueError(f"{scene_name} 的 native 缓存维度不一致")
    return processed, points_by_superpoint, base_tracks, prompt_tracks, masks, scores, classes


def _build_scene(scene_name, args, staging_root):
    (
        processed,
        points_by_superpoint,
        base_tracks,
        prompt_tracks,
        native_masks,
        scores,
        classes,
    ) = _scene_inputs(scene_name, args)

    changed_track_ids = sorted(
        track_id
        for track_id, track in prompt_tracks.items()
        if bool(track.get("core_prompt_variant_changed", False))
    )
    iou_rows = []
    added_points_by_track = {}
    for track_id in changed_track_ids:
        base_points = _load_track_points(
            base_tracks[track_id], len(processed), scene_name, "v0"
        )
        prompt_points = _load_track_points(
            prompt_tracks[track_id], len(processed), scene_name, "prompt"
        )
        common_core = np.intersect1d(base_points, prompt_points, assume_unique=True)
        iou_rows.append(native_overlap_vectors(common_core, native_masks)["iou"])
        added_superpoints = [
            int(item)
            for item in prompt_tracks[track_id].get(
                "core_prompt_added_superpoint_ids", []
            )
        ]
        added_points_by_track[track_id] = np.concatenate(
            [points_by_superpoint[item] for item in added_superpoints]
        ).astype(np.int64, copy=False)

    iou_matrix = (
        np.stack(iou_rows, axis=0)
        if iou_rows
        else np.zeros((0, native_masks.shape[1]), dtype=np.float64)
    )
    pairs = mutual_best_positive_pairs(iou_matrix, changed_track_ids)
    completed_masks, pair_rows = complete_native_masks(
        native_masks, pairs, added_points_by_track
    )

    np.save(staging_root / f"{scene_name}_pred_masks.npy", completed_masks)
    shutil.copy2(
        args.native_prediction_cache / f"{scene_name}_pred_scores.npy",
        staging_root / f"{scene_name}_pred_scores.npy",
    )
    shutil.copy2(
        args.native_prediction_cache / f"{scene_name}_pred_classes.npy",
        staging_root / f"{scene_name}_pred_classes.npy",
    )
    if not np.array_equal(
        scores, np.load(staging_root / f"{scene_name}_pred_scores.npy")
    ) or not np.array_equal(
        classes, np.load(staging_root / f"{scene_name}_pred_classes.npy")
    ):
        raise AssertionError(f"{scene_name} 的类别或分数未原样保留")

    summary = {
        "scene_name": scene_name,
        "native_candidate_count": int(native_masks.shape[1]),
        "core_prompt_changed_track_count": len(changed_track_ids),
        "mutual_best_positive_pair_count": len(pair_rows),
        "effective_refined_candidate_count": sum(
            row["actual_added_point_count"] > 0 for row in pair_rows
        ),
        "declared_added_point_count": sum(
            row["declared_added_point_count"] for row in pair_rows
        ),
        "actual_added_point_count": sum(
            row["actual_added_point_count"] for row in pair_rows
        ),
        "pairs": pair_rows,
    }
    (staging_root / f"{scene_name}_paired_completion.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    del processed, native_masks, completed_masks
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--base-track-root", type=Path, required=True)
    parser.add_argument("--prompt-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument(
        "--expected-cache-mode",
        choices=("mask3d_yoloworld_only",),
        required=True,
    )
    parser.add_argument(
        "--processed-scene-root", type=Path, default=Path("data/scannet200")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args()
    for name in (
        "scene_list",
        "base_track_root",
        "prompt_track_root",
        "native_prediction_cache",
        "processed_scene_root",
        "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes 必须为正数")
    if args.output_dir.exists():
        raise SystemExit(f"输出目录已存在，为避免覆盖已拒绝执行：{args.output_dir}")

    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    cache_contract = _native_cache_contract(
        args.native_prediction_cache, args.expected_cache_mode
    )
    staging_root = args.output_dir.with_name(f".{args.output_dir.name}.writing")
    if staging_root.exists():
        raise SystemExit(f"临时输出目录已存在：{staging_root}")
    staging_root.mkdir(parents=True)
    summaries = []
    try:
        for index, scene_name in enumerate(scenes, start=1):
            summary = _build_scene(scene_name, args, staging_root)
            summaries.append(summary)
            print(
                f"[场景完成] {index}/{len(scenes)} {scene_name}: "
                f"配对 {summary['mutual_best_positive_pair_count']}，"
                f"实际补全 {summary['effective_refined_candidate_count']} 个候选",
                flush=True,
            )
        manifest = {
            "gt_usage": "none；匹配和补全不读取 GT、类别或分数。",
            "decision_state": (
                "只对 core-prompt 改变轨迹的共同核心与纯 Mask3D 做正 IoU "
                "互为最佳配对；原 Mask3D 仅并入声明的提示新增 superpoint。"
            ),
            "candidate_inputs": {"mode": "mask3d_yoloworld_paired_completion"},
            "source_native_cache_contract": cache_contract,
            "scene_count": len(summaries),
            "native_candidate_count": sum(
                item["native_candidate_count"] for item in summaries
            ),
            "core_prompt_changed_track_count": sum(
                item["core_prompt_changed_track_count"] for item in summaries
            ),
            "mutual_best_positive_pair_count": sum(
                item["mutual_best_positive_pair_count"] for item in summaries
            ),
            "effective_refined_candidate_count": sum(
                item["effective_refined_candidate_count"] for item in summaries
            ),
            "declared_added_point_count": sum(
                item["declared_added_point_count"] for item in summaries
            ),
            "actual_added_point_count": sum(
                item["actual_added_point_count"] for item in summaries
            ),
            "integrity": {
                "candidate_count_unchanged": True,
                "classes_unchanged": True,
                "scores_unchanged": True,
                "masks_only_grow": True,
                "unpaired_masks_unchanged": True,
                "consensus_v0_not_read_or_modified": True,
            },
            "scenes": summaries,
            "params": {key: value for key, value in vars(args).items()},
        }
        (staging_root / "paired_mask3d_completion_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            + "\n"
        )
        os.replace(staging_root, args.output_dir)
    except BaseException:
        # 保留 staging 便于排查，不发布不完整缓存。
        raise
    print(json.dumps({
        key: manifest[key]
        for key in (
            "scene_count",
            "core_prompt_changed_track_count",
            "mutual_best_positive_pair_count",
            "effective_refined_candidate_count",
            "actual_added_point_count",
            "integrity",
        )
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

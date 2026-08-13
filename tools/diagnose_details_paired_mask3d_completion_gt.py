#!/usr/bin/env python3
"""GT-only：归因固定成对 Mask3D 局部补全的对象错配与边界过扩张。

输入必须是已经物化并完成 AP 的固定原始/补全缓存。对每个实际改变候选，本工具
以原 Mask3D 的最佳 GT 实例为固定目标，比较共同核心目标、新增点纯度以及补全前后
IoU。它不生成候选、不修改 mask/类别/分数/阈值，也不运行或分解 AP。
"""

import argparse
import csv
import json
import sys
from collections import Counter
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
)
from diagnose_details_core_prompt_native_relation_gt import (
    _load_gt,
    best_gt_instance,
    target_metrics,
)
from diagnose_details_paired_completion_class_agnostic_ap import (
    _refined_cache_contract,
)


AP_THRESHOLDS = (0.25,) + tuple(np.arange(0.50, 0.91, 0.05).round(2))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stats(values):
    values = np.asarray(
        [item for item in values if item is not None], dtype=np.float64
    )
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def attribute_completion(original_points, refined_points, core_points, gt_ids, gt_sizes):
    """比较一个固定候选，类别只描述事后失败类型，不形成选择规则。"""
    original_points = np.unique(np.asarray(original_points, dtype=np.int64))
    refined_points = np.unique(np.asarray(refined_points, dtype=np.int64))
    core_points = np.unique(np.asarray(core_points, dtype=np.int64))
    if np.any(~np.isin(original_points, refined_points, assume_unique=True)):
        raise ValueError("补全候选删除了原 Mask3D 点")
    added_points = np.setdiff1d(refined_points, original_points, assume_unique=True)
    original_target_id, original_best_iou = best_gt_instance(
        original_points, gt_ids, gt_sizes
    )
    refined_target_id, refined_best_iou = best_gt_instance(
        refined_points, gt_ids, gt_sizes
    )
    core_target_id, core_best_iou = best_gt_instance(core_points, gt_ids, gt_sizes)
    target_size = int(gt_sizes.get(original_target_id, 0))
    original_target = target_metrics(
        original_points, gt_ids, original_target_id, target_size
    )
    refined_target = target_metrics(
        refined_points, gt_ids, original_target_id, target_size
    )
    core_target = target_metrics(core_points, gt_ids, original_target_id, target_size)
    added_target = target_metrics(
        added_points, gt_ids, original_target_id, target_size
    )
    delta = float(refined_target["iou"] - original_target["iou"])
    eps = 1e-12
    if original_target_id <= 0:
        category = "no_valid_original_target"
    elif core_target_id <= 0:
        category = "no_valid_core_target"
    elif core_target_id != original_target_id:
        category = "core_native_target_mismatch"
    elif delta < -eps:
        category = (
            "aligned_overexpansion_original_iou_ge50"
            if original_target["iou"] >= 0.50
            else "aligned_overexpansion_original_iou_lt50"
        )
    elif delta > eps:
        category = "aligned_improvement"
    else:
        category = "aligned_unchanged"
    return {
        "failure_category": category,
        "original_best_gt_instance_id": int(original_target_id),
        "original_best_gt_iou": float(original_best_iou),
        "refined_best_gt_instance_id": int(refined_target_id),
        "refined_best_gt_iou": float(refined_best_iou),
        "core_best_gt_instance_id": int(core_target_id),
        "core_best_gt_iou": float(core_best_iou),
        "core_native_target_aligned": (
            bool(core_target_id == original_target_id)
            if original_target_id > 0 and core_target_id > 0 else None
        ),
        "refined_best_target_switched": (
            bool(refined_target_id != original_target_id)
            if original_target_id > 0 and refined_target_id > 0 else None
        ),
        "original_target_iou": float(original_target["iou"]),
        "refined_target_iou": float(refined_target["iou"]),
        "refined_minus_original_target_iou": delta,
        "original_target_precision": float(original_target["precision"]),
        "refined_target_precision": float(refined_target["precision"]),
        "original_target_coverage": float(original_target["coverage"]),
        "refined_target_coverage": float(refined_target["coverage"]),
        "core_original_target_iou": float(core_target["iou"]),
        "added_point_count": int(len(added_points)),
        "added_original_target_precision": (
            float(added_target["precision"]) if len(added_points) else None
        ),
        "expansion_ratio": float(len(added_points) / max(1, len(original_points))),
    }


def threshold_transitions(original_iou, refined_iou, thresholds=AP_THRESHOLDS):
    output = {}
    for threshold in thresholds:
        name = f"iou{int(round(float(threshold) * 100)):02d}"
        output[name] = {
            "up": bool(original_iou < threshold <= refined_iou),
            "down": bool(refined_iou < threshold <= original_iou),
        }
    return output


def summarize_rows(rows):
    categories = Counter(row["failure_category"] for row in rows)
    valid = [row for row in rows if row["original_best_gt_instance_id"] > 0]
    improved = [
        row for row in valid if row["refined_minus_original_target_iou"] > 1e-12
    ]
    worsened = [
        row for row in valid if row["refined_minus_original_target_iou"] < -1e-12
    ]
    transitions = {}
    for threshold in AP_THRESHOLDS:
        name = f"iou{int(round(float(threshold) * 100)):02d}"
        transitions[name] = {
            "original_qualified_count": int(sum(
                row["original_target_iou"] >= threshold for row in valid
            )),
            "refined_qualified_count": int(sum(
                row["refined_target_iou"] >= threshold for row in valid
            )),
            "upward_cross_count": int(sum(row[f"{name}_up"] for row in valid)),
            "downward_cross_count": int(sum(row[f"{name}_down"] for row in valid)),
        }
    return {
        "effective_changed_candidate_count": len(rows),
        "valid_original_target_count": len(valid),
        "failure_category_counts": dict(sorted(categories.items())),
        "core_native_target_aligned_count": sum(
            row["core_native_target_aligned"] is True for row in rows
        ),
        "core_native_target_mismatch_count": sum(
            row["core_native_target_aligned"] is False for row in rows
        ),
        "core_or_native_target_unassigned_count": sum(
            row["core_native_target_aligned"] is None for row in rows
        ),
        "target_iou_improved_count": len(improved),
        "target_iou_worsened_count": len(worsened),
        "target_iou_unchanged_count": len(valid) - len(improved) - len(worsened),
        "refined_best_target_switched_count": sum(
            row["refined_best_target_switched"] is True for row in rows
        ),
        "worsened_original_iou_ge25_count": sum(
            row["original_target_iou"] >= 0.25 for row in worsened
        ),
        "worsened_original_iou_ge50_count": sum(
            row["original_target_iou"] >= 0.50 for row in worsened
        ),
        "worsened_original_iou_ge75_count": sum(
            row["original_target_iou"] >= 0.75 for row in worsened
        ),
        "threshold_transitions": transitions,
        "refined_minus_original_target_iou": _stats(
            row["refined_minus_original_target_iou"] for row in valid
        ),
        "added_original_target_precision": _stats(
            row["added_original_target_precision"] for row in valid
        ),
        "expansion_ratio": _stats(row["expansion_ratio"] for row in rows),
        "native_score": _stats(row["native_score"] for row in rows),
        "native_score_one_count": sum(
            abs(row["native_score"] - 1.0) <= 1e-12 for row in rows
        ),
        "native_score_unique_count": len({row["native_score"] for row in rows}),
        "native_score_scene_percentile": _stats(
            row["native_score_scene_percentile"] for row in rows
        ),
    }


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(
        args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size
    )
    original_masks = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
    )
    refined_masks = np.load(
        args.refined_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
    )
    scores = np.asarray(np.load(
        args.native_prediction_cache / f"{scene_name}_pred_scores.npy", mmap_mode="r"
    ), dtype=np.float64)
    if original_masks.shape[0] != len(gt_ids) and original_masks.shape[1] == len(gt_ids):
        original_masks = original_masks.T
    if refined_masks.shape[0] != len(gt_ids) and refined_masks.shape[1] == len(gt_ids):
        refined_masks = refined_masks.T
    if original_masks.shape != refined_masks.shape or original_masks.shape != (
        len(gt_ids), len(scores)
    ):
        raise ValueError(f"{scene_name} 的 GT/原始/补全缓存维度不一致")
    base_tracks = _load_tracks(args.base_track_root, scene_name)
    prompt_tracks = _load_tracks(args.prompt_track_root, scene_name)
    pair_payload = json.loads(
        (args.refined_prediction_cache / f"{scene_name}_paired_completion.json").read_text()
    )
    effective_pairs = [
        row for row in pair_payload["pairs"] if int(row["actual_added_point_count"]) > 0
    ]
    rows = []
    for pair in effective_pairs:
        track_id = int(pair["track_id"])
        candidate_id = int(pair["native_candidate_id"])
        if track_id not in base_tracks or track_id not in prompt_tracks:
            raise ValueError(f"{scene_name} 配对引用未知 track {track_id}")
        original_points = np.flatnonzero(original_masks[:, candidate_id])
        refined_points = np.flatnonzero(refined_masks[:, candidate_id])
        actual_added = np.setdiff1d(
            refined_points, original_points, assume_unique=True
        )
        if len(actual_added) != int(pair["actual_added_point_count"]):
            raise ValueError(f"{scene_name} candidate {candidate_id} 实际增量与 manifest 不符")
        base_points = _load_track_points(
            base_tracks[track_id], len(gt_ids), scene_name, "v0"
        )
        prompt_points = _load_track_points(
            prompt_tracks[track_id], len(gt_ids), scene_name, "prompt"
        )
        core_points = np.intersect1d(base_points, prompt_points, assume_unique=True)
        attribution = attribute_completion(
            original_points, refined_points, core_points, gt_ids, gt_sizes
        )
        transitions = threshold_transitions(
            attribution["original_target_iou"], attribution["refined_target_iou"]
        )
        score = float(scores[candidate_id])
        row = {
            "scene_name": scene_name,
            "track_id": track_id,
            "native_candidate_id": candidate_id,
            "native_score": score,
            "native_score_descending_rank": int(np.sum(scores > score) + 1),
            "native_score_scene_percentile": float(np.mean(scores <= score)),
            "common_core_native_iou": float(pair["common_core_native_iou"]),
            "native_point_count_before": int(pair["native_point_count_before"]),
            "native_point_count_after": int(pair["native_point_count_after"]),
            **attribution,
        }
        for name, transition in transitions.items():
            row[f"{name}_up"] = transition["up"]
            row[f"{name}_down"] = transition["down"]
        rows.append(row)
    del original_masks, refined_masks
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--base-track-root", type=Path, required=True)
    parser.add_argument("--prompt-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--refined-prediction-cache", type=Path, required=True)
    parser.add_argument(
        "--gt-instance-dir", type=Path, default=Path("data/scannet200/ground_truth")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--allow-gt-diagnostics", action="store_true")
    args = parser.parse_args()
    if not args.allow_gt_diagnostics:
        raise SystemExit("必须显式传入 --allow-gt-diagnostics；GT 只能用于离线归因。")
    for name in (
        "scene_list",
        "base_track_root",
        "prompt_track_root",
        "native_prediction_cache",
        "refined_prediction_cache",
        "gt_instance_dir",
        "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise SystemExit(f"输出目录已存在，为避免覆盖已拒绝执行：{args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    native_contract = _native_cache_contract(
        args.native_prediction_cache, "mask3d_yoloworld_only"
    )
    refined_manifest = _refined_cache_contract(
        args.refined_prediction_cache, args.native_prediction_cache, len(scenes)
    )
    if int(refined_manifest["effective_refined_candidate_count"]) <= 0:
        raise ValueError("补全 manifest 没有实际改变候选")

    rows = []
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{len(scene_rows)} 个实际改变候选",
            flush=True,
        )
    if len(rows) != int(refined_manifest["effective_refined_candidate_count"]):
        raise ValueError("实际归因候选数与补全 manifest 不一致")
    if sum(row["added_point_count"] for row in rows) != int(
        refined_manifest["actual_added_point_count"]
    ):
        raise ValueError("实际归因新增点数与补全 manifest 不一致")

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "paired_completion_gt_attribution.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    harmful = sorted(
        rows,
        key=lambda row: (
            row["refined_minus_original_target_iou"],
            row["scene_name"],
            row["native_candidate_id"],
        ),
    )
    with (args.output_dir / "most_harmful_candidates.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(harmful)
    payload = {
        "diagnostic_type": (
            "GT-only 固定 Mask3D 局部补全失败归因；候选级 IoU 变化不是 AP 的精确分解。"
        ),
        "decision_constraint": (
            "GT 不得回流至配对、mask、superpoint、类别、分数或阈值；"
            "本报告只区分固定候选的对象错配、过扩张和改善。"
        ),
        "scene_count": len(scenes),
        "native_cache_contract": native_contract,
        "fixed_completion_contract": {
            key: refined_manifest[key]
            for key in (
                "effective_refined_candidate_count",
                "actual_added_point_count",
                "mutual_best_positive_pair_count",
            )
        },
        "attribution": summarize_rows(rows),
        "params": {key: value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps(payload["attribution"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

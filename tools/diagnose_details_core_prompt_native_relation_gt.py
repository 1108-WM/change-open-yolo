#!/usr/bin/env python3
"""GT-only：归因冻结 v0/prompt/native 几何关系中的收益、冗余与竞争。

输入必须是先验生成的无 GT 关系账本。GT 只在本工具中为每条冻结轨迹指定共同
核心的事后匹配实例，并比较同一实例上的 v0、prompt 与 native IoU。工具不输出
推理规则，不改变 mask、分数、候选或阈值，也不运行 AP。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("场景列表为空或含重复场景")
    return scenes


def _native_cache_contract(root, expected_mode):
    manifest_path = root / "native_cache_no_gt_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"native 缓存缺少 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    inputs = manifest.get("candidate_inputs", {})
    mode = inputs.get("mode")
    if mode is None:
        mode = "strong_native" if int(inputs.get("loaded", 0)) > 0 else "unknown"
    if mode != expected_mode:
        raise ValueError(f"native 缓存模式不符：期望 {expected_mode}，实际 {mode}")
    return {
        "mode": mode,
        "manifest_path": str(manifest_path),
        "decision_state": manifest.get("decision_state"),
    }


def _load_gt(path, min_region_size):
    gt_ids = np.loadtxt(path, dtype=np.int64)
    valid_classes = {int(item) for item in VALID_CLASS_IDS_200_INST}
    ids, counts = np.unique(gt_ids, return_counts=True)
    sizes = {
        int(instance_id): int(count)
        for instance_id, count in zip(ids, counts)
        if int(instance_id) > 0
        and int(instance_id) // 1000 in valid_classes
        and int(count) >= int(min_region_size)
    }
    return gt_ids, sizes


def _load_points(track, point_count):
    path = Path(track["points_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    points = np.unique(np.asarray(np.load(path)["point_indices"], dtype=np.int64))
    return points[(points >= 0) & (points < point_count)]


def best_gt_instance(points, gt_ids, gt_sizes):
    points = np.unique(np.asarray(points, dtype=np.int64))
    best_id, best_iou = -1, 0.0
    if not len(points):
        return best_id, best_iou
    ids, counts = np.unique(gt_ids[points], return_counts=True)
    for instance_id, intersection in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id not in gt_sizes:
            continue
        iou = float(
            int(intersection)
            / max(1, len(points) + gt_sizes[instance_id] - int(intersection))
        )
        if iou > best_iou:
            best_id, best_iou = instance_id, iou
    return best_id, best_iou


def target_metrics(points, gt_ids, instance_id, gt_size):
    points = np.unique(np.asarray(points, dtype=np.int64))
    if not len(points) or instance_id <= 0:
        return {"iou": 0.0, "precision": 0.0, "coverage": 0.0}
    intersection = int(np.sum(gt_ids[points] == int(instance_id)))
    return {
        "iou": float(intersection / max(1, len(points) + gt_size - intersection)),
        "precision": float(intersection / len(points)),
        "coverage": float(intersection / max(1, gt_size)),
    }


def native_target_relation(
    native_masks, native_scores, gt_ids, instance_id, gt_size, native_sizes=None
):
    if instance_id <= 0 or native_masks.shape[1] == 0:
        return {
            "best_iou": 0.0,
            "best_candidate_id": -1,
            "best_candidate_score": None,
            "max_score_at_iou25": None,
            "max_score_at_iou50": None,
        }
    if native_sizes is None:
        native_sizes = native_masks.sum(axis=0, dtype=np.int64)
    intersections = native_masks[gt_ids == int(instance_id)].sum(
        axis=0, dtype=np.int64
    )
    ious = intersections / np.maximum(1, native_sizes + int(gt_size) - intersections)
    best_iou = float(ious.max(initial=0.0))
    tied = np.flatnonzero(np.abs(ious - best_iou) <= 1e-12)
    best_id = int(sorted(tied.tolist(), key=lambda item: (-float(native_scores[item]), item))[0])

    def max_score(threshold):
        eligible = native_scores[ious >= threshold]
        return float(eligible.max()) if len(eligible) else None

    return {
        "best_iou": best_iou,
        "best_candidate_id": best_id,
        "best_candidate_score": float(native_scores[best_id]),
        "max_score_at_iou25": max_score(0.25),
        "max_score_at_iou50": max_score(0.50),
    }


def _track_map(root, scene_name):
    tracks = json.loads((root / scene_name / "automatic_tracks.json").read_text())["tracks"]
    output = {int(track["track_id"]): track for track in tracks}
    if len(output) != len(tracks):
        raise ValueError(f"{scene_name} 含重复 track_id")
    return output


def _scene_rows(scene_name, args):
    gt_ids, gt_sizes = _load_gt(
        args.gt_instance_dir / f"{scene_name}.txt", args.min_region_size
    )
    base = _track_map(args.base_track_root, scene_name)
    prompt = _track_map(args.prompt_track_root, scene_name)
    ledger_rows = json.loads(
        (args.relation_ledger_root / scene_name / "native_relation_ledger.json").read_text()
    )
    ledger = {int(row["track_id"]): row for row in ledger_rows}
    if set(base) != set(prompt) or set(base) != set(ledger):
        raise ValueError(f"{scene_name} 的 v0/prompt/ledger track_id 不一致")

    native_masks = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
    )
    native_scores = np.asarray(np.load(
        args.native_prediction_cache / f"{scene_name}_pred_scores.npy", mmap_mode="r"
    ), dtype=np.float64)
    if native_masks.shape[0] != len(gt_ids) and native_masks.shape[1] == len(gt_ids):
        native_masks = native_masks.T
    if native_masks.shape != (len(gt_ids), len(native_scores)):
        raise ValueError(f"{scene_name} native 缓存维度不匹配")
    native_sizes = native_masks.sum(axis=0, dtype=np.int64)
    native_target_cache = {}

    rows = []
    for track_id in sorted(base):
        base_points = _load_points(base[track_id], len(gt_ids))
        prompt_points = _load_points(prompt[track_id], len(gt_ids))
        common_points = np.intersect1d(base_points, prompt_points, assume_unique=True)
        added_points = np.setdiff1d(prompt_points, base_points, assume_unique=True)
        target_id, core_best_iou = best_gt_instance(common_points, gt_ids, gt_sizes)
        gt_size = int(gt_sizes.get(target_id, 0))
        base_metrics = target_metrics(base_points, gt_ids, target_id, gt_size)
        prompt_metrics = target_metrics(prompt_points, gt_ids, target_id, gt_size)
        added_metrics = target_metrics(added_points, gt_ids, target_id, gt_size)
        if target_id not in native_target_cache:
            native_target_cache[target_id] = native_target_relation(
                native_masks, native_scores, gt_ids, target_id, gt_size, native_sizes
            )
        native = native_target_cache[target_id]
        frozen = ledger[track_id]
        prompt_top = frozen["prompt_native_relation"]["top_matches"]
        prompt_top_iou = float(prompt_top[0]["iou"]) if prompt_top else 0.0
        track_score = float(frozen["mean_node_quality"])
        rows.append({
            "scene_name": scene_name,
            "track_id": track_id,
            "core_prompt_variant_changed": bool(
                frozen["core_prompt_metadata"]["variant_changed"]
            ),
            "final_branch_changed_from_v0": bool(
                frozen["point_relation"]["changed"]
            ),
            "target_gt_instance_id": int(target_id),
            "common_core_best_gt_iou": float(core_best_iou),
            "v0_target_iou": float(base_metrics["iou"]),
            "prompt_target_iou": float(prompt_metrics["iou"]),
            "prompt_minus_v0_target_iou": float(
                prompt_metrics["iou"] - base_metrics["iou"]
            ),
            "v0_target_precision": float(base_metrics["precision"]),
            "prompt_target_precision": float(prompt_metrics["precision"]),
            "v0_target_coverage": float(base_metrics["coverage"]),
            "prompt_target_coverage": float(prompt_metrics["coverage"]),
            "added_point_count": int(len(added_points)),
            "added_target_precision": (
                float(added_metrics["precision"]) if len(added_points) else None
            ),
            "native_target_best_iou": float(native["best_iou"]),
            "native_target_best_candidate_id": int(native["best_candidate_id"]),
            "native_target_best_candidate_score": native["best_candidate_score"],
            "native_target_max_score_at_iou25": native["max_score_at_iou25"],
            "native_target_max_score_at_iou50": native["max_score_at_iou50"],
            "track_score": track_score,
            "track_score_above_native_iou25_max": (
                track_score > native["max_score_at_iou25"]
                if native["max_score_at_iou25"] is not None else None
            ),
            "track_score_above_native_iou50_max": (
                track_score > native["max_score_at_iou50"]
                if native["max_score_at_iou50"] is not None else None
            ),
            "prompt_top_native_iou": prompt_top_iou,
            "prompt_native_mutual_best": bool(frozen["prompt_native_mutual_best"]),
            "prompt_touching_native_count": int(
                frozen["prompt_native_relation"]["touching_native_count"]
            ),
            "added_inside_any_native_ratio": frozen[
                "added_point_native_coverage"
            ]["inside_any_native_ratio"],
            "added_reachable_from_core_ratio": frozen[
                "added_superpoint_connectivity"
            ]["reachable_from_core_ratio"],
        })
    return rows


def _stats(values):
    values = np.asarray([item for item in values if item is not None], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def summarize_rows(rows):
    valid = [row for row in rows if row["target_gt_instance_id"] > 0]
    eps = 1e-12
    improved = [row for row in valid if row["prompt_minus_v0_target_iou"] > eps]
    worsened = [row for row in valid if row["prompt_minus_v0_target_iou"] < -eps]
    up25 = [row for row in valid if row["v0_target_iou"] < 0.25 <= row["prompt_target_iou"]]
    up50 = [row for row in valid if row["v0_target_iou"] < 0.50 <= row["prompt_target_iou"]]
    down25 = [row for row in valid if row["prompt_target_iou"] < 0.25 <= row["v0_target_iou"]]
    down50 = [row for row in valid if row["prompt_target_iou"] < 0.50 <= row["v0_target_iou"]]
    return {
        "track_count": len(rows),
        "valid_target_track_count": len(valid),
        "target_iou_improved_count": len(improved),
        "target_iou_unchanged_count": len(valid) - len(improved) - len(worsened),
        "target_iou_worsened_count": len(worsened),
        "upward_cross_ap25_count": len(up25),
        "upward_cross_ap50_count": len(up50),
        "downward_cross_ap25_count": len(down25),
        "downward_cross_ap50_count": len(down50),
        "unique_vs_native_upward_cross_ap25_count": sum(
            row["native_target_best_iou"] < 0.25 for row in up25
        ),
        "unique_vs_native_upward_cross_ap50_count": sum(
            row["native_target_best_iou"] < 0.50 for row in up50
        ),
        "improved_but_native_already_ap25_count": sum(
            row["native_target_best_iou"] >= 0.25 for row in improved
        ),
        "improved_but_native_already_ap50_count": sum(
            row["native_target_best_iou"] >= 0.50 for row in improved
        ),
        "prompt_mask_duplicate_iou50_count": sum(
            row["prompt_top_native_iou"] >= 0.50 for row in valid
        ),
        "prompt_native_mutual_best_count": sum(
            row["prompt_native_mutual_best"] for row in valid
        ),
        "track_score_above_native_iou25_max_count": sum(
            row["track_score_above_native_iou25_max"] is True for row in valid
        ),
        "track_score_below_or_equal_native_iou25_max_count": sum(
            row["track_score_above_native_iou25_max"] is False for row in valid
        ),
        "prompt_minus_v0_target_iou": _stats(
            row["prompt_minus_v0_target_iou"] for row in valid
        ),
        "added_target_precision": _stats(
            row["added_target_precision"] for row in valid
        ),
        "native_target_best_iou": _stats(
            row["native_target_best_iou"] for row in valid
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--base-track-root", type=Path, required=True)
    parser.add_argument("--prompt-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument(
        "--expected-cache-mode",
        choices=("mask3d_yoloworld_only", "strong_native"),
        required=True,
    )
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
        "scene_list", "relation_ledger_root", "base_track_root",
        "prompt_track_root", "native_prediction_cache", "gt_instance_dir", "output_dir",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_dir}")
    scenes = _read_scenes(args.scene_list)
    args.native_cache_contract = _native_cache_contract(
        args.native_prediction_cache, args.expected_cache_mode
    )
    ledger_summary = json.loads(
        (args.relation_ledger_root / "native_relation_ledger_summary.json").read_text()
    )
    ledger_cache = Path(ledger_summary["params"]["native_prediction_cache"])
    if ledger_cache.resolve() != args.native_prediction_cache.resolve():
        raise ValueError("GT-only 归因的 native 缓存与冻结无 GT 账本不一致")
    rows = []
    for index, scene_name in enumerate(scenes, start=1):
        scene_rows = _scene_rows(scene_name, args)
        rows.extend(scene_rows)
        print(f"[场景完成] {index}/{len(scenes)} {scene_name}: {len(scene_rows)} 条", flush=True)
    args.output_dir.mkdir(parents=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "track_gt_attribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "diagnostic_type": "GT-only 冻结 v0/prompt/native 几何关系归因；不运行 AP。",
        "decision_constraint": (
            "GT 不得回流到轨迹、superpoint、候选、分数、阈值、类别或融合；"
            "本报告只验证无GT结构假设。"
        ),
        "scene_count": len(scenes),
        "native_cache_contract": args.native_cache_contract,
        "all_tracks": summarize_rows(rows),
        "final_branch_changed_from_v0": summarize_rows([
            row for row in rows if row["final_branch_changed_from_v0"]
        ]),
        "core_prompt_changed_only": summarize_rows([
            row for row in rows if row["core_prompt_variant_changed"]
        ]),
        "params": {key: value for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "scene_count": payload["scene_count"],
        "core_prompt_changed_only": payload["core_prompt_changed_only"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

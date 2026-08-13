#!/usr/bin/env python3
"""建立共识 v0、核心点提示变体与冻结 native 候选的无 GT 几何关系账本。

本工具只读取点级 mask、原始 superpoint 和冻结 native mask。它不读取 native
类别/分数、GT、语义或 AP，也不生成、筛选、排序或修改任何候选。账本用于定位
提示几何进入基线后被候选重叠/竞争稀释的位置，不定义新的接受规则。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_automatic_sam_track_growth_ledger import _raw_superpoint_context


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path):
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"场景列表含重复项：{path}")
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
        raise ValueError(
            f"native 缓存模式不符：期望 {expected_mode}，实际 {mode}；"
            f"缓存为 {root}"
        )
    return {
        "mode": mode,
        "manifest_path": str(manifest_path),
        "decision_state": manifest.get("decision_state"),
        "scene_count": int(manifest.get("scene_count", 0)),
    }


def _load_tracks(root, scene_name):
    path = root / scene_name / "automatic_tracks.json"
    payload = json.loads(path.read_text())
    tracks = payload.get("tracks", [])
    by_id = {int(track["track_id"]): track for track in tracks}
    if len(by_id) != len(tracks):
        raise ValueError(f"{scene_name} 在 {root} 中含重复 track_id")
    return by_id


def _load_track_points(track, point_count, scene_name, source_name):
    path = Path(track["points_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    points = np.unique(
        np.asarray(np.load(path)["point_indices"], dtype=np.int64)
    )
    if len(points) != int(track.get("point_count", len(points))):
        raise ValueError(
            f"{scene_name} track {track['track_id']} 的 {source_name} 点数声明与文件不符"
        )
    if len(points) and (int(points[0]) < 0 or int(points[-1]) >= point_count):
        raise ValueError(
            f"{scene_name} track {track['track_id']} 的 {source_name} 点索引越界"
        )
    return points


def set_relation(left, right):
    """返回两个已去重点集的对称几何关系。"""
    left = np.unique(np.asarray(left, dtype=np.int64))
    right = np.unique(np.asarray(right, dtype=np.int64))
    intersection = np.intersect1d(left, right, assume_unique=True)
    union_count = len(left) + len(right) - len(intersection)
    return {
        "intersection_point_count": int(len(intersection)),
        "union_point_count": int(union_count),
        "iou": float(len(intersection) / max(1, union_count)),
        "left_inside_right_ratio": float(len(intersection) / max(1, len(left))),
        "right_inside_left_ratio": float(len(intersection) / max(1, len(right))),
    }


def native_overlap_vectors(points, native_masks):
    """计算一个点集与全部 native mask 的无阈值重叠向量。"""
    points = np.unique(np.asarray(points, dtype=np.int64))
    candidate_count = native_masks.shape[1]
    native_sizes = native_masks.sum(axis=0, dtype=np.int64)
    if len(points) == 0 or candidate_count == 0:
        zeros = np.zeros(candidate_count, dtype=np.float64)
        return {
            "intersection": np.zeros(candidate_count, dtype=np.int64),
            "iou": zeros,
            "source_inside_native": zeros.copy(),
            "native_inside_source": zeros.copy(),
            "native_sizes": native_sizes,
        }
    intersection = native_masks[points].sum(axis=0, dtype=np.int64)
    union = len(points) + native_sizes - intersection
    return {
        "intersection": intersection,
        "iou": intersection / np.maximum(1, union),
        "source_inside_native": intersection / len(points),
        "native_inside_source": intersection / np.maximum(1, native_sizes),
        "native_sizes": native_sizes,
    }


def summarize_native_relation(vectors, top_k=2):
    intersection = vectors["intersection"]
    iou = vectors["iou"]
    positive = np.flatnonzero(intersection > 0)
    ordered = sorted(positive.tolist(), key=lambda item: (-float(iou[item]), item))
    top = []
    for candidate_id in ordered[:top_k]:
        top.append({
            "native_candidate_id": int(candidate_id),
            "intersection_point_count": int(intersection[candidate_id]),
            "native_point_count": int(vectors["native_sizes"][candidate_id]),
            "iou": float(iou[candidate_id]),
            "source_inside_native_ratio": float(
                vectors["source_inside_native"][candidate_id]
            ),
            "native_inside_source_ratio": float(
                vectors["native_inside_source"][candidate_id]
            ),
        })
    return {
        "native_candidate_count": int(len(intersection)),
        "touching_native_count": int(len(positive)),
        "iou_ge_0_10_native_count": int(np.sum(iou >= 0.10)),
        "top_matches": top,
    }


def greedy_native_cover(points, native_masks, targets=(0.50, 0.80, 0.90)):
    """描述 native 并集覆盖核心所需数量；仅作账本统计，不形成接受阈值。"""
    points = np.unique(np.asarray(points, dtype=np.int64))
    targets = tuple(float(item) for item in targets)
    if not len(points):
        return {
            "native_union_coverage_ratio": 0.0,
            "greedy_native_candidate_ids": [],
            **{f"greedy_count_to_{int(target * 100)}pct": -1 for target in targets},
        }
    support = np.asarray(native_masks[points], dtype=bool)
    union_coverage = float(np.any(support, axis=1).mean()) if support.shape[1] else 0.0
    uncovered = np.ones(len(points), dtype=bool)
    available = np.ones(support.shape[1], dtype=bool)
    selected = []
    counts = {target: -1 for target in targets}
    while support.shape[1] and np.any(uncovered):
        gains = support[uncovered].sum(axis=0, dtype=np.int64)
        gains[~available] = -1
        candidate_id = int(np.argmax(gains))
        if int(gains[candidate_id]) <= 0:
            break
        selected.append(candidate_id)
        available[candidate_id] = False
        uncovered[support[:, candidate_id]] = False
        coverage = 1.0 - float(uncovered.mean())
        for target in targets:
            if counts[target] < 0 and coverage + 1e-12 >= target:
                counts[target] = len(selected)
        if all(counts[target] >= 0 for target in targets):
            break
    return {
        "native_union_coverage_ratio": union_coverage,
        "greedy_native_candidate_ids": selected,
        **{
            f"greedy_count_to_{int(target * 100)}pct": int(counts[target])
            for target in targets
        },
    }


def added_point_native_coverage(added_points, native_masks, best_native_id):
    added_points = np.unique(np.asarray(added_points, dtype=np.int64))
    if not len(added_points):
        return {
            "added_point_count": 0,
            "inside_best_native_count": 0,
            "inside_other_native_count": 0,
            "inside_any_native_count": 0,
            "inside_best_native_ratio": None,
            "inside_other_native_ratio": None,
            "inside_any_native_ratio": None,
        }
    support = np.asarray(native_masks[added_points], dtype=bool)
    if 0 <= int(best_native_id) < support.shape[1]:
        best = support[:, int(best_native_id)]
        other = np.any(
            np.delete(support, int(best_native_id), axis=1), axis=1
        ) if support.shape[1] > 1 else np.zeros(len(added_points), dtype=bool)
    else:
        best = np.zeros(len(added_points), dtype=bool)
        other = np.any(support, axis=1) if support.shape[1] else best.copy()
    any_native = np.any(support, axis=1) if support.shape[1] else best.copy()
    return {
        "added_point_count": int(len(added_points)),
        "inside_best_native_count": int(best.sum()),
        "inside_other_native_count": int(other.sum()),
        "inside_any_native_count": int(any_native.sum()),
        "inside_best_native_ratio": float(best.mean()),
        "inside_other_native_ratio": float(other.mean()),
        "inside_any_native_ratio": float(any_native.mean()),
    }


def added_superpoint_connectivity(core_ids, added_ids, neighbors):
    """记录新增 superpoint 对共同核心的直接与经新增原子的传递连通性。"""
    core = set(int(item) for item in core_ids)
    added = set(int(item) for item in added_ids)
    allowed = core | added
    reachable = set(core)
    frontier = list(core)
    while frontier:
        current = frontier.pop()
        for edge in neighbors.get(current, []):
            target = int(edge["neighbor_superpoint_id"])
            if target in allowed and target not in reachable:
                reachable.add(target)
                frontier.append(target)
    records = []
    for superpoint_id in sorted(added):
        edges = [
            edge for edge in neighbors.get(superpoint_id, [])
            if int(edge["neighbor_superpoint_id"]) in allowed
        ]
        core_edges = [
            edge for edge in edges
            if int(edge["neighbor_superpoint_id"]) in core
        ]
        records.append({
            "superpoint_id": superpoint_id,
            "directly_adjacent_to_core": bool(core_edges),
            "reachable_from_core_via_added": superpoint_id in reachable,
            "adjacent_core_superpoint_count": len(core_edges),
            "adjacent_added_superpoint_count": sum(
                int(edge["neighbor_superpoint_id"]) in added for edge in edges
            ),
            "core_boundary_contact_count_sum": int(sum(
                int(edge["boundary_contact_count"]) for edge in core_edges
            )),
        })
    direct_count = sum(item["directly_adjacent_to_core"] for item in records)
    reachable_count = sum(item["reachable_from_core_via_added"] for item in records)
    return {
        "added_superpoint_count": len(records),
        "directly_adjacent_to_core_count": int(direct_count),
        "reachable_from_core_count": int(reachable_count),
        "directly_adjacent_to_core_ratio": (
            float(direct_count / len(records)) if records else None
        ),
        "reachable_from_core_ratio": (
            float(reachable_count / len(records)) if records else None
        ),
        "added_superpoints": records,
    }


def _validate_track_contract(scene_name, base_tracks, prompt_tracks, valid_superpoints):
    if set(base_tracks) != set(prompt_tracks):
        missing_prompt = sorted(set(base_tracks) - set(prompt_tracks))
        missing_base = sorted(set(prompt_tracks) - set(base_tracks))
        raise ValueError(
            f"{scene_name} 的 v0/prompt track_id 不一致："
            f"prompt缺少{missing_prompt[:5]}，v0缺少{missing_base[:5]}"
        )
    for track_id in sorted(base_tracks):
        base = base_tracks[track_id]
        prompt = prompt_tracks[track_id]
        if float(base.get("mean_node_quality", 0.0)) != float(
            prompt.get("mean_node_quality", 0.0)
        ):
            raise ValueError(f"{scene_name} track {track_id} 的 mean_node_quality 被修改")
        for source_name, track in (("v0", base), ("prompt", prompt)):
            ids = [int(item) for item in track.get("superpoint_ids", [])]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"{scene_name} track {track_id} 的 {source_name} superpoint 重复"
                )
            unknown = sorted(set(ids) - valid_superpoints)
            if unknown:
                raise ValueError(
                    f"{scene_name} track {track_id} 的 {source_name} 引用未知 superpoint："
                    f"{unknown[:5]}"
                )
        declared_added = [
            int(item) for item in prompt.get("core_prompt_added_superpoint_ids", [])
        ]
        declared_count = int(
            prompt.get("core_prompt_added_superpoint_count", len(declared_added))
        )
        declared_changed = bool(prompt.get("core_prompt_variant_changed", False))
        if (
            len(declared_added) != len(set(declared_added))
            or declared_count != len(declared_added)
            or declared_changed != bool(declared_added)
            or not set(declared_added).issubset(
                set(int(item) for item in prompt.get("superpoint_ids", []))
            )
        ):
            raise ValueError(
                f"{scene_name} track {track_id} 的 core prompt 增量元数据不一致"
            )


def _stats(values):
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _scene_summary(scene_name, rows, native_count):
    added_rows = [row for row in rows if row["point_relation"]["added_point_count"] > 0]
    removed_rows = [row for row in rows if row["point_relation"]["removed_point_count"] > 0]
    prompt_top_ious = [
        row["prompt_native_relation"]["top_matches"][0]["iou"]
        if row["prompt_native_relation"]["top_matches"] else 0.0
        for row in rows
    ]
    base_top_ious = [
        row["v0_native_relation"]["top_matches"][0]["iou"]
        if row["v0_native_relation"]["top_matches"] else 0.0
        for row in rows
    ]
    deltas = np.asarray(prompt_top_ious) - np.asarray(base_top_ious)
    return {
        "scene_name": scene_name,
        "track_count": len(rows),
        "native_candidate_count": int(native_count),
        "changed_track_count": sum(row["point_relation"]["changed"] for row in rows),
        "core_prompt_changed_track_count": sum(
            row["core_prompt_metadata"]["variant_changed"] for row in rows
        ),
        "core_prompt_added_superpoint_count": sum(
            row["core_prompt_metadata"]["added_superpoint_count"] for row in rows
        ),
        "core_prompt_added_point_count": sum(
            row["core_prompt_metadata"]["added_point_count"] for row in rows
        ),
        "track_with_added_points_count": len(added_rows),
        "track_with_removed_points_count": len(removed_rows),
        "added_point_count": sum(
            row["point_relation"]["added_point_count"] for row in rows
        ),
        "removed_point_count": sum(
            row["point_relation"]["removed_point_count"] for row in rows
        ),
        "mutual_best_prompt_native_count": sum(
            row["prompt_native_mutual_best"] for row in rows
        ),
        "prompt_top_iou_improved_count": int(np.sum(deltas > 1e-12)),
        "prompt_top_iou_unchanged_count": int(np.sum(np.abs(deltas) <= 1e-12)),
        "prompt_top_iou_worsened_count": int(np.sum(deltas < -1e-12)),
        "v0_top_native_iou": _stats(base_top_ious),
        "prompt_top_native_iou": _stats(prompt_top_ious),
        "prompt_minus_v0_top_native_iou": _stats(deltas),
        "added_inside_best_native_count": sum(
            row["added_point_native_coverage"]["inside_best_native_count"]
            for row in rows
        ),
        "added_inside_other_native_count": sum(
            row["added_point_native_coverage"]["inside_other_native_count"]
            for row in rows
        ),
        "added_inside_any_native_count": sum(
            row["added_point_native_coverage"]["inside_any_native_count"]
            for row in rows
        ),
        "added_superpoint_count": sum(
            row["added_superpoint_connectivity"]["added_superpoint_count"]
            for row in rows
        ),
        "added_superpoint_direct_core_count": sum(
            row["added_superpoint_connectivity"]["directly_adjacent_to_core_count"]
            for row in rows
        ),
        "added_superpoint_reachable_core_count": sum(
            row["added_superpoint_connectivity"]["reachable_from_core_count"]
            for row in rows
        ),
        "integrity": {
            "track_id_sets_equal": True,
            "mean_node_quality_unchanged": True,
            "point_and_superpoint_indices_valid": True,
        },
    }


def _build_scene(scene_name, args):
    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    raw_superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    valid_superpoints = set(int(item) for item in np.unique(raw_superpoints))
    base_tracks = _load_tracks(args.base_track_root, scene_name)
    prompt_tracks = _load_tracks(args.prompt_track_root, scene_name)
    _validate_track_contract(
        scene_name, base_tracks, prompt_tracks, valid_superpoints
    )

    native_masks = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
    )
    if native_masks.ndim != 2:
        raise ValueError(f"{scene_name} 的 native mask 维度异常：{native_masks.shape}")
    if native_masks.shape[0] != len(processed) and native_masks.shape[1] == len(processed):
        native_masks = native_masks.T
    if native_masks.shape[0] != len(processed):
        raise ValueError(f"{scene_name} 的 native mask 点数不匹配")
    native_masks = np.asarray(native_masks, dtype=bool)

    context = _raw_superpoint_context(
        processed,
        args.adjacency_knn,
        args.adjacency_max_distance,
        args.min_contact_points,
        args.min_contact_ratio,
    )
    track_cache = {}
    prompt_iou_rows = []
    track_ids = sorted(base_tracks)
    for track_id in track_ids:
        base = base_tracks[track_id]
        prompt = prompt_tracks[track_id]
        base_points = _load_track_points(
            base, len(processed), scene_name, "v0"
        )
        prompt_points = _load_track_points(
            prompt, len(processed), scene_name, "prompt"
        )
        common_points = np.intersect1d(base_points, prompt_points, assume_unique=True)
        added_points = np.setdiff1d(prompt_points, base_points, assume_unique=True)
        removed_points = np.setdiff1d(base_points, prompt_points, assume_unique=True)
        base_vectors = native_overlap_vectors(base_points, native_masks)
        prompt_vectors = native_overlap_vectors(prompt_points, native_masks)
        core_vectors = native_overlap_vectors(common_points, native_masks)
        prompt_iou_rows.append(prompt_vectors["iou"])
        track_cache[track_id] = {
            "base_points": base_points,
            "prompt_points": prompt_points,
            "common_points": common_points,
            "added_points": added_points,
            "removed_points": removed_points,
            "base_vectors": base_vectors,
            "prompt_vectors": prompt_vectors,
            "core_vectors": core_vectors,
        }

    if native_masks.shape[1] and track_ids:
        prompt_iou_matrix = np.stack(prompt_iou_rows, axis=0)
        best_track_indices = np.argmax(prompt_iou_matrix, axis=0)
        best_track_ious = prompt_iou_matrix[
            best_track_indices, np.arange(native_masks.shape[1])
        ]
    else:
        best_track_indices = np.zeros(native_masks.shape[1], dtype=np.int64)
        best_track_ious = np.zeros(native_masks.shape[1], dtype=np.float64)

    rows = []
    for track_index, track_id in enumerate(track_ids):
        base = base_tracks[track_id]
        prompt = prompt_tracks[track_id]
        cached = track_cache[track_id]
        base_points = cached["base_points"]
        prompt_points = cached["prompt_points"]
        common_points = cached["common_points"]
        added_points = cached["added_points"]
        removed_points = cached["removed_points"]
        base_superpoints = set(int(item) for item in base.get("superpoint_ids", []))
        prompt_superpoints = set(int(item) for item in prompt.get("superpoint_ids", []))
        common_superpoints = base_superpoints & prompt_superpoints
        added_superpoints = prompt_superpoints - base_superpoints
        removed_superpoints = base_superpoints - prompt_superpoints
        base_relation = summarize_native_relation(cached["base_vectors"])
        prompt_relation = summarize_native_relation(cached["prompt_vectors"])
        core_relation = summarize_native_relation(cached["core_vectors"])
        best_native_id = (
            int(prompt_relation["top_matches"][0]["native_candidate_id"])
            if prompt_relation["top_matches"] else -1
        )
        native_best_track_id = -1
        native_best_prompt_iou = 0.0
        mutual_best = False
        if best_native_id >= 0 and best_track_ious[best_native_id] > 0.0:
            native_best_track_id = int(track_ids[int(best_track_indices[best_native_id])])
            native_best_prompt_iou = float(best_track_ious[best_native_id])
            mutual_best = native_best_track_id == track_id
        relation = set_relation(base_points, prompt_points)
        relation.update({
            "v0_point_count": int(len(base_points)),
            "prompt_point_count": int(len(prompt_points)),
            "common_core_point_count": int(len(common_points)),
            "added_point_count": int(len(added_points)),
            "removed_point_count": int(len(removed_points)),
            "changed": bool(len(added_points) or len(removed_points)),
        })
        rows.append({
            "scene_name": scene_name,
            "track_id": track_id,
            "source_track_id_v0": int(base.get("source_track_id", track_id)),
            "source_track_id_prompt": int(prompt.get("source_track_id", track_id)),
            "mean_node_quality": float(base.get("mean_node_quality", 0.0)),
            "mean_node_quality_unchanged": True,
            "core_prompt_metadata": {
                "variant_changed": bool(prompt.get("core_prompt_variant_changed", False)),
                "added_superpoint_ids": [
                    int(item)
                    for item in prompt.get("core_prompt_added_superpoint_ids", [])
                ],
                "added_superpoint_count": int(
                    prompt.get("core_prompt_added_superpoint_count", 0)
                ),
                "added_point_count": int(
                    prompt.get("core_prompt_added_point_count", 0)
                ),
            },
            "point_relation": relation,
            "superpoint_relation": {
                "v0_superpoint_count": len(base_superpoints),
                "prompt_superpoint_count": len(prompt_superpoints),
                "common_core_superpoint_count": len(common_superpoints),
                "added_superpoint_ids": sorted(added_superpoints),
                "removed_superpoint_ids": sorted(removed_superpoints),
            },
            "v0_native_relation": base_relation,
            "prompt_native_relation": prompt_relation,
            "common_core_native_relation": core_relation,
            "prompt_native_mutual_best": mutual_best,
            "prompt_top_native_id": best_native_id,
            "top_native_best_prompt_track_id": native_best_track_id,
            "top_native_best_prompt_iou": native_best_prompt_iou,
            "common_core_native_cover": greedy_native_cover(
                common_points, native_masks
            ),
            "added_point_native_coverage": added_point_native_coverage(
                added_points, native_masks, best_native_id
            ),
            "added_superpoint_connectivity": added_superpoint_connectivity(
                common_superpoints, added_superpoints, context["neighbors"]
            ),
            "decision_state": (
                "仅记录 v0/prompt/native 几何关系；不生成、筛选、排序或修改候选。"
            ),
        })

    summary = _scene_summary(scene_name, rows, native_masks.shape[1])
    staging = args.output_root / f".{scene_name}.writing"
    published = args.output_root / scene_name
    if staging.exists() or published.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published}")
    staging.mkdir(parents=True)
    (staging / "native_relation_ledger.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging, published)
    del processed, native_masks, context, track_cache, prompt_iou_rows
    return summary


def _aggregate(summaries, args):
    totals = {
        key: sum(item[key] for item in summaries)
        for key in (
            "track_count",
            "native_candidate_count",
            "changed_track_count",
            "core_prompt_changed_track_count",
            "core_prompt_added_superpoint_count",
            "core_prompt_added_point_count",
            "track_with_added_points_count",
            "track_with_removed_points_count",
            "added_point_count",
            "removed_point_count",
            "mutual_best_prompt_native_count",
            "prompt_top_iou_improved_count",
            "prompt_top_iou_unchanged_count",
            "prompt_top_iou_worsened_count",
            "added_inside_best_native_count",
            "added_inside_other_native_count",
            "added_inside_any_native_count",
            "added_superpoint_count",
            "added_superpoint_direct_core_count",
            "added_superpoint_reachable_core_count",
        )
    }
    added_points = totals["added_point_count"]
    added_superpoints = totals["added_superpoint_count"]
    return {
        "purpose": "定位核心点提示几何进入冻结 native 候选集合后的重叠与竞争关系。",
        "gt_usage": (
            "none；不读取 GT、native 类别/分数或语义，不生成候选、不修改分数、不做 AP。"
        ),
        "decision_state": (
            "无 GT 描述性账本；不依据本账本自动接受/拒绝轨迹，不定义新阈值。"
        ),
        "scene_count": len(summaries),
        **totals,
        "added_inside_best_native_ratio": (
            totals["added_inside_best_native_count"] / added_points
            if added_points else None
        ),
        "added_inside_other_native_ratio": (
            totals["added_inside_other_native_count"] / added_points
            if added_points else None
        ),
        "added_inside_any_native_ratio": (
            totals["added_inside_any_native_count"] / added_points
            if added_points else None
        ),
        "added_superpoint_direct_core_ratio": (
            totals["added_superpoint_direct_core_count"] / added_superpoints
            if added_superpoints else None
        ),
        "added_superpoint_reachable_core_ratio": (
            totals["added_superpoint_reachable_core_count"] / added_superpoints
            if added_superpoints else None
        ),
        "integrity": {
            "all_scenes_passed": True,
            "track_id_sets_equal": True,
            "mean_node_quality_unchanged": True,
            "point_and_superpoint_indices_valid": True,
        },
        "native_cache_contract": args.native_cache_contract,
        "scenes": summaries,
        "params": {key: value for key, value in vars(args).items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--base-track-root", type=Path, required=True)
    parser.add_argument("--prompt-track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument(
        "--expected-cache-mode",
        choices=("mask3d_yoloworld_only", "strong_native"),
        required=True,
        help="强制核对缓存 manifest，防止创新点一误用 SAM-fused/BPR strong/native。",
    )
    parser.add_argument(
        "--processed-scene-root", type=Path, default=Path("data/scannet200")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adjacency-knn", type=int, default=12)
    parser.add_argument("--adjacency-max-distance", type=float, default=0.05)
    parser.add_argument("--min-contact-points", type=int, default=3)
    parser.add_argument("--min-contact-ratio", type=float, default=0.02)
    args = parser.parse_args()
    for name in (
        "scene_list",
        "base_track_root",
        "prompt_track_root",
        "native_prediction_cache",
        "processed_scene_root",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes 必须为正数")
    if args.adjacency_knn <= 0 or args.adjacency_max_distance <= 0:
        raise SystemExit("邻接参数必须为正数")
    if args.min_contact_points <= 0 or args.min_contact_ratio < 0:
        raise SystemExit("接触参数不合法")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    scenes = _read_scenes(args.scene_list)
    args.native_cache_contract = _native_cache_contract(
        args.native_prediction_cache, args.expected_cache_mode
    )
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if args.resume and existing.is_file():
            summary = json.loads(existing.read_text())
            print(
                f"[跳过已有] {index}/{len(scenes)} {scene_name}: "
                f"{summary['track_count']} 条轨迹",
                flush=True,
            )
        else:
            summary = _build_scene(scene_name, args)
            print(
                f"[场景完成] {index}/{len(scenes)} {scene_name}: "
                f"改变 {summary['changed_track_count']}/{summary['track_count']} 条，"
                f"互为最佳 {summary['mutual_best_prompt_native_count']} 条",
                flush=True,
            )
        summaries.append(summary)

    payload = _aggregate(summaries, args)
    (args.output_root / "native_relation_ledger_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps({
        key: payload[key]
        for key in (
            "scene_count",
            "track_count",
            "changed_track_count",
            "core_prompt_changed_track_count",
            "core_prompt_added_superpoint_count",
            "core_prompt_added_point_count",
            "mutual_best_prompt_native_count",
            "prompt_top_iou_improved_count",
            "prompt_top_iou_unchanged_count",
            "prompt_top_iou_worsened_count",
            "added_inside_any_native_ratio",
            "added_superpoint_reachable_core_ratio",
            "integrity",
        )
    }, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

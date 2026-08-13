#!/usr/bin/env python3
"""为 prompt 新增 superpoint 建立独立视角局部 GVC 与边界正反证据账本。

每个原子以共识 v0 为回退，只在排除轨迹成员/重观测帧和 prompt 帧后的 uniform30
帧中比较 ``v0`` 与 ``v0 + 单个新增 superpoint``。工具同时记录核心锚定自动
SAM mask 的支持/排除、原始 superpoint 接触边界及分数 1.0 Mask3D 覆盖关系。
全过程不读取 GT、类别或语义，不生成、筛选、排序或修改候选。
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_automatic_sam_track_growth_ledger import _raw_superpoint_context
from build_details_core_prompt_native_relation_ledger import (
    _load_tracks,
    _native_cache_contract,
    _read_scenes,
    native_overlap_vectors,
)
from refine_details_automatic_tracks_consensus import _superpoint_points
from refine_details_consensus_all_view_reobservation import (
    _load_observations,
    frame_match_metrics,
)
from score_details_consensus_track_out_holdout import (
    _best_frame_siou,
    _mask_is_visible,
    load_selected_frame_indices,
    visible_counts_for_frames,
)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def best_core_anchored_observation(
    core_superpoints, observations, min_visible_points
):
    """按核心覆盖、sIoU、纯度和固定 SAM 质量选择一条描述性观测。"""
    best = None
    best_rank = None
    for observation in observations:
        metrics = frame_match_metrics(
            core_superpoints,
            observation["lifted_superpoints"],
            observation["visible_counts"],
            min_visible_points,
        )
        rank = (
            float(metrics["core_coverage"]),
            float(metrics["siou"]),
            float(metrics["observation_purity"]),
            float(observation["quality"]),
            -int(observation["observation_id"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = (observation, metrics)
    return best


def atomic_holdout_evidence(
    base_superpoints,
    core_superpoints,
    added_superpoint_id,
    selected_frame_indices,
    excluded_frame_indices,
    observations_by_frame,
    visible_counts_by_frame,
    min_visible_points,
    support_siou,
    min_anchor_core_coverage,
):
    """在相同独立共同可见帧域比较 v0 与单原子生长变体。"""
    base = tuple(sorted(set(int(item) for item in base_superpoints)))
    core = tuple(sorted(set(int(item) for item in core_superpoints)))
    added = int(added_superpoint_id)
    grown = tuple(sorted(set(base) | {added}))
    excluded = set(int(item) for item in excluded_frame_indices)
    frames = []
    for frame_index in sorted(set(int(item) for item in selected_frame_indices)):
        if frame_index in excluded:
            continue
        visible_counts = visible_counts_by_frame[frame_index]
        if not _mask_is_visible(base, visible_counts, min_visible_points):
            continue
        if int(visible_counts.get(added, 0)) < int(min_visible_points):
            continue
        observations = observations_by_frame.get(frame_index, [])
        anchored = best_core_anchored_observation(
            core, observations, min_visible_points
        )
        row = {
            "frame_index": int(frame_index),
            "observation_count": int(len(observations)),
            "base_best_siou": float(_best_frame_siou(
                base, observations, min_visible_points
            )),
            "grown_best_siou": float(_best_frame_siou(
                grown, observations, min_visible_points
            )),
            "selected_core_observation_id": -1,
            "selected_core_coverage": 0.0,
            "selected_core_siou": 0.0,
            "selected_core_observation_purity": 0.0,
            "selected_added_inside_visible_ratio": 0.0,
            "anchor_reliable": False,
            "anchor_supports_added": False,
            "anchor_excludes_added": False,
        }
        if anchored is not None:
            observation, metrics = anchored
            visible_count = int(observation["visible_counts"].get(added, 0))
            inside_count = int(observation["inside_counts"].get(added, 0))
            reliable = float(metrics["core_coverage"]) >= float(
                min_anchor_core_coverage
            )
            supports = reliable and added in set(
                int(item) for item in observation["lifted_superpoints"]
            )
            row.update({
                "selected_core_observation_id": int(observation["observation_id"]),
                "selected_core_coverage": float(metrics["core_coverage"]),
                "selected_core_siou": float(metrics["siou"]),
                "selected_core_observation_purity": float(
                    metrics["observation_purity"]
                ),
                "selected_added_inside_visible_ratio": float(
                    inside_count / max(1, visible_count)
                ),
                "anchor_reliable": bool(reliable),
                "anchor_supports_added": bool(supports),
                "anchor_excludes_added": bool(reliable and not supports),
            })
        frames.append(row)

    base_scores = np.asarray(
        [row["base_best_siou"] for row in frames], dtype=np.float64
    )
    grown_scores = np.asarray(
        [row["grown_best_siou"] for row in frames], dtype=np.float64
    )
    reliable_count = sum(row["anchor_reliable"] for row in frames)
    positive_count = sum(row["anchor_supports_added"] for row in frames)
    exclusion_count = sum(row["anchor_excludes_added"] for row in frames)
    base_mean = float(base_scores.mean()) if len(base_scores) else 0.0
    grown_mean = float(grown_scores.mean()) if len(grown_scores) else 0.0
    base_support_rate = (
        float(np.mean(base_scores >= float(support_siou))) if len(base_scores) else 0.0
    )
    grown_support_rate = (
        float(np.mean(grown_scores >= float(support_siou))) if len(grown_scores) else 0.0
    )
    return {
        "excluded_frame_count": int(len(excluded)),
        "joint_visible_independent_frame_count": int(len(frames)),
        "zero_observation_frame_count": int(sum(
            row["observation_count"] == 0 for row in frames
        )),
        "reliable_anchor_frame_count": int(reliable_count),
        "positive_anchor_frame_count": int(positive_count),
        "exclusion_anchor_frame_count": int(exclusion_count),
        "positive_anchor_rate": float(positive_count / max(1, reliable_count)),
        "exclusion_anchor_rate": float(exclusion_count / max(1, reliable_count)),
        "base_mean_best_siou": base_mean,
        "grown_mean_best_siou": grown_mean,
        "delta_mean_best_siou": float(grown_mean - base_mean),
        "base_support_frame_rate": base_support_rate,
        "grown_support_frame_rate": grown_support_rate,
        "delta_support_frame_rate": float(grown_support_rate - base_support_rate),
        "frames": frames,
    }


def graph_hops_from_core(core_superpoints, added_superpoints, neighbors):
    core = set(int(item) for item in core_superpoints)
    added = set(int(item) for item in added_superpoints)
    allowed = core | added
    distance = {item: 0 for item in core}
    queue = deque(sorted(core))
    while queue:
        current = queue.popleft()
        for edge in neighbors.get(current, []):
            target = int(edge["neighbor_superpoint_id"])
            if target in allowed and target not in distance:
                distance[target] = distance[current] + 1
                queue.append(target)
    return {item: distance.get(item) for item in added}


def boundary_relation(added_superpoint_id, core_superpoints, neighbors):
    core = set(int(item) for item in core_superpoints)
    edges = [
        edge for edge in neighbors.get(int(added_superpoint_id), [])
        if int(edge["neighbor_superpoint_id"]) in core
    ]
    contact_sum = sum(int(edge["boundary_contact_count"]) for edge in edges)
    if not edges:
        return {
            "direct_core_neighbor_count": 0,
            "direct_core_contact_count_sum": 0,
            "max_direct_core_contact_density": 0.0,
            "contact_weighted_boundary_distance": None,
            "contact_weighted_normal_difference": None,
            "contact_weighted_color_difference": None,
        }

    def weighted(field):
        return float(sum(
            float(edge[field]) * int(edge["boundary_contact_count"])
            for edge in edges
        ) / max(1, contact_sum))

    return {
        "direct_core_neighbor_count": int(len(edges)),
        "direct_core_contact_count_sum": int(contact_sum),
        "max_direct_core_contact_density": float(max(
            edge["boundary_contact_ratio"] for edge in edges
        )),
        "contact_weighted_boundary_distance": weighted("mean_boundary_distance"),
        "contact_weighted_normal_difference": weighted("mean_normal_difference"),
        "contact_weighted_color_difference": weighted("mean_color_difference"),
    }


def native_ownership(
    added_points, core_points, native_masks, native_scores
):
    added_points = np.unique(np.asarray(added_points, dtype=np.int64))
    core_vectors = native_overlap_vectors(core_points, native_masks)
    core_owner = int(np.argmax(core_vectors["iou"])) if native_masks.shape[1] else -1
    if core_owner >= 0 and float(core_vectors["iou"][core_owner]) <= 0.0:
        core_owner = -1
    score_one_ids = np.flatnonzero(
        np.abs(np.asarray(native_scores, dtype=np.float64) - 1.0) <= 1e-12
    )
    if len(added_points):
        added_coverage = native_masks[added_points].mean(axis=0)
    else:
        added_coverage = np.zeros(native_masks.shape[1], dtype=np.float64)
    core_owner_ratio = (
        float(added_coverage[core_owner]) if core_owner >= 0 else None
    )
    other_score_one = score_one_ids[score_one_ids != core_owner]
    if len(other_score_one):
        best_index = int(np.argmax(added_coverage[other_score_one]))
        best_other_id = int(other_score_one[best_index])
        best_other_ratio = float(added_coverage[best_other_id])
        if best_other_ratio <= 0.0:
            best_other_id = -1
    else:
        best_other_id, best_other_ratio = -1, 0.0
    return {
        "core_best_native_candidate_id": core_owner,
        "core_best_native_iou": (
            float(core_vectors["iou"][core_owner]) if core_owner >= 0 else 0.0
        ),
        "added_inside_core_best_native_ratio": core_owner_ratio,
        "score_one_native_candidate_count": int(len(score_one_ids)),
        "best_other_score_one_native_candidate_id": best_other_id,
        "best_other_score_one_native_coverage_ratio": best_other_ratio,
        "score_one_native_touching_added_count": int(sum(
            added_coverage[candidate_id] > 0.0 for candidate_id in score_one_ids
        )),
    }


def _load_plan_map(root, scene_name):
    rows = json.loads((root / scene_name / "core_prompt_plan.json").read_text())
    mapped = {int(row["track_id"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"{scene_name} 的 prompt plan 含重复 track_id")
    return mapped


def _load_prompt_support_map(root, scene_name):
    rows = json.loads((root / scene_name / "prompt_track_ledger.json").read_text())
    output = {}
    for row in rows:
        track_id = int(row["track_id"])
        if track_id in output:
            raise ValueError(f"{scene_name} 的 prompt ledger 含重复 track_id")
        output[track_id] = {
            int(item["superpoint_id"]): int(item["support_frame_count"])
            for item in row.get("new_superpoint_support_all_hypotheses", [])
        }
    return output


def _scene(scene_name, args):
    from utils import WORLD_2_CAM

    processed_path = (
        args.processed_scene_root / scene_name / f"{scene_name.replace('scene', '')}.npy"
    )
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10:
        raise ValueError(f"{scene_name} 缺少原始 superpoint 列")
    superpoints = np.asarray(processed[:, 9], dtype=np.int64)
    points_by_superpoint = _superpoint_points(superpoints)
    superpoint_sizes = {
        superpoint_id: int(len(points))
        for superpoint_id, points in points_by_superpoint.items()
    }
    v0_tracks = _load_tracks(args.v0_track_root, scene_name)
    prompt_tracks = _load_tracks(args.prompt_track_root, scene_name)
    if set(v0_tracks) != set(prompt_tracks):
        raise ValueError(f"{scene_name} 的 v0/prompt track_id 不一致")
    plans = _load_plan_map(args.prompt_plan_root, scene_name)
    prompt_support = _load_prompt_support_map(args.prompt_ledger_root, scene_name)

    native_masks = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_masks.npy", mmap_mode="r"
    )
    native_scores = np.load(
        args.native_prediction_cache / f"{scene_name}_pred_scores.npy", mmap_mode="r"
    )
    if native_masks.shape[0] != len(processed) and native_masks.shape[1] == len(processed):
        native_masks = native_masks.T
    if native_masks.shape != (len(processed), len(native_scores)):
        raise ValueError(f"{scene_name} 的 native 缓存维度不一致")
    native_masks = np.asarray(native_masks, dtype=bool)

    automatic_scene_root = args.automatic_root / scene_name
    selected_frames = load_selected_frame_indices(automatic_scene_root)
    world = WORLD_2_CAM(str(args.dataset_root / scene_name), args.depth_scale, args.config)
    _, raw_visibility = world.get_mesh_projections()
    visibility = raw_visibility.detach().cpu().numpy().astype(bool, copy=False)
    visible_counts = visible_counts_for_frames(
        superpoints, visibility, selected_frames
    )
    _, observations_by_frame = _load_observations(
        automatic_scene_root,
        superpoints,
        superpoint_sizes,
        visibility,
        args,
    )
    context = _raw_superpoint_context(
        processed,
        args.adjacency_knn,
        args.adjacency_max_distance,
        args.min_contact_points,
        args.min_contact_ratio,
    )

    rows = []
    changed_track_count = 0
    for track_id in sorted(prompt_tracks):
        prompt = prompt_tracks[track_id]
        added_ids = sorted(set(
            int(item) for item in prompt.get("core_prompt_added_superpoint_ids", [])
        ))
        if not added_ids:
            continue
        changed_track_count += 1
        if track_id not in plans or track_id not in prompt_support:
            raise ValueError(f"{scene_name} track {track_id} 缺少 prompt 计划或支持账本")
        plan = plans[track_id]
        v0_ids = sorted(set(
            int(item) for item in v0_tracks[track_id].get("superpoint_ids", [])
        ))
        if set(added_ids) & set(v0_ids):
            raise ValueError(f"{scene_name} track {track_id} 的 prompt 新增原子已在 v0")
        core_ids = sorted(set(int(item) for item in plan["common_core_superpoint_ids"]))
        if not core_ids or not set(core_ids).issubset(set(v0_ids)):
            raise ValueError(f"{scene_name} track {track_id} 的共同核心不属于 v0")
        prompt_frames = sorted(
            int(row["frame_index"]) for row in plan["prompt_frames"]
        )
        excluded_frames = sorted(
            set(int(item) for item in plan["excluded_frame_indices"]) | set(prompt_frames)
        )
        core_points = np.concatenate(
            [points_by_superpoint[item] for item in core_ids]
        ).astype(np.int64, copy=False)
        hops = graph_hops_from_core(core_ids, added_ids, context["neighbors"])
        for added_id in added_ids:
            if added_id not in points_by_superpoint:
                raise ValueError(f"{scene_name} track {track_id} 引用未知新增 SP {added_id}")
            support_count = prompt_support[track_id].get(added_id)
            if support_count is None or support_count < 2:
                raise ValueError(
                    f"{scene_name} track {track_id} SP {added_id} 不满足固定 prompt 支持合同"
                )
            evidence = atomic_holdout_evidence(
                v0_ids,
                core_ids,
                added_id,
                selected_frames,
                excluded_frames,
                observations_by_frame,
                visible_counts,
                args.min_visible_points_per_superpoint,
                args.support_siou,
                args.min_anchor_core_coverage,
            )
            rows.append({
                "scene_name": scene_name,
                "track_id": int(track_id),
                "added_superpoint_id": int(added_id),
                "added_point_count": int(superpoint_sizes[added_id]),
                "prompt_support_frame_count": int(support_count),
                "v0_superpoint_count": int(len(v0_ids)),
                "common_core_superpoint_count": int(len(core_ids)),
                "original_track_frame_count": int(len(plan["excluded_frame_indices"])),
                "prompt_frame_count": int(len(prompt_frames)),
                "selected_uniform_frame_count": int(len(selected_frames)),
                "reachable_from_core_via_prompt_added": hops[added_id] is not None,
                "graph_hops_from_core": hops[added_id],
                **boundary_relation(added_id, core_ids, context["neighbors"]),
                **native_ownership(
                    points_by_superpoint[added_id],
                    core_points,
                    native_masks,
                    native_scores,
                ),
                **evidence,
                "decision_state": (
                    "无GT原子级描述账本；不接受/拒绝superpoint，不生成或修改候选。"
                ),
            })

    scene_summary = {
        "scene_name": scene_name,
        "changed_track_count": int(changed_track_count),
        "added_superpoint_count": int(len(rows)),
        "added_point_count": int(sum(row["added_point_count"] for row in rows)),
        "with_independent_joint_visible_frame_count": int(sum(
            row["joint_visible_independent_frame_count"] > 0 for row in rows
        )),
        "with_reliable_anchor_frame_count": int(sum(
            row["reliable_anchor_frame_count"] > 0 for row in rows
        )),
        "with_positive_anchor_count": int(sum(
            row["positive_anchor_frame_count"] > 0 for row in rows
        )),
        "with_exclusion_anchor_count": int(sum(
            row["exclusion_anchor_frame_count"] > 0 for row in rows
        )),
        "delta_gvc_positive_count": int(sum(
            row["delta_mean_best_siou"] > 1e-12 for row in rows
        )),
        "delta_gvc_negative_count": int(sum(
            row["delta_mean_best_siou"] < -1e-12 for row in rows
        )),
    }
    staging = args.output_root / f".{scene_name}.writing"
    published = args.output_root / scene_name
    if staging.exists() or published.exists():
        raise FileExistsError(f"输出或临时目录已存在：{published}")
    staging.mkdir(parents=True)
    (staging / "added_superpoint_boundary_ledger.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (staging / "summary.json").write_text(
        json.dumps(scene_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging, published)
    del world, raw_visibility, visibility, processed, native_masks, context
    return scene_summary


def _aggregate(summaries, args):
    fields = (
        "changed_track_count",
        "added_superpoint_count",
        "added_point_count",
        "with_independent_joint_visible_frame_count",
        "with_reliable_anchor_frame_count",
        "with_positive_anchor_count",
        "with_exclusion_anchor_count",
        "delta_gvc_positive_count",
        "delta_gvc_negative_count",
    )
    return {
        "gt_usage": "none；不读取GT、类别或语义，不运行AP。",
        "decision_state": (
            "固定prompt新增superpoint的独立视角局部GVC、边界和native所有权描述账本；"
            "不生成、筛选、排序或修改候选。"
        ),
        "scene_count": int(len(summaries)),
        **{field: int(sum(item[field] for item in summaries)) for field in fields},
        "native_cache_contract": args.native_cache_contract,
        "integrity": {
            "only_declared_prompt_added_superpoints": True,
            "v0_and_prompt_files_unchanged": True,
            "track_and_prompt_frames_excluded": True,
            "single_added_superpoint_comparisons": True,
            "no_candidate_actions": True,
        },
        "scenes": summaries,
        "params": {
            key: value for key, value in vars(args).items() if key != "config"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--v0-track-root", type=Path, required=True)
    parser.add_argument("--prompt-track-root", type=Path, required=True)
    parser.add_argument("--prompt-plan-root", type=Path, required=True)
    parser.add_argument("--prompt-ledger-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument(
        "--processed-scene-root", type=Path, default=Path("data/scannet200")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument(
        "--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-superpoint-visible-ratio", type=float, default=0.10)
    parser.add_argument("--min-initial-mask-support", type=float, default=0.30)
    parser.add_argument("--min-visible-points-per-superpoint", type=int, default=3)
    parser.add_argument("--support-siou", type=float, default=0.30)
    parser.add_argument("--min-anchor-core-coverage", type=float, default=0.30)
    parser.add_argument("--adjacency-knn", type=int, default=12)
    parser.add_argument("--adjacency-max-distance", type=float, default=0.05)
    parser.add_argument("--min-contact-points", type=int, default=3)
    parser.add_argument("--min-contact-ratio", type=float, default=0.02)
    args = parser.parse_args()
    for name in (
        "scene_list",
        "v0_track_root",
        "prompt_track_root",
        "prompt_plan_root",
        "prompt_ledger_root",
        "automatic_root",
        "native_prediction_cache",
        "processed_scene_root",
        "dataset_root",
        "config_path",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise SystemExit("--max-scenes 必须为正数")
    if not 0.0 <= args.min_anchor_core_coverage <= 1.0:
        raise SystemExit("--min-anchor-core-coverage 必须在零到一之间")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录已存在且非空，为避免覆盖已拒绝执行：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    args.native_cache_contract = _native_cache_contract(
        args.native_prediction_cache, "mask3d_yoloworld_only"
    )
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, scene_name in enumerate(scenes, start=1):
        existing = args.output_root / scene_name / "summary.json"
        if args.resume and existing.is_file():
            summary = json.loads(existing.read_text())
        else:
            summary = _scene(scene_name, args)
        summaries.append(summary)
        print(
            f"[场景完成] {index}/{len(scenes)} {scene_name}: "
            f"{summary['added_superpoint_count']} 个新增SP，"
            f"{summary['with_reliable_anchor_frame_count']} 个有独立锚定证据",
            flush=True,
        )
    payload = _aggregate(summaries, args)
    (args.output_root / "added_superpoint_boundary_ledger_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    print(json.dumps({
        key: payload[key]
        for key in (
            "scene_count",
            "changed_track_count",
            "added_superpoint_count",
            "with_independent_joint_visible_frame_count",
            "with_reliable_anchor_frame_count",
            "with_positive_anchor_count",
            "with_exclusion_anchor_count",
            "delta_gvc_positive_count",
            "delta_gvc_negative_count",
            "integrity",
        )
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

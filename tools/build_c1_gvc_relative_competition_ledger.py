#!/usr/bin/env python3
"""Build no-GT public-view relative GVC evidence for frozen track--native pairs.

Only positive-overlap pairs from the frozen C1 relation ledger are compared.
For each pair, every D2b source frame is removed, then both candidates are
evaluated on exactly the same remaining depth-consistent views.  The common
views are selected by the smaller of the two visible-point counts.  Pure-track
components and pairs without a common eligible view are recorded as missing
comparison rather than assigned a fabricated native score.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_c1_gvc_paper_reference_quality_ledger import (
    _candidate_view_rows,
    _frame_contract,
    _load_2d_observations,
    _load_tracks,
    _read_scenes,
    _resolve,
    _summary,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _public_track_frames(track: dict, frame_id_to_index: dict[str, int]) -> tuple[set[int], int]:
    raw = {str(value) for value in track.get("frame_ids", [])}
    resolved = {frame_id_to_index[value] for value in raw if value in frame_id_to_index}
    return resolved, len(raw - set(frame_id_to_index))


def _pair_metrics(track_rows: dict[int, dict], native_rows: dict[int, dict], max_views: int) -> dict:
    common = sorted(set(track_rows) & set(native_rows))
    selected_ids = sorted(
        common,
        key=lambda frame: (-min(track_rows[frame]["visible_point_count"], native_rows[frame]["visible_point_count"]), frame),
    )[:max_views]
    selected = [(track_rows[frame], native_rows[frame]) for frame in selected_ids]
    track_gvc = _summary([left["gvc_frame_score"] for left, _ in selected])
    native_gvc = _summary([right["gvc_frame_score"] for _, right in selected])
    track_box = _summary([left["box_iou"] for left, _ in selected])
    native_box = _summary([right["box_iou"] for _, right in selected])
    track_support = _summary([left["mask_point_support"] for left, _ in selected])
    native_support = _summary([right["mask_point_support"] for _, right in selected])
    native_mean = native_gvc["mean"]
    return {
        "common_depth_consistent_view_count": len(common),
        "selected_public_common_view_count": len(selected),
        "selected_public_common_frames": selected_ids,
        "track_gvc_public_common": track_gvc,
        "native_gvc_public_common": native_gvc,
        "track_minus_native_gvc": float(track_gvc["mean"] - native_mean),
        "track_over_native_gvc_ratio": (float(track_gvc["mean"] / native_mean) if native_mean > 0.0 else None),
        "native_gvc_zero_denominator": native_mean == 0.0,
        "track_minus_native_box_iou": float(track_box["mean"] - native_box["mean"]),
        "track_minus_native_mask_support": float(track_support["mean"] - native_support["mean"]),
        "track_box_iou_public_common": track_box,
        "native_box_iou_public_common": native_box,
        "track_mask_support_public_common": track_support,
        "native_mask_support_public_common": native_support,
    }


def _scene(scene: str, args) -> tuple[list[dict], list[dict], dict]:
    from utils import WORLD_2_CAM

    relations = _jsonl(args.relation_ledger_root / scene / "track_native_relations.jsonl")
    components = _jsonl(args.relation_ledger_root / scene / "relation_components.jsonl")
    pairs = {(int(row["proposal_id"]), int(row["native_candidate_id"])): row for row in relations}
    if len(pairs) != len(relations):
        raise ValueError(f"{scene}: track--native relation 重复")
    tracks_needed = {track for track, _ in pairs}
    natives_needed = {native for _, native in pairs}
    automatic_scene = args.automatic_root / scene
    frame_indices, frame_id_to_index = _frame_contract(automatic_scene)
    observations = _load_2d_observations(args.yoloworld_sam_root / scene)
    track_map = {int(row["track_id"]): row for row in _load_tracks(args.track_root / scene)}
    if not tracks_needed <= set(track_map):
        raise ValueError(f"{scene}: relation track 不在冻结 D2b 根中")
    masks = np.load(args.native_prediction_cache / f"{scene}_pred_masks.npy", mmap_mode="r")
    if masks.ndim != 2 or not natives_needed <= set(range(masks.shape[1])):
        raise ValueError(f"{scene}: relation native 不在冻结缓存中")
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    projections_t, visibility_t = world.get_mesh_projections()
    projections = projections_t.detach().cpu().numpy().astype(np.float64)
    visibility = visibility_t.detach().cpu().numpy().astype(bool)
    if masks.shape[0] != visibility.shape[1]:
        raise ValueError(f"{scene}: native 与可见性点数不一致")
    scaling = (world.depth_resolution[0] / world.image_resolution[0], world.depth_resolution[1] / world.image_resolution[1])
    track_views, native_views, track_source = {}, {}, {}
    for track_id in sorted(tracks_needed):
        track = track_map[track_id]
        points = np.unique(np.asarray(np.load(track["points_path"])["point_indices"], dtype=np.int64))
        points = points[(points >= 0) & (points < visibility.shape[1])]
        excluded, unresolved = _public_track_frames(track, frame_id_to_index)
        rows = _candidate_view_rows(points, frame_indices, projections, visibility, scaling, observations, args.min_visible_points)
        track_views[track_id] = {row["frame_index"]: row for row in rows if row["frame_index"] not in excluded}
        track_source[track_id] = {"source_frame_count": len(excluded), "unresolved_source_frame_id_count": unresolved}
    for native_id in sorted(natives_needed):
        points = np.flatnonzero(masks[:, native_id]).astype(np.int64)
        rows = _candidate_view_rows(points, frame_indices, projections, visibility, scaling, observations, args.min_visible_points)
        native_views[native_id] = {row["frame_index"]: row for row in rows}
    pair_rows = []
    per_track: dict[int, list[dict]] = defaultdict(list)
    for (track_id, native_id), relation in sorted(pairs.items()):
        metrics = _pair_metrics(track_views[track_id], native_views[native_id], args.max_views)
        row = {
            "scene_name": scene, "track_id": track_id, "native_candidate_id": native_id,
            "component_id": int(relation["component_id"]), "point_iou": float(relation["point_iou"]),
            "track_inside_native_ratio": float(relation["track_inside_native_ratio"]),
            "native_inside_track_ratio": float(relation["native_inside_track_ratio"]),
            **track_source[track_id], **metrics,
            "ground_truth_usage": "none", "proposal_materialization_applied": False,
        }
        pair_rows.append(row)
        per_track[track_id].append(row)
    track_component = {}
    for component in components:
        for track_id in component["track_ids"]:
            track_component[int(track_id)] = int(component["component_id"])
    track_rows = []
    for track_id in sorted(track_map):
        rows = per_track.get(track_id, [])
        valid = [row for row in rows if row["selected_public_common_view_count"] > 0]
        if not rows:
            state, comparator = "no_overlapping_native", None
        elif not valid:
            state, comparator = "overlap_without_public_common_view", None
        else:
            state = "relative_public_evidence_available"
            comparator = max(valid, key=lambda row: (row["native_gvc_public_common"]["mean"], -row["native_candidate_id"]))
        summary = {
            "scene_name": scene, "track_id": track_id, "component_id": track_component[track_id],
            "comparison_state": state, "overlapping_native_count": len(rows),
            "public_comparison_native_candidate_id": None if comparator is None else comparator["native_candidate_id"],
            "public_comparison_pair": None if comparator is None else {
                key: comparator[key] for key in (
                    "selected_public_common_view_count", "track_gvc_public_common", "native_gvc_public_common",
                    "track_minus_native_gvc", "track_over_native_gvc_ratio", "native_gvc_zero_denominator",
                    "track_minus_native_box_iou", "track_minus_native_mask_support",
                )
            },
            **track_source.get(track_id, {"source_frame_count": 0, "unresolved_source_frame_id_count": 0}),
            "ground_truth_usage": "none", "proposal_materialization_applied": False,
        }
        track_rows.append(summary)
    summary = {
        "scene_name": scene, "relation_pair_count": len(pair_rows), "relative_track_count": len(track_rows),
        "with_public_relative_evidence_count": sum(row["comparison_state"] == "relative_public_evidence_available" for row in track_rows),
    }
    del world, projections_t, visibility_t, projections, visibility, masks
    return pair_rows, track_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--relation-ledger-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--native-prediction-cache", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--yoloworld-sam-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannet200"))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=10)
    parser.add_argument("--min-visible-points", type=int, default=30)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("scene_list", "relation_ledger_root", "track_root", "native_prediction_cache", "automatic_root", "yoloworld_sam_root", "dataset_root", "config_path", "output_root"):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.max_views < 1 or args.min_visible_points < 1:
        raise SystemExit("--max-views 与 --min-visible-points 必须为正数")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录非空，拒绝覆盖：{args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    summaries = []
    for index, scene in enumerate(scenes, start=1):
        scene_root = args.output_root / scene
        if (scene_root / "track_relative_gvc.jsonl").is_file():
            if not args.resume:
                raise SystemExit(f"输出场景已存在：{scene}")
            summaries.append(json.loads((scene_root / "summary.json").read_text()))
            print(f"[跳过已有] {index}/{len(scenes)} {scene}", flush=True)
            continue
        pairs, tracks, summary = _scene(scene, args)
        scene_root.mkdir()
        (scene_root / "pair_relative_gvc.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs))
        (scene_root / "track_relative_gvc.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in tracks))
        (scene_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summaries.append(summary)
        print(f"[场景完成] {index}/{len(scenes)} {scene}: {summary['relation_pair_count']} 对", flush=True)
    payload = {
        "diagnostic_type": "no-GT C1 public-view relative track-native GVC ledger",
        "ground_truth_usage": "none", "proposal_materialization_applied": False,
        "track_source_frames_excluded": True, "native_score_fabrication_applied": False,
        "scene_count": len(summaries),
        "relation_pair_count": sum(row["relation_pair_count"] for row in summaries),
        "track_count": sum(row["relative_track_count"] for row in summaries),
        "with_public_relative_evidence_count": sum(row["with_public_relative_evidence_count"] for row in summaries),
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "config"},
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

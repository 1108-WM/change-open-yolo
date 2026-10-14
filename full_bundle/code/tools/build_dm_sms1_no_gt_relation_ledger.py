#!/usr/bin/env python3
"""Build DM-SMS-1 frozen candidate relation features without GT or AP."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fi1_superpoint_geometry import _raw_superpoint_context
from tools.build_c1_gvc_paper_reference_quality_ledger import _frame_contract, _load_2d_observations
from tools.fi1_shared.build_safety60_structured_action_plan import _positive_pairs
from tools.fi1_shared.build_train_candidate_relation_feature_ledger import (
    ADJACENCY_KNN,
    ADJACENCY_MAX_DISTANCE,
    MIN_CONTACT_POINTS,
    MIN_CONTACT_RATIO,
    _candidate_view_rows_with_box,
    build_inference_relation_rows,
)
from tools.fi1_shared.evaluate_candidate_quality_reranking_class_agnostic_ap import load_safety_scene


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scenes(path: Path) -> list[str]:
    result = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or contains duplicates")
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))


def _scene(scene: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    from utils import WORLD_2_CAM

    masks, scores, _classes, _feature_rows, tracks, candidate_audit = load_safety_scene(
        scene, args.native_cache, args.quality_ledger_root, args.track_root,
    )
    pairs, track_points, native_points, _point_paths = _positive_pairs(
        scene, masks, scores, tracks
    )
    scene_stem = scene.removeprefix("scene")
    processed_path = args.dataset_root / scene / f"{scene_stem}.npy"
    processed = np.load(processed_path, mmap_mode="r")
    if processed.ndim != 2 or processed.shape[1] < 10 or len(processed) != masks.shape[0]:
        raise ValueError(f"{scene}: processed scene contract differs from candidates")
    context = _raw_superpoint_context(
        processed, ADJACENCY_KNN, ADJACENCY_MAX_DISTANCE,
        MIN_CONTACT_POINTS, MIN_CONTACT_RATIO,
    )
    automatic_scene = args.automatic_root / scene
    frame_indices, frame_id_to_index = _frame_contract(automatic_scene)
    observations = _load_2d_observations(args.yoloworld_sam_root / scene)
    world = WORLD_2_CAM(str(args.dataset_root / scene), args.depth_scale, args.config)
    projections_t, visibility_t = world.get_mesh_projections()
    projections = projections_t.detach().cpu().numpy().astype(np.float64)
    visibility = visibility_t.detach().cpu().numpy().astype(bool)
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    camera_centers = {
        frame: np.asarray(np.loadtxt(world.poses[frame]), dtype=np.float64)[:3, 3]
        for frame in frame_indices
    }
    track_by_id = {int(row["track_id"]): row for row in tracks}
    track_views = {}
    for track_id, points in track_points.items():
        raw_frames = {str(value) for value in track_by_id[track_id].get("frame_ids", [])}
        source_frames = {
            frame_id_to_index[value] for value in raw_frames if value in frame_id_to_index
        }
        all_rows = _candidate_view_rows_with_box(
            points, frame_indices, projections, visibility, scaling, observations,
        )
        track_views[track_id] = {
            frame: row for frame, row in all_rows.items() if frame not in source_frames
        }
    native_views = {
        group_id: _candidate_view_rows_with_box(
            points, frame_indices, projections, visibility, scaling, observations,
        )
        for group_id, points in native_points.items()
    }
    relation_rows, components, relation_summary = build_inference_relation_rows(
        scene, pairs, processed, context, track_points, native_points,
        track_views, native_views, camera_centers,
    )
    for row in relation_rows:
        if row.get("contracts", {}).get("feature_ground_truth_usage") != "none":
            raise ValueError(f"{scene}: relation row is not GT-free")
    summary = {
        "scene_name": scene,
        **relation_summary,
        "native_candidate_count": int(masks.shape[1]),
        "track_candidate_count": len(tracks),
        "relation_component_count": len(components),
        "candidate_audit": candidate_audit,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_evaluation_run": False,
        "candidate_mutation": False,
    }
    del world, projections_t, visibility_t, projections, visibility, processed, masks
    return relation_rows, summary


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "native_cache", "quality_ledger_root", "track_root",
        "automatic_root", "yoloworld_sam_root", "dataset_root", "config_path",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"scene count {len(scenes)} differs from expected {args.expected_scene_count}"
        )
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"output root is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    relation_count = 0
    component_count = 0
    try:
        for index, scene in enumerate(scenes, 1):
            scene_root = args.output_root / scene
            record_path = scene_root / "relation_features_no_gt.jsonl"
            summary_path = scene_root / "summary.json"
            if args.resume and record_path.is_file() and summary_path.is_file():
                rows = [
                    json.loads(line) for line in record_path.read_text().splitlines()
                    if line.strip()
                ]
                summary = json.loads(summary_path.read_text())
                if len(rows) != int(summary.get("relation_count", -1)):
                    raise ValueError(f"{scene}: relation resume count mismatch")
                print(
                    f"[DM-SMS-1 relations skip] {index}/{len(scenes)} {scene}: "
                    f"relations={len(rows)}",
                    flush=True,
                )
            else:
                rows, summary = _scene(scene, args)
                scene_root.mkdir(exist_ok=False)
                _write_jsonl(record_path, rows)
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                )
            summaries.append(summary)
            relation_count += len(rows)
            component_count += int(summary["relation_component_count"])
            print(
                f"[DM-SMS-1 relations] {index}/{len(scenes)} {scene}: "
                f"relations={len(rows)} components={summary['relation_component_count']}",
                flush=True,
            )
        output = {
            "version": "dm_sms1_no_gt_relation_ledger_v1",
            "scene_count": len(scenes),
            "relation_count": relation_count,
            "relation_component_count": component_count,
            "ground_truth_usage": "none",
            "ground_truth_read": False,
            "ap_evaluation_run": False,
            "candidate_mutation": False,
            "feature_contract": "frozen official100 candidate relation inference features",
            "relation_kind_counts": dict(sorted(Counter(
                key for summary in summaries
                for key, count in summary.get("relation_kind_counts", {}).items()
                for _ in range(int(count))
            ).items())),
            "params": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key != "config"
            },
            "scene_summaries": summaries,
        }
        (args.output_root / "summary.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return output
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"),
    )
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--quality-ledger-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--automatic-root", type=Path, required=True)
    parser.add_argument("--yoloworld-sam-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"),
    )
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independently reproject and audit the DM-SMS-1 Alpha/SAM view manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_dm_sms1_unique_geometry_ledger import GeometryResolver
from tools.build_dm_sms1_alpha_view_manifest import (
    SCALE_EXPANSIONS,
    select_top_visible_views,
    square_crop_box,
)
from tools.dm_sms_core import geometry_hash


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenes(path: Path) -> list[str]:
    result = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or contains duplicates")
    return result


def _audit_scene(
    scene: str, manifest_rows: list[dict], ledger_by_hash: dict[str, dict],
    args: argparse.Namespace, resolver: GeometryResolver,
) -> dict:
    from utils import WORLD_2_CAM

    world = WORLD_2_CAM(str(args.prepared_root / scene), args.depth_scale, args.config)
    projections_t, visibility_t = world.get_mesh_projections()
    projections = projections_t.detach().cpu().numpy().astype(np.float64)
    visibility = visibility_t.detach().cpu().numpy().astype(bool)
    frame_ids = [Path(path).stem for path in world.color_paths]
    image_height, image_width = map(int, world.image_resolution)
    scaling = (
        float(world.depth_resolution[0]) / image_height,
        float(world.depth_resolution[1]) / image_width,
    )
    audited_views = 0
    audited_scales = 0
    no_view_count = 0
    for row in manifest_rows:
        digest = str(row["geometry_hash"])
        ledger = ledger_by_hash.get(digest)
        if ledger is None or str(ledger["geometry_key"]) != str(row["geometry_key"]):
            raise ValueError(f"{scene}/{digest}: manifest-to-ledger join mismatch")
        if row.get("ground_truth_usage") != "none" or row.get("embedding_computed") is not False:
            raise ValueError(f"{scene}/{digest}: manifest row violates no-GT/no-embedding contract")
        points = resolver.points(ledger["canonical_geometry_locator"])
        if len(points) != int(row["point_count"]) or geometry_hash(points) != digest:
            raise ValueError(f"{scene}/{digest}: geometry locator mismatch")
        counts = visibility[:, points].sum(axis=1, dtype=np.int64)
        selected = select_top_visible_views(counts, frame_ids, args.max_views)
        views = list(row["views"])
        if int(row["eligible_visible_view_count"]) != int(np.count_nonzero(counts > 0)):
            raise ValueError(f"{scene}/{digest}: eligible view count mismatch")
        if int(row["selected_view_count"]) != len(selected) or len(views) != len(selected):
            raise ValueError(f"{scene}/{digest}: selected view count mismatch")
        no_view_count += int(not views)
        for rank, (view, frame_index) in enumerate(zip(views, selected)):
            frame_id = frame_ids[frame_index]
            if (
                int(view["view_rank"]) != rank
                or int(view["frame_index"]) != frame_index
                or str(view["frame_id"]) != frame_id
            ):
                raise ValueError(f"{scene}/{digest}: top-view order mismatch")
            visible_points = points[visibility[frame_index, points]]
            if int(view["visible_point_count"]) != len(visible_points):
                raise ValueError(f"{scene}/{digest}/{frame_id}: visible count mismatch")
            expected_ratio = float(len(visible_points) / len(points))
            if not np.isclose(float(view["visible_ratio"]), expected_ratio, rtol=0.0, atol=1e-12):
                raise ValueError(f"{scene}/{digest}/{frame_id}: visible ratio mismatch")
            coords = projections[frame_index, visible_points]
            xs = coords[:, 0] / scaling[1]
            ys = coords[:, 1] / scaling[0]
            expected_tight = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            if not np.allclose(view["tight_projected_bbox_xyxy"], expected_tight, rtol=0.0, atol=1e-9):
                raise ValueError(f"{scene}/{digest}/{frame_id}: tight bbox mismatch")
            if view["sam_box_prompt_xyxy"] != view["tight_projected_bbox_xyxy"]:
                raise ValueError(f"{scene}/{digest}/{frame_id}: SAM prompt differs from tight bbox")
            expected_assets = {
                "rgb_path": str(Path(world.color_paths[frame_index])),
                "depth_path": str(args.prepared_root / scene / "depth" / f"{frame_id}.png"),
                "pose_path": str(Path(world.poses[frame_index])),
                "intrinsics_path": str(Path(world.intrinsics[frame_index])),
            }
            for field, expected_path in expected_assets.items():
                if str(view[field]) != expected_path or not Path(expected_path).is_file():
                    raise ValueError(f"{scene}/{digest}/{frame_id}: {field} mismatch")
            crops = list(view["crop_scales"])
            if len(crops) != len(SCALE_EXPANSIONS):
                raise ValueError(f"{scene}/{digest}/{frame_id}: scale count mismatch")
            for scale_index, (crop, expansion) in enumerate(zip(crops, SCALE_EXPANSIONS)):
                square, integer = square_crop_box(
                    expected_tight, expansion, image_width, image_height
                )
                if (
                    int(crop["scale_index"]) != scale_index
                    or float(crop["bbox_expansion_fraction_per_side"]) != expansion
                    or not np.allclose(
                        crop["square_bbox_xyxy_float"], square, rtol=0.0, atol=1e-9
                    )
                    or list(crop["crop_xyxy_integer_exclusive"]) != integer
                ):
                    raise ValueError(f"{scene}/{digest}/{frame_id}: crop scale mismatch")
                audited_scales += 1
            audited_views += 1
    del world, projections_t, visibility_t, projections, visibility
    return {
        "scene_name": scene,
        "geometry_count": len(manifest_rows),
        "audited_view_count": audited_views,
        "audited_scale_count": audited_scales,
        "no_visible_view_geometry_count": no_view_count,
    }


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "manifest_root", "ledger_root", "prepared_root", "config_path",
        "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"scene count {len(scenes)} differs from expected {args.expected_scene_count}"
        )
    manifest_summary_path = args.manifest_root / "summary.json"
    manifest_path = args.manifest_root / "alpha_view_manifest.jsonl"
    manifest_summary = json.loads(manifest_summary_path.read_text())
    if (
        manifest_summary.get("version") != "dm_sms1_alpha_view_manifest_v1"
        or manifest_summary.get("manifest_valid") is not True
        or int(manifest_summary.get("scene_count", -1)) != len(scenes)
        or manifest_summary.get("ground_truth_read") is not False
        or manifest_summary.get("ap_computed") is not False
        or manifest_summary.get("embedding_computed") is not False
        or manifest_summary.get("sam_inference_computed") is not False
    ):
        raise ValueError("Alpha view manifest summary contract is invalid")
    ledger_summary = json.loads((args.ledger_root / "summary.json").read_text())
    ledger_rows = _read_jsonl(args.ledger_root / "unique_geometry_ledger.jsonl")
    manifest_rows = _read_jsonl(manifest_path)
    ledger_by_scene = defaultdict(dict)
    manifest_by_scene = defaultdict(list)
    for row in ledger_rows:
        ledger_by_scene[str(row["scene_name"])][str(row["geometry_hash"])] = row
    for row in manifest_rows:
        manifest_by_scene[str(row["scene_name"])].append(row)
    if set(ledger_by_scene) != set(scenes) or set(manifest_by_scene) != set(scenes):
        raise ValueError("scene coverage mismatch")
    manifest_order = [(row["scene_name"], row["geometry_hash"]) for row in manifest_rows]
    if manifest_order != sorted(manifest_order) or len(manifest_order) != len(set(manifest_order)):
        raise ValueError("manifest geometry order or uniqueness mismatch")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    resolver = GeometryResolver()
    summaries = []
    for index, scene in enumerate(scenes, 1):
        rows = sorted(manifest_by_scene[scene], key=lambda row: row["geometry_hash"])
        summary = _audit_scene(scene, rows, ledger_by_scene[scene], args, resolver)
        summaries.append(summary)
        print(
            f"[DM-SMS-1 Alpha audit] {index}/{len(scenes)} {scene}: "
            f"geometry={summary['geometry_count']} views={summary['audited_view_count']}",
            flush=True,
        )
    derived = {
        "scene_count": len(scenes),
        "geometry_count": sum(row["geometry_count"] for row in summaries),
        "selected_view_count": sum(row["audited_view_count"] for row in summaries),
        "crop_scale_count": sum(row["audited_scale_count"] for row in summaries),
        "no_visible_view_geometry_count": sum(
            row["no_visible_view_geometry_count"] for row in summaries
        ),
    }
    for key, value in derived.items():
        if int(manifest_summary.get(key, -1)) != value:
            raise ValueError(f"manifest aggregate mismatch for {key}")
    if derived["geometry_count"] != int(ledger_summary["unique_geometry_count"]):
        raise ValueError("manifest geometry count differs from unique ledger")
    args.output_root.mkdir(parents=True, exist_ok=False)
    output = {
        "version": "dm_sms1_alpha_view_manifest_audit_v1",
        "audit_valid": True,
        **derived,
        "duplicate_geometry_join_count": 0,
        "asset_missing_count": 0,
        "projection_error_count": 0,
        "view_order_error_count": 0,
        "bbox_error_count": 0,
        "crop_scale_error_count": 0,
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "embedding_computed": False,
        "sam_inference_computed": False,
        "candidate_mutation": False,
        "input_provenance": {
            "scene_list": str(args.scene_list),
            "scene_list_sha256": _sha256(args.scene_list),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "manifest_summary_sha256": _sha256(manifest_summary_path),
            "unique_ledger_sha256": _sha256(
                args.ledger_root / "unique_geometry_ledger.jsonl"
            ),
        },
        "scene_summaries": summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"),
    )
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path,
        default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"),
    )
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-views", type=int, default=20)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

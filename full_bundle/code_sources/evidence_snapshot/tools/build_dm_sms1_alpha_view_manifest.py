#!/usr/bin/env python3
"""Build the preregistered DM-SMS-1 Alpha/SAM view manifest without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_dm_sms1_unique_geometry_ledger import GeometryResolver
from tools.dm_sms_core import geometry_hash


SCALE_EXPANSIONS = (0.0, 0.2, 0.4)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenes(path: Path) -> list[str]:
    result = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or contains duplicates")
    return result


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_top_visible_views(
    visible_counts: np.ndarray, frame_ids: list[str], max_views: int,
) -> list[int]:
    counts = np.asarray(visible_counts, dtype=np.int64)
    if counts.shape != (len(frame_ids),):
        raise ValueError("visible counts and frame ids differ")
    if max_views <= 0:
        raise ValueError("max_views must be positive")
    eligible = [index for index, count in enumerate(counts) if int(count) > 0]
    return sorted(
        eligible,
        key=lambda index: (-int(counts[index]), int(index), str(frame_ids[index])),
    )[:max_views]


def square_crop_box(
    tight_xyxy: list[float], expansion: float, image_width: int, image_height: int,
) -> tuple[list[float], list[int]]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("invalid image size")
    if expansion < 0.0:
        raise ValueError("expansion must be nonnegative")
    x1, y1, x2, y2 = map(float, tight_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
        raise ValueError("invalid tight bbox")
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    x1 -= expansion * width
    x2 += expansion * width
    y1 -= expansion * height
    y2 += expansion * height
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, 1.0)
    square = [
        max(0.0, center_x - 0.5 * side),
        max(0.0, center_y - 0.5 * side),
        min(float(image_width - 1), center_x + 0.5 * side),
        min(float(image_height - 1), center_y + 0.5 * side),
    ]
    integer = [
        max(0, int(math.floor(square[0]))),
        max(0, int(math.floor(square[1]))),
        min(image_width, int(math.ceil(square[2])) + 1),
        min(image_height, int(math.ceil(square[3])) + 1),
    ]
    if integer[2] <= integer[0] or integer[3] <= integer[1]:
        raise ValueError("square crop became empty after clipping")
    return [float(value) for value in square], integer


def _asset(path: Path, kind: str, scene: str, frame_id: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{scene}/{frame_id}: missing {kind}: {path}")
    return str(path)


def _scene_rows(
    scene: str, ledger_rows: list[dict], args: argparse.Namespace,
    resolver: GeometryResolver,
) -> tuple[list[dict], dict]:
    from utils import WORLD_2_CAM

    world = WORLD_2_CAM(str(args.prepared_root / scene), args.depth_scale, args.config)
    projections_t, visibility_t = world.get_mesh_projections()
    projections = projections_t.detach().cpu().numpy().astype(np.float64)
    visibility = visibility_t.detach().cpu().numpy().astype(bool)
    frame_ids = [Path(path).stem for path in world.color_paths]
    if len(frame_ids) != projections.shape[0] or visibility.shape != projections.shape[:2]:
        raise ValueError(f"{scene}: frame/projection/visibility dimensions disagree")
    image_height, image_width = map(int, world.image_resolution)
    scaling = (
        float(world.depth_resolution[0]) / image_height,
        float(world.depth_resolution[1]) / image_width,
    )
    output = []
    view_count = 0
    scale_count = 0
    no_view_count = 0
    for row in ledger_rows:
        points = resolver.points(row["canonical_geometry_locator"])
        if len(points) != int(row["point_count"]) or geometry_hash(points) != row["geometry_hash"]:
            raise ValueError(f"{row['geometry_key']}: canonical locator differs from unique ledger")
        if points[-1] >= visibility.shape[1]:
            raise ValueError(f"{row['geometry_key']}: point index exceeds projection domain")
        counts = visibility[:, points].sum(axis=1, dtype=np.int64)
        selected = select_top_visible_views(counts, frame_ids, args.max_views)
        views = []
        for rank, frame_index in enumerate(selected):
            visible_points = points[visibility[frame_index, points]]
            coords = projections[frame_index, visible_points]
            xs = coords[:, 0] / scaling[1]
            ys = coords[:, 1] / scaling[0]
            valid = (
                np.isfinite(xs) & np.isfinite(ys)
                & (xs >= 0.0) & (xs <= image_width - 1)
                & (ys >= 0.0) & (ys <= image_height - 1)
            )
            if int(valid.sum()) != len(visible_points):
                raise ValueError(
                    f"{row['geometry_key']}/{frame_ids[frame_index]}: visible projection is invalid"
                )
            tight = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            crops = []
            for scale_index, expansion in enumerate(SCALE_EXPANSIONS):
                square, integer = square_crop_box(
                    tight, expansion, image_width, image_height
                )
                crops.append({
                    "scale_index": scale_index,
                    "bbox_expansion_fraction_per_side": expansion,
                    "square_bbox_xyxy_float": square,
                    "crop_xyxy_integer_exclusive": integer,
                })
            frame_id = frame_ids[frame_index]
            rgb_path = Path(world.color_paths[frame_index])
            depth_path = args.prepared_root / scene / "depth" / f"{frame_id}.png"
            pose_path = Path(world.poses[frame_index])
            intrinsics_path = Path(world.intrinsics[frame_index])
            views.append({
                "view_rank": rank,
                "frame_index": int(frame_index),
                "frame_id": frame_id,
                "visible_point_count": int(len(visible_points)),
                "visible_ratio": float(len(visible_points) / len(points)),
                "tight_projected_bbox_xyxy": tight,
                "sam_box_prompt_xyxy": tight,
                "sam_model_type": "vit_b",
                "sam_selection_contract": "highest predicted_iou; exact tie smallest mask index",
                "rgb_path": _asset(rgb_path, "RGB", scene, frame_id),
                "depth_path": _asset(depth_path, "depth", scene, frame_id),
                "pose_path": _asset(pose_path, "pose", scene, frame_id),
                "intrinsics_path": _asset(intrinsics_path, "intrinsics", scene, frame_id),
                "image_height": image_height,
                "image_width": image_width,
                "crop_scales": crops,
            })
            view_count += 1
            scale_count += len(crops)
        no_view_count += int(not views)
        output.append({
            "scene_name": scene,
            "geometry_hash": str(row["geometry_hash"]),
            "geometry_key": str(row["geometry_key"]),
            "point_count": int(row["point_count"]),
            "canonical_candidate_source": str(row["canonical_candidate_source"]),
            "canonical_candidate_id": int(row["canonical_candidate_id"]),
            "canonical_frozen_class_index": int(row["canonical_frozen_class_index"]),
            "canonical_frozen_class_valid": bool(row["canonical_frozen_class_valid"]),
            "canonical_frozen_score": float(row["canonical_frozen_score"]),
            "member_count": int(row.get("member_count", 1)),
            "members": [dict(member) for member in row.get("members", [])],
            "eligible_visible_view_count": int(np.count_nonzero(counts > 0)),
            "selected_view_count": len(views),
            "feature_missing_if_no_valid_sam_or_incomplete_scales": True,
            "views": views,
            "ground_truth_usage": "none",
            "embedding_computed": False,
            "sam_inference_computed": False,
        })
    del world, projections_t, visibility_t, projections, visibility
    return output, {
        "scene_name": scene,
        "geometry_count": len(output),
        "member_count": sum(int(row.get("member_count", 1)) for row in output),
        "selected_view_count": view_count,
        "crop_scale_count": scale_count,
        "no_visible_view_geometry_count": no_view_count,
    }


def run(args: argparse.Namespace) -> dict:
    for name in (
        "scene_list", "ledger_root", "prepared_root", "config_path", "output_root",
        "preregistration_path",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    scenes = _scenes(args.scene_list)
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError(
            f"scene count {len(scenes)} differs from expected {args.expected_scene_count}"
        )
    input_summary_path = args.ledger_root / "summary.json"
    input_summary = json.loads(input_summary_path.read_text())
    if (
        int(input_summary.get("scene_count", -1)) != len(scenes)
        or input_summary.get("contract_valid") is not True
        or input_summary.get("ground_truth_usage") != "none"
        or input_summary.get("ap_computed") is not False
    ):
        raise ValueError("unique geometry ledger contract is invalid")
    ledger_rows = _read_jsonl(args.ledger_root / "unique_geometry_ledger.jsonl")
    rows_by_scene = defaultdict(list)
    for row in ledger_rows:
        rows_by_scene[str(row["scene_name"])].append(row)
    if set(rows_by_scene) != set(scenes):
        raise ValueError("unique ledger scene coverage differs")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    args.output_root.mkdir(parents=True, exist_ok=False)
    resolver = GeometryResolver()
    summaries = []
    all_rows = []
    try:
        with (args.output_root / "alpha_view_manifest.jsonl").open("w") as output_file:
            for index, scene in enumerate(scenes, 1):
                rows, summary = _scene_rows(
                    scene, sorted(rows_by_scene[scene], key=lambda row: row["geometry_hash"]),
                    args, resolver,
                )
                for row in rows:
                    output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                all_rows.extend(rows)
                summaries.append(summary)
                print(
                    f"[DM-SMS-1 Alpha manifest] {index}/{len(scenes)} {scene}: "
                    f"geometry={len(rows)} views={summary['selected_view_count']}",
                    flush=True,
                )
        identities = [(row["scene_name"], row["geometry_hash"]) for row in all_rows]
        selected_view_count = sum(row["selected_view_count"] for row in summaries)
        scale_count = sum(row["crop_scale_count"] for row in summaries)
        output = {
            "version": "dm_sms1_alpha_view_manifest_v1",
            "scene_count": len(scenes),
            "geometry_count": len(all_rows),
            "member_count": sum(int(row.get("member_count", 1)) for row in all_rows),
            "selected_view_count": selected_view_count,
            "crop_scale_count": scale_count,
            "no_visible_view_geometry_count": sum(
                row["no_visible_view_geometry_count"] for row in summaries
            ),
            "duplicate_geometry_join_count": len(identities) - len(set(identities)),
            "asset_missing_count": 0,
            "projection_error_count": 0,
            "max_views": args.max_views,
            "scale_expansions": list(SCALE_EXPANSIONS),
            "view_order_contract": "visible point count descending; tie frame index then frame id",
            "crop_contract": "expand tight bbox per side; center-preserving square; clip to image",
            "sam_prompt_contract": "tight projected bbox; ViT-B; highest predicted-IoU mask",
            "aggregation_contract": "L2Norm(sum_v sum_l visible_ratio[v] * feature[v,l])",
            "manifest_valid": (
                len(all_rows) == int(input_summary["unique_geometry_count"])
                and sum(int(row.get("member_count", 1)) for row in all_rows)
                == int(input_summary["member_count"])
                and len(identities) == len(set(identities))
                and scale_count == 3 * selected_view_count
            ),
            "ground_truth_usage": "none",
            "ground_truth_read": False,
            "ap_computed": False,
            "embedding_computed": False,
            "sam_inference_computed": False,
            "candidate_mutation": False,
            "input_provenance": {
                "scene_list": str(args.scene_list),
                "scene_list_sha256": _sha256(args.scene_list),
                "unique_ledger": str(args.ledger_root / "unique_geometry_ledger.jsonl"),
                "unique_ledger_sha256": _sha256(
                    args.ledger_root / "unique_geometry_ledger.jsonl"
                ),
                "unique_ledger_summary_sha256": _sha256(input_summary_path),
                "prepared_root": str(args.prepared_root),
                "config_path": str(args.config_path),
                "config_sha256": _sha256(args.config_path),
                "preregistration_path": str(args.preregistration_path),
                "preregistration_sha256": _sha256(args.preregistration_path),
            },
            "scene_summaries": summaries,
        }
        if not output["manifest_valid"]:
            raise AssertionError("Alpha view manifest aggregate contract failed")
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
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path,
        default=Path("/media/jia/软件1/scannet_train_stream/prepared_ncs"),
    )
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preregistration-path", type=Path,
        default=Path("docs/DM_SMS1A_PREREGISTRATION_20260817.md"),
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-views", type=int, default=20)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

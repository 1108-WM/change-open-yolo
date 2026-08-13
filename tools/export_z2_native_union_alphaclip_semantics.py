#!/usr/bin/env python3
"""Export GT-free limited-context Alpha-CLIP for native and pair-union Z1 nodes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from export_automatic_track_alphaclip_semantics import (  # noqa: E402
    _finalize_record,
    _flush,
    _select_track_views,
)
from export_multiview_object_clip_features import (  # noqa: E402
    _load_alpha_clip,
    _make_crop_alpha_mask,
    _make_crop_image,
)


SOURCE_NAMES = ("native", "pair_union")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _geometry_hash(points: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(points, dtype=np.int64).tobytes()).hexdigest()


def _points(path: Path, point_count: int) -> np.ndarray:
    with np.load(path) as payload:
        points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if len(points) == 0 or np.any(points < 0) or np.any(points >= point_count):
        raise ValueError(f"invalid point indices: {path}")
    return points


def _load_z1(root: Path, scenes: set[str]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    evidence = {}
    with (root / "semantic_evidence_nodes.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["scene_name"]) in scenes:
                key = str(row["semantic_evidence_node_key"])
                if key in evidence:
                    raise ValueError(f"duplicate Z1 evidence key: {key}")
                evidence[key] = row
    bindings = defaultdict(list)
    with (root / "candidate_bindings.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene = str(row["scene_name"])
            source = str(row["candidate_source"])
            if scene in scenes and source in SOURCE_NAMES:
                key = str(row["semantic_evidence_node_key"])
                if key not in evidence:
                    raise ValueError(f"Z1 binding references missing evidence: {key}")
                bindings[key].append(row)
    return evidence, bindings


def _load_union_rows(path: Path, scenes: set[str]) -> dict[tuple[str, int], dict]:
    result = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene = str(row["scene_name"])
            key = (scene, int(row["candidate_id"]))
            if scene in scenes:
                if key in result:
                    raise ValueError(f"duplicate pair-union row: {key}")
                result[key] = row
    return result


def _native_masks(root: Path, scene: str) -> np.ndarray:
    prefix = root / scene / "native_cache" / f"{scene}_pred_"
    masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
    classes = np.load(str(prefix) + "classes.npy", mmap_mode="r")
    if masks.ndim != 2 or masks.shape[1] != len(classes):
        raise ValueError(f"{scene}: native cache dimensions disagree")
    return masks


def _scene_specs(
    scene: str,
    evidence: dict[str, dict],
    bindings: dict[str, list[dict]],
    union_rows: dict[tuple[str, int], dict],
    stream_root: Path,
) -> tuple[list[dict], int]:
    masks = _native_masks(stream_root, scene)
    point_count = int(masks.shape[0])
    specs = []
    for key, rows in bindings.items():
        if not rows or str(rows[0]["scene_name"]) != scene:
            continue
        sources = {str(row["candidate_source"]) for row in rows}
        if len(sources) != 1:
            raise ValueError(f"{key}: one evidence key has mixed native/pair-union bindings")
        source = next(iter(sources))
        representative = min(rows, key=lambda row: int(row["candidate_id"]))
        candidate_id = int(representative["candidate_id"])
        if source == "native":
            points = np.flatnonzero(np.asarray(masks[:, candidate_id], dtype=bool)).astype(np.int64)
            frame_ids = [str(value) for value in evidence[key]["input_frame_ids"]]
        else:
            union = union_rows.get((scene, candidate_id))
            if union is None:
                raise ValueError(f"{scene}: missing pair-union plan row {candidate_id}")
            points = _points(Path(union["points_path"]), point_count)
            frame_ids = [str(value) for value in evidence[key]["track_support_frame_ids"]]
            if not frame_ids:
                frame_ids = [str(value) for value in evidence[key]["input_frame_ids"]]
        if len(points) == 0:
            raise ValueError(f"{key}: empty geometry")
        actual_hash = _geometry_hash(points)
        if actual_hash != str(evidence[key]["geometry_hash"]):
            raise ValueError(f"{key}: geometry hash disagrees with Z1")
        specs.append({
            "scene_name": scene,
            "candidate_source": source,
            "representative_candidate_id": candidate_id,
            "bound_candidate_count": len(rows),
            "semantic_evidence_node_key": key,
            "semantic_evidence_node_id": int(evidence[key]["semantic_evidence_node_id"]),
            "geometry_node_id": int(evidence[key]["geometry_node_id"]),
            "geometry_hash": actual_hash,
            "point_count": int(len(points)),
            "points": points,
            "frame_ids": frame_ids,
            "selected_track_id": (
                int(representative["selected_track_id"]) if source == "pair_union" else None
            ),
            "selected_track_semantic_evidence_node_key": (
                str(representative["selected_track_semantic_evidence_node_key"])
                if source == "pair_union" else None
            ),
        })
    specs.sort(key=lambda row: (SOURCE_NAMES.index(row["candidate_source"]), row["semantic_evidence_node_id"]))
    return specs, point_count


def _scene_records(scene: str, args, alpha_state, labels, evidence, bindings, union_rows):
    from utils import WORLD_2_CAM

    specs, point_count = _scene_specs(
        scene, evidence, bindings, union_rows, args.stream_records_root
    )
    world = WORLD_2_CAM(str(args.prepared_dataset_root / scene), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    if visibility.shape[1] != point_count:
        raise ValueError(f"{scene}: projection point count disagrees with candidate geometry")
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    records = []
    pending_images, pending_masks = [], []
    pending = {"records": [], "views": []}
    for spec in specs:
        points = spec.pop("points")
        frame_ids = spec.pop("frame_ids")
        selected = _select_track_views(
            points,
            frame_ids,
            frame_lookup,
            projections,
            visibility,
            scaling,
            world.image_resolution,
            args.top_views,
            args.min_visible_points,
            args.crop_padding_ratio,
        )
        record = {
            **spec,
            "input_frame_count": len(frame_ids),
            "crop_contract": {
                "crop_mode": "limited_context",
                "crop_padding_ratio": float(args.crop_padding_ratio),
                "alpha_mask_dilation_iters": int(args.alpha_mask_dilation_iters),
                "mask_background": False,
            },
            "views": [],
            "_probs": [],
            "_logits": [],
        }
        for view in selected:
            image = np.asarray(imageio.imread(world.color_paths[view["frame_index"]]))
            x1, y1, x2, y2 = view["bbox_xyxy"]
            coords = projections[view["frame_index"], view["visible_point_ids"]]
            coords_color = np.stack([
                np.round(coords[:, 0] / scaling[1]).astype(np.int64),
                np.round(coords[:, 1] / scaling[0]).astype(np.int64),
            ], axis=1)
            crop = _make_crop_image(
                image, (x1, y1, x2, y2), coords=coords_color, mask_background=False
            )
            alpha_mask = _make_crop_alpha_mask(
                image.shape[:2],
                (x1, y1, x2, y2),
                coords=coords_color,
                dilation_iters=args.alpha_mask_dilation_iters,
            )
            view_record = {
                "frame_id": view["frame_id"],
                "visible_points": view["visible_points"],
                "bbox_xyxy": view["bbox_xyxy"],
            }
            record["views"].append(view_record)
            pending_images.append(crop)
            pending_masks.append(alpha_mask)
            pending["records"].append(record)
            pending["views"].append(view_record)
            if len(pending_images) >= args.batch_size:
                _flush(alpha_state, args.device, pending_images, pending_masks, pending)
        records.append(record)
    _flush(alpha_state, args.device, pending_images, pending_masks, pending)
    records = [_finalize_record(record, labels) for record in records]
    del world, projections, visibility
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--z1-root", type=Path, required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--prepared-dataset-root", type=Path, required=True)
    parser.add_argument("--combined-plan-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--alpha-clip-source", type=Path, default=Path("_external/AlphaCLIP/AlphaCLIP-main"))
    parser.add_argument("--alpha-clip-base-model", type=Path, default=Path("pretrained/alpha_clip/checkpoints/ViT-L-14.pt"))
    parser.add_argument("--alpha-clip-checkpoint", type=Path, default=Path("pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth"))
    parser.add_argument("--prompt-template", default="a photo of a {label}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-views", type=int, default=3)
    parser.add_argument("--min-visible-points", type=int, default=20)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.50)
    parser.add_argument("--alpha-mask-dilation-iters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "scene_list", "z1_root", "stream_records_root", "prepared_dataset_root", "combined_plan_root",
        "config_path", "alpha_clip_source", "alpha_clip_base_model", "alpha_clip_checkpoint", "output_root",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing Alpha-CLIP CPU fallback")
    if args.crop_padding_ratio != 0.50:
        raise SystemExit("this frozen limited-context export requires --crop-padding-ratio 0.50")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")
    with args.config_path.open() as handle:
        args.config = yaml.safe_load(handle)
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    labels = [str(value) for value in args.config["network2d"]["text_prompts"]]
    if len(labels) != 198:
        raise ValueError(f"expected 198 class prompts, got {len(labels)}")
    scenes = _read_scenes(args.scene_list)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    evidence, bindings = _load_z1(args.z1_root, set(scenes))
    union_rows = _load_union_rows(
        args.combined_plan_root / "pair_union_append_candidates.jsonl", set(scenes)
    )
    alpha_state = _load_alpha_clip(args, labels, args.device)
    args.output_root.mkdir(parents=True)
    all_records = []
    for index, scene in enumerate(scenes, 1):
        records = _scene_records(
            scene, args, alpha_state, labels, evidence, bindings, union_rows
        )
        all_records.extend(records)
        scene_root = args.output_root / scene
        scene_root.mkdir()
        (scene_root / "geometry_node_alphaclip_semantics.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        counts = Counter(row["candidate_source"] for row in records)
        print(
            f"[Z2 node Alpha] {index}/{len(scenes)} {scene}: "
            f"native={counts['native']}, pair_union={counts['pair_union']}",
            flush=True,
        )
    source_counts = Counter(row["candidate_source"] for row in all_records)
    semantic_counts = Counter(
        row["candidate_source"] for row in all_records if row["alphaclip_class_index"] >= 0
    )
    summary = {
        "diagnostic_type": "Z2 GT-free native exact-geometry and pair-union geometry Alpha-CLIP ledger",
        "ground_truth_usage": "none",
        "candidate_mutation": False,
        "scene_count": len(scenes),
        "class_prompt_count": len(labels),
        "record_count": len(all_records),
        "source_record_counts": dict(sorted(source_counts.items())),
        "source_with_semantics_counts": dict(sorted(semantic_counts.items())),
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "config"
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

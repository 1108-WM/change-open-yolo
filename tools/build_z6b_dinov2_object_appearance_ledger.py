#!/usr/bin/env python3
"""Build no-GT DINOv2 top-3 object-appearance features from frozen Z6b views.

This executable deliberately refuses CPU fallback. It consumes the exact Z6b
manifest, reconstructs a sparse projected foreground mask from the frozen 3D
geometry for each registered view, encodes mask-aware limited-context crops
with DINOv2 ViT-S/14, and stores per-view plus node-level consistency features.
It never changes geometry, candidates, classes, scores, or inference plans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "vit_small_patch14_dinov2"
EXPECTED_MANIFEST_SHA256 = "503261f316a0e9e642eb09c63c87d1fd10c9be60b3bd71041d3dfa8fb149107b"
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("DINOv2 embedding has invalid norm")
    return values / norm


def _cosine_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def _aggregate_embeddings(features: np.ndarray) -> dict:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("features must be a finite non-empty 2D array")
    similarities = _cosine_matrix(features)
    if len(features) == 1:
        medoid_index = 0
        pairwise = np.asarray([], dtype=np.float32)
    else:
        scores = (similarities.sum(axis=1) - 1.0) / (len(features) - 1)
        medoid_index = int(np.argmax(scores))
        pairwise = similarities[np.triu_indices(len(features), k=1)]
    mean_embedding = _l2_normalize(features.mean(axis=0))
    medoid_embedding = _l2_normalize(features[medoid_index])
    return {
        "mean_embedding": mean_embedding,
        "medoid_embedding": medoid_embedding,
        "medoid_view_index": medoid_index,
        "pairwise_cosine_mean": float(pairwise.mean()) if len(pairwise) else 1.0,
        "pairwise_cosine_min": float(pairwise.min()) if len(pairwise) else 1.0,
        "pairwise_cosine_std": float(pairwise.std()) if len(pairwise) else 0.0,
        "dispersion_one_minus_pairwise_mean": float(1.0 - pairwise.mean()) if len(pairwise) else 0.0,
    }


def _masked_crop_tensor(
    image: np.ndarray,
    bbox_xyxy: list[int],
    coords_color: np.ndarray,
    output_size: int,
    dilation_radius: int,
) -> torch.Tensor:
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("empty registered crop")
    mask = np.zeros(crop.shape[:2], dtype=bool)
    coords = np.asarray(coords_color, dtype=np.int64)
    local_x = np.clip(coords[:, 0] - x1, 0, max(0, crop.shape[1] - 1))
    local_y = np.clip(coords[:, 1] - y1, 0, max(0, crop.shape[0] - 1))
    mask[local_y, local_x] = True
    if dilation_radius > 0:
        try:
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=dilation_radius)
        except ImportError as error:
            raise RuntimeError("scipy is required for deterministic sparse-mask dilation") from error
    crop[~mask] = 127
    resized = np.asarray(
        Image.fromarray(crop).resize((output_size, output_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    return torch.from_numpy(((resized - MEAN) / STD).transpose(2, 0, 1))


def _load_model(checkpoint: Path):
    try:
        import timm
    except ImportError as error:
        raise RuntimeError("timm is required to run the Z6b DINOv2 GPU ledger") from error
    state = torch.load(checkpoint, map_location="cpu")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or set(unexpected) != {"mask_token"}:
        raise ValueError(
            f"DINOv2 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return model.cuda().eval()


def _geometry_points(row: dict, scene: str, args, native_masks, tracks, unions) -> np.ndarray:
    source = str(row["candidate_source"])
    candidate_id = int(row["geometry_reference"]["candidate_id"])
    if source == "native":
        points = np.flatnonzero(np.asarray(native_masks[:, candidate_id], dtype=bool)).astype(np.int64)
    elif source == "track":
        track = tracks.get(candidate_id)
        if track is None:
            raise ValueError(f"{scene}: missing track {candidate_id}")
        with np.load(Path(track["points_path"])) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    else:
        union = unions.get(candidate_id)
        if union is None:
            raise ValueError(f"{scene}: missing pair-union {candidate_id}")
        with np.load(Path(union["points_path"])) as payload:
            points = np.unique(np.asarray(payload["point_indices"], dtype=np.int64))
    if not len(points):
        raise ValueError(f'{row["semantic_evidence_node_key"]}: empty geometry')
    actual_hash = hashlib.sha1(points.astype(np.int64).tobytes()).hexdigest()
    if actual_hash != str(row["geometry_hash"]):
        raise ValueError(f'{row["semantic_evidence_node_key"]}: geometry hash mismatch')
    return points


def _scene_rows(
    scene: str, manifest_rows: list[dict], model, args
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    from utils import WORLD_2_CAM

    prefix = args.stream_records_root / scene / "native_cache" / f"{scene}_pred_"
    native_masks = np.load(str(prefix) + "masks.npy", mmap_mode="r")
    tracks_path = args.stream_records_root / scene / "d2b_tracks_filtered" / scene / "automatic_tracks.json"
    tracks = {int(row["track_id"]): row for row in json.loads(tracks_path.read_text())["tracks"]}
    unions = {}
    for row in _read_jsonl(args.combined_plan_root / "pair_union_append_candidates.jsonl"):
        if str(row["scene_name"]) == scene:
            unions[int(row["candidate_id"])] = row

    world = WORLD_2_CAM(str(args.prepared_dataset_root / scene), args.depth_scale, args.config)
    projections, visibility = world.get_mesh_projections()
    projections = projections.detach().cpu().numpy().astype(np.int64)
    visibility = visibility.detach().cpu().numpy().astype(bool)
    frame_lookup = {Path(path).stem: index for index, path in enumerate(world.color_paths)}
    scaling = (
        world.depth_resolution[0] / world.image_resolution[0],
        world.depth_resolution[1] / world.image_resolution[1],
    )
    output_size = int(model.patch_embed.img_size[0])
    pending, pending_keys, encoded = [], [], {}

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            output = model(torch.stack(pending).cuda(non_blocking=True)).detach().float().cpu().numpy()
        for key, feature in zip(pending_keys, output):
            encoded[key] = _l2_normalize(feature)
        pending.clear()
        pending_keys.clear()

    image_cache = {}
    for row in manifest_rows:
        if not row["views"]:
            continue
        points = _geometry_points(row, scene, args, native_masks, tracks, unions)
        for view in row["views"]:
            frame_id = str(view["frame_id"])
            frame_index = frame_lookup.get(frame_id)
            if frame_index is None or int(view["frame_index"]) != frame_index:
                raise ValueError(f'{row["semantic_evidence_node_key"]}: frame index mismatch')
            visible_points = points[visibility[frame_index, points]]
            if len(visible_points) != int(view["visible_point_count"]):
                raise ValueError(f'{row["semantic_evidence_node_key"]}: visible point count mismatch')
            if frame_id not in image_cache:
                image_cache[frame_id] = np.asarray(Image.open(view["rgb_path"]).convert("RGB"))
            coords = projections[frame_index, visible_points]
            coords_color = np.stack([
                np.round(coords[:, 0] / scaling[1]).astype(np.int64),
                np.round(coords[:, 1] / scaling[0]).astype(np.int64),
            ], axis=1)
            pending.append(_masked_crop_tensor(
                image_cache[frame_id], view["bbox_xyxy"], coords_color,
                output_size, args.mask_dilation_radius,
            ))
            pending_keys.append((str(row["semantic_evidence_node_key"]), int(view["view_rank"])))
            if len(pending) >= args.batch_size:
                flush()
    flush()

    feature_dim = int(model.num_features)
    node_features = np.full((len(manifest_rows), feature_dim * 2), np.nan, dtype=np.float32)
    per_view_features = np.full((len(manifest_rows), 3, feature_dim), np.nan, dtype=np.float32)
    ledger = []
    for local_index, row in enumerate(manifest_rows):
        view_features = [
            encoded[(str(row["semantic_evidence_node_key"]), int(view["view_rank"]))]
            for view in row["views"]
        ]
        if view_features:
            for view_offset, feature in enumerate(view_features):
                per_view_features[local_index, view_offset] = feature
            aggregate = _aggregate_embeddings(np.stack(view_features))
            node_features[local_index, :feature_dim] = aggregate.pop("mean_embedding")
            node_features[local_index, feature_dim:] = aggregate.pop("medoid_embedding")
            state = "available"
        else:
            aggregate = {
                "medoid_view_index": None,
                "pairwise_cosine_mean": None,
                "pairwise_cosine_min": None,
                "pairwise_cosine_std": None,
                "dispersion_one_minus_pairwise_mean": None,
            }
            state = "no_registered_view"
        ledger.append({
            "scene_name": scene,
            "node_index": int(row["node_index"]),
            "semantic_evidence_node_key": str(row["semantic_evidence_node_key"]),
            "candidate_source": str(row["candidate_source"]),
            "geometry_hash": str(row["geometry_hash"]),
            "view_count": len(view_features),
            "embedding_state": state,
            **aggregate,
            "ground_truth_usage": "none",
            "candidate_mutation": False,
            "class_mutation": False,
            "score_mutation": False,
            "inference_plan_written": False,
        })
    return ledger, node_features, per_view_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path(
        "docs/diagnostics/z6b_object_view_manifest_official100_20260812"
    ))
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/official_train100_20260808/official_train100.txt"
    ))
    parser.add_argument("--stream-records-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/records"
    ))
    parser.add_argument("--prepared-dataset-root", type=Path, default=Path(
        "/media/jia/软件1/scannet_train_stream/prepared"
    ))
    parser.add_argument("--combined-plan-root", type=Path, default=Path(
        "output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1"
    ))
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "pretrained/checkpoints/dinov2_vits14_pretrain.pth"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "docs/diagnostics/z6b_dinov2_object_appearance_official100_20260812"
    ))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mask-dilation-radius", type=int, default=5)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-nonofficial-manifest", action="store_true",
                        help="permit a separately audited frozen safety60 manifest")
    parser.add_argument("--safety60-transfer", action="store_true")
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing CPU fallback for full Z6b DINOv2 ledger")
    if not args.checkpoint.is_file() or not args.checkpoint.stat().st_size:
        raise SystemExit("DINOv2 checkpoint is missing or empty")
    manifest_path = args.manifest_root / "object_view_manifest.jsonl"
    actual_sha = _sha256(manifest_path)
    if actual_sha != EXPECTED_MANIFEST_SHA256 and not args.allow_nonofficial_manifest:
        raise SystemExit(f"Z6b manifest SHA-256 mismatch: {actual_sha}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise SystemExit("output root is non-empty; use --resume only for this exact frozen ledger")
    if args.batch_size <= 0 or args.mask_dilation_radius < 0 or args.scene_offset < 0:
        raise SystemExit("batch size must be positive and dilation radius non-negative")
    args.config = yaml.safe_load(args.config_path.read_text())
    args.depth_scale = float(args.config["openyolo3d"]["depth_scale"])
    all_scenes = _read_scenes(args.scene_list)
    scenes = all_scenes[args.scene_offset:]
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise SystemExit("max scenes must be positive")
        scenes = scenes[:args.max_scenes]
    if not scenes:
        raise SystemExit("selected scene range is empty")
    manifest = _read_jsonl(manifest_path)
    rows_by_scene = defaultdict(list)
    for row in manifest:
        rows_by_scene[str(row["scene_name"])].append(row)
    if set(rows_by_scene) != set(all_scenes):
        raise ValueError("manifest does not exactly cover the registered full scene list")
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.checkpoint)
    scene_summaries = []
    for ordinal, scene in enumerate(scenes, 1):
        target = args.output_root / scene
        if args.resume and (target / "summary.json").is_file():
            scene_summaries.append(json.loads((target / "summary.json").read_text()))
            print(f"[Z6b DINOv2] {ordinal}/{len(scenes)} {scene}: existing", flush=True)
            continue
        scene_rows = sorted(rows_by_scene[scene], key=lambda row: int(row["node_index"]))
        ledger, features, per_view_features = _scene_rows(scene, scene_rows, model, args)
        if (
            not np.isfinite(features[np.isfinite(features)]).all()
            or not np.isfinite(per_view_features[np.isfinite(per_view_features)]).all()
        ):
            raise ValueError(f"{scene}: non-finite DINOv2 feature")
        stage = args.output_root / f".{scene}.tmp.{os.getpid()}"
        stage.mkdir()
        with (stage / "dinov2_object_appearance_ledger.jsonl").open("w") as handle:
            for row in ledger:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        np.savez_compressed(
            stage / "dinov2_node_embeddings.npz",
            mean_medoid=features,
            per_view=per_view_features,
        )
        summary = {
            "scene_name": scene,
            "node_count": len(ledger),
            "node_with_embedding_count": sum(row["embedding_state"] == "available" for row in ledger),
            "node_without_embedding_count": sum(row["embedding_state"] != "available" for row in ledger),
            "encoded_view_count": sum(row["view_count"] for row in ledger),
            "ground_truth_usage": "none",
        }
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if target.exists():
            shutil.rmtree(stage)
            raise FileExistsError(f"{scene}: target exists without resume")
        os.replace(stage, target)
        scene_summaries.append(summary)
        print(f"[Z6b DINOv2] {ordinal}/{len(scenes)} {scene}: {summary['encoded_view_count']}", flush=True)
    summary = {
        "diagnostic_type": "Z6b GT-free frozen top-3 DINOv2 object-appearance ledger",
        "model": MODEL_NAME,
        "checkpoint": str(args.checkpoint),
        "manifest_sha256": actual_sha,
        "scene_count": len(scene_summaries),
        "node_count": sum(row["node_count"] for row in scene_summaries),
        "node_with_embedding_count": sum(row["node_with_embedding_count"] for row in scene_summaries),
        "node_without_embedding_count": sum(row["node_without_embedding_count"] for row in scene_summaries),
        "encoded_view_count": sum(row["encoded_view_count"] for row in scene_summaries),
        "feature_contract": "L2-normalized per-view CLS in [node, top3, dim]; L2-normalized mean and cosine-medoid; pairwise cosine dispersion",
        "ground_truth_usage": "none",
        "candidate_mutation": False,
        "geometry_mutation": False,
        "class_mutation": False,
        "score_mutation": False,
        "inference_plan_written": False,
        "mllm_invoked": False,
        "safety60_read": bool(args.safety60_transfer),
        "even48_read": False,
        "test60_read": False,
        "params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "config"
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

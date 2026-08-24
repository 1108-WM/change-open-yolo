#!/usr/bin/env python3
"""Build the preregistered GPU-only DM-SMS-1 Alpha-CLIP and SMS ledger."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dm_sms_core import compute_sms, sms_keep_mask, visible_ratio_multiscale_feature


PROMPT_TEMPLATE = "a blurry photo of {CLASS_NAME} in a room."
SCALE_COUNT = 3


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=bool)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(values, bitorder="little").tobytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenes(path: Path) -> list[str]:
    result = sorted(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("scene list is empty or contains duplicates")
    return result


def _ensure_loralib_stub() -> None:
    if importlib.util.find_spec("loralib") is not None:
        return
    module = types.ModuleType("loralib")
    module.Linear = torch.nn.Linear

    class MergedLinear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__(in_features, out_features)

    module.MergedLinear = MergedLinear
    sys.modules["loralib"] = module


def _load_alpha_clip(args: argparse.Namespace, prompts: list[str]) -> dict:
    source = str(args.alpha_clip_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    _ensure_loralib_stub()
    import alpha_clip

    model, preprocess = alpha_clip.load(
        str(args.alpha_clip_base),
        alpha_vision_ckpt_pth=str(args.alpha_clip_checkpoint),
        device="cuda",
    )
    model.eval()
    tokens = alpha_clip.tokenize(prompts).cuda()
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    input_resolution = int(model.visual.input_resolution)
    mask_transform = transforms.Compose([
        transforms.Resize(input_resolution, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(input_resolution),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.26),
    ])
    return {
        "model": model,
        "preprocess": preprocess,
        "mask_transform": mask_transform,
        "text_features": text_features.detach().cpu().numpy(),
        "feature_dim": int(text_features.shape[1]),
        "input_resolution": input_resolution,
    }


def _load_sam(args: argparse.Namespace):
    source = str(args.sam_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    model.cuda().eval()
    return SamPredictor(model)


@torch.no_grad()
def _encode_alpha_batch(
    state: dict, images: list[Image.Image], masks: list[Image.Image]
) -> np.ndarray:
    image_tensors = torch.stack([state["preprocess"](item) for item in images]).cuda()
    mask_tensors = torch.stack([state["mask_transform"](item) for item in masks]).cuda()
    image_tensors = image_tensors.half()
    mask_tensors = mask_tensors.half()
    features = state["model"].visual(image_tensors, mask_tensors)
    features = features / features.norm(dim=-1, keepdim=True)
    result = features.detach().float().cpu().numpy().astype(np.float32)
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all():
        raise ValueError("Alpha-CLIP produced nonfinite visual features")
    return result


def finalize_scene_semantics(
    records: list[dict], scale_features: np.ndarray, text_features: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Aggregate view/scale features, compute cosine and complete-population SMS."""
    feature_dim = int(text_features.shape[1])
    aggregate = np.full((len(records), feature_dim), np.nan, dtype=np.float32)
    similarities = np.full((len(records), len(text_features)), np.nan, dtype=np.float32)
    alpha_valid = np.zeros(len(records), dtype=bool)
    for geometry_index, row in enumerate(records):
        views = row["views"]
        incomplete_scale = False
        per_view = []
        ratios = []
        for view in views:
            if view.get("sam_mask_valid") is not True:
                continue
            scales = view["scales"]
            valid_scales = [item for item in scales if item.get("feature_valid") is True]
            if len(valid_scales) != SCALE_COUNT:
                incomplete_scale = True
                break
            valid_scales = sorted(valid_scales, key=lambda item: int(item["scale_index"]))
            if [int(item["scale_index"]) for item in valid_scales] != list(range(SCALE_COUNT)):
                incomplete_scale = True
                break
            per_view.append(np.stack([
                scale_features[int(item["feature_index"])] for item in valid_scales
            ]))
            ratios.append(float(view["visible_ratio"]))
        if per_view and not incomplete_scale:
            aggregate[geometry_index] = visible_ratio_multiscale_feature(
                np.stack(per_view), np.asarray(ratios, dtype=np.float32)
            )
            similarities[geometry_index] = (
                aggregate[geometry_index] @ text_features.T
            ).astype(np.float32)
            alpha_valid[geometry_index] = True
    population_complete = bool(alpha_valid.all())
    class_means = np.full(len(text_features), np.nan, dtype=np.float32)
    class_stds = np.full(len(text_features), np.nan, dtype=np.float32)
    deleted = 0
    if population_complete:
        sms = compute_sms(similarities)
        keep = sms_keep_mask(sms, threshold)
        class_means = sms.class_means
        class_stds = sms.class_stds
        for index, row in enumerate(records):
            top = int(sms.top_classes[index])
            row.update({
                "alpha_feature_valid": True,
                "alpha_class_index": top,
                "alpha_top_similarity": float(similarities[index, top]),
                "sms_score": float(sms.scores[index]),
                "sms_valid": bool(sms.valid[index]),
                "sms_keep": bool(keep[index]),
                "sms_reason": "valid" if sms.valid[index] else "degenerate_class_variance",
            })
        deleted = int((~keep).sum())
    else:
        for index, row in enumerate(records):
            if alpha_valid[index]:
                top = int(np.argmax(similarities[index]))
                row["alpha_class_index"] = top
                row["alpha_top_similarity"] = float(similarities[index, top])
            else:
                row["alpha_class_index"] = None
                row["alpha_top_similarity"] = None
            row.update({
                "alpha_feature_valid": bool(alpha_valid[index]),
                "sms_score": None,
                "sms_valid": False,
                "sms_keep": True,
                "sms_reason": "incomplete_scene_population",
            })
    return aggregate, similarities, np.stack([class_means, class_stds]), {
        "population_complete": population_complete,
        "alpha_valid_count": int(alpha_valid.sum()),
        "alpha_invalid_count": int((~alpha_valid).sum()),
        "sms_deleted_count": deleted,
        "sms_kept_count": len(records) - deleted,
    }


def _scene_complete(scene_root: Path, manifest_sha256: str) -> bool:
    path = scene_root / "summary.json"
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text())
    except Exception:
        return False
    return (
        summary.get("scene_complete") is True
        and summary.get("input_manifest_sha256") == manifest_sha256
        and all((scene_root / name).is_file() for name in (
            "records.jsonl", "scale_features.npy", "aggregate_features.npy",
            "similarities.npy", "sms_class_stats.npy",
        ))
    )


def _process_scene(
    scene: str, input_rows: list[dict], args: argparse.Namespace,
    alpha_state: dict, sam_predictor, text_features: np.ndarray,
    manifest_sha256: str,
) -> dict:
    scene_root = args.output_root / "scenes" / scene
    if _scene_complete(scene_root, manifest_sha256):
        return json.loads((scene_root / "summary.json").read_text())
    if scene_root.exists():
        raise FileExistsError(f"incomplete scene output exists: {scene_root}")
    scene_root.mkdir(parents=True)

    records = []
    frame_refs = defaultdict(list)
    feature_count = 0
    for geometry_index, source in enumerate(input_rows):
        row = {
            "scene_name": scene,
            "geometry_index": geometry_index,
            "geometry_hash": str(source["geometry_hash"]),
            "geometry_key": str(source["geometry_key"]),
            "point_count": int(source["point_count"]),
            "canonical_candidate_source": str(source["canonical_candidate_source"]),
            "canonical_candidate_id": int(source["canonical_candidate_id"]),
            "canonical_frozen_class_index": int(source["canonical_frozen_class_index"]),
            "canonical_frozen_class_valid": bool(source["canonical_frozen_class_valid"]),
            "canonical_frozen_score": float(source["canonical_frozen_score"]),
            "member_count": int(source.get("member_count", 1)),
            "members": [dict(member) for member in source.get("members", [])],
            "selected_view_count": int(source["selected_view_count"]),
            "views": [],
            "ground_truth_usage": "none",
        }
        for view_index, source_view in enumerate(source["views"]):
            scales = []
            for source_scale in source_view["crop_scales"]:
                scales.append({
                    "scale_index": int(source_scale["scale_index"]),
                    "bbox_expansion_fraction_per_side": float(
                        source_scale["bbox_expansion_fraction_per_side"]
                    ),
                    "crop_xyxy_integer_exclusive": list(
                        map(int, source_scale["crop_xyxy_integer_exclusive"])
                    ),
                    "feature_index": feature_count,
                    "feature_valid": False,
                    "feature_sha256": None,
                    "alpha_crop_mask_sha256": None,
                })
                feature_count += 1
            view = {
                "view_rank": int(source_view["view_rank"]),
                "frame_index": int(source_view["frame_index"]),
                "frame_id": str(source_view["frame_id"]),
                "rgb_path": str(source_view["rgb_path"]),
                "visible_point_count": int(source_view["visible_point_count"]),
                "visible_ratio": float(source_view["visible_ratio"]),
                "sam_box_prompt_xyxy": list(map(float, source_view["sam_box_prompt_xyxy"])),
                "sam_mask_valid": False,
                "sam_selected_mask_index": None,
                "sam_predicted_iou": None,
                "sam_mask_area": 0,
                "sam_mask_sha256": None,
                "scales": scales,
            }
            row["views"].append(view)
            frame_refs[str(source_view["rgb_path"])].append((geometry_index, view_index))
        records.append(row)
    if feature_count != SCALE_COUNT * sum(len(row["views"]) for row in records):
        raise AssertionError("feature indexing differs from three-scale manifest")
    scale_features = np.full(
        (feature_count, int(alpha_state["feature_dim"])), np.nan, dtype=np.float32
    )

    pending_images: list[Image.Image] = []
    pending_masks: list[Image.Image] = []
    pending_meta: list[tuple[int, dict]] = []

    def flush_alpha() -> None:
        nonlocal pending_images, pending_masks, pending_meta
        if not pending_images:
            return
        encoded = _encode_alpha_batch(alpha_state, pending_images, pending_masks)
        for feature, (feature_index, scale) in zip(encoded, pending_meta):
            scale_features[feature_index] = feature
            scale["feature_valid"] = True
            scale["feature_sha256"] = _array_sha256(feature)
        pending_images, pending_masks, pending_meta = [], [], []

    for rgb_path in sorted(frame_refs, key=lambda value: (Path(value).stem, value)):
        image = np.asarray(imageio.imread(rgb_path))[:, :, :3]
        image_pil = Image.fromarray(image).convert("RGB")
        refs = frame_refs[rgb_path]
        sam_predictor.set_image(np.ascontiguousarray(image))
        for start in range(0, len(refs), args.sam_batch_size):
            batch_refs = refs[start:start + args.sam_batch_size]
            boxes = np.asarray([
                records[g]["views"][v]["sam_box_prompt_xyxy"] for g, v in batch_refs
            ], dtype=np.float32)
            boxes_t = torch.as_tensor(boxes, device="cuda")
            transformed = sam_predictor.transform.apply_boxes_torch(boxes_t, image.shape[:2])
            with torch.no_grad():
                masks_t, ious_t, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=True,
                    return_logits=False,
                )
            selected = torch.argmax(ious_t.float(), dim=1)
            rows_t = torch.arange(len(batch_refs), device=selected.device)
            selected_masks = masks_t[rows_t, selected].detach().cpu().numpy().astype(bool)
            selected_ious = ious_t[rows_t, selected].detach().float().cpu().numpy()
            selected_ids = selected.detach().cpu().numpy()
            for (geometry_index, view_index), mask, predicted_iou, mask_index in zip(
                batch_refs, selected_masks, selected_ious, selected_ids
            ):
                view = records[geometry_index]["views"][view_index]
                area = int(mask.sum())
                if area <= 0 or not np.isfinite(predicted_iou):
                    continue
                view.update({
                    "sam_mask_valid": True,
                    "sam_selected_mask_index": int(mask_index),
                    "sam_predicted_iou": float(predicted_iou),
                    "sam_mask_area": area,
                    "sam_mask_sha256": _mask_sha256(mask),
                })
                for scale in view["scales"]:
                    crop_box = tuple(scale["crop_xyxy_integer_exclusive"])
                    rgb_crop = image_pil.crop(crop_box)
                    x1, y1, x2, y2 = crop_box
                    alpha_crop_np = mask[y1:y2, x1:x2]
                    if alpha_crop_np.shape != (y2 - y1, x2 - x1):
                        raise ValueError("SAM alpha crop shape differs from manifest crop")
                    alpha_crop = Image.fromarray(
                        alpha_crop_np.astype(np.uint8) * 255, mode="L"
                    )
                    scale["alpha_crop_mask_sha256"] = _mask_sha256(alpha_crop_np)
                    pending_images.append(rgb_crop)
                    pending_masks.append(alpha_crop)
                    pending_meta.append((int(scale["feature_index"]), scale))
                    if len(pending_images) >= args.alpha_batch_size:
                        flush_alpha()
            del masks_t, ious_t, transformed
        sam_predictor.reset_image()
        flush_alpha()
    flush_alpha()

    aggregate, similarities, class_stats, semantic_summary = finalize_scene_semantics(
        records, scale_features, text_features, args.sms_threshold
    )
    np.save(scene_root / "scale_features.npy", scale_features, allow_pickle=False)
    np.save(scene_root / "aggregate_features.npy", aggregate, allow_pickle=False)
    np.save(scene_root / "similarities.npy", similarities, allow_pickle=False)
    np.save(scene_root / "sms_class_stats.npy", class_stats, allow_pickle=False)
    with (scene_root / "records.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    source_counts = Counter(row["canonical_candidate_source"] for row in records)
    deleted_sources = Counter(
        row["canonical_candidate_source"] for row in records if not row["sms_keep"]
    )
    summary = {
        "version": "dm_sms1_alpha_embedding_scene_v1",
        "scene_name": scene,
        "scene_complete": True,
        "geometry_count": len(records),
        "member_count": sum(int(row.get("member_count", 1)) for row in records),
        "selected_view_count": sum(len(row["views"]) for row in records),
        "scale_feature_count": len(scale_features),
        "sam_missing_view_count": sum(
            int(view["sam_mask_valid"] is not True)
            for row in records for view in row["views"]
        ),
        "feature_dim": int(alpha_state["feature_dim"]),
        **semantic_summary,
        "source_geometry_counts": dict(sorted(source_counts.items())),
        "sms_deleted_source_counts": dict(sorted(deleted_sources.items())),
        "sms_threshold": float(args.sms_threshold),
        "input_manifest_sha256": manifest_sha256,
        "records_sha256": None,
        "scale_features_sha256": _sha256(scene_root / "scale_features.npy"),
        "aggregate_features_sha256": _sha256(scene_root / "aggregate_features.npy"),
        "similarities_sha256": _sha256(scene_root / "similarities.npy"),
        "sms_class_stats_sha256": _sha256(scene_root / "sms_class_stats.npy"),
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "candidate_mutation": False,
    }
    summary["records_sha256"] = _sha256(scene_root / "records.jsonl")
    (scene_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _validate_assets(args: argparse.Namespace, provenance: dict) -> None:
    expected = provenance["models"]
    checks = (
        (args.alpha_clip_base, expected["alpha_clip_base_sha256"]),
        (args.alpha_clip_checkpoint, expected["alpha_clip_checkpoint_sha256"]),
        (args.sam_checkpoint, expected["sam_checkpoint_sha256"]),
        (args.config_path, provenance["prompt_contract"]["config_sha256"]),
    )
    for path, digest in checks:
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"asset hash mismatch: {path}")


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("DM-SMS-1 Stage D requires CUDA; CPU fallback is forbidden")
    for name in (
        "scene_list", "manifest_root", "manifest_audit_root", "output_root", "config_path",
        "asset_provenance", "alpha_clip_source", "alpha_clip_base",
        "alpha_clip_checkpoint", "sam_source", "sam_checkpoint",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    if args.sms_threshold != 0.0:
        raise ValueError("DM-SMS-1A freezes tau_SMS=0; threshold scanning is forbidden")
    provenance = json.loads(args.asset_provenance.read_text())
    _validate_assets(args, provenance)
    all_scenes = _scenes(args.scene_list)
    scenes = all_scenes[:args.max_scenes] if args.max_scenes is not None else all_scenes
    if args.expected_scene_count is not None and len(scenes) != args.expected_scene_count:
        raise ValueError("scene count differs from frozen Stage D contract")
    manifest_path = args.manifest_root / "alpha_view_manifest.jsonl"
    manifest_summary_path = args.manifest_root / "summary.json"
    manifest_audit_path = args.manifest_audit_root / "summary.json"
    manifest_summary = json.loads(manifest_summary_path.read_text())
    manifest_audit = json.loads(manifest_audit_path.read_text())
    scene_list_sha256 = _sha256(args.scene_list)
    if (
        manifest_summary.get("manifest_valid") is not True
        or manifest_summary.get("ground_truth_read") is not False
        or manifest_summary.get("embedding_computed") is not False
        or manifest_audit.get("audit_valid") is not True
        or manifest_audit.get("embedding_computed") is not False
    ):
        raise ValueError("Stage C manifest/audit contract is invalid")
    if manifest_summary.get("input_provenance", {}).get("scene_list_sha256") != scene_list_sha256:
        raise ValueError("Stage C manifest scene list differs from Stage D scene list")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_audit["input_provenance"]["manifest_sha256"] != manifest_sha256:
        raise ValueError("Stage C manifest hash differs from its audit")
    input_rows = _read_jsonl(manifest_path)
    by_scene = defaultdict(list)
    for row in input_rows:
        by_scene[str(row["scene_name"])].append(row)
    if not set(scenes).issubset(by_scene):
        raise ValueError("Stage C scene coverage differs")
    for scene in scenes:
        by_scene[scene].sort(key=lambda row: row["geometry_hash"])

    with args.config_path.open() as handle:
        config = yaml.safe_load(handle)
    labels = list(config["network2d"]["text_prompts"])
    if len(labels) != 198:
        raise ValueError("DM-SMS-1 requires exactly 198 configured classes")
    prompts = [PROMPT_TEMPLATE.format(CLASS_NAME=name) for name in labels]

    if args.output_root.exists() and not args.resume:
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "scenes").mkdir(exist_ok=True)
    alpha_state = _load_alpha_clip(args, prompts)
    text_features = np.asarray(alpha_state["text_features"], dtype=np.float32)
    np.save(args.output_root / "text_features.npy", text_features, allow_pickle=False)
    (args.output_root / "text_prompts.json").write_text(
        json.dumps({"class_names": labels, "prompts": prompts}, ensure_ascii=False, indent=2) + "\n"
    )
    sam_predictor = _load_sam(args)
    model_provenance = {
        "alpha_clip_source": str(args.alpha_clip_source),
        "alpha_clip_source_commit": provenance["models"]["alpha_clip_source_commit"],
        "alpha_clip_base": str(args.alpha_clip_base),
        "alpha_clip_base_sha256": _sha256(args.alpha_clip_base),
        "alpha_clip_checkpoint": str(args.alpha_clip_checkpoint),
        "alpha_clip_checkpoint_sha256": _sha256(args.alpha_clip_checkpoint),
        "sam_source": str(args.sam_source),
        "sam_source_commit": provenance["models"]["sam_source_commit"],
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_checkpoint_sha256": _sha256(args.sam_checkpoint),
        "sam_model_type": args.sam_model_type,
        "prompt_template": PROMPT_TEMPLATE,
        "class_count": len(labels),
        "text_features_sha256": _sha256(args.output_root / "text_features.npy"),
        "text_prompts_sha256": _sha256(args.output_root / "text_prompts.json"),
        "feature_dim": int(alpha_state["feature_dim"]),
        "input_resolution": int(alpha_state["input_resolution"]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "parameters_frozen": True,
    }
    (args.output_root / "model_provenance.json").write_text(
        json.dumps(model_provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    summaries = []
    for index, scene in enumerate(scenes, 1):
        summary = _process_scene(
            scene, by_scene[scene], args, alpha_state, sam_predictor,
            text_features, manifest_sha256,
        )
        summaries.append(summary)
        print(
            f"[DM-SMS-1 Stage D] {index}/{len(scenes)} {scene}: "
            f"geometry={summary['geometry_count']} valid={summary['alpha_valid_count']} "
            f"deleted={summary['sms_deleted_count']}",
            flush=True,
        )
    output = {
        "version": "dm_sms1_alpha_embedding_ledger_v1",
        "stage": "D_gpu_alpha_sms",
        "stage_complete": True,
        "formal_full_scene_run": args.max_scenes is None,
        "scene_count": len(scenes),
        "geometry_count": sum(row["geometry_count"] for row in summaries),
        "member_count": sum(row["member_count"] for row in summaries),
        "selected_view_count": sum(row["selected_view_count"] for row in summaries),
        "scale_feature_count": sum(row["scale_feature_count"] for row in summaries),
        "sam_missing_view_count": sum(row["sam_missing_view_count"] for row in summaries),
        "alpha_valid_count": sum(row["alpha_valid_count"] for row in summaries),
        "alpha_invalid_count": sum(row["alpha_invalid_count"] for row in summaries),
        "sms_population_complete_scene_count": sum(
            int(row["population_complete"]) for row in summaries
        ),
        "sms_deleted_count": sum(row["sms_deleted_count"] for row in summaries),
        "sms_kept_count": sum(row["sms_kept_count"] for row in summaries),
        "sms_threshold": float(args.sms_threshold),
        "class_count": len(labels),
        "prompt_template": PROMPT_TEMPLATE,
        "sam_prompt_contract": "tight projected bbox; ViT-B; highest predicted-IoU mask; exact tie first index",
        "feature_contract": "L2-normalized Alpha-CLIP feature per view/scale",
        "aggregation_contract": "L2Norm(sum_v sum_l visible_ratio[v] * feature[v,l])",
        "similarity_contract": "raw cosine against 198 L2-normalized text features",
        "sms_contract": "complete unique-geometry scene population; delete valid SMS<0",
        "ground_truth_usage": "none",
        "ground_truth_read": False,
        "ap_computed": False,
        "candidate_geometry_mutation": False,
        "candidate_count_mutation_before_sms": False,
        "input_provenance": {
            "scene_list": str(args.scene_list),
            "scene_list_sha256": _sha256(args.scene_list),
            "alpha_view_manifest": str(manifest_path),
            "alpha_view_manifest_sha256": manifest_sha256,
            "alpha_view_manifest_summary_sha256": _sha256(manifest_summary_path),
            "alpha_view_manifest_audit_sha256": _sha256(manifest_audit_path),
            "asset_provenance": str(args.asset_provenance),
            "asset_provenance_sha256": _sha256(args.asset_provenance),
            "model_provenance_sha256": _sha256(args.output_root / "model_provenance.json"),
        },
        "scene_summaries": summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=Path(
        "output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt"
    ))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--manifest-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained/config_scannet200.yaml"))
    parser.add_argument("--asset-provenance", type=Path, default=Path(
        "docs/DM_SMS1_ASSET_PROVENANCE_20260817.json"
    ))
    parser.add_argument("--alpha-clip-source", type=Path, default=Path(
        "_external/AlphaCLIP/AlphaCLIP-main"
    ))
    parser.add_argument("--alpha-clip-base", type=Path, default=Path(
        "pretrained/alpha_clip/checkpoints/ViT-L-14.pt"
    ))
    parser.add_argument("--alpha-clip-checkpoint", type=Path, default=Path(
        "pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth"
    ))
    parser.add_argument("--sam-source", type=Path, default=Path(
        "_external/segment-anything/segment-anything-main"
    ))
    parser.add_argument("--sam-checkpoint", type=Path, default=Path(
        "pretrained/checkpoints/sam_vit_b_01ec64.pth"
    ))
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-batch-size", type=int, default=8)
    parser.add_argument("--alpha-batch-size", type=int, default=32)
    parser.add_argument("--sms-threshold", type=float, default=0.0)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

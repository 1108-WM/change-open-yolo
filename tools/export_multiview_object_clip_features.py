import argparse
import gc
import importlib.util
import json
import os
import os.path as osp
import sys
import types

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import binary_dilation
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast
from torchvision import transforms
from torchvision.transforms import InterpolationMode

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200
from utils import OpenYolo3D, get_visibility_mat
from utils.backprojection_fusion import append_backprojection_proposals, load_backprojection_candidates


def _load_yaml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _load_clip_processor(clip_model_path):
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_path, local_files_only=True)
    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": 224},
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    return CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _ensure_loralib_stub():
    if importlib.util.find_spec("loralib") is not None:
        return
    module = types.ModuleType("loralib")
    module.Linear = torch.nn.Linear

    class MergedLinear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__(in_features, out_features)

    module.MergedLinear = MergedLinear
    sys.modules["loralib"] = module


def _load_alpha_clip(args, labels, device):
    alpha_source = osp.abspath(args.alpha_clip_source)
    if alpha_source not in sys.path:
        sys.path.insert(0, alpha_source)
    _ensure_loralib_stub()
    import alpha_clip

    model, preprocess = alpha_clip.load(
        osp.abspath(args.alpha_clip_base_model),
        alpha_vision_ckpt_pth=osp.abspath(args.alpha_clip_checkpoint),
        device=device,
    )
    model.eval()
    text = alpha_clip.tokenize(_text_prompts(labels, args.prompt_template)).to(device)
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().detach().float()
    input_resolution = int(model.visual.input_resolution)
    mask_transform = transforms.Compose(
        [
            transforms.Resize(input_resolution, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(input_resolution),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.26),
        ]
    )
    return {
        "module": alpha_clip,
        "model": model,
        "preprocess": preprocess,
        "mask_transform": mask_transform,
        "text_features": text_features,
        "logit_scale": logit_scale,
        "input_resolution": input_resolution,
    }


def _text_prompts(labels, template):
    return [template.format(label=label.replace("-", " ")) for label in labels]


def _softmax(values):
    values = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.maximum(exp_values.sum(axis=-1, keepdims=True), 1e-12)


def _clear_openyolo_state(openyolo3d, unload_3d_network=False):
    for attr in (
        "world2cam",
        "mesh_projections",
        "preds_3d",
        "preds_2d",
        "predicted_masks",
        "predicated_scores",
        "predicated_classes",
    ):
        if hasattr(openyolo3d, attr):
            setattr(openyolo3d, attr, None)
    if unload_3d_network and hasattr(openyolo3d, "network_3d"):
        openyolo3d.network_3d = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _source_for_prediction(pred_id, original_count, applied_records):
    if pred_id < original_count:
        return "mask3d", "mask3d", None
    applied_idx = pred_id - original_count
    if 0 <= applied_idx < len(applied_records):
        applied = applied_records[applied_idx]
        return applied.get("source_kind") or "proposal", applied.get("source_name") or "proposal", applied
    return "proposal", "proposal", None


def _parse_filter_set(value):
    if value is None:
        return None
    parsed = {item.strip() for item in str(value).split(",") if item.strip()}
    return parsed or None


def _should_rescore_prediction(pred_score, source_kind, source_name, class_name, args):
    policy = args.rescore_policy
    if policy == "all":
        selected = True
    elif policy == "low_score":
        selected = args.rescore_max_base_score is None or float(pred_score) <= float(args.rescore_max_base_score)
    elif policy == "proposals":
        selected = source_kind != "mask3d"
    elif policy == "proposals_or_low_score":
        selected = source_kind != "mask3d" or (
            args.rescore_max_base_score is None or float(pred_score) <= float(args.rescore_max_base_score)
        )
    elif policy == "source_or_low_score":
        source_kinds = _parse_filter_set(args.rescore_source_kinds)
        source_names = _parse_filter_set(args.rescore_source_names)
        source_match = False
        if source_kinds is not None and source_kind in source_kinds:
            source_match = True
        if source_names is not None and source_name in source_names:
            source_match = True
        selected = source_match or (
            args.rescore_max_base_score is None or float(pred_score) <= float(args.rescore_max_base_score)
        )
    else:
        selected = True

    if not selected:
        return False, f"policy_{policy}"
    if args.rescore_min_base_score is not None and float(pred_score) < float(args.rescore_min_base_score):
        return False, "below_min_base_score"
    if args.rescore_classes is not None:
        allowed_classes = _parse_filter_set(args.rescore_classes)
        if allowed_classes is not None and class_name not in allowed_classes:
            return False, "class_not_selected"
    return True, "selected"


def _bbox_from_visible_points(coords, scaling_params, image_shape, padding_ratio=0.15, min_crop_size=16):
    if len(coords) == 0:
        return None
    x1 = float(coords[:, 0].min() / scaling_params[1])
    x2 = float((coords[:, 0].max() + 1) / scaling_params[1])
    y1 = float(coords[:, 1].min() / scaling_params[0])
    y2 = float((coords[:, 1].max() + 1) / scaling_params[0])
    h, w = int(image_shape[0]), int(image_shape[1])
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(padding_ratio) * max(bw, bh)
    x1 -= pad
    x2 += pad
    y1 -= pad
    y2 += pad
    if x2 - x1 < min_crop_size:
        mid = 0.5 * (x1 + x2)
        x1 = mid - 0.5 * min_crop_size
        x2 = mid + 0.5 * min_crop_size
    if y2 - y1 < min_crop_size:
        mid = 0.5 * (y1 + y2)
        y1 = mid - 0.5 * min_crop_size
        y2 = mid + 0.5 * min_crop_size
    x1 = int(max(0, np.floor(x1)))
    y1 = int(max(0, np.floor(y1)))
    x2 = int(min(w, np.ceil(x2)))
    y2 = int(min(h, np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _square_bbox(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    side = max(x2 - x1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    x1 = int(np.floor(cx - 0.5 * side))
    x2 = int(np.ceil(cx + 0.5 * side))
    y1 = int(np.floor(cy - 0.5 * side))
    y2 = int(np.ceil(cy + 0.5 * side))
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_w:
        x1 -= x2 - image_w
        x2 = image_w
    if y2 > image_h:
        y1 -= y2 - image_h
        y2 = image_h
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)
    return x1, y1, x2, y2


def _select_views(
    mask,
    projections,
    keep_visible_points,
    scaling_params,
    image_resolution,
    top_views,
    min_visible_points,
    max_bbox_area_ratio,
    square_crops=False,
):
    visibility = get_visibility_mat(
        torch.from_numpy(mask[None, :].astype(np.float32)),
        torch.from_numpy(keep_visible_points.astype(np.float32)),
        topk=min(top_views * 4, keep_visible_points.shape[0]),
    ).numpy()[0]
    candidates = []
    image_h, image_w = int(image_resolution[0]), int(image_resolution[1])
    for frame_id in np.where(visibility)[0]:
        visible = (keep_visible_points[frame_id].squeeze() * mask).astype(bool)
        visible_count = int(visible.sum())
        if visible_count < min_visible_points:
            continue
        coords = projections[frame_id][visible].astype(np.int64)
        bbox = _bbox_from_visible_points(coords, scaling_params, (image_h, image_w))
        if bbox is None:
            continue
        if square_crops:
            bbox = _square_bbox(bbox, (image_h, image_w))
        x1, y1, x2, y2 = bbox
        area_ratio = float(((x2 - x1) * (y2 - y1)) / max(1, image_w * image_h))
        if max_bbox_area_ratio is not None and area_ratio > max_bbox_area_ratio:
            continue
        candidates.append(
            {
                "frame_id": int(frame_id),
                "visible_points": visible_count,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_area_ratio": area_ratio,
            }
        )
    candidates = sorted(candidates, key=lambda item: (-item["visible_points"], item["bbox_area_ratio"]))
    return candidates[:top_views]


def _make_crop_image(image, bbox, coords=None, mask_background=False, dilation_iters=5):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    if not mask_background or coords is None or len(coords) == 0:
        return Image.fromarray(crop).convert("RGB")

    local_x = coords[:, 0] - x1
    local_y = coords[:, 1] - y1
    valid = (local_x >= 0) & (local_x < crop.shape[1]) & (local_y >= 0) & (local_y < crop.shape[0])
    if valid.sum() == 0:
        return Image.fromarray(crop).convert("RGB")
    mask = np.zeros(crop.shape[:2], dtype=bool)
    mask[local_y[valid], local_x[valid]] = True
    if dilation_iters > 0:
        mask = binary_dilation(mask, iterations=int(dilation_iters))
    background = np.full_like(crop, 128)
    crop = np.where(mask[:, :, None], crop, background)
    return Image.fromarray(crop).convert("RGB")


def _make_crop_alpha_mask(image_shape, bbox, coords=None, dilation_iters=5):
    x1, y1, x2, y2 = bbox
    crop_h = max(1, y2 - y1)
    crop_w = max(1, x2 - x1)
    mask = np.zeros((crop_h, crop_w), dtype=bool)
    if coords is None or len(coords) == 0:
        return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    local_x = coords[:, 0] - x1
    local_y = coords[:, 1] - y1
    valid = (local_x >= 0) & (local_x < crop_w) & (local_y >= 0) & (local_y < crop_h)
    if valid.sum() > 0:
        mask[local_y[valid], local_x[valid]] = True
    if dilation_iters > 0 and mask.any():
        mask = binary_dilation(mask, iterations=int(dilation_iters))
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


@torch.no_grad()
def _encode_images(model, processor, text_features, logit_scale, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).detach().cpu().numpy().astype(np.float32)
    logits = (float(logit_scale.detach().cpu()) * similarities).astype(np.float32)
    return {
        "probs": _softmax(logits).astype(np.float32),
        "logits": logits,
        "similarities": similarities,
    }


@torch.no_grad()
def _encode_alpha_clip_images(alpha_state, images, alpha_masks, device):
    model = alpha_state["model"]
    image_tensors = torch.stack([alpha_state["preprocess"](image) for image in images], dim=0).to(device)
    alpha_tensors = torch.stack([alpha_state["mask_transform"](mask) for mask in alpha_masks], dim=0).to(device)
    if str(device).startswith("cuda"):
        image_tensors = image_tensors.half()
        alpha_tensors = alpha_tensors.half()
    image_features = model.visual(image_tensors, alpha_tensors)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ alpha_state["text_features"].T).detach().cpu().numpy().astype(np.float32)
    logits = (float(alpha_state["logit_scale"].detach().cpu()) * similarities).astype(np.float32)
    return {
        "probs": _softmax(logits).astype(np.float32),
        "logits": logits,
        "similarities": similarities,
    }


def _encode_pending(args, clip_state, pending_images, pending_alpha_masks, device):
    if args.vision_encoder == "alpha_clip":
        return _encode_alpha_clip_images(clip_state, pending_images, pending_alpha_masks, device)
    return _encode_images(
        clip_state["model"],
        clip_state["processor"],
        clip_state["text_features"],
        clip_state["logit_scale"],
        pending_images,
        device,
    )


def _aggregate_rows(rows, mode, probability=False):
    if len(rows) == 0:
        return None
    stacked = np.stack(rows, axis=0).astype(np.float32)
    if mode == "max":
        return stacked.max(axis=0)
    if probability and mode == "noisy_or":
        return 1.0 - np.prod(1.0 - np.clip(stacked, 0.0, 1.0), axis=0)
    return stacked.mean(axis=0)


def export_features(args):
    config = _load_yaml(osp.join("./pretrained", f"config_{args.dataset_name}.yaml"))
    labels = config["network2d"]["text_prompts"]
    depth_scale = config["openyolo3d"]["depth_scale"]
    path_2_dataset = osp.join("./data", args.dataset_name)
    datatype = "point cloud" if args.dataset_name == "replica" else "mesh"
    if args.scene_names:
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    elif args.dataset_name == "replica":
        scene_names = SCENE_NAMES_REPLICA
    else:
        scene_names = SCENE_NAMES_SCANNET200

    candidates_by_scene, candidate_summary = load_backprojection_candidates(args.backprojection_candidates)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.vision_encoder == "alpha_clip":
        clip_state = _load_alpha_clip(args, labels, device)
    else:
        processor = _load_clip_processor(args.clip_model_path)
        model = CLIPModel.from_pretrained(args.clip_model_path, local_files_only=True).to(device)
        model.eval()
        text_inputs = processor(text=_text_prompts(labels, args.prompt_template), padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp().detach().float()
        clip_state = {
            "model": model,
            "processor": processor,
            "text_features": text_features,
            "logit_scale": logit_scale,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    openyolo3d = OpenYolo3D(f"./pretrained/config_{args.dataset_name}.yaml")
    summary = {
        "dataset_name": args.dataset_name,
        "scene_names": scene_names,
        "labels": labels,
        "vision_encoder": args.vision_encoder,
        "clip_model_path": args.clip_model_path,
        "alpha_clip_source": args.alpha_clip_source,
        "alpha_clip_base_model": args.alpha_clip_base_model,
        "alpha_clip_checkpoint": args.alpha_clip_checkpoint,
        "candidate_summary": candidate_summary,
        "params": vars(args),
        "total_records": 0,
        "total_view_crops": 0,
        "scenes": {},
    }

    for scene_name in tqdm(scene_names):
        scene_dir = osp.join(args.output_dir, scene_name)
        crop_dir = osp.join(scene_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        scene_id = scene_name.replace("scene", "")
        processed_file = (
            osp.join(path_2_dataset, scene_name, f"{scene_id}.npy")
            if args.dataset_name == "scannet200"
            else None
        )
        prediction = openyolo3d.predict(
            path_2_scene_data=osp.join(path_2_dataset, scene_name),
            depth_scale=depth_scale,
            datatype=datatype,
            processed_scene=processed_file,
            path_to_3d_masks=args.path_to_3d_masks,
            is_gt=args.is_gt,
            path_to_2d_preds=args.path_to_2d_preds,
            save_2d_preds=args.save_2d_preds,
            reuse_2d_preds=args.reuse_2d_preds,
        )
        scene_prediction = prediction[scene_name]
        original_count = int(scene_prediction[0].shape[1])
        points_xyz, _ = openyolo3d.world2cam.load_ply(openyolo3d.world2cam.mesh)
        fusion_report = {"loaded": 0, "applied": [], "skipped": []}
        if args.backprojection_candidates is not None:
            point_segments = None
            point_visibility = None
            superpoint_box_context = None
            if args.backprojection_superpoint_refine:
                if processed_file is None:
                    raise ValueError("--backprojection_superpoint_refine requires a ScanNet200 processed scene file.")
                point_segments = np.load(processed_file, mmap_mode="r")[:, 9].astype(np.int64)
                projections_for_sp, point_visibility = openyolo3d.mesh_projections
                if float(args.backprojection_superpoint_min_box_positive_ratio or 0.0) > 0.0:
                    superpoint_box_context = {
                        "projections": projections_for_sp,
                        "scaling_params": openyolo3d.scaling_params,
                    }
            fused = append_backprojection_proposals(
                scene_name,
                scene_prediction[0],
                scene_prediction[1],
                scene_prediction[2],
                candidates_by_scene,
                points_xyz=points_xyz[:, :3],
                point_segments=point_segments,
                point_visibility=point_visibility,
                min_score=args.backprojection_min_score,
                min_seed_points=args.backprojection_min_seed_points,
                max_existing_iou=args.backprojection_max_existing_iou,
                max_seed_in_existing_mask_ratio=args.backprojection_max_seed_in_existing_mask_ratio,
                max_proposal_iou=args.backprojection_max_proposal_iou,
                max_candidates=args.backprojection_max_candidates_per_scene,
                score_scale=args.backprojection_score_scale,
                use_candidate_fusion_score=args.backprojection_use_candidate_fusion_score,
                allowed_classes=args.backprojection_allowed_classes,
                blocked_classes=args.backprojection_blocked_classes,
                cc_cleanup=args.backprojection_cc_cleanup,
                cc_radius=args.backprojection_cc_radius,
                cc_min_component_points=args.backprojection_cc_min_component_points,
                cc_keep_topk=args.backprojection_cc_keep_topk,
                source_priorities=args.backprojection_source_priorities,
                source_max_candidates=args.backprojection_source_max_candidates,
                source_score_scales=args.backprojection_source_score_scales,
                superpoint_refine=args.backprojection_superpoint_refine,
                superpoint_min_coverage=args.backprojection_superpoint_min_coverage,
                superpoint_max_expansion_ratio=args.backprojection_superpoint_max_expansion_ratio,
                superpoint_min_view_siou=args.backprojection_superpoint_min_view_siou,
                superpoint_box_context=superpoint_box_context,
                superpoint_min_box_positive_ratio=args.backprojection_superpoint_min_box_positive_ratio,
                superpoint_max_box_negative_ratio=args.backprojection_superpoint_max_box_negative_ratio,
                superpoint_box_min_visible_points=args.backprojection_superpoint_box_min_visible_points,
                superpoint_box_min_views=args.backprojection_superpoint_box_min_views,
            )
            scene_prediction = fused[:3]
            fusion_report = fused[3]

        pred_masks = _to_numpy(scene_prediction[0]).astype(bool)
        pred_classes = _to_numpy(scene_prediction[1]).astype(np.int64)
        pred_scores = _to_numpy(scene_prediction[2]).astype(np.float32)
        projections, keep_visible_points = openyolo3d.mesh_projections
        projections = _to_numpy(projections)
        keep_visible_points = _to_numpy(keep_visible_points)
        applied_records = fusion_report.get("applied", [])

        records = []
        pending_images = []
        pending_alpha_masks = []
        pending_meta = []
        meta_probs = {}
        meta_logits = {}
        meta_similarities = {}
        skipped_rescore = {}
        for pred_id in range(pred_masks.shape[1]):
            source_kind, source_name, applied = _source_for_prediction(pred_id, original_count, applied_records)
            class_id = int(pred_classes[pred_id])
            class_name = labels[class_id] if 0 <= class_id < len(labels) else "unknown"
            should_rescore, skip_reason = _should_rescore_prediction(
                pred_scores[pred_id],
                source_kind,
                source_name,
                class_name,
                args,
            )
            if not should_rescore:
                skipped_rescore[skip_reason] = skipped_rescore.get(skip_reason, 0) + 1
                continue
            mask = pred_masks[:, pred_id]
            selected_views = _select_views(
                mask,
                projections,
                keep_visible_points,
                openyolo3d.scaling_params,
                openyolo3d.world2cam.image_resolution,
                args.top_views,
                args.min_visible_points,
                args.max_bbox_area_ratio,
                square_crops=args.square_crops,
            )
            if not selected_views:
                continue
            view_records = []
            per_view_probs = []
            for view_rank, view in enumerate(selected_views):
                image = imageio.imread(openyolo3d.world2cam.color_paths[view["frame_id"]])
                x1, y1, x2, y2 = view["bbox_xyxy"]
                visible = (keep_visible_points[view["frame_id"]].squeeze() * mask).astype(bool)
                coords = projections[view["frame_id"]][visible].astype(np.int64)
                coords_color = np.stack(
                    [
                        np.round(coords[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64),
                        np.round(coords[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64),
                    ],
                    axis=1,
                )
                crop = _make_crop_image(
                    image,
                    (x1, y1, x2, y2),
                    coords=coords_color,
                    mask_background=args.mask_background,
                    dilation_iters=args.mask_dilation_iters,
                )
                crop_name = f"pred{pred_id:04d}_view{view_rank:02d}_frame{view['frame_id']:04d}.jpg"
                crop_path = osp.join(crop_dir, crop_name)
                if args.save_crops:
                    crop.save(crop_path, quality=95)
                else:
                    crop_path = None
                alpha_mask = None
                if args.vision_encoder == "alpha_clip":
                    alpha_mask = _make_crop_alpha_mask(
                        image.shape[:2],
                        (x1, y1, x2, y2),
                        coords=coords_color,
                        dilation_iters=args.alpha_mask_dilation_iters,
                    )
                    if args.save_alpha_masks:
                        alpha_name = f"pred{pred_id:04d}_view{view_rank:02d}_frame{view['frame_id']:04d}_alpha.png"
                        alpha_path = osp.join(crop_dir, alpha_name)
                        alpha_mask.save(alpha_path)
                    else:
                        alpha_path = None
                pending_images.append(crop)
                if args.vision_encoder == "alpha_clip":
                    pending_alpha_masks.append(alpha_mask)
                pending_meta.append((pred_id, len(view_records)))
                view_record = {**view, "crop_path": crop_path}
                if args.vision_encoder == "alpha_clip" and alpha_path is not None:
                    view_record["alpha_mask_path"] = alpha_path
                view_records.append(view_record)
                if len(pending_images) >= args.batch_size:
                    encoded = _encode_pending(args, clip_state, pending_images, pending_alpha_masks, device)
                    for meta, probs, logits, similarities in zip(
                        pending_meta,
                        encoded["probs"],
                        encoded["logits"],
                        encoded["similarities"],
                    ):
                        meta_probs.setdefault(meta, probs)
                        meta_logits.setdefault(meta, logits)
                        meta_similarities.setdefault(meta, similarities)
                    pending_images = []
                    pending_alpha_masks = []
                    pending_meta = []

            records.append(
                {
                    "scene_name": scene_name,
                    "prediction_id": int(pred_id),
                    "source_kind": source_kind,
                    "source_name": source_name,
                    "pred_class_id": class_id,
                    "pred_class_name": class_name,
                    "pred_score": float(pred_scores[pred_id]),
                    "views": view_records,
                    "candidate_id": applied.get("candidate_id") if applied else None,
                }
            )

        if pending_images:
            encoded = _encode_pending(args, clip_state, pending_images, pending_alpha_masks, device)
            for meta, probs, logits, similarities in zip(
                pending_meta,
                encoded["probs"],
                encoded["logits"],
                encoded["similarities"],
            ):
                meta_probs.setdefault(meta, probs)
                meta_logits.setdefault(meta, logits)
                meta_similarities.setdefault(meta, similarities)

        for record in records:
            prob_rows = []
            logit_rows = []
            similarity_rows = []
            for view_id, view in enumerate(record["views"]):
                row = meta_probs.get((record["prediction_id"], view_id))
                if row is None:
                    continue
                logits = meta_logits.get((record["prediction_id"], view_id))
                similarities = meta_similarities.get((record["prediction_id"], view_id))
                prob_rows.append(row)
                if logits is not None:
                    logit_rows.append(logits)
                if similarities is not None:
                    similarity_rows.append(similarities)
                top_ids = np.argsort(-row)[: args.topk_classes]
                view["clip_topk"] = [
                    {"class_id": int(class_id), "class_name": labels[int(class_id)], "prob": float(row[int(class_id)])}
                    for class_id in top_ids
                ]
                if args.save_raw_scores and similarities is not None:
                    view["clip_similarities"] = [float(item) for item in similarities.tolist()]
                if args.save_raw_scores and logits is not None:
                    view["clip_logits"] = [float(item) for item in logits.tolist()]
            aggregate_probs = _aggregate_rows(prob_rows, args.aggregate, probability=True)
            aggregate_logits = _aggregate_rows(logit_rows, args.raw_aggregate, probability=False)
            aggregate_similarities = _aggregate_rows(similarity_rows, args.raw_aggregate, probability=False)
            if aggregate_probs is None:
                record["clip_probs"] = []
                record["clip_topk"] = []
                continue
            top_ids = np.argsort(-aggregate_probs)[: args.topk_classes]
            record["clip_probs"] = [float(item) for item in aggregate_probs.tolist()]
            record["clip_topk"] = [
                {"class_id": int(class_id), "class_name": labels[int(class_id)], "prob": float(aggregate_probs[int(class_id)])}
                for class_id in top_ids
            ]
            if args.save_raw_scores and aggregate_similarities is not None:
                top_similarity_ids = np.argsort(-aggregate_similarities)[: args.topk_classes]
                record["clip_similarities"] = [float(item) for item in aggregate_similarities.tolist()]
                record["clip_similarity_topk"] = [
                    {
                        "class_id": int(class_id),
                        "class_name": labels[int(class_id)],
                        "score": float(aggregate_similarities[int(class_id)]),
                    }
                    for class_id in top_similarity_ids
                ]
            if args.save_raw_scores and aggregate_logits is not None:
                top_logit_ids = np.argsort(-aggregate_logits)[: args.topk_classes]
                record["clip_logits"] = [float(item) for item in aggregate_logits.tolist()]
                record["clip_logit_topk"] = [
                    {"class_id": int(class_id), "class_name": labels[int(class_id)], "score": float(aggregate_logits[int(class_id)])}
                    for class_id in top_logit_ids
                ]

        output_json = osp.join(scene_dir, "multiview_object_clip_features.json")
        with open(output_json, "w") as f:
            json.dump(
                {
                    "scene_name": scene_name,
                    "labels": labels,
                    "features": records,
                },
                f,
                indent=2,
            )
        view_count = sum(len(record.get("views", [])) for record in records)
        summary["scenes"][scene_name] = {
            "records": len(records),
            "view_crops": view_count,
            "fusion_applied": len(applied_records),
            "skipped_rescore": skipped_rescore,
        }
        summary["total_records"] += len(records)
        summary["total_view_crops"] += view_count
        _clear_openyolo_state(openyolo3d, unload_3d_network=args.path_to_3d_masks is None)

    summary_path = osp.join(args.output_dir, "multiview_object_clip_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary['total_records']} object CLIP records to {args.output_dir}")
    print(f"Saved {summary['total_view_crops']} crops")
    print(f"Saved summary to {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--scene_names", default=None)
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--is_gt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_2d_preds", default=None)
    parser.add_argument("--save_2d_preds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--reuse_2d_preds", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_candidates", default=None)
    parser.add_argument("--output_dir", default="./output/multiview_object_clip_replica_s5_m30_bpr")
    parser.add_argument("--vision_encoder", default="clip", choices=["clip", "alpha_clip"])
    parser.add_argument("--clip_model_path", default="./pretrained/clip-vit-base-patch32")
    parser.add_argument("--alpha_clip_source", default="./_external/AlphaCLIP/AlphaCLIP-main")
    parser.add_argument("--alpha_clip_base_model", default="./pretrained/alpha_clip/checkpoints/ViT-L-14.pt")
    parser.add_argument("--alpha_clip_checkpoint", default="./pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth")
    parser.add_argument("--prompt_template", default="a photo of a {label}")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--top_views", default=3, type=int)
    parser.add_argument("--topk_classes", default=5, type=int)
    parser.add_argument("--aggregate", default="mean", choices=["mean", "max", "noisy_or"])
    parser.add_argument("--raw_aggregate", default="mean", choices=["mean", "max"])
    parser.add_argument("--save_raw_scores", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--rescore_policy",
        default="all",
        choices=["all", "low_score", "proposals", "proposals_or_low_score", "source_or_low_score"],
    )
    parser.add_argument("--rescore_max_base_score", default=None, type=float)
    parser.add_argument("--rescore_min_base_score", default=None, type=float)
    parser.add_argument("--rescore_source_kinds", default=None)
    parser.add_argument("--rescore_source_names", default=None)
    parser.add_argument("--rescore_classes", default=None)
    parser.add_argument("--min_visible_points", default=50, type=int)
    parser.add_argument("--max_bbox_area_ratio", default=0.60, type=float)
    parser.add_argument("--square_crops", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mask_background", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mask_dilation_iters", default=5, type=int)
    parser.add_argument("--save_crops", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--alpha_mask_dilation_iters", default=5, type=int)
    parser.add_argument("--save_alpha_masks", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--backprojection_min_score", default=0.40, type=float)
    parser.add_argument("--backprojection_min_seed_points", default=80, type=int)
    parser.add_argument("--backprojection_max_existing_iou", default=0.30, type=float)
    parser.add_argument("--backprojection_max_seed_in_existing_mask_ratio", default=0.70, type=float)
    parser.add_argument("--backprojection_max_proposal_iou", default=0.50, type=float)
    parser.add_argument("--backprojection_max_candidates_per_scene", default=30, type=int)
    parser.add_argument("--backprojection_score_scale", default=0.50, type=float)
    parser.add_argument("--backprojection_use_candidate_fusion_score", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_allowed_classes", default=None)
    parser.add_argument("--backprojection_blocked_classes", default=None)
    parser.add_argument("--backprojection_cc_cleanup", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_cc_radius", default=0.03, type=float)
    parser.add_argument("--backprojection_cc_min_component_points", default=50, type=int)
    parser.add_argument("--backprojection_cc_keep_topk", default=1, type=int)
    parser.add_argument("--backprojection_source_priorities", default=None)
    parser.add_argument("--backprojection_source_max_candidates", default=None)
    parser.add_argument("--backprojection_source_score_scales", default=None)
    parser.add_argument("--backprojection_superpoint_refine", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_superpoint_min_coverage", default=0.30, type=float)
    parser.add_argument("--backprojection_superpoint_max_expansion_ratio", default=2.0, type=float)
    parser.add_argument("--backprojection_superpoint_min_view_siou", default=0.0, type=float)
    parser.add_argument("--backprojection_superpoint_min_box_positive_ratio", default=0.0, type=float)
    parser.add_argument("--backprojection_superpoint_max_box_negative_ratio", default=1.0, type=float)
    parser.add_argument("--backprojection_superpoint_box_min_visible_points", default=5, type=int)
    parser.add_argument("--backprojection_superpoint_box_min_views", default=1, type=int)
    return parser


if __name__ == "__main__":
    export_features(build_parser().parse_args())

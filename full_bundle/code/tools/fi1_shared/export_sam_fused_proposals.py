"""Shared frozen Full definitions; retired experiment CLI removed."""
import os.path as osp

import sys

import imageio.v2 as imageio

import numpy as np

import torch

def _add_sam_to_path(sam_source):
    sam_source = osp.abspath(sam_source)
    if sam_source not in sys.path:
        sys.path.insert(0, sam_source)

def _load_sam_predictor(checkpoint, model_type, device, sam_source):
    _add_sam_to_path(sam_source)
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)

def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _frame_key(image_path):
    return osp.basename(image_path).split(".")[0]

def _safe_label(labels, label_id):
    if label_id < 0 or label_id >= len(labels):
        return "unknown"
    return labels[label_id]

def _clamp_box(box, width, height, min_size=2.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + min_size, min(width * 1.0, x2))
    y2 = max(y1 + min_size, min(height * 1.0, y2))
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)

def _prepare_image(path):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)

def _sam_mask_to_indices(openyolo3d, frame_idx, sam_mask, projections_np, visible_np):
    image_height, image_width = openyolo3d.world2cam.image_resolution
    visible_indices = np.flatnonzero(visible_np[frame_idx].astype(bool))
    if len(visible_indices) == 0:
        return np.asarray([], dtype=np.int64)

    coords = projections_np[frame_idx, visible_indices].astype(np.float32)
    xs = np.round(coords[:, 0] / openyolo3d.scaling_params[1]).astype(np.int64)
    ys = np.round(coords[:, 1] / openyolo3d.scaling_params[0]).astype(np.int64)
    valid = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    if not valid.any():
        return np.asarray([], dtype=np.int64)

    visible_indices = visible_indices[valid]
    xs = xs[valid]
    ys = ys[valid]
    inside = sam_mask[ys, xs].astype(bool)
    return np.unique(visible_indices[inside]).astype(np.int64)

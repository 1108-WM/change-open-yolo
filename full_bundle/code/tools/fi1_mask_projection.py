"""Shared mask RLE, visible-point projection and fixed observation ordering."""
import numpy as np


def select_frame_indices(total_frames, max_frames, frame_stride, selection):
    """从配置已加载的视角中选择帧；默认 first 保持历史 f30 行为。"""
    available = np.arange(0, total_frames, max(1, frame_stride), dtype=np.int64)
    if max_frames is None or len(available) <= max_frames:
        return available.tolist()
    if selection == "first":
        return available[:max_frames].tolist()
    if selection == "uniform":
        positions = np.rint(np.linspace(0, len(available) - 1, max_frames)).astype(np.int64)
        return available[positions].tolist()
    raise ValueError(f"未知帧采样方式：{selection}")

def mask_to_visible_points(mask, frame_index, projections, visibility, scaling_params):
    """将二维自动 mask 映射到该帧实际可见的三维点编号。"""
    point_indices = np.flatnonzero(visibility[frame_index])
    coords = projections[frame_index, point_indices].astype(np.float32)
    xs = np.round(coords[:, 0] / float(scaling_params[1])).astype(np.int64)
    ys = np.round(coords[:, 1] / float(scaling_params[0])).astype(np.int64)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    return point_indices[valid][mask[ys[valid], xs[valid]]].astype(np.int64)

def encode_binary_mask_rle(mask):
    """以 COCO 兼容的列优先游程保存二值 mask，避免保存完整二维数组。"""
    mask = np.asarray(mask, dtype=bool)
    pixels = np.asfortranarray(mask).reshape(-1, order="F").astype(np.uint8)
    changes = np.flatnonzero(np.diff(np.concatenate((np.asarray([0], dtype=np.uint8), pixels, np.asarray([0], dtype=np.uint8)))))
    counts = np.diff(np.concatenate((np.asarray([0], dtype=np.int64), changes, np.asarray([len(pixels)], dtype=np.int64))))
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(value) for value in counts]}

def decode_binary_mask_rle(payload):
    """测试和后续离线关系构建使用的 RLE 还原函数。"""
    height, width = (int(value) for value in payload["size"])
    values, value = [], False
    for count in payload["counts"]:
        values.extend([value] * int(count))
        value = not value
    return np.asarray(values, dtype=bool).reshape((height, width), order="F")

def exact_mask_relation(left, right):
    """保存同帧 mask 的连续关系，不把其解释为 parent/child 决定。"""
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return {
        "intersection_pixel_count": intersection,
        "iou": float(intersection / max(1, union)),
        "left_coverage": float(intersection / max(1, int(left.sum()))),
        "right_coverage": float(intersection / max(1, int(right.sum()))),
    }

def _select_masks(items, max_masks_per_frame):
    """SAM 内部已做 NMS；这里只限制存储预算，不做类别或 GT 选择。"""
    ordered = sorted(
        items,
        key=lambda item: (
            -float(item["predicted_iou"] * item["stability_score"]),
            int(item["area"]),
        ),
    )
    return ordered[:max_masks_per_frame]

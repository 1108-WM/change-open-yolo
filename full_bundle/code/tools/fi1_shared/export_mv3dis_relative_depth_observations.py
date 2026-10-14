"""Shared frozen Full definitions; retired experiment CLI removed."""
import numpy as np

RELATIVE_DEPTH_ALPHA = 0.05

PROJECTION_CONTRACT = "mv3dis_relative_depth_rle_reference"

def relative_depth_weights(projected_depth, measured_depth, in_image, alpha=RELATIVE_DEPTH_ALPHA):
    """Return paper visibility and Eq. (2) weights with strict inequality."""
    projected_depth = np.asarray(projected_depth, dtype=np.float64)
    measured_depth = np.asarray(measured_depth, dtype=np.float64)
    in_image = np.asarray(in_image, dtype=bool)
    if projected_depth.shape != measured_depth.shape or projected_depth.shape != in_image.shape:
        raise ValueError("depth and in-image arrays must have identical shapes")
    tolerance = float(alpha) * measured_depth
    error = np.abs(projected_depth - measured_depth)
    visible = (
        in_image
        & np.isfinite(projected_depth)
        & np.isfinite(measured_depth)
        & (projected_depth > 0.0)
        & (measured_depth > 0.0)
        & (error < tolerance)
    )
    weights = np.zeros(projected_depth.shape, dtype=np.float32)
    weights[visible] = (1.0 - error[visible] / tolerance[visible]).astype(np.float32)
    if np.any(weights[visible] <= 0.0) or np.any(weights > 1.0):
        raise ValueError("relative depth weights are outside (0, 1]")
    return visible, weights

def project_relative_depth_frame(points_h, camera_to_world, intrinsic, depth_map, alpha=RELATIVE_DEPTH_ALPHA):
    """Project one frame using MV3DIS Eqs. (1)-(2)."""
    points_h = np.asarray(points_h, dtype=np.float64)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if points_h.ndim != 2 or points_h.shape[1] != 4:
        raise ValueError("homogeneous points must have shape N x 4")
    if camera_to_world.shape != (4, 4) or intrinsic.shape != (4, 4):
        raise ValueError("pose and intrinsic must be 4 x 4")
    if depth_map.ndim != 2:
        raise ValueError("depth map must be two-dimensional")

    world_to_camera = np.linalg.inv(camera_to_world)
    camera = points_h @ world_to_camera.T
    projected = camera @ intrinsic.T
    depth = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth) & (depth > 0.0)
    pixels = np.zeros((len(points_h), 2), dtype=np.int64)
    pixels[finite, 0] = np.floor(u[finite]).astype(np.int64)
    pixels[finite, 1] = np.floor(v[finite]).astype(np.int64)
    height, width = depth_map.shape
    in_image = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    measured = np.zeros(len(points_h), dtype=np.float64)
    selected = np.flatnonzero(in_image)
    measured[selected] = depth_map[pixels[selected, 1], pixels[selected, 0]]
    visible, weights = relative_depth_weights(depth, measured, in_image, alpha)
    return pixels, visible, weights

def mask_points_from_projection(mask, pixels, visible, weights, depth_shape):
    """Map depth-grid projections into the source RGB/RLE resolution."""
    mask = np.asarray(mask, dtype=bool)
    pixels = np.asarray(pixels, dtype=np.int64)
    visible = np.asarray(visible, dtype=bool)
    weights = np.asarray(weights, dtype=np.float32)
    if pixels.shape != (len(visible), 2) or weights.shape != visible.shape:
        raise ValueError("projection arrays have inconsistent shapes")
    point_indices = np.flatnonzero(visible)
    if not len(point_indices):
        return point_indices.astype(np.int64), np.empty(0, dtype=np.float32)
    depth_height, depth_width = map(int, depth_shape)
    xs = np.rint(
        pixels[point_indices, 0].astype(np.float64) * mask.shape[1] / depth_width
    ).astype(np.int64)
    ys = np.rint(
        pixels[point_indices, 1].astype(np.float64) * mask.shape[0] / depth_height
    ).astype(np.int64)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    selected = valid & mask[ys.clip(0, mask.shape[0] - 1), xs.clip(0, mask.shape[1] - 1)]
    result_points = point_indices[selected].astype(np.int64, copy=False)
    result_weights = weights[result_points].astype(np.float32, copy=False)
    if np.any(result_weights <= 0.0) or np.any(result_weights > 1.0):
        raise ValueError("mask point weights are outside (0, 1]")
    return result_points, result_weights

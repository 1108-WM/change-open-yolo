"""Diagnostic ceilings for a frozen candidate-conditioned editable point domain.

These functions accept annotations only for post-seal diagnosis. They are not
prediction operators and must never supply actions, sampling, or inference labels.
"""
import numpy as np
from scipy.spatial import cKDTree


def halo(parent, xyz, radius=.15):
    distance, _ = cKDTree(xyz[parent]).query(xyz, distance_upper_bound=radius)
    return np.flatnonzero(np.isfinite(distance))


def ideal_repair(parent, editable, target, observable=None):
    """Keep unknown/outside points; perfectly correct only the permitted domain."""
    editable = np.asarray(editable, dtype=np.int64)
    if observable is not None:
        editable = editable[observable[editable]]
    kept = np.setdiff1d(parent, editable, assume_unique=True)
    added = np.intersect1d(editable, target, assume_unique=True)
    result = np.union1d(kept, added)
    # Evaluator ignores masks with <100 points. Retain the fewest pollution
    # points needed to remain eligible, rather than overstate a tiny-mask gain.
    if len(result) < 100:
        fill = np.setdiff1d(parent, result, assume_unique=True)
        result = np.union1d(result, fill[:100-len(result)])
    return result


def unit_ceiling(parent, editable, labels, target_id, units):
    """Exact unconstrained IoU ceiling for whole units clipped to editable domain.

IoU = (fixed true + selected true)/(target size + fixed false + selected
false). Its optimum is a prefix sorted by true/false ratio. Clipping a unit
at the editable boundary is optimistic for an indivisible full superpoint.
The returned ratio is an upper bound if its mask has fewer than 100 points.
"""
    outside = np.setdiff1d(parent, editable, assume_unique=True)
    a = int(np.sum(labels[outside] == target_id))
    b = len(outside)-a
    n = int(np.sum(labels == target_id))
    _, inverse = np.unique(units[editable], axis=0, return_inverse=True)
    total = np.bincount(inverse)
    true = np.bincount(inverse, weights=(labels[editable] == target_id)).astype(np.int64)
    false = total-true
    ratio = np.divide(true, false, out=np.full(len(true), np.inf), where=false > 0)
    order = np.argsort(-ratio, kind='stable')
    hits = np.r_[a, a+np.cumsum(true[order])]
    noise = np.r_[b, b+np.cumsum(false[order])]
    iou = hits / (n+noise)
    k = int(np.argmax(iou))
    return dict(iou=float(iou[k]), points=int(hits[k]+noise[k]),
                evaluator_eligible=bool(hits[k]+noise[k] >= 100),
                mixed_target_background_units=int(np.sum((true > 0) & (false > 0))),
                target_points_in_mixed_units=int(true[(true > 0) & (false > 0)].sum()))


def adapt_intrinsic(intrinsic, rgb_hw, depth_hw):
    """Match the frozen WORLD_2_CAM depth-resolution convention exactly."""
    if rgb_hw == depth_hw:
        return intrinsic.copy()
    h, w = rgb_hw; dh, dw = depth_hw
    resized_width = int(np.floor(dw*h/w))
    result = intrinsic.copy()
    result[0, 0] *= resized_width/h
    result[1, 1] *= dw/w
    result[0, 2] *= (dh-1)/(h-1)
    result[1, 2] *= (dw-1)/(w-1)
    return result


def reliable_visibility(xyz, pose, intrinsic, sensor, tolerance=.02):
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera = homogeneous @ np.linalg.inv(pose).T
    projected = camera @ intrinsic.T
    finite = np.isfinite(projected).all(axis=1) & (camera[:, 2] > 0)
    pixel = np.zeros((len(xyz), 2), np.int64)
    pixel[finite] = np.trunc(projected[finite, :2]/projected[finite, 2, None]).astype(np.int64)
    dh, dw = sensor.shape
    # Frozen frontend excludes the first row/column, not just off-image pixels.
    inside = finite & (pixel[:, 0] > 0) & (pixel[:, 0] < dw) & (pixel[:, 1] > 0) & (pixel[:, 1] < dh)
    ids = np.flatnonzero(inside)
    measured = sensor[pixel[ids, 1], pixel[ids, 0]]
    result = np.zeros(len(xyz), bool)
    result[ids] = (measured > 0) & (np.abs(measured-camera[ids, 2]) <= tolerance)
    return result


def add_group_observation(first, counts, visible, group):
    """Saturate at two distinct physical groups; repeated frames count once."""
    new = visible & (counts == 0)
    different = visible & (counts == 1) & (first != group)
    first[new] = group
    counts[new] = 1
    counts[different] = 2


def physical_groups(poses, translation=.1, angle_degrees=15.):
    representatives = []; groups = {}
    for frame in sorted(poses):
        pose = poses[frame]
        group = None
        if representatives:
            reps = np.asarray(representatives)
            near = np.flatnonzero(np.linalg.norm(reps[:, :3, 3]-pose[:3, 3], axis=1) < translation)
            if len(near):
                trace = np.einsum('gij,ij->g', reps[near, :3, :3], pose[:3, :3])
                angles = np.degrees(np.arccos(np.clip((trace-1)/2, -1, 1)))
                eligible = near[angles < angle_degrees]
                if len(eligible):
                    group = int(eligible[0])
        if group is None:
            group = len(representatives); representatives.append(pose)
        groups[frame] = group
    return groups

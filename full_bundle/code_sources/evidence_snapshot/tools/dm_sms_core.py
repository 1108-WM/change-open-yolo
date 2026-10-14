#!/usr/bin/env python3
"""Pure, GT-free primitives for the preregistered DM-SMS-1A experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


SMS_EPSILON = 1e-6


@dataclass(frozen=True)
class SMSResult:
    scores: np.ndarray
    valid: np.ndarray
    top_classes: np.ndarray
    class_means: np.ndarray
    class_stds: np.ndarray


def geometry_hash(point_indices: np.ndarray) -> str:
    """Hash a canonical sorted set of scene point indices."""
    points = np.unique(np.asarray(point_indices, dtype=np.int64))
    return hashlib.sha1(points.tobytes()).hexdigest()


def exact_geometry_groups(masks: np.ndarray) -> list[list[int]]:
    """Group identical boolean mask columns without using class or score."""
    values = np.asarray(masks, dtype=bool)
    if values.ndim != 2:
        raise ValueError("masks must have shape [point, prediction]")
    groups: dict[bytes, list[int]] = {}
    for candidate_id in range(values.shape[1]):
        packed = np.packbits(values[:, candidate_id]).tobytes()
        groups.setdefault(packed, []).append(candidate_id)
    return list(groups.values())


def canonical_member(
    member_ids: list[int],
    scores: np.ndarray,
    source_ranks: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
) -> int:
    """Choose the preregistered canonical member of an exact geometry group.

    The formal cross-source tie order is frozen score descending, then source
    rank (native, track, pair-union), then the source-local candidate id.  The
    optional arrays preserve the original two-argument helper behavior for
    callers that operate inside a single source.
    """
    if not member_ids:
        raise ValueError("geometry group has no members")
    values = np.asarray(scores, dtype=np.float64)
    ranks = (
        np.zeros(len(values), dtype=np.int64)
        if source_ranks is None else np.asarray(source_ranks, dtype=np.int64)
    )
    local_ids = (
        np.arange(len(values), dtype=np.int64)
        if candidate_ids is None else np.asarray(candidate_ids, dtype=np.int64)
    )
    if ranks.shape != values.shape or local_ids.shape != values.shape:
        raise ValueError("scores, source_ranks, and candidate_ids must have equal shape")
    return min(
        (int(item) for item in member_ids),
        key=lambda item: (
            -float(values[item]), int(ranks[item]), int(local_ids[item]), item,
        ),
    )


def visible_ratio_multiscale_feature(features: np.ndarray, visible_ratios: np.ndarray) -> np.ndarray:
    """Implement Eq. (3): sum over views and scales, weighted by visibility."""
    values = np.asarray(features, dtype=np.float64)
    ratios = np.asarray(visible_ratios, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("features must have shape [view, scale, dimension]")
    if ratios.shape != (values.shape[0],):
        raise ValueError("visible_ratios must have one value per view")
    if not np.isfinite(values).all() or not np.isfinite(ratios).all():
        raise ValueError("features and visible ratios must be finite")
    if (ratios < 0).any() or (ratios > 1).any():
        raise ValueError("visible ratios must be in [0, 1]")
    aggregated = (values * ratios[:, None, None]).sum(axis=(0, 1))
    norm = float(np.linalg.norm(aggregated))
    if norm <= 0.0:
        raise ValueError("aggregated feature has zero norm")
    return (aggregated / norm).astype(np.float32)


def compute_sms(similarities: np.ndarray, epsilon: float = SMS_EPSILON) -> SMSResult:
    """Compute Details Matter SMS over all unique proposals in one scene.

    For proposal k, statistics for its top class c_max are taken from the
    complete scene population L[:, c_max], not only proposals whose top class
    is c_max. Classes with degenerate population variance are marked invalid;
    the preregistered filtering rule conservatively retains those proposals.
    """
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("similarities must have non-empty shape [proposal, class]")
    if not np.isfinite(values).all():
        raise ValueError("similarities must be finite")
    means = values.mean(axis=0)
    stds = np.sqrt(np.mean((values - means[None, :]) ** 2, axis=0))
    top = np.argmax(values, axis=1).astype(np.int64)
    selected_stds = stds[top]
    valid = selected_stds > float(epsilon)
    scores = np.zeros(values.shape[0], dtype=np.float64)
    row_ids = np.arange(values.shape[0])
    scores[valid] = (
        values[row_ids[valid], top[valid]] - means[top[valid]]
    ) / selected_stds[valid]
    return SMSResult(
        scores=scores.astype(np.float32),
        valid=valid.astype(bool),
        top_classes=top,
        class_means=means.astype(np.float32),
        class_stds=stds.astype(np.float32),
    )


def sms_keep_mask(result: SMSResult, threshold: float = 0.0) -> np.ndarray:
    """Delete only valid SMS values strictly below the frozen threshold."""
    return np.logical_or(~result.valid, result.scores >= float(threshold))

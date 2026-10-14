"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import hashlib

import json

import math

from collections import defaultdict

from pathlib import Path

import numpy as np

EXPECTED_OOF_SHA256 = "a25f737e44da7d1409bbdc37f52c90a943d1ce4ab21901a0de1e974db2ca40ef"

def load_oof_predictions(path: Path) -> dict[tuple[str, str, int], dict]:
    predictions = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["scene_name"]),
            str(row["candidate_source"]),
            int(row["candidate_id"]),
        )
        if key in predictions:
            raise ValueError(f"折外预测候选身份重复：{key}")
        value = float(row["predictions"]["D_plus_gvc"]["q"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"折外预测质量越界：{key}={value}")
        predictions[key] = {
            "q": value,
            "label_best_gt_iou": float(row["label_best_gt_iou"]),
            "label_valid_iou25": int(row["label_valid_iou25"]),
            "label_valid_iou50": int(row["label_valid_iou50"]),
        }
    if not predictions:
        raise ValueError("折外预测文件为空")
    return predictions

def geometry_groups_and_audit(
    masks: np.ndarray,
    point_counts: list[int],
    expected_group_sizes: list[int] | None,
) -> tuple[list[list[int]], dict]:
    """用有界内存计算每个基线候选的完全相同掩码分组。"""
    if masks.ndim != 2 or masks.shape[1] != len(point_counts):
        raise ValueError("基线掩码数量与候选账本不一致")
    if expected_group_sizes is not None and len(expected_group_sizes) != masks.shape[1]:
        raise ValueError("完全相同掩码组大小字段数量不一致")
    digests = [None] * masks.shape[1]
    actual_counts = np.empty(masks.shape[1], dtype=np.int64)
    whole_digest = hashlib.sha256()
    whole_digest.update(json.dumps({"shape": list(masks.shape), "dtype": str(masks.dtype)}).encode())
    for start in range(0, masks.shape[1], 16):
        stop = min(start + 16, masks.shape[1])
        block = np.asarray(masks[:, start:stop], dtype=bool)
        packed = np.packbits(block, axis=0)
        whole_digest.update(start.to_bytes(8, "little"))
        whole_digest.update(np.packbits(block, axis=None).tobytes())
        actual_counts[start:stop] = np.count_nonzero(block, axis=0)
        for column in range(stop - start):
            digests[start + column] = hashlib.sha256(
                np.ascontiguousarray(packed[:, column]).tobytes()
            ).hexdigest()
    if not np.array_equal(actual_counts, np.asarray(point_counts, dtype=np.int64)):
        bad = np.flatnonzero(actual_counts != np.asarray(point_counts, dtype=np.int64))[:5]
        raise ValueError(f"基线掩码点数与账本不一致：{bad.tolist()}")
    grouped = defaultdict(list)
    for candidate_id, digest in enumerate(digests):
        grouped[digest].append(candidate_id)
    groups = sorted(grouped.values(), key=lambda members: members[0])
    actual_group_sizes = np.empty(masks.shape[1], dtype=np.int64)
    for members in groups:
        actual_group_sizes[members] = len(members)
    if expected_group_sizes is not None and not np.array_equal(
        actual_group_sizes, np.asarray(expected_group_sizes, dtype=np.int64)
    ):
        bad = np.flatnonzero(
            actual_group_sizes != np.asarray(expected_group_sizes, dtype=np.int64)
        )[:5]
        raise ValueError(f"完全相同掩码组大小与训练账本不一致：{bad.tolist()}")
    return groups, {
        "candidate_count": masks.shape[1],
        "geometry_group_count": len(groups),
        "singleton_group_count": sum(len(members) == 1 for members in groups),
        "maximum_group_size": max(len(members) for members in groups),
        "canonical_mask_element_sha256": whole_digest.hexdigest(),
    }

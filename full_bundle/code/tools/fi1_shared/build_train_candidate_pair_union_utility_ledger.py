"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import math

import numpy as np

from tools.fi1_shared.evaluate_candidate_quality_reranking_class_agnostic_ap import MIN_REGION_SIZE

LEARNED_SCORE_TOKENS = (
    "C_plus_geometry_track_structure",
    "D_plus_gvc",
)

def proposal_points(kind: str, track: np.ndarray, native: np.ndarray) -> np.ndarray:
    if kind == "pair_union":
        return np.union1d(track, native).astype(np.int64)
    if kind == "pair_intersection":
        return np.intersect1d(track, native, assume_unique=True).astype(np.int64)
    raise ValueError(f"unsupported pair proposal kind: {kind}")

def _pure_inference_relation_features(features: dict) -> dict[str, float]:
    output = {}
    for name, value in features.items():
        if any(token in name for token in LEARNED_SCORE_TOKENS):
            continue
        if isinstance(value, bool):
            output[f"relation__{name}"] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[f"relation__{name}"] = float(value)
    if not output:
        raise ValueError("relation row has no pure inference features")
    return output

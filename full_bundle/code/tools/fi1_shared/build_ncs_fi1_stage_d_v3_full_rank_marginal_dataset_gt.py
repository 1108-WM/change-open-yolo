"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from tools.fi1_shared.build_ncs_fi1_stage_d_v2_rank_marginal_dataset_gt import FEATURE_NAMES as V2_FEATURE_NAMES

EXTRA_FEATURE_NAMES = (
    "rank_prefix_native_count", "rank_prefix_track_count",
    "rank_prefix_original_union_count", "rank_prefix_refined_union_count",
    "rank_prefix_append_fraction",
)

FEATURE_NAMES = (*V2_FEATURE_NAMES, *EXTRA_FEATURE_NAMES)

def candidate_key(scene: str, union_candidate_id: int, variant: str) -> str:
    return f"{scene}:union:{union_candidate_id:04d}:{variant}"

def append_precedes(left: dict, right: dict) -> bool:
    """Frozen descending-score, ascending-key order among append candidates."""
    left_score = float(left["reference_score"])
    right_score = float(right["reference_score"])
    return left_score > right_score or (
        left_score == right_score and str(left["candidate_key"]) < str(right["candidate_key"])
    )

def full_rank_prefix(
    baseline: list[dict], append_candidates: list[dict], current: dict,
) -> list[dict]:
    score = float(current["reference_score"])
    return [entry for entry in baseline if float(entry["reference_score"]) >= score] + [
        entry for entry in append_candidates
        if entry["candidate_key"] != current["candidate_key"] and append_precedes(entry, current)
    ]

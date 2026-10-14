"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

from tools.fi1_shared.build_ncs_fi1_stage_d_marginal_gain_dataset_gt import FEATURE_NAMES as V1_FEATURE_NAMES, label_candidate

def _prefix_name(name: str) -> str:
    return "rank_prefix_" + name[len("baseline_"):] if name.startswith("baseline_") else name

FEATURE_NAMES = (
    *tuple(_prefix_name(name) for name in V1_FEATURE_NAMES),
    "rank_prefix_geometry_count",
    "rank_prefix_fraction_of_native_track",
)

def rank_labels(candidate_iou: dict[int, float], prefix_best: dict[int, float]) -> dict:
    labels = label_candidate(candidate_iou, prefix_best)
    labels["rank_conditioned_marginal_iou_gain"] = labels.pop("marginal_iou_gain")
    labels["rank_conditioned_marginal_q_gain"] = labels.pop("marginal_q_gain")
    labels["rank_conditioned_marginal_soft_quality_gain"] = labels.pop("marginal_soft_quality_gain")
    labels["rank_prefix_best_iou_for_selected_gain_target"] = labels.pop(
        "existing_best_iou_for_selected_gain_target"
    )
    return labels

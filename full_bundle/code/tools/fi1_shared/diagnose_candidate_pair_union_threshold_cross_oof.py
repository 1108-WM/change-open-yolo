"""Shared frozen Full definitions; retired experiment CLI removed."""
from __future__ import annotations

import numpy as np

VERSIONS = {
    "pair_union": "official100_pair_union_threshold_cross_oof_v1",
    "pair_intersection": "official100_pair_intersection_threshold_cross_oof_v1",
}

def prior_correct_balanced_probability(
    balanced_probability: np.ndarray, natural_positive_rate: float,
) -> np.ndarray:
    """Undo the artificial 50/50 class prior used for rare-state fitting."""
    probability = np.clip(np.asarray(balanced_probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    prior = min(1 - 1e-6, max(1e-6, float(natural_positive_rate)))
    odds = probability / (1.0 - probability)
    corrected_odds = odds * prior / (1.0 - prior)
    return corrected_odds / (1.0 + corrected_odds)

import pytest

from tools.run_frozen_safety60_champion_pair_union_ap import (
    corrected_pair_union_probability,
)


def test_pair_union_probability_restores_full_training_natural_prior():
    metadata = {"training_component_balanced_natural_positive_rate": 0.05}
    assert corrected_pair_union_probability(0.5, metadata) == pytest.approx(0.05)


def test_pair_union_probability_rejects_missing_natural_prior():
    with pytest.raises(ValueError, match="natural positive rate"):
        corrected_pair_union_probability(0.5, {})

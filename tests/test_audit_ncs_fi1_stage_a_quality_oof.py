import pytest

from tools.audit_ncs_fi1_stage_a_quality_oof import _basic_metrics, _close


def test_oof_audit_metric_recomputation_is_exact_for_basic_fields():
    result = _basic_metrics([0.0, 1.0], [0.25, 0.75])
    assert result["count"] == 2
    assert result["mae"] == pytest.approx(0.25)
    assert result["rmse"] == pytest.approx(0.25)
    assert result["absolute_mean_bias"] == 0.0


def test_oof_audit_close_rejects_missing_or_different_values():
    assert _close(0.1, 0.1 + 1e-13)
    assert not _close(None, 0.1)
    assert not _close(0.1, 0.2)

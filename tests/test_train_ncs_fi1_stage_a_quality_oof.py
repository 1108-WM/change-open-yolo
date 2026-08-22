import numpy as np
import pytest

from tools.train_ncs_fi1_stage_a_quality_oof import (
    _calibrate,
    _fit_calibrator,
    metrics,
    reliability,
)


def test_reliability_uses_fixed_ten_equal_width_bins():
    result = reliability(np.asarray([0.0, 1.0]), np.asarray([0.05, 0.95]))
    assert len(result["bins"]) == 10
    assert sum(row["count"] for row in result["bins"]) == 2
    assert result["expected_calibration_error"] == pytest.approx(0.05)


def test_degenerate_calibration_has_explicit_constant_fallback():
    calibrator, kind = _fit_calibrator(np.asarray([0.2, 0.2]), np.asarray([0.0, 1.0]))
    assert kind == "constant_calibration_fallback"
    assert _calibrate(calibrator, np.asarray([0.1, 0.9])).tolist() == [0.5, 0.5]


def test_metrics_report_shared_quality_error_and_rank():
    result = metrics(np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.4, 0.9]))
    assert result["mae"] == pytest.approx(0.0666666667)
    assert result["rmse"] > 0.0
    assert result["spearman"] == pytest.approx(1.0)

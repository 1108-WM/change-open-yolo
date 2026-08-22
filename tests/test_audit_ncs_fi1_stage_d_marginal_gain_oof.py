from tools.audit_ncs_fi1_stage_d_marginal_gain_oof import _close


def test_stage_d_result_audit_close_is_strict():
    assert _close(0.2, 0.2 + 1e-13)
    assert not _close(0.2, 0.2 + 1e-8)
    assert _close(None, None)
    assert not _close(None, 0.0)

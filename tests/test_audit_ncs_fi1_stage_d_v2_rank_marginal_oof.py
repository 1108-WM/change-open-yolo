from tools.audit_ncs_fi1_stage_d_v2_rank_marginal_oof import _close


def test_stage_d_v2_result_audit_close_is_strict_and_handles_null():
    assert _close(None, None)
    assert _close(0.3, 0.3 + 1e-13)
    assert not _close(None, 0.0)
    assert not _close(0.3, 0.3 + 1e-8)

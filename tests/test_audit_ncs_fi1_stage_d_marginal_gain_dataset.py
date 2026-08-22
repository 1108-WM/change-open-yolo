from tools.audit_ncs_fi1_stage_d_marginal_gain_dataset import _close


def test_stage_d_dataset_audit_close_is_strict_and_handles_none():
    assert _close(0.1, 0.1 + 1e-13)
    assert not _close(0.1, 0.1 + 1e-8)
    assert _close(None, None)
    assert not _close(None, 0.0)

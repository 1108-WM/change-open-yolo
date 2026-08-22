from tools.audit_ncs_fi1_stage_d_v2_rank_marginal_dataset import _close


def test_stage_d_v2_dataset_audit_close_handles_null_and_is_strict():
    assert _close(None, None)
    assert _close(0.1, 0.1 + 1e-13)
    assert not _close(None, 0.0)
    assert not _close(0.1, 0.1 + 1e-8)

from tools.audit_ncs_fi1_stage_b_relation_rerank_plan import _close


def test_stage_b_audit_close_accepts_serialization_noise_only():
    assert _close(0.25, 0.25 + 1e-13)
    assert not _close(0.25, 0.25 + 1e-9)
    assert not _close(None, 0.25)

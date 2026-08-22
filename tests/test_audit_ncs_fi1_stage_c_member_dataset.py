from tools.audit_ncs_fi1_stage_c_member_dataset import _close


def test_stage_c_dataset_audit_close_accepts_only_serialization_noise():
    assert _close(0.5, 0.5 + 1e-13)
    assert not _close(0.5, 0.5 + 1e-8)
    assert not _close(None, 0.5)

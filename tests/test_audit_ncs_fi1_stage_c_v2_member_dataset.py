from tools.audit_ncs_fi1_stage_c_v2_member_dataset import _close


def test_c_v2_dataset_audit_close_is_strict():
    assert _close(0.2, 0.2 + 1e-13)
    assert not _close(0.2, 0.2 + 1e-8)
    assert not _close(None, 0.2)

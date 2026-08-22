from tools.audit_ncs_fi1_stage_c_v2_refinement_oof import _close


def test_c_v2_result_audit_close_is_strict():
    assert _close(-0.1, -0.1 + 1e-13)
    assert not _close(-0.1, -0.1 + 1e-8)
    assert not _close(None, -0.1)

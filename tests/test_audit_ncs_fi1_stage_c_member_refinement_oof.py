from tools.audit_ncs_fi1_stage_c_member_refinement_oof import _close


def test_stage_c_result_audit_close_rejects_material_difference():
    assert _close(0.1, 0.1 + 1e-13)
    assert not _close(0.1, 0.1 + 1e-7)
    assert not _close(None, 0.1)

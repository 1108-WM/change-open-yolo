from tools.audit_ncs_fi1_stage_a_quality_dataset import _forbidden_feature_name


def test_feature_leakage_check_allows_scene_normalized_geometry_but_rejects_identity():
    assert not _forbidden_feature_name("point_fraction_of_scene")
    assert not _forbidden_feature_name("public_track_minus_native_gvc")
    assert _forbidden_feature_name("scene_name")
    assert _forbidden_feature_name("label_quality_q")
    assert _forbidden_feature_name("best_iou")

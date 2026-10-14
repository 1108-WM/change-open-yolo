import numpy as np
import pytest

from tools.run_fi1_core_ablation_val312 import AccessGuard, reference_selection, MODEL_HASH


def test_validation_GT_requires_seal_and_exact_whitelist(tmp_path):
    gt = tmp_path / 'ground_truth/scene0011_00.txt'
    guard = AccessGuard(['scene0011_00'], [gt])
    with pytest.raises(PermissionError, match='all936'):
        guard.check(gt)
    guard.gt_enabled = True
    guard.check(gt)
    with pytest.raises(PermissionError, match='outside frozen'):
        guard.check(tmp_path / 'ground_truth/scene0000_00.txt')
    with pytest.raises(PermissionError, match='validation whitelist'):
        guard.check(tmp_path / 'other/ground_truth/scene0011_00.txt')


def test_official_test_and_symlink_are_blocked(tmp_path):
    guard = AccessGuard(['scene0011_00'], [])
    for name in ['test/scene0011_00.npz','scannet200_test/scene0011_00.txt','scannetv2_test.txt']:
        with pytest.raises(PermissionError, match='official final test'):
            guard.check(tmp_path/name)
    link=tmp_path/'innocent.npz'; link.symlink_to(tmp_path/'test/scene0011_00.npz')
    with pytest.raises(PermissionError, match='official final test'):
        guard.check(link)


def test_GT_symlink_cannot_bypass_phase_guard(tmp_path):
    directory=tmp_path/'ground_truth'; directory.mkdir()
    link=directory/'scene0011_00.txt'; link.symlink_to(tmp_path/'payload.txt')
    guard=AccessGuard(['scene0011_00'],[link])
    with pytest.raises(PermissionError,match='all936'):
        guard.check(link)
    guard.gt_enabled=True; guard.check(link)


def test_cached_control_extends_K600_to_K800_without_rescoring():
    members=[dict(geometry_key=f'g{i:04d}',class_index=0,score=.5,has_original_native_alias=False)
             for i in range(900)]
    scores=np.arange(900,dtype=float)/1000
    old=[{**members[i],'score':float(scores[i])} for i in range(899,299,-1)]
    reference=reference_selection(members,scores,old)
    assert len(reference)==800 and reference[:600]==old
    assert reference[-1]['geometry_key']=='g0100'
    with pytest.raises(ValueError,match='K600 control'):
        reference_selection(members,scores,list(reversed(old)))


def test_model_identity_is_exact_current_GD10_B26():
    assert MODEL_HASH=='48f1d79df57bd9e2ba70802ed18494d6390c5bbdfec6f990fc259449d84ea43a'

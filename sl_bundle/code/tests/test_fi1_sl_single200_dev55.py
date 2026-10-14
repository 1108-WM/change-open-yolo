from pathlib import Path
import pytest
from tools import run_fi1_sl_single200_dev55 as run


def contract():
    return dict(train_scenes=['scene0002_00'],scenes=['scene0005_00'],code={},
        bindings={'scene0005_00':dict(gt_path='/tmp/ground_truth_dev55/scene0005_00.txt')})


def test_physical_scene_identity_catches_other_scan():
    assert run.mains(['scene0005_00'])==run.mains(['scene0005_01'])
    assert not run.mains(['scene0002_00'])&run.mains(['scene0005_00'])


def test_original_optimization_loop_is_unchanged():
    assert run.training_loop_ast(run.fit)==run.training_loop_ast(run.parent.original.fit)


def test_guard_separates_validation_targets_and_train_only_labels():
    g=run.Guard(contract(),'fit')
    g.check(run.L/'dataset/scene0002_00/labels.npz')
    for p in (run.L/'dataset/scene0005_00/labels.npz',Path('/tmp/ground_truth_dev55/scene0005_00.txt'),
              Path('/tmp/evaluation/scene0005_00/match.json')):
        with pytest.raises(PermissionError):g.check(p)
    for mode in ('predict','verify'):
        with pytest.raises(PermissionError):run.Guard(contract(),mode).check(run.L/'dataset/scene0002_00/labels.npz')


def test_raw_gt_requires_sealed_predictions_and_bound_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    g=run.Guard(contract(),'evaluate')
    with pytest.raises(PermissionError):g.check('/tmp/ground_truth_dev55/scene0005_00.txt')
    (tmp_path/'predictions_frozen.json').write_text('{}')
    g.check('/tmp/ground_truth_dev55/scene0005_00.txt')
    with pytest.raises(PermissionError):g.check('/tmp/ground_truth_dev55/scene0006_00.txt')

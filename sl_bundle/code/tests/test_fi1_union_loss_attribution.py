import numpy as np
import pytest
from tools.fi1_union_loss_attribution import factorial_selections, member_diagnostics, key


def row(g, c=0, native=False):
    return dict(geometry_key=g, class_index=c, has_original_native_alias=native)


def test_new_class_on_existing_pointset_is_an_added_member():
    old = [row('g', 0)]; new = [row('g', 0), row('g', 1)]
    out = factorial_selections(old, [.8], new, [.7, .9], budget=1)
    assert key(out['A'][0]) == key(out['C'][0]) == ('g', 0)
    assert key(out['B'][0]) == key(out['D'][0]) == ('g', 1)


def test_C_keeps_full_pool_scores_even_when_added_members_cannot_output():
    old = [row('x'), row('y')]; new = old + [row('u')]
    out = factorial_selections(old, [.9, .8], new, [.1, .7, .6], budget=1)
    assert out['A'][0]['geometry_key'] == 'x'
    assert out['B'][0]['geometry_key'] == 'x'
    assert out['C'][0]['geometry_key'] == out['D'][0]['geometry_key'] == 'y'
    assert out['C'][0]['score'] == .7


def test_all_added_members_compete_and_B_D_use_identical_added_scores():
    old = [row('x')]; new = old + [row('u'), row('v')]
    out = factorial_selections(old, [.8], new, [.95, .85, .7], budget=2)
    assert [m['geometry_key'] for m in out['B']] == ['u', 'x']
    assert [m['geometry_key'] for m in out['D']] == ['x', 'u']
    assert next(m['score'] for m in out['B'] if m['geometry_key'] == 'u') == .85
    assert next(m['score'] for m in out['D'] if m['geometry_key'] == 'u') == .85


def test_no_added_members_and_no_score_changes_collapse_all_controls():
    old = [row('z'), row('a', native=True)]
    out = factorial_selections(old, [.5, .5], old, [.5, .5], budget=1)
    assert all(v == out['A'] for v in out.values()) and out['A'][0]['geometry_key'] == 'a'


def test_identity_and_tie_errors_are_rejected():
    with pytest.raises(ValueError, match='subset'):
        factorial_selections([row('a')], [.1], [row('b')], [.2])
    with pytest.raises(ValueError, match='duplicate'):
        factorial_selections([row('a'), row('a')], [.1, .2], [row('a')], [.2])
    with pytest.raises(ValueError, match='tie'):
        factorial_selections([row('a', native=True)], [.1], [row('a')], [.2])


def test_scores_are_frozen_finite_probabilities():
    with pytest.raises(ValueError, match='scores'):
        factorial_selections([row('a')], [np.nan], [row('a')], [.2])


def test_descriptive_diagnostics_distinguish_wrong_class_duplicate_and_small():
    keys = ['a', 'u', 'small']; sizes = np.array([200, 200, 99])
    iou = np.array([[.8, .1], [.9, .2], [.9, 0]])
    inter = np.array([[180, 20], [190, 110], [99, 0]])
    selected = [row('a', 0), row('u', 0), row('u', 1), row('small', 0)]
    diag, cov = member_diagnostics(keys, sizes, iou, inter, [0, 1], selected, [row('a')], {'a', 'small'})
    d = diag['all']['thresholds']['0.75']
    assert d['correct_range'] == 2 and d['unique_correct_GT'] == 1 and d['redundant_correct_members'] == 1
    assert d['wrong_class_good_range'] == 1 and diag['all']['small_ignored'] == 1
    assert diag['new_member']['multi_significant_GT'] == 2
    assert cov['0.75'] == {0}


def test_threshold_is_strict_and_no_good_range_is_not_called_FP():
    diag, _ = member_diagnostics(['a'], np.array([100]), np.array([[.75]]), np.array([[100]]), [0],
                               [row('a')], [row('a')], {'a'})
    assert diag['all']['thresholds']['0.75']['no_range_at_threshold'] == 1


def test_training_guard_forbids_validation_artifacts_and_final_test(tmp_path):
    from tools.run_fi1_union_loss_attribution_train90 import Guard
    guard = Guard(['scene0000_00'], [])
    with pytest.raises(PermissionError, match='validation312'):
        guard.check('/tmp/.local/fi1_any_val312/results.json')
    with pytest.raises(PermissionError, match='heldout'):
        guard.check('/media/jia/scannet_validation_stream/scene0000_00/data.npz')
    with pytest.raises(PermissionError, match='final test'):
        guard.check('/tmp/test/data.npz')


def test_training_GT_requires_frozen_phase_and_exact_whitelist(tmp_path):
    from tools.run_fi1_union_loss_attribution_train90 import Guard
    gt = tmp_path / 'ground_truth/scene0000_00.txt'
    guard = Guard(['scene0000_00'], [str(gt)])
    with pytest.raises(PermissionError, match='GT requires'): guard.check(gt)
    guard.gt_enabled = True; guard.check(gt)
    with pytest.raises(PermissionError): guard.check(tmp_path / 'ground_truth/scene0001_00.txt')

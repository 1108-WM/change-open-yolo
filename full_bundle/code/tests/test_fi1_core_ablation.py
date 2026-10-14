import copy
import numpy as np
import pytest

from tools.fi1_core_ablation import source_pool, features_and_members, direct_selection, learned_selection
from tools.fi1_r3_core import pair_overlap, independent_context


def geom(key, sources, refined=False):
    return dict(geometry_key='scene0000_00:geometry:' + key, points_sha256=key,
                point_count=3, legacy_geometry_keys=[],
                legacy_members=[dict(candidate_source=s, candidate_id=i, frozen_class_index=0)
                                for i, s in enumerate(sources)],
                dv3_plans=[dict(append_only=True, plan_key=key + ':refine', frozen_class_index=1)]
                if refined else [])


def evidence(g):
    v = np.zeros(198, dtype=int); v[:2] = [3, 1]
    return dict(geometry_key=g['geometry_key'], points_sha256=g['points_sha256'],
                yolo=dict(acquisition_complete=True, available=True, votes=v),
                alpha=dict(acquisition_complete=True, available=False, missing_reason='no view'))


def test_shared_geometry_retained_without_union_aliases_and_input_unchanged():
    gs = [geom('a', ['native', 'pair_union'], True), geom('b', ['track']),
          geom('c', ['pair_union']), geom('d', [], True)]
    before = copy.deepcopy(gs)
    pool = source_pool(gs)
    assert gs == before
    assert [g['points_sha256'] for g in pool] == ['a', 'b']
    assert [m['candidate_source'] for m in pool[0]['legacy_members']] == ['native']
    assert pool[0]['dv3_plans'] == []


def test_native_only_shared_pointset_loses_nonnative_class_expansion():
    g = geom('a', ['native', 'pair_union'])
    p = {g['geometry_key']: np.array([0, 1, 2])}
    e = {g['geometry_key']: evidence(g)}
    full, _ = features_and_members([g], p, e, 4)
    nt, x = features_and_members(source_pool([g]), p, e, 4)
    assert [m['class_index'] for m in full] == [0, 1]
    assert [m['class_index'] for m in nt] == [0]
    assert np.array_equal(x[0, 14:18], [1, 0, 0, 0])


def test_context_is_recomputed_after_removing_neighbor():
    gs = [geom('a', ['track']), geom('b', ['pair_union'])]
    p = {g['geometry_key']: np.array([0, 1, 2]) for g in gs}
    e = {g['geometry_key']: evidence(g) for g in gs}
    full, x = features_and_members(gs, p, e, 4)
    iou, cover = pair_overlap(list(p.values()), 4)
    assert np.array_equal(x[:, 20:], independent_context(full, list(p), iou, cover))
    pool = source_pool(gs)
    nt, nx = features_and_members(pool, {pool[0]['geometry_key']: p[pool[0]['geometry_key']]},
                                  {pool[0]['geometry_key']: e[pool[0]['geometry_key']]}, 4)
    assert np.any(x[:, 20:] != 0)
    assert np.all(nx[:, 20:] == 0)


def test_direct_scores_and_membership_remain_exact_and_budget_refills():
    members = [dict(geometry_key=str(i), class_index=0, score=float(i)/1000,
                    has_original_native_alias=False) for i in range(900)]
    before = copy.deepcopy(members)
    chosen = direct_selection(members)
    assert len(chosen) == 800 and chosen[0]['geometry_key'] == '899'
    assert members == before
    assert len(direct_selection(members[:3])) == 3


def test_learned_head_uses_changed_features_not_old_member_scores():
    class Model:
        n_features_in_ = 26
        def predict(self, x): return x[:, 20]
    members = [dict(geometry_key=str(i), class_index=0, score=float(i),
                    has_original_native_alias=False) for i in range(2)]
    x = np.zeros((2, 26)); x[:, 20] = [.9, .1]
    selected, scores = learned_selection(members, x, Model())
    assert selected[0]['geometry_key'] == '0'
    assert np.array_equal(scores, [.9, .1])


def test_invalid_source_rejected():
    with pytest.raises(ValueError, match='unknown source'):
        source_pool([], ['invented'])


def test_empty_nt_pool_is_supported():
    assert source_pool([geom('a', ['pair_union'])]) == []
    members, x = features_and_members([], {}, {}, 4)
    assert members == [] and x.shape == (0, 26)


def test_GT_and_heldout_guard_blocks_actual_resolved_paths(tmp_path):
    from tools.run_fi1_core_ablation_train90 import AccessGuard
    gt = tmp_path / 'ground_truth/scene0000_00.txt'
    guard = AccessGuard(['scene0000_00'], [gt])
    with pytest.raises(PermissionError, match='GT requires'):
        guard.check(gt)
    guard.gt_enabled = True
    guard.check(gt)
    with pytest.raises(PermissionError, match='outside frozen'):
        guard.check(tmp_path / 'scene0001_00/data.npz')
    heldout = tmp_path / 'scannet_validation_stream/record.json'
    link = tmp_path / 'innocent.json'; link.symlink_to(heldout)
    with pytest.raises(PermissionError, match='heldout'):
        guard.check(link)


def test_evaluator_UUID_serialization_preserves_AP_and_unique_identity(tmp_path):
    from tools.run_fi1_core_ablation_train90 import serializable_matches
    from tools.fi1_r3b_delta_diagnostic_core import evaluator, metrics
    ev = evaluator()
    gt = tmp_path / 'labels.txt'
    np.savetxt(gt, np.full(120, int(ev.VALID_CLASS_IDS[0])*1000 + 1), fmt='%d')
    gm, pm = ev.assign_instances_for_scan(dict(pred_masks=np.ones((120,2), dtype=bool),
        pred_classes=np.array([0,0]), pred_scores=np.array([.9,.8])), str(gt))
    original = dict(gt=gm, pred=pm)
    encoded = serializable_matches(original)
    assert len({p['uuid'] for p in encoded['pred'][ev.CLASS_LABELS[0]]}) == 2
    before, *_ = ev.evaluate_matches({'synthetic': original})
    after, *_ = ev.evaluate_matches({'synthetic': encoded})
    assert np.array_equal(before,after,equal_nan=True)

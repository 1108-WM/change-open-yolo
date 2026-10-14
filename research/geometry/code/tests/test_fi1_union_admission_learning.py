import copy
import numpy as np
import pytest

from tools.fi1_union_admission_learning import (
    admitted_pool, anchor_targets, scene_folds, sample_weights, admission_keys,
    boundary_features, region_view_features)
from tools.fi1_core_ablation import source_pool,features_and_members
from tests.test_fi1_core_ablation import geom, evidence


def test_admission_removes_alias_but_protects_shared_native_and_rebuilds_classes():
    gs=[geom('a',['native','pair_union']),geom('b',['track']),geom('c',['pair_union'])]
    old=copy.deepcopy(gs)
    assert admitted_pool(gs,set())==source_pool(gs)
    assert admitted_pool(gs,{gs[0]['geometry_key'],gs[2]['geometry_key']})==source_pool(gs,('native','track','pair_union'))
    assert gs==old
    pool=admitted_pool(gs,set());ps={g['geometry_key']:np.arange(3) for g in pool};ev={g['geometry_key']:evidence(g) for g in pool}
    members,x=features_and_members(pool,ps,ev,4)
    assert {m['class_index'] for m in members if m['geometry_key']==gs[0]['geometry_key']}=={0}
    assert np.all(x[:,16:18]==0)
    with pytest.raises(ValueError):admitted_pool(gs,{'unknown'})


def test_admission_changes_context_even_if_old_output_did_not_select_U():
    gs=[geom('a',['track']),geom('b',['pair_union'])]
    ps={g['geometry_key']:np.arange(3) for g in gs};ev={g['geometry_key']:evidence(g) for g in gs}
    mm,xx=features_and_members(gs,ps,ev,4)
    pool=admitted_pool(gs,set());k=pool[0]['geometry_key']
    nn,yy=features_and_members(pool,{k:ps[k]},{k:ev[k]},4)
    assert np.any(xx[:len(nn),20:]!=yy[:,20:])


def test_supervision_compares_best_parent_without_switching_GT_to_union_winner():
    t=anchor_targets([.6,.1],[.85,.2],[.75,.9],[1001,1002])
    assert t['anchor']==1001 and t['gain']<0 and t['absolute_iou']==.75
    assert t['delta_iou']==pytest.approx(-.1)


def test_small_high_precision_completion_preserves_threshold_crossing():
    t=anchor_targets([.89],[.6],[.91],[1001])
    assert t['delta_iou']==pytest.approx(.02)
    assert t['delta_threshold']==pytest.approx(1/9)
    assert t['gain']>.02


def test_unmatched_and_unreliable_parents_not_fabricated_negative_gain():
    t=anchor_targets([0],[.9],[.8],[1001]);assert t['gain'] is None and not t['reliable']
    t=anchor_targets([.1],[.2],[.3],[1001]);assert not t['reliable'] and t['absolute_iou']==.3
    t=anchor_targets([],[],[],[]);assert t['anchor'] is None


def test_folds_group_repeated_scans_and_ignore_input_order():
    scenes=[f'scene{i:04d}_00' for i in range(200)]+['scene0001_01']
    a=scene_folds(scenes);assert a==scene_folds(scenes[::-1]);assert a['scene0001_00']==a['scene0001_01']
    assert [sum(a[s]==i for s in scenes[:-1]) for i in range(5)]==[40]*5


def test_weights_balance_scenes_and_reliable_directions():
    rr=[dict(scene='scene0001_00',union_key='a'),dict(scene='scene0001_00',union_key='a'),
        dict(scene='scene0001_00',union_key='b'),dict(scene='scene0002_00',union_key='c')]
    w=sample_weights(rr);assert w.sum()==4 and w[:3].sum()==w[3]
    assert w[:2].sum()==w[2]


def test_admission_requires_both_gain_and_quality_without_GT_validity():
    rr=[dict(union_key='u'),dict(union_key='u'),dict(union_key='v'),dict(union_key='w')]
    assert admission_keys(rr,[-.1,.001,.2,0],[.9,.51,.49,.9])=={'u'}
    with pytest.raises(ValueError):admission_keys(rr,[np.nan]*4,[.9]*4)


def test_boundary_has_explicit_empty_increment_and_rejects_label_columns():
    raw=np.zeros((5,9));raw[:,:3]=np.arange(5)[:,None];raw[:,6]=1
    out=boundary_features(raw,np.arange(3),np.array([3,4]));assert out['distance_min']>0
    assert out['normal_mean']==0 and out['normal_valid_fraction']==1
    assert boundary_features(raw,np.arange(5),np.array([],dtype=int))['empty_increment']==1
    with pytest.raises(ValueError):boundary_features(np.zeros((5,12)),np.arange(3),np.array([3,4]))


def test_independent_views_exclude_sources_and_repeated_pose_votes():
    parent=np.arange(5);extra=np.arange(5,10)
    frames=[dict(group=g,visible=np.ones(10,dtype=bool),masks=[np.arange(10)]) for g in [0,1,1]]
    x=region_view_features(parent,extra,frames,{0})
    assert x['log1p_independent_groups']==pytest.approx(np.log(2)) and x['joint_mean']==1
    assert x['separate_max']==0
    assert region_view_features(parent,extra,frames,{0,1})['unknown']==1


def test_joint_and_separation_can_coexist_and_invisibility_is_unknown():
    a=np.arange(5);b=np.arange(5,10)
    f=dict(group=1,visible=np.ones(10,dtype=bool),masks=[np.arange(10),a,b])
    x=region_view_features(a,b,[f],set());assert x['joint_max']==x['separate_max']==1
    f['visible'][b]=False
    assert region_view_features(a,b,[f],set())['unknown']==1


def test_identical_mask_aliases_cannot_form_separating_pair():
    a=np.arange(6);b=np.arange(6,12);mask=np.array([0,1,2,6,7,8])
    f=dict(group=1,visible=np.ones(12,dtype=bool),masks=[mask,mask.copy()])
    x=region_view_features(a,b,[f],set())
    assert x['joint_max']==.5 and x['separate_max']==0


def test_training_guard_blocks_other_fold_labels_reserve_and_AP(tmp_path):
    from tools.run_fi1_union_admission_development200 import AdmissionGuard
    c=dict(scenes=['scene0001_00','scene0002_00'],bindings={s:dict(gt_path=f'/tmp/ground_truth/{s}.txt') for s in ['scene0001_00','scene0002_00']})
    g=AdmissionGuard(c,'fit',['scene0001_00']);g.check(tmp_path/'labels/scene0001_00.json')
    with pytest.raises(PermissionError,match='fold'):g.check(tmp_path/'labels/scene0002_00.json')
    with pytest.raises(PermissionError):g.check(tmp_path/'scenes/scene0003_00/geometry.json')
    with pytest.raises(PermissionError):g.check('/tmp/ground_truth/scene0001_00.txt')
    with pytest.raises(PermissionError):g.check(tmp_path/'evaluation/summary.json')

import numpy as np
import pytest
from tools.fi1_class_agnostic_ablation import geometry_selection, collapse_members, remap_gt, class_agnostic_evaluator, project_matches
from tools.fi1_core_ablation import source_pool
from tests.test_fi1_core_ablation import geom
from tools.fi1_r3b_delta_diagnostic_core import evaluator, metrics


def test_geometry_scores_filter_refined_alias_before_max_and_keep_shared():
    g=geom('a',['track'],True)
    g['dv3_plans'][0]['challenger_score']=.9
    g['dv3_plans'].append(dict(append_only=False,candidate_source='track',challenger_score=.4))
    assert geometry_selection([g])[0]['score']==.9
    pool=source_pool([g],('native','track','pair_union'))
    assert geometry_selection(pool)[0]['score']==.4
    assert pool[0]['geometry_key']==g['geometry_key']


def test_class_duplicate_collapse_keeps_max_score_and_never_refills():
    rows=[dict(geometry_key='a',class_index=2,score=.8),
          dict(geometry_key='a',class_index=1,score=.8),dict(geometry_key='b',class_index=0,score=.3)]
    chosen=collapse_members(rows)
    assert len(chosen)==2 and chosen[0]==dict(geometry_key='a',score=.8,winning_class_index=1)
    assert rows[0]['class_index']==2
    with pytest.raises(ValueError,match='invalid B26'):
        collapse_members([dict(geometry_key='a',class_index=0,score=np.nan)])


def test_remapping_preserves_boundaries_small_objects_and_void():
    gt=np.array([1001,1001,3001,3001,1002,4001,0,999])
    mapped=remap_gt(gt,[1,3],5)
    assert mapped.tolist()==[5001,5001,5003,5003,5002,0,0,0]


def test_single_class_evaluator_exactly_matches_historical_full_label_mapping(tmp_path):
    ev,valid,uid=class_agnostic_evaluator()
    gt=np.concatenate([np.full(120,int(valid[0])*1000+1),np.full(120,int(valid[1])*1000+1),
                       np.full(40,int(valid[1])*1000+2),np.zeros(20,dtype=int)])
    mapped=remap_gt(gt,valid,uid); p=tmp_path/'gt.txt'; np.savetxt(p,mapped,fmt='%d')
    masks=np.zeros((len(gt),3),dtype=bool); masks[:120,0]=True; masks[120:240,1]=True; masks[:120,2]=True
    pred=dict(pred_masks=masks,pred_classes=np.zeros(3,dtype=int),pred_scores=np.array([.9,.8,.7]))
    gm,pm=ev.assign_instances_for_scan(pred,str(p)); aps,*_=ev.evaluate_matches({'scan':dict(gt=gm,pred=pm)})
    historical=evaluator(); gm2,pm2=historical.assign_instances_for_scan(pred,str(p))
    aps2,*_=historical.evaluate_matches({'scan':dict(gt=gm2,pred=pm2)})
    assert np.array_equal(aps[0,0],aps2[0,0],equal_nan=True)
    assert metrics(ev,aps)==metrics(historical,aps2)
    assert sum(g['vert_count']>=100 for g in gm[ev.CLASS_LABELS[0]])==2


def test_dropping_wrong_class_allows_geometric_match_and_duplicates_penalized(tmp_path):
    ev,valid,uid=class_agnostic_evaluator()
    gt=np.full(120,int(valid[1])*1000+1); mapped=remap_gt(gt,valid,uid)
    path=tmp_path/'gt.txt'; np.savetxt(path,mapped,fmt='%d')
    gm,pm=ev.assign_instances_for_scan(dict(pred_masks=np.ones((120,1),dtype=bool),
        pred_classes=np.array([0]),pred_scores=np.array([.8])),str(path))
    aps,*_=ev.evaluate_matches({'scan':dict(gt=gm,pred=pm)})
    assert metrics(ev,aps)['ap']==1.0
    duplicate_gt=tmp_path/'gt2.txt'; np.savetxt(duplicate_gt,np.r_[mapped,mapped+1],fmt='%d')
    masks=np.zeros((240,3),dtype=bool); masks[:120,:2]=True; masks[120:,2]=True
    gm,pm=ev.assign_instances_for_scan(dict(pred_masks=masks,pred_classes=np.zeros(3,dtype=int),
        pred_scores=np.array([.9,.85,.8])),str(duplicate_gt))
    aps,*_=ev.evaluate_matches({'scan':dict(gt=gm,pred=pm)})
    assert 0 < metrics(ev,aps)['ap'] < 1.0


def test_subset_and_rescore_projection_matches_fresh_original_assignment(tmp_path):
    ev,valid,uid=class_agnostic_evaluator(); label=ev.CLASS_LABELS[0]
    gt=np.r_[np.full(120,uid*1000+1),np.full(120,uid*1000+2),np.zeros(20,dtype=int)]
    path=tmp_path/'gt.txt';np.savetxt(path,gt,fmt='%d')
    masks=np.zeros((260,4),dtype=bool);masks[:120,0]=True;masks[:120,1]=True
    masks[120:240,2]=True;masks[240:,3]=True
    gm,pm=ev.assign_instances_for_scan(dict(pred_masks=masks,pred_classes=np.zeros(4,dtype=int),
        pred_scores=np.array([.9,.8,.7,.6])),str(path))
    template=dict(gt=gm,pred=pm)
    selected=[dict(geometry_key='b',score=.99),dict(geometry_key='c',score=.8),dict(geometry_key='tiny',score=.7)]
    projected=project_matches(template,['a','b','c'],selected,label)
    gm2,pm2=ev.assign_instances_for_scan(dict(pred_masks=masks[:,[1,2,3]],pred_classes=np.zeros(3,dtype=int),
        pred_scores=np.array([.99,.8,.7])),str(path))
    a,*_=ev.evaluate_matches({'scan':projected});b,*_=ev.evaluate_matches({'scan':dict(gt=gm2,pred=pm2)})
    assert np.array_equal(a,b,equal_nan=True)
    assert [p['confidence'] for p in projected['pred'][label]]==[.99,.8]
    assert [p['confidence'] for p in template['pred'][label]]==[.9,.8,.7]

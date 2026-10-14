import numpy as np
import pytest
from tools.fi1_union_development_replay import Guard,project_multiclass
from tools.fi1_r3b_delta_diagnostic_core import evaluator


@pytest.mark.parametrize('indices,scores', [([4,2,0,3],[.95,.8,.7,.6]),([1,0,5],[.9,.9,.5])])
def test_multiclass_subset_rescore_equals_fresh_original_matching_and_AP(tmp_path,indices,scores):
    ev=evaluator();ids=ev.VALID_CLASS_IDS
    gt=np.r_[np.full(120,int(ids[0])*1000+1),np.full(120,int(ids[1])*1000+1),
             np.full(40,int(ids[0])*1000+2),np.zeros(20,dtype=int)]
    path=tmp_path/'synthetic.txt';np.savetxt(path,gt,fmt='%d')
    masks=np.zeros((300,6),dtype=bool);masks[:120,0]=True;masks[:120,1]=True
    masks[240:,2]=True;masks[120:240,3]=True;masks[:120,4]=True;masks[120:,5]=True
    cls=np.array([0,0,0,1,1,1]);original_scores=np.array([.9,.85,.8,.7,.6,.5])
    universe=[dict(geometry_key=str(i),class_index=int(cls[i]),score=float(original_scores[i])) for i in range(6)]
    sizes={str(i):int(masks[:,i].sum()) for i in range(6)}
    gm,pm=ev.assign_instances_for_scan(dict(pred_masks=masks,pred_classes=cls,pred_scores=original_scores),str(path))
    selected=[{**universe[i],'score':score} for i,score in zip(indices,scores)]
    projected=project_multiclass(dict(gt=gm,pred=pm),universe,selected,sizes,ev.CLASS_LABELS,ids)
    gm2,pm2=ev.assign_instances_for_scan(dict(pred_masks=masks[:,indices],pred_classes=cls[indices],pred_scores=np.asarray(scores)),str(path))
    a,*_=ev.evaluate_matches({'scene':projected});b,*_=ev.evaluate_matches({'scene':dict(gt=gm2,pred=pm2)})
    assert np.array_equal(a,b,equal_nan=True)
    for label in ev.CLASS_LABELS:
        def fields(p):return {k:v for k,v in p.items() if k not in ('uuid','matched_gt')}
        assert [fields(p) for p in projected['pred'][label]]==[fields(p) for p in pm2[label]]
        for pg,fg in zip(projected['gt'][label],gm2[label]):
            assert [fields(p) for p in pg['matched_pred']]==[fields(p) for p in fg['matched_pred']]


def test_guard_preserves_reserve_and_recognizes_legacy_GT_mirror():
    path='/tmp/ground_truth_train/scene0001_01.txt'
    c=dict(scenes=['scene0001_01'],bindings={'scene0001_01':dict(gt_path=path)},label_inputs={})
    g=Guard(c,raw_GT_allowed=True)
    with pytest.raises(PermissionError,match='sealed'):g.check(path)
    g.gt_enabled=True;g.check(path)
    with pytest.raises(PermissionError):g.check('/tmp/ground_truth_train/scene0002_00.txt')
    with pytest.raises(PermissionError):g.check('/tmp/scenes/scene0002_00/geometry.json')
    checker=Guard(c);checker.gt_enabled=True
    with pytest.raises(PermissionError):checker.check(path)


def test_guard_blocks_cached_targets_and_labels_before_seal():
    p='/tmp/.local/old/evaluation/scene0001_01/A_matches.json'
    c=dict(scenes=['scene0001_01'],bindings={'scene0001_01':dict(gt_path='/tmp/ground_truth/scene0001_01.txt')},label_inputs={p:'hash'})
    g=Guard(c)
    with pytest.raises(PermissionError,match='cached labels'):g.check(p)
    g.gt_enabled=True;g.check(p)
    with pytest.raises(PermissionError,match='targets'):g.check('/tmp/learning/dataset/scene0001_01/features.npz')

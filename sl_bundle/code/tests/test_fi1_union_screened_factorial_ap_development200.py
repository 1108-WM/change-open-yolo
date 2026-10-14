import copy
import numpy as np
import pytest
from tools.run_fi1_union_screened_factorial_ap_development200 import assign,check_match,canonical_match,target_review
from tools.fi1_class_agnostic_ablation import collapse_members,class_agnostic_evaluator,remap_gt,project_matches
from tools.fi1_r3b_delta_diagnostic_core import evaluator


def test_census_checker_rejects_reverse_confidence_corruption(tmp_path):
    ev=evaluator();gt=np.r_[np.full(120,int(ev.VALID_CLASS_IDS[0])*1000+1),np.zeros(20,dtype=int)]
    q=tmp_path/'gt.txt';np.savetxt(q,gt,fmt='%d');pts={'g':np.arange(130)}
    rows=[dict(geometry_key='g',class_index=0,score=.8)]
    m=assign(ev,rows,pts,q,len(gt))
    assert check_match(m,rows,pts,gt,ev)==1
    bad=copy.deepcopy(m);bad['gt'][ev.CLASS_LABELS[0]][0]['matched_pred'][0]['confidence']=.7
    with pytest.raises(AssertionError):check_match(bad,rows,pts,gt,ev)


def test_wrong_class_and_tiny_masks_preserve_distinct_evaluation_views(tmp_path):
    ev=evaluator();ca,valid,uid=class_agnostic_evaluator()
    gt=np.r_[np.full(120,int(valid[1])*1000+1),np.full(30,int(valid[0])*1000+2),np.zeros(20,dtype=int)]
    q=tmp_path/'gt.txt';np.savetxt(q,gt,fmt='%d')
    pts={'g':np.arange(120),'tiny':np.arange(120,150)}
    rows=[dict(geometry_key='g',class_index=0,score=.8),dict(geometry_key='g',class_index=1,score=.6),dict(geometry_key='tiny',class_index=0,score=.7)]
    sm=assign(ev,rows,pts,q,len(gt));check_match(sm,rows,pts,gt,ev)
    collapsed=collapse_members(rows);assert len(collapsed)==2 and collapsed[0]['score']==.8
    mapped=remap_gt(gt,valid,uid);ca.util_3d.load_ids=lambda path:mapped
    full=assign(ca,collapsed,pts,q,len(gt),True)
    assert check_match(full,collapsed,pts,mapped,ca,True)==1
    projected=project_matches(full,['g'],collapsed,ca.CLASS_LABELS[0])
    assert canonical_match(projected)==canonical_match(full)
    assert len(full['gt'][ca.CLASS_LABELS[0]])==2


def test_target_review_distinguishes_coverage_from_confidence(tmp_path):
    ev=evaluator();gt=np.full(120,int(ev.VALID_CLASS_IDS[0])*1000+1);q=tmp_path/'gt.txt';np.savetxt(q,gt,fmt='%d')
    pts={'g':np.arange(120)};rows=[dict(geometry_key='g',class_index=0,score=.8)]
    a=assign(ev,rows,pts,q,len(gt));c=assign(ev,[dict(rows[0],score=.5)],pts,q,len(gt));b=assign(ev,[],pts,q,len(gt))
    review=target_review(dict(A=a,B=b,C=c,D=c),ev)
    assert review['GT']==1
    assert review['counts']['B_minus_A']['0.50']['lost']==1
    assert review['counts']['C_minus_A']['0.50']['shared_score_down']==1
    assert review['counts']['C_minus_A']['0.50']['lost']==0

import numpy as np
import pytest
from tools.fi1_sr_cross_control import fixed_rescore, independent_rescore_match
from tools.fi1_class_agnostic_ablation import collapse_members


def row(g,c,s):
    return dict(geometry_key=g,class_index=c,score=s,has_original_native_alias=False)


def test_fixed_set_no_refill_and_zero_scores_preserved():
    pool=[row('g',0,.9),row('h',0,.8),row('new',0,.7)]
    selected=fixed_rescore(pool[:2],pool,np.array([0,.2,1.]))
    assert [m['geometry_key'] for m in selected]==['h','g']
    assert len(selected)==2 and selected[-1]['score']==0


def test_geometry_winner_recomputed_from_selected_only():
    pool=[row('g',0,.9),row('g',1,.8),row('g',2,.7)]
    selected=fixed_rescore(pool[:2],pool,np.array([.1,.2,1.]))
    assert collapse_members(selected)==[dict(geometry_key='g',score=.2,winning_class_index=1)]


def test_match_replay_keeps_void_ignore_and_reverse_edges():
    rows=[row('g',0,.9),row('h',0,.8),row('small',0,.7)]
    g=dict(instance_id=1001,vert_count=150)
    p=dict(uuid='g',pred_id=0,confidence=.9,label_id=1,vert_count=100,void_intersection=13,
           matched_gt=[{**g,'intersection':75}])
    h=dict(uuid='h',pred_id=1,confidence=.8,label_id=1,vert_count=100,void_intersection=0,matched_gt=[])
    template=dict(gt={'x':[{**g,'matched_pred':[{**{k:v for k,v in p.items() if k!='matched_gt'},'intersection':75}]}]},pred={'x':[p,h]})
    new=fixed_rescore(rows,rows,np.array([0,.2,1.]))
    result=independent_rescore_match(template,rows,new,dict(g=100,h=100,small=99))
    assert [r['uuid'] for r in result['pred']['x']]==['h','g']
    assert result['pred']['x'][1]['void_intersection']==13
    assert result['gt']['x'][0]['matched_pred'][0]['confidence']==0
    assert result['gt']['x'][0]['matched_pred'][0]['pred_id']==1
    assert template['pred']['x'][0]['confidence']==.9


def test_identity_mismatch_rejected():
    with pytest.raises(ValueError):fixed_rescore([row('unknown',0,.1)],[row('g',0,.1)],[.2])

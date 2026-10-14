import numpy as np
import torch
from tools import fi1_region_affiliation_core as c


def test_physical_split_does_not_depend_on_input_order_or_scan_identity():
    scenes=[f'scene{i:04d}_00' for i in range(200)]+['scene0001_01'];folds={s:int(s[5:9])%5 for s in scenes}
    a=c.split_scenes(scenes,folds);assert a==c.split_scenes(scenes[::-1],folds)
    owner={s.split('_')[0]:k for k,ss in a.items() for s in ss}
    assert all(owner[s.split('_')[0]]==k for k,ss in a.items() for s in ss)


def test_missing_mask_is_explicit_unknown_not_negative_support():
    parent=np.ones(8,bool);member=np.zeros((8,4),bool);seen=np.ones(8,bool)
    v=c.local_view(np.array([0,1]),parent,member,seen,np.empty((8,0),bool),np.empty(0))
    assert v[0]==1 and v[1]==1 and v[4]==0 and v[5]==0 and v[19]==1


def test_ambiguous_matching_and_shared_mask_are_explicit():
    parent=np.ones(8,bool);member=np.zeros((8,4),bool);member[:,0]=True;seen=np.ones(8,bool);masks=np.ones((8,2),bool)
    v=c.local_view(np.array([0]),parent,member,seen,masks,np.array([.9,.9]))
    assert v[5]==v[6]==v[7]==v[8]==v[15]==v[19]==1


def test_retained_anchor_excludes_inspected_region_and_cache_equivalence():
    parent=np.ones(5,bool);member=np.zeros((5,4),bool);seen=np.ones(5,bool);masks=np.array([[1],[1],[1],[0],[0]],bool);block=np.arange(3)
    v=c.local_view(block,parent,member,seen,masks,np.array([.95]));cached=c.local_view(block,parent,member,seen,masks,np.array([.95]),c.view_context(parent,member,seen,masks))
    assert v[5]==0 and np.array_equal(v,cached)


def test_union_deduplicates_points_and_uses_conservative_overlapping_risk():
    parent=np.arange(110);halo=parent.copy();blocks=[np.array([0,1]),np.array([1,2])]
    pr=np.array([[.01,.99,0,.99],[.04,.96,0,.99]],np.float32);rule=dict(safe=.9,target=.1,unknown=.1,union_risk=.1)
    q,a=c.execute(parent,halo,blocks,pr,rule)
    assert a['deleted']==3 and np.array_equal(q,np.arange(3,110)) and abs(a['risk']-.03)<1e-6
    q,a=c.execute(parent,halo,blocks,pr,dict(rule,union_risk=.025));assert np.array_equal(q,parent) and a['reason']=='union_risk_abstain'


def test_minimum_size_and_calibration_abstention_preserve_parent():
    p=np.arange(101);blocks=[np.array([0,1])];pr=np.array([[0,1,0,1]],np.float32);rule=dict(safe=.9,target=.1,unknown=.1,union_risk=.1)
    q,a=c.execute(p,p,blocks,pr,rule);assert np.array_equal(q,p) and a['reason']=='minimum_mask_abstain'
    q,a=c.execute(p,p,blocks,pr,{'abstain':True});assert np.array_equal(q,p)


def test_point_graph_reads_spatial_tokens_and_C_has_same_parameter_count():
    rng=np.random.default_rng(4);xyz=rng.normal(size=(32,3)).astype(np.float32)*.03;x=np.zeros((32,64),np.float32);parent=np.ones(32,bool);member=np.zeros((32,4),bool);nn=np.tile(np.arange(8),(32,1))
    t,mask,roles,idx,w,stats,tid=c.tokens(x,xyz,parent,nn,member,np.arange(4))
    assert len(np.unique(tid[:4]))==4 and (tid[mask]>=0).all() and np.isfinite(t).all()
    model=c.RegionAffiliation().eval();kwargs=[torch.tensor(a[None]) for a in (t.astype(np.float32),mask,roles.astype(np.int64),idx.astype(np.int64),w.astype(np.float32),stats)]
    with torch.inference_mode():out=model(*kwargs,torch.zeros((1,6,20)))
    assert out.shape==(1,4) and torch.isfinite(out).all()
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in c.RegionAffiliation().parameters())


def test_mixed_and_unknown_fraction_supervision_keeps_distinct_targets():
    logits=torch.zeros((2,4),requires_grad=True);truth=torch.tensor([[.5,.5,0.],[0.,0.,1.]])
    value=c.loss(logits,truth,torch.zeros(2),torch.ones(2));value.backward()
    assert logits.grad[0,2]>0 and logits.grad[1,2]<0 and torch.isfinite(value)

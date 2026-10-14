import numpy as np
import torch
from tools import fi1_decoupled_region_core as c


def test_out_of_scope_known_identity_is_not_relation_unknown():
    assert np.array_equal(c.pair_truth([1000,1000],[1000]),[1,0,0])
    assert np.array_equal(c.pair_truth([1000],[2001]),[0,1,0])


def test_mixed_reference_is_fractional_without_forced_majority():
    assert np.allclose(c.pair_truth([2001,3001],[2001,3001]),[.5,.5,0])
    assert np.allclose(c.pair_truth([2001,0],[2001,3001,0]),[1/6,1/6,2/3])
    assert np.array_equal(c.pair_truth([2001],[]),[0,0,1])


def fixture():
    rng=np.random.default_rng(3);x=np.zeros((96,75),np.float32);mask=np.zeros(96,bool);roles=np.full(96,-1,np.int8)
    for k,lo in enumerate((0,24,48,72)):
        mask[lo:lo+4]=True;roles[lo:lo+4]=k;x[lo:lo+4,:3]=rng.normal(size=(4,3))*.02+k*.05
        x[lo:lo+4,3:9]=rng.normal(size=(4,6));x[lo:lo+4,16:64]=rng.normal(size=(4,48))
    x[72:76,71]=1;nn=np.broadcast_to(np.arange(96)[:,None],(96,4)).copy();w=np.ones((96,4),np.float32)/4
    return x,mask,roles,nn,w


def test_parent_member_flags_cannot_shortcut_instance_identity_input():
    x,mask,roles,nn,w=fixture();a=c.relation_context(x,mask,roles,nn,w);x[:,9:16]=99;b=c.relation_context(x,mask,roles,nn,w)
    assert all(np.array_equal(i,j) for i,j in zip(a,b))


def test_spatial_relation_context_is_translation_invariant():
    x,mask,roles,nn,w=fixture();a=c.relation_context(x,mask,roles,nn,w);x[:,:3]+=np.array([1.,-2.,3.]);b=c.relation_context(x,mask,roles,nn,w)
    assert np.allclose(a[0],b[0],atol=.001) and np.allclose(a[-1],b[-1],atol=1e-5)


def test_missing_reference_predicts_unknown_and_models_have_disjoint_weights():
    x,mask,roles,nn,w=fixture();nodes,group,weights,cross,valid,distance,stats=c.relation_context(x,mask,roles,nn,w)
    relation=c.InstanceRelation().eval();parent=c.ParentEditability().eval()
    args=[torch.tensor(a[None],dtype=torch.bool if a.dtype==bool else torch.long if np.issubdtype(a.dtype,np.integer) else torch.float32) for a in (nodes,mask,group,nn,weights,cross,valid,distance,stats)]
    with torch.inference_mode():p=relation(*args).softmax(-1)
    assert p.shape==(1,6,3) and p[0,3,2]>.999
    assert not ({id(p) for p in relation.parameters()}&{id(p) for p in parent.parameters()})


def test_symmetric_boundary_embedding_has_no_directed_identity_bias():
    model=c.InstanceRelation().eval();a=torch.randn(2,48);b=torch.randn(2,48);distance=torch.ones(2,1)
    with torch.inference_mode():
        first=model.edge(torch.cat((a+b,(a-b).abs(),a*b,distance),1));second=model.edge(torch.cat((b+a,(b-a).abs(),b*a,distance),1))
    assert torch.equal(first,second)


def prediction():
    r=np.zeros((2,6,3),np.float32);r[:,:,2]=1;r[:,1]=[.04,.95,.01];r[:,2]=[.95,.04,.01]
    return dict(relations=r,parent=np.array([.99,.002,.002,.003,.003]),valid=np.ones((2,6),bool))


def test_deletion_requires_both_relation_and_independent_parent_permission():
    p=np.arange(110);blocks=[np.array([0,1]),np.array([1,2])];pred=prediction();rule=dict(edit=.9,relation=.9)
    q,act=c.execute(p,p,blocks,pred,rule);assert act['deleted']==3 and np.array_equal(q,np.arange(3,110))
    pred['parent'][0]=.1;q,act=c.execute(p,p,blocks,pred,rule);assert np.array_equal(q,p) and act['reason']=='parent_not_editable'


def test_parent_difference_alone_does_not_certify_neighbor_identity():
    p=np.arange(110);pred=prediction();pred['valid'][:,2:]=False
    q,act=c.execute(p,p,[np.array([0,1]),np.array([1,2])],pred,dict(edit=.9,relation=.9));assert np.array_equal(q,p)


def test_minimum_size_and_locked_calibration_abstention_preserve_parent():
    p=np.arange(102);blocks=[np.array([0,1]),np.array([1,2])]
    q,act=c.execute(p,p,blocks,prediction(),dict(edit=.9,relation=.9));assert np.array_equal(q,p) and act['reason']=='minimum_mask_abstain'
    q,act=c.execute(p,p,blocks,prediction(),{'abstain':True});assert np.array_equal(q,p)

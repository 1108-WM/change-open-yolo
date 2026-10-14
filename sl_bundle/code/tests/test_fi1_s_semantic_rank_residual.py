import numpy as np
import torch
from tools.fi1_s_semantic_rank_residual import adjusted,member_states,make_pairs,Residual


def test_residual_identity_endpoints_and_two_directions():
    s=np.array([0,.1,.3,1]);assert np.array_equal(adjusted(s,np.zeros(4)),s)
    a=adjusted(s,np.array([1,1,-1,-1]));assert a[0]==0 and a[-1]==1 and a[1]>s[1] and a[2]<s[2]
    m=Residual();assert torch.equal(m(torch.randn(5,28)),torch.zeros(5))


def test_threshold_ignore_duplicate_and_small_mask():
    gi=np.arange(5);ci=np.zeros(5,dtype=int);s=[.2,.1,.8,.6,.9]
    own=np.array([[.9],[.8],[0],[0],[1.]]);targets=np.array([[1001],[1001],[-1],[-1],[1002]])
    frac=np.array([[0],[0],[.6],[0],[0]]);ignored=np.array([[0],[0],[0],[.85],[0]])
    state,_=member_states(s,gi,ci,np.array([100,100,100,100,99]),own,targets,frac,ignored,[.5,.75,.9])
    assert np.array_equal(state,np.array([[1,1,0],[2,2,0],[-1,0,0],[-1,-1,0],[-1,-1,-1]]))


def test_pairs_respect_classes_and_scene_target_identity():
    s=np.array([.2,.1,.9,.3,.8,.05]);ci=np.array([0,0,0,1,1,0]);scene=np.array([0,0,1,0,1,1])
    ids=np.array([1001,1001,-1,2001,-1,1001]);states=np.array([[1],[2],[0],[1],[0],[2]],dtype=np.int8)
    p=make_pairs(s,ci,scene,ids,states,42);assert np.all(ci[p['pairs'][:,0]]==ci[p['pairs'][:,1]])
    assert not any(a==0 and b==5 and k==3 for (a,b),k in zip(p['pairs'],p['kinds']))
    assert np.isclose(p['weights'][ci[p['pairs'][:,0]]==0].sum(),.5)

import numpy as np
from tools.fi1_candidate_observation_selection import region_weights,choose_views


def test_selector_preserves_subject_then_complementary_boundary():
    halo=np.arange(100);roles,w=region_weights(np.arange(50),halo,np.arange(40,50),np.arange(45,60))
    visible=np.zeros((3,100),bool)
    visible[0,:50]=True;visible[2,:50]=True
    visible[1,:10]=True;visible[1,40:]=True
    ix,trace=choose_views(np.array([10,20,30]),np.arange(3),visible,visible,np.ones(3,bool),roles,w,budget=2)
    assert ix==[0,1] and trace[0]['subject_fraction']==1


def test_no_duplicate_physical_groups_or_empty_mask_views():
    halo=np.arange(100);roles,w=region_weights(halo,halo,halo,np.array([],int))
    v=np.ones((4,100),bool)
    ix,_=choose_views(np.array([0,10,20,30]),np.array([0,0,1,2]),v,v,np.array([1,1,1,0],bool),roles,w)
    assert ix==[0,2]


def test_missing_evidence_does_not_supply_coverage_and_budget_is_fixed():
    halo=np.arange(100);roles,w=region_weights(halo,halo,halo,np.array([],int))
    v=np.zeros((8,100),bool);v[:,:30]=True
    ix,_=choose_views(np.arange(8)*10,np.arange(8),v,v,np.ones(8,bool),roles,w)
    assert len(ix)==2
    assert v[ix].sum(axis=0)[30:].sum()==0


def test_outside_and_disagreement_are_weighted_without_being_labels():
    roles,w=region_weights(np.arange(50),np.arange(100),np.arange(40,50),np.arange(45,60))
    assert roles[2][50:].all() and w[50:].sum()>0
    assert len(roles)==4 and np.isfinite(w).all()

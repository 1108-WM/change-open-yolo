import numpy as np
import torch
import pytest
from tools import fi1_local_edit_ablation_core as core
from tools import fi1_candidate_local_segmenter as base


def test_sparse_corrections_receive_their_fixed_loss_budget():
    groups=[np.arange(2),np.arange(2,5),np.arange(5,15),np.arange(15,10015),np.arange(10015,10025)]
    indices=core.balanced_sample(groups,np.random.default_rng(1))
    actual=[int(np.isin(indices,g).sum()) for g in groups]
    assert actual==[1024,1024,819,819,410] and len(indices)==4096


def test_outside_is_not_automatically_a_negative_and_unseen_edits_are_not_stratified():
    x=np.zeros((6,64));x[:,10]=.1;y=np.array([1,1,0,1,0,2]);p=np.array([1,1,0,0,1,0],bool);visible=np.array([2,1,2,2,2,2])
    groups=core.strata(y,p,visible,x)
    assert groups[0].tolist()==[0] and groups[1].tolist()==[2] and groups[2].tolist()==[3] and groups[4].tolist()==[5]


def test_candidate_scope_can_abstain_despite_confident_point_additions():
    parent=np.arange(100);halo=np.arange(120);p=np.tile([.97,.02,.01],(120,1));groups=np.full(120,2)
    actual,rec=core.repair(parent,halo,p,groups,'C',np.array([.1,.85,.01,.01,.02,.01]))
    assert np.array_equal(actual,parent) and not rec['accepted']
    actual,_=core.repair(parent,halo,p,groups,'C',np.array([.85,.1,.01,.01,.02,.01]));assert len(actual)==120


def test_B_reuses_exact_original_network_prior_and_state_layout():
    torch.manual_seed(10);a=base.CandidateLocalSegmenter();torch.manual_seed(10);b=core.model('B')
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


def test_identity_network_has_valid_gradients_with_competitor_and_no_competitor():
    torch.set_num_threads(1);torch.manual_seed(11)
    for has_neighbor in (False,True):
        x=torch.randn(120,64);x[:,9]=0;x[:100,9]=1;nn,w=base.graph(x[:,:3].numpy());m=core.model('C')
        mask=torch.zeros(120,4,dtype=torch.bool);mask[100:,0]=has_neighbor;valid=torch.tensor([has_neighbor,False,False,False])
        condition=dict(representatives=torch.tensor(core.representatives(x.numpy(),x[:,9].numpy()>.5)).long(),membership=mask,distance=torch.zeros(120,4),valid=valid,overlap=torch.zeros(4))
        logit,candidate,relation=m(x,torch.tensor(nn).long(),torch.tensor(w).float(),[(0,120)],[condition])
        loss=logit.square().mean()+candidate.square().mean()+relation[0].square().mean();loss.backward()
        assert torch.isfinite(loss) and m.identity[-1].weight.grad.abs().sum()>0 and m.edit[-1].weight.grad.abs().sum()>0

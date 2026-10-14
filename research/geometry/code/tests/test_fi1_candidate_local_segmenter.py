import numpy as np
import torch
import pytest
from tools.fi1_candidate_local_segmenter import CandidateLocalSegmenter,graph,repair,segmentation_loss


def test_edits_are_pointwise_and_unobserved_points_are_preserved():
    parent=np.arange(100);halo=np.arange(105);p=np.zeros((105,3),np.float32);p[:,0]=.95;p[:,1]=.05
    p[2:4]=[.05,.95,0];p[103]=[.01,.01,.98];seen=np.full(105,2);seen[2]=1;seen[104]=0
    actual,rec=repair(parent,halo,p,seen)
    assert 2 in actual and 3 not in actual and 100 in actual and 102 in actual
    assert 103 not in actual and 104 not in actual and rec['deleted']==1 and rec['added']==3


def test_small_mask_abstains_without_a_gt_chosen_refill():
    p=np.tile([.01,.98,.01],(100,1));parent=np.arange(100)
    actual,rec=repair(parent,parent,p,np.full(100,2))
    assert np.array_equal(actual,parent) and rec['reason']=='minimum_mask_abstain'


def test_reference_initialization_preserves_parent_and_gradient_is_real():
    torch.set_num_threads(1);torch.manual_seed(12)
    x=torch.randn(120,64);x[:,9]=0;x[:100,9]=1
    nn,w=graph(x[:,:3].numpy());model=CandidateLocalSegmenter().eval()
    logits=model(x,torch.tensor(nn).long(),torch.tensor(w).float(),[(0,120)])
    actual,_=repair(np.arange(100),np.arange(120),logits.softmax(1).detach().numpy(),np.full(120,2))
    assert np.array_equal(actual,np.arange(100))
    y=torch.ones(120,dtype=torch.long);y[:90]=0;y[119]=2
    before=model.decoder[-1].weight.detach().clone();loss=segmentation_loss(logits,y,x[:,9]>.5,False)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01);loss.backward();optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before,model.decoder[-1].weight)


def test_unknown_supervision_is_a_third_class():
    a=segmentation_loss(torch.tensor([[0.,0.,8.]],requires_grad=True),torch.tensor([2]),torch.tensor([True]),False)
    b=segmentation_loss(torch.tensor([[0.,8.,0.]],requires_grad=True),torch.tensor([2]),torch.tensor([True]),False)
    assert a<b


def test_graph_context_is_permutation_equivariant():
    torch.set_num_threads(1);torch.manual_seed(13);model=CandidateLocalSegmenter().eval()
    torch.nn.init.normal_(model.decoder[-1].weight,std=.02)
    x=torch.randn(40,64);x[:,9]=0;x[:25,9]=1
    nn,w=graph(x[:,:3].numpy());perm=np.random.default_rng(14).permutation(40);inverse=np.argsort(perm)
    a=model(x,torch.tensor(nn).long(),torch.tensor(w).float(),[(0,40)])
    b=model(x[perm],torch.tensor(inverse[nn[perm]]).long(),torch.tensor(w[perm]).float(),[(0,40)])
    assert torch.allclose(a,b[inverse],atol=1e-5)


def test_invalid_membership_and_nan_cannot_be_silent_actions():
    with pytest.raises(ValueError):repair(np.arange(100),np.arange(99),np.ones((99,3))/3,np.full(99,2))
    with pytest.raises(ValueError):repair(np.arange(100),np.arange(100),np.full((100,3),np.nan),np.full(100,2))

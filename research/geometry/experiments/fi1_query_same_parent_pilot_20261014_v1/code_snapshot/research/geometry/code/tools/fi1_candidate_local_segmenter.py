"""Small candidate-conditioned point graph; frozen image backbone is upstream."""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

PARAMETERS=dict(seed=20261012,candidates_per_source=4,epochs=12,batch_candidates=4,
                loss_points=4096,learning_rate=.001,weight_decay=.0001,hidden=64,
                neighbors=8,confidence=.8,min_observation_groups=2,min_mask_points=100,
                visual_dimension=32,dice_weight=.5,good_consistency_weight=.25,
                good_iou=.95)
FEATURE_DIM=64
ARMS=('old30_selected6','gd10_selected6')


def graph(xyz,k=8):
    n=len(xyz);distance,index=cKDTree(xyz).query(xyz,k=min(k+1,n))
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    # Keep the exact point domain; no voxel or superpoint membership decision.
    if index.shape[1]>1:distance,index=distance[:,1:],index[:,1:]
    while index.shape[1]<k:
        index=np.column_stack((index,np.arange(n)));distance=np.column_stack((distance,np.zeros(n)))
    weight=np.exp(-distance/.05);weight/=np.maximum(weight.sum(axis=1,keepdims=True),1e-12)
    return index.astype(np.int32),weight.astype(np.float16)


class CandidateLocalSegmenter(nn.Module):
    def __init__(self,dimension=FEATURE_DIM,width=64):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Linear(width,width),nn.LayerNorm(width))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width),nn.LayerNorm(width)) for _ in range(2)])
        self.decoder=nn.Sequential(nn.Linear(4*width,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,3))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,x,neighbors,weights,parts):
        h=self.embed(x)
        for layer in self.messages:
            nearby=(h[neighbors]*weights[...,None]).sum(1)
            h=h+layer(torch.cat((h,nearby),1))
        queries=[]
        for start,end in parts:
            member=x[start:end,9]>.5
            if not member.any():raise ValueError('candidate has no parent points')
            own=h[start:end][member]
            mean=own.mean(0).expand(end-start,-1);maximum=own.max(0).values.expand(end-start,-1)
            queries.append(torch.cat((h[start:end],mean,maximum,h[start:end]-mean),1))
        logits=self.decoder(torch.cat(queries))
        prior=4*x[:,9]-2
        return logits+torch.stack((prior,-prior,torch.zeros_like(prior)),1)


def segmentation_loss(logits,labels,parent,good,indices=None):
    if indices is not None:logits,labels,parent=logits[indices],labels[indices],parent[indices]
    # target / assigned other instance / annotation or identity unknown.
    loss=torch.nn.functional.cross_entropy(logits,labels)
    probability=logits.softmax(1)[:,0];truth=(labels==0).float();known=labels!=2
    dice=1-(2*(probability[known]*truth[known]).sum()+1)/(probability[known].sum()+truth[known].sum()+1)
    loss=loss+PARAMETERS['dice_weight']*dice
    if good:
        keep=torch.nn.functional.binary_cross_entropy(probability.clamp(1e-6,1-1e-6),parent.float())
        loss=loss+PARAMETERS['good_consistency_weight']*keep
    return loss


def repair(parent,halo,probability,observed_groups,parameters=PARAMETERS):
    if probability.shape!=(len(halo),3) or len(observed_groups)!=len(halo):raise ValueError('unaligned point evidence')
    if not np.array_equal(halo,np.unique(halo)) or not np.isin(parent,halo).all():raise ValueError('invalid local domain')
    p=np.asarray(probability,dtype=np.float32)
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=.002):raise ValueError('invalid probabilities')
    present=np.isin(halo,parent);visible=observed_groups>=parameters['min_observation_groups']
    add=~present&visible&(p[:,0]>=parameters['confidence'])
    remove=present&visible&(p[:,1]>=parameters['confidence'])
    kept=halo[(present&~remove)|add]
    if len(kept)<parameters['min_mask_points']:
        return parent.copy(),dict(accepted=False,reason='minimum_mask_abstain',added=0,deleted=0)
    return kept,dict(accepted=bool(add.any() or remove.any()),reason='changed' if add.any() or remove.any() else 'keep',
                     added=int(add.sum()),deleted=int(remove.sum()),unobserved_parent_retained=int((present&~visible).sum()))

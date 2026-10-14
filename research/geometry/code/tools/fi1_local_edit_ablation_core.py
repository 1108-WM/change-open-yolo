"""Fixed-observation correction supervision and candidate identity ablations."""
import numpy as np
import torch
from torch import nn
from tools import fi1_candidate_local_segmenter as base

VARIANTS=('A0','B','C')
STRATA=('delete','add','neighbor','keep','unknown')
SCOPE=('edit','keep','outside_scope','unannotated','ambiguous','small_instance')
PARAMETERS=dict(seed=20261012,epochs=12,pilot_epochs=40,pilot_per_stratum=4,loss_points=4096,
                strata_weights=(.25,.25,.20,.20,.10),candidate_weight=.5,relation_weight=.25,
                learning_rate=.001,weight_decay=.0001,batch_candidates=4,
                candidate_edit_threshold=.7,point_threshold=.8,parent_representatives=8,competitors=4)


def strata(labels,parent,visible,x):
    labels=np.asarray(labels);parent=np.asarray(parent,bool);visible=np.asarray(visible)>=2
    delete=parent&(labels==1)&visible;add=~parent&(labels==0)&visible
    # Assigned neighboring points are negatives only through training GT, not membership alone.
    hard=~parent&(labels==1)&visible&(x[:,10]<=1/3)
    other=~parent&(labels==1)&visible&~hard
    masks=[delete,add,hard,(parent&(labels==0))|other,labels==2]
    return [np.flatnonzero(m) for m in masks]


def balanced_sample(groups,rng,n=4096):
    active=[i for i,g in enumerate(groups) if len(g)]
    if not active:raise ValueError('no supervised point')
    mass=np.asarray(PARAMETERS['strata_weights'])[active];mass=mass/mass.sum()
    quota=np.floor(n*mass).astype(int);quota[-1]+=n-int(quota.sum())
    indices=np.concatenate([rng.choice(groups[i],int(q),replace=len(groups[i])<q) for i,q in zip(active,quota)])
    rng.shuffle(indices)
    return indices


def balanced_loss(logits,labels,parent,good,groups,rng):
    indices=balanced_sample(groups,rng);indices=torch.tensor(indices,device=logits.device)
    return base.segmentation_loss(logits,labels,parent,good,indices)


def representatives(x,parent,k=8):
    points=np.flatnonzero(parent);xyz=x[points,:3]
    first=int(np.argmin(np.sum((xyz-xyz.mean(0))**2,axis=1)));selected=[first]
    distance=np.sum((xyz-xyz[first])**2,axis=1)
    for _ in range(1,min(k,len(points))):
        nxt=int(np.argmax(distance));selected.append(nxt);distance=np.minimum(distance,np.sum((xyz-xyz[nxt])**2,axis=1))
    return points[np.asarray(selected)].astype(np.int32)


class IdentitySegmenter(nn.Module):
    def __init__(self):
        super().__init__();self.base=base.CandidateLocalSegmenter()
        self.relation=nn.Sequential(nn.Linear(64*3+2,64),nn.GELU(),nn.Linear(64,3))
        self.identity=nn.Sequential(nn.Linear(64*3+7,64),nn.GELU(),nn.Linear(64,3))
        self.edit=nn.Sequential(nn.Linear(64*3+8,64),nn.GELU(),nn.Linear(64,len(SCOPE)))
        nn.init.zeros_(self.identity[-1].weight);nn.init.zeros_(self.identity[-1].bias)

    def forward(self,x,neighbors,weights,parts,conditions):
        h=self.base.embed(x)
        for layer in self.base.messages:
            h=h+layer(torch.cat((h,(h[neighbors]*weights[...,None]).sum(1)),1))
        outputs=[];candidate=[];relations=[]
        for (start,end),condition in zip(parts,conditions):
            xx=x[start:end];hh=h[start:end];member=xx[:,9]>.5;own=hh[member]
            mean=own.mean(0);maximum=own.max(0).values;n=len(hh)
            original=self.base.decoder(torch.cat((hh,mean.expand(n,-1),maximum.expand(n,-1),hh-mean),1))
            prior=4*xx[:,9]-2;original=original+torch.stack((prior,-prior,torch.zeros_like(prior)),1)
            prototypes=hh[condition['representatives']]
            normalized=torch.nn.functional.normalize(hh,dim=-1)
            own_normal=torch.nn.functional.normalize(prototypes,dim=-1)
            spatial=(condition['own_distance'] if 'own_distance' in condition else torch.cdist(xx[:,:3],xx[condition['representatives'],:3]))/0.15
            attention=(3*(normalized@own_normal.T)-spatial).softmax(1)
            local_own=attention@prototypes
            masks=condition['membership'];distance=condition['distance'];valid=condition['valid']
            queries=[]
            for k in range(masks.shape[1]):
                q=hh[masks[:,k]].mean(0) if valid[k] else torch.zeros_like(mean);queries.append(q)
            queries=torch.stack(queries);overlap=condition['overlap']
            relation_logits=self.relation(torch.cat((mean.expand(len(queries),-1),queries,queries-mean,overlap[:,None],valid.float()[:,None]),1));relations.append(relation_logits)
            similarity=normalized@torch.nn.functional.normalize(queries,dim=-1).T
            score=(3*similarity-distance/.15).masked_fill(~valid[None],-1e4)
            neighbor_attention=score.softmax(1)*valid[None]
            neighbor_attention=neighbor_attention/neighbor_attention.sum(1,keepdim=True).clamp_min(1e-8)
            competitor=neighbor_attention@queries
            near=(neighbor_attention*distance).sum(1,keepdim=True)
            support=(neighbor_attention*masks.float()).sum(1,keepdim=True)
            relation_probability=neighbor_attention@relation_logits.softmax(1)
            own_sim=(normalized*torch.nn.functional.normalize(local_own,dim=-1)).sum(1,keepdim=True)
            other_sim=(neighbor_attention*similarity).sum(1,keepdim=True)
            local_overlap=(neighbor_attention*overlap[None]).sum(1,keepdim=True)
            extra=torch.cat((hh,hh-local_own,hh-competitor,own_sim-other_sim,xx[:,10:11]-near/.15,support,local_overlap,relation_probability),1)
            logits=original+self.identity(extra);outputs.append(logits)
            prob=logits.softmax(1)
            outside=~member;stats=torch.stack((xx[:,59].mean(),torch.log1p(member.sum().float())/10,torch.log1p(outside.sum().float())/10,
                valid.float().mean(),prob[member,0].mean(),prob[member,1].mean(),prob[outside,0].mean() if outside.any() else prob[:,0].mean(),prob[:,2].mean()))
            candidate.append(self.edit(torch.cat((mean,maximum,queries[valid].mean(0) if valid.any() else torch.zeros_like(mean),stats))))
        return torch.cat(outputs),torch.stack(candidate),relations


def model(variant):
    return IdentitySegmenter() if variant=='C' else base.CandidateLocalSegmenter()


def repair(parent,halo,probability,groups,variant,candidate_probability=None):
    if variant=='C':
        cp=np.asarray(candidate_probability,dtype=np.float32)
        if cp.shape!=(len(SCOPE),) or not np.isfinite(cp).all() or not np.isclose(cp.sum(),1,atol=.002):raise ValueError('candidate probability')
        if cp[0]<PARAMETERS['candidate_edit_threshold']:
            return parent.copy(),dict(accepted=False,reason='candidate_abstain_'+SCOPE[int(cp.argmax())],added=0,deleted=0)
    return base.repair(parent,halo,probability,groups)

"""Independent instance relations and candidate editability.

Relation targets never depend on evaluation classes or parent eligibility.
"""
from collections import Counter
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from tools import fi1_region_affiliation_core as old

PARAMETERS=dict(seed=20261013,pair_epochs=24,parent_epochs=60,region_batch=256,
                parent_batch=32,width=48,learning_rate=.001,weight_decay=.0001,
                parent_tokens=128,pair_input_dimension=57,references=6,
                relation_thresholds=(.7,.8,.9,.95),edit_thresholds=(.5,.7,.9,.95),min_points=100)
PARENT_STATES=('edit','keep','out_of_scope','unannotated','ambiguous')
REFERENCE_NAMES=('retained_boundary','retained_parent','competitor0','competitor1','competitor2','competitor3')


def token_groups(tokens,mask,roles):
    group=np.where(mask,roles,-1).astype(np.int8)
    for k in range(4):group[mask&(roles==3)&(tokens[:,71+k]>.5)]=3+k
    return group


def relation_context(tokens,mask,roles,neighbors,weights):
    """Keep frozen spatial samples; remove parent-role shortcuts from identity input."""
    x=np.asarray(tokens,np.float32);group=token_groups(x,mask,roles);xyz=x[:,:3];nodes=np.zeros((96,57),np.float32)
    centers={};extent={}
    for k in range(7):
        ids=np.flatnonzero(group==k)
        if not len(ids):continue
        centers[k]=xyz[ids].mean(0);extent[k]=np.ptp(xyz[ids],axis=0)
        nodes[ids]=np.c_[np.clip((xyz[ids]-centers[k])/.15,-10,10),x[ids,3:9],x[ids,16:64]]
    w=np.asarray(weights,np.float32).copy();w*=mask[:,None]&mask[neighbors]&(group[:,None]==group[neighbors]);w/=np.maximum(w.sum(1,keepdims=True),1e-12)
    cross=np.zeros((6,48,2),np.int16);valid=np.zeros((6,48),bool);distance=np.zeros((6,48),np.float32);stats=np.zeros((6,12),np.float32)
    own=np.flatnonzero(group==0)
    for slot in range(6):
        ref=np.flatnonzero(group==slot+1)
        if not len(own) or not len(ref):continue
        _,j=cKDTree(xyz[ref]).query(xyz[own]);_,i=cKDTree(xyz[own]).query(xyz[ref]);edges=np.unique(np.r_[np.c_[own,ref[j]],np.c_[own[i],ref]],axis=0)
        assert len(edges)<=48;cross[slot,:len(edges)]=edges;valid[slot,:len(edges)]=True;distance[slot,:len(edges)]=np.minimum(np.linalg.norm(xyz[edges[:,0]]-xyz[edges[:,1]],axis=1)/.3,10)
        color=np.abs(x[own,3:6].mean(0)-x[ref,3:6].mean(0));a=x[own,6:9].mean(0);b=x[ref,6:9].mean(0);normal=1-abs(float(a@b/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b))))
        a=x[own,16:48].mean(0);b=x[ref,16:48].mean(0);visual=1-float(a@b/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b)))
        stats[slot]=np.r_[min(np.linalg.norm(centers[0]-centers[slot+1])/.3,10),np.minimum((extent[0]+extent[slot+1])/2,3),np.minimum(np.abs(extent[0]-extent[slot+1]),3),color,normal,visual]
    return nodes.astype(np.float16),group,w.astype(np.float16),cross,valid,distance.astype(np.float16),stats


def instance_distribution(values):
    values=np.asarray(values);n=len(values);known=Counter(int(v) for v in values if v>=1000)
    return known,n


def pair_truth(region_values,reference_values):
    a,n=instance_distribution(region_values);b,m=instance_distribution(reference_values)
    if not n or not m:return np.array([0.,0.,1.],np.float32)
    same=sum(v*b.get(g,0) for g,v in a.items())/(n*m);known=sum(a.values())*sum(b.values())/(n*m)
    return np.array([same,max(0.,known-same),max(0.,1-known)],np.float32)


class PointEncoder(nn.Module):
    def __init__(self,dimension,width=48):
        super().__init__();self.embed=nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Linear(width,width),nn.LayerNorm(width))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width)) for _ in range(2)])
    def forward(self,x,mask,neighbors,weights):
        h=self.embed(x)*mask[:,:,None];batch=torch.arange(len(x),device=x.device)[:,None,None]
        for layer in self.messages:
            q=(h[batch,neighbors]*weights[:,:,:,None]).sum(2);h=(h+layer(torch.cat((h,q),-1)))*mask[:,:,None]
        return h


def pool(h,valid):
    mean=(h*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
    maximum=h.masked_fill(~valid[:,:,None],-1e4).max(1).values;maximum=torch.where(valid.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
    return torch.cat((mean,maximum),1)


class InstanceRelation(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=PointEncoder(57)
        self.edge=nn.Sequential(nn.Linear(145,48),nn.GELU(),nn.Linear(48,48))
        self.head=nn.Sequential(nn.Linear(396,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,3))
    def forward(self,nodes,mask,group,neighbors,weights,cross,cross_valid,distance,stats):
        h=self.encoder(nodes,mask,neighbors,weights);left=pool(h,mask&(group==0));batch=torch.arange(len(h),device=h.device)[:,None];outputs=[]
        for k in range(6):
            right=pool(h,mask&(group==k+1));idx=cross[:,k];a=h[batch,idx[:,:,0]];b=h[batch,idx[:,:,1]]
            edge=self.edge(torch.cat((a+b,(a-b).abs(),a*b,distance[:,k,:,None]),-1));edges=pool(edge,cross_valid[:,k])
            relation=torch.cat((left+right,(left-right).abs(),left*right,edges,stats[:,k]),1)
            pred=self.head(relation);exists=(mask&(group==k+1)).any(1)
            fallback=torch.tensor([-20.,-20.,20.],device=h.device).expand_as(pred);outputs.append(torch.where(exists[:,None],pred,fallback))
        return torch.stack(outputs,1)


class ParentEditability(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=PointEncoder(64);self.relation=nn.Sequential(nn.Linear(384,48),nn.GELU(),nn.Linear(48,48))
        self.head=nn.Sequential(nn.Linear(192+16,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,5))
    def forward(self,x,mask,group,neighbors,weights,stats):
        h=self.encoder(x,mask,neighbors,weights);own=pool(h,mask&(group==0));context=[];valid=[]
        for k in range(1,5):
            sel=mask&(group==k);other=pool(h,sel);context.append(self.relation(torch.cat((own,other,own-other,(own-other).abs()),1)));valid.append(sel.any(1))
        r=torch.stack(context,1);v=torch.stack(valid,1);mean=(r*v[:,:,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
        maximum=r.masked_fill(~v[:,:,None],-1e4).max(1).values;maximum=torch.where(v.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
        return self.head(torch.cat((own,mean,maximum,stats),1))


def parent_context(x,xyz,parent,halo,membership,visible):
    member=np.isin(halo,parent);own=old.spread(np.flatnonzero(member),xyz,64);sets=[own]
    for k in range(4):sets.append(old.spread(np.flatnonzero(membership[:,k]),xyz,16))
    output=np.zeros((128,64),np.float32);valid=np.zeros(128,bool);group=np.full(128,-1,np.int8);indices=np.full(128,-1,np.int32)
    for k,ids in enumerate(sets):
        lo=0 if k==0 else 64+(k-1)*16;hi=lo+len(ids);output[lo:hi]=x[ids];valid[lo:hi]=True;group[lo:hi]=k;indices[lo:hi]=ids
    nn=np.broadcast_to(np.arange(128)[:,None],(128,4)).copy();w=np.zeros((128,4),np.float32)
    ids=np.flatnonzero(valid);d,j=cKDTree(xyz[indices[ids]]).query(xyz[indices[ids]],k=min(5,len(ids)))
    if d.ndim==1:d=d[:,None];j=j[:,None]
    d,j=d[:,1:],j[:,1:]
    for k in range(min(4,j.shape[1])):nn[ids,k]=ids[j[:,k]];w[ids,k]=np.exp(-d[:,k]/.05)*(d[:,k]<=.15)
    w/=np.maximum(w.sum(1,keepdims=True),1e-12)
    extent=np.ptp(xyz[member],axis=0);fraction=member.mean();stats=np.r_[np.log1p(len(parent))/12,np.log1p(len(halo))/12,fraction,np.minimum(extent,4),
          float((visible[member]>=2).mean()),x[member,11].mean(),membership[member].mean(0),
          [float(membership[:,k].any()) for k in range(4)]].astype(np.float32);assert len(stats)==16
    return output.astype(np.float16),valid,group,nn.astype(np.int16),w.astype(np.float16),stats,indices


def relation_score(probability,valid):
    """Require positive neighbor identity as well as difference from parent body."""
    pr=np.asarray(probability,np.float32);neighbor=np.where(valid[:,2:],pr[:,2:,0],0).max(1)
    return np.minimum(pr[:,1,1],neighbor)*valid[:,1]


def execute(parent,halo,blocks,prediction,rule):
    if rule.get('abstain',False):return parent.copy(),dict(reason='calibration_abstain',selected=[],deleted=0)
    if float(prediction['parent'][0])<rule['edit']:return parent.copy(),dict(reason='parent_not_editable',selected=[],deleted=0)
    scores=relation_score(prediction['relations'],prediction['valid']);selected=np.flatnonzero(scores>=rule['relation'])
    union=np.unique(np.concatenate([blocks[j] for j in selected])) if len(selected) else np.empty(0,np.int32)
    # Recheck the genuine union, with overlap assigned worst same-parent belief.
    same=np.zeros(len(halo),np.float32)
    for j in selected:same[blocks[j]]=np.maximum(same[blocks[j]],prediction['relations'][j,1,0])
    if len(union) and float(same[union].mean())>1-rule['relation']+1e-7:
        return parent.copy(),dict(reason='union_relation_conflict',selected=selected.tolist(),deleted=0)
    q=np.setdiff1d(parent,halo[union],assume_unique=True)
    if len(q)<100:return parent.copy(),dict(reason='minimum_mask_abstain',selected=selected.tolist(),deleted=0)
    return q,dict(reason='changed' if len(union) else 'keep',selected=selected.tolist(),deleted=len(parent)-len(q))

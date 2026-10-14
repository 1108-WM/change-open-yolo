"""GT-free deletion components and supervised region/action consequences."""
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tools import fi1_candidate_local_segmenter as base

PARAMETERS=dict(seed=20261013,proposal_epochs=12,verifier_epochs=40,proposal_other_cost=3.,
                confidence=.8,visible_groups=2,block_acceptance=.9,min_points=100,
                descriptor_dimension=140,regions=7,verifier_batch=128,
                learning_rate=.001,weight_decay=.0001)


def deletion_blocks(parent,halo,probability,visible,neighbors):
    """Exactly the audit's undirected frozen 8-NN components, before min100."""
    removed=np.flatnonzero(np.isin(halo,parent)&(visible>=2)&(probability[:,1]>=.8))
    if not len(removed):return []
    lookup=np.full(len(halo),-1,np.int32);lookup[removed]=np.arange(len(removed))
    edges=lookup[neighbors[removed]];i,j=np.nonzero(edges>=0)
    adjacency=csr_matrix((np.ones(len(i)),(i,edges[i,j])),shape=(len(removed),len(removed)))
    n,labels=connected_components(adjacency,directed=False)
    order=np.argsort(labels,kind='stable');cuts=np.cumsum(np.bincount(labels))[:-1]
    return [np.sort(removed[k]).astype(np.int32) for k in np.split(order,cuts)]


def delete(parent,halo,blocks,selected=None):
    if selected is None:selected=np.ones(len(blocks),bool)
    removed=np.concatenate([b for b,yes in zip(blocks,selected) if yes]) if any(selected) else np.empty(0,np.int32)
    result=np.setdiff1d(parent,halo[removed],assume_unique=True)
    return result if len(result)>=PARAMETERS['min_points'] else parent.copy()


def corrected_loss(logits,y,parent,good):
    """Every point once; bounded cost, full-domain Dice and good preservation."""
    cost=torch.ones(len(y),device=y.device);cost[parent&(y==1)]=PARAMETERS['proposal_other_cost']
    ce=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*cost).sum()/cost.sum()
    probability=logits.softmax(1)[:,0];known=y!=2;truth=(y==0).float()
    dice=1-(2*(probability[known]*truth[known]).sum()+1)/(probability[known].sum()+truth[known].sum()+1)
    loss=ce+.5*dice
    if good:loss=loss+.25*torch.nn.functional.binary_cross_entropy(probability.clamp(1e-6,1-1e-6),parent.float())
    return loss


def stable_supervision(gt,parent,halo,meta,classes):
    """Never force a mixed/out-of-scope parent to follow a small best-IoU hit."""
    target=meta['target'];purity=0. if target is None else float((gt[parent]==target).mean())
    stable=target is not None and target//1000 in classes and purity>=.6
    y=np.full(len(halo),2,np.uint8)
    if stable:y[gt[halo]>=1000]=1;y[gt[halo]==target]=0
    return y,dict(stable=bool(stable),target=target if stable else None,purity=purity,
                  good=bool(stable and meta['good']),original_target=target)


def descriptor(x,probability,visible,indices):
    if not len(indices):return np.zeros(140,np.float32)
    values=np.concatenate((x[indices],probability[indices].astype(np.float32)),1)
    return np.concatenate((values.mean(0),values.std(0),np.ptp(x[indices,:3],axis=0),
                           [np.log1p(len(indices))/10,float((visible[indices]>=2).mean()),1.])).astype(np.float32)


def region_features(e,probability,blocks):
    x=e['x'];nn=e['neighbors'];parent=x[:,9]>.5;visible=e['visible'];condition=e['condition']
    result=[]
    for block in blocks:
        remain=parent.copy();remain[block]=False
        around=np.unique(nn[block]);around=around[remain[around]]
        # Local retained context; parent and competing regions remain separate.
        regions=[block,around,np.flatnonzero(remain)]+[np.flatnonzero(condition['membership'][:,k]) for k in range(4)]
        result.append(np.stack([descriptor(x,probability,visible,r) for r in regions]))
    return np.stack(result) if result else np.empty((0,7,140),np.float32)


def consequences(gt,parent,halo,blocks,meta):
    """Specific proposed actions, not theoretical candidate reparability."""
    result=[];gid=meta['target'];stable=meta['stable']
    total=0 if not stable else int((gt==gid).sum());hit=0 if not stable else int((gt[parent]==gid).sum())
    before=hit/(len(parent)+total-hit) if stable else 0.
    for block in blocks:
        values=gt[halo[block]];n=len(block)
        if stable:
            target=int((values==gid).sum());other=int(((values>=1000)&(values!=gid)).sum());unknown=n-target-other
            after=(hit-target)/(len(parent)-n+total-(hit-target)) if len(parent)-n>=100 else before
            delta=after-before;safe=target==0 and unknown==0 and delta>0
            output=[float(safe),other/n,target/n,unknown/n,float(np.clip(delta*10,-1,1))]
        else:target=other=0;unknown=n;delta=None;output=[0.,0.,0.,1.,0.]
        result.append((output,dict(stable=stable,target_deleted=target,pollution_deleted=other,unknown_deleted=unknown,delta=delta,points=n)))
    return result


class BlockVerifier(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='summary':self.network=nn.Sequential(nn.Linear(7*140,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,5))
        else:
            self.embed=nn.Sequential(nn.Linear(140,64),nn.GELU(),nn.Linear(64,64))
            self.relation=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,64))
            self.network=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,5))

    def forward(self,x):
        if self.kind=='summary':return self.network(x.flatten(1))
        h=self.embed(x);block=h[:,:1];q=h[:,1:];b=block.expand(-1,6,-1)
        relation=self.relation(torch.cat((b,q,b-q,(b-q).abs()),-1));valid=x[:,1:,-1]>.5
        mean=(relation*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        maximum=relation.masked_fill(~valid[:,:,None],-1e4).max(1).values
        maximum=torch.where(valid.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
        return self.network(torch.cat((h[:,0],mean,maximum),1))


def decoded(logits):return torch.cat((logits[:,:4].sigmoid(),logits[:,4:5].tanh()/10),1)


def verifier_loss(logits,truth,weights):
    classification=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],truth[:,0],reduction='none')
    fractions=(logits[:,1:4].sigmoid()-truth[:,1:4]).square().mean(1)
    delta=torch.nn.functional.smooth_l1_loss(logits[:,4].tanh(),truth[:,4],reduction='none')
    return ((classification+2*fractions+.5*delta)*weights).mean()

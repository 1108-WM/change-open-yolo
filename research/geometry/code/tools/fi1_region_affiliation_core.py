"""Same-source region ownership, explicit missing views, and union risk.

No labels are accepted by token, view-evidence, or execution functions.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from tools.fi1_frontend_protocol import digest

PARAMETERS=dict(seed=20261013,outer_fold=0,train_scenes=112,calibration_scenes=24,
                holdout_scenes=24,budget=96,views=6,tokens_per_role=24,
                token_dimension=75,view_dimension=20,stats_dimension=16,
                width=48,epochs=24,batch=256,learning_rate=.001,weight_decay=.0001,
                graph_k=4,graph_radius=.15,min_points=100,
                calibration_safe=(.5,.7,.8,.9,.95,.98),
                calibration_target=(.02,.05,.1),calibration_unknown=(.02,.05,.1),
                calibration_union_risk=(.005,.01,.02))
VIEW_NAMES=('present','region_visible','retained_visible','neighbor_visible','has_masks',
            'parent_match','parent_ambiguous','neighbor_match','neighbor_ambiguous',
            'parent_region_support','neighbor_region_support','parent_anchor_support',
            'neighbor_anchor_support','any_region_support','parent_neighbor_difference',
            'same_mask','parent_quality','neighbor_quality','mask_count','missing_observation')


def split_scenes(scenes,folds):
    scope=[s for s in scenes if folds[s]!=PARAMETERS['outer_fold']]
    groups={}
    for s in scope:groups.setdefault(s.split('_')[0],[]).append(s)
    ordered=sorted(groups,key=lambda g:(digest([PARAMETERS['seed'],'region-affiliation',g]),g))
    n=len(ordered);a=int(n*.7);b=a+int(n*.15)
    return {name:sorted(s for g in gg for s in groups[g]) for name,gg in
            (('train',ordered[:a]),('calibration',ordered[a:b]),('holdout',ordered[b:]))}


def spread(indices,xyz,limit):
    """Deterministic spatial samples without replacement; zero padding is masked."""
    ids=np.unique(indices).astype(np.int32)
    if len(ids)<=limit:return ids
    pts=xyz[ids];distance=np.sum((pts-pts.mean(0))**2,1);selected=[]
    for _ in range(limit):
        j=int(np.argmax(distance));selected.append(j)
        distance=np.minimum(distance,np.sum((pts-pts[j])**2,1));distance[selected]=-1
    return ids[selected]


def tokens(x,xyz,parent,neighbors,membership,block):
    remain=parent.copy();remain[block]=False
    ring=np.unique(neighbors[block]);ring=ring[remain[ring]]
    sets=[spread(block,xyz,24),spread(ring,xyz,24),spread(np.flatnonzero(remain),xyz,24)]
    competing=[];roles=[]
    for k in range(4):
        ids=np.flatnonzero(membership[:,k]);ids=np.setdiff1d(ids,block)
        if len(ids):
            # Competing anchors closest to this region, then spread locally.
            d,_=cKDTree(xyz[block]).query(xyz[ids]);ids=ids[np.argsort(d,kind='stable')[:128]]
        sample=spread(ids,xyz,6);competing.extend(sample);roles.extend([k]*len(sample))
    sets.append(np.asarray(competing,np.int32));center=xyz[block].mean(0)
    out=np.zeros((96,75),np.float32);mask=np.zeros(96,bool);role=np.full(96,-1,np.int8)
    index=np.full(96,-1,np.int32)
    for k,ids in enumerate(sets):
        lo=24*k;hi=lo+len(ids);index[lo:hi]=ids;mask[lo:hi]=True;role[lo:hi]=k
        out[lo:hi,:64]=x[ids];out[lo:hi,64:67]=np.clip((xyz[ids]-center)/.3,-4,4)
        out[lo:hi,67+k]=1
        if k==3:
            for j,competitor in enumerate(roles):out[lo+j,71+competitor]=1
    valid=np.flatnonzero(mask);nnidx=np.broadcast_to(np.arange(96)[:,None],(96,4)).copy();weights=np.zeros((96,4),np.float32)
    if len(valid):
        pts=xyz[index[valid]];dd,ii=cKDTree(pts).query(pts,k=min(5,len(valid)))
        if dd.ndim==1:dd=dd[:,None];ii=ii[:,None]
        if dd.shape[1]>1:dd,ii=dd[:,1:],ii[:,1:]
        else:dd=np.zeros((len(valid),1));ii=np.arange(len(valid))[:,None]
        for k in range(min(4,dd.shape[1])):
            nnidx[valid,k]=valid[ii[:,k]];weights[valid,k]=np.exp(-dd[:,k]/.05)*(dd[:,k]<=.15)
        weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
    extent=np.ptp(xyz[block],axis=0)
    stats=np.r_[np.log1p(len(block))/10,np.log1p(parent.sum())/12,len(block)/parent.sum(),
                np.minimum(extent,2),x[block,11].mean(),x[block,59].mean(),
                len(ring)/max(1,len(block)),membership[block].mean(0),
                float(len(ring)>0),float(len(competing)>0),float(remain.any())].astype(np.float32)
    assert len(stats)==16
    return out.astype(np.float16),mask,role,nnidx.astype(np.int16),weights.astype(np.float16),stats,index


def match(mask_membership,visible,anchor):
    anchor=anchor[visible[anchor]]
    if len(anchor)<3 or not mask_membership.shape[1]:return None,False,0.,0.
    hits=mask_membership[anchor].sum(0);sizes=(mask_membership&visible[:,None]).sum(0)
    return match_counts(hits,len(anchor),sizes)


def match_counts(hits,count,sizes):
    if count<3 or not len(sizes):return None,False,0.,0.
    score=hits/np.maximum(1,count+sizes-hits)
    eligible=np.flatnonzero((hits>=3)&(score>=.05))
    if not len(eligible):return None,False,0.,0.
    order=sorted(eligible,key=lambda k:(-float(score[k]),int(k)));best=order[0]
    ambiguous=len(order)>1 and (score[order[1]]>=.9*score[best] or score[best]-score[order[1]]<.02)
    return int(best),bool(ambiguous),float(hits[best]/count),float(score[best])


def view_context(parent,membership,visible,masks):
    known=masks&visible[:,None]
    return dict(sizes=known.sum(0),parent_hits=known[parent].sum(0),parent_count=int((parent&visible).sum()),
                neighbor_hits=[known[membership[:,k]].sum(0) for k in range(4)],
                neighbor_count=[int((membership[:,k]&visible).sum()) for k in range(4)])


def local_view(block,parent,membership,visible,masks,quality,context=None):
    """Mask IDs share a frame; absence and ambiguous matching are explicit."""
    remain=np.flatnonzero(parent);remain=np.setdiff1d(remain,block)
    regional=block[visible[block]];v=np.zeros(20,np.float32);v[0]=1
    v[1]=len(regional)/len(block);v[2]=visible[remain].mean() if len(remain) else 0
    v[3]=max([float(visible[membership[:,k]].mean()) for k in range(4) if membership[:,k].any()] or [0])
    v[4]=bool(masks.shape[1]);v[18]=np.log1p(masks.shape[1])/5
    context=view_context(parent,membership,visible,masks) if context is None else context
    hits=masks[regional].sum(0)
    own,ambiguity,coverage,_=match_counts(context['parent_hits']-hits,context['parent_count']-len(regional),context['sizes'])
    v[5]=own is not None;v[6]=ambiguity;v[11]=coverage
    if own is not None:v[16]=quality[own]
    competitors=[]
    for k in range(4):
        ids=np.flatnonzero(membership[:,k]);ids=np.setdiff1d(ids,block)
        excluded=regional[membership[regional,k]]
        chosen,amb,cov,score=match_counts(context['neighbor_hits'][k]-masks[excluded].sum(0),context['neighbor_count'][k]-len(excluded),context['sizes'])
        if chosen is not None:competitors.append((chosen,amb,cov,score))
    if competitors:
        # Neighbor whose matched instance most supports the region; no GT.
        selected=max(competitors,key=lambda t:(float(masks[regional,t[0]].mean()) if len(regional) else 0,t[3],-t[0]))
        other,amb,cov,_=selected;v[7]=1;v[8]=amb;v[12]=cov;v[17]=quality[other];v[15]=own==other if own is not None else 0
        if len(regional):v[10]=masks[regional,other].mean()
    if len(regional):
        if own is not None:v[9]=masks[regional,own].mean()
        if masks.shape[1]:v[13]=masks[regional].any(1).mean()
    v[14]=v[9]-v[10]
    v[19]=not (len(regional)>0 and masks.shape[1]>0 and own is not None and not ambiguity)
    return v


class RegionAffiliation(nn.Module):
    def __init__(self):
        super().__init__();w=48
        self.embed=nn.Sequential(nn.Linear(75,w),nn.GELU(),nn.Linear(w,w),nn.LayerNorm(w))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*w,w),nn.GELU(),nn.Linear(w,w)) for _ in range(2)])
        self.relation=nn.Sequential(nn.Linear(4*w,w),nn.GELU(),nn.Linear(w,w))
        self.view=nn.Sequential(nn.Linear(20,32),nn.GELU(),nn.Linear(32,32))
        self.output=nn.Sequential(nn.Linear(4*w+64+16,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,4))

    def forward(self,x,mask,roles,neighbors,weights,stats,views):
        h=self.embed(x)*mask[:,:,None];batch=torch.arange(len(x),device=x.device)[:,None,None]
        for layer in self.messages:
            q=(h[batch,neighbors]*weights[:,:,:,None]).sum(2)
            h=(h+layer(torch.cat((h,q),-1)))*mask[:,:,None]
        def pool(sel):
            denom=sel.sum(1,keepdim=True).clamp_min(1);mean=(h*sel[:,:,None]).sum(1)/denom
            maximum=h.masked_fill(~sel[:,:,None],-1e4).max(1).values
            maximum=torch.where(sel.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
            return mean,maximum
        own,maximum=pool(mask&(roles==0));relations=[];valid=[]
        selections=[mask&(roles==k) for k in (1,2)]+[mask&(roles==3)&(x[:,:,71+k]>.5) for k in range(4)]
        for sel in selections:
            q,_=pool(sel);relations.append(self.relation(torch.cat((own,q,own-q,(own-q).abs()),1)));valid.append(sel.any(1))
        rel=torch.stack(relations,1);valid=torch.stack(valid,1)
        relmean=(rel*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        relmax=rel.masked_fill(~valid[:,:,None],-1e4).max(1).values;relmax=torch.where(valid.any(1,keepdim=True),relmax,torch.zeros_like(relmax))
        vv=self.view(views);present=views[:,:,0]>.5
        vmean=(vv*present[:,:,None]).sum(1)/present.sum(1,keepdim=True).clamp_min(1)
        vmax=vv.masked_fill(~present[:,:,None],-1e4).max(1).values;vmax=torch.where(present.any(1,keepdim=True),vmax,torch.zeros_like(vmax))
        return self.output(torch.cat((own,maximum,relmean,relmax,vmean,vmax,stats),1))


def loss(logits,fractions,safe,weights):
    soft=-(fractions*logits[:,:3].log_softmax(1)).sum(1)
    binary=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,3],safe,reduction='none')
    return ((soft+binary)*weights).sum()/weights.sum().clamp_min(1)


def decode(logits):return torch.cat((logits[:,:3].softmax(1),logits[:,3:4].sigmoid()),1)


def execute(parent,halo,blocks,probability,rule):
    if rule.get('abstain',False):return parent.copy(),dict(reason='calibration_abstain',selected=[],deleted=0,risk=0.,unknown=0.)
    pr=np.asarray(probability,np.float32);assert pr.shape==(len(blocks),4) and np.isfinite(pr).all()
    selected=np.flatnonzero((pr[:,3]>=rule['safe'])&(pr[:,0]<=rule['target'])&(pr[:,2]<=rule['unknown']))
    # Risk on actual point-set union: duplicates never increase support.
    target=np.zeros(len(halo),np.float32);unknown=np.zeros(len(halo),np.float32);covered=np.zeros(len(halo),bool)
    for i in selected:
        b=blocks[i];covered[b]=True;target[b]=np.maximum(target[b],pr[i,0]);unknown[b]=np.maximum(unknown[b],pr[i,2])
    ids=np.flatnonzero(covered);risk=float(target[ids].mean()) if len(ids) else 0.;missing=float(unknown[ids].mean()) if len(ids) else 0.
    if risk>rule['union_risk'] or missing>rule['unknown']:
        return parent.copy(),dict(reason='union_risk_abstain',selected=selected.tolist(),deleted=0,risk=risk,unknown=missing)
    result=np.setdiff1d(parent,halo[ids],assume_unique=True)
    if len(result)<100:return parent.copy(),dict(reason='minimum_mask_abstain',selected=selected.tolist(),deleted=0,risk=risk,unknown=missing)
    return result,dict(reason='changed' if len(ids) else 'keep',selected=selected.tolist(),deleted=len(parent)-len(result),risk=risk,unknown=missing)

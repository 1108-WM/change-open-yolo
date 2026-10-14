"""Bounded class-member residual with multi-IoU, class-balanced pair ranking."""
import numpy as np
from scipy.special import expit,logit
import torch
from torch import nn

BOUND=1.0
ANCHOR=.1
EPOCHS=20
SEED=20261010


def base_logits(scores):
    # Numerical floor for model input/loss only; actual score endpoints remain exact.
    return logit(np.clip(np.asarray(scores),1e-6,1-1e-6))


def adjusted(scores,delta):
    s=np.asarray(scores,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
    if s.shape!=delta.shape or not np.isfinite(s).all() or not np.isfinite(delta).all():raise ValueError('alignment/finite')
    if np.any((s<0)|(s>1)) or np.any(abs(delta)>BOUND+1e-7):raise ValueError('domain')
    odds=np.exp(delta)
    return s*odds/(1-s+s*odds)


class Residual(nn.Module):
    def __init__(self,dimensions=28):
        super().__init__();self.hidden=nn.Linear(dimensions,32);self.head=nn.Linear(32,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return BOUND*torch.tanh(self.head(torch.tanh(self.hidden(x))).squeeze(-1))


def member_states(scores,geometry_index,classes,sizes,own,targets,ignore_fraction,ignored_match,thresholds):
    """S-fixed representatives; 1 representative,2 duplicate,0 FP,-1 ignore.

    This is a training ranking surrogate, not frozen TP flags for new AP.
    own/targets are geometry x class valid-target max IoU and target identity.
    ignored_match includes evaluator's same-class small/group targets.
    """
    gi=np.asarray(geometry_index);ci=np.asarray(classes);scores=np.asarray(scores)
    q=own[gi,ci];identity=targets[gi,ci]
    states=np.full((len(scores),len(thresholds)),-1,dtype=np.int8)
    for ti,t in enumerate(thresholds):
        valid=sizes[gi]>=100;positive=valid & (q>t)
        fp=valid & ~positive & (ignored_match[gi,ci]<=t) & (ignore_fraction[gi,ci]<=t)
        states[fp,ti]=0;states[positive,ti]=2
        best={}
        for i in np.flatnonzero(positive):
            g=int(identity[i]);assert g>=1000
            if g not in best or scores[i]>scores[best[g]]:best[g]=int(i)
        states[list(best.values()),ti]=1
    return states,identity


def make_pairs(scores,classes,scene_ids,identities,states,seed):
    """At most128 reps/class/threshold; hard, cross-scene, random and duplicate.

    Identical GT numeric IDs across scenes are different targets. Class IDs and
    GT identities are supervision/sampling only, never model input.
    """
    scores=np.asarray(scores);classes=np.asarray(classes);scene_ids=np.asarray(scene_ids)
    rng=np.random.default_rng(seed);pairs=[];groups=[];kinds=[];group=0
    byclass={c:np.flatnonzero(classes==c) for c in np.unique(classes)}
    for ci,ii in byclass.items():
        for ti in range(states.shape[1]):
            pp=ii[states[ii,ti]==1];nnn=ii[states[ii,ti]==0];dd=ii[states[ii,ti]==2]
            if not len(pp) or not (len(nnn) or len(dd)):continue
            if len(pp)>128:pp=np.sort(rng.choice(pp,128,replace=False))
            hard=nnn[np.lexsort((nnn,-scores[nnn]))][:64]
            dup={}
            for i in dd:dup.setdefault((scene_ids[i],identities[i]),[]).append(int(i))
            start=len(pairs)
            for p in pp:
                choices=[]
                if len(hard):choices.extend((int(n),0) for n in rng.choice(hard,min(4,len(hard)),replace=False))
                cross=nnn[scene_ids[nnn]!=scene_ids[p]]
                if len(cross):
                    top=cross[np.lexsort((cross,-scores[cross]))][:64]
                    choices.extend((int(n),1) for n in rng.choice(top,min(2,len(top)),replace=False))
                if len(nnn):choices.extend((int(n),2) for n in rng.choice(nnn,min(2,len(nnn)),replace=False))
                ds=dup.get((scene_ids[p],identities[p]),[])
                ds=sorted(ds,key=lambda n:(-scores[n],n))[:2]
                choices.extend((n,3) for n in ds)
                seen=set()
                for n,kind in choices:
                    if n in seen or scores[p] in (0,1) or scores[n] in (0,1):continue
                    seen.add(n);pairs.append((int(p),n));groups.append(group);kinds.append(kind)
            if len(pairs)>start:group+=1
    pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2);groups=np.asarray(groups,dtype=np.int64);kinds=np.asarray(kinds,dtype=np.int8)
    assert len(pairs)>0
    # Equal total per class, then its available IoU groups. No inverse frequency explosion.
    gc=np.asarray([classes[p] for p,n in pairs]);weights=np.zeros(len(pairs))
    for ci in np.unique(gc):
        mask=gc==ci;gg=np.unique(groups[mask])
        for g in gg:weights[groups==g]=1/(len(gg)*np.sum(groups==g))
    weights/=weights.sum()
    assert np.all(classes[pairs[:,0]]==classes[pairs[:,1]])
    return dict(pairs=pairs,groups=groups,kinds=kinds,weights=weights)


def ranking_metrics(scores,pairdata):
    p,n=pairdata['pairs'].T;w=pairdata['weights'];margin=base_logits(scores[p])-base_logits(scores[n])
    values=dict(pair_loss=float(np.sum(w*np.logaddexp(0,-margin))),
                inversion=float(np.sum(w*((scores[p]<scores[n])+.5*(scores[p]==scores[n])))),
                pairs=len(p),cross_scene_pairs=int((pairdata['kinds']==1).sum()),duplicate_pairs=int((pairdata['kinds']==3).sum()))
    return values

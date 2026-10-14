"""One feature intervention in the existing candidate-conditioned point graph."""
import hashlib
import numpy as np
import torch
from tools.fi1_candidate_local_segmenter import CandidateLocalSegmenter
from tools.fi1_deletion_block_core import corrected_loss

SEED=20261013
BASE_DIM=64
PRIOR_DIM=128
DIM=192
THRESHOLDS=(.7,.8,.9,.95,.98)

def safe_columns(raw):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]<9:raise ValueError('XYZ/RGB/normal columns required')
    x=np.array(raw[:,:9],dtype=np.float32,copy=True)
    if not np.isfinite(x).all():raise ValueError('invalid allowed inputs')
    return x

def selection_split(scenes):
    groups={}
    for s in scenes:groups.setdefault(s.split('_')[0],[]).append(s)
    order=sorted(groups,key=lambda g:(hashlib.sha256(f'{SEED}:mask3d-feature-selection:{g}'.encode()).hexdigest(),g))
    chosen=order[:16]
    return {'fit':sorted(s for g in order[16:] for s in groups[g]),'selection':sorted(s for g in chosen for s in groups[g])}

def inputs(base,prior,mean,std,arm):
    base=np.asarray(base,np.float32);prior=np.asarray(prior,np.float32)
    if base.shape[1]!=64 or prior.shape!=(len(base),128):raise ValueError('unaligned feature channels')
    extra=np.clip((prior-mean)/std,-10,10) if arm=='B' else np.zeros_like(prior)
    if arm not in ('A','B'):raise ValueError(arm)
    return np.column_stack((base,extra)).astype(np.float32)

def model():return CandidateLocalSegmenter(dimension=DIM,width=64)

def repair(parent,halo,probability,visible,threshold):
    p=np.asarray(probability,np.float32)
    if p.shape!=(len(halo),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=.002):raise ValueError('probabilities')
    member=np.isin(halo,parent);seen=np.asarray(visible)>=2
    q=halo[(member&~(seen&(p[:,1]>=threshold)))|(~member&seen&(p[:,0]>=threshold))]
    return q if len(q)>=100 else parent.copy()

def protection(m):
    edited=m.get('added',0)+m.get('deleted',0)
    return dict(add_precision=not m.get('added',0) or m.get('added_target',0)/m['added']>=.9,
        delete_precision=not m.get('deleted',0) or m.get('deleted_other',0)/m['deleted']>=.9,
        target_loss=m.get('deleted_target',0)/max(1,m.get('changed_parent_target',0))<=.01,
        unknown=m.get('unknown_edits',0)/max(1,edited)<=.1,
        good_rate=m.get('good_degraded',0)/max(1,m.get('good_parents',0))<=.02,
        good_tail=m.get('good_loss_gt_0.1',0)==0,drift=m.get('drift',0)==0,
        support=m.get('actions',0)>=3 and m.get('action_scenes',0)>=2)

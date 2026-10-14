#!/usr/bin/env python3
"""GT-free S800 -> learned L confidence, with complete S-pool context.

Inputs are geometry, actual own evidence, B26 and L weights/normalization.
No historical member/feature/score cache is needed. Historical 8192-row
CPU batch shapes are preserved using zero padding for bitwise replay.
Padding rows supply no evidence and never become outputs.
"""
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch
from tools import run_fi1_native_symmetric_eligibility_development200 as source
from tools import fi1_s_semantic_rank_residual as core
from tools.fi1_frontend_protocol import freeze_json, sha256_file


def build_pool(binding, b26):
    gs, all_points, evidence = source.ctl.base.load_geometry(binding)
    geometry = source.source_pool(gs)  # remove U/Uprime, including shared aliases
    keys = [g['geometry_key'] for g in geometry]
    evidence = {k:evidence[k] for k in keys}
    members = source.symmetric_members(geometry, evidence)
    own = {g['geometry_key']:source.geometry_features(g, evidence[g['geometry_key']]) for g in geometry}
    iou, coverage = source.pair_overlap([all_points[k] for k in keys], binding['point_count'])
    x26 = np.column_stack((np.asarray([own[m['geometry_key']][m['class_index']] for m in members]),
                           source.context_features(members, keys, iou, coverage)))
    assert x26.shape == (len(members),26) and np.isfinite(x26).all()
    scores = np.clip(b26.predict(x26),0,1)
    return members, x26, scores


def load_L(model_path, normalization_path):
    model = core.Residual()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    with np.load(normalization_path, allow_pickle=False) as z:mean, std = z['mean'], z['std']
    assert mean.shape == std.shape == (28,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    return model, mean, std


def score_selected(model, x28, baseline, indices, mean, std):
    """Only selected rows have meaningful inputs; original batch shape/order.

    No attention, pooling or batch normalization exists in this row-wise MLP.
    Zero padding therefore has no statistical effect, and preserves the CPU
    GEMM shape used by the sealed full-pool inference (including last batch).
    """
    indices = np.asarray(indices, dtype=np.int64)
    assert len(np.unique(indices))==len(indices) and ((indices>=0)&(indices<len(x28))).all()
    delta = np.empty(len(indices),dtype=np.float64)
    with torch.no_grad():
        for start in range(0,len(x28),8192):
            mask = (indices>=start)&(indices<start+8192)
            if not mask.any():continue
            padded = np.zeros((min(8192,len(x28)-start),28),dtype=np.float32)
            positions = indices[mask]-start
            padded[positions] = ((x28[indices[mask]]-mean)/std).astype(np.float32)
            delta[mask] = model(torch.from_numpy(padded)).numpy()[positions]
    return core.adjusted(baseline[indices], delta), delta


def rescore_pool(members, x26, baseline, model, mean, std, budget=800):
    if len(members)<budget:raise ValueError('pool smaller than the registered S800 budget')
    selected = source.rank(members,baseline)[:budget]
    lookup = {source.ctl.shared.key(m):i for i,m in enumerate(members)}
    indices = np.asarray([lookup[source.ctl.shared.key(m)] for m in selected])
    x28 = np.column_stack((x26, baseline, core.base_logits(baseline))).astype(np.float32)
    scores, delta = score_selected(model,x28,baseline,indices,mean,std)
    rows = source.rank([{**m,'score':float(v)} for m,v in zip(selected,scores)])
    return rows, source.ctl.collapse_members(rows), indices, delta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','points','evidence','b26','l-model','normalization','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--point-count',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    # Standalone manifest binds the exact supplied inference inputs, never GT.
    paths={n:getattr(a,n.replace('-','_')) for n in ('geometry','points','evidence','b26','l-model','normalization')}
    binding={n:paths[n] for n in ('geometry','points','evidence')};binding['point_count']=a.point_count
    members,x26,scores=build_pool(binding,joblib.load(a.b26))
    rows,ca,_,_=rescore_pool(members,x26,scores,*load_L(a.l_model,a.normalization))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    freeze_json(out/'semantic_SL.json',rows);freeze_json(out/'class_agnostic_SL.json',ca)
    freeze_json(out/'receipt.json',dict(complete=True,inputs={str(q):sha256_file(q) for q in paths.values()},
        outputs={str(out/n):sha256_file(out/n) for n in ('semantic_SL.json','class_agnostic_SL.json')},
        full_pool_members=len(members),context='full symmetric S pool',selected=800,GT_used=False))


if __name__=='__main__':main()

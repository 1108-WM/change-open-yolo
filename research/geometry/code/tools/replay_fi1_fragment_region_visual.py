#!/usr/bin/env python3
"""GT-free GPU evidence replay, one preregistered first scene in each fold."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from tools import run_fi1_fragment_affiliation_learnability as run
from tools import fi1_fragment_region_visual_evidence as visual
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha

def main():
    c=run.diag.read(run.OUT/'contract.json');parent=run.diag.read(run.SOURCE/'contract.json');assert run.diag.read(run.OUT/'independent_verified.json')['complete'];sys.addaudithook(run.diag.Guard(parent).hook)
    cc=dict(c);cc['version']=c['version']+'/visual_replay';model=visual.load_model(c);records=[]
    scenes=[next(s for s in c['scenes'] if c['folds'][s]==fold) for fold in range(5)]
    for s in scenes:
        print('GPU_replay',s,flush=True);_,pts=run.diag.load_nt(parent['jobs'][s]);raw=np.asarray(np.load(parent['jobs'][s]['paths']['processed'],mmap_mode='r')[:,:9]);pairs=[r for r in run.diag.read(run.SOURCE/'features'/s/'pairs.json') if r['retrieval']['combined']]
        evidence,meta=visual.visual_scene(cc,parent,s,raw,pts,pairs,model)
        with np.load(run.OUT/'scenes'/s/'features.npz') as z:cached=z['visual'][:,-61:]
        assert np.allclose(np.asarray(evidence),cached,atol=1e-6,rtol=0),np.abs(np.asarray(evidence)-cached).max()
        old=run.diag.read(run.OUT/'scenes'/s/'region_views.json');assert len(old)==len(meta)
        for a,b in zip(old,meta):
            assert len(a['views'])==len(b)
            for av,bv in zip(a['views'],b):
                for key in av:
                    if key=='dino_cosine':assert abs(av[key]-bv[key])<=1e-6
                    else:assert av[key]==bv[key]
        with np.load(run.OUT/'scenes'/s/'region_embeddings.npz') as z:oldemb=z['embeddings']
        with np.load(run.OUT/'visual_replay/scenes'/s/'region_embeddings.npz') as z:newemb=z['embeddings']
        assert np.allclose(oldemb.astype(float),newemb.astype(float),atol=.001,rtol=0)
        records.append(dict(scene=s,fold=c['folds'][s],pairs=len(meta),views=sum(len(r) for r in meta),max_feature_difference=float(np.abs(np.asarray(evidence)-cached).max(initial=0))))
    freeze_json(run.OUT/'visual_replay_verified.json',dict(complete=True,GT_read=False,model_checkpoint_sha256=c['model_sha256'],scenes=records,source_sha256=sha(__file__),own_semantic_or_instance_AP=False))
    print('visual_replay_verified',records,flush=True)

if __name__=='__main__':
    with threadpool_limits(4):main()

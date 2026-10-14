#!/usr/bin/env python3
"""Post-seal witnesses only: which whole known-other blocks were missed?"""
from pathlib import Path
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_deletion_blocks as run


def diagnose():
    c=run.read(run.OUT/'contract.json');assert run.read(run.OUT/'completion_verified.json')['complete']
    with (run.OUT/'evaluation/candidates.csv').open() as f:base={(r['scene'],r['candidate_id']):r for r in csv.DictReader(f) if r['arm']=='B_raw'}
    witnesses=[]
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);fold=c['folds'][s];meta={r['candidate_id']:r for r in run.read(run.OUT/'labels'/s/'meta.json')}
        with np.load(run.OUT/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz') as raw,np.load(run.OUT/'predictions'/s/'predictions.npz') as prediction,np.load(run.SOURCE/'features'/s/'data.npz') as z:
            for r in run.read(run.SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];b=base[s,cid]
                if not b['target']:continue
                gid=int(b['target']);parent=z[cid+'_parent'];halo=z[cid+'_halo'];blocks=run.unpack(raw,cid,'_B');pure=[bool((gt[halo[block]]>=1000).all() and (gt[halo[block]]!=gid).all()) for block in blocks]
                ideal=run.core.delete(parent,halo,blocks,pure);hit=int((gt[ideal]==gid).sum());total=int((gt==gid).sum());score=hit/(len(ideal)+total-hit);baseline=float(b['original_NT_best'])
                if score-baseline<.02-1e-12:continue
                pureindices=np.flatnonzero(pure);actions=[]
                for kind in ('summary','relation'):
                    pr=prediction[cid+'_B_'+kind+'_block_prediction'];selected=pr[:,0].astype(np.float32)>=.9
                    actual=prediction[cid+'_B_'+kind+'_points'];ah=int((gt[actual]==gid).sum());aiou=ah/(len(actual)+total-ah)
                    actions.append(dict(kind=kind,actual_iou=aiou,actual_pool_delta=aiou-baseline,selected_pure_blocks=int(selected[pureindices].sum()),pure_blocks=len(pureindices),
                                        all_pure_block_probabilities=[dict(block=int(k),points=len(blocks[k]),p_safe=float(pr[k,0]),selected=bool(selected[k])) for k in pureindices],
                                        selected_target_points=int((gt[np.setdiff1d(parent,actual)]==gid).sum())))
                witnesses.append(dict(scene=s,candidate_id=cid,source=r['source'],target=gid,stable=meta[cid]['stable'],parent_iou=float(b['parent_iou']),NT_best=baseline,oracle_iou=score,oracle_gain=score-baseline,pure_deleted=len(parent)-len(ideal),actions=actions))
    targets={(r['scene'],r['target']) for r in witnesses};summary=run.read(run.OUT/'summary.json');assert len(targets)==summary['oracle']['B']['known_other_oracle_gain02']
    run.seal(run.OUT/'block_opportunity_diagnosis.json',dict(complete=True,posthoc_GT_only=True,used_for_training_or_action_selection=False,witnesses=witnesses,targets=len(targets),code={str(Path(__file__).resolve()):run.sha(__file__)}))
    print('GT-only opportunity witnesses',len(witnesses),'parents',len(targets),'targets',flush=True)


if __name__=='__main__':diagnose()

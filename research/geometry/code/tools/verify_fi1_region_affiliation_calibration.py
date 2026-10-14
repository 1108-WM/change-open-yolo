#!/usr/bin/env python3
"""Recount every rejected calibration union independently; no new rules."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_region_affiliation as run
from tools.verify_fi1_region_affiliation import replay,scores


def verify():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c,'calibrate').hook);examples=[]
    for s in c['splits']['calibration']:
        _,meta,gt,ids,sizes=run.supervision(s,c);meta={r['candidate_id']:r for r in meta}
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank:
            prob={a:np.load(run.OUT/'probabilities'/a/s/'data.npz') for a in run.ARMS}
            for row in run.read(run.OUT/'features'/s/'meta.json'):
                cid=row['candidate_id'];blocks=run.prior.old.unpack(bank,cid)
                examples.append(dict(scene=s,candidate_id=cid,parent=z[cid+'_parent'],halo=z[cid+'_halo'],blocks=[blocks[i] for i in bank[cid+'_budget96']],probability={a:prob[a][cid+'_probability'] for a in run.ARMS},meta=meta[cid],gt=gt,ids=ids,sizes=sizes))
            for p in prob.values():p.close()
    counts=Counter();scopes={};allhashes={}
    for arm in run.ARMS:
        path=run.OUT/'calibration'/f'{arm}.json';grid=run.read(path)['grid'];allhashes[str(path)]=run.sha(path)
        for row in grid:
            total=Counter();changedscenes=set()
            for e in examples:
                selected=any(p[3]>=row['rule']['safe'] and p[0]<=row['rule']['target'] and p[2]<=row['rule']['unknown'] for p in e['probability'][arm])
                q=replay(e['parent'],e['halo'],e['blocks'],e['probability'][arm],row['rule']) if selected else e['parent']
                removed=np.setdiff1d(e['parent'],q);total['deleted']+=len(removed)
                if not len(removed):continue
                changedscenes.add(e['scene']);total['actions']+=1;m=e['meta']
                if not m['stable']:total['unknown']+=len(removed);total['unknown_identity_actions']+=1;continue
                gid=m['target'];values=Counter(e['gt'][removed].tolist());target=values[gid];other=sum(n for g,n in values.items() if g>=1000 and g!=gid);unknown=len(removed)-target-other
                total.update(target=target,other=other,unknown=unknown,parent_target=int((e['gt'][e['parent']]==gid).sum()))
                after=scores(q,e['gt'],e['ids'],e['sizes']);k=int(np.flatnonzero(e['ids']==gid)[0]);total['good_degraded']+=m['parent_iou']>=.9 and after[k]-m['parent_iou']<-.02-1e-12
                best=int(np.argmax(after));total['drift']+=e['ids'][best]!=gid and (e['gt'][q]==e['ids'][best]).sum()>=20
            total['good_parents']=sum(e['meta']['good'] for e in examples);total['action_scenes']=len(changedscenes)
            expected=row['metrics']
            for k,value in total.items():assert expected.get(k,0)==value,(arm,k,value,expected.get(k))
            assert np.isclose(expected['precision'],total['other']/total['deleted'])
            assert np.isclose(expected['target_loss'],total['target']/max(1,total['parent_target']))
            assert np.isclose(expected['unknown_fraction'],total['unknown']/total['deleted'])
            assert row['qualified']==False;counts['grid_rules_replayed']+=1;counts['candidate_union_checks']+=len(examples)
        # Separate unknown identity scope only for the reported rejected rule.
        best=max(grid,key=lambda r:r['metrics']['precision']);scopes[arm]=Counter()
        for e in examples:
            q=replay(e['parent'],e['halo'],e['blocks'],e['probability'][arm],best['rule']);n=len(e['parent'])-len(q)
            if not e['meta']['stable']:scopes[arm][e['meta']['scope']]+=n
    run.seal(run.OUT/'calibration_independent_verified.json',dict(complete=True,counts=dict(counts),highest_precision_rejected_unknown_scopes={a:dict(v) for a,v in scopes.items()},inputs=allhashes,verification_code_sha256=run.sha(Path(__file__)),rules_unchanged_sha256=run.sha(run.OUT/'rules_frozen.json'),holdout_GT_read=False,no_new_rules=True));print('calibration_independent_verified',dict(counts),{a:dict(v) for a,v in scopes.items()},flush=True)


if __name__=='__main__':verify()

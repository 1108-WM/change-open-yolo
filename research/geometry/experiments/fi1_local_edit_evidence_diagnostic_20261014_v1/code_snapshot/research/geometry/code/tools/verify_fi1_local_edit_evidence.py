#!/usr/bin/env python3
"""Independent proposal identity/query/GT edit/pool replay; no learning."""
from pathlib import Path
import ast,csv,sys,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_local_edit_evidence as run
from verify_fi1_nt_region_identity import vector

def verify():
    c=run.read(run.OUT/'contract.json');run.diag.verify(c['code']);run.diag.verify(c['inputs']);run.diag.verify(c['GT_inputs']);run.diag.verify(c['frozen_SL_models']);run.diag.verify(run.read(run.OUT/'proposals_frozen.json')['outputs']);run.diag.verify(run.read(run.OUT/'evaluation_frozen.json')['outputs'])
    assert (run.OUT/'proposals_frozen.json').stat().st_mtime<=(run.OUT/'evaluation_frozen.json').stat().st_mtime
    rows=run.read(run.OUT/'evaluation/rows.json');lookup={(r['scene'],r['proposal_id']):r for r in rows};pool=list(csv.DictReader((run.OUT/'evaluation/pool.csv').open()));pl={(r['scene'],r['mode'],int(r['gt_id'])):r for r in pool};n=targets=0
    for s,j in c['jobs'].items():
        root=Path(j['asset_root']);gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok];_,nt=run.diag.load_nt(j);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,vector(set(map(int,p)),gt,ids,sizes))
        best={};unions={};parents={};evidence=run.read(run.OUT/'proposals'/s/'evidence.json')
        with np.load(run.OUT/'proposals'/s/'edits.npz') as edits,np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as prior:
            for e in evidence:
                cid=e['candidate_id'];own=set(map(int,st[cid+'_parent']));parents[cid]=own;pp=edits[e['proposal_id']];pts=set(map(int,pp));assert len(pts)==e['points'];assert hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest()==e['point_ids_sha256']
                halo=st[cid+'_halo'];domain={int(p) for p,l,g in zip(halo,st[cid+'_labels'],st[cid+'_visible']) if l==e['region'] and g>=2};assert pts<=domain
                if e['family']=='region':assert pts==domain
                assert (pts<=own if e['action']=='delete' else not pts&own)
                ix=np.searchsorted(prior['point_ids'],pp);p=prior['P1_probability'][ix];q=e['query_id'];assert abs(float(p[:,q].mean())-e['query_mean'])<1e-7
                cm=p.mean(0,dtype=np.float64);cm[q]=-1;assert int(np.lexsort((np.arange(len(cm)),-cm))[0])==e['competitor_id']
                actual=own|pts if e['action']=='add' else own-pts
                if len(actual)<100:actual=own.copy()
                before=vector(own,gt,ids,sizes);after=vector(actual,gt,ids,sizes);idx=int(before.argmax()) if len(ids) and before.max()>0 else None;g=int(ids[idx]) if idx is not None else None;hits=sum(gt[p]==g for p in own) if g is not None else 0;stable=g is not None and hits>=20 and hits/len(own)>=.6;row=lookup[s,e['proposal_id']];assert stable==row['stable']
                if stable:
                    correct=sum((gt[p]==g if e['action']=='add' else gt[p]>=1000 and gt[p]!=g) for p in pts);wrong=sum((gt[p]>=1000 and gt[p]!=g if e['action']=='add' else gt[p]==g) for p in pts);unknown=sum(gt[p]<1000 for p in pts)
                    assert (correct,wrong,unknown)==(row['affected_correct'],row['affected_wrong'],row['affected_unknown']);assert abs(row['repaired_iou']-after[idx])<1e-12;assert abs(row['parent_iou']-before[idx])<1e-12
                for screen,keep in run.core.screens(e).items():
                    assert e['screen_'+screen]==keep
                    if not keep:continue
                    mode=f'{e["family"]}/{e["source"]}/{e["action"]}/{screen}';best.setdefault(mode,baseline.copy());best[mode]=np.maximum(best[mode],after);unions.setdefault((cid,e['family'],screen,e['action'],e['source']),set()).update(pts)
                n+=1
            for (cid,family,screen,action,source),pts in unions.items():
                p=parents[cid]|pts if action=='add' else parents[cid]-pts
                if len(p)<100:p=parents[cid].copy()
                mode=f'{family}/{source}/{action}/{screen}/union';after=vector(p,gt,ids,sizes);best.setdefault(mode,baseline.copy());best[mode]=np.maximum(best[mode],after)
        for mode,v in best.items():
            for i,g in enumerate(ids):
                r=pl[s,mode,int(g)];assert abs(float(r['baseline'])-baseline[i])<1e-12 and abs(float(r['expanded'])-v[i])<1e-12;assert ast.literal_eval(r['gain02'])==bool(v[i]-baseline[i]>=.02-1e-12);targets+=1
        print('verified',s,n,targets,flush=True)
    run.seal(run.OUT/'verification.json',dict(complete=True,proposals=n,pool_rows=targets,independent_set_actions_and_IoU=True,source_and_SL_unchanged=True,GT_after_frozen_evidence=True,training=False,production_modified=False,time=time.time()))

if __name__=='__main__':verify()

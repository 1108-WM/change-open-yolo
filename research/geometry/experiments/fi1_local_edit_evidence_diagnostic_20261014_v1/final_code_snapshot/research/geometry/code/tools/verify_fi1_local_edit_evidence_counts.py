#!/usr/bin/env python3
"""Supplementary independent count replay; avoids repeated Python-set scans.

Added after evidence/evaluation freeze. Does not replace the sealed verifier.
GT histograms and candidate IoUs are computed independently of evaluator code.
"""
from pathlib import Path
import ast,csv,hashlib,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_local_edit_evidence as run

def counts(points,gt):
    return np.bincount(gt[points],minlength=int(gt.max())+1)

def iou(hist,total,ids,sizes):
    if total<100:return np.zeros(len(ids))
    hit=hist[ids]
    return hit/(total+sizes-hit)

def verify():
    c=run.read(run.OUT/'contract.json')
    for key in ('code','inputs','GT_inputs','frozen_SL_models'):run.diag.verify(c[key])
    for file in ('proposals_frozen.json','evaluation_frozen.json'):run.diag.verify(run.read(run.OUT/file)['outputs'])
    assert (run.OUT/'proposals_frozen.json').stat().st_mtime<=(run.OUT/'evaluation_frozen.json').stat().st_mtime
    rows=run.read(run.OUT/'evaluation/rows.json');lookup={(r['scene'],r['proposal_id']):r for r in rows}
    pl={(r['scene'],r['mode'],int(r['gt_id'])):r for r in csv.DictReader((run.OUT/'evaluation/pool.csv').open())}
    n=targets=0
    for s,j in c['jobs'].items():
        root=Path(j['asset_root']);gt=np.loadtxt(j['GT'],dtype=np.int64)
        ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok]
        _,nt=run.diag.load_nt(j);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(counts(p,gt),len(p),ids,sizes))
        best={};unions={};parents={};evidence=run.read(run.OUT/'proposals'/s/'evidence.json');domains={}
        with np.load(run.OUT/'proposals'/s/'edits.npz') as edits,np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as prior:
            prior_points=prior['point_ids'];prior_probability=prior['P1_probability'];structure={}
            for e in evidence:
                cid=e['candidate_id']
                if cid not in parents:
                    pp=st[cid+'_parent'];hist=counts(pp,gt);before=iou(hist,len(pp),ids,sizes)
                    structure[cid]=(st[cid+'_halo'],st[cid+'_labels'],st[cid+'_visible'])
                    idx=int(before.argmax()) if len(ids) and before.max()>0 else None;g=int(ids[idx]) if idx is not None else None
                    stable=g is not None and hist[g]>=20 and hist[g]/len(pp)>=.6
                    parents[cid]=(set(map(int,pp)),hist,before,idx,g,stable)
                own,hist,before,idx,g,stable=parents[cid];pp=edits[e['proposal_id']];pts=set(map(int,pp))
                assert len(pts)==e['points'];assert hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest()==e['point_ids_sha256']
                dk=(cid,e['region'])
                if dk not in domains:
                    halo,labels,visible=structure[cid]
                    domains[dk]=set(map(int,halo[(labels==e['region'])&(visible>=2)]))
                assert pts<=domains[dk]
                if e['family']=='region':assert pts==domains[dk]
                assert (pts<=own if e['action']=='delete' else not pts&own)
                ix=np.searchsorted(prior_points,pp);p=prior_probability[ix];q=e['query_id']
                assert abs(float(p[:,q].mean())-e['query_mean'])<1e-7
                cm=p.mean(0,dtype=np.float64);cm[q]=-1;assert int(np.lexsort((np.arange(len(cm)),-cm))[0])==e['competitor_id']
                sign=1 if e['action']=='add' else -1;total=len(own)+sign*len(pp)
                after=before if total<100 else iou(hist+sign*counts(pp,gt),total,ids,sizes)
                row=lookup[s,e['proposal_id']];assert stable==row['stable']
                if stable:
                    target=gt[pp]==g;other=(gt[pp]>=1000)&~target
                    correct=int((target if sign==1 else other).sum());wrong=int((other if sign==1 else target).sum());unknown=int((gt[pp]<1000).sum())
                    assert (correct,wrong,unknown)==(row['affected_correct'],row['affected_wrong'],row['affected_unknown'])
                    assert abs(row['repaired_iou']-after[idx])<1e-12 and abs(row['parent_iou']-before[idx])<1e-12
                for screen,keep in run.core.screens(e).items():
                    assert e['screen_'+screen]==keep
                    if not keep:continue
                    mode=f'{e["family"]}/{e["source"]}/{e["action"]}/{screen}'
                    best.setdefault(mode,baseline.copy());best[mode]=np.maximum(best[mode],after)
                    unions.setdefault((cid,e['family'],screen,e['action'],e['source']),set()).update(pts)
                n+=1
            for (cid,family,screen,action,source),pts in unions.items():
                own,hist,before,_,_,_=parents[cid];sign=1 if action=='add' else -1;total=len(own)+sign*len(pts)
                after=before if total<100 else iou(hist+sign*counts(np.array(sorted(pts)),gt),total,ids,sizes)
                mode=f'{family}/{source}/{action}/{screen}/union';best.setdefault(mode,baseline.copy());best[mode]=np.maximum(best[mode],after)
        for mode,v in best.items():
            for i,g in enumerate(ids):
                r=pl[s,mode,int(g)];assert abs(float(r['baseline'])-baseline[i])<1e-12 and abs(float(r['expanded'])-v[i])<1e-12
                assert ast.literal_eval(r['gain02'])==bool(v[i]-baseline[i]>=.02-1e-12);targets+=1
        print('verified',s,n,targets,flush=True)
    run.seal(run.OUT/'verification_counts.json',dict(complete=True,proposals=n,pool_rows=targets,independent_GT_histograms_and_IoU=True,source_and_SL_unchanged=True,GT_after_frozen_evidence=True,training=False,production_modified=False,screen_definition_reused=True,script_sha256=run.diag.sha(Path(__file__)),time=time.time()))

if __name__=='__main__':verify()

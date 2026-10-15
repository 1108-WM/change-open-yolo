#!/usr/bin/env python3
"""Independent rule/block/union and all-GT complete-NT pool replay."""
from pathlib import Path
import ast,csv,hashlib,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_t_add_union as run

def hist(points,gt):return np.bincount(gt[points],minlength=int(gt.max())+1)
def iou(h,n,ids,sizes):
    if n<100:return np.zeros(len(ids))
    x=h[ids];return x/(n+sizes-x)

def verify():
    c=run.read(run.OUT/'contract.json')
    for k in ('code','inputs','GT_inputs','frozen_SL_models'):run.diag.verify(c[k])
    for stage in ('structure','priors','predictions','evaluation'):run.diag.verify(run.read(run.OUT/(stage+'_frozen.json'))['outputs'])
    assert (run.OUT/'predictions_frozen.json').stat().st_mtime<=(run.OUT/'evaluation_frozen.json').stat().st_mtime
    physical=lambda ss:{s.split('_')[0] for s in ss}
    assert len(c['scenes'])==len(physical(c['scenes']))==24 and not physical(c['scenes'])&set(c['excluded_physical_scenes'])
    rows=run.read(run.OUT/'evaluation/rows.json');rl={(r['scene'],r['candidate_id'],r['arm']):r for r in rows}
    pools=list(csv.DictReader((run.OUT/'evaluation/pool.csv').open()));pl={(r['scene'],r['mode'],int(r['gt_id'])):r for r in pools};nw=0;nb=0;nr=0;npool=0
    for s,j in c['jobs'].items():
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok]
        _,nt=run.diag.load_nt(j);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(hist(p,gt),len(p),ids,sizes))
        best={a:baseline.copy() for a in ('A','B')};single={a:baseline.copy() for a in ('A','B')};blocks=run.read(run.OUT/'predictions'/s/'blocks.json');bl={(r['candidate_id'],r['region']):r for r in blocks};unions=run.read(run.OUT/'predictions'/s/'unions.json');ul={(r['candidate_id'],r['arm']):r for r in unions}
        with np.load(run.OUT/'structure'/s/'data.npz') as st,np.load(run.OUT/'priors'/s/'data.npz') as pr,np.load(run.OUT/'predictions'/s/'data.npz') as pred:
            roi=pr['point_ids'];prob=pr['P1_probability'];volume=pr['P1_full_volume'];used=set()
            for meta in c['parents'][s]:
                assert meta['source']=='track';cid=meta['candidate_id'];parent=st[cid+'_parent'];halo=st[cid+'_halo'];labels=st[cid+'_labels'];visible=st[cid+'_visible'];member=np.isin(halo,parent);ix=np.searchsorted(roi,halo);p=prob[ix];ownp=p[member].sum(0,dtype=np.float64);scores=ownp/(len(parent)+volume-ownp);q=int(np.lexsort((np.arange(len(scores)),-scores))[0]);selected={'A':set(),'B':set()};bid={'A':[],'B':[]};h=hist(parent,gt);before=iou(h,len(parent),ids,sizes);idx=int(before.argmax()) if len(ids) and before.max()>0 else None;g=int(ids[idx]) if idx is not None else None;stable=g is not None and h[g]>=20 and h[g]/len(parent)>=.6
                for rid in range(int(labels.max())+1):
                    domain=np.flatnonzero((labels==rid)&~member&(visible>=2))
                    if not len(domain):continue
                    b=bl[cid,rid];used.add((cid,rid));pp=pred[b['block_id']+'_block'];assert np.array_equal(pp,halo[domain]);assert b['query_id']==q
                    means=np.array([float(p[domain,k].astype(np.float64).sum()/len(domain)) for k in range(p.shape[1])]);own=means[q];means[q]=-1;comp=int(np.lexsort((np.arange(len(means)),-means))[0]);margin=own-means[comp]
                    assert b['competitor_id']==comp and abs(b['query_mean']-own)<1e-12 and abs(b['margin']-margin)<1e-12
                    decisions={'A':bool(own>=.5),'B':bool(own>=.5 and margin>=.1)}
                    after=iou(h+hist(pp,gt),len(parent)+len(pp),ids,sizes)
                    for arm,accept in decisions.items():
                        assert b[arm]==accept
                        if accept:selected[arm].update(map(int,pp));bid[arm].append(b['block_id']);single[arm]=np.maximum(single[arm],after)
                    nb+=1
                for arm in ('A','B'):
                    u=ul[cid,arm];pp=pred[cid+'_'+arm+'_points'];add=pred[cid+'_'+arm+'_added'];actual=set(map(int,parent))|selected[arm];assert set(map(int,pp))==actual and set(map(int,add))==selected[arm];assert u['blocks']==bid[arm] and u['extra_candidate']==bool(selected[arm]);assert hashlib.sha256(pp.astype('<i8').tobytes()).hexdigest()==u['points_sha256']
                    after=iou(hist(pp,gt),len(pp),ids,sizes);r=rl[s,cid,arm];assert r['stable']==stable and r['deleted']==0 and r['added']==len(add)
                    assert r['unknown_annotation_added']==int((gt[add]<1000).sum())
                    if stable:
                        assert r['target']==g and r['added_target']==int((gt[add]==g).sum()) and r['added_other']==int(((gt[add]>=1000)&(gt[add]!=g)).sum())
                        assert abs(r['parent_iou']-before[idx])<1e-12 and abs(r['repaired_iou']-after[idx])<1e-12
                    if len(add):best[arm]=np.maximum(best[arm],after)
                    nr+=1
                assert selected['B']<=selected['A']
            assert used==set(bl)
        for mode,v in [('NT',baseline)]+[(a,best[a]) for a in ('A','B')]+[(a+'_single_diagnostic',single[a]) for a in ('A','B')]:
            for k,g in enumerate(ids):
                r=pl[s,mode,int(g)];assert abs(float(r['baseline'])-baseline[k])<1e-12 and abs(float(r['expanded'])-v[k])<1e-12;assert ast.literal_eval(r['gain02'])==bool(v[k]-baseline[k]>=.02-1e-12);npool+=1
        print('verified',s,'blocks',nb,'unions',nr,'pool',npool,flush=True)
    run.seal(run.OUT/'verification.json',dict(complete=True,blocks=nb,unions=nr,pool_rows=npool,independent_rule_selection_and_union_and_GT_IoU=True,original_NT_preserved=True,N_edits=0,T_deletions=0,GT_after_predictions=True,code_inputs_SL_unchanged=True,physical_isolation_verified=True,training=False,production_modified=False,time=time.time()))

if __name__=='__main__':verify()

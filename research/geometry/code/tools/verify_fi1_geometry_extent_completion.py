#!/usr/bin/env python3
"""Independent point-count reconstruction of diagnostic and sealed new extents."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_witnessed_local_completion as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
from tools.fi1_r0_r1_protocol import original_aliases
from tools.fi1_witnessed_local_completion import witnessed_addition
read=run.read;diag=run.diag

def independent(points,gt,ids):
    sizes=np.array([np.sum(gt==i) for i in ids]);inter=np.array([[np.sum(gt[p]==i) for i in ids] for p in points]);n=np.array([len(p) for p in points])
    io=inter/(n[:,None]+sizes[None,:]-inter);io[n<100]=0
    return io,inter,n,sizes

def main():
    c=read(run.OUT/'contract.json');records=[json.loads(x) for x in (diag.OUT/'targets.jsonl').read_text().splitlines()];counts=Counter();new=Counter();observ=Counter();small_best=[];quality=[]
    run.diag.prior.src.ctl.base.verify_files(read(run.OUT/'geometry_frozen.json')['outputs'])
    for i,s in enumerate(c['scenes']):
        print('verify',i,s,flush=True);gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);assert sha(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']]
        rr=[r for r in records if r['scene']==s];ids=np.array([r['gt_id'] for r in rr]);gs,pts,_=diag.prior.src.ctl.base.load_geometry(c['jobs'][s]['binding']);nt=run.source_pool(gs);keys=[g['geometry_key'] for g in nt];ix={k:j for j,k in enumerate(keys)}
        io,inter,n,nn=independent([pts[k] for k in keys],gt,ids)
        for j,r in enumerate(rr):
            best=sorted(range(len(keys)),key=lambda k:(-io[k,j],keys[k]))[0];assert keys[best]==r['best_geometry_key'] and abs(io[best,j]-r['best_iou'])<1e-14
            if n[best]<100:small_best.append(dict(scene=s,gt_id=r['gt_id'],points=int(n[best]),physical_intersection=int(inter[best,j])))
            else:assert abs(inter[best,j]/n[best]-r['best_purity'])<1e-14 and abs(inter[best,j]/nn[j]-r['best_coverage'])<1e-14
            for src,reps in r['representatives'].items():
                for rep in reps:
                    k=ix[rep['geometry_key']];assert any(a['source']==src for a in original_aliases(nt[k]));assert abs(io[k,j]-rep['iou'])<1e-14
                    if n[k]>=100:assert abs(inter[k,j]/n[k]-rep['purity'])<1e-14 and abs(inter[k,j]/nn[j]-rep['coverage'])<1e-14
            counts['targets']+=1;counts['gap050']+=r['best_iou']<=c['thresholds'][0];counts['gap075']+=r['missing075'];counts['gap090']+=r['best_iou']<=c['thresholds'][2]
            if r['missing075']:
                counts[r['kind']]+=1;pair=r['NT_top12_pair_union_oracle']
                if pair:
                    p=np.union1d(pts[pair['left']],pts[pair['right']]);a=int(np.sum(gt[p]==r['gt_id']));assert len(p)==pair['points'] and a==pair['intersection'];assert abs(a/(len(p)+nn[j]-a)-pair['iou'])<1e-14
                    counts['pair_repair075']+=pair['iou']>c['thresholds'][1]
                observ['visible_half']+=r['visible_in30_fraction']>=.5;observ['rawmask_half']+=r['any_rawmask30_target_coverage']>=.5;observ['no_rawmask']+=r['any_rawmask30_target_coverage']==0;observ['no_visibility']+=r['visible_in30_fraction']==0
        rows=read(run.OUT/'geometry'/s/'geometry.json');dec=read(run.OUT/'geometry'/s/'decisions.json');decindex={(r['parent'],r['donor']):r for r in dec if r['accepted']}
        with np.load(run.OUT/'geometry'/s/'points.npz') as z:added=[z[r['array_key']] for r in rows]
        if rows:
            fs,bound=diag.load_frames(c,s);raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');xyz=np.asarray(raw[:,:3]).copy();del raw
            for r,p in zip(rows,added):
                assert hashlib.sha256(p.astype('<i8').tobytes()).hexdigest()==r['points_sha256'];assert r['geometry_key'] not in keys and np.isin(pts[r['parent']],p).all()
                expected,receipt=witnessed_addition(pts[r['parent']],pts[r['donor']],fs,xyz,c['parameters']);assert np.array_equal(expected,p) and receipt['accepted'];assert receipt['joint_groups']==r['joint_groups'] and receipt['added_points']==r['added_points']
                decisions=[d for d in dec if d['accepted'] and d['parent']==r['parent']];best=max(len(d['joint_groups']) for d in decisions);assert len(r['joint_groups'])==best
                counts['action_replayed']+=1
            ni,_,_,_=independent(added,gt,ids);nb=ni.max(axis=0,initial=0);before=io.max(axis=0,initial=0)
            for t in c['thresholds']:new[format(t,'.2f')]+=int(np.sum((before<=t)&(nb>t)))
            for k,r in enumerate(rows):
                gid=int(ids[ni[k].argmax()]);p=added[k];par=pts[r['parent']];inc=np.setdiff1d(p,par);q=int(np.sum(gt[p]==gid));old=int(np.sum(gt[par]==gid));extra=int(np.sum(gt[inc]==gid));size=int(np.sum(gt==gid))
                quality.append(dict(scene=s,geometry_key=r['geometry_key'],diagnostic_best_gt=gid,iou=float(ni[k].max()),purity=q/len(p),coverage=q/size,parent_purity=old/len(par),parent_coverage=old/size,added_target_points=extra,added_pollution_points=len(inc)-extra))
    assert counts['targets']==4689 and counts['gap075']==877 and counts['action_replayed']==201
    expected=read(run.OUT/'geometry_quality.json')['groups']['development200'];assert all(new[t]==expected[t]['new_over_complete_NT'] for t in new)
    freeze_json(run.OUT/'extent_quality_supplement.json',dict(complete=True,rows=quality,counts=dict(counts),observations=dict(observ),small_best=small_best))
    freeze_json(run.OUT/'independent_geometry_verified.json',dict(complete=True,source_sha256=sha(__file__),counts=dict(counts),new_geometry_gain=dict(new),diagnosis_targets_sha256=sha(diag.OUT/'targets.jsonl'),geometry_seal_sha256=sha(run.OUT/'geometry_frozen.json'),observations=dict(observ),small_best=small_best,GT_diagnostic_only=True,actions_replayed_without_GT_inputs=True,production_modified=False))

if __name__=='__main__':main()

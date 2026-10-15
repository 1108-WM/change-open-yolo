#!/usr/bin/env python3
"""Post-evaluation redundancy/supply audit. Never changes frozen A/B outputs."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_t_add_union as run
from verify_fi1_t_add_union import hist,iou

def analysis():
    c=run.read(run.OUT/'contract.json');run.diag.verify(run.read(run.OUT/'evaluation_frozen.json')['outputs']);d=run.OUT/'failure_diagnosis';d.mkdir(exist_ok=True)
    rows=run.read(run.OUT/'evaluation/rows.json');lookup={(r['scene'],r['candidate_id'],r['arm']):r for r in rows};redundancy=[];supply=[];discard=[];origin=[];scope=[];totals={};mapping=0
    for s,j in c['jobs'].items():
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok]
        geometry,nt=run.diag.load_nt(j);baseN=np.zeros(len(ids));baseT=baseN.copy()
        for g in geometry:
            sources={a['source'] for a in run.diag.original_aliases(g)};v=iou(hist(nt[g['geometry_key']],gt),len(nt[g['geometry_key']]),ids,sizes)
            if 'native' in sources:baseN=np.maximum(baseN,v)
            if 'track' in sources:baseT=np.maximum(baseT,v)
        baseline=np.maximum(baseN,baseT);bestpoint=baseline.copy();bestregion=baseline.copy()
        blocks=run.read(run.OUT/'predictions'/s/'blocks.json')
        with np.load(run.OUT/'structure'/s/'data.npz') as st,np.load(run.OUT/'predictions'/s/'data.npz') as pred,np.load(run.OUT/'priors'/s/'data.npz') as prior:
            raw=np.load(j['paths']['processed'],mmap_mode='r');coordinates=np.floor(np.asarray(raw[:,:3],np.float32)/.02).astype(np.int32);sparse=prior['input_sparse_coordinates'];mapped=prior['original_to_sparse_row'];assert np.array_equal(sparse[mapped,1:],coordinates);mapping+=len(raw)
            for meta in c['parents'][s]:
                cid=meta['candidate_id'];r=lookup[s,cid,'A'];parent=st[cid+'_parent'];aa=pred[cid+'_A_added'];bb=pred[cid+'_B_added'];dropped=np.setdiff1d(aa,bb)
                scope.append(dict(scene=s,candidate_id=cid,scope=r['scope'],stable=r['stable'],parent_points=len(parent),A_added=len(aa),B_added=len(bb),parent_iou=r['parent_iou']))
                known=(gt[dropped]>=1000);dr=dict(scene=s,candidate_id=cid,stable=r['stable'],dropped=len(dropped),unannotated=int((~known).sum()),unknown_identity=len(dropped) if not r['stable'] else 0)
                if r['stable']:dr.update(target=int((gt[dropped]==r['target']).sum()),other=int((known&(gt[dropped]!=r['target'])).sum()))
                discard.append(dr)
                for arm in ('A','B'):
                    arr=pred[cid+'_'+arm+'_added'];nset=set(map(int,arr));selected=[b for b in blocks if b['candidate_id']==cid and b[arm]]
                    for b in selected:origin.append(dict(b,arm=arm,scope=r['scope'],stable=r['stable']))
                if not r['stable']:continue
                idx=int(np.flatnonzero(ids==r['target'])[0]);g=r['target'];parenthist=hist(parent,gt)
                for arm in ('A','B'):
                    rr=lookup[s,cid,arm];after=rr['repaired_iou'];redundancy.append(dict(scene=s,candidate_id=cid,arm=arm,target=g,parent_iou=r['parent_iou'],repaired_iou=after,parent_delta=rr['parent_delta'],original_N_best=float(baseN[idx]),original_T_best=float(baseT[idx]),original_NT_best=float(baseline[idx]),gain_over_N=float(after-baseN[idx]),gain_over_T=float(after-baseT[idx]),gain_over_NT=float(after-baseline[idx]),improved_ge02=rr['parent_delta']>=.02-1e-12,already_covered_N=baseN[idx]>=after-1e-12,already_covered_T=baseT[idx]>=after-1e-12))
                points=[];pure=[];cb=[b for b in blocks if b['candidate_id']==cid]
                for b in cb:
                    p=pred[b['block_id']+'_block'];points.extend(map(int,p[gt[p]==g]))
                    if np.all(gt[p]==g):pure.extend(map(int,p))
                points=np.array(sorted(set(points)),np.int64);pure=np.array(sorted(set(pure)),np.int64)
                pointv=iou(parenthist+hist(points,gt),len(parent)+len(points),ids,sizes);regionv=iou(parenthist+hist(pure,gt),len(parent)+len(pure),ids,sizes)
                # Same-parent add-only optimum over supplied points: every correct
                # point is beneficial, every other/unknown point detrimental.
                # GT construction is diagnostic, never promoted to candidate pool.
                bestpoint[idx]=max(bestpoint[idx],pointv[idx]);bestregion[idx]=max(bestregion[idx],regionv[idx])
                supply.append(dict(scene=s,candidate_id=cid,target=g,parent_iou=r['parent_iou'],original_N_best=float(baseN[idx]),original_T_best=float(baseT[idx]),original_NT_best=float(baseline[idx]),available_target_points=len(points),entirely_target_region_points=len(pure),point_add_only_optimal_iou=float(pointv[idx]),pure_region_union_iou=float(regionv[idx]),point_gain_over_complete_NT=float(pointv[idx]-baseline[idx]),pure_region_gain_over_complete_NT=float(regionv[idx]-baseline[idx]),point_gain02=bool(pointv[idx]-baseline[idx]>=.02-1e-12),pure_region_gain02=bool(regionv[idx]-baseline[idx]>=.02-1e-12),posthoc_GT_only=True))
        totals[s]=dict(point_same_parent_pool_gain02=int((bestpoint-baseline>=.02-1e-12).sum()),pure_region_same_parent_pool_gain02=int((bestregion-baseline>=.02-1e-12).sum()),point_pool_continuous_gain=float((bestpoint-baseline).sum()),pure_region_pool_continuous_gain=float((bestregion-baseline).sum()))
    for name,rr in [('redundancy',redundancy),('supply_opportunity',supply),('B_discarded_edits',discard),('selected_block_origin',origin),('parent_scopes',scope)]:run.met.csv_write(d/(name+'.csv'),rr)
    summary=dict(posthoc=True,GT_constructs_not_predictions=True,original_A_B_outputs_unchanged=True,physical_scene_replacement=False,threshold_changes=False,sparse_original_point_mappings_verified=mapping,point_add_only_optimum_scope='only stable same original parent target, supplied visible region domains, no deletions/new points; not global optimum or executed GT edits',point_pool_gain02_targets=sum(t['point_same_parent_pool_gain02'] for t in totals.values()),point_pool_gain02_scenes=sum(t['point_same_parent_pool_gain02']>0 for t in totals.values()),pure_region_pool_gain02_targets=sum(t['pure_region_same_parent_pool_gain02'] for t in totals.values()),pure_region_pool_gain02_scenes=sum(t['pure_region_same_parent_pool_gain02']>0 for t in totals.values()),point_pool_continuous_gain=sum(t['point_pool_continuous_gain'] for t in totals.values()),scenes=totals)
    for arm in ('A','B'):
        rr=[r for r in redundancy if r['arm']==arm];improved=[r for r in rr if r['improved_ge02']]
        summary[arm]=dict(improved_ge02=len(improved),improved_already_covered_N=sum(r['already_covered_N'] for r in improved),improved_already_covered_T=sum(r['already_covered_T'] for r in improved),improved_covered_either=sum(r['already_covered_N'] or r['already_covered_T'] for r in improved),improved_GT_targets=len({(r['scene'],r['target']) for r in improved}))
    summary['B_discarded']=dict(target=sum(r.get('target',0) for r in discard),other=sum(r.get('other',0) for r in discard),unannotated=sum(r['unannotated'] for r in discard),unknown_identity=sum(r['unknown_identity'] for r in discard),all=sum(r['dropped'] for r in discard))
    run.seal(d/'summary.json',summary);run.seal(run.OUT/'failure_diagnosis_frozen.json',dict(complete=True,posthoc_not_rules=True,outputs={str(p):run.diag.sha(p) for p in d.glob('*')},code_sha256=run.diag.sha(Path(__file__))))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':analysis()

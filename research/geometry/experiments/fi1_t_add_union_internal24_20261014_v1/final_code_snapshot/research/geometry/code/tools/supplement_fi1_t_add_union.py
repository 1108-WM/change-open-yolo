#!/usr/bin/env python3
"""Independent oracle-supply validation and missed blocks, post hoc only."""
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_t_add_union as run
from verify_fi1_nt_region_identity import vector

def main():
    c=run.read(run.OUT/'contract.json');run.diag.verify(run.read(run.OUT/'failure_diagnosis_frozen.json')['outputs']);run.diag.verify(run.read(run.OUT/'effective_runtime_frozen.json')['runtime_code'])
    supply=list(csv.DictReader((run.OUT/'failure_diagnosis/supply_opportunity.csv').open()));sl={(r['scene'],r['candidate_id']):r for r in supply};rows=run.read(run.OUT/'evaluation/rows.json');rl={(r['scene'],r['candidate_id'],r['arm']):r for r in rows};missed=[];verified=0;targets=0
    for s,j in c['jobs'].items():
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok]
        _,nt=run.diag.load_nt(j);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,vector(set(map(int,p)),gt,ids,sizes))
        blocks=run.read(run.OUT/'predictions'/s/'blocks.json');bp=baseline.copy();br=baseline.copy()
        with np.load(run.OUT/'structure'/s/'data.npz') as st,np.load(run.OUT/'predictions'/s/'data.npz') as pred:
            for meta in c['parents'][s]:
                cid=meta['candidate_id'];r=rl[s,cid,'A']
                if not r['stable']:continue
                g=r['target'];k=int(np.flatnonzero(ids==g)[0]);parent=set(map(int,st[cid+'_parent']));correct=set();pure=set();cb=[b for b in blocks if b['candidate_id']==cid]
                for b in cb:
                    pts=set(map(int,pred[b['block_id']+'_block']));good={p for p in pts if gt[p]==g};correct|=good
                    if good==pts:pure|=pts
                pv=vector(parent|correct,gt,ids,sizes);rv=vector(parent|pure,gt,ids,sizes);x=sl[s,cid];assert len(correct)==int(x['available_target_points']) and len(pure)==int(x['entirely_target_region_points']);assert abs(pv[k]-float(x['point_add_only_optimal_iou']))<1e-12 and abs(rv[k]-float(x['pure_region_union_iou']))<1e-12
                bp[k]=max(bp[k],pv[k]);br[k]=max(br[k],rv[k]);verified+=1
                if pv[k]-baseline[k]<.02-1e-12:continue
                for b in cb:
                    pp=pred[b['block_id']+'_block'];good=int((gt[pp]==g).sum());other=int(((gt[pp]>=1000)&(gt[pp]!=g)).sum());unknown=int((gt[pp]<1000).sum())
                    missed.append(dict(b,target=g,target_points=good,other_points=other,unknown_points=unknown,target_fraction=good/len(pp),entirely_target=good==len(pp),point_add_only_optimal_iou=float(pv[k]),pure_region_iou=float(rv[k]),original_NT_best=float(baseline[k]),A_union_iou=rl[s,cid,'A']['repaired_iou'],B_union_iou=rl[s,cid,'B']['repaired_iou'],posthoc_not_rule=True))
            assert run.read(run.OUT/'failure_diagnosis/summary.json')['scenes'][s]['point_same_parent_pool_gain02']==int((bp-baseline>=.02-1e-12).sum());assert run.read(run.OUT/'failure_diagnosis/summary.json')['scenes'][s]['pure_region_same_parent_pool_gain02']==int((br-baseline>=.02-1e-12).sum());targets+=len(ids)
        print('supply_verified',s,verified,flush=True)
    d=run.OUT/'supplement';d.mkdir(exist_ok=True);run.met.csv_write(d/'missed_opportunity_blocks.csv',missed)
    allblocks=[];unions=[]
    for s in c['scenes']:
        allblocks+=run.read(run.OUT/'predictions'/s/'blocks.json');unions+=run.read(run.OUT/'predictions'/s/'unions.json')
    dups={a:sum(r['duplicate_original'] for r in rows if r['arm']==a) for a in ('A','B')};scopes={}
    for r in rows:
        if r['arm']=='A':scopes[r['scope']]=scopes.get(r['scope'],0)+1
    actions=[]
    for arm in ('A','B'):
        rr=[r for r in rows if r['arm']==arm];stable=[r for r in rr if r['stable']];changed=[r for r in stable if r['added']];wrong=sum(r['added_other'] for r in stable);target=sum(r['added_target'] for r in stable);unknown=sum(r['unknown_annotation_added'] for r in stable);ss=[b for b in allblocks if b[arm]]
        actions.append(dict(arm=arm,action_scenes=len({r['scene'] for r in rr if r['added']}),stable_action_scenes=len({r['scene'] for r in changed}),unknown_annotation_fraction_stable=(unknown/(target+wrong+unknown) if target+wrong+unknown else None),known_precision_stable=target/max(1,target+wrong),stable_added_target=target,stable_added_other=wrong,stable_unannotated=unknown,unstable_action_parents=sum(not r['stable'] and bool(r['added']) for r in rr),good_action_parents=sum(r.get('good_parent',False) and bool(r['added']) for r in rr),max_added=sum(sorted([r['added'] for r in rr],reverse=True)[:1]),max_other=max((r['added_other'] for r in stable),default=0),full_NT_neither_selected_blocks=sum(b['full_neither_fraction']==1 for b in ss),full_N_supported_selected_blocks=sum(b['full_N_fraction']>0 for b in ss),entirely_correct_stable_unions=sum(r['added_other']==0 and r['unknown_annotation_added']==0 and bool(r['added']) for r in stable),duplicate_original_candidates=dups[arm]))
    run.met.csv_write(d/'action_breakdown.csv',actions)
    coverage=run.met.pool_summary([r for r in []]) if False else {}
    pool=list(csv.DictReader((run.OUT/'evaluation/pool.csv').open()))
    for mode in ('NT','A','B','A_single_diagnostic','B_single_diagnostic'):
        pp=[r for r in pool if r['mode']==mode];coverage[mode]={f'cross_{t}':sum(r[f'cross_{t}']=='True' for r in pp) for t in (.5,.75,.9,.95)}
    run.seal(d/'summary.json',dict(complete=True,independent_python_set_supply_parents=verified,valid_GT_targets=targets,scope_counts=scopes,actions=actions,pool_coverage_new_crossings=coverage,all_rules_and_predictions_unchanged=True,runtime_addendum_code_hashes_verified=True,posthoc_only=True,code_sha256=run.diag.sha(Path(__file__)),time=time.time()))

if __name__=='__main__':main()

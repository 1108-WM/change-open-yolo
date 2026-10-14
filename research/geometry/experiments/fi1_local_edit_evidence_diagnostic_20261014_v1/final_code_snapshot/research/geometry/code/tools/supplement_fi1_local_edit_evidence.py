#!/usr/bin/env python3
"""Post-freeze diagnostic strata and complete-NT witnesses, never edit rules."""
from pathlib import Path
import csv,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_local_edit_evidence as run
from verify_fi1_local_edit_evidence_counts import counts,iou

def main():
    c=run.read(run.OUT/'contract.json');run.diag.verify(run.read(run.OUT/'evaluation_frozen.json')['outputs'])
    rows=run.read(run.OUT/'evaluation/rows.json');lookup={(r['scene'],r['proposal_id']):r for r in rows}
    d=run.OUT/'supplement_v2';d.mkdir(exist_ok=True);witnesses=[];denominators={};frames=[]
    for s,j in c['jobs'].items():
        gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[ok],sizes[ok]
        denominators[j['cohort']]=denominators.get(j['cohort'],0)+len(ids)
        _,nt=run.diag.load_nt(j);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(counts(p,gt),len(p),ids,sizes))
        parents={}
        with np.load(run.OUT/'proposals'/s/'edits.npz') as edits,np.load(Path(j['asset_root'])/'structure'/s/'data.npz') as st:
            for r in (r for r in rows if r['scene']==s):
                cid=r['candidate_id']
                if cid not in parents:
                    p=st[cid+'_parent'];parents[cid]=(counts(p,gt),len(p))
                hist,n=parents[cid];pp=edits[r['proposal_id']];sign=1 if r['action']=='add' else -1
                after=iou(hist,n,ids,sizes) if n+sign*len(pp)<100 else iou(hist+sign*counts(pp,gt),n+sign*len(pp),ids,sizes)
                for k in np.flatnonzero(after-baseline>=.02-1e-12):
                    w=dict(r,pool_target=int(ids[k]),pool_baseline=float(baseline[k]),pool_after=float(after[k]),pool_gain=float(after[k]-baseline[k]),same_parent_target=bool(r['stable'] and r['target']==int(ids[k])))
                    w['strict_correct_witness']=bool(w['same_parent_target'] and r['net_gain'] and r['pure90'] and r['known_fraction']>=.9)
                    witnesses.append(w)
        for f in csv.DictReader((run.OUT/'proposals'/s/'frames.csv').open()):
            r=lookup[s,f['proposal_id']]
            if r['family']!='region':continue
            f.update(cohort=r['cohort'],source=r['source'],action=r['action']);frames.append(f)
    strata=[];purity=[];coverage=[]
    for cohort in c['cohorts']:
      for source in ('native','track'):
       for action in ('add','delete'):
        for family in ('region','query_boundary'):
          rr=[r for r in rows if (r['cohort'],r['source'],r['action'],r['family'])==(cohort,source,action,family)]
          group='/'.join((cohort,source,action,family))
          for size,lo,hi in (('1',1,1),('2-19',2,19),('20-99',20,99),('100+',100,float('inf'))):
            br=[r for r in rr if lo<=r['points']<=hi];sr=[r for r in br if r['stable']]
            strata.append(dict(group=group,size=size,proposals=len(br),stable=len(sr),net_gain=sum(r['net_gain'] for r in sr),net_harm=sum(r['net_harm'] for r in sr),pure90=sum(r['pure90'] for r in sr),pure90_known90=sum(r['pure90'] and r['known_fraction']>=.9 for r in sr),fallback=sum(r['minimum100_fallback'] for r in br)))
          known=[r for r in rr if r['stable'] and r['known_fraction']>=.9]
          for feature in ('query','margin','views','contact'):
            def value(r):
                sign=1 if action=='add' else -1
                return {'query':sign*r['query_mean'],'margin':sign*r['query_margin_mean'],'views':sign*(r['common_frames']-r['conflict_frames']),'contact':r['parent_contact05_fraction'] or 0}[feature]
            sceneauc=[run.auc([r['pure90'] for r in known if r['scene']==s],[value(r) for r in known if r['scene']==s]) for s in c['cohorts'][cohort]]
            valid=[a for a in sceneauc if a is not None]
            purity.append(dict(group=group,feature=feature,known90=len(known),pure90=sum(r['pure90'] for r in known),auc=run.auc([r['pure90'] for r in known],[value(r) for r in known]),macro_scene_auc=float(np.mean(valid)) if valid else None,scene_auc_supported=len(valid)))
          for kind in ('all','full_neither_positive','full_neither_all'):
            br=[r for r in rr if kind=='all' or (r['full_neither_fraction']>0 if kind=='full_neither_positive' else r['full_neither_fraction']==1)]
            sr=[r for r in br if r['stable']];ww=[w for w in witnesses if w['cohort']==cohort and w['source']==source and w['action']==action and w['family']==family and (kind=='all' or (w['full_neither_fraction']>0 if kind=='full_neither_positive' else w['full_neither_fraction']==1))]
            coverage.append(dict(group=group,coverage=kind,proposals=len(br),stable=len(sr),net_gain=sum(r['net_gain'] for r in sr),net_harm=sum(r['net_harm'] for r in sr),pool_targets=len({(w['scene'],w['pool_target']) for w in ww}),strict_correct_pool_targets=len({(w['scene'],w['pool_target']) for w in ww if w['strict_correct_witness']})))
    screens=run.core.screens(dict(action='add',query_mean=0,query_margin_mean=0,competitor_mean=0,common_frames=0,conflict_frames=0))
    poolsummary=[];pool=list(csv.DictReader((run.OUT/'evaluation/pool.csv').open()))
    for cohort in c['cohorts']:
      for source in ('native','track'):
       for action in ('add','delete'):
        for family in ('region','query_boundary'):
         for screen in screens:
            mode='/'.join((family,source,action,screen));ww=[w for w in witnesses if (w['cohort'],w['source'],w['action'],w['family'])==(cohort,source,action,family) and w['screen_'+screen]]
            for union in (False,True):
                pr=[p for p in pool if p['cohort']==cohort and p['mode']==mode+('/union' if union else '')]
                den=denominators[cohort]
                poolsummary.append(dict(cohort=cohort,mode=mode+('/union' if union else ''),full_target_denominator=den,present_pool_rows=len(pr),missing_rows_assumed_original_NT=den-len(pr),mean_gain_over_all_targets=sum(float(p['delta']) for p in pr)/den,pool_gain02=sum(p['gain02']=='True' for p in pr),same_parent_correct_targets=None if union else len({(w['scene'],w['pool_target']) for w in ww if w['same_parent_target'] and w['net_gain']}),pure90_known90_targets=None if union else len({(w['scene'],w['pool_target']) for w in ww if w['strict_correct_witness']})))
    frame_summary=[]
    for cohort in c['cohorts']:
     for source in ('native','track'):
      for action in ('add','delete'):
        ff=[f for f in frames if (f['cohort'],f['source'],f['action'])==(cohort,source,action)]
        frame_summary.append(dict(cohort=cohort,source=source,action=action,frames=len(ff),missing=sum(f['missing']=='True' for f in ff),no_visible_edit=sum(int(f['edit_visible'])==0 for f in ff),no_visible_reference=sum(int(f['reference_visible'])==0 for f in ff),mask_or_positive_coverage_missing=sum(f['missing']=='True' and int(f['edit_visible'])>0 and int(f['reference_visible'])>0 for f in ff),common=sum(f['common']=='True' for f in ff),conflict=sum(f['conflict']=='True' for f in ff),reference_mask_coverage_below_half=sum(f['missing']=='False' and float(f['reference_best_mask_coverage'])<.5 for f in ff)))
    for name,records in [('pool_witnesses',witnesses),('size_strata',strata),('pure90_comparison',purity),('full_NT_coverage',coverage),('complete_pool_summary',poolsummary),('frame_availability',frame_summary)]:run.met.csv_write(d/(name+'.csv'),records)
    run.seal(d/'summary.json',dict(posthoc_only=True,changes_to_proposals_evidence_or_evaluation=False,independent_validation=False,full_GT_targets=denominators,pool_witnesses=len(witnesses),frame_summary=frame_summary,code_sha256=run.diag.sha(Path(__file__))))
    print('supplement complete',denominators,'pool witnesses',len(witnesses),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Post-seal failure diagnostics; no inference or rule adjustment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_query_transfer_eval as run

def analyze():
    c=run.read(run.OUT/'contract.json');assert (run.OUT/'verification.json').exists();run.diag.verify(run.read(run.OUT/'predictions_frozen.json')['outputs'])
    rows=[];classifications=[];ref=[]
    for cohort,scenes in (('transfer',c['scenes']),('training',run.read(run.PILOT/'contract.json')['scenes'])):
        for s in scenes:
            gt=None if cohort=='training' else np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64)
            if cohort=='transfer':
                pr=np.load(run.OUT/'predictions'/s/'data.npz');ee=list(run.examples(s))
            else:ee=[e for (ss,cid),e in run.pilot.load().items() if ss==s]
            for e in ee:
                cid=e['candidate_id'];member=np.isin(e['halo'],e['parent']);visible=e['visible']>=2
                if cohort=='transfer':
                    ids,sizes=run.met.valid_targets(gt,c['class_ids']);im=run.met.identity(e['parent'],gt,ids,sizes);stable=im['stable'];y=np.full(len(e['halo']),2,np.int64)
                    if stable:y[gt[e['halo']]>=1000]=1;y[gt[e['halo']]==im['target']]=0
                else:y=e['y'];stable=True
                ref.append(dict(cohort=cohort,scene=s,candidate_id=cid,source=e['source'],stable=stable,points=len(y),parent_points=int(member.sum()),visible_fraction=float(visible.mean()),target_fraction=float((y==0).mean()),other_fraction=float((y==1).mean()),unknown_fraction=float((y==2).mean()),query_parent_mean=float(e['query'][member].mean())))
                for mode in ('A','B','B_zero'):
                    z=None if cohort=='transfer' else np.load(run.PILOT/'predictions'/s/(cid+('_A.npz' if mode=='A' else '_B.npz')))
                    prob=pr[cid+'_'+mode+'_probability'] if z is None else z[mode+'_probability'];points=pr[cid+'_'+mode+'_points'] if z is None else z[mode+'_points'];mask=np.isin(e['halo'],points)
                    pred=prob.argmax(1);known=y!=2
                    classifications.append(dict(cohort=cohort,scene=s,candidate_id=cid,source=e['source'],mode=mode,stable=stable,known_points=int(known.sum()),known_correct=int((pred[known]==y[known]).sum()),points=len(y),correct=int((pred==y).sum()),target_points=int((y==0).sum()),target_pred_other=int(((y==0)&(pred==1)).sum()),other_points=int((y==1).sum()),other_pred_target=int(((y==1)&(pred==0)).sum())))
                    if stable:
                        slices=dict(correct_add=(~member&mask&(y==0)),wrong_add=(~member&mask&(y==1)),correct_delete=(member&~mask&(y==1)),wrong_delete=(member&~mask&(y==0)),missed_add=(~member&visible&~mask&(y==0)),missed_delete=(member&visible&mask&(y==1)))
                        for kind,select in slices.items():
                            n=int(select.sum());col=1 if 'delete' in kind else 0
                            rows.append(dict(cohort=cohort,scene=s,candidate_id=cid,source=e['source'],mode=mode,kind=kind,points=n,query_probability_mean=float(e['query'][select].mean()) if n else None,action_probability_mean=float(prob[select,col].mean()) if n else None,query_ge05_points=int((e['query'][select]>=.5).sum()) if n else 0,action_probability_sum=float(prob[select,col].sum(dtype=np.float64)) if n else 0))
                    if z is not None:z.close()
            if cohort=='transfer':pr.close()
    d=run.OUT/'failure_diagnosis';d.mkdir(exist_ok=True)
    for name,rr in (('edit_probability',rows),('classification',classifications),('cohort_input',ref)):run.met.csv_write(d/(name+'.csv'),rr)
    summary={}
    for cohort in ('training','transfer'):
        for mode in ('A','B','B_zero'):
            rr=[r for r in classifications if r['cohort']==cohort and r['mode']==mode];known=sum(r['known_points'] for r in rr)
            stats=dict(known_points=known,known_accuracy=sum(r['known_correct'] for r in rr)/known,other_pred_target=sum(r['other_pred_target'] for r in rr),target_pred_other=sum(r['target_pred_other'] for r in rr))
            for kind in ('correct_add','wrong_add','correct_delete','wrong_delete','missed_add','missed_delete'):
                er=[r for r in rows if r['cohort']==cohort and r['mode']==mode and r['kind']==kind];n=sum(r['points'] for r in er)
                stats[kind]=dict(points=n,query_ge05_points=sum(r['query_ge05_points'] for r in er),action_probability_mean=sum(r['action_probability_sum'] for r in er)/n if n else None)
            summary[cohort+'/'+mode]=stats
    run.seal(d/'summary.json',summary)
    print(run.diag.plain(summary),flush=True)

if __name__=='__main__':analyze()

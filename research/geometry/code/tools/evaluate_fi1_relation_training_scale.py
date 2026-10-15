#!/usr/bin/env python3
"""Postseal relationship and full-NT assessment for the single scale comparison."""
from collections import Counter
import csv
import numpy as np
import run_fi1_relation_training_scale as r
import evaluate_fi1_nt_region_identity as met


def main():
    r.guard(gt=True);c=r.contract();a=r.a
    r.verify(a.read(r.OUT/'predictions_frozen.json')['outputs']);r.verify(c['label_files'])
    r.verify(a.read(r.SOURCE/'evaluation_frozen.json')['outputs'])
    rr,x,nold=r.combined()
    labels={}
    for s in c['scenes']+c['extra_scenes']:
        root=r.BASE if s in c['scenes'] else r.OUT/'training_data'
        labels.update({(s,q['pair_id']):q['relation'] for q in a.read(root/'labels'/f'{s}.json')['pairs']})
    y=np.array([1 if labels[q['scene'],q['pair_id']]=='same' else 0 if labels[q['scene'],q['pair_id']]=='different' else -1 for q in rr])
    scenes=np.array([q['scene'] for q in rr]);hard=x[:,4]==0
    diagnostic={};outputs={}
    for arm in r.ARMS:
        per_fold={};acc={p:dict(y=[],score=[],scenes=[]) for p in ('train','common_half','heldout')}
        for f in range(5):
            d=r.OUT/'models'/f'fold{f}'
            score=np.load(d/(arm+'_scores.npy'))
            idx=np.load(d/(arm+'_training.npz'))['row_indices']
            halfidx=np.load(d/'half_training.npz')['row_indices']
            test=np.array([i for i,q in enumerate(rr[:nold]) if c['folds'][q['scene']]==f and y[i]>=0])
            per_fold[str(f)]={}
            for part,ix in (('train',idx),('common_half',halfidx),('heldout',test)):
                per_fold[str(f)][part]={key:r.metrics.metrics(y[ii],score[ii],scenes[ii]) for key,ii in (
                    ('all',ix),('disjoint',ix[hard[ix]]),('overlap',ix[~hard[ix]]))}
                ii=ix[hard[ix]]
                for key,value in (('y',y[ii]),('score',score[ii]),('scenes',scenes[ii])):acc[part][key].extend(value)
        diagnostic[arm]=dict(folds=per_fold,pooled={part:r.metrics.metrics(**values) for part,values in acc.items()})
    steps=[]
    for left,right in zip(r.ARMS[:-1],r.ARMS[1:]):
        lc,rc=(diagnostic[arm]['pooled']['heldout'] for arm in (left,right))
        better=sum(diagnostic[right]['folds'][str(f)]['heldout']['disjoint']['balanced_logloss']<diagnostic[left]['folds'][str(f)]['heldout']['disjoint']['balanced_logloss'] for f in range(5))
        checks=dict(lower_balanced_logloss=rc['balanced_logloss']<lc['balanced_logloss'],at_least3_folds_lower=better>=3,
            different_high90_not_higher=rc['negative']['high90_fraction']<=lc['negative']['high90_fraction'],
            same_high90_not_lower=rc['positive']['high90_fraction']>=lc['positive']['high90_fraction'])
        steps.append(dict(left=left,right=right,checks=checks,lower_folds=better,passed=all(checks.values())))
    trend=all(q['passed'] for q in steps) and diagnostic['double']['pooled']['heldout']['negative']['high90_fraction']<diagnostic['half']['pooled']['heldout']['negative']['high90_fraction']
    a.seal(r.OUT/'learning_diagnostics.json',dict(arms=diagnostic,steps=steps,relationship_trend_passed=trend,
        heldout_clear_difficult_pairs=int(((y[:nold]>=0)&hard[:nold]).sum()),all52_exposed_development=True,
        training_occurrences_not_independent=True))
    oldrows={(q['scene'],q['pair_id']):q for q in a.read(r.SOURCE/'evaluation/C_pairs.json')}
    base={(q['scene'],q['gt_id']):q['baseline'] for q in a.read(r.SOURCE/'evaluation/C_targets.json')}
    jobs=a.read(r.AUDIT/'contract.json')['jobs'];old=a.read(a.OLD/'contract.json')
    results={};joinedarms={}
    for arm in r.ARMS:
        pred=a.read(r.OUT/(arm+'_decisions.json'));joined=[];targets=[]
        for s in c['scenes']:
            gp=jobs[s]['job']['GT'];assert a.sha(gp)==old['GT_inputs'][gp]
            gt=np.loadtxt(gp,dtype=np.int64);ids,sizes=met.valid_targets(gt,old['class_ids'])
            baseline=np.array([base[s,int(g)] for g in ids]);best=baseline.copy()
            with np.load(r.GEOMETRY/'predictions'/s/'data.npz') as z:
                for q in (q for q in pred if q['scene']==s):
                    row=dict(q);prior=oldrows[s,q['pair_id']]
                    for k in ('GT_relation_strict','GT_relation_previous','safe_gain02','same_target','same_target_pool_delta',
                              'union_target_purity','mixed_endpoint','good_endpoint_loss_gt02','union_GT'):row[k]=prior[k]
                    joined.append(row)
                    if q['emit']:best=np.maximum(best,met.ious(z[q['pair_id']+'_union'],gt,ids,sizes))
            targets.extend(met.pool_rows(s,arm,baseline,best,ids))
        joinedarms[arm]={(q['scene'],q['pair_id']):q for q in joined}
        accepted=[q for q in joined if q['state']=='accept'];emitted=[q for q in accepted if q['emit']]
        safe=[q for q in emitted if q['safe_gain02']]
        unknown=sum(q['union_GT']['unknown'] for q in emitted)/max(1,sum(q['union_GT']['points'] for q in emitted))
        gates=dict(safe_gain_scenes=len({q['scene'] for q in safe})>=2,
            no_confirmed_cross_identity=not any(q['GT_relation_previous']=='different' for q in accepted),
            no_unresolved_accept=not any(q['GT_relation_previous']=='unresolved' for q in accepted),
            full_endpoint_identity=bool(accepted) and all(q['GT_relation_strict']=='same' for q in accepted),
            good_copy_safety=not any(q['good_endpoint_loss_gt02'] for q in accepted),annotation_unknown=unknown<=.1)
        results[arm]=dict(decisions=dict(Counter(q['state'] for q in joined)),emitted=len(emitted),
            strict_accepted=dict(Counter(q['GT_relation_strict'] for q in accepted)),
            previous_accepted=dict(Counter(q['GT_relation_previous'] for q in accepted)),
            accepted_mixed=sum(q['mixed_endpoint'] for q in accepted),good_endpoint_loss_gt02=sum(q['good_endpoint_loss_gt02'] for q in accepted),
            safe_gain_pairs=len(safe),safe_gain_targets=len({(q['scene'],q['same_target']) for q in safe}),
            safe_gain_scenes=sorted({q['scene'] for q in safe}),pool=met.pool_summary(targets),
            annotation_unknown_fraction=unknown,gates=gates,passed=all(gates.values()))
        for name,value in (('pairs',joined),('targets',targets)):
            path=r.OUT/'evaluation'/(arm+'_'+name+'.json');a.seal(path,value);outputs[str(path)]=a.sha(path)
        compact=[{k:r.json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in q.items() if k!='views'} for q in joined]
        path=r.OUT/(arm+'_evidence.csv')
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
        outputs[str(path)]=a.sha(path)
    opportunities=[]
    for i,q in enumerate(rr[:nold]):
        if not oldrows[q['scene'],q['pair_id']]['safe_gain02']:continue
        f=c['folds'][q['scene']];stats={}
        for arm in r.ARMS:
            p=joinedarms[arm][q['scene'],q['pair_id']]
            trained=[];other=[]
            for j in range(5):
                idx=np.load(r.OUT/'models'/f'fold{j}'/(arm+'_training.npz'))['row_indices']
                v=float(np.load(r.OUT/'models'/f'fold{j}'/(arm+'_scores.npy'))[i])
                if i in idx:trained.append(dict(fold=j,score=v))
                elif j!=f:other.append(dict(fold=j,score=v))
            stats[arm]=dict(oof=p['score'],state=p['state'],emit=p['emit'],reason=p['reason'],
                actual_training_predictions=trained,other_nonfitting_predictions=other)
        opportunities.append(dict(scene=q['scene'],pair_id=q['pair_id'],relation=labels[q['scene'],q['pair_id']],
            delta=oldrows[q['scene'],q['pair_id']]['same_target_pool_delta'],arms=stats))
    assert len(opportunities)==18
    a.seal(r.OUT/'opportunity_diagnostics.json',dict(posthoc_only=True,opportunities=opportunities))
    safe=results['double']['passed']
    decision='prepare_unexposed_validation_proposal' if trend and safe else ('coverage_trend_only_output_unsafe' if trend else 'stop_expanding_this_fixed_configuration')
    a.seal(r.OUT/'results.json',dict(complete=True,results=results,relationship_trend_passed=trend,
        geometry_repair_passed=safe,decision=decision,AP=False,all52_exposed_development=True))
    for name in ('learning_diagnostics.json','opportunity_diagnostics.json','results.json'):outputs[str(r.OUT/name)]=a.sha(r.OUT/name)
    a.seal(r.OUT/'evaluation_frozen.json',dict(complete=True,prediction_sha256=a.sha(r.OUT/'predictions_frozen.json'),outputs=outputs))
    print(r.json.dumps(dict(steps=steps,relationship_trend_passed=trend,results=results,decision=decision),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

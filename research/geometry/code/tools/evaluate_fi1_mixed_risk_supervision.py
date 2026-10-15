#!/usr/bin/env python3
"""Postseal learning, mixed-risk and full-NT outcome assessment."""
from collections import Counter
import csv
import numpy as np
import run_fi1_mixed_risk_supervision as r
import evaluate_fi1_nt_region_identity as met


def describe(v):
    v=np.asarray(v)
    return dict(rows=len(v),mean=float(v.mean()),median=float(np.median(v)),
        q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)),high90=int((v>=.9).sum())) if len(v) else dict(rows=0)


def multiclass(y,p):
    known=y>=0; yy=y[known]; pp=p[known]
    return dict(rows=len(y),supervised_rows=len(yy),
        logloss=float(-np.log(np.clip(pp[np.arange(len(yy)),yy],1e-12,1)).mean()),
        confusion_true_rows_pred_columns=[[int(((yy==k)&(pp.argmax(1)==j)).sum()) for j in range(3)] for k in range(3)],
        by_label={name:dict(same=describe(p[y==k,1]),mixed=describe(p[y==k,2])) for name,k in r.CLASSES.items()})


def main():
    r.guard(gt=True);c=r.contract();a=r.a
    r.verify(a.read(r.OUT/'predictions_frozen.json')['outputs'])
    r.verify(a.read(r.SCALE/'evaluation_frozen.json')['outputs'])
    r.verify(a.read(r.controls.OUT/'evaluation_frozen.json')['outputs'])
    rr,x,n=r.combined();y=r.labels_for(c['scenes']+c['extra_scenes'],c,rr)
    scenes=np.array([q['scene'] for q in rr]);hard=x[:,4]==0
    clear=(y==0)|(y==1);diag={};risk={};scores={};outputs={}
    for arm in r.ARMS:
        per={};acc={p:dict(y=[],score=[],scenes=[]) for p in ('train','common_clear_train','heldout')}
        train_y=[];train_p=[]
        for f in range(5):
            d=r.OUT/'models'/f'fold{f}'; old=r.SCALE/'models'/f'fold{f}'
            if arm=='binary':
                score=np.load(old/'double_scores.npy');idx=np.load(old/'double_training.npz')['row_indices']
            else:
                prob=np.load(d/'probabilities.npy');score=prob[:,1];idx=np.load(d/'training.npz')['row_indices']
                train_y.extend(y[idx]);train_p.extend(prob[idx])
            common=np.load(old/'double_training.npz')['row_indices']
            test=np.array([i for i,q in enumerate(rr[:n]) if c['folds'][q['scene']]==f])
            per[str(f)]={}
            for part,ix in (('train',idx),('common_clear_train',common),('heldout',test)):
                ix=ix[clear[ix]]
                per[str(f)][part]={key:r.metrics.metrics(y[ii],score[ii],scenes[ii]) for key,ii in (
                    ('all',ix),('disjoint',ix[hard[ix]]),('overlap',ix[~hard[ix]]))}
                ii=ix[hard[ix]]
                for key,val in (('y',y[ii]),('score',score[ii]),('scenes',scenes[ii])):acc[part][key].extend(val)
            if arm=='mixed':
                per[str(f)]['multiclass_train']=multiclass(y[idx],prob[idx])
                per[str(f)]['multiclass_heldout']=multiclass(y[test],prob[test])
        diag[arm]=dict(folds=per,pooled={part:r.metrics.metrics(**v) for part,v in acc.items()})
        raw=np.load(r.OUT/(arm+'_scores.npz'))['oof'];scores[arm]=raw[:,1] if arm=='mixed' else raw
        if arm=='mixed':risk=dict(train=multiclass(np.array(train_y),np.array(train_p)),heldout=multiclass(y[:n],raw))
    lc,rc=(diag[arm]['pooled']['heldout'] for arm in r.ARMS)
    lower=sum(diag['mixed']['folds'][str(f)]['heldout']['disjoint']['balanced_logloss']<diag['binary']['folds'][str(f)]['heldout']['disjoint']['balanced_logloss'] for f in range(5))
    checks=dict(lower_balanced_logloss=rc['balanced_logloss']<lc['balanced_logloss'],at_least3_folds_lower=lower>=3,
        different_high90_lower=rc['negative']['high90_fraction']<lc['negative']['high90_fraction'],
        same_high90_not_lower=rc['positive']['high90_fraction']>=lc['positive']['high90_fraction'])
    a.seal(r.OUT/'learning_diagnostics.json',dict(arms=diag,mixed_risk=risk,checks=checks,lower_folds=lower,
        classification_passed=all(checks.values()),raw_same_probability=True,training_occurrences_not_independent=True))
    oldrows={(q['scene'],q['pair_id']):q for q in a.read(r.controls.OUT/'evaluation/C_pairs.json')}
    base={(q['scene'],q['gt_id']):q['baseline'] for q in a.read(r.controls.OUT/'evaluation/C_targets.json')}
    jobs=a.read(r.AUDIT/'contract.json')['jobs'];old=a.read(a.OLD/'contract.json');results={};joinedarms={}
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
                    if arm=='binary':row['mixed_probability']=None
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
        with (r.OUT/(arm+'_evidence.csv')).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
        outputs[str(r.OUT/(arm+'_evidence.csv'))]=a.sha(r.OUT/(arm+'_evidence.csv'))
    opportunities=[]
    for i,q in enumerate(rr[:n]):
        if not oldrows[q['scene'],q['pair_id']]['safe_gain02']:continue
        stats={};f=c['folds'][q['scene']]
        for arm in r.ARMS:
            p=joinedarms[arm][q['scene'],q['pair_id']];trained=[];other=[]
            for j in range(5):
                if arm=='binary':
                    d=r.SCALE/'models'/f'fold{j}';idx=np.load(d/'double_training.npz')['row_indices'];v=float(np.load(d/'double_scores.npy')[i]);mix=None
                else:
                    d=r.OUT/'models'/f'fold{j}';idx=np.load(d/'training.npz')['row_indices'];pp=np.load(d/'probabilities.npy')[i];v=float(pp[1]);mix=float(pp[2])
                if i in idx:trained.append(dict(fold=j,score=v,mixed_probability=mix))
                elif j!=f:other.append(dict(fold=j,score=v,mixed_probability=mix))
            competitors=[dict(pair_id=t['pair_id'],score=t['score'],state=t['state'],emit=t['emit'],
                relation=t['GT_relation_strict'],mixed_probability=t.get('mixed_probability')) for t in joinedarms[arm].values()
                if t['scene']==q['scene'] and t['pair_id']!=q['pair_id'] and {q['left'],q['right']}&{t['left'],t['right']}]
            competitors.sort(key=lambda t:(-t['score'],t['pair_id']))
            stats[arm]=dict(oof=p['score'],state=p['state'],emit=p['emit'],reason=p['reason'],mixed_probability=p.get('mixed_probability'),
                actual_training_predictions=trained,other_nonfitting_predictions=other,shared_endpoint_competitors=competitors,
                different_competitors_above=sum(t['relation']=='different' and t['score']>p['score'] for t in competitors),
                mixed_competitors_above=sum(t['relation']=='mixed' and t['score']>p['score'] for t in competitors))
        opportunities.append(dict(scene=q['scene'],pair_id=q['pair_id'],relation=next(k for k,v in r.CLASSES.items() if v==y[i]),
            delta=oldrows[q['scene'],q['pair_id']]['same_target_pool_delta'],arms=stats))
    assert len(opportunities)==18
    a.seal(r.OUT/'opportunity_diagnostics.json',dict(posthoc_only=True,opportunities=opportunities))
    a.seal(r.OUT/'results.json',dict(complete=True,results=results,classification_checks=checks,
        classification_passed=all(checks.values()),geometry_repair_passed=results['mixed']['passed'],
        decision='prepare_unexposed_validation' if results['mixed']['passed'] else 'stop_fixed_summary_configuration_prioritize_spatial_part_representation',
        AP=False,mixed_is_not_cannot_link=True,all52_exposed_development=True))
    for name in ('learning_diagnostics.json','opportunity_diagnostics.json','results.json'):outputs[str(r.OUT/name)]=a.sha(r.OUT/name)
    a.seal(r.OUT/'evaluation_frozen.json',dict(complete=True,prediction_sha256=a.sha(r.OUT/'predictions_frozen.json'),outputs=outputs))
    print(r.json.dumps(dict(checks=checks,results=results),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

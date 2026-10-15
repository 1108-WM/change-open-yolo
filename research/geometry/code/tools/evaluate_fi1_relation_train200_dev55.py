#!/usr/bin/env python3
"""Postseal dev55 relationship, geometry, risk and candidate opportunity diagnosis."""
from collections import Counter
import csv
import numpy as np
import run_fi1_relation_train200_dev55 as r
import evaluate_fi1_nt_region_identity as met


def y_for(rr):
    lookup={}
    for s in sorted({q['scene'] for q in rr}):lookup.update({(s,q['pair_id']):q['relation'] for q in r.a.read(r.OUT/'labels'/f'{s}.json')['pairs']})
    return np.array([1 if lookup[q['scene'],q['pair_id']]=='same' else 0 if lookup[q['scene'],q['pair_id']]=='different' else -1 for q in rr]),lookup


def relation_metrics(rr,x,y,p,idx):
    ss=np.array([q['scene'] for q in rr]);physical=np.array([s.rsplit('_',1)[0] for s in ss]);idx=idx[y[idx]>=0]
    return {name:dict(scan=r.scale.metrics.metrics(y[ii],p[ii],ss[ii]),physical=r.scale.metrics.metrics(y[ii],p[ii],physical[ii]))
        for name,ii in (('all',idx),('disjoint',idx[x[idx,4]==0]),('overlap',idx[x[idx,4]>0]))}


def composition(pp,gt,target=None):
    ids,ns=np.unique(gt[pp],return_counts=True)
    return dict(points=len(pp),distribution={str(g):int(n) for g,n in zip(ids,ns)},known_instances=int((ids>=1000).sum()),
        unknown=int(ns[ids<1000].sum()),target=int((gt[pp]==target).sum()) if target is not None else None)


def main():
    r.guard(gt=True);c=r.contract();a=r.a;predseal=a.read(r.OUT/'predictions_frozen.json');r.verify(predseal['outputs'])
    a.seal(r.OUT/'dev_labels_frozen.json',r.make_labels(c['scenes'],c))
    rr,x=r.matrix('dev55');y,lookup=y_for(rr);tr,tx=r.matrix('train200');ty,_=y_for(tr)
    diagnostics={};scores={};outputs={}
    common=np.array([i for i,q in enumerate(tr) if q['scene'] in c['small_scenes'] and ty[i]>=0])
    for arm in r.ARMS:
        p=np.load(r.OUT/(arm+'_scores.npy'));tp=np.load(r.OUT/(arm+'_train_scores.npy'));scores[arm]=p
        actual=common if arm=='reference82' else np.flatnonzero(ty>=0)
        diagnostics[arm]=dict(dev55=relation_metrics(rr,x,y,p,np.arange(len(rr))),
            actual_train=relation_metrics(tr,tx,ty,tp,actual),common82_train=relation_metrics(tr,tx,ty,tp,common),
            by_relation={name:dict(rows=len(v),mean=float(v.mean()),median=float(np.median(v)),high90=int((v>=.9).sum()),low10=int((v<=.1).sum())) if len(v) else dict(rows=0)
                for name in ('same','different','mixed','unknown') for v in [p[np.array([lookup[q['scene'],q['pair_id']]==name for q in rr])]]},
            per_physical={s:relation_metrics(rr,x,y,p,np.array([i for i,q in enumerate(rr) if q['scene'].rsplit('_',1)[0]==s])) for s in c['dev_physical']})
    a.seal(r.OUT/'learning_diagnostics.json',dict(arms=diagnostics,comparison=r.scale.metrics.improvement(diagnostics['reference82']['dev55']['disjoint']['scan'],diagnostics['train200']['dev55']['disjoint']['scan']),
        scans=55,physical_sites=29,physical_sites_not_scans_are_independent_units=True,no_old52_denominator=True))
    classes=a.read(r.SPLIT)['class_ids'];allpairs=[];targetarms={arm:[] for arm in r.ARMS};supplytargets=[];eligibletargets=[]
    preds={arm:{(q['scene'],q['pair_id']):q for q in a.read(r.OUT/(arm+'_decisions.json'))} for arm in r.ARMS}
    joined={arm:[] for arm in r.ARMS}
    for si,s in enumerate(c['scenes']):
        spec=c['jobs'][s];gp=spec['job']['GT'];assert a.sha(gp)==spec['GT_sha256'];gt=np.loadtxt(gp,dtype=np.int64)
        ids,sizes=met.valid_targets(gt,classes);labs=a.read(r.OUT/'labels'/f'{s}.json');strict={q['pair_id']:q['relation'] for q in labs['pairs']}
        nodes=a.read(r.OUT/'geometry/predictions'/s/'nodes.json');rows=[q for q in rr if q['scene']==s]
        with np.load(r.OUT/'geometry/predictions'/s/'data.npz') as z:
            baseline=np.zeros(len(ids))
            for n in nodes:
                if n['in_original_NT']:baseline=np.maximum(baseline,met.ious(z['n%04d'%n['node']],gt,ids,sizes))
            metas={n['node']:r.scale.candidate.track_identity(z['n%04d'%n['node']],gt,ids,sizes) for n in nodes if n['final_track_ids']}
            best={arm:baseline.copy() for arm in r.ARMS};supply=baseline.copy();eligible=baseline.copy()
            for q in rows:
                pp=z[q['pair_id']+'_union'];v=met.ious(pp,gt,ids,sizes);ml,mr=metas[q['left']],metas[q['right']]
                relation='same' if ml['stable'] and mr['stable'] and ml['target']==mr['target'] else 'different' if ml['stable'] and mr['stable'] else 'unresolved'
                target=ml['target'] if relation=='same' else None;j=ml['index'] if relation=='same' else None
                comp=composition(pp,gt,target);purity=comp['target']/len(pp) if target is not None else None
                delta=float(v[j]-baseline[j]) if j is not None else None
                losses=[dict(endpoint=end,target=m['target'],parent_iou=m['parent_iou'],union_iou=float(v[m['index']]),loss=float(m['parent_iou']-v[m['index']]))
                    for end,m in (('left',ml),('right',mr)) if m['stable'] and m['parent_iou']>=.9]
                row=dict(q,GT_relation_strict=strict[q['pair_id']],GT_relation_previous=relation,
                    same_target=target,same_target_pool_delta=delta,union_target_purity=purity,union_GT=comp,
                    safe_gain02=bool(relation=='same' and purity>=.9 and delta>=.02-1e-12),
                    any_target_gain02_ids=ids[v-baseline>=.02-1e-12].tolist(),mixed_endpoint=strict[q['pair_id']]=='mixed',
                    good_endpoint_losses=losses,good_endpoint_loss_gt02=any(t['loss']>.02+1e-12 for t in losses))
                allpairs.append(row);supply=np.maximum(supply,v)
                if q['eligible']:eligible=np.maximum(eligible,v)
                for arm in r.ARMS:
                    action=preds[arm][s,q['pair_id']];joined[arm].append(dict(row,**{k:action[k] for k in ('score','state','reason','emit')}))
                    if action['emit']:best[arm]=np.maximum(best[arm],v)
            for arm in r.ARMS:targetarms[arm].extend(met.pool_rows(s,arm,baseline,best[arm],ids))
            supplytargets.extend(met.pool_rows(s,'all_proposed',baseline,supply,ids));eligibletargets.extend(met.pool_rows(s,'eligible_proposed',baseline,eligible,ids))
        print('POSTSEAL_GT',si+1,s,len(rows),flush=True)
    results={}
    for arm in r.ARMS:
        accepted=[q for q in joined[arm] if q['state']=='accept'];emitted=[q for q in accepted if q['emit']];safe=[q for q in emitted if q['safe_gain02']]
        unknown=sum(q['union_GT']['unknown'] for q in emitted)/max(1,sum(q['union_GT']['points'] for q in emitted))
        gates=dict(safe_gain_scenes=len({q['scene'].rsplit('_',1)[0] for q in safe})>=2,
            no_confirmed_cross_identity=not any(q['GT_relation_previous']=='different' for q in accepted),
            no_unresolved_accept=not any(q['GT_relation_previous']=='unresolved' for q in accepted),
            full_endpoint_identity=bool(accepted) and all(q['GT_relation_strict']=='same' for q in accepted),
            good_copy_safety=not any(q['good_endpoint_loss_gt02'] for q in accepted),annotation_unknown=unknown<=.1)
        results[arm]=dict(decisions=dict(Counter(q['state'] for q in joined[arm])),emitted=len(emitted),strict_accepted=dict(Counter(q['GT_relation_strict'] for q in accepted)),
            previous_accepted=dict(Counter(q['GT_relation_previous'] for q in accepted)),accepted_mixed=sum(q['mixed_endpoint'] for q in accepted),
            good_endpoint_loss_gt02=sum(q['good_endpoint_loss_gt02'] for q in accepted),safe_gain_pairs=len(safe),safe_gain_targets=len({(q['scene'],q['same_target']) for q in safe}),
            safe_gain_physical_sites=sorted({q['scene'].rsplit('_',1)[0] for q in safe}),pool=met.pool_summary(targetarms[arm]),annotation_unknown_fraction=unknown,gates=gates,passed=all(gates.values()))
        for name,value in (('pairs',joined[arm]),('targets',targetarms[arm])):
            path=r.OUT/'evaluation'/(arm+'_'+name+'.json');a.seal(path,value);outputs[str(path)]=a.sha(path)
        path=r.OUT/(arm+'_evidence.csv')
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=sorted({k for q in joined[arm] for k in q if k!='views'}));w.writeheader()
            w.writerows([{k:r.scale.json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in q.items() if k!='views'} for q in joined[arm]])
        outputs[str(path)]=a.sha(path)
    opportunities=[]
    for q in allpairs:
        if not q['safe_gain02'] and not q['any_target_gain02_ids']:continue
        opportunities.append(dict(scene=q['scene'],pair_id=q['pair_id'],strict_relation=q['GT_relation_strict'],safe_gain02=q['safe_gain02'],
            eligible=q['eligible'],same_target=q['same_target'],delta=q['same_target_pool_delta'],any_target_gain02_ids=q['any_target_gain02_ids'],
            arms={arm:{k:preds[arm][q['scene'],q['pair_id']][k] for k in ('score','state','emit','reason')} for arm in r.ARMS}))
    safe=[q for q in allpairs if q['safe_gain02']]
    a.seal(r.OUT/'opportunity_diagnostics.json',dict(posthoc_only=True,all_proposed_pool=met.pool_summary(supplytargets),eligible_proposed_pool=met.pool_summary(eligibletargets),
        safe_pairs=len(safe),eligible_safe_pairs=sum(q['eligible'] for q in safe),safe_targets=len({(q['scene'],q['same_target']) for q in safe}),
        safe_physical_sites=sorted({q['scene'].rsplit('_',1)[0] for q in safe}),strict_safe_counts=dict(Counter(q['GT_relation_strict'] for q in safe)),opportunities=opportunities))
    for name,value in (('supply_pairs',allpairs),('supply_targets',supplytargets),('eligible_supply_targets',eligibletargets)):
        path=r.OUT/'evaluation'/(name+'.json');a.seal(path,value);outputs[str(path)]=a.sha(path)
    a.seal(r.OUT/'results.json',dict(complete=True,results=results,AP=False,comparison_on_same_dev55=True,scans=55,physical_sites=29,
        training200_all_predefined_scenes=True,geometry_repair_passed=results['train200']['passed'],
        decision='prepare_separate_unexposed_validation' if results['train200']['passed'] else 'baseline_does_not_pass_safe_geometry_repair'))
    for name in ('learning_diagnostics.json','opportunity_diagnostics.json','results.json','dev_labels_frozen.json'):outputs[str(r.OUT/name)]=a.sha(r.OUT/name)
    outputs.update(a.read(r.OUT/'dev_labels_frozen.json')['outputs'])
    a.seal(r.OUT/'evaluation_frozen.json',dict(complete=True,time_unix=r.time.time(),prediction_sha256=a.sha(r.OUT/'predictions_frozen.json'),outputs=outputs))
    print(r.scale.json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

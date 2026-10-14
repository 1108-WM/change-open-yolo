#!/usr/bin/env python3
"""Post-seal diagnostics: raw prior information separately from keep/fallback probes.

No new association, threshold, scene selection or output action is introduced.
The original frozen inference and evaluation artifacts remain intact.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import rankdata
import run_fi1_nt_region_identity as run
import evaluate_fi1_nt_region_identity as ev
import fi1_nt_region_identity_core as core


def main():
    c=run.read(run.OUT/'contract.json')
    for name in ('priors','associations','prior_diagnosis'):run.verify(run.read(run.OUT/(name+'_frozen.json'))['outputs'])
    d=run.OUT/'supplement';d.mkdir(exist_ok=True)
    rawrows=[];parentrows=[];regions=[];poolrows=[];scenes=[]
    for s in c['scenes']:
        gt=ev.load_gt(c,s);ids,sizes=ev.valid_targets(gt,c['class_ids']);_,nt=run.load_nt(c['jobs'][s]);baseline=ev.pool_base(nt,gt,ids,sizes)
        best={mode+'_'+kind:baseline.copy() for mode in ('P0','P1') for kind in ('all_query_probe_oracle','nonGT_set_probe_oracle')}
        links={(r['candidate_id'],r['prior']):r for r in run.read(run.OUT/'associations'/s/'links.json')}
        with np.load(run.OUT/'structure'/s/'data.npz') as z,np.load(run.OUT/'priors'/s/'data.npz') as ps:
            roi=ps['point_ids']
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];truth=gt[halo];labels=z[cid+'_labels'];meta=ev.identity(parent,gt,ids,sizes)
                localids,localcounts=np.unique(truth[truth>=1000],return_counts=True)
                for mode in ('P0','P1'):
                    p=ps[mode+'_probability'][np.searchsorted(roi,halo)];hard=p>=.5;nq=p.shape[1];sizesq=hard.sum(0)
                    hit=np.zeros((nq,len(localids)),np.int64)
                    for i,g in enumerate(localids):hit[:,i]=hard[truth==g].sum(0)
                    rawiou=hit/np.maximum(sizesq[:,None]+localcounts[None,:]-hit,1)
                    dominant=[int(localids[rawiou[q].argmax()]) if len(localids) and rawiou[q].max()>0 else None for q in range(nq)]
                    association=links[cid,mode];chosen=association['parent']['top'][0];queryset=association['query_set']
                    valid_hit=[];fallback=[]
                    for q in range(nq):
                        output=core.probe(parent,halo,p[:,q],z[cid+'_visible']);valid_hit.append(ev.ious(output,gt,ids,sizes))
                        member=np.isin(halo,parent);seen=z[cid+'_visible']>=2;tentative=np.where(seen,hard[:,q],member)
                        fallback.append(int(tentative.sum())<100)
                        row=dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,query=q,raw_halo_mask_points=int(sizesq[q]),
                            raw_halo_unknown_points=int(hard[truth<1000,q].sum()),raw_halo_dominant_identity=dominant[q],
                            raw_halo_dominant_IoU=float(rawiou[q].max(initial=0)),probe_min100_fallback=fallback[q],
                            selected_top1=q==chosen,in_nonGT_set=q in queryset)
                        if meta['stable']:
                            g=meta['target'];a=np.flatnonzero(localids==g)[0];target=truth==g;other=(truth>=1000)&~target
                            tp=int(hit[q,a]);row.update(target=g,raw_target_iou=float(rawiou[q,a]),
                                raw_target_precision=tp/max(1,int(sizesq[q])),raw_target_recall=tp/max(1,int(target.sum())),
                                target_mean_probability=float(p[target,q].mean()),
                                neighbor_mean_probability=float(p[other,q].mean()) if other.any() else None)
                            if target.any() and other.any():
                                values=p[target|other,q];positive=target[target|other];ranks=rankdata(values)
                                npos=int(positive.sum());nneg=len(values)-npos
                                row['known_target_vs_neighbor_AUROC']=float((ranks[positive].sum()-npos*(npos+1)/2)/(npos*nneg))
                        rawrows.append(row)
                    valid_hit=np.array(valid_hit)
                    # All-query maxima are GT diagnostic references only. They do not replace top1 probes.
                    if len(ids):
                        best[mode+'_all_query_probe_oracle']=np.maximum(best[mode+'_all_query_probe_oracle'],valid_hit.max(0))
                        best[mode+'_nonGT_set_probe_oracle']=np.maximum(best[mode+'_nonGT_set_probe_oracle'],valid_hit[queryset].max(0))
                    info=dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,scope=meta['scope'],stable=meta['stable'],
                        top1=chosen,query_set_size=len(queryset),raw_top1_dominant_identity=dominant[chosen],top1_min100_fallback=fallback[chosen],
                        parent_distribution=meta['distribution'])
                    if meta['stable']:
                        a=np.flatnonzero(localids==meta['target'])[0];rawscores=rawiou[:,a];qbest=int(rawscores.argmax());qset=max(queryset,key=lambda q:(rawscores[q],-q))
                        idx=meta['index'];qb=int(valid_hit[:,idx].argmax())
                        info.update(target=meta['target'],raw_top1_same_instance=dominant[chosen]==meta['target'],
                            raw_top1_target_iou=float(rawscores[chosen]),raw_oracle_target_iou=float(rawscores[qbest]),
                            raw_set_target_iou=float(rawscores[qset]),raw_oracle_query=qbest,raw_set_query=qset,
                            raw_top1_oracle_gap=float(rawscores[qbest]-rawscores[chosen]),raw_set_oracle_gap=float(rawscores[qbest]-rawscores[qset]),
                            probe_oracle_query=qb,probe_oracle_is_fallback=fallback[qb],
                            raw_oracle_neighbor_probability=float(p[(truth>=1000)&(truth!=meta['target']),qbest].mean()) if ((truth>=1000)&(truth!=meta['target'])).any() else None)
                    parentrows.append(info)
                    for rl in association['regions']:
                        rid=rl['region'];b=np.flatnonzero(labels==rid);v,n=np.unique(truth[b],return_counts=True);g=int(v[n.argmax()]);q=rl['top'][0]
                        regions.append(dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,region=rid,points=len(b),
                            GT_dominant_identity=g,GT_dominant_fraction=float(n.max()/len(b)),GT_pure_known=len(v)==1 and g>=1000,
                            associated_query=q,query_raw_halo_dominant_identity=dominant[q],association_ambiguous=rl['ambiguous'],
                            region_query_mean=float(p[b,q].mean()),region_query_std=float(p[b,q].std()),
                            dominant_identity_in_query=bool(g>=1000 and dominant[q]==g)))
        for mode,expanded in best.items():poolrows+=ev.pool_rows(s,mode,baseline,expanded,ids)
        print('supplement raw prior',s,flush=True)
    summary=dict(complete=True,posthoc_diagnostic_only=True,no_prediction_or_association_changed=True,raw_prior={},
                 probe_oracle_pools={m:ev.pool_summary([r for r in poolrows if r['mode']==m]) for m in {r['mode'] for r in poolrows}})
    for mode in ('P0','P1'):
        rr=[r for r in parentrows if r['prior']==mode and r['stable']]
        summary['raw_prior'][mode]=dict(stable_parents=len(rr),top1_same_instance=sum(r['raw_top1_same_instance'] for r in rr),
            oracle_IoU_ge50=sum(r['raw_oracle_target_iou']>=.5 for r in rr),oracle_IoU_ge75=sum(r['raw_oracle_target_iou']>=.75 for r in rr),
            top1_IoU_ge50=sum(r['raw_top1_target_iou']>=.5 for r in rr),top1_IoU_ge75=sum(r['raw_top1_target_iou']>=.75 for r in rr),
            raw_top1_loss_ge02=sum(r['raw_top1_oracle_gap']>=.02-1e-12 for r in rr),raw_set_loss_ge02=sum(r['raw_set_oracle_gap']>=.02-1e-12 for r in rr),
            mean_top1_IoU=float(np.mean([r['raw_top1_target_iou'] for r in rr])),mean_oracle_IoU=float(np.mean([r['raw_oracle_target_iou'] for r in rr])),
            mean_set_IoU=float(np.mean([r['raw_set_target_iou'] for r in rr])),
            top1_min100_fallback=sum(r['top1_min100_fallback'] for r in rr),probe_oracle_min100_fallback=sum(r['probe_oracle_is_fallback'] for r in rr))
    for name,rows in [('all_raw_queries',rawrows),('raw_parent_identity',parentrows),('region_association_identity',regions),('query_probe_oracle_targets',poolrows)]:ev.csv_write(d/(name+'.csv'),rows)
    run.seal(d/'summary.json',summary)
    run.stage_seal('supplement',list(d.glob('*')),GT_read=True,code={str(Path(__file__).resolve()):run.sha(Path(__file__).resolve())},
                   associations_unchanged=run.sha(run.OUT/'associations_frozen.json'))
    print(summary,flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Post-seal GT diagnostics, feasible constructions, and actual NT-pool metrics."""
from pathlib import Path
from collections import Counter
import argparse, csv, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_nt_region_identity as run
import fi1_nt_region_identity_core as core
OUT=run.OUT
THRESHOLDS=(.5,.75,.9,.95)


def csv_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def valid_targets(gt, classes):
    ids,sizes=np.unique(gt,return_counts=True)
    mask=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,classes)
    return ids[mask],sizes[mask]


def ious(points,gt,ids,sizes):
    if len(points)<100:return np.zeros(len(ids))
    labels,counts=np.unique(gt[points],return_counts=True);loc=np.searchsorted(labels,ids)
    hit=np.zeros(len(ids));ok=loc<len(labels);ok[ok]&=labels[loc[ok]]==ids[ok];hit[ok]=counts[loc[ok]]
    return hit/(len(points)+sizes-hit)


def identity(parent,gt,ids,sizes):
    before=ious(parent,gt,ids,sizes);idx=int(before.argmax()) if len(ids) and before.max()>0 else None
    gid=int(ids[idx]) if idx is not None else None;hit=int((gt[parent]==gid).sum()) if gid is not None else 0
    stable=gid is not None and hit>=20 and hit/len(parent)>=.6
    labels,counts=np.unique(gt[parent],return_counts=True);dom=int(labels[counts.argmax()]);fraction=float(counts.max()/len(parent))
    scope='stable' if stable else 'out_of_scope' if dom>=1000 and dom//1000 not in set(ids//1000) and fraction>=.6 else 'unannotated' if (gt[parent]<1000).mean()>=.5 else 'ambiguous'
    return dict(stable=stable,target=gid,index=idx,hits=hit,purity=hit/len(parent),scope=scope,
                distribution={str(g):int(n) for g,n in zip(labels,counts)},parent_iou=float(before[idx]) if idx is not None else 0)


def pool_base(nt,gt,ids,sizes):
    baseline=np.zeros(len(ids))
    for p in nt.values():baseline=np.maximum(baseline,ious(p,gt,ids,sizes))
    return baseline


def outcome(parent,p,gt,ids,sizes,meta,baseline):
    add=np.setdiff1d(p,parent);delete=np.setdiff1d(parent,p)
    row=dict(changed=bool(len(add)+len(delete)),added=len(add),deleted=len(delete),edits=len(add)+len(delete),
             stable=meta['stable'],scope=meta['scope'],parent_iou=meta['parent_iou'],target=meta['target'],
             parent_points=len(parent),result_points=len(p),unknown_edits=len(add)+len(delete))
    after=ious(p,gt,ids,sizes)
    if meta['stable']:
        g=meta['target'];idx=meta['index'];score=float(after[idx]);delta=score-meta['parent_iou']
        at=int((gt[add]==g).sum());dt=int((gt[delete]==g).sum())
        ao=int(((gt[add]>=1000)&(gt[add]!=g)).sum());do=int(((gt[delete]>=1000)&(gt[delete]!=g)).sum())
        row.update(repaired_iou=score,parent_delta=delta,added_target=at,added_other=ao,deleted_target=dt,deleted_other=do,
                   unknown_edits=len(add)+len(delete)-at-dt-ao-do,unknown_annotation_edits=len(add)+len(delete)-at-dt-ao-do,
                   changed_parent_target=meta['hits'] if row['changed'] else 0,
                   good_parent=meta['parent_iou']>=.9,good_loss=max(0,-delta) if meta['parent_iou']>=.9 else 0,
                   drift=bool(row['changed'] and len(ids) and int(ids[after.argmax()])!=g and (gt[p]==int(ids[after.argmax()])).sum()>=20),
                   pool_delta=score-float(baseline[idx]))
    else:row['unknown_identity_edits']=row['unknown_edits']
    return row,after


def summarize_rows(rows):
    m={k:sum(r.get(k,0) for r in rows) for k in ('added','deleted','edits','added_target','added_other','deleted_target','deleted_other',
        'unknown_edits','unknown_annotation_edits','unknown_identity_edits','changed_parent_target','parent_delta','drift','duplicate_original','duplicate_generated')}
    m.update(parents=len(rows),stable_parents=sum(r['stable'] for r in rows),actions=sum(r['changed'] for r in rows),
        action_scenes=len({r['scene'] for r in rows if r['changed']}),
        parent_improved_02=sum(r.get('parent_delta',0)>=.02-1e-12 for r in rows),
        parent_degraded_02=sum(r.get('parent_delta',0)<-.02-1e-12 for r in rows),
        good_parents=sum(r.get('good_parent',False) for r in rows),
        good_loss_gt_02=sum(r.get('good_loss',0)>.02+1e-12 for r in rows),
        good_loss_gt_10=sum(r.get('good_loss',0)>.1+1e-12 for r in rows),
        worst_good_loss=max((r.get('good_loss',0) for r in rows),default=0))
    for name,n,d in [('add_target_precision','added_target','added'),('delete_pollution_precision','deleted_other','deleted'),
                     ('unknown_fraction','unknown_edits','edits'),('target_loss','deleted_target','changed_parent_target')]:
        m[name]=m[n]/m[d] if m[d] else None
    losses=[r['good_loss'] for r in rows if r.get('good_parent')]
    m['good_loss_quantiles']=dict(zip(('min','median','p90','p95','max'),np.quantile(losses,[0,.5,.9,.95,1]).tolist())) if losses else {}
    return m


def pool_rows(scene,mode,baseline,expanded,ids):
    rows=[]
    for g,b,e in zip(ids,baseline,expanded):
        rows.append(dict(scene=scene,mode=mode,gt_id=int(g),baseline=float(b),expanded=float(e),delta=float(e-b),
            value=bool(e-b>=.02-1e-12 or any(b<=t<e for t in (.75,.9,.95))),
            gain02=bool(e-b>=.02-1e-12),**{f'cross_{t}':bool(b<=t<e) for t in THRESHOLDS}))
    return rows


def pool_summary(rows):
    return dict(targets=len(rows),pool_gain02=sum(r['gain02'] for r in rows),pool_value_targets=sum(r['value'] for r in rows),
        pool_value_scenes=len({r['scene'] for r in rows if r['value']}),pool_delta_sum=sum(r['delta'] for r in rows),
        **{f'new_cross_{t}':sum(r[f'cross_{t}'] for r in rows) for t in THRESHOLDS},
        **{f'baseline_coverage_{t}':sum(r['baseline']>t for r in rows) for t in THRESHOLDS},
        **{f'expanded_coverage_{t}':sum(r['expanded']>t for r in rows) for t in THRESHOLDS})


def load_gt(c,s):
    path=c['jobs'][s]['GT'];assert run.sha(path)==c['GT_inputs'][path]
    return np.loadtxt(path,dtype=np.int64)


def structure():
    c=run.read(OUT/'contract.json');assert run.read(OUT/'structure_frozen.json')['complete']
    run.verify(run.read(OUT/'structure_frozen.json')['outputs'])
    d=OUT/'structure_diagnosis';d.mkdir(exist_ok=True);regional=[];parents=[];targets=[];pointsets={};coverage=[]
    modes=('point_feasible','region_feasible','point_delete_feasible','region_delete_feasible','old96_delete_feasible')
    for s in c['scenes']:
        gt=load_gt(c,s);ids,sizes=valid_targets(gt,c['class_ids']);_,nt=run.load_nt(c['jobs'][s]);baseline=pool_base(nt,gt,ids,sizes)
        best={m:baseline.copy() for m in modes};scene_arrays={}
        with np.load(OUT/'structure'/s/'data.npz') as z,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank:
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];visible=z[cid+'_visible'];labels=z[cid+'_labels']
                order=z[cid+'_order'];offset=z[cid+'_offsets'];regions=[order[a:b] for a,b in zip(offset[:-1],offset[1:])]
                flat=bank[cid+'_block_indices'];off=bank[cid+'_block_offsets'];old=[flat[off[i]:off[i+1]] for i in bank[cid+'_budget96']]
                editable=np.isin(halo,parent)&(visible>=2);oldcover=np.zeros(len(halo),bool)
                for b in old:oldcover[b]=True
                meta=identity(parent,gt,ids,sizes)
                coverage.append(dict(scene=s,candidate_id=cid,source=r['source'],halo_points=len(halo),new_regions=len(regions),
                    old96_regions=len(old),new_halo_coverage=len(halo),old96_halo_coverage=int(oldcover.sum()),
                    common_parent_visible_domain=int(editable.sum()),old96_parent_visible_covered=int((editable&oldcover).sum()),
                    old96_uncovered_parent_visible=int((editable&~oldcover).sum()),stable=meta['stable'],scope=meta['scope'],
                    parent_identity_distribution=run.plain(meta['distribution'])))
                for rid,b in enumerate(regions):
                    vals,counts=np.unique(gt[halo[b]],return_counts=True);idx=counts.argmax();dominant=int(vals[idx]);pure=len(vals)==1 and dominant>=1000
                    regional.append(dict(scene=s,candidate_id=cid,source=r['source'],region=rid,points=len(b),
                        parent_points=int(np.isin(halo[b],parent).sum()),editable_points=int((visible[b]>=2).sum()),
                        dominant_identity=dominant,dominant_fraction=float(counts[idx]/len(b)),known_fraction=float((gt[halo[b]]>=1000).mean()),
                        pure_known=bool(pure),mixed_known=len(vals[vals>=1000])>1,has_unknown=bool((vals<1000).any()),
                        membership_vector=z[cid+'_membership'][b[0]].astype(int).tolist(),
                        identity_distribution={str(g):int(n) for g,n in zip(vals,counts)}))
                for mode in modes:
                    if not meta['stable']:p=parent.copy()
                    else:
                        reg=regions if mode.startswith('region') else old if mode.startswith('old96') else None
                        p=core.feasible(parent,halo,gt,meta['target'],visible,reg,deletion_only='delete' in mode)
                    scene_arrays[cid+'_'+mode]=p
                    row,after=outcome(parent,p,gt,ids,sizes,meta,baseline);row.update(scene=s,candidate_id=cid,source=r['source'],mode=mode)
                    parents.append(row);best[mode]=np.maximum(best[mode],after)
        path=d/(s+'_feasible.npz');np.savez_compressed(path,**scene_arrays);pointsets[str(path)]=run.sha(path)
        for m in modes:targets+=pool_rows(s,m,baseline,best[m],ids)
        run.stage('stage1_postseal_structure_diagnosis',scene=s)
    summary=dict(complete=True,post_hoc_feasible_not_optimum=True,modes={m:dict(**summarize_rows([r for r in parents if r['mode']==m]),
        **pool_summary([r for r in targets if r['mode']==m])) for m in modes},
        regions=len(regional),region_points=sum(r['points'] for r in regional),
        pure_known_regions=sum(r['pure_known'] for r in regional),pure_known_points=sum(r['points'] for r in regional if r['pure_known']),
        mixed_known_regions=sum(r['mixed_known'] for r in regional),mixed_known_points=sum(r['points'] for r in regional if r['mixed_known']),
        has_unknown_regions=sum(r['has_unknown'] for r in regional),tiny_le5_regions=sum(r['points']<=5 for r in regional),
        tiny_le5_points=sum(r['points'] for r in regional if r['points']<=5),
        region_size_quantiles=np.quantile([r['points'] for r in regional],[0,.25,.5,.75,.9,.95,.99,1]).tolist(),
        coverage=coverage)
    for name,rows in [('regions',regional),('parents',parents),('targets',targets),('coverage',coverage)]:csv_write(d/(name+'.csv'),rows)
    run.seal(d/'summary.json',summary)
    run.stage_seal('structure_diagnosis',list(d.glob('*')),GT_read=True,structure_seal_sha256=run.sha(OUT/'structure_frozen.json'))


def prior():
    c=run.read(OUT/'contract.json')
    for name in ('structure','priors','associations'):run.verify(run.read(OUT/(name+'_frozen.json'))['outputs'])
    d=OUT/'prior_diagnosis';d.mkdir(exist_ok=True)
    parents=[];targets=[];queries=[];links=[];mixed=[];scene_rows=[]
    modes=c['modes']
    for s in c['scenes']:
        gt=load_gt(c,s);ids,sizes=valid_targets(gt,c['class_ids']);_,nt=run.load_nt(c['jobs'][s]);baseline=pool_base(nt,gt,ids,sizes)
        original_keys={p.astype('<i8').tobytes() for p in nt.values()};newkeys={m:set() for m in modes};best={m:baseline.copy() for m in modes}
        linkage={(r['candidate_id'],r['prior']):r for r in run.read(OUT/'associations'/s/'links.json')}
        with np.load(OUT/'structure'/s/'data.npz') as z,np.load(OUT/'priors'/s/'data.npz') as ps,np.load(OUT/'associations'/s/'data.npz') as actions:
            roi=ps['point_ids']
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];meta=identity(parent,gt,ids,sizes)
                labels=z[cid+'_labels'];ix=np.searchsorted(roi,halo)
                for mode in modes:
                    p=actions[cid+'_'+mode];row,after=outcome(parent,p,gt,ids,sizes,meta,baseline)
                    code=p.astype('<i8').tobytes();row.update(scene=s,candidate_id=cid,source=r['source'],mode=mode,
                        duplicate_original=bool(row['changed'] and code in original_keys),duplicate_generated=bool(row['changed'] and code in newkeys[mode]))
                    newkeys[mode].add(code);parents.append(row);best[mode]=np.maximum(best[mode],after)
                for mode in ('P0','P1'):
                    probability=ps[mode+'_probability'][ix];link=linkage[cid,mode];top=link['parent']['top'][0];queryset=link['query_set']
                    all_iou=[];dominants=[]; query_points=[]
                    for q in range(probability.shape[1]):
                        p=core.probe(parent,halo,probability[:,q],z[cid+'_visible']);score=ious(p,gt,ids,sizes);all_iou.append(score)
                        dom=int(ids[score.argmax()]) if len(ids) and score.max()>0 else None;dominants.append(dom)
                        query_points.append(len(p))
                        queries.append(dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,query=q,
                            full_soft_volume=float(ps[mode+'_full_volume'][q]),full_hard_volume=int(ps[mode+'_hard_volume'][q]),
                            local_probe_points=len(p),dominant_valid_instance=dom,dominant_iou=float(score.max(initial=0)),
                            stable_parent_target_iou=float(score[meta['index']]) if meta['stable'] else None,
                            selected_top1=q==top,in_nonGT_set=q in queryset))
                    all_iou=np.array(all_iou)
                    info=dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,stable=meta['stable'],scope=meta['scope'],
                        selected_query=top,selected_soft_IoU=link['parent']['scores'][0],gap=link['parent']['gap'],ratio=link['parent']['ratio'],
                        ambiguous=link['parent']['ambiguous'],query_set_size=len(queryset),parent_identity_distribution=meta['distribution'])
                    if meta['stable']:
                        idx=meta['index'];scores=all_iou[:,idx];qbest=int(np.argmax(scores));sbest=max(queryset,key=lambda q:(scores[q],-q))
                        info.update(target=meta['target'],parent_iou=meta['parent_iou'],selected_target_iou=float(scores[top]),
                            selected_dominant_instance=dominants[top],selected_same_instance=dominants[top]==meta['target'],
                            oracle_best_query=qbest,oracle_target_iou=float(scores[qbest]),nonGT_set_best_query=sbest,
                            nonGT_set_target_iou=float(scores[sbest]),oracle_gain_over_parent=float(scores[qbest]-meta['parent_iou']),
                            top1_oracle_gap=float(scores[qbest]-scores[top]),set_oracle_gap=float(scores[qbest]-scores[sbest]))
                    else:
                        for gid,num in meta['distribution'].items():
                            gid=int(gid)
                            if gid<1000:continue
                            where=np.flatnonzero(ids==gid)
                            mixed.append(dict(scene=s,candidate_id=cid,prior=mode,instance=gid,parent_points=num,
                                valid_eval_instance=bool(len(where)),
                                best_query=int(all_iou[:,where[0]].argmax()) if len(where) else None,
                                best_local_probe_iou=float(all_iou[:,where[0]].max()) if len(where) else None,
                                nonGT_set_best_iou=float(all_iou[queryset,where[0]].max()) if len(where) else None,
                                top1_iou=float(all_iou[top,where[0]]) if len(where) else None))
                    links.append(info)
        for mode in modes:
            rr=pool_rows(s,mode,baseline,best[mode],ids);targets+=rr
            scene_rows.append(dict(scene=s,mode=mode,**summarize_rows([r for r in parents if r['scene']==s and r['mode']==mode]),**pool_summary(rr)))
        run.stage('stage3_postseal_prior_diagnosis',scene=s)
    summaries={m:dict(**summarize_rows([r for r in parents if r['mode']==m]),**pool_summary([r for r in targets if r['mode']==m])) for m in modes}
    association={}
    for mode in ('P0','P1'):
        rr=[r for r in links if r['prior']==mode];stable=[r for r in rr if r['stable']]
        association[mode]=dict(parents=len(rr),stable=len(stable),ambiguous=sum(r['ambiguous'] for r in rr),
            same_instance=sum(r['selected_same_instance'] for r in stable),
            oracle_gain02=sum(r['oracle_gain_over_parent']>=.02-1e-12 for r in stable),
            oracle_iou_ge50=sum(r['oracle_target_iou']>=.5 for r in stable),
            oracle_iou_ge75=sum(r['oracle_target_iou']>=.75 for r in stable),
            top1_loss_ge02=sum(r['top1_oracle_gap']>=.02-1e-12 for r in stable),
            set_loss_ge02=sum(r['set_oracle_gap']>=.02-1e-12 for r in stable),
            mean_top1_iou=float(np.mean([r['selected_target_iou'] for r in stable])),
            mean_oracle_iou=float(np.mean([r['oracle_target_iou'] for r in stable])),
            mean_set_iou=float(np.mean([r['nonGT_set_target_iou'] for r in stable])))
    for name,rows in [('parents',parents),('targets',targets),('queries',queries),('association_identity',links),('mixed_parent_identities',mixed),('scenes',scene_rows)]:csv_write(d/(name+'.csv'),rows)
    summary=dict(complete=True,modes=summaries,association=association,
        per_source={source:{m:summarize_rows([r for r in parents if r['source']==source and r['mode']==m]) for m in modes} for source in ('native','track')},
        scenes=scene_rows,paired_interfaces={s:run.read(OUT/'priors'/s/'receipt.json')['paired'] for s in c['scenes']},
        diagnostic_only=True,all_query_oracle_is_posthoc_local_visible_readout_not_full_instance_AP=True,
        mixed_parent_identities=mixed)
    run.seal(d/'summary.json',summary);run.stage_seal('prior_diagnosis',list(d.glob('*')),GT_read=True,
        prior_sha256=run.sha(OUT/'priors_frozen.json'),association_sha256=run.sha(OUT/'associations_frozen.json'))


def report():
    c=run.read(OUT/'contract.json');structure=run.read(OUT/'structure_diagnosis/summary.json');prior=run.read(OUT/'prior_diagnosis/summary.json')
    summary=dict(complete=True,stages_completed=[0,1,2,3,4],scenes=c['scenes'],scope='fit96 hash-selected12; exposed training diagnostic',
        structure=structure,prior=prior,training=False,AP_run=False,SL_modified=False)
    run.seal(OUT/'summary.json',summary)
    # Conclusions are authored after evidence review; do not invent a success gate here.
    run.stage('stage4_evidence_ready',modes=prior['modes'],association=prior['association'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['structure','prior','report']);args=p.parse_args();globals()[args.mode]()

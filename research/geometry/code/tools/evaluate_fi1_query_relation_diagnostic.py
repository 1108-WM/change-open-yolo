#!/usr/bin/env python3
"""Post-seal relation labels, negative examples and local information-supply references."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import combinations
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import rankdata
import run_fi1_query_relation_diagnostic as run
import evaluate_fi1_nt_region_identity as met
OUT=run.OUT;read=run.read;seal=run.seal;sha=run.sha

def counts(points,gt,target):
    labels,n=np.unique(gt[points],return_counts=True);known=labels>=1000;total=len(points);targetpoints=int(n[labels==target].sum()) if target is not None else 0;other=int(n[known].sum())-targetpoints;unknown=total-targetpoints-other;knownn=total-unknown
    dominant=int(labels[known][n[known].argmax()]) if known.any() else None;domhits=int(n[labels==dominant].sum()) if dominant is not None else 0
    category='unknown'
    if total>=20 and knownn/total>=.9:
        if targetpoints/knownn>=.9:category='same_target_local'
        elif dominant is not None and dominant!=target and domhits/knownn>=.9 and domhits>=20:category='neighbor_local'
        else:category='mixed_local'
    return dict(points=total,target=targetpoints,other=other,unknown=unknown,known_fraction=knownn/total if total else None,target_known_fraction=targetpoints/knownn if knownn else None,dominant_known=dominant,dominant_known_points=domhits,category=category,distribution={str(g):int(v) for g,v in zip(labels,n)})

def evaluate():
    c=read(OUT/'contract.json');frozen=read(OUT/'relations_frozen.json');assert frozen['complete'];run.diag.verify(frozen['outputs']);run.diag.verify(c['GT_inputs']);outfiles={};allparents=[];pairlabels=[];regionlabels=[];queries=[];references=[];allpool=[]
    for si,(s,j) in enumerate(c['jobs'].items()):
        run.stage('postseal_GT_relations',scene=s,completed=si,total=52);root=Path(j['asset_root']);gt=np.loadtxt(j['GT'],dtype=np.int64);ids,sizes=met.valid_targets(gt,c['class_ids']);_,nt=run.diag.load_nt(j);baseline=met.pool_base(nt,gt,ids,sizes);d=OUT/'relations'/s;parentrows=read(d/'parents.json');bips=read(d/'query_regions.json');pairs=read(d/'query_pairs.json');scene_parents=[];scene_queries=[];scene_pairs=[];scene_regions=[];scene_refs=[];best={k:baseline.copy() for k in ('pure_regions','point_optimum','candidate_safe_regions','all150_safe_regions','best_candidate_single','best_all150_single','best_candidate_anchor_pair','all_candidate_union')}
        with np.load(d/'data.npz') as z,np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as pr:
            roi=pr['point_ids'];fullhard=pr['P1_hard_volume'];prob=pr['P1_probability']
            for p in parentrows:
                cid=p['candidate_id'];halo=z[cid+'_halo'];parent=z[cid+'_parent'];qs=z[cid+'_queries'];member=np.isin(halo,parent);visible=st[cid+'_visible']>=2;labels=st[cid+'_labels'];editable=(~member)&visible;meta=met.identity(parent,gt,ids,sizes);target=meta['target'] if meta['stable'] else None;idx=meta['index'];ix=np.searchsorted(roi,halo);hp=prob[ix];hard=hp>=.5;selectedhard=hard[:,qs];assert np.array_equal(selectedhard,np.unpackbits(z[cid+'_hard_masks'],axis=0,bitorder='little')[:len(halo)])
                qstats={};qrecord={}
                for q in range(150):
                    hc=counts(halo[hard[:,q]],gt,target);rc=counts(roi[prob[:,q]>=.5],gt,target);unobserved=int(fullhard[q]) - rc['points'];assert unobserved>=0
                    row=dict(scene=s,cohort=j['cohort'],candidate_id=cid,query=q,is_candidate=q in qs,is_anchor=q==p['anchor'],stable_parent=meta['stable'],parent_target=target,halo=hc,ROI=rc,full_scene_hard_volume=int(fullhard[q]),full_scene_soft_volume=float(pr['P1_full_volume'][q]),outside_stored_ROI_hard_points=unobserved,full_scene_composition_unavailable=unobserved>0,global_target_precision_upper_bound=(rc['target']+unobserved)/int(fullhard[q]) if fullhard[q] else None,global_known_other_lower_bound=rc['other'],parent_hard_coverage=float(hard[member,q].mean()),parent_soft_mean=float(hp[member,q].mean()),editable_hard_points=int(hard[editable,q].sum()))
                    qstats[q]=hc;qrecord[q]=row
                    # All150 is a posthoc supply reference; not the GT-free candidate set.
                    scene_queries.append(row)
                for r in [r for r in pairs if r['candidate_id']==cid]:
                    a,b=r['query_a'],r['query_b'];ha,hb=hard[:,a],hard[:,b];ac,bc=qstats[a],qstats[b];ax=counts(halo[ha&~hb],gt,target);bx=counts(halo[hb&~ha],gt,target);combined=counts(halo[ha|hb],gt,target);rel='unknown'
                    if meta['stable']:
                        if ac['category']=='same_target_local' and bc['category']=='same_target_local':
                            if r['hard_iou'] is not None and r['hard_iou']>=.5:rel='duplicate_same_target_local'
                            elif ax['target']>0 and bx['target']>0:rel='complement_same_target_local'
                            else:rel='contained_same_target_local'
                        elif {ac['category'],bc['category']}=={'same_target_local','neighbor_local'}:rel='true_neighbor_local'
                        elif ac['category']=='neighbor_local' and bc['category']=='neighbor_local':rel='neighbor_duplicate_local' if ac['dominant_known']==bc['dominant_known'] else 'different_neighbors_local'
                        elif ac['category']=='mixed_local' or bc['category']=='mixed_local':rel='mixed_local'
                    local_parent_shared_target=ax['target']>0 and bx['target']>0
                    # No global identity assertion from halo purity, explicit outside/mixed ROI counts.
                    scene_pairs.append(dict(r,stable_parent=meta['stable'],parent_target=target,relation=rel,a_halo=ac,b_halo=bc,a_exclusive_GT=ax,b_exclusive_GT=bx,union_GT=combined,a_ROI=qrecord[a]['ROI'],b_ROI=qrecord[b]['ROI'],a_unobserved_full_points=qrecord[a]['outside_stored_ROI_hard_points'],b_unobserved_full_points=qrecord[b]['outside_stored_ROI_hard_points'],same_target_exclusive_points_both=local_parent_shared_target,global_identity_claim=False))
                parent_bips=[r for r in bips if r['candidate_id']==cid]
                for r in parent_bips:
                    block=z[f"{cid}_r{r['region']:04d}_block"];fullregion=halo[labels==r['region']];bc=counts(block,gt,target);fc=counts(fullregion,gt,target);relation='unknown'
                    if meta['stable'] and len(block):
                        if bc['target']==len(block):relation='pure_target_block'
                        elif bc['other']==len(block):relation='pure_other_block'
                        elif bc['unknown']==len(block):relation='unannotated_block'
                        else:relation='mixed_block'
                    scene_regions.append(dict(r,stable_parent=meta['stable'],parent_target=target,block_GT=bc,full_region_GT=fc,block_relation=relation,query_halo_category=qstats[r['query']]['category'],query_ROI=qrecord[r['query']]['ROI']))
                prrow=dict(p,**meta)
                if meta['stable']:
                    supplied=[z[f'{cid}_r{rid:04d}_block'] for rid in p['regions']];pure=[b for b in supplied if np.all(gt[b]==target)];correct=np.unique(np.concatenate([b[gt[b]==target] for b in supplied])) if supplied else np.empty(0,np.int64);purepts=np.unique(np.concatenate(pure)) if pure else np.empty(0,np.int64)
                    m=z[cid+'_region_means_all150'];qcover=hard[member].mean(0);safe_blocks=[i for i,b in enumerate(supplied) if np.all(gt[b]==target)];candidate_supported=[i for i in safe_blocks if np.any(m[i,qs]>=.5)];all_supported=[i for i in safe_blocks if np.any(m[i]>=.5)]
                    def blockunion(inds):return np.unique(np.concatenate([supplied[i] for i in inds])) if len(inds) else np.empty(0,np.int64)
                    purev=met.ious(np.union1d(parent,purepts),gt,ids,sizes);pointv=met.ious(np.union1d(parent,correct),gt,ids,sizes);candv=met.ious(np.union1d(parent,blockunion(candidate_supported)),gt,ids,sizes);allv=met.ious(np.union1d(parent,blockunion(all_supported)),gt,ids,sizes)
                    # Actual local hard readout obeys original editable domain; may pollute.
                    queryresults={q:met.ious(np.union1d(parent,halo[editable&hard[:,q]]),gt,ids,sizes) for q in range(150)};singles=np.array([queryresults[q][idx] for q in range(150)]);bestall=int(np.lexsort((np.arange(150),-singles))[0]);bestq=int(qs[np.lexsort((qs,-singles[qs]))[0]])
                    anchor=p['anchor'];pairvals={int(q):met.ious(np.union1d(parent,halo[editable&(hard[:,anchor]|hard[:,q])]),gt,ids,sizes) for q in qs};pairq=min(pairvals,key=lambda q:(-pairvals[q][idx],q));allcandv=met.ious(np.union1d(parent,halo[editable&selectedhard.any(1)]),gt,ids,sizes)
                    vals={'pure_regions':purev,'point_optimum':pointv,'candidate_safe_regions':candv,'all150_safe_regions':allv,'best_candidate_single':queryresults[bestq],'best_all150_single':queryresults[bestall],'best_candidate_anchor_pair':pairvals[pairq],'all_candidate_union':allcandv}
                    for k,v in vals.items():best[k][idx]=max(best[k][idx],v[idx])
                    query_association_single=[];query_joint=[]
                    for bi,b in enumerate(supplied):
                        if not np.all(gt[b]==target):continue
                        supported=np.flatnonzero(m[bi]>=.5);joint=[int(q) for q in supported if qcover[q]>=.5];candjoint=[q for q in joint if q in qs];supportcandidate=[int(q) for q in supported if q in qs]
                        query_association_single.extend(candjoint);query_joint.append(dict(region=p['regions'][bi],points=len(b),anchor_mean=float(m[bi,anchor]),candidate_support_queries=supportcandidate,all150_support_queries=supported.tolist(),candidate_single_explains_parent_and_block=candjoint,all150_single_explains_parent_and_block=joint,multiquery_needed_by_half_support=bool(qcover[anchor]>=.5 and len(supportcandidate)>0 and not candjoint)))
                    ref=dict(scene=s,cohort=j['cohort'],candidate_id=cid,target=target,parent_iou=meta['parent_iou'],original_NT_best=float(baseline[idx]),regions=p['regions'],pure_target_blocks=len(safe_blocks),candidate_supported_pure_blocks=len(candidate_supported),all150_supported_pure_blocks=len(all_supported),pure_target_blocks_no_candidate_support=[p['regions'][i] for i in safe_blocks if i not in candidate_supported],best_candidate_query=bestq,best_all150_query=bestall,best_anchor_pair_query=pairq,all150_best_in_candidates=bestall in qs,block_single_vs_multiple=query_joint,posthoc_GT_only=True,no_generated_candidate=True)
                    for k,v in vals.items():ref[k+'_iou']=float(v[idx]);ref[k+'_gain_over_NT']=float(v[idx]-baseline[idx]);ref[k+'_gain02']=bool(v[idx]-baseline[idx]>=.02-1e-12)
                    scene_refs.append(ref);prrow.update(candidate_single_iou=float(singles[bestq]),all150_single_iou=float(singles[bestall]),candidate_pair_iou=float(pairvals[pairq][idx]))
                scene_parents.append(prrow)
        ed=OUT/'evaluation'/s;ed.mkdir(parents=True,exist_ok=True)
        for name,rr in [('parents',scene_parents),('queries_all150',scene_queries),('pairs',scene_pairs),('query_regions',scene_regions),('posthoc_supply',scene_refs)]:seal(ed/(name+'.json'),rr);outfiles[str(ed/(name+'.json'))]=sha(ed/(name+'.json'))
        pool=[]
        for mode,bb in best.items():pool.extend(met.pool_rows(s,mode,baseline,bb,ids))
        seal(ed/'pool.json',pool);outfiles[str(ed/'pool.json')]=sha(ed/'pool.json');allpool+=pool;allparents+=scene_parents;queries+=scene_queries;pairlabels+=scene_pairs;regionlabels+=scene_regions;references+=scene_refs
    summary=dict(parents=len(allparents),stable_parents=sum(r['stable'] for r in allparents),pair_categories=dict(Counter(r['relation'] for r in pairlabels)),pair_category_scenes={k:len({r['scene'] for r in pairlabels if r['relation']==k}) for k in {r['relation'] for r in pairlabels}},candidate_queries=sum(r['is_candidate'] for r in queries),regions=len({(r['scene'],r['candidate_id'],r['region']) for r in regionlabels}),query_region_rows=len(regionlabels),posthoc_pool={mode:met.pool_summary([r for r in allpool if r['mode']==mode]) for mode in sorted({r['mode'] for r in allpool})},ROI_identity_only=True,unknown_kept=True)
    seal(OUT/'evaluation_summary.json',summary);outfiles[str(OUT/'evaluation_summary.json')]=sha(OUT/'evaluation_summary.json');seal(OUT/'evaluation_frozen.json',dict(complete=True,outputs=outfiles,relations_seal_sha256=sha(OUT/'relations_frozen.json'),GT_after_all_relations=True,evaluation_code_sha256=sha(Path(__file__))));run.stage('GT_relation_diagnosis_completed',summary=summary)

def auc(y,x):
    y=np.array(y,bool);x=np.array(x,float)
    if not len(y) or not y.any() or y.all():return None
    n=y.sum();r=rankdata(x);return float((r[y].sum()-n*(n+1)/2)/(n*(len(y)-n)))
def compare(rows,feature,labelkey,pos,neg,stratum):
    rr=[r for r in rows if r[labelkey] in pos|neg and r.get(feature) is not None and np.isfinite(r[feature])];ys=[r[labelkey] in pos for r in rr];xx=[r[feature] for r in rr];scenes=sorted({r['scene'] for r in rr});macro=[];bs=[]
    for s in scenes:
        sub=[r for r in rr if r['scene']==s];v=auc([r[labelkey] in pos for r in sub],[r[feature] for r in sub]);
        if v is not None:macro.append(v)
    rng=np.random.default_rng(20261014);by={s:[r for r in rr if r['scene']==s] for s in scenes}
    for _ in range(200):
        if not scenes:break
        selection=rng.choice(scenes,len(scenes));sub=[r for s in selection for r in by[s]];v=auc([r[labelkey] in pos for r in sub],[r[feature] for r in sub]);
        if v is not None:bs.append(v)
    return dict(stratum=stratum,feature=feature,positive=sum(ys),negative=len(ys)-sum(ys),scene_count=len(scenes),scene_count_both_labels=len(macro),auc=auc(ys,xx),macro_scene_auc=float(np.mean(macro)) if macro else None,bootstrap_p025=float(np.quantile(bs,.025)) if bs else None,bootstrap_p975=float(np.quantile(bs,.975)) if bs else None,positive_mean=float(np.mean([x for x,y in zip(xx,ys) if y])) if any(ys) else None,negative_mean=float(np.mean([x for x,y in zip(xx,ys) if not y])) if ys and not all(ys) else None,missing_or_excluded=len(rows)-len(rr))

def analyze():
    c=read(OUT/'contract.json');run.diag.verify(read(OUT/'evaluation_frozen.json')['outputs']);pairs=[];bips=[];refs=[];parents=[]
    for s in c['scenes']:
        d=OUT/'evaluation'/s;pairs+=read(d/'pairs.json');bips+=read(d/'query_regions.json');refs+=read(d/'posthoc_supply.json');parents+=read(d/'parents.json')
    pos={'complement_same_target_local'};neg={'true_neighbor_local'};features=['hard_iou','soft_iou','a_containment','b_containment','parent_union_coverage','minimum_distance','centroid_distance','region_contact_edges','shared_regions','common_frames','conflict_frames','valid_frames'];comparisons=[]
    for r in pairs:
        r['max_parent_coverage']=max(r['parent_a_coverage'],r['parent_b_coverage']);r['min_parent_coverage']=min(r['parent_a_coverage'],r['parent_b_coverage']);r['volume_ratio']=min(r['a_full_volume'],r['b_full_volume'])/max(r['a_full_volume'],r['b_full_volume'],1e-12);r['view_direction']=r['common_frames']-r['conflict_frames'] if r['valid_frames'] else None
    features+=['max_parent_coverage','min_parent_coverage','volume_ratio','view_direction']
    strata={'all':pairs,'anchor_pairs':[r for r in pairs if r['has_anchor']],'no_overlap':[r for r in pairs if r['no_overlap']],'small_support':[r for r in pairs if r['small_support']],'non_small_support':[r for r in pairs if not r['small_support']]}
    for cohort in c['cohorts']:strata[cohort]=[r for r in pairs if r['cohort']==cohort]
    for name,rr in strata.items():
        for feature in features:comparisons.append(compare(rr,feature,'relation',pos,neg,name))
    # Duplicate comparisons separate, so overlap cannot disguise fragment recall.
    duplicatecomparisons=[compare(pairs,f,'relation',{'duplicate_same_target_local'},neg,'duplicates_vs_neighbor') for f in features]
    regionfeatures=['query_mean','query_parent_mean','hard_parent_fraction','hard_region_fraction','hard_halo_fraction_of_full','local_common_frames','local_conflict_frames','local_valid_frames','whole_common_frames','whole_conflict_frames']
    regioncomparisons=[]
    for cohort in ['all']+list(c['cohorts']):
        rr=[r for r in bips if r['query_mean']>=.5 and (cohort=='all' or r['cohort']==cohort)]
        for f in regionfeatures:regioncomparisons.append(compare(rr,f,'block_relation',{'pure_target_block'},{'pure_other_block'},cohort+'/qmean>=.5'))
    bins=[]
    for kind in sorted({r['relation'] for r in pairs}):
        rr=[r for r in pairs if r['relation']==kind];bins.append(dict(relation=kind,pairs=len(rr),scenes=len({r['scene'] for r in rr}),no_overlap=sum(r['no_overlap'] for r in rr),small_support=sum(r['small_support'] for r in rr),no_valid_frames=sum(r['valid_frames']==0 for r in rr),any_common=sum(r['common_frames']>0 for r in rr),any_conflict=sum(r['conflict_frames']>0 for r in rr),global_ROI_known_other=sum(r['a_ROI']['other']+r['b_ROI']['other'] for r in rr),global_unobserved_hard_points=sum(r['a_unobserved_full_points']+r['b_unobserved_full_points'] for r in rr)))
    opportunities=[]
    for r in refs:
        if r['pure_regions_gain02'] or r['point_optimum_gain02'] or r['best_all150_single_gain02'] or r['best_candidate_anchor_pair_gain02']:opportunities.append(r)
    # Preserve necessary true-neighbor negative records together with positives,
    # scene-balanced deterministic representative sampling for readable examples.
    examples=[]
    for kind in ('complement_same_target_local','true_neighbor_local','duplicate_same_target_local','mixed_local','unknown'):
        for s in c['scenes']:
            rr=[r for r in pairs if r['scene']==s and r['relation']==kind];rr.sort(key=lambda r:(not r['has_anchor'],r['candidate_id'],r['query_a'],r['query_b']));examples+=rr[:2]
    d=OUT/'analysis';d.mkdir(exist_ok=True)
    for name,rr in [('pair_comparisons',comparisons),('duplicate_comparisons',duplicatecomparisons),('region_comparisons',regioncomparisons),('category_bins',bins),('pool_opportunities',opportunities),('balanced_examples',examples)]:seal(d/(name+'.json'),rr);met.csv_write(d/(name+'.csv'),[r for r in rr if name not in ('balanced_examples','pool_opportunities')])
    summary=read(OUT/'evaluation_summary.json');supported=[r for r in comparisons if r['stratum']=='anchor_pairs' and r['macro_scene_auc'] is not None and max(r['macro_scene_auc'],1-r['macro_scene_auc'])>=.7 and r['scene_count_both_labels']>=2];summary['candidate_evidence_signals_exploratory']=supported;summary['global_safety_not_established']=True
    seal(d/'summary.json',summary);seal(OUT/'analysis_frozen.json',dict(complete=True,outputs={str(p):sha(p) for p in d.iterdir()},code_sha256=sha(Path(__file__)),no_fitted_grouping_rule=True));run.stage('relation_analysis_completed',pair_counts=summary['pair_categories'],posthoc_pool=summary['posthoc_pool'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('evaluate','analyze'));a=p.parse_args();globals()[a.mode]()

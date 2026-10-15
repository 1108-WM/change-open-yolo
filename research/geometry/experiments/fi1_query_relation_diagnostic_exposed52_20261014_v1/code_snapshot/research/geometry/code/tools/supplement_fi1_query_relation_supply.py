#!/usr/bin/env python3
"""Independent same-parent supply IoUs and evidence attached to missed blocks."""
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_query_relation_diagnostic as run
OUT=run.OUT;read=run.read;seal=run.seal;sha=run.sha

def independent_iou(point,gt,gid,size):
    if len(point)<100:return 0.
    inter=Counter(map(int,gt[point])).get(gid,0);return inter/(len(point)+size-inter)
def main():
    c=read(OUT/'contract.json');run.diag.verify(read(OUT/'evaluation_frozen.json')['outputs']);totals=Counter();evidence=[];cohortcounts={};poolcheck={}
    for s,j in c['jobs'].items():
        print('independent supply',s,flush=True);root=Path(j['asset_root']);gt=np.loadtxt(j['GT'],dtype=np.int64);hist=Counter(map(int,gt));ids=sorted(g for g,n in hist.items() if g>=1000 and n>=100 and g//1000 in c['class_ids']);_,nt=run.diag.load_nt(j);baseline={g:max((independent_iou(p,gt,g,hist[g]) for p in nt.values()),default=0.) for g in ids};best={mode:dict(baseline) for mode in ('pure_regions','point_optimum','candidate_safe_regions','all150_safe_regions','best_candidate_single','best_all150_single','best_candidate_anchor_pair','all_candidate_union')};parentrows={r['candidate_id']:r for r in read(OUT/'relations'/s/'parents.json')};refs=read(OUT/'evaluation'/s/'posthoc_supply.json');pairrows=read(OUT/'evaluation'/s/'pairs.json');bips=read(OUT/'evaluation'/s/'query_regions.json');qrows=read(OUT/'evaluation'/s/'queries_all150.json');qby={(r['candidate_id'],r['query']):r for r in qrows}
        with np.load(OUT/'relations'/s/'data.npz') as z,np.load(root/'structure'/s/'data.npz') as st,np.load(root/'priors'/s/'data.npz') as pr:
            for r in refs:
                cid=r['candidate_id'];gid=r['target'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];p=pr['P1_probability'][np.searchsorted(pr['point_ids'],halo)];hard=p>=.5;editable=(~np.isin(halo,parent))&(st[cid+'_visible']>=2);qs=z[cid+'_queries'];anchor=parentrows[cid]['anchor'];blocks=[z[f'{cid}_r{rid:04d}_block'] for rid in r['regions']];means=z[cid+'_region_means_all150'];pure=[i for i,b in enumerate(blocks) if np.all(gt[b]==gid)];cand=[i for i in pure if (means[i,qs]>=.5).any()];allq=[i for i in pure if (means[i]>=.5).any()]
                def union(indices):return np.unique(np.concatenate([blocks[i] for i in indices])) if len(indices) else np.empty(0,np.int64)
                targetpts=np.unique(np.concatenate([b[gt[b]==gid] for b in blocks])) if blocks else np.empty(0,np.int64);sets={'pure_regions':np.union1d(parent,union(pure)),'candidate_safe_regions':np.union1d(parent,union(cand)),'all150_safe_regions':np.union1d(parent,union(allq)),'point_optimum':np.union1d(parent,targetpts)}
                single=[independent_iou(np.union1d(parent,halo[editable&hard[:,q]]),gt,gid,hist[gid]) for q in range(150)];bestall=min(range(150),key=lambda q:(-single[q],q));bestq=min(qs,key=lambda q:(-single[q],q));assert bestq==r['best_candidate_query'] and bestall==r['best_all150_query'];pairvals={int(q):independent_iou(np.union1d(parent,halo[editable&(hard[:,anchor]|hard[:,q])]),gt,gid,hist[gid]) for q in qs};pairq=min(pairvals,key=lambda q:(-pairvals[q],q));assert pairq==r['best_anchor_pair_query'];sets['all_candidate_union']=np.union1d(parent,halo[editable&hard[:,qs].any(1)]);values={mode:independent_iou(v,gt,gid,hist[gid]) for mode,v in sets.items()};values.update(best_candidate_single=single[bestq],best_all150_single=single[bestall],best_candidate_anchor_pair=pairvals[pairq]);assert abs(r['original_NT_best']-baseline[gid])<1e-12
                for mode,value in values.items():assert abs(r[mode+'_iou']-value)<1e-12;best[mode][gid]=max(best[mode][gid],value);totals['independent_parent_mode_ious']+=1
                if r['pure_regions_gain02'] or r['point_optimum_gain02'] or r['best_candidate_single_gain02']:
                    relevant=[]
                    for bi in pure:
                        if means[bi,anchor]>=.5:continue
                        rid=r['regions'][bi];support=[q for q in qs if means[bi,q]>=.5];regions=[rr for rr in bips if rr['candidate_id']==cid and rr['region']==rid and rr['query'] in support];relations=[rr for rr in pairrows if rr['candidate_id']==cid and rr['has_anchor'] and (rr['query_a'] in support or rr['query_b'] in support)];relevant.append(dict(region=rid,points=len(blocks[bi]),anchor_mean=float(means[bi,anchor]),support_queries=list(map(int,support)),support_edges=regions,anchor_pairs=relations))
                    newq=int(bestq);assoc='original_anchor_best' if newq==anchor else 'alternate_single_query';evidence.append(dict(scene=s,cohort=j['cohort'],candidate_id=cid,target=gid,anchor=anchor,best_candidate_query=newq,interpretation=assoc,best_candidate_roi=qby[cid,newq]['ROI'],best_candidate_outside_ROI=qby[cid,newq]['outside_stored_ROI_hard_points'],best_candidate_local=qby[cid,newq]['halo'],single_gain_over_NT=r['best_candidate_single_gain_over_NT'],pair_gain_over_NT=r['best_candidate_anchor_pair_gain_over_NT'],pure_gain_over_NT=r['pure_regions_gain_over_NT'],point_gain_over_NT=r['point_optimum_gain_over_NT'],omitted_pure_block_evidence=relevant,all_candidate_union_gain_over_NT=r['all_candidate_union_gain_over_NT'],posthoc_only=True))
        pooled=read(OUT/'evaluation'/s/'pool.json')
        for rr in pooled:
            assert abs(rr['baseline']-baseline[rr['gt_id']])<1e-12 and abs(rr['expanded']-best[rr['mode']][rr['gt_id']])<1e-12;totals['independent_pool_target_modes']+=1
        poolcheck[s]={mode:sum(best[mode][g]-baseline[g]>=.02-1e-12 for g in ids) for mode in best}
    d=OUT/'supplement';d.mkdir(exist_ok=True);seal(d/'opportunity_evidence.json',evidence);seal(d/'pool_recomputed.json',poolcheck);seal(OUT/'supply_verification.json',dict(complete=True,counts=dict(totals),evidence_outputs={str(p):sha(p) for p in d.iterdir()},code_sha256=sha(Path(__file__)),all_references_GT_posthoc=True,new_rules=False));print(dict(totals),flush=True)
if __name__=='__main__':main()

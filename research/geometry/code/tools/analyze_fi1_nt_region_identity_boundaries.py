#!/usr/bin/env python3
"""Post-hoc boundary failure localization, without any new inference rules."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_nt_region_identity as run
import evaluate_fi1_nt_region_identity as ev


def main():
    c=run.read(run.OUT/'contract.json');rows=[];mixed=[];oldregions=[];opportunity=[];edges=[]
    d=run.OUT/'boundary_diagnosis';d.mkdir(exist_ok=True)
    for s in c['scenes']:
        gt=ev.load_gt(c,s);ids,sizes=ev.valid_targets(gt,c['class_ids'])
        linkage={(r['candidate_id'],r['prior']):r for r in run.read(run.OUT/'associations'/s/'links.json')}
        with np.load(run.OUT/'structure'/s/'data.npz') as z,np.load(run.OUT/'priors'/s/'data.npz') as ps,\
             np.load(run.REGIONS/'regions'/s/'regions.npz') as bank:
            roi=ps['point_ids'];sparse_row=ps['original_to_sparse_row']
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];labels=z[cid+'_labels'];visible=z[cid+'_visible']>=2
                truth=gt[halo];member=np.isin(halo,parent);meta=ev.identity(parent,gt,ids,sizes)
                flat=bank[cid+'_block_indices'];off=bank[cid+'_block_offsets'];old=[flat[off[i]:off[i+1]] for i in bank[cid+'_budget96']]
                for rid,b in zip(bank[cid+'_budget96'],old):
                    b=b[member[b]&visible[b]]
                    if not len(b):continue
                    vals,counts=np.unique(truth[b],return_counts=True)
                    oldregions.append(dict(scene=s,candidate_id=cid,source=r['source'],old_region=int(rid),domain='parent_visible_delete',
                        points=len(b),pure_known=bool(len(vals)==1 and vals[0]>=1000),
                        mixed_known=bool(len(vals[vals>=1000])>1),has_unknown=bool((vals<1000).any()),dominant_fraction=float(counts.max()/len(b))))
                # Exact common-mask versus disjoint-mask support, rather than best-ID alone.
                for u,v in z[cid+'_edges']:
                    same=distinct=missing=0
                    for f in r['views'][run.OBS]['frames']:
                        hits=z[cid+f'_frame{f}_mask_hits'];counts=z[cid+f'_frame{f}_visible']
                        a=set(np.flatnonzero(hits[u]));b=set(np.flatnonzero(hits[v]))
                        if counts[u]==0 or counts[v]==0 or not a or not b:missing+=1
                        else:
                            same+=bool(a&b);distinct+=bool(a-b) and bool(b-a)
                    edges.append(dict(scene=s,candidate_id=cid,u=int(u),v=int(v),any_common_mask_frames=same,
                        both_have_distinct_mask_frames=distinct,missing_frames=missing,
                        common_and_distinct_can_coexist=True))
                for mode in ('P0','P1'):
                    p=ps[mode+'_probability'][np.searchsorted(roi,halo)];link=linkage[cid,mode];q=link['parent']['top'][0]
                    hard=p>=.5;querycount=hard.sum(0);localids,counts=np.unique(truth[truth>=1000],return_counts=True)
                    hits=np.array([hard[truth==g].sum(0) for g in localids]).T if len(localids) else np.zeros((p.shape[1],0))
                    softiou=hits/np.maximum(querycount[:,None]+counts[None,:]-hits,1)
                    if not meta['stable']:
                        # Multiple GT identities are described, never a unique target assigned.
                        for i,g in enumerate(localids):
                            own=int((gt[parent]==g).sum())
                            if not own:continue
                            best=int(softiou[:,i].argmax());qs=max(link['query_set'],key=lambda j:(softiou[j,i],-j))
                            mixed.append(dict(scene=s,candidate_id=cid,scope=meta['scope'],prior=mode,GT_instance=int(g),
                                parent_points=own,halo_points=int(counts[i]),
                                all_query_best_raw_halo_iou=float(softiou[best,i]),best_query=best,
                                nonGT_set_best_raw_halo_iou=float(softiou[qs,i]),top1_raw_halo_iou=float(softiou[q,i])))
                        continue
                    g=meta['target'];target=truth==g;other=(truth>=1000)&~target;unknown=truth<1000
                    qbest=int(softiou[:,np.flatnonzero(localids==g)[0]].argmax())
                    voxel=sparse_row[halo];mixed_voxel=np.zeros(len(halo),bool)
                    # A Mask3D query produces one score per voxel; native point probe cannot split it.
                    for v in np.unique(voxel):
                        ix=np.flatnonzero(voxel==v)
                        if target[ix].any() and other[ix].any():mixed_voxel[ix]=True
                    for kind,query in [('top1',q),('GT_raw_oracle',qbest)]:
                        add=~member&visible&target;delete=member&visible&other
                        rows.append(dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,query_role=kind,query=query,
                            target_in_parent_visible=int((member&visible&target).sum()),
                            target_parent_false_delete=int((member&visible&target&~hard[:,query]).sum()),
                            target_add_opportunity=int(add.sum()),target_add_recovered=int((add&hard[:,query]).sum()),
                            pollution_delete_opportunity=int(delete.sum()),pollution_delete_recovered=int((delete&~hard[:,query]).sum()),
                            neighbor_false_add=int((~member&visible&other&hard[:,query]).sum()),
                            unknown_edit=int((visible&unknown&((member&~hard[:,query])|(~member&hard[:,query]))).sum()),
                            mixed_instance_voxel_points=int(mixed_voxel.sum()),mixed_voxel_add_opportunity=int((add&mixed_voxel).sum()),
                            mixed_voxel_delete_opportunity=int((delete&mixed_voxel).sum())))
                    # Region purity on the comparable parent-visible deletion domain.
                    region_delete=0;olddelete=0
                    for rid in np.unique(labels):
                        b=(labels==rid)&member&visible
                        if b.any() and np.all(other[b]):region_delete+=int(b.sum())
                    for b in old:
                        b=b[member[b]&visible[b]]
                        if len(b) and np.all(other[b]):olddelete+=len(b)
                    opportunity.append(dict(scene=s,candidate_id=cid,source=r['source'],prior=mode,
                        pollution_delete_points=int((member&visible&other).sum()),new_region_pure_delete_points=region_delete,
                        old96_pure_delete_points_before_overlap_dedup=olddelete,mixed_voxel_pollution_delete_points=int((member&visible&other&mixed_voxel).sum())))
        print('boundary diagnosis',s,flush=True)
    summary=dict(complete=True,posthoc_only=True,inference_unchanged=True,
        mechanism_counts={mode:{kind:{k:sum(r[k] for r in rows if r['prior']==mode and r['query_role']==kind)
            for k in rows[0] if isinstance(rows[0][k],int) and k!='query'} for kind in ('top1','GT_raw_oracle')} for mode in ('P0','P1')},
        mixed_identity_rows=len(mixed),old96_common_domain_regions=len(oldregions),
        old96_common_domain_pure_regions=sum(r['pure_known'] for r in oldregions),
        old96_common_domain_mixed_regions=sum(r['mixed_known'] for r in oldregions),
        common_mask_edges=sum(r['any_common_mask_frames']>0 for r in edges),
        distinct_mask_edges=sum(r['both_have_distinct_mask_frames']>0 for r in edges),
        no_mask_evidence_edges=sum(r['missing_frames']>0 and r['any_common_mask_frames']==0 and r['both_have_distinct_mask_frames']==0 for r in edges))
    for name,rr in [('mechanisms',rows),('mixed_raw_identities',mixed),('old96_common_domain_purity',oldregions),('opportunity',opportunity),('mask_contact_evidence',edges)]:ev.csv_write(d/(name+'.csv'),rr)
    run.seal(d/'summary.json',summary);run.stage_seal('boundary_diagnosis',list(d.glob('*')),GT_read=True,
        code={str(Path(__file__).resolve()):run.sha(Path(__file__).resolve())})
    print(summary,flush=True)


if __name__=='__main__':main()

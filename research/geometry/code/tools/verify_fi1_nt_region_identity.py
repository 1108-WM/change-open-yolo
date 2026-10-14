#!/usr/bin/env python3
"""Independent Python-set pointset/IoU/pool replay of every new diagnostic output."""
from pathlib import Path
import ast, csv, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_nt_region_identity as run


def vector(pointset,gt,ids,sizes):
    if len(pointset)<100:return np.zeros(len(ids))
    labels=[int(gt[i]) for i in pointset];counts={}
    for g in labels:counts[g]=counts.get(g,0)+1
    return np.array([counts.get(int(g),0)/(len(pointset)+int(size)-counts.get(int(g),0)) for g,size in zip(ids,sizes)])


def verify():
    c=run.read(run.OUT/'contract.json');run.verify(c['inputs']);run.verify(c['prior_inputs']);run.verify(c['frozen_SL_models'])
    for name in ('structure','structure_diagnosis','priors','associations','prior_diagnosis'):
        run.verify(run.read(run.OUT/(name+'_frozen.json'))['outputs'])
    parents=list(csv.DictReader((run.OUT/'prior_diagnosis/parents.csv').open()))
    targetrows=list(csv.DictReader((run.OUT/'prior_diagnosis/targets.csv').open()))
    lookup={(r['scene'],r['candidate_id'],r['mode']):r for r in parents}
    targetlookup={(r['scene'],r['mode'],int(r['gt_id'])):r for r in targetrows}
    n=ntargets=mapchecks=regionchecks=0
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);assert run.sha(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']]
        ids,sizes=np.unique(gt,return_counts=True);valid=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[valid],sizes[valid]
        _,nt=run.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,vector(set(map(int,p)),gt,ids,sizes))
        best={m:baseline.copy() for m in c['modes']};original={frozenset(map(int,p)) for p in nt.values()};generated={m:set() for m in c['modes']}
        with np.load(run.OUT/'structure'/s/'data.npz') as z,np.load(run.OUT/'associations'/s/'data.npz') as actions,np.load(run.OUT/'priors'/s/'data.npz') as prior:
            raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');coords=np.floor(raw[:,:3]/.02).astype(np.int32)
            unique=prior['unique_map'];inverse=prior['inverse_map'];rows=prior['original_to_sparse_row'];sparse=prior['input_sparse_coordinates']
            assert np.array_equal(coords[unique][inverse],coords);assert np.array_equal(sparse[rows,1:],coords);mapchecks+=len(coords)
            roi=prior['point_ids'];links={(r['candidate_id'],r['prior']):r for r in run.read(run.OUT/'associations'/s/'links.json')}
            for r in c['parents'][s]:
                cid=r['candidate_id'];parent=set(map(int,z[cid+'_parent']));halo=z[cid+'_halo'];labels=z[cid+'_labels'];order=z[cid+'_order'];offset=z[cid+'_offsets'];members=z[cid+'_membership'];visible=z[cid+'_visible']
                assert sorted(map(int,order))==list(range(len(halo)))
                for rid,(a,b) in enumerate(zip(offset[:-1],offset[1:])):
                    region=order[a:b];assert np.all(labels[region]==rid) and np.all(members[region]==members[region[0]]);regionchecks+=1
                before=vector(parent,gt,ids,sizes);idx=int(before.argmax()) if len(ids) and before.max()>0 else None
                gid=int(ids[idx]) if idx is not None else None;hits=sum(gt[i]==gid for i in parent) if gid is not None else 0;stable=gid is not None and hits>=20 and hits/len(parent)>=.6
                ix=np.searchsorted(roi,halo)
                for mode in c['modes']:
                    family,kind=mode.split('_');q=links[cid,family]['parent']['top'][0];p=prior[family+'_probability'][ix,q]
                    if kind=='region':
                        means=np.array([np.mean(p[labels==rid],dtype=np.float64) for rid in range(len(offset)-1)])
                        p=means[labels]
                    expected={int(point) for i,point in enumerate(halo) if (p[i]>=.5 if visible[i]>=2 else int(point) in parent)}
                    if len(expected)<100:expected=parent.copy()
                    actual=set(map(int,actions[cid+'_'+mode]));assert expected==actual,(s,cid,mode,'probe readout')
                    add=actual-parent;delete=parent-actual;row=lookup[s,cid,mode]
                    assert len(add)==int(row['added']) and len(delete)==int(row['deleted'])
                    assert ast.literal_eval(row['stable'])==stable
                    after=vector(actual,gt,ids,sizes);best[mode]=np.maximum(best[mode],after)
                    changed=bool(add or delete);frozen=frozenset(actual)
                    assert ast.literal_eval(row['duplicate_original'])==(changed and frozen in original)
                    assert ast.literal_eval(row['duplicate_generated'])==(changed and frozen in generated[mode]);generated[mode].add(frozen)
                    if stable:
                        at=sum(gt[i]==gid for i in add);dt=sum(gt[i]==gid for i in delete)
                        ao=sum(gt[i]>=1000 and gt[i]!=gid for i in add);do=sum(gt[i]>=1000 and gt[i]!=gid for i in delete)
                        for field,value in [('added_target',at),('deleted_target',dt),('added_other',ao),('deleted_other',do),('unknown_edits',len(add)+len(delete)-at-dt-ao-do)]:assert int(row[field])==value
                        assert abs(float(row['parent_iou'])-before[idx])<1e-10 and abs(float(row['repaired_iou'])-after[idx])<1e-10
                    else:assert int(row['unknown_edits'])==len(add)+len(delete)
                    n+=1
        for mode in c['modes']:
            for i,g in enumerate(ids):
                row=targetlookup[s,mode,int(g)];b=baseline[i];e=best[mode][i]
                assert abs(float(row['baseline'])-b)<1e-10 and abs(float(row['expanded'])-e)<1e-10
                assert ast.literal_eval(row['gain02'])==bool(e-b>=.02-1e-12)
                for t in (.5,.75,.9,.95):assert ast.literal_eval(row[f'cross_{t}'])==bool(b<=t<e)
                ntargets+=1
        print('independent verified',s,n,ntargets,flush=True)
    receipt=dict(complete=True,probes=n,pool_targets=ntargets,original_points_mapping_checked=mapchecks,regions_partition_checked=regionchecks,
        source_inputs_unchanged=True,old_prior_unchanged=True,SL_unchanged=True,checkpoint_unchanged=run.sha(c['checkpoint'])==c['checkpoint_sha256'],
        independent_set_readout_and_IoU=True,training=False,AP_run=False,time=__import__('time').time())
    run.seal(run.OUT/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':verify()

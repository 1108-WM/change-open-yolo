#!/usr/bin/env python3
"""Independent set readout/GT/pool and unchanged saved-model replay."""
from pathlib import Path
import ast,csv,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import run_fi1_query_transfer_eval as run
from verify_fi1_nt_region_identity import vector

def verify():
    out=run.OUT;c=run.read(out/'contract.json')
    for key in ('inputs','model_bindings','GT_inputs','code','frozen_SL_models'):run.diag.verify(c[key])
    for name in ('structure','priors','associations','predictions'):run.diag.verify(run.read(out/(name+'_frozen.json'))['outputs'])
    assert not {s.split('_')[0] for s in c['scenes']}&{s.split('_')[0] for s in c['pilot_training_scenes']}
    assert (out/'contract.json').stat().st_mtime < (out/'predictions_frozen.json').stat().st_mtime <= (out/'decision.json').stat().st_mtime
    rows=list(csv.DictReader((out/'evaluation/parents.csv').open()));pool=list(csv.DictReader((out/'evaluation/pool.csv').open()));lookup={(r['scene'],r['candidate_id'],r['mode']):r for r in rows};pl={(r['scene'],r['mode'],r['source'],int(r['gt_id'])):r for r in pool}
    torch.set_num_threads(4);models={}
    for arm in ('A','B'):
        net=run.pilot.CandidateLocalSegmenter(193,64).cuda().eval();net.load_state_dict(torch.load(c['model_paths'][arm],map_location='cuda',weights_only=True));models[arm]=net
    n=targets=replays=points=0;effects=[]
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);valid=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[valid],sizes[valid]
        _,nt=run.diag.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,vector(set(map(int,p)),gt,ids,sizes))
        original={frozenset(map(int,p)) for p in nt.values()};seen={m:set() for m in run.MODES};best={(m,source):baseline.copy() for m in ('original_NT',*run.MODES) for source in ('all','native','track')}
        with np.load(out/'predictions'/s/'data.npz') as pr,np.load(out/'associations'/s/'data.npz') as reg,np.load(out/'priors'/s/'data.npz') as prior,np.load(out/'structure'/s/'data.npz') as st:
            raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');coords=np.floor(raw[:,:3]/.02).astype(np.int32);np.testing.assert_array_equal(prior['input_sparse_coordinates'][prior['original_to_sparse_row'],1:],coords);points+=len(coords)
            for e in run.examples(s):
                cid=e['candidate_id'];parent=set(map(int,e['parent']));halo=e['halo'];before=vector(parent,gt,ids,sizes);idx=int(before.argmax()) if len(ids) and before.max()>0 else None;gid=int(ids[idx]) if idx is not None else None
                hits=sum(gt[i]==gid for i in parent) if gid is not None else 0;stable=gid is not None and hits>=20 and hits/len(parent)>=.6
                pix=np.searchsorted(prior['point_ids'],e['parent']);inter=prior['P1_probability'][pix].sum(0,dtype=np.float64);score=inter/(len(parent)+prior['P1_full_volume']-inter);q=int(np.lexsort((np.arange(len(score)),-score))[0]);assert q==e['query_id']
                ix=np.searchsorted(prior['point_ids'],halo);np.testing.assert_array_equal(e['query'],prior['P1_probability'][ix,q])
                labels=st[cid+'_labels'];means=np.array([e['query'][labels==rid].mean(dtype=np.float64) for rid in range(int(labels.max())+1)])
                for mode in run.MODES:
                    if mode=='region_average':
                        actual=set(map(int,reg[cid+'_P1_region']));expected={int(i) for k,i in enumerate(halo) if (means[labels[k]]>=.5 if e['visible'][k]>=2 else int(i) in parent)}
                    else:
                        p=pr[cid+'_'+mode+'_probability'];actual=set(map(int,pr[cid+'_'+mode+'_points']))
                        expected={int(i) for k,i in enumerate(halo) if (int(i) in parent and not (e['visible'][k]>=2 and p[k,1]>=.8)) or (int(i) not in parent and e['visible'][k]>=2 and p[k,0]>=.8)}
                        with torch.inference_mode():
                            x,nn,w,parts=run.pilot.collate([e],mode,'cuda');replay=models['A' if mode=='A' else 'B'](x,nn,w,parts).softmax(1).cpu().numpy()
                        np.testing.assert_allclose(replay,p,atol=1e-7,rtol=0);replays+=1
                        if mode not in ('A','B'):
                            b=pr[cid+'_B_probability'];effects.append(dict(scene=s,candidate_id=cid,mode=mode,mean_probability_change=float(np.abs(b-p).mean()),changed_points=len(set(map(int,pr[cid+'_B_points']))^actual)))
                    if len(expected)<100:expected=parent.copy()
                    assert actual==expected,(s,cid,mode,'readout')
                    row=lookup[s,cid,mode];assert ast.literal_eval(row['stable'])==stable;after=vector(actual,gt,ids,sizes)
                    add=actual-parent;delete=parent-actual;assert int(row['added'])==len(add) and int(row['deleted'])==len(delete)
                    if stable:
                        counts=dict(added_target=sum(gt[i]==gid for i in add),deleted_target=sum(gt[i]==gid for i in delete),added_other=sum(gt[i]>=1000 and gt[i]!=gid for i in add),deleted_other=sum(gt[i]>=1000 and gt[i]!=gid for i in delete))
                        counts['unknown_edits']=len(add)+len(delete)-sum(counts.values())
                        for k,v in counts.items():assert int(row[k])==v,(s,cid,mode,k)
                        assert abs(float(row['repaired_iou'])-after[idx])<1e-12 and abs(float(row['parent_iou'])-before[idx])<1e-12
                    else:assert int(row['unknown_edits'])==len(add)+len(delete)
                    code=frozenset(actual);changed=bool(add or delete);assert ast.literal_eval(row['duplicate_original'])==(changed and code in original);assert ast.literal_eval(row['duplicate_generated'])==(changed and code in seen[mode]);seen[mode].add(code)
                    best[mode,'all']=np.maximum(best[mode,'all'],after);best[mode,e['source']]=np.maximum(best[mode,e['source']],after);n+=1
        for (mode,source),v in best.items():
            for i,g in enumerate(ids):
                r=pl[s,mode,source,int(g)];assert abs(float(r['baseline'])-baseline[i])<1e-12 and abs(float(r['expanded'])-v[i])<1e-12
                assert ast.literal_eval(r['gain02'])==bool(v[i]-baseline[i]>=.02-1e-12)
                for t in (.5,.75,.9,.95):assert ast.literal_eval(r[f'cross_{t}'])==bool(baseline[i]<=t<v[i])
                targets+=1
        print('verified',s,n,targets,replays,flush=True)
    run.met.csv_write(out/'evaluation/intervention_effect.csv',effects)
    run.diag.verify(c['model_bindings']);run.diag.verify(c['frozen_SL_models'])
    run.seal(out/'verification.json',dict(complete=True,independent_pointsets=n,independent_pool_rows=targets,model_probability_replays=replays,original_points_mapping=points,physical_isolation=True,full_scene_GT_free_top1_verified=True,all_unstable_parents_retained=True,models_and_normalization_unchanged=True,SL_unchanged=True,training=False,AP_run=False,time=time.time()))

if __name__=='__main__':verify()

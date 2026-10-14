#!/usr/bin/env python3
"""Independent set action/GT/pool checks plus sealed final-model GPU replay."""
from pathlib import Path
import ast, csv, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import run_fi1_query_same_parent_pilot as run
from verify_fi1_nt_region_identity import vector

def verify():
    out=run.OUT;c=run.read(out/'contract.json');run.diagnostic.verify(c['code']);run.diagnostic.verify(c['outputs']);run.diagnostic.verify(c['input_bindings']);run.diagnostic.verify(c['GT_inputs']);run.diagnostic.verify(c['frozen_SL_models'])
    for name in ('models','predictions'):run.diagnostic.verify(run.read(out/(name+'_frozen.json'))['outputs'])
    assert (out/'models_frozen.json').stat().st_mtime <= (out/'predictions_frozen.json').stat().st_mtime <= (out/'decision.json').stat().st_mtime
    parents=list(csv.DictReader((out/'evaluation/parents.csv').open()));pool=list(csv.DictReader((out/'evaluation/pool.csv').open()));lookup={(r['scene'],r['candidate_id'],r['mode']):r for r in parents};pl={(r['scene'],r['mode'],int(r['gt_id'])):r for r in pool}
    n=targets=0;by_mode={m:set() for m in (*run.MODES,'region_average')}
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True);keep=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,c['class_ids']);ids,sizes=ids[keep],sizes[keep]
        _,nt=run.diagnostic.load_nt(c['jobs'][s]);base=np.zeros(len(ids))
        for p in nt.values():base=np.maximum(base,vector(set(map(int,p)),gt,ids,sizes))
        best={m:base.copy() for m in (*run.MODES,'region_average','original_NT')};generated={m:set() for m in best}
        with np.load(run.DIAG/'associations'/s/'data.npz') as region:
            for r in (r for r in c['samples'] if r['scene']==s):
                cid=r['candidate_id'];g=r['target'];idx=list(ids).index(g)
                with np.load(out/'inputs'/s/(cid+'.npz')) as z:
                    parent=set(map(int,z['parent']));halo=z['halo'];visible=z['visible'];before=vector(parent,gt,ids,sizes)
                    with np.load(run.DIAG/'priors'/s/'data.npz') as prior, np.load(run.SOURCE/'features'/s/'data.npz') as original:
                        roi=prior['point_ids'];ix=np.searchsorted(roi,halo);np.testing.assert_array_equal(roi[ix],halo)
                        parentix=np.searchsorted(roi,z['parent']);intersection=prior['P1_probability'][parentix].sum(0,dtype=np.float64)
                        scores=intersection/(len(parent)+prior['P1_full_volume']-intersection)
                        selected=int(np.lexsort((np.arange(len(scores)),-scores))[0]);assert selected==r['query_id']
                        np.testing.assert_array_equal(z['query'],prior['P1_probability'][ix,selected]);np.testing.assert_array_equal(z['head'],prior['P1_head_roi'][ix]);np.testing.assert_array_equal(z['base'],original[cid+'_gd10_selected6_x'].astype(np.float32))
                    yy=np.array([0 if gt[i]==g else 1 if gt[i]>=1000 else 2 for i in halo]);np.testing.assert_array_equal(yy,z['y']);assert run.digest(z['y'])==r['labels_sha256']
                    for mode in (*run.MODES,'region_average'):
                        if mode=='region_average':actual=set(map(int,region[cid+'_P1_region']))
                        else:
                            arm='A' if mode=='A' else 'B'
                            with np.load(out/'predictions'/s/(cid+'_'+arm+'.npz')) as pr:
                                prob=pr[mode+'_probability'];actual=set(map(int,pr[mode+'_points']))
                            expected={int(i) for k,i in enumerate(halo) if (int(i) in parent and not (visible[k]>=2 and prob[k,1]>=.8)) or (int(i) not in parent and visible[k]>=2 and prob[k,0]>=.8)}
                            if len(expected)<100:expected=parent.copy()
                            assert actual==expected,(s,cid,mode,'independent .8 actions')
                        row=lookup[s,cid,mode];add=actual-parent;delete=parent-actual;after=vector(actual,gt,ids,sizes);best[mode]=np.maximum(best[mode],after)
                        at=sum(gt[i]==g for i in add);ao=sum(gt[i]>=1000 and gt[i]!=g for i in add);dt=sum(gt[i]==g for i in delete);do=sum(gt[i]>=1000 and gt[i]!=g for i in delete)
                        for name,value in (('added',len(add)),('deleted',len(delete)),('added_target',at),('added_other',ao),('deleted_target',dt),('deleted_other',do),('unknown_edits',len(add)+len(delete)-at-ao-dt-do)):
                            assert int(row[name])==value,(s,cid,mode,name)
                        assert abs(float(row['parent_iou'])-before[idx])<1e-12 and abs(float(row['repaired_iou'])-after[idx])<1e-12
                        code=frozenset(actual);assert ast.literal_eval(row['duplicate_generated'])==(bool(add or delete) and code in generated[mode]);generated[mode].add(code)
                        if mode in ('A','B') and after[idx]-base[idx]>=.02-1e-12:by_mode[mode].add((s,g))
                        n+=1
        for mode,v in best.items():
            for i,g in enumerate(ids):
                row=pl[s,mode,int(g)];assert abs(float(row['baseline'])-base[i])<1e-12 and abs(float(row['expanded'])-v[i])<1e-12
                for t in (.5,.75,.9,.95):assert ast.literal_eval(row[f'cross_{t}'])==bool(base[i]<=t<v[i])
                assert ast.literal_eval(row['gain02'])==bool(v[i]-base[i]>=.02-1e-12);targets+=1
    torch.set_num_threads(4);data=run.load(False);replays=0;interventions=[];weights={}
    for arm in ('A','B'):
        net=run.CandidateLocalSegmenter(193,64).cuda().eval();net.load_state_dict(torch.load(out/'models'/arm/'model.pt',map_location='cuda',weights_only=True))
        weights[arm]=dict(query_embed_column_norm=float(net.embed[0].weight[:,192].norm()),other_embed_column_mean_norm=float(net.embed[0].weight[:,:192].norm(dim=0).mean()))
        for (s,cid),e in sorted(data.items()):
            with np.load(out/'predictions'/s/(cid+'_'+arm+'.npz')) as z:
                for mode in (('A',) if arm=='A' else run.MODES[1:]):
                    with torch.inference_mode():
                        x,nn,w,parts=run.collate([e],mode,'cuda');p=net(x,nn,w,parts).softmax(1).cpu().numpy()
                    np.testing.assert_allclose(p,z[mode+'_probability'],rtol=0,atol=1e-7);replays+=1
                    if arm=='B' and mode!='B':
                        bp=z['B_probability'];diff=np.abs(p-bp);changed=len(set(map(int,z['B_points']))^set(map(int,z[mode+'_points'])))
                        interventions.append(dict(scene=s,candidate_id=cid,mode=mode,mean_absolute_probability_change=float(diff.mean()),max_absolute_probability_change=float(diff.max()),changed_points=changed))
        del net;torch.cuda.empty_cache()
    ha,hb=[run.read(out/'models'/a/'history.json') for a in ('A','B')];assert ha['initial_state_sha256']==hb['initial_state_sha256'];assert len(ha['epochs'])==len(hb['epochs'])==64
    assert all(a['batch_order']==b['batch_order'] for a,b in zip(ha['epochs'],hb['epochs']))
    run.metrics.csv_write(out/'evaluation/intervention_effect.csv',interventions)
    receipt=dict(complete=True,independent_pointsets=n,independent_pool_rows=targets,final_model_probability_replays=replays,same_initialization_and_order=True,same_original_parent_labels=True,query_interventions_reach_forward=True,intervention_effect=interventions,weights=weights,B_pool_targets_not_in_A=sorted(by_mode['B']-by_mode['A']),identical_gain02_target_sets=by_mode['B']==by_mode['A'],source_and_SL_unchanged=True,model_before_prediction_before_metrics=True,time=time.time(),code={str(Path(__file__).resolve()):run.sha(Path(__file__).resolve())})
    run.seal(out/'verification.json',receipt);print('verified',n,targets,replays,'B only pool targets',receipt['B_pool_targets_not_in_A'],flush=True)

if __name__=='__main__':verify()

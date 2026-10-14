#!/usr/bin/env python3
"""Independent point actions, pooled geometry, model and fold/access replay."""
from pathlib import Path
from collections import Counter
import sys,csv
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_local_edit_ablation as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def iou(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    values,counts=np.unique(gt[p],return_counts=True);dd=dict(zip(values,counts));hit=np.array([dd.get(g,0) for g in ids]);return hit/(len(p)+sizes-hit)


def verify():
    out=run.OUT;c=run.read(out/'contract.json');stats=Counter();original=run.original
    for mapping in (c['code'],c['source_bindings'],c['frozen_SL_models']):original.audit.prior.old.verify_files(mapping)
    for name in ('conditions','labels','pilot','models','predictions','evaluation'):original.audit.prior.old.verify_files(run.read(out/(name+'_frozen.json'))['outputs'])
    for fold in range(5):
        for variant in run.learn.VARIANTS:
            h=run.read(out/'models'/variant/f'fold{fold}'/'history.json');assert len(h['epochs'])==12 and len(h['train_scenes'])==160 and len(h['test_scenes'])==40
            assert set(s[:9] for s in h['train_scenes']).isdisjoint(s[:9] for s in h['test_scenes']);assert h['train_scenes']==[s for s in c['scenes'] if c['folds'][s]!=fold];assert h['full_training_from_scratch'];stats['models']+=1
        for variant in ('B','C'):
            fit=run.read(out/'pilot'/variant/f'fold{fold}'/'fit.json');assert all(c['folds'][s]!=fold for s,cid in fit['keys']);assert len(fit['epochs'])==40 and len(fit['keys'])<=20;assert not fit['used_for_parameter_choice'] and not fit['used_for_full_training_warm_start'];stats['pilot_fits']+=1
        s=next(s for s in c['scenes'] if c['folds'][s]==fold)
        for mode,p in [('train',out/'labels'/s/'data.npz'),('pilot',out/'labels'/s/'meta.json'),('predict',out/'labels'/s/'data.npz'),('conditions',Path(c['jobs'][s]['GT'])),('train',Path(c['jobs'][s]['GT'])),('predict',run.AUDIT/'candidate_diagnosis.csv')]:
            try:run.Guard(c,mode,fold).hook('open',(str(p),'r',0))
            except PermissionError:stats['guard_rejections']+=1
            else:raise AssertionError((mode,p))
    with (out/'evaluation/candidates.csv').open() as f:reported={(r['scene'],r['candidate_id'],r['arm']):r for r in csv.DictReader(f)}
    with (out/'evaluation/targets.csv').open() as f:target_report={(r['scene'],r['gt_id'],r['arm'],r['mode']):r for r in csv.DictReader(f)}
    totals={a:Counter() for a in run.ARMS};delete_den=0;add_den=0
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=original.audit.prior.valid_targets(gt,c['class_ids']);geometry,pp=original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in pp.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
        best={a:{m:baseline.copy() for m in ('full','delete_only','add_only')} for a in run.ARMS};rows=run.read(run.SOURCE/'plan'/s/'candidates.json')
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(out/'conditions'/s/'data.npz') as cond,np.load(out/'predictions'/s/'predictions.npz') as prediction,np.load(run.SOURCE/'predictions'/s/'predictions.npz') as old:
            cm={r['candidate_id']:r for r in run.read(out/'conditions'/s/'meta.json')}
            for r in rows:
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];present=np.isin(halo,parent);seen=z[cid+'_'+run.OBS+'_groups']>=2
                assert np.array_equal(parent,pp[r['parent']])
                for k,key in enumerate(cm[cid]['competitors']):assert np.array_equal(cond[cid+'_membership'][:,k],np.isin(halo,pp[key]));stats['competitor_bindings']+=1
                assert np.array_equal(halo[cond[cid+'_representatives']],np.asarray(cm[cid]['parent_representatives'])) and np.isin(halo[cond[cid+'_representatives']],parent).all()
                for a in run.ARMS:
                    prob=prediction[cid+'_'+a+'_probability'].astype(np.float32);state=present.copy();state[present&seen&(prob[:,1]>=.8)]=False;state[~present&seen&(prob[:,0]>=.8)]=True
                    if a=='C' and prediction[cid+'_C_candidate_probability'].astype(np.float32)[0]<.7:state=present.copy()
                    p=halo[state] if state.sum()>=100 else parent;assert np.array_equal(p,prediction[cid+'_'+a+'_points']);assert np.array_equal(np.isin(halo[~seen],p),present[~seen]);stats['action_replays']+=1
                    if a=='A':assert np.array_equal(p,old[cid+'_'+run.OBS+'_points']) and np.array_equal(prob,old[cid+'_'+run.OBS+'_probability'].astype(np.float32));stats['frozen_A_replays']+=1
                    add=np.setdiff1d(p,parent);remove=np.setdiff1d(parent,p);d=np.setdiff1d(parent,remove);d=parent if len(d)<100 else d;ap=np.union1d(parent,add);report=reported[s,cid,a];counts=totals[a]
                    counts.update(parents=1,actions=bool(len(add)+len(remove)),added=len(add),deleted=len(remove));assert len(add)==int(report['added']) and len(remove)==int(report['deleted'])
                    for mode,mask in (('full',p),('delete_only',d),('add_only',ap)):best[a][mode]=np.maximum(best[a][mode],iou(mask,gt,ids,sizes))
                    if report['target']:
                        gid=int(report['target']);k=int(np.flatnonzero(ids==gid)[0]);before=iou(parent,gt,ids,sizes)[k];after=iou(p,gt,ids,sizes)[k];assert abs(before-float(report['parent_iou']))<1e-12 and abs(after-float(report['repaired_iou']))<1e-12
                        aa=int((gt[add]==gid).sum());ao=int(((gt[add]>=1000)&(gt[add]!=gid)).sum());dt=int((gt[remove]==gid).sum());do=int(((gt[remove]>=1000)&(gt[remove]!=gid)).sum())
                        for key,value in [('added_target',aa),('added_other',ao),('deleted_target',dt),('deleted_other',do)]:assert value==int(report[key]);counts[key]+=value
                        dden=int((present&seen&(gt[halo]>=1000)&(gt[halo]!=gid)).sum());aden=int((~present&seen&(gt[halo]==gid)).sum());assert dden==int(report['delete_opportunity']) and aden==int(report['add_opportunity'])
                        counts.update(delete_opportunity=dden,add_opportunity=aden)
                    else:counts.update(unknown_identity_add=len(add),unknown_identity_delete=len(remove))
        for a in run.ARMS:
            for mode in ('full','delete_only','add_only'):
                for k,g in enumerate(ids):
                    rr=target_report[s,str(g),a,mode];assert abs(float(rr['original_NT_best'])-baseline[k])<1e-12 and abs(float(rr['expanded_NT_best'])-best[a][mode][k])<1e-12;stats['pool_target_replays']+=1
        print('independent four-arm geometry verified',s,flush=True)
    summary=run.read(out/'summary.json')
    for a in run.ARMS:
        for key,value in totals[a].items():assert value==summary['arms'][a][key],(a,key,value,summary['arms'][a][key])
    torch.set_num_threads(2);model_replays=[]
    for fold in range(5):
        s=next(s for s in c['scenes'] if c['folds'][s]==fold);data=run.examples([s],False);keys=sorted(data)[:2]
        with np.load(out/'predictions'/s/'predictions.npz') as prediction:
            for variant in run.learn.VARIANTS:
                model=run.learn.model(variant).eval();model.load_state_dict(torch.load(out/'models'/variant/f'fold{fold}'/'model.pt',map_location='cpu',weights_only=True))
                for key in keys:
                    e=data[key];cid=key[1]
                    with torch.inference_mode():x,parts,logits,candidate,relation=run.forward(model,variant,[e],device='cpu');p=logits.softmax(1).numpy().astype(np.float16);cp=None if candidate is None else candidate.softmax(1)[0].numpy().astype(np.float16)
                    difference=float(np.abs(p.astype(np.float32)-prediction[cid+'_'+variant+'_probability'].astype(np.float32)).max());assert difference<=.001
                    actual,_=run.learn.repair(e['parent'],e['halo'],p,e['visible'],variant,cp);assert np.array_equal(actual,prediction[cid+'_'+variant+'_points']);model_replays.append(dict(fold=fold,scene=s,candidate_id=cid,variant=variant,max_point_probability_difference=difference))
    assert stats['models']==15 and stats['pilot_fits']==10 and stats['guard_rejections']==30 and stats['action_replays']==6396 and stats['frozen_A_replays']==1599 and stats['pool_target_replays']==56268
    freeze_json(out/'independent_verified.json',dict(complete=True,counts=dict(stats),model_replays=model_replays,all_point_actions_and_geometry_independent=True,contract_sha256=sha(out/'contract.json'),summary_sha256=sha(out/'summary.json')))
    freeze_json(out/'completion_verified.json',dict(complete=True,training_models=15,training_only_pilots=10,OOF_predictions=6396,independent_verified_sha256=sha(out/'independent_verified.json'),SL_preserved=True,AP_run=False));run.stage('complete_verified',**dict(stats))

if __name__=='__main__':verify()

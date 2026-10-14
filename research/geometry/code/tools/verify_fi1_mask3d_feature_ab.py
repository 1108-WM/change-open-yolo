#!/usr/bin/env python3
"""Independent original-index, point-action/IoU, calibration, source and CPU checks."""
from pathlib import Path
from collections import Counter
import csv,sys,time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_mask3d_feature_ab as run

def score(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    values=gt[p];hit=np.asarray([np.count_nonzero(values==g) for g in ids]);return hit/(len(p)+sizes-hit)

def point_action(parent,halo,p,visible,threshold):
    member=set(map(int,parent));q=set(member)
    for j,point in enumerate(halo):
        if visible[j]<2:continue
        if int(point) in member and float(np.float32(p[j,1]))>=threshold:q.discard(int(point))
        elif int(point) not in member and float(np.float32(p[j,0]))>=threshold:q.add(int(point))
    return np.asarray(sorted(q),parent.dtype) if len(q)>=100 else parent.copy()

def verify():
    c=run.read(run.OUT/'contract.json');run.integrity(c['code']);run.integrity(c['source_bindings']);run.integrity(c['frozen_SL_models'])
    for name in ('instance_features','models','probabilities','predictions','evaluation'):run.integrity(run.read(run.OUT/(name+'_frozen.json'))['outputs'])
    rules=run.read(run.OUT/'rules_frozen.json');assert rules['calibration_sha256']==run.sha(run.OUT/'calibration.json') and not rules['holdout_labels_read']
    guard=run.Guard(c,'predict');sys.addaudithook(guard.hook);denied=0
    for mode in ('predict','apply','train'):
        guard.mode=mode;guard.allowed=set(c['splits']['fit']+c['splits']['selection']) if mode=='train' else set()
        for split in ('calibration','holdout'):
            s=c['splits'][split][0]
            for p in (c['jobs'][s]['GT'],str(run.LABELS/'labels'/s/'data.npz')):
                try:Path(p).read_bytes()
                except PermissionError:denied+=1
                else:raise AssertionError('forbidden supervision opened')
    guard.mode='verify';physical=[{s.split('_')[0] for s in c['splits'][k]} for k in ('fit','selection','calibration','holdout')]
    assert all(not physical[i]&physical[j] for i in range(4) for j in range(i))
    for arm in ('A','B'):
        h=run.read(run.OUT/'models'/arm/'history.json');assert h['fit_scenes']==c['splits']['fit'] and h['selection_scenes']==c['splits']['selection']
        best=min(h['epochs'],key=lambda v:(v['selection_loss'],v['epoch']));assert best['epoch']==h['selected_epoch'];assert len(h['epochs'])==24
    aa,bb=[run.read(run.OUT/'models'/a/'history.json') for a in ('A','B')];assert aa['parameters']==bb['parameters'] and aa['initial_state_sha256']==bb['initial_state_sha256'];assert [h['steps'] for h in aa['epochs']]==[h['steps'] for h in bb['epochs']]
    with (run.OUT/'evaluation/candidates.csv').open() as f:rows=list(csv.DictReader(f))
    with (run.OUT/'evaluation/targets.csv').open() as f:targets=list(csv.DictReader(f))
    rowmap={(r['scene'],r['candidate_id'],r['arm'],r['mode']):r for r in rows};targetmap={(r['scene'],int(r['gt_id']),r['arm'],r['mode']):r for r in targets};summary=run.read(run.OUT/'summary.json');totals={};counts=Counter();maxdiff=0.
    with np.load(run.OUT/'normalization.npz') as z:mean,std=z['mean'],z['std']
    nets={}
    for arm in ('A','B'):
        net=run.core.model().eval();net.load_state_dict(torch.load(run.OUT/'models'/arm/'model.pt',map_location='cpu',weights_only=True));nets[arm]=net
    probes={s for k in ('fit','selection','calibration','holdout') for s in c['splits'][k][:2]}
    for ii,s in enumerate(c['scenes']):
        gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=run.original.audit.prior.valid_targets(gt,c['class_ids']);_,nt=run.original.audit.prior.old.load_nt(c['jobs'][s]);base=np.zeros(len(ids))
        for p in nt.values():base=np.maximum(base,score(p,gt,ids,sizes))
        expanded={(arm,mode):base.copy() for arm in ('A','B') for mode in ('generated','accepted')}
        split=next(k for k in ('fit','selection','calibration','holdout') if s in c['splits'][k])
        raw=np.load(c['jobs'][s]['paths']['processed'],mmap_mode='r');coordinates=np.floor(np.asarray(raw[:,:3],np.float32)/.02)
        with np.load(run.OUT/'instance_features'/s/'data.npz') as inst:
            unique=inst['unique_map'];inverse=inst['inverse_map'];roi=inst['point_ids'];prior=inst['features'];assert np.array_equal(coordinates[unique][inverse],coordinates)
            assert np.array_equal(roi,np.unique(roi)) and prior.shape==(len(roi),128);counts['original_index_maps']+=len(inverse)
            order=np.argsort(inverse[roi],kind='stable');same=inverse[roi][order];repeated=np.flatnonzero(same[1:]==same[:-1]);assert np.array_equal(prior[order[repeated]],prior[order[repeated+1]])
        receipt=run.read(run.OUT/'instance_features'/s/'receipt.json');assert receipt['columns_used']==list(range(9)) and receipt['mask_logits_bitwise_unchanged'] and receipt['hard_masks_bitwise_unchanged'];counts['mask_interface_invariance_scenes']+=1
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.OUT/'probabilities'/s/'data.npz') as pr,np.load(run.OUT/'predictions'/s/'data.npz') as out:
            for ri,r in enumerate(run.read(run.SOURCE/'plan'/s/'candidates.json')):
                cid=r['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];visible=z[cid+'_'+run.OBS+'_groups'];before=score(parent,gt,ids,sizes);index=int(np.argmax(before)) if len(ids) and before.max()>0 else None;gid=int(ids[index]) if index is not None else None;stable=gid is not None and (gt[parent]==gid).sum()>=20 and (gt[parent]==gid).mean()>=.6
                if s in c['splits']['train']:
                    with np.load(run.LABELS/'labels'/s/'data.npz') as labels:
                        truth=np.full(len(halo),2,dtype=np.uint8)
                        if stable:truth[gt[halo]>=1000]=1;truth[gt[halo]==gid]=0
                        assert np.array_equal(truth,labels[cid+'_y']);counts['supervision_point_checks']+=len(truth)
                for arm in ('A','B'):
                    p=pr[cid+'_'+arm+'_probability'];generation=point_action(parent,halo,p,visible,.8);assert np.array_equal(generation,pr[cid+'_'+arm+'_generated']);rule=rules['chosen'][arm]
                    acceptance=parent.copy() if rule['abstain'] else point_action(parent,halo,p,visible,rule['threshold']);assert np.array_equal(acceptance,out[cid+'_'+arm+'_accepted'])
                    if s in probes and ri==0:
                        e=run.load_scenes([s])[s,cid]
                        with torch.no_grad():logits,_,_=run.forward(nets[arm],[e],arm,mean,std,device='cpu');cpu=logits.softmax(1).numpy()
                        maxdiff=max(maxdiff,float(np.max(np.abs(cpu-p.astype(np.float32)))));assert np.allclose(cpu,p,atol=.002)
                        assert np.array_equal(point_action(parent,halo,cpu.astype(np.float16),visible,.8),generation);counts['CPU_model_replays']+=1
                    for mode,q in (('generated',generation),('accepted',acceptance)):
                        counts['actual_pointsets']+=1;rr=rowmap[s,cid,arm,mode];added=np.setdiff1d(q,parent);deleted=np.setdiff1d(parent,q);assert int(rr['added'])==len(added) and int(rr['deleted'])==len(deleted)
                        after=score(q,gt,ids,sizes);expanded[arm,mode]=np.maximum(expanded[arm,mode],after)
                        if stable:
                            assert np.isclose(float(rr['repaired_iou']),after[index],atol=1e-12);assert int(rr['added_target'])==int((gt[added]==gid).sum());assert int(rr['deleted_target'])==int((gt[deleted]==gid).sum());assert int(rr['added_other'])==int(((gt[added]>=1000)&(gt[added]!=gid)).sum());assert int(rr['deleted_other'])==int(((gt[deleted]>=1000)&(gt[deleted]!=gid)).sum())
        for (arm,mode),best in expanded.items():
            key=(split,arm+'_'+mode);totals.setdefault(key,Counter());totals[key]['pool_gain02']+=int((best-base>=.02-1e-12).sum());totals[key]['pool_delta_sum']+=float((best-base).sum())
            for g,value,initial in zip(ids,best,base):rr=targetmap[s,int(g),arm,mode];assert np.isclose(float(rr['baseline']),initial,atol=1e-12) and np.isclose(float(rr['expanded']),value,atol=1e-12);counts['full_NT_targets']+=1
        if (ii+1)%20==0:run.stage('independent_AB_verification',completed=ii+1,total=160)
    for (split,mode),v in totals.items():
        for k,val in v.items():assert np.isclose(summary['splits'][split][mode][k],val,atol=1e-10)
    # Recompute the original five calibration settings, not new thresholds.
    from tools.evaluate_fi1_mask3d_feature_ab import evaluate_scenes
    rr=evaluate_scenes(c,c['splits']['calibration'],[(a,t) for a in ('A','B') for t in run.core.THRESHOLDS],calibration=True)
    for r in run.read(run.OUT/'calibration.json')['grid']:
        m=rr[f"{r['arm']}:{r['threshold']}"]['metrics'];assert m==r['metrics'] and run.core.protection(m)==r['protection'];counts['calibration_rules_replayed']+=1
    run.seal(run.OUT/'independent_verified.json',dict(complete=True,counts=dict(counts),access_rejections=denied,CPU_max_probability_difference=maxdiff,summary_sha256=run.sha(run.OUT/'summary.json'),SL_preserved=True,Mask3D_frozen=True))
    run.seal(run.OUT/'completion_verified.json',dict(complete=True,independent_sha256=run.sha(run.OUT/'independent_verified.json'),summary_sha256=run.sha(run.OUT/'summary.json'),report_sha256=run.sha(run.OUT/'report.md'),experiment_finished=True,arms=2,new_frames=False,new_quality_heads=False,SL_modified=False,AP_run=False,decision=summary['decision']))
    run.stage('AB_experiment_completed',decision=summary['decision'],counts=dict(counts))

if __name__=='__main__':verify()

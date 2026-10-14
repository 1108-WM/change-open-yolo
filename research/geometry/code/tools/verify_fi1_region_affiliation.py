#!/usr/bin/env python3
"""Independent point-set, input/split, label, pool, and CPU model replays."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_region_affiliation as run
from tools.evaluate_fi1_region_affiliation import THRESHOLDS


def replay(parent,halo,blocks,pr,rule):
    if rule.get('abstain'):return parent.copy()
    selected=[j for j,p in enumerate(pr) if p[3]>=rule['safe'] and p[0]<=rule['target'] and p[2]<=rule['unknown']]
    risk={};unknown={}
    for j in selected:
        for i in blocks[j]:risk[int(i)]=max(risk.get(int(i),0),float(pr[j,0]));unknown[int(i)]=max(unknown.get(int(i),0),float(pr[j,2]))
    if risk and (np.mean(list(risk.values()))>rule['union_risk']+1e-8 or np.mean(list(unknown.values()))>rule['unknown']+1e-8):return parent.copy()
    removed={int(halo[i]) for i in risk};q=np.asarray([p for p in parent if int(p) not in removed],parent.dtype)
    return q if len(q)>=100 else parent.copy()


def scores(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    hist=Counter(gt[p].tolist());hit=np.asarray([hist[int(g)] for g in ids]);return hit/(len(p)+sizes-hit)


def verify():
    c=run.read(run.OUT/'contract.json');assert c['splits']==run.core.split_scenes(list(c['folds']),c['folds'])
    assert len({s.split('_')[0] for s in c['scenes']})==160 and not any(c['folds'][s]==0 for s in c['scenes'])
    for a in c['splits']:
        for b in c['splits']:
            if a!=b:assert not ({s[:9] for s in c['splits'][a]}&{s[:9] for s in c['splits'][b]})
    for k in ('code','source_bindings','frozen_SL_models'):run.integrity(c[k])
    for name in ('features','labels_train','models','probabilities','predictions','evaluation'):
        run.integrity(run.read(run.OUT/(name+'_frozen.json'))['outputs'])
    run.integrity(run.read(run.OUT/'features_frozen.json')['inputs'])
    rules=run.read(run.OUT/'rules_frozen.json');run.integrity(rules['outputs']);assert rules['holdout_labels_read']==False
    counts=Counter();guard=run.Guard(c,'predict');sys.addaudithook(guard.hook)
    for mode in ('features','predict','apply','train','labels_train','calibrate'):
        guard.mode=mode
        for scope in c['splits']:
            s=c['splits'][scope][0];allowed=scope=='train' and mode=='labels_train' or scope=='calibration' and mode=='calibrate'
            try:guard.hook('open',(c['jobs'][s]['GT'],))
            except PermissionError:assert not allowed;counts['GT_access_rejections']+=1
            else:assert allowed
    guard.mode='verify';summary=run.read(run.OUT/'summary.json');split={s:k for k,ss in c['splits'].items() for s in ss};metrics={k:{a:Counter() for a in run.ARMS} for k in c['splits']}
    fit={k:{a:[] for a in run.ARMS} for k in c['splits']};torch.set_num_threads(4)
    models={a:run.core.RegionAffiliation().eval() for a in run.ARMS}
    for a in run.ARMS:
        models[a].load_state_dict(torch.load(run.OUT/'models'/a/'model.pt',map_location='cpu',weights_only=True));history=run.read(run.OUT/'models'/a/'history.json')
        assert history['train_scenes']==c['splits']['train'] and len(history['epochs'])==24 and not history['region_replacement'];counts['optimization_updates']+=sum(r['updates'] for r in history['epochs'])
    sampled=set(s for ss in c['splits'].values() for s in ss[:3]);max_error=0.
    for i,s in enumerate(c['scenes']):
        scope=split[s];gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);assert run.sha(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']]
        ids,sizes=run.original.audit.prior.valid_targets(gt,c['class_ids']);_,nt=run.original.audit.prior.old.load_nt(c['jobs'][s]);base=np.zeros(len(ids))
        for p in nt.values():base=np.maximum(base,scores(p,gt,ids,sizes))
        best={a:base.copy() for a in run.ARMS};meta=run.supervision(s,c)[1];meta={r['candidate_id']:r for r in meta}
        with np.load(run.OUT/'features'/s/'data.npz') as feature,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank,np.load(run.SOURCE/'features'/s/'data.npz') as original,np.load(run.OUT/'predictions'/s/'data.npz') as output:
            pr={a:np.load(run.OUT/'probabilities'/a/s/'data.npz') for a in run.ARMS};lab=np.load(run.OUT/'labels'/s/'data.npz') if scope=='train' else None
            for row in run.read(run.OUT/'features'/s/'meta.json'):
                cid=row['candidate_id'];parent=original[cid+'_parent'];halo=original[cid+'_halo'];mm=meta[cid];blocks=run.prior.old.unpack(bank,cid);chosen=bank[cid+'_budget96'];assert np.array_equal(feature[cid+'_region_ids'],chosen)
                assert len(chosen)<=96;selected=[blocks[j] for j in chosen]
                for j,b in enumerate(selected):
                    assert np.isin(halo[b],parent).all() and (original[cid+'_'+run.OBS+'_groups'][b]>=2).all()
                    tid=feature[cid+'_token_ids'][j];valid=feature[cid+'_mask'][j];assert np.array_equal(tid>=0,valid)
                    roles=feature[cid+'_roles'][j];assert np.isin(tid[valid&(roles==0)],b).all()
                    if mm['stable']:
                        values=gt[halo[b]];t=int((values==mm['target']).sum());o=int(((values>=1000)&(values!=mm['target'])).sum());u=len(b)-t-o
                    else:t=o=0;u=len(b)
                    if lab is not None:
                        assert np.array_equal(lab[cid+'_counts'][j],[t,o,u]);assert np.allclose(lab[cid+'_fractions'][j],[t/len(b),o/len(b),u/len(b)])
                        assert lab[cid+'_safe'][j]==(o==len(b) and len(parent)-len(b)>=100)
                    counts['same_source_regions_verified']+=1
                views=feature[cid+'_views'];assert ((views[:,:,0]>.5)|(views[:,:,19]>.5)).all();assert np.isfinite(views).all()
                before=scores(parent,gt,ids,sizes);targetindex=None if not mm['stable'] else int(np.flatnonzero(ids==mm['target'])[0])
                for arm in run.ARMS:
                    probability=pr[arm][cid+'_probability'];q=replay(parent,halo,selected,probability,rules['rules'][arm]['rule']);assert np.array_equal(q,output[cid+'_'+arm+'_points']);after=scores(q,gt,ids,sizes);best[arm]=np.maximum(best[arm],after)
                    removed=np.setdiff1d(parent,q);m=metrics[scope][arm];m.update(parents=1,actions=bool(len(removed)),deleted=len(removed));counts['actual_union_replays']+=1
                    if mm['stable']:
                        t=int((gt[removed]==mm['target']).sum());o=int(((gt[removed]>=1000)&(gt[removed]!=mm['target'])).sum());u=len(removed)-t-o;delta=after[targetindex]-before[targetindex]
                        m.update(deleted_target=t,deleted_other=o,annotation_unknown_deleted=u,parent_improved02=delta>=.02-1e-12,parent_degraded02=delta<-.02-1e-12,good_parents=before[targetindex]>=.9,good_degraded02=before[targetindex]>=.9 and delta<-.02-1e-12,good_degraded10=before[targetindex]>=.9 and delta<-.1-1e-12,good_degraded25=before[targetindex]>=.9 and delta<-.25-1e-12,drift=bool(len(removed) and after.max(initial=0)>0 and np.argmax(after)!=targetindex and (gt[q]==ids[np.argmax(after)]).sum()>=20))
                    else:m['identity_unknown_deleted']+=len(removed)
            if lab is not None:lab.close()
            for p in pr.values():p.close()
        if s in sampled:
            data=run.load_data([s]);idx=np.arange(min(128,len(data['items'])))
            for arm in run.ARMS:
                with torch.inference_mode():cpu=run.core.decode(run.logits(models[arm],run.batch_data(data,idx,arm,device='cpu'))).numpy()
                withdata=[]
                with np.load(run.OUT/'probabilities'/arm/s/'data.npz') as p:
                    for ss,cid,j in data['items'][:len(idx)]:withdata.append(p[cid+'_probability'][j])
                error=float(np.max(np.abs(cpu-np.asarray(withdata)),initial=0));assert error<.0001,(s,arm,error);max_error=max(max_error,error);counts['CPU_probability_replays']+=len(idx)
        for arm in run.ARMS:
            metrics[scope][arm]['pool_gain02']+=int((best[arm]-base>=.02-1e-12).sum())
            for t in THRESHOLDS:metrics[scope][arm]['new_cross_'+str(t)]+=int(((base<=t)&(best[arm]>t)).sum())
            counts['pool_target_replays']+=len(ids)
        if (i+1)%20==0:run.stage('independent_internal_test_replay',completed=i+1,total=160)
    for scope in metrics:
        for arm,m in metrics[scope].items():
            expected=summary['splits'][scope]['arms'][arm]
            for key,value in m.items():assert expected.get(key,0)==value,(scope,arm,key,value,expected.get(key))
    run.integrity(c['frozen_SL_models']);run.seal(run.OUT/'independent_verified.json',dict(complete=True,counts=dict(counts),CPU_max_probability_error=max_error,models_sha256={a:run.sha(run.OUT/'models'/a/'model.pt') for a in run.ARMS},summary_sha256=run.sha(run.OUT/'summary.json'),splits_and_input_hashes_verified=True))
    run.seal(run.OUT/'completion_verified.json',dict(complete=True,report_sha256=run.sha(run.OUT/'report.md'),summary_sha256=run.sha(run.OUT/'summary.json'),independent_sha256=run.sha(run.OUT/'independent_verified.json'),new_training=True,models=2,formal_OOF=False,SL_preserved=True,AP_run=False));run.stage('completed_verified',**dict(counts))


if __name__=='__main__':verify()

#!/usr/bin/env python3
"""Independent relation labels, new calibration, actual geometry and CPU replay."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_decoupled_region as run
from tools.verify_fi1_region_affiliation import scores
from tools.evaluate_fi1_region_affiliation import THRESHOLDS


def execute(parent,halo,blocks,pred,rule):
    if rule.get('abstain') or pred['parent'][0]<rule['edit']:return parent.copy()
    selected=[]
    for i,p in enumerate(pred['relations']):
        if not pred['valid'][i,1]:continue
        same=max([float(p[k,0]) for k in range(2,6) if pred['valid'][i,k]] or [0])
        if min(float(p[1,1]),same)>=rule['relation']:selected.append(i)
    risk={}
    for i in selected:
        for point in blocks[i]:risk[int(point)]=max(risk.get(int(point),0),float(pred['relations'][i,1,0]))
    if risk and np.mean(list(risk.values()))>1-rule['relation']+1e-7:return parent.copy()
    removed={int(halo[j]) for j in risk};q=np.asarray([p for p in parent if int(p) not in removed],parent.dtype)
    return q if len(q)>=100 else parent.copy()


def independent_truth(a,b):
    if not len(a) or not len(b):return np.array([0,0,1.])
    aa=Counter(int(v) for v in a if v>=1000);bb=Counter(int(v) for v in b if v>=1000);n=len(a)*len(b);same=sum(v*bb[g] for g,v in aa.items())/n;known=sum(aa.values())*sum(bb.values())/n
    return np.array([same,known-same,1-known])


def verify():
    c=run.read(run.OUT/'contract.json');assert c['splits']==run.read(run.OLD/'contract.json')['splits'];assert not c['C_enabled']
    for name in ('code','source_bindings','frozen_SL_models'):run.integrity(c[name])
    for name in ('contexts','labels_train','models','probabilities','predictions','evaluation'):run.integrity(run.read(run.OUT/(name+'_frozen.json'))['outputs'])
    rules=run.read(run.OUT/'rules_frozen.json');assert not rules['holdout_labels_read'] and not rules['old_A_recalibrated'];assert run.sha(run.OUT/'calibration.json')==rules['calibration_sha256'];summary=run.read(run.OUT/'summary.json');count=Counter()
    guard=run.Guard(c,'predict');sys.addaudithook(guard.hook)
    for mode in ('contexts','predict','train','apply','labels_train','calibrate'):
        guard.mode=mode
        for split in c['splits']:
            s=c['splits'][split][0];allowed=mode=='labels_train' and split=='train' or mode=='calibrate' and split=='calibration'
            try:guard.hook('open',(c['jobs'][s]['GT'],))
            except PermissionError:assert not allowed;count['GT_access_rejections']+=1
            else:assert allowed
    for mode in ('contexts','train','predict'):
        guard.mode=mode
        for folder in ('probabilities','predictions','calibration'):
            try:guard.hook('open',(str(run.OLD/folder/'dummy.npz'),))
            except PermissionError:count['old_output_input_rejections']+=1
            else:raise AssertionError('old outputs allowed')
    guard.mode='verify';torch.set_num_threads(4);rel=run.core.InstanceRelation().eval();rel.load_state_dict(torch.load(run.OUT/'models/relations/model.pt',map_location='cpu',weights_only=True));parentmodel=run.core.ParentEditability().eval();parentmodel.load_state_dict(torch.load(run.OUT/'models/parent/model.pt',map_location='cpu',weights_only=True))
    for name,epochs in (('relations',24),('parent',60)):
        h=run.read(run.OUT/'models'/name/'history.json');assert h['train_scenes']==c['splits']['train'] and len(h['epochs'])==epochs and not h['other_model_predictions_used'];count['optimization_updates']+=sum(e['updates'] for e in h['epochs'])
    splits={s:k for k,ss in c['splits'].items() for s in ss};metrics={k:{a:Counter() for a in run.ARMS} for k in c['splits']};sampled={s for ss in c['splits'].values() for s in ss[:3]};maxdiff=0.
    for ii,s in enumerate(c['scenes']):
        split=splits[s];gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);assert run.sha(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']];ids,sizes=run.legacy.original.audit.prior.valid_targets(gt,c['class_ids']);_,nt=run.legacy.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,scores(p,gt,ids,sizes))
        best={a:baseline.copy() for a in run.ARMS};meta={m['candidate_id']:m for m in run.supervision(s,c)[1]}
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.OLD/'features'/s/'data.npz') as f,np.load(run.OUT/'contexts'/s/'data.npz') as ctx,np.load(run.REGIONS/'regions'/s/'regions.npz') as bank,np.load(run.OUT/'probabilities'/s/'data.npz') as pr,np.load(run.OUT/'predictions'/s/'data.npz') as out,np.load(run.OLD/'predictions'/s/'data.npz') as a:
            lab=np.load(run.OUT/'labels'/s/'data.npz') if split=='train' else None
            for row in run.read(run.OUT/'contexts'/s/'meta.json'):
                cid=row['candidate_id'];m=meta[cid];parent=z[cid+'_parent'];halo=z[cid+'_halo'];allblocks=run.legacy.prior.old.unpack(bank,cid);rr=bank[cid+'_budget96'];blocks=[allblocks[k] for k in rr];pred={k:pr[cid+'_'+k] for k in ('relations','parent','valid')}
                for j,b in enumerate(blocks):
                    assert np.isin(halo[b],parent).all() and (z[cid+'_'+run.OBS+'_groups'][b]>=2).all()
                    group=ctx[cid+'_group'][j];tokenids=f[cid+'_token_ids'][j]
                    for k in range(6):
                        reference=np.unique(tokenids[group==k+1]);truth=independent_truth(gt[halo[b]],gt[halo[reference]])
                        if lab is not None:assert np.allclose(truth,lab[cid+'_pair_truth'][j,k],atol=1e-7)
                        assert pred['valid'][j,k]==bool(len(reference));count['relation_target_replays']+=1
                    count['fixed_region_replays']+=1
                if lab is not None:
                    if m['stable']:
                        covered=set(int(halo[j]) for b in blocks for j in b);poll=sum(gt[p]>=1000 and gt[p]!=m['target'] for p in covered);target=0 if poll and len(parent)-poll>=100 else 1
                    else:target={'out_of_scope':2,'unannotated':3,'ambiguous':4}[m['scope']]
                    assert lab[cid+'_parent_label']==target;count['parent_supervision_replays']+=1
                q=execute(parent,halo,blocks,pred,rules['chosen']['rule']);assert np.array_equal(q,out[cid+'_D_decoupled_points']);assert np.array_equal(a[cid+'_A_points'],out[cid+'_A_sealed_points']);before=scores(parent,gt,ids,sizes)
                for arm in run.ARMS:
                    q=out[cid+'_'+arm+'_points'];removed=np.setdiff1d(parent,q);after=scores(q,gt,ids,sizes);best[arm]=np.maximum(best[arm],after);total=metrics[split][arm];total.update(parents=1,actions=bool(len(removed)),deleted=len(removed));count['actual_pointset_replays']+=1
                    if m['stable']:
                        gid=m['target'];k=int(np.flatnonzero(ids==gid)[0]);hit=Counter(gt[removed].tolist());other=sum(n for g,n in hit.items() if g>=1000 and g!=gid);delta=float(after[k]-before[k]);drift=bool(len(removed) and after.max(initial=0)>0 and np.argmax(after)!=k and (gt[q]==ids[np.argmax(after)]).sum()>=20)
                        total.update(deleted_target=hit[gid],deleted_other=other,annotation_unknown_deleted=len(removed)-hit[gid]-other,parent_improved02=delta>=.02-1e-12,parent_degraded02=delta<-.02-1e-12,good_parents=before[k]>=.9,good_degraded02=before[k]>=.9 and delta<-.02-1e-12,good_degraded10=before[k]>=.9 and delta<-.1-1e-12,good_degraded25=before[k]>=.9 and delta<-.25-1e-12,drift=drift)
                    else:total['identity_unknown_deleted']+=len(removed)
            if lab is not None:lab.close()
        if s in sampled:
            pair,parent=run.load_data([s]);idx=np.arange(min(128,len(pair['items'])))
            with torch.inference_mode():rr=run.pair_forward(rel,pair,idx,device='cpu').softmax(-1).numpy();pp=run.parent_forward(parentmodel,parent,np.arange(len(parent['items'])),device='cpu').softmax(-1).numpy()
            with np.load(run.OUT/'probabilities'/s/'data.npz') as pr:
                expected=np.stack([pr[cid+'_relations'][j] for ss,cid,j in pair['items'][:len(idx)]]) if len(idx) else np.empty((0,6,3));diff=float(np.max(np.abs(expected-rr),initial=0));assert diff<.0001;maxdiff=max(maxdiff,diff)
                for k,(ss,cid) in enumerate(parent['items']):diff=float(np.max(np.abs(pr[cid+'_parent']-pp[k])));assert diff<.0001;maxdiff=max(maxdiff,diff)
            count['CPU_relation_replays']+=len(idx)*6;count['CPU_parent_replays']+=len(parent['items'])
        for arm in run.ARMS:
            metrics[split][arm]['pool_gain02']+=int((best[arm]-baseline>=.02-1e-12).sum())
            for t in THRESHOLDS:metrics[split][arm]['new_cross_'+str(t)]+=int(((baseline<=t)&(best[arm]>t)).sum())
            count['pool_target_replays']+=len(ids)
        if (ii+1)%20==0:run.stage('independent_decoupled_replay',completed=ii+1,total=160)
    for split in metrics:
        for arm,total in metrics[split].items():
            for k,v in total.items():assert summary['splits'][split]['arms'][arm].get(k,0)==v,(split,arm,k)
    # Recount all16 calibration rules; this reads calibration GT only.
    guard.mode='calibrate';examples=run.calibration_examples(c)
    for row in run.read(run.OUT/'calibration.json')['grid']:
        total=Counter();scene=set()
        for e in examples:
            q=execute(e['parent'],e['halo'],e['blocks'],e['prediction'],row['rule']);removed=np.setdiff1d(e['parent'],q);total['deleted']+=len(removed)
            if not len(removed):continue
            total['actions']+=1;scene.add(e['scene']);m=e['meta']
            if not m['stable']:total['unknown']+=len(removed);continue
            gid=m['target'];h=Counter(e['gt'][removed].tolist());other=sum(n for g,n in h.items() if g>=1000 and g!=gid);total.update(target=h[gid],other=other,unknown=len(removed)-h[gid]-other)
        for k,value in total.items():assert row['metrics'].get(k,0)==value,(k,value,row['metrics'])
        assert row['metrics']['action_scenes']==len(scene);count['calibration_rules_replayed']+=1
    run.integrity(c['source_bindings']);run.integrity(c['frozen_SL_models']);run.seal(run.OUT/'independent_verified.json',dict(complete=True,counts=dict(count),CPU_max_probability_difference=maxdiff,old_A_unchanged=True,old_C_not_used=True,models_independent=True,summary_sha256=run.sha(run.OUT/'summary.json')))
    run.seal(run.OUT/'completion_verified.json',dict(complete=True,report_sha256=run.sha(run.OUT/'report.md'),summary_sha256=run.sha(run.OUT/'summary.json'),independent_sha256=run.sha(run.OUT/'independent_verified.json'),new_models=2,old_A_unchanged=True,old_threshold_search=False,C_enabled=False,stop_from_scratch_recipe=summary['stop_from_scratch_recipe'],SL_preserved=True,AP_run=False));run.stage('completed_verified',**dict(count))


if __name__=='__main__':verify()

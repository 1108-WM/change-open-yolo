#!/usr/bin/env python3
"""Independent replay of five fitted models and complete geometry, zero refits."""
from collections import Counter
import subprocess
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import run_fi1_mixed_risk_supervision as r


def main():
    r.guard(gt=True);c=r.contract();a=r.a;checks=Counter()
    for name in ('predictions_frozen.json','evaluation_frozen.json'):r.verify(a.read(r.OUT/name)['outputs'])
    source=a.read(r.SCALE/'contract.json')
    assert c['scenes']==source['scenes'] and c['extra_scenes']==source['extra_scenes'] and c['splits']==source['splits']
    physical=lambda ss:{s.rsplit('_',1)[0] for s in ss}
    assert physical(c['extra_scenes']).isdisjoint(physical(c['scenes']))
    rr,x,n=r.combined();y=r.labels_for(c['scenes']+c['extra_scenes'],c,rr)
    assert Counter(y)==Counter({1:3050,0:952,2:3200,-1:1714})
    diag=a.read(r.OUT/'learning_diagnostics.json');pooled={arm:[] for arm in r.ARMS}
    for f in range(5):
        split=c['splits'][str(f)];assert physical(split['double']).isdisjoint(physical(split['heldout']))
        d=r.OUT/'models'/f'fold{f}';rec=a.read(d/'receipt.json');z=np.load(d/'training.npz')
        assert set(rec['label_inputs'])=={str(r.labelpath(s,c)) for s in split['double']}
        r.verify(rec['label_inputs'])
        idx=np.array([i for i,q in enumerate(rr) if q['scene'] in split['double'] and y[i]>=0])
        np.testing.assert_array_equal(z['row_indices'],idx);np.testing.assert_array_equal(z['y'],y[idx])
        assert set(z['y'])=={0,1,2} and (y[idx]==2).any()
        count=Counter(rr[i]['scene'] for i in idx)
        w=np.array([len(idx)/len(count)/count[rr[i]['scene']] for i in idx])
        np.testing.assert_allclose(z['weights'],w,rtol=1e-13,atol=1e-13)
        checks['equal_scene_mass']+=len(count)
        model=joblib.load(d/'mixed.joblib');baseline=joblib.load(r.SCALE/'models'/f'fold{f}'/'double.joblib')
        assert model.get_params()==baseline.get_params()
        np.testing.assert_array_equal(model.classes_,[0,1,2]);assert model.n_trees_per_iteration_==3 and baseline.n_trees_per_iteration_==1
        with threadpool_limits(4):p=r.probability_columns(model,x)
        np.testing.assert_array_equal(p,np.load(d/'probabilities.npy'));np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
        assert (p>=0).all() and (p<=1).all();checks['model_reloads_no_fit']+=1
        for arm,score,trainidx in (('mixed',p[:,1],idx),('binary',np.load(r.SCALE/'models'/f'fold{f}'/'double_scores.npy'),
                np.load(r.SCALE/'models'/f'fold{f}'/'double_training.npz')['row_indices'])):
            for part,ix in (('train',trainidx),('common_clear_train',np.load(r.SCALE/'models'/f'fold{f}'/'double_training.npz')['row_indices']),
                    ('heldout',np.array([i for i,q in enumerate(rr[:n]) if c['folds'][q['scene']]==f]))):
                ii=ix[((y[ix]==0)|(y[ix]==1))&(x[ix,4]==0)]
                v=np.clip(score[ii],1e-12,1-1e-12);loss=-y[ii]*np.log(v)-(1-y[ii])*np.log1p(-v)
                bal=(loss[y[ii]==1].mean()+loss[y[ii]==0].mean())/2
                assert abs(bal-diag['arms'][arm]['folds'][str(f)][part]['disjoint']['balanced_logloss'])<1e-12
                checks['binary_metrics_recomputed']+=1
                if part=='heldout':pooled[arm].extend([(int(y[i]),float(score[i])) for i in ii])
        test=np.array([i for i,q in enumerate(rr[:n]) if c['folds'][q['scene']]==f and y[i]>=0])
        loss=-np.log(np.clip(p[test,y[test]],1e-12,1)).mean()
        assert abs(loss-diag['arms']['mixed']['folds'][str(f)]['multiclass_heldout']['logloss'])<1e-12
        print('VERIFIED fold',f,flush=True)
    assert len(list((r.OUT/'models').glob('fold*/*.joblib')))==5
    for arm in r.ARMS:
        yy,ss=np.array(pooled[arm]).T;v=np.clip(ss,1e-12,1-1e-12);loss=-yy*np.log(v)-(1-yy)*np.log1p(-v)
        bal=(loss[yy==1].mean()+loss[yy==0].mean())/2
        assert abs(bal-diag['arms'][arm]['pooled']['heldout']['balanced_logloss'])<1e-12
    for target,src in (('binary_scores.npz','double_scores.npz'),('binary_decisions.json','double_decisions.json')):
        assert a.sha(r.OUT/target)==a.sha(r.SCALE/src)
    folds=np.array([c['folds'][q['scene']] for q in rr[:n]])
    z=np.load(r.OUT/'mixed_scores.npz');np.testing.assert_array_equal(z['oof'],z['fold_all'][folds,np.arange(n)])
    for f in range(5):np.testing.assert_array_equal(z['fold_all'][f],np.load(r.OUT/'models'/f'fold{f}'/'probabilities.npy')[:n])
    results=a.read(r.OUT/'results.json')['results']
    for arm in r.ARMS:
        raw=np.load(r.OUT/(arm+'_scores.npz'))['oof'];score=raw[:,1] if arm=='mixed' else raw
        pred=a.read(r.OUT/(arm+'_decisions.json'));lookup={(q['scene'],q['pair_id']):q for q in pred}
        joined=a.read(r.OUT/'evaluation'/(arm+'_pairs.json'));accepted=[q for q in joined if q['state']=='accept']
        assert results[arm]['strict_accepted']==dict(Counter(q['GT_relation_strict'] for q in accepted))
        assert results[arm]['good_endpoint_loss_gt02']==sum(q['good_endpoint_loss_gt02'] for q in accepted)
        for s in c['scenes']:
            ix=[i for i,q in enumerate(rr[:n]) if q['scene']==s];used=set();expected=set()
            for i in sorted(ix,key=lambda i:(-score[i],rr[i]['pair_id'])):
                q=rr[i]
                if q['eligible'] and score[i]>=.9 and not ({q['left'],q['right']}&used):
                    used.update((q['left'],q['right']));expected.add(q['pair_id'])
            assert expected=={q['pair_id'] for q in pred if q['scene']==s and q['state']=='accept'}
            emitted=set()
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as unions,np.load(r.GEOMETRY/'predictions'/s/'data.npz') as src:
                for i in ix:
                    q=lookup[s,rr[i]['pair_id']];assert q['score']==score[i]
                    if arm=='mixed':assert q['mixed_probability']==raw[i,2] and q['different_probability']==raw[i,0]
                    if q['state']!='accept':assert q['state']==('reject' if rr[i]['eligible'] and score[i]<=.1 else 'unknown')
                    should_emit=q['state']=='accept' and not q['duplicate_original_NT'] and q['union_sha256'] not in emitted
                    assert q['emit']==should_emit
                    if should_emit:
                        emitted.add(q['union_sha256']);pp=unions[q['pair_id']+'_union']
                        np.testing.assert_array_equal(pp,np.union1d(src['n%04d'%q['left']],src['n%04d'%q['right']]))
                        assert a.digest(pp)==q['union_sha256'];checks['whole_unions']+=1
                assert set(unions.files)=={q['pair_id']+'_union' for q in pred if q['scene']==s and q['emit']}
            if arm=='binary':
                for name in ('pairs.json','unions.npz'):assert a.sha(r.OUT/'predictions/binary'/s/name)==a.sha(r.SCALE/'predictions/double'/s/name)
    # Compute original NT and additions directly from point memberships, independently of evaluator.
    jobs=a.read(r.AUDIT/'contract.json')['jobs'];classes=a.read(a.OLD/'contract.json')['class_ids']
    targets={arm:{(q['scene'],q['gt_id']):q for q in a.read(r.OUT/'evaluation'/(arm+'_targets.json'))} for arm in r.ARMS}
    def iou(pp,gt,ids,sizes):
        hit=np.array([np.count_nonzero(gt[pp]==g) for g in ids])
        return hit/(len(pp)+sizes-hit) if len(pp)>=100 else np.zeros(len(ids))
    gains=Counter()
    for s in c['scenes']:
        gt=np.loadtxt(jobs[s]['job']['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        keep=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,classes);ids,sizes=ids[keep],sizes[keep];base=np.zeros(len(ids))
        with np.load(r.GEOMETRY/'predictions'/s/'data.npz') as src:
            for node in a.read(r.GEOMETRY/'predictions'/s/'nodes.json'):
                if node['in_original_NT']:base=np.maximum(base,iou(src['n%04d'%node['node']],gt,ids,sizes))
        for arm in r.ARMS:
            best=base.copy()
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as unions:
                for key in unions.files:best=np.maximum(best,iou(unions[key],gt,ids,sizes))
            gains[arm]+=int(((best-base)>=.02).sum())
            for j,g in enumerate(ids):
                q=targets[arm][s,int(g)];assert abs(q['baseline']-base[j])<1e-12 and abs(q['expanded']-best[j])<1e-12
                checks['full_NT_target_rows']+=1
    for arm in r.ARMS:assert gains[arm]==results[arm]['pool']['pool_gain02']
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r.HERE),'-p','test_fi1_mixed_risk_supervision.py','-v'],capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    a.seal(r.OUT/'verification.json',dict(complete=True,additional_model_fits=0,checks=dict(checks),tests=tests.stderr,
        limitation='Endpoint composition and identity reuse certified full-endpoint labels and prior pair tables; complete NT IoU independently recomputed. Evaluation52 exposed development.'))
    print('VERIFIED',dict(checks),flush=True)


if __name__=='__main__':main()

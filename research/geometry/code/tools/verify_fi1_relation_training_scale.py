#!/usr/bin/env python3
"""Independent split, training-weight, score, action and complete NT checks; no fits."""
from collections import Counter
import json
import subprocess
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import run_fi1_relation_training_scale as r


def main():
    r.guard(gt=True);c=r.contract();a=r.a;checks=Counter()
    for name in ('predictions_frozen.json','evaluation_frozen.json'):r.verify(a.read(r.OUT/name)['outputs'])
    assert a.read(r.OUT/'pilot_parity.json')['complete']
    inv=a.read(r.OUT/'inventory.json');protected=set(inv['protected_physical'])
    assert {r.physical(s) for s in c['extra_scenes']}.isdisjoint(protected|{r.physical(s) for s in c['scenes']})
    available=[q['scene'] for q in inv['scenes'] if q['eligible']]
    assert c['extra_scenes']==r.ordered(available,'extra')[:len(c['extra_scenes'])]
    rr,x,nold=r.combined();oldscore=np.load(r.SOURCE/'C_scores.npz')['fold_all']
    diag=a.read(r.OUT/'learning_diagnostics.json');lookup={}
    for s in c['scenes']+c['extra_scenes']:
        root=r.BASE if s in c['scenes'] else r.OUT/'training_data'
        lookup.update({(s,q['pair_id']):q['relation'] for q in a.read(root/'labels'/f'{s}.json')['pairs']})
    yy=np.array([1 if lookup[q['scene'],q['pair_id']]=='same' else 0 if lookup[q['scene'],q['pair_id']]=='different' else -1 for q in rr])
    for f in range(5):
        split=c['splits'][str(f)];full=[s for s in c['scenes'] if c['folds'][s]!=f]
        assert split['original']==full
        assert split['half']==r.ordered(full,'half')[:len(full)//2]
        assert split['double']==full+c['extra_scenes'][:len(full)]
        for arm in r.ARMS:
            assert {r.physical(s) for s in split[arm]}.isdisjoint({r.physical(s) for s in split['heldout']})
            d=r.OUT/'models'/f'fold{f}';receipt=a.read(d/(arm+'_receipt.json'))
            assert set(receipt['label_inputs'])=={str((r.BASE if s in c['scenes'] else r.OUT/'training_data')/'labels'/f'{s}.json') for s in split[arm]}
            r.verify(receipt['label_inputs'])
            z=np.load(d/(arm+'_training.npz'))
            idx=np.array([i for i,q in enumerate(rr) if q['scene'] in split[arm] and yy[i]>=0])
            np.testing.assert_array_equal(idx,z['row_indices']);np.testing.assert_array_equal(yy[idx],z['y'])
            counts=Counter(rr[i]['scene'] for i in idx)
            expected=np.array([len(idx)/len(counts)/counts[rr[i]['scene']] for i in idx])
            np.testing.assert_allclose(z['weights'],expected,rtol=1e-13,atol=1e-13)
            for s,n in counts.items():
                np.testing.assert_allclose(z['weights'][[rr[i]['scene']==s for i in idx]].sum(),len(idx)/len(counts),rtol=1e-13)
                checks['scene_weight_mass']+=1
            model=joblib.load(r.SOURCE/'models'/f'fold{f}'/'C.joblib' if arm=='original' else d/(arm+'.joblib'))
            assert model.get_params()==joblib.load(r.SOURCE/'models'/f'fold{f}'/'C.joblib').get_params()
            with threadpool_limits(4):score=model.predict_proba(x)[:,1]
            np.testing.assert_array_equal(score,np.load(d/(arm+'_scores.npy')))
            if arm=='original':np.testing.assert_array_equal(score[:nold],oldscore[f])
            checks['models_reloaded_no_refit']+=1
            for part,ii in (('train',idx),('common_half',np.load(d/'half_training.npz')['row_indices']),
                ('heldout',np.array([i for i,q in enumerate(rr[:nold]) if c['folds'][q['scene']]==f and yy[i]>=0]))):
                ii=ii[x[ii,4]==0];y=yy[ii];p=np.clip(score[ii],1e-12,1-1e-12)
                loss=-y*np.log(p)-(1-y)*np.log1p(-p)
                expectedloss=(loss[y==1].mean()+loss[y==0].mean())/2
                assert abs(expectedloss-diag['arms'][arm]['folds'][str(f)][part]['disjoint']['balanced_logloss'])<1e-12
                checks['difficult_metrics_reconstructed']+=1
        print('VERIFIED fold',f,flush=True)
    assert len(list((r.OUT/'models').glob('fold*/*.joblib')))==10
    for arm in r.ARMS:
        scores=np.load(r.OUT/(arm+'_scores.npz'));folds=np.array([c['folds'][q['scene']] for q in rr[:nold]])
        np.testing.assert_array_equal(scores['oof'],scores['fold_all'][folds,np.arange(nold)])
        pred=a.read(r.OUT/(arm+'_decisions.json'))
        if arm=='original':assert a.sha(r.OUT/(arm+'_decisions.json'))==a.sha(r.SOURCE/'C_decisions.json')
        for s in c['scenes']:
            ix=[i for i,q in enumerate(rr[:nold]) if q['scene']==s];used=set();accepted=set()
            for i in sorted(ix,key=lambda i:(-scores['oof'][i],rr[i]['pair_id'])):
                q=rr[i]
                if q['eligible'] and scores['oof'][i]>=.9 and q['left'] not in used and q['right'] not in used:
                    used.update((q['left'],q['right']));accepted.add(q['pair_id'])
            assert accepted=={pred[i]['pair_id'] for i in ix if pred[i]['state']=='accept'}
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as z,np.load(r.GEOMETRY/'predictions'/s/'data.npz') as old:
                assert set(z.files)=={pred[i]['pair_id']+'_union' for i in ix if pred[i]['emit']}
                for i in ix:
                    q=pred[i];assert q['score']==scores['oof'][i]
                    if q['emit']:
                        pp=z[q['pair_id']+'_union'];np.testing.assert_array_equal(pp,np.union1d(old['n%04d'%q['left']],old['n%04d'%q['right']]))
                        assert a.digest(pp)==q['union_sha256'];checks['complete_unions']+=1
            if arm=='original':
                for name in ('pairs.json','unions.npz'):assert a.sha(r.OUT/'predictions'/arm/s/name)==a.sha(r.SOURCE/'predictions/C'/s/name)
    # Independently rebuild complete-NT IoU from exact point memberships.
    jobs=a.read(r.AUDIT/'contract.json')['jobs'];classes=a.read(a.OLD/'contract.json')['class_ids']
    targets={arm:{(q['scene'],q['gt_id']):q for q in a.read(r.OUT/'evaluation'/(arm+'_targets.json'))} for arm in r.ARMS}
    def iou(pp,gt,ids,sizes):
        hit=np.array([np.count_nonzero(gt[pp]==g) for g in ids])
        return hit/(len(pp)+sizes-hit) if len(pp)>=100 else np.zeros(len(ids))
    for s in c['scenes']:
        gt=np.loadtxt(jobs[s]['job']['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        keep=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,classes);ids,sizes=ids[keep],sizes[keep];base=np.zeros(len(ids))
        with np.load(r.GEOMETRY/'predictions'/s/'data.npz') as z:
            for n in a.read(r.GEOMETRY/'predictions'/s/'nodes.json'):
                if n['in_original_NT']:base=np.maximum(base,iou(z['n%04d'%n['node']],gt,ids,sizes))
        for arm in r.ARMS:
            best=base.copy()
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as z:
                for k in z.files:best=np.maximum(best,iou(z[k],gt,ids,sizes))
            for j,g in enumerate(ids):
                q=targets[arm][s,int(g)];assert abs(q['baseline']-base[j])<1e-12 and abs(q['expanded']-best[j])<1e-12
                checks['complete_NT_target_rows']+=1
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r.HERE),'-p','test_fi1_relation_training_scale.py','-v'],capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    a.seal(r.OUT/'verification.json',dict(complete=True,checks=dict(checks),additional_model_fits=0,tests=tests.stderr,
        limitation='No new independent evaluation scenes; endpoint identity/composition reuse previous certified tables; full NT point IoU independently recalculated.'))
    print('VERIFIED',dict(checks),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Verify exact200/55 scope, one fit, cache provenance, labels and full-NT outcomes."""
from collections import Counter
import subprocess
import sys
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import run_fi1_relation_train200_dev55 as r


def main():
    r.guard(gt=True);c=r.contract();a=r.a;checks=Counter()
    for name in ('data_frozen.json','training_labels_frozen.json','predictions_frozen.json','evaluation_frozen.json'):r.verify(a.read(r.OUT/name)['outputs'])
    old=a.read(r.SPLIT);assert c['train_scenes']==old['train_scenes'] and c['scenes']==old['scenes']
    assert not (r.physical(c['train_scenes'])&r.physical(c['scenes']))
    seal=a.read(r.OUT/'data_frozen.json');assert len(seal['scenes'])==255 and all(q['complete'] for q in seal['scenes'])
    assert seal['reference82_feature_parity']==c['small_scenes']
    for s in c['train_scenes']+c['scenes']:
        raw=r.OUT/'geometry/predictions'/s;meas=r.OUT/'measurements/predictions'/s
        for name in ('nodes.json','pairs.json'):assert a.sha(raw/name)==a.sha(meas/name)
        corrections=a.read(meas/'cache_corrections.json');vp=a.read(raw/'visibility.json');mv=a.read(meas/'visibility.json')
        assert corrections['frames_checked']==len(vp)==len(mv)
        by_frame={str(q['frame_id']):q for q in corrections['corrections']}
        with np.load(raw/'data.npz') as z,np.load(meas/'data.npz') as zz:
            for key in z.files:
                if key!='frame_node_visible' and not key.endswith('_intersection_visible'):np.testing.assert_array_equal(z[key],zz[key])
            for j,(v,m) in enumerate(zip(vp,mv)):
                before=np.load(v['path']);after=np.load(m['path']);assert not (after&~before).any()
                if str(v['frame_id']) in by_frame:
                    q=by_frame[str(v['frame_id'])];np.testing.assert_array_equal(np.flatnonzero(before&~after),q['removed_point_ids'])
                    checks['corrected_visibility_frames']+=1
                else:np.testing.assert_array_equal(before,after)
                for n in a.read(raw/'nodes.json'):
                    assert zz['frame_node_visible'][j,n['node']]==after[z['n%04d'%n['node']]].sum()
                for p in a.read(raw/'pairs.json'):
                    pp=np.intersect1d(z['n%04d'%p['left']],z['n%04d'%p['right']])
                    assert zz[p['pair_id']+'_intersection_visible'][j]==after[pp].sum()
                checks['visibility_frames']+=1
        checks['scene_geometries_and_measurements']+=1
    tr,tx=r.matrix('train200');rr,x=r.matrix('dev55');look={}
    for s in c['train_scenes']+c['scenes']:
        lab=a.read(r.OUT/'labels'/f'{s}.json');gt=np.loadtxt(c['jobs'][s]['job']['GT'],dtype=np.int64)
        kinds={}
        with np.load(r.OUT/'geometry/predictions'/s/'data.npz') as z:
            for key,meta in lab['nodes'].items():
                pp=z['n%04d'%int(key)];ids,counts=np.unique(gt[pp],return_counts=True);known=ids>=1000
                assert meta['distribution']=={str(g):int(n) for g,n in zip(ids,counts)}
                mixed=((counts>=20)&known).sum()>=2;j=int(np.argmax(np.where(known,counts,0))) if known.any() else None
                clear=not mixed and j is not None and counts[j]>=20 and counts[j]/len(pp)>=.9
                kind='mixed' if mixed else 'clear' if clear else 'unknown';assert meta['kind']==kind
                kinds[int(key)]=(kind,int(ids[j]) if j is not None else None);checks['full_endpoint_labels']+=1
        features=a.read(r.OUT/'scene_data'/s/'features'/s/'rows.json');labels={q['pair_id']:q['relation'] for q in lab['pairs']}
        for q in features:
            l,h=kinds[q['left']],kinds[q['right']]
            rel=('same' if l[1]==h[1] else 'different') if l[0]==h[0]=='clear' else 'mixed' if 'mixed' in (l[0],h[0]) else 'unknown'
            assert labels[q['pair_id']]==rel;look[s,q['pair_id']]=rel;checks['pair_labels']+=1
    idx=np.array([i for i,q in enumerate(tr) if look[q['scene'],q['pair_id']] in ('same','different')]);y=np.array([int(look[tr[i]['scene'],tr[i]['pair_id']]=='same') for i in idx])
    z=np.load(r.OUT/'models/training.npz');np.testing.assert_array_equal(z['row_indices'],idx);np.testing.assert_array_equal(z['y'],y)
    count=Counter(tr[i]['scene'] for i in idx);expected=np.array([len(idx)/len(count)/count[tr[i]['scene']] for i in idx])
    np.testing.assert_allclose(z['weights'],expected,rtol=1e-13,atol=1e-13);checks['scene_weight_mass']=len(count)
    assert a.sha(r.OUT/'models/reference82.joblib')==a.sha(r.SMALL/'models/fold0/double.joblib')
    assert a.read(r.OUT/'models/receipt.json')['fits']==1 and len(list((r.OUT/'models').glob('*.joblib')))==2
    joined={arm:a.read(r.OUT/'evaluation'/(arm+'_pairs.json')) for arm in r.ARMS};targets={arm:{(q['scene'],q['gt_id']):q for q in a.read(r.OUT/'evaluation'/(arm+'_targets.json'))} for arm in r.ARMS}
    diag=a.read(r.OUT/'learning_diagnostics.json')
    for arm in r.ARMS:
        model=joblib.load(r.OUT/'models'/(arm+'.joblib'));assert all(model.get_params()[k]==v for k,v in c['model'].items())
        with threadpool_limits(4):p=model.predict_proba(x)[:,1];tp=model.predict_proba(tx)[:,1]
        np.testing.assert_array_equal(p,np.load(r.OUT/(arm+'_scores.npy')));np.testing.assert_array_equal(tp,np.load(r.OUT/(arm+'_train_scores.npy')))
        checks['models_replayed_no_fit']+=1
        ix=np.array([i for i,q in enumerate(rr) if look[q['scene'],q['pair_id']] in ('same','different') and x[i,4]==0])
        yy=np.array([int(look[rr[i]['scene'],rr[i]['pair_id']]=='same') for i in ix]);v=np.clip(p[ix],1e-12,1-1e-12);loss=-yy*np.log(v)-(1-yy)*np.log1p(-v)
        assert abs((loss[yy==1].mean()+loss[yy==0].mean())/2-diag['arms'][arm]['dev55']['disjoint']['scan']['balanced_logloss'])<1e-12
        pred={(q['scene'],q['pair_id']):q for q in a.read(r.OUT/(arm+'_decisions.json'))}
        for s in c['scenes']:
            ix=[i for i,q in enumerate(rr) if q['scene']==s];used=set();accept=set();emitted=set()
            for i in sorted(ix,key=lambda i:(-p[i],rr[i]['pair_id'])):
                q=rr[i]
                if q['eligible'] and p[i]>=.9 and not ({q['left'],q['right']}&used):used.update((q['left'],q['right']));accept.add(q['pair_id'])
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as unions,np.load(r.OUT/'geometry/predictions'/s/'data.npz') as source:
                for i in ix:
                    q=pred[s,rr[i]['pair_id']];assert q['score']==p[i]
                    assert q['state']==('accept' if q['pair_id'] in accept else 'reject' if q['eligible'] and p[i]<=.1 else 'unknown')
                    emit=q['state']=='accept' and not q['duplicate_original_NT'] and q['union_sha256'] not in emitted;assert q['emit']==emit
                    if emit:
                        emitted.add(q['union_sha256']);pp=unions[q['pair_id']+'_union'];np.testing.assert_array_equal(pp,np.union1d(source['n%04d'%q['left']],source['n%04d'%q['right']]))
                        assert a.digest(pp)==q['union_sha256'];checks['emitted_complete_unions']+=1
                assert set(unions.files)=={q['pair_id']+'_union' for q in pred.values() if q['scene']==s and q['emit']}
    classes=a.read(r.SPLIT)['class_ids'];supply={(q['scene'],q['gt_id']):q for q in a.read(r.OUT/'evaluation/supply_targets.json')}
    pair_rows={(q['scene'],q['pair_id']):q for q in a.read(r.OUT/'evaluation/supply_pairs.json')}
    def iou(pp,gt,ids,sizes,allow_small=False):
        hit=np.array([np.count_nonzero(gt[pp]==g) for g in ids]);return hit/(len(pp)+sizes-hit) if len(pp)>=100 or allow_small else np.zeros(len(ids))
    for s in c['scenes']:
        gt=np.loadtxt(c['jobs'][s]['job']['GT'],dtype=np.int64);ids,sizes=np.unique(gt,return_counts=True)
        keep=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,classes);ids,sizes=ids[keep],sizes[keep]
        rows=[q for q in pair_rows.values() if q['scene']==s]
        with np.load(r.OUT/'geometry/predictions'/s/'data.npz') as z:
            base=np.zeros(len(ids));nodes=a.read(r.OUT/'geometry/predictions'/s/'nodes.json')
            for n in nodes:
                if n['in_original_NT']:base=np.maximum(base,iou(z['n%04d'%n['node']],gt,ids,sizes))
            candidate_best=base.copy()
            for q in rows:
                v=iou(z[q['pair_id']+'_union'],gt,ids,sizes);candidate_best=np.maximum(candidate_best,v)
                np.testing.assert_array_equal(q['any_target_gain02_ids'],ids[v-base>=.02-1e-12])
                meta=[]
                for node in (q['left'],q['right']):
                    pp=z['n%04d'%node];vv=iou(pp,gt,ids,sizes,True);j=int(vv.argmax()) if len(vv) and vv.max()>0 else None
                    hit=np.count_nonzero(gt[pp]==ids[j]) if j is not None else 0
                    meta.append(dict(stable=bool(j is not None and hit>=20 and hit/len(pp)>=.6),j=j,parent_iou=float(vv[j]) if j is not None and len(pp)>=100 else 0))
                l,h=meta;rel=('same' if l['j']==h['j'] else 'different') if l['stable'] and h['stable'] else 'unresolved'
                assert q['GT_relation_previous']==rel
                safe=rel=='same' and (gt[z[q['pair_id']+'_union']]==ids[l['j']]).mean()>=.9 and v[l['j']]-base[l['j']]>=.02-1e-12
                assert q['safe_gain02']==safe
                damage=any(m['stable'] and m['parent_iou']>=.9 and m['parent_iou']-v[m['j']]>.02+1e-12 for m in meta)
                assert q['good_endpoint_loss_gt02']==damage;checks['candidate_opportunities_and_copy_risk']+=1
            for j,g in enumerate(ids):assert abs(supply[s,int(g)]['expanded']-candidate_best[j])<1e-12
            for arm in r.ARMS:
                best=base.copy()
                with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as unions:
                    for key in unions.files:best=np.maximum(best,iou(unions[key],gt,ids,sizes))
                for j,g in enumerate(ids):
                    q=targets[arm][s,int(g)];assert abs(q['baseline']-base[j])<1e-12 and abs(q['expanded']-best[j])<1e-12
                    checks['complete_NT_targets']+=1
        print('VERIFIED dev',s,flush=True)
    results=a.read(r.OUT/'results.json')['results']
    for arm in r.ARMS:
        accepted=[q for q in joined[arm] if q['state']=='accept'];emitted=[q for q in accepted if q['emit']]
        assert results[arm]['strict_accepted']==dict(Counter(q['GT_relation_strict'] for q in accepted))
        assert results[arm]['good_endpoint_loss_gt02']==sum(q['good_endpoint_loss_gt02'] for q in accepted)
        assert results[arm]['pool']['pool_gain02']==sum(q['delta']>=.02-1e-12 for q in targets[arm].values())
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r.HERE),'-p','test_fi1_relation_train200_dev55.py','-v'],capture_output=True,text=True);assert test.returncode==0,test.stderr
    a.seal(r.OUT/'verification.json',dict(complete=True,additional_fits=0,checks=dict(checks),tests=test.stderr,
        limitations='One fixed82/200 comparison on development55 scans/29 sites. Not final holdout and no AP. Source cache corrections retained explicitly; original masks/candidates/geometry unchanged.'))
    print('VERIFIED',dict(checks),flush=True)


if __name__=='__main__':main()

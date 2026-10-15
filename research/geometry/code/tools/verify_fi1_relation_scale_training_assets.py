#!/usr/bin/env python3
"""Additional independent checks of newly prepared training assets, without fits."""
from collections import Counter
import numpy as np
import run_fi1_relation_training_scale_checked as entry

r=entry.r


def main():
    r.guard(gt=True);c=r.contract();a=r.a
    for name in ('features_frozen.json','labels_frozen.json'):r.verify(a.read(r.OUT/'training_data'/name)['outputs'])
    rows=[];g=[];v=[];checks=Counter();labelcounts=Counter()
    for s in c['extra_scenes']:
        d=r.OUT/'training_data/features'/s
        for name in ('features.npz','rows.json','receipt.json'):
            assert a.sha(d/name)==a.sha(r.OUT/'feature_cache/features'/s/name)
        sr=a.read(d/'rows.json');rows+=sr
        with np.load(d/'features.npz') as z:g.extend(z['geometry']);v.extend(z['rgbd'])
        gp=c['jobs'][s]['job']['GT'];assert a.sha(gp)==c['jobs'][s]['GT_sha256']
        gt=np.loadtxt(gp,dtype=np.int64)
        assert len(gt)==c['jobs'][s]['job']['binding']['point_count']
        labs=a.read(r.OUT/'training_data/labels'/f'{s}.json');identities={}
        with np.load(r.OUT/'training_geometry/predictions'/s/'data.npz') as z:
            for key,old in labs['nodes'].items():
                pp=z['n%04d'%int(key)];ids,counts=np.unique(gt[pp],return_counts=True)
                assert old['distribution']=={str(i):int(n) for i,n in zip(ids,counts)}
                known=ids>=1000;multi=np.count_nonzero(known&(counts>=20))>=2
                j=np.argmax(np.where(known,counts,0)) if known.any() else None
                clear=not multi and j is not None and counts[j]>=20 and counts[j]/len(pp)>=.9
                kind='mixed' if multi else 'clear' if clear else 'unknown'
                assert old['kind']==kind
                identities[int(key)]=(kind,int(ids[j]) if j is not None else None)
                checks['new_endpoint_labels_from_raw_GT']+=1
            lm={q['pair_id']:q['relation'] for q in labs['pairs']}
            for q in sr:
                left,right=identities[q['left']],identities[q['right']]
                rel=('same' if left[1]==right[1] else 'different') if left[0]==right[0]=='clear' else ('mixed' if 'mixed' in (left[0],right[0]) else 'unknown')
                assert rel==lm[q['pair_id']];labelcounts[rel]+=1
                union=np.union1d(z['n%04d'%q['left']],z['n%04d'%q['right']])
                np.testing.assert_array_equal(z[q['pair_id']+'_union'],union)
                assert a.digest(union)==q['union_sha256'];checks['new_pair_label_and_full_union']+=1
        checks['new_scene_feature_files_exact']=checks['new_scene_feature_files_exact']+1
    assert rows==a.read(r.OUT/'training_data/feature_rows.json')
    assert len({(q['scene'],q['pair_id']) for q in rows})==len(rows)
    with np.load(r.OUT/'training_data/features.npz') as z:
        np.testing.assert_array_equal(z['geometry'],np.asarray(g));np.testing.assert_array_equal(z['rgbd'],np.asarray(v))
    diag=a.read(r.OUT/'learning_diagnostics.json');steps=[]
    for left,right in zip(r.ARMS[:-1],r.ARMS[1:]):
        lc,rc=(diag['arms'][arm]['pooled']['heldout'] for arm in (left,right))
        lower=sum(diag['arms'][right]['folds'][str(f)]['heldout']['disjoint']['balanced_logloss']<diag['arms'][left]['folds'][str(f)]['heldout']['disjoint']['balanced_logloss'] for f in range(5))
        checks0=dict(lower_balanced_logloss=rc['balanced_logloss']<lc['balanced_logloss'],at_least3_folds_lower=lower>=3,
            different_high90_not_higher=rc['negative']['high90_fraction']<=lc['negative']['high90_fraction'],
            same_high90_not_lower=rc['positive']['high90_fraction']>=lc['positive']['high90_fraction'])
        steps.append(dict(left=left,right=right,checks=checks0,lower_folds=lower,passed=all(checks0.values())))
    assert steps==diag['steps']
    a.seal(r.OUT/'training_assets_verified.json',dict(complete=True,additional_fits=0,checks=dict(checks),
        training_pairs=len(rows),labels=dict(labelcounts),new_features_exact=True,
        supplement_code={str(r.HERE/'verify_fi1_relation_scale_training_assets.py'):a.sha(__file__)},
        no_model_or_threshold_selection=True))
    print('TRAINING_ASSETS_VERIFIED',dict(checks),dict(labelcounts),flush=True)


if __name__=='__main__':main()

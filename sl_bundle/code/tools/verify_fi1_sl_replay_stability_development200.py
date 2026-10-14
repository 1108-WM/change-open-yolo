#!/usr/bin/env python3
"""Postseal audit: seeds/folds, weights, frozen method, identities and AP stats."""
from pathlib import Path
from collections import Counter
import inspect,sys,warnings
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_sl_replay_stability_development200 as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def main():
    out=run.OUT;c=run.read(out/'contract.json');assert run.read(out/'summary.json')['complete']
    run.ctl.base.verify_files(c['code']);run.ctl.base.verify_files(c['original_models'])
    run.ctl.base.verify_files(c['inputs']);run.ctl.base.verify_files(c['label_inputs'])
    assert c['seeds']==[20261010,20261011,20261012]
    assert __import__('hashlib').sha256(inspect.getsource(run.original.fit).encode()).hexdigest()==c['original_fit_source_sha256']
    ts=np.asarray(run.ctl.base.evaluator().opt['overlaps']);counts=Counter();stats={}
    for seed in c['seeds']:
        root=run.arm(seed);seal=run.read(root/'predictions_frozen.json');run.ctl.base.verify_files(seal['outputs'])
        assert run.read(root/'inference_verified.json')['all_selected_positive']
        assert run.read(root/'coverage_verified.json')['recovered424']==424
        models=run.L/'models' if seed==c['seeds'][0] else root/'models'
        stats[seed]={}
        for fold in range(5):
            d=models/f'fold{fold}';r=run.read(d/'receipt.json');run.ctl.base.verify_files(r['outputs'])
            train=[s for s in c['scenes'] if c['folds'][s]!=fold];test=[s for s in c['scenes'] if c['folds'][s]==fold]
            assert train==r['train_scenes'] and test==r['test_scenes'] and len(train)==160 and len(test)==40
            assert not {s[:9] for s in train}&{s[:9] for s in test} and not r['heldout_labels_opened']
            with np.load(d/'normalization.npz') as z:
                with np.load(run.L/'models'/f'fold{fold}'/'normalization.npz') as ref:
                    assert all(np.array_equal(z[k],ref[k]) for k in ('mean','std'))
            t=run.read(d/'training_metrics.json');assert [h['epoch'] for h in t['history']]==list(range(1,21))
            with np.load(d/'training_pairs.npz') as z:pairs=dict(z)
            assert np.isfinite(pairs['weights']).all() and abs(pairs['weights'].sum()-1)<1e-12
            # Label/ID arrays are used solely to verify sampling; never inferred inputs.
            a=run.load_training_scenes(c,train);p,n=pairs['pairs'].T
            assert np.array_equal(a['classes'][p],a['classes'][n])
            assert ((a['S'][p]>0)&(a['S'][p]<1)&(a['S'][n]>0)&(a['S'][n]<1)).all()
            active=np.unique(a['classes'][p]);weights={int(ci):float(pairs['weights'][a['classes'][p]==ci].sum()) for ci in active}
            assert all(abs(w-1/len(active))<1e-12 for w in weights.values())
            expected=run.original.core.make_pairs(a['S'],a['classes'],a['scene_ids'],a['identity'],a['states'],seed+fold)
            assert all(np.array_equal(pairs[k],expected[k]) for k in pairs)
            # Recompute training-only normalization, independently of saved parameters.
            with np.load(d/'normalization.npz') as normalization:
                assert np.array_equal(a['x'].mean(0,dtype=np.float64).astype(np.float32),normalization['mean'])
                assert np.array_equal(np.maximum(a['x'].std(0,dtype=np.float64),1e-6).astype(np.float32),normalization['std'])
            stats[seed][fold]=dict(training_pairs=len(p),active_classes=len(active),epochs=20,seed=seed+fold)
            counts['models']=counts['models']+1
            print('model_verified',seed,fold,flush=True)
            del a
        for s in c['scenes']:
            old=run.read(run.S/'predictions'/s/'semantic_S.json');new=run.read(root/'predictions'/s/'semantic_SL.json')
            def identity(m):return {k:v for k,v in m.items() if k!='score'}
            assert {run.ctl.shared.key(m):identity(m) for m in old}=={run.ctl.shared.key(m):identity(m) for m in new}
            assert all(m['score']>0 for m in new)
            best={}
            for m in new:
                k=m['geometry_key']
                if k not in best or (m['score'],-m['class_index'])>(best[k]['score'],-best[k]['class_index']):best[k]=m
            expected=sorted([dict(geometry_key=k,score=m['score'],winning_class_index=m['class_index']) for k,m in best.items()],key=lambda m:(-m['score'],m['geometry_key']))
            assert expected==run.read(root/'predictions'/s/'class_agnostic_SL.json')
            counts['members']+=800;counts['CA_geometries']+=len(expected)
            run.ctl.base.verify_files(run.read(root/'evaluation'/s/'receipt.json')['outputs'])
        summary=run.read(root/'evaluation_summary.json');run.ctl.base.verify_files(summary['AP_arrays'])
        for group in c['reporting_groups']:
            for v in run.original.VIEWS:
                arr=np.load(root/'evaluation'/f'{group}_{v}_SL_AP.npy')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    metrics=dict(ap=float(np.nanmean(arr[0,:,~np.isclose(ts,.25)])),ap50=float(np.nanmean(arr[0,:,np.isclose(ts,.5)])),ap25=float(np.nanmean(arr[0,:,np.isclose(ts,.25)])))
                assert all(abs(metrics[k]-summary['AP'][group][v]['SL'][k])<1e-14 for k in metrics)
                counts['AP_aggregates']+=1
    summary=run.read(out/'summary.json')
    for group in c['reporting_groups']:
        for v in run.original.VIEWS:
            for k in ('ap','ap50','ap25'):
                values=np.array([run.read(run.arm(seed)/'evaluation_summary.json')['AP'][group][v]['SL'][k] for seed in c['seeds']]);z=summary['groups'][group][v][k]
                assert values.tolist()==z['values'] and float(values.mean())==z['mean'] and float(values.min())==z['min'] and float(values.max())==z['max']
    assert run.read(out/'replay/AP_replay_verified.json')['arrays_exact']==16
    files={str(q):sha(q) for q in (out/'summary.json',out/'report.md',out/'contract.json')}
    freeze_json(out/'independent_verified.json',dict(complete=True,counts=dict(counts),models=stats,
        source_method_unchanged=True,train_only_normalization_exact=True,training_pair_regeneration_exact=True,
        three_seed_aggregation_exact=True,original_replay_exact=True,outputs=files,verifier_sha256=sha(__file__),raw_GT_read=False))
    print('independent verification complete',dict(counts),flush=True)


if __name__=='__main__':main()

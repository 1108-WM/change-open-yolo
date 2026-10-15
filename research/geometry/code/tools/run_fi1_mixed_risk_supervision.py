#!/usr/bin/env python3
"""One fixed three-class objective comparison; five fits, no risk cutoff."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import audit_fi1_t_source_chain as a
import run_fi1_relation_controls as controls
import fi1_pair_relation_core as relation
import fi1_relation_controls_core as metrics

HERE = Path(__file__).resolve().parent
SCALE = a.EXP/'fi1_relation_training_scale_exposed52_20261015_v2'
BASE, GEOMETRY, AUDIT = controls.BASE, controls.GEOMETRY, a.OUT
OUT = a.EXP/'fi1_mixed_risk_supervision_exposed52_20261015_v1'
CLASSES = dict(different=0, same=1, mixed=2, unknown=-1)
ARMS = ('binary', 'mixed')


def guard(labels=(), gt=False):
    allowed = {str(Path(p).resolve()) for p in labels}
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(k in mode for k in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not p.is_relative_to(OUT):
            raise PermissionError('Writes restricted to new experiment: '+str(p))
        if not gt and any(k in str(p) for k in ('ground_truth', '/labels/', '/evaluation/',
                'opportunity_diagnostics', '_evidence.csv', '/results.json')) and str(p) not in allowed:
            raise PermissionError('Labels/outcomes forbidden in this stage: '+str(p))
    sys.addaudithook(hook)


def verify(refs):
    for p, h in refs.items():
        assert a.sha(p) == h, p


def contract():
    c = a.read(OUT/'contract.json')
    verify(c['code']); verify(c['source_files'])
    return c


def combined():
    old = controls.rows()
    new = a.read(SCALE/'training_data/feature_rows.json')
    xold, _ = controls.features()
    with np.load(SCALE/'training_data/features.npz') as z:
        xnew = np.c_[z['geometry'], z['rgbd']]
    return old+new, np.r_[xold, xnew], len(old)


def labelpath(s, c):
    return (BASE if s in c['scenes'] else SCALE/'training_data')/'labels'/f'{s}.json'


def labels_for(scenes, c, rr):
    lookup = {}
    for s in scenes:
        p = labelpath(s, c)
        assert a.sha(p) == c['label_files'][str(p)]
        lookup.update({(s, q['pair_id']):CLASSES[q['relation']] for q in a.read(p)['pairs']})
    return np.array([lookup.get((q['scene'], q['pair_id']), -1) for q in rr])


def probability_columns(model, x):
    p = model.predict_proba(x)
    assert set(model.classes_) == {0, 1, 2}
    return p[:, [list(model.classes_).index(k) for k in (0, 1, 2)]]


def prepare():
    guard(); a.reserve(); assert not OUT.exists()
    c = a.read(SCALE/'contract.json')
    delivery = a.read(SCALE/'delivery_verified.json'); assert delivery['complete']
    refs = {str(SCALE/n):a.sha(SCALE/n) for n in ('contract.json', 'delivery_verified.json',
        'predictions_frozen.json', 'training_assets_verified.json', 'training_data/features_frozen.json',
        'training_data/labels_frozen.json')}
    for p, h in delivery['outputs'].items():
        path = Path(p)
        if path.name in ('double_scores.npz', 'double_decisions.json') or '/predictions/double/' in p or (
                '/models/' in p and path.name.startswith('double')) or p.endswith('training_data/features.npz') or p.endswith('training_data/feature_rows.json'):
            refs[p] = h
    for root, names in ((BASE, ('features.npz', 'feature_rows.json', 'labels_frozen.json')),
                        (GEOMETRY, ('predictions_frozen.json',))):
        refs.update({str(root/n):a.sha(root/n) for n in names})
    geom = a.read(GEOMETRY/'predictions_frozen.json')['outputs']
    for p, h in geom.items():
        if Path(p).name in ('data.npz', 'nodes.json', 'pairs.json'): refs[p] = h
    verify(refs)
    labelrefs = dict(c['label_files'])
    labelrefs.update(a.read(SCALE/'training_data/labels_frozen.json')['outputs'])
    code = [HERE/n for n in ('run_fi1_mixed_risk_supervision.py', 'evaluate_fi1_mixed_risk_supervision.py',
        'verify_fi1_mixed_risk_supervision.py', 'test_fi1_mixed_risk_supervision.py',
        'fi1_pair_relation_core.py', 'fi1_relation_controls_core.py', 'run_fi1_relation_controls.py',
        'audit_fi1_t_source_chain.py', 'evaluate_fi1_nt_region_identity.py')]
    a.seal(OUT/'contract.json', dict(version=OUT.name, time_unix=time.time(),
        scenes=c['scenes'], extra_scenes=c['extra_scenes'], folds=c['folds'], splits=c['splits'],
        model=c['model'], class_mapping=CLASSES, label_files=labelrefs, source_files=refs,
        code={str(p):a.sha(p) for p in code}, feature_dim=154, evaluation_pairs=4865,
        baseline='Reuse sealed scale-v2 double binary scores, decisions, models and whole unions; zero refits.',
        objective='Three-class multinomial HGB: clear different0, clear same1, mixed2; unknown excluded. Complete endpoint labels unchanged. Necessary class-specific tree expansion, no capacity search.',
        weights='Exact C recipe over supervised rows: equal scene total, natural three-class proportions, sum weights equals fitting row count. No class/relation/risk balancing; adding mixed rows necessarily changes per-row and class masses.',
        actions='Use raw P(same)>=0.9 propose accept; raw P(same)<=0.1 reject, not cannot-link; otherwise unknown. Original eligibility, descending same probability then pair_id greedy disjoint partner matching; whole unions; deduplicate original NT and emitted geometry. No mixed-risk cutoff or renormalization.',
        diagnostics='Postseal only: clear zero-overlap train/common-clear/OOF binary diagnostics on raw P(same); mixed scores, three-class confusion and loss; unknown unlabelled; all endpoints for output risk. 18 opportunities only posthoc, never tuning.',
        classification_checks='Lower difficult OOF balanced loss in pooled and>=3/5 folds; lower clear-neighbour high90; correct complementary high90 not lower. Report flags separately from geometry gates.',
        geometry_gates=c['geometry_gates'], isolation=c['isolation'], all52_exposed_development=True,
        limits=dict(fits=5, baseline_refits=0, threshold_search=False, feature_change=False,
            further_expansion=False, AP=False, production_modified=False, old_seals_modified=False)))
    for p in code: shutil.copyfile(p, OUT/('code_'+p.name))
    (OUT/'tmp').mkdir()
    print('FROZEN', OUT, flush=True)


def fit(fold):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    c = a.read(OUT/'contract.json'); scenes = c['splits'][str(fold)]['double']
    guard([labelpath(s, c) for s in scenes]); c = contract(); start = time.time()
    rr, x, _ = combined(); assert x.shape == (8916, 154)
    y = labels_for(scenes, c, rr); idx = np.flatnonzero(y >= 0)
    w = metrics.weights([rr[i]['scene'] for i in idx], x[idx, 4]>0, y[idx])
    d = OUT/'models'/f'fold{fold}'; d.mkdir(parents=True)
    model = HistGradientBoostingClassifier(**c['model'])
    with threadpool_limits(4):
        model.fit(x[idx], y[idx], sample_weight=w)
        prob = probability_columns(model, x)
    joblib.dump(model, d/'mixed.joblib')
    np.savez_compressed(d/'training.npz', row_indices=idx, y=y[idx], weights=w, classes=model.classes_)
    np.save(d/'probabilities.npy', prob)
    a.seal(d/'receipt.json', dict(fold=fold, fits=1, seconds=time.time()-start, scenes=scenes,
        effective_scenes=sorted({rr[i]['scene'] for i in idx}), rows=len(idx), counts=dict(Counter(y[idx].tolist())),
        label_inputs={str(labelpath(s,c)):c['label_files'][str(labelpath(s,c))] for s in scenes},
        outputs={str(p):a.sha(p) for p in d.iterdir()}))
    print('FIT',fold,len(idx),Counter(y[idx]),'seconds',time.time()-start,flush=True)


def predict():
    guard(); c = contract(); rr = controls.rows(); n = len(rr); outputs = {}
    probs = []
    for f in range(5):
        d = OUT/'models'/f'fold{f}'; rec = a.read(d/'receipt.json'); verify(rec['outputs'])
        outputs.update(rec['outputs']); outputs[str(d/'receipt.json')] = a.sha(d/'receipt.json')
        probs.append(np.load(d/'probabilities.npy')[:n])
    probs = np.asarray(probs); folds = np.array([c['folds'][q['scene']] for q in rr])
    oof = probs[folds, np.arange(n)]
    np.savez_compressed(OUT/'mixed_scores.npz', fold_all=probs, oof=oof)
    for name, source in (('binary_scores.npz','double_scores.npz'),('binary_decisions.json','double_decisions.json')):
        shutil.copyfile(SCALE/source,OUT/name)
    allpred = []
    for s in c['scenes']:
        dest = OUT/'predictions/mixed'/s; dest.mkdir(parents=True)
        ix = [i for i,q in enumerate(rr) if q['scene']==s]
        pred = relation.decide([rr[i] for i in ix], oof[ix,1], [rr[i]['eligible'] for i in ix])
        emitted, arrays = set(), {}
        with np.load(GEOMETRY/'predictions'/s/'data.npz') as z:
            for i,q in zip(ix,pred):
                q.update(same_probability=float(oof[i,1]), different_probability=float(oof[i,0]), mixed_probability=float(oof[i,2]))
                if q['state']!='accept': continue
                pp = z[q['pair_id']+'_union']; assert a.digest(pp)==q['union_sha256']
                if q['duplicate_original_NT'] or q['union_sha256'] in emitted:
                    q['reason'] += '_duplicate_geometry_not_emitted'; continue
                emitted.add(q['union_sha256']); arrays[q['pair_id']+'_union']=pp; q['emit']=True
        np.savez_compressed(dest/'unions.npz', **arrays); a.seal(dest/'pairs.json',pred); allpred.extend(pred)
        olddest = OUT/'predictions/binary'/s; olddest.mkdir(parents=True)
        for name in ('pairs.json','unions.npz'): shutil.copyfile(SCALE/'predictions/double'/s/name,olddest/name)
        outputs.update({str(p):a.sha(p) for root in (dest,olddest) for p in root.iterdir()})
    a.seal(OUT/'mixed_decisions.json',allpred)
    for name in ('mixed_scores.npz','mixed_decisions.json','binary_scores.npz','binary_decisions.json'):
        outputs[str(OUT/name)]=a.sha(OUT/name)
    a.seal(OUT/'predictions_frozen.json',dict(complete=True,time_unix=time.time(),GT_used=False,
        fitted_models=5,baseline_exact_reused=True,outputs=outputs))
    print('PREDICT',Counter(q['state'] for q in allpred),'emitted',sum(q['emit'] for q in allpred),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','fit','predict'));p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    if args.mode=='fit': assert args.fold is not None; fit(args.fold)
    else: globals()[args.mode]()

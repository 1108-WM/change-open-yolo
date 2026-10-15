#!/usr/bin/env python3
"""Audit C reuse, train-only physical grouping, invariant masses and output IoU."""
from collections import Counter, defaultdict
import json
import subprocess
import sys
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
import run_fi1_physical_relation_balance as r


def main():
    r.guard(False)
    c = r.contract()
    for name in ('predictions_frozen.json', 'evaluation_frozen.json'):
        r.previous.prior.verify(r.a.read(r.OUT/name)['outputs'])
    r.previous.prior.verify(c['label_files'])
    rr = r.previous.rows()
    x, overlap = r.previous.features()
    folds = np.array([c['folds'][q['scene']] for q in rr])
    lookup, labels = {}, {}
    for s in c['scenes']:
        labels[s] = r.a.read(r.BASE/'labels'/f'{s}.json')
        lookup.update({(s, q['pair_id']): q['relation'] for q in labels[s]['pairs']})
    y = np.array([1 if lookup[q['scene'], q['pair_id']] == 'same' else 0 if lookup[q['scene'], q['pair_id']] == 'different' else -1 for q in rr])
    allscore = {arm: np.load(r.OUT/f'{arm}_scores.npz')['fold_all'] for arm in r.ARMS}
    checks = Counter()
    maximum_mass_error = 0.
    assert len(list((r.OUT/'models').glob('fold*/D.joblib'))) == 5
    assert not list((r.OUT/'models').glob('fold*/C.joblib'))
    for name in ('C_scores.npz', 'C_decisions.json'):
        assert r.a.sha(r.OUT/name) == r.a.sha(r.SOURCE/name)
    for fold in range(5):
        d = r.OUT/'models'/f'fold{fold}'
        receipt = r.a.read(d/'receipt.json')
        idx = np.flatnonzero((y >= 0) & (folds != fold))
        assert receipt['training_rows'] == idx.tolist()
        assert {s.rsplit('_', 1)[0] for s in receipt['train_scenes']}.isdisjoint({s.rsplit('_', 1)[0] for s in receipt['test_scenes']})
        assert set(receipt['label_inputs']) == {str(r.BASE/'labels'/f'{s}.json') for s in receipt['train_scenes']}
        r.previous.prior.verify(receipt['label_inputs'])
        origin = np.load(r.SOURCE/'models'/f'fold{fold}'/'training.npz')
        actual = np.load(d/'training.npz')
        np.testing.assert_array_equal(idx, origin['row_indices'])
        np.testing.assert_array_equal(idx, actual['row_indices'])
        np.testing.assert_array_equal(y[idx], actual['y'])
        np.testing.assert_array_equal(origin['weights_C'], actual['weights_C'])
        keys = []
        for i in idx:
            q = rr[i]
            n = labels[q['scene']]['nodes']
            left, right = n[str(q['left'])], n[str(q['right'])]
            assert left['kind'] == right['kind'] == 'clear'
            aa, bb = sorted((int(left['target']), int(right['target'])))
            assert (aa == bb) == bool(y[i])
            keys.append((q['scene'], int(y[i]), aa, bb))
        assert keys == [tuple(json.loads(v)) for v in actual['group_keys']]
        categories = {(k[0], k[1]) for k in keys}
        expected = np.empty(len(idx))
        for category in sorted(categories):
            positions = [j for j, k in enumerate(keys) if k[:2] == category]
            groups = {keys[j] for j in positions}
            counts = Counter(keys[j] for j in positions)
            mass = origin['weights_C'][positions].sum()
            for j in positions:
                expected[j] = mass/len(groups)/counts[keys[j]]
            error = abs(actual['weights_D'][positions].sum()-mass)
            assert error <= 1e-11
            maximum_mass_error = max(maximum_mass_error, error)
            group_masses = [actual['weights_D'][[j for j in positions if keys[j] == k]].sum() for k in groups]
            np.testing.assert_allclose(group_masses, mass/len(groups), rtol=0, atol=1e-11)
            checks['scene_class_mass_and_group_balance'] += 1
        np.testing.assert_allclose(actual['weights_D'], expected, rtol=0, atol=1e-12)
        for s in set(k[0] for k in keys):
            ii = [j for j, k in enumerate(keys) if k[0] == s]
            assert abs(actual['weights_D'][ii].sum()-origin['weights_C'][ii].sum()) <= 1e-11
            checks['scene_mass_preserved'] += 1
        model = joblib.load(d/'D.joblib')
        cmodel = joblib.load(r.SOURCE/'models'/f'fold{fold}'/'C.joblib')
        assert model.get_params() == cmodel.get_params()
        with threadpool_limits(4):
            score = model.predict_proba(x)[:, 1]
        np.testing.assert_array_equal(score, allscore['D'][fold])
        checks['D_models_verified_without_additional_fitting'] += 1
        print('verify D weights/model', fold, flush=True)
    diag = r.a.read(r.OUT/'learning_diagnostics.json')
    for arm in r.ARMS:
        oof = np.load(r.OUT/f'{arm}_scores.npz')['oof']
        np.testing.assert_array_equal(oof, allscore[arm][folds, np.arange(len(rr))])
        for part in ('train', 'heldout'):
            yy, pp = [], []
            for fold in range(5):
                mask = (y >= 0) & ~overlap & ((folds != fold) if part == 'train' else (folds == fold))
                v, target = allscore[arm][fold, mask], y[mask]
                yy.extend(target);pp.extend(v)
                p = np.clip(v, 1e-12, 1-1e-12)
                loss = -target*np.log(p)-(1-target)*np.log1p(-p)
                balanced = float((loss[target == 1].mean()+loss[target == 0].mean())/2)
                assert abs(balanced-diag['arms'][arm]['folds'][str(fold)][part]['disjoint']['balanced_logloss']) < 1e-12
            target, score = np.asarray(yy), np.asarray(pp)
            p = np.clip(score, 1e-12, 1-1e-12)
            loss = -target*np.log(p)-(1-target)*np.log1p(-p)
            m = diag['arms'][arm]['pooled'][part]['disjoint']
            assert abs((loss[target == 1].mean()+loss[target == 0].mean())/2-m['balanced_logloss']) < 1e-12
            assert m['positive']['high90'] == int(((score >= .9) & (target == 1)).sum())
            assert m['negative']['high90'] == int(((score >= .9) & (target == 0)).sum())
            checks['difficult_training_or_OOF_metrics_recomputed'] += 1
        pred = r.a.read(r.OUT/f'{arm}_decisions.json')
        for s in c['scenes']:
            ii = [i for i, q in enumerate(rr) if q['scene'] == s]
            used, accepted = set(), set()
            for i in sorted(ii, key=lambda j: (-oof[j], rr[j]['pair_id'])):
                q = rr[i]
                if q['eligible'] and oof[i] >= .9 and q['left'] not in used and q['right'] not in used:
                    used.update([q['left'], q['right']]);accepted.add(q['pair_id'])
            assert accepted == {pred[i]['pair_id'] for i in ii if pred[i]['state'] == 'accept'}
            d = r.OUT/'predictions'/arm/s
            if arm == 'C':
                for name in ('pairs.json', 'unions.npz'):
                    assert r.a.sha(d/name) == r.a.sha(r.SOURCE/'predictions/C'/s/name)
            with np.load(d/'unions.npz') as z, np.load(r.GEOMETRY/'predictions'/s/'data.npz') as original:
                assert set(z.files) == {pred[i]['pair_id']+'_union' for i in ii if pred[i]['emit']}
                for i in ii:
                    q = pred[i]
                    assert q['score'] == oof[i]
                    if q['state'] == 'reject':
                        assert q['eligible'] and q['score'] <= .1
                    if q['emit']:
                        pp = z[q['pair_id']+'_union']
                        np.testing.assert_array_equal(pp, np.union1d(original['n%04d' % q['left']], original['n%04d' % q['right']]))
                        assert r.a.digest(pp) == q['union_sha256']
                        checks['unchanged_emitted_unions'] += 1
    # Independent point-count IoU reconstruction, including complete original NT.
    jobs = r.a.read(r.a.OUT/'contract.json')['jobs']
    classes = r.a.read(r.a.OLD/'contract.json')['class_ids']
    targets = {arm: {(q['scene'], q['gt_id']): q for q in r.a.read(r.OUT/'evaluation'/f'{arm}_targets.json')} for arm in r.ARMS}
    def iou(pp, gt, ids, sizes):
        hit = np.array([np.count_nonzero(gt[pp] == g) for g in ids])
        return hit/(len(pp)+sizes-hit) if len(pp) >= 100 else np.zeros(len(ids))
    for s in c['scenes']:
        gt = np.loadtxt(jobs[s]['job']['GT'], dtype=np.int64)
        ids, sizes = np.unique(gt, return_counts=True)
        keep = (ids >= 1000) & (sizes >= 100) & np.isin(ids//1000, classes)
        ids, sizes = ids[keep], sizes[keep]
        base = np.zeros(len(ids))
        with np.load(r.GEOMETRY/'predictions'/s/'data.npz') as z:
            for n in r.a.read(r.GEOMETRY/'predictions'/s/'nodes.json'):
                if n['in_original_NT']:
                    base = np.maximum(base, iou(z['n%04d' % n['node']], gt, ids, sizes))
        for arm in r.ARMS:
            best = base.copy()
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as z:
                for key in z.files:
                    best = np.maximum(best, iou(z[key], gt, ids, sizes))
            for j, g in enumerate(ids):
                q = targets[arm][s, int(g)]
                assert abs(q['baseline']-base[j]) < 1e-12 and abs(q['expanded']-best[j]) < 1e-12
                checks['full_NT_target_rows_recomputed'] += 1
    left, right = (diag['arms'][arm]['pooled']['heldout']['disjoint'] for arm in r.ARMS)
    better = sum(diag['arms']['D']['folds'][str(f)]['heldout']['disjoint']['balanced_logloss'] <
                 diag['arms']['C']['folds'][str(f)]['heldout']['disjoint']['balanced_logloss'] for f in range(5))
    passed = (right['balanced_logloss'] < left['balanced_logloss'] and better >= 3 and
              right['negative']['high90_fraction'] < left['negative']['high90_fraction'] and
              right['positive']['high90_fraction'] >= left['positive']['high90_fraction'])
    assert passed == diag['relation_diagnosis_passed']
    assert passed == r.a.read(r.OUT/'results.json')['relation_diagnosis_passed']
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(r.HERE),
                           '-p', 'test_fi1_physical_relation_balance.py', '-v'], capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    r.a.seal(r.OUT/'verification.json', dict(complete=True, checks=dict(checks), max_mass_error=maximum_mass_error,
        test_stdout=tests.stdout, test_stderr=tests.stderr, additional_model_fits=0,
        limitations='Same frozen features and prior certified union-composition/identity table reused. Group formula, masses, diagnostic losses, matching and complete NT IoU independently reconstructed. No independent scenes or AP.'))
    print('VERIFIED', checks, 'max mass error', maximum_mass_error, flush=True)


if __name__ == '__main__':
    main()

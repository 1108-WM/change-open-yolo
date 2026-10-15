#!/usr/bin/env python3
"""Independent weight reconstruction, fixed-model refit, exact output and pool checks."""
from collections import Counter
import subprocess
import sys
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_fi1_relation_controls as r


def main():
    r.guard(False)
    c = r.contract()
    for name in ('predictions_frozen.json', 'evaluation_frozen.json'):
        r.prior.verify(r.a.read(r.OUT/name)['outputs'])
    r.prior.verify(c['label_files'])
    rr = r.rows()
    x, overlap = r.features()
    lookup = {}
    for s in c['scenes']:
        lookup.update({(s, q['pair_id']): q['relation'] for q in r.a.read(r.BASE/'labels'/f'{s}.json')['pairs']})
    label = np.array([1 if lookup[q['scene'], q['pair_id']] == 'same' else 0 if lookup[q['scene'], q['pair_id']] == 'different' else -1 for q in rr])
    folds = np.array([c['folds'][q['scene']] for q in rr])
    scores = {arm: np.load(r.OUT/f'{arm}_scores.npz')['fold_all'] for arm in r.ARMS}
    checks = Counter()
    assert c['models']['A'] == c['models']['B']
    assert {k for k in c['models']['C'] if c['models']['C'][k] != c['models']['A'][k]} == {'max_leaf_nodes', 'max_depth', 'min_samples_leaf'}
    np.testing.assert_array_equal(np.load(r.OUT/'A_scores.npz')['oof'], np.load(r.BASE/'geometry_rgbd_oof.npy'))
    assert r.a.sha(r.OUT/'A_decisions.json') == r.a.sha(r.BASE/'geometry_rgbd_decisions.json')
    for fold in range(5):
        d = r.OUT/'models'/f'fold{fold}'
        receipt = r.a.read(d/'receipt.json')
        expected = np.flatnonzero((folds != fold) & (label >= 0))
        assert receipt['training_rows'] == expected.tolist()
        assert {s.rsplit('_', 1)[0] for s in receipt['train_scenes']}.isdisjoint({s.rsplit('_', 1)[0] for s in receipt['test_scenes']})
        assert set(receipt['label_inputs']) == {str(r.BASE/'labels'/f'{s}.json') for s in receipt['train_scenes']}
        old = r.a.read(r.BASE/'models'/f'fold{fold}'/'receipt.json')
        assert old['training_rows'] == expected.tolist()
        r.prior.verify(receipt['label_inputs'])
        train = np.load(d/'training.npz')
        np.testing.assert_equal(train['row_indices'], expected)
        np.testing.assert_equal(train['y'], label[expected])
        scene = [rr[i]['scene'] for i in expected]
        present = {s: {(bool(overlap[i]), int(label[i])) for i in expected if rr[i]['scene'] == s} for s in set(scene)}
        count = Counter((rr[i]['scene'], bool(overlap[i]), int(label[i])) for i in expected)
        a_weights = np.array([1/scene.count(rr[i]['scene']) for i in expected])
        b_weights = np.array([1/(len(present[rr[i]['scene']])*count[rr[i]['scene'], bool(overlap[i]), int(label[i])]) for i in expected])
        a_weights *= len(expected)/a_weights.sum()
        b_weights *= len(expected)/b_weights.sum()
        np.testing.assert_allclose(train['weights_A'], a_weights, rtol=0, atol=1e-14)
        np.testing.assert_allclose(train['weights_B'], b_weights, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(train['weights_A'], train['weights_C'])
        for arm, weight in (('B', b_weights), ('C', a_weights)):
            model = HistGradientBoostingClassifier(**c['models'][arm])
            with threadpool_limits(4):
                model.fit(x[expected], label[expected], sample_weight=weight)
                actual = model.predict_proba(x)[:, 1]
            np.testing.assert_allclose(actual, scores[arm][fold], rtol=0, atol=0)
            checks['fixed_models_independently_refit'] += 1
        checks['fold_splits_and_weight_vectors'] += 1
        print('verify fit', fold, flush=True)
    for arm in r.ARMS:
        oof = np.load(r.OUT/f'{arm}_scores.npz')['oof']
        np.testing.assert_array_equal(oof, scores[arm][folds, np.arange(len(rr))])
        pred = r.a.read(r.OUT/f'{arm}_decisions.json')
        for s in c['scenes']:
            ii = [i for i, q in enumerate(rr) if q['scene'] == s]
            used, accepted = set(), set()
            for i in sorted(ii, key=lambda j: (-oof[j], rr[j]['pair_id'])):
                q = rr[i]
                if q['eligible'] and oof[i] >= .9 and q['left'] not in used and q['right'] not in used:
                    used.update([q['left'], q['right']]);accepted.add(q['pair_id'])
            pp = [pred[i] for i in ii]
            assert accepted == {q['pair_id'] for q in pp if q['state'] == 'accept'}
            dest = r.OUT/'predictions'/arm/s
            if arm == 'A':
                for name in ('pairs.json', 'unions.npz'):
                    assert r.a.sha(dest/name) == r.a.sha(r.BASE/'predictions/geometry_rgbd'/s/name)
            with np.load(dest/'unions.npz') as emitted, np.load(r.GEOMETRY/'predictions'/s/'data.npz') as source:
                assert set(emitted.files) == {q['pair_id']+'_union' for q in pp if q['emit']}
                for q in pp:
                    assert q['score'] == oof[next(i for i in ii if rr[i]['pair_id'] == q['pair_id'])]
                    if q['state'] == 'reject':
                        assert q['eligible'] and q['score'] <= .1
                    if q['emit']:
                        points = emitted[q['pair_id']+'_union']
                        np.testing.assert_array_equal(points, np.union1d(source['n%04d' % q['left']], source['n%04d' % q['right']]))
                        assert r.a.digest(points) == q['union_sha256']
                        checks['unchanged_emitted_unions'] += 1
    # Separate intersection-count implementation: recompute original NT and target pool.
    jobs = r.a.read(r.a.OUT/'contract.json')['jobs']
    classes = r.a.read(r.a.OLD/'contract.json')['class_ids']
    targets = {arm: {(q['scene'], q['gt_id']): q for q in r.a.read(r.OUT/'evaluation'/f'{arm}_targets.json')} for arm in r.ARMS}
    def ious(pp, gt, ids, sizes):
        hits = np.array([np.count_nonzero(gt[pp] == g) for g in ids])
        return hits/(len(pp)+sizes-hits) if len(pp) >= 100 else np.zeros(len(ids))
    for s in c['scenes']:
        gt = np.loadtxt(jobs[s]['job']['GT'], dtype=np.int64)
        ids, sizes = np.unique(gt, return_counts=True)
        keep = (ids >= 1000) & (sizes >= 100) & np.isin(ids//1000, classes)
        ids, sizes = ids[keep], sizes[keep]
        baseline = np.zeros(len(ids))
        with np.load(r.GEOMETRY/'predictions'/s/'data.npz') as source:
            for n in r.a.read(r.GEOMETRY/'predictions'/s/'nodes.json'):
                if n['in_original_NT']:
                    baseline = np.maximum(baseline, ious(source['n%04d' % n['node']], gt, ids, sizes))
        for arm in r.ARMS:
            best = baseline.copy()
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as z:
                for key in z.files:
                    best = np.maximum(best, ious(z[key], gt, ids, sizes))
            for j, g in enumerate(ids):
                q = targets[arm][s, int(g)]
                assert abs(q['baseline']-baseline[j]) < 1e-12 and abs(q['expanded']-best[j]) < 1e-12
                checks['full_NT_target_rows_recomputed'] += 1
    opportunities = r.a.read(r.OUT/'opportunity_diagnostics.json')['opportunities']
    assert len(opportunities) == 18
    for q in opportunities:
        i = next(i for i, row in enumerate(rr) if (row['scene'], row['pair_id']) == (q['scene'], q['pair_id']))
        for arm in r.ARMS:
            expected = [f for f in range(5) if f != folds[i]] if label[i] >= 0 else []
            assert q['arms'][arm]['actually_fitted_other_folds'] == expected
            checks['opportunity_training_memberships'] += 1
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(r.HERE),
                            '-p', 'test_fi1_relation_controls.py', '-v'], capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    r.a.seal(r.OUT/'verification.json', dict(complete=True, checks=dict(checks),
        stdout=tests.stdout, stderr=tests.stderr,
        limitations='Same frozen feature representation and source post-GT pair-composition table reused. Full NT IoU independently recomputed; this is reproducibility verification, not an isolated-scene validation.'))
    print('VERIFIED', checks, flush=True)


if __name__ == '__main__':
    main()

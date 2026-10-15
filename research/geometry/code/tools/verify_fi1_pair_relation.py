#!/usr/bin/env python3
"""Post-assessment independent structural, feature, model and geometry checks."""
from collections import Counter
import json
import subprocess
import sys
import time
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
import run_fi1_pair_relation as r


def main():
    r.guard(False)
    c = r.contract()
    seals = [r.a.read(r.OUT/name) for name in ('features_frozen.json', 'labels_frozen.json',
                                             'predictions_frozen.json', 'evaluation_frozen.json')]
    for seal in seals:
        r.verify(seal['outputs'])
    rows = r.a.read(r.OUT/'feature_rows.json')
    assert len(rows) == 4865 and len({(x['scene'], x['pair_id']) for x in rows}) == 4865
    matrices = r.matrix()
    scenes = r.a.read(r.a.OUT/'contract.json')['jobs']
    identities, counts = {}, Counter()
    for s in c['scenes']:
        source = r.SOURCE/'predictions'/s
        expected = r.a.read(source/'pairs.json')
        actual = [x for x in rows if x['scene'] == s]
        assert [(p['pair_id'], p['left'], p['right']) for p in expected] == [(p['pair_id'], p['left'], p['right']) for p in actual]
        receipt = r.a.read(r.OUT/'features'/s/'receipt.json')
        r.verify(receipt['inputs'])
        counts['hashed_inputs'] += len(receipt['inputs'])
        labels = r.a.read(r.OUT/'labels'/f'{s}.json')
        assert r.a.sha(labels['GT_path']) == labels['GT_sha256']
        gt = np.loadtxt(labels['GT_path'], dtype=np.int64)
        identities.update({(s, y['pair_id']): y['relation'] for y in labels['pairs']})
        raw = np.asarray(np.load(scenes[s]['job']['paths']['processed'], mmap_mode='r')[:, :9])
        with np.load(source/'data.npz') as z:
            for old, new in zip(expected, actual):
                left, right = z['n%04d' % new['left']], z['n%04d' % new['right']]
                union = np.union1d(left, right)
                assert np.array_equal(union, z[new['pair_id']+'_union'])
                assert r.a.digest(union) == new['union_sha256'] == old['union_sha256']
                for node, pp in ((new['left'], left), (new['right'], right)):
                    lab = r.core.endpoint_label(*np.unique(gt[pp], return_counts=True))
                    assert lab == labels['nodes'][str(node)]
                relation = r.core.labels(labels['nodes'][str(new['left'])], labels['nodes'][str(new['right'])])
                assert identities[s, new['pair_id']] == relation
                counts['exact_frozen_unions'] += 1
            # Recompute whole-surface/contact features for every pair, not just successes.
            f = np.load(r.OUT/'features'/s/'features.npz')
            assert f['geometry'].shape == (len(actual), 23)
            assert f['rgbd'].shape == (len(actual), 131)
            for j, new in enumerate(actual):
                g, v, *_ = r.core.surface(raw, z['n%04d' % new['left']], z['n%04d' % new['right']])
                np.testing.assert_allclose(g, f['geometry'][j, :21], equal_nan=True)
                np.testing.assert_allclose(v, f['rgbd'][j, :80], equal_nan=True)
                assert len(new['views']) <= 3
                assert len({v['frame_id'] for v in new['views']}) == len(new['views'])
                assert set(new['selected_frame_ids']) == {v['frame_id'] for v in new['views']}
                counts['surface_features_recomputed'] += 1
        print('verify', s, counts['exact_frozen_unions'], flush=True)
    for fold in range(5):
        d = r.OUT/'models'/f'fold{fold}'
        receipt = r.a.read(d/'receipt.json')
        expected_train = [i for i, x in enumerate(rows) if c['folds'][x['scene']] != fold and identities[x['scene'], x['pair_id']] in ('same', 'different')]
        assert receipt['training_rows'] == expected_train
        train = {s.rsplit('_', 1)[0] for s in receipt['train_scenes']}
        test = {s.rsplit('_', 1)[0] for s in receipt['test_scenes']}
        assert not train & test
        assert set(receipt['label_inputs']) == {str(r.OUT/'labels'/f'{s}.json') for s in receipt['train_scenes']}
        r.verify(receipt['label_inputs'])
        ix = np.array([c['folds'][x['scene']] == fold for x in rows])
        for arm in r.ARMS:
            with threadpool_limits(4):
                score = joblib.load(d/f'{arm}.joblib').predict_proba(matrices[arm][ix])[:, 1]
            np.testing.assert_allclose(score, np.load(r.OUT/f'{arm}_oof.npy')[ix], rtol=0, atol=0)
    for arm in r.ARMS:
        score = np.load(r.OUT/f'{arm}_oof.npy')
        pred = r.a.read(r.OUT/f'{arm}_decisions.json')
        for s in c['scenes']:
            ix = [i for i, x in enumerate(rows) if x['scene'] == s]
            local = [pred[i] for i in ix]
            used = set()
            expected_accept = set()
            for i in sorted(ix, key=lambda j: (-score[j], rows[j]['pair_id'])):
                row = rows[i]
                if row['eligible'] and score[i] >= .9 and row['left'] not in used and row['right'] not in used:
                    used.update([row['left'], row['right']]);expected_accept.add(row['pair_id'])
            assert expected_accept == {p['pair_id'] for p in local if p['state'] == 'accept'}
            for p in local:
                if p['state'] == 'reject':
                    assert p['eligible'] and p['score'] <= .1
                if p['state'] == 'unknown':
                    assert p['pair_id'] not in expected_accept
            with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as z:
                assert set(z.files) == {p['pair_id']+'_union' for p in local if p['emit']}
                hashes = set()
                for p in local:
                    if p['emit']:
                        assert p['state'] == 'accept' and not p['duplicate_original_NT']
                        h = r.a.digest(z[p['pair_id']+'_union'])
                        assert h == p['union_sha256'] and h not in hashes
                        hashes.add(h)
                        counts['emitted_exact_unions'] += 1
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(r.HERE),
                           '-p', 'test_fi1_pair_relation.py', '-v'], capture_output=True, text=True)
    assert test.returncode == 0, test.stderr
    r.a.seal(r.OUT/'verification.json', dict(complete=True, time_unix=time.time(), checks=dict(counts),
        test_stdout=test.stdout, test_stderr=test.stderr,
        independent_process=True, independent_implementation_of_matching=True,
        limitation='Surface feature recomputation shares core measurement routines; RGB-D paths verified by hashes, view receipts and tests, not an independent second sensor reconstruction. No independent validation scenes or instance AP.'))
    print('VERIFIED', counts, flush=True)


if __name__ == '__main__':
    main()

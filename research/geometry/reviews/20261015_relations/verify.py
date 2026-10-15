#!/usr/bin/env python3
"""Recompute published scalar diagnostics without local experiment assets."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-11), (left, right)


def physical(scene):
    return scene.rsplit('_', 1)[0]


def load_csv(name):
    def decode(value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    with (HERE / name).open(newline='') as stream:
        return [{k: decode(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def main():
    evidence = json.loads((HERE / 'evidence.json').read_text())
    for path, expected in evidence['published_code'].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, path
    for name, expected in evidence['scalar_files'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    dataset = evidence['dataset']
    train, dev, small = map(set, (dataset['train200'], dataset['dev55'], dataset['reference82']))
    assert (len(train), len(dev), len(small)) == (200, 55, 82)
    assert small < train and train.isdisjoint(dev)
    assert {physical(s) for s in train}.isdisjoint(physical(s) for s in dev)
    assert len({physical(s) for s in dev}) == 29
    pairs = load_csv('dev55_pairs.csv')
    targets = load_csv('dev55_targets.csv')
    assert len(pairs) == len({(q['scene'], q['pair_id']) for q in pairs}) == 4014
    assert {q['scene'] for q in pairs} <= dev
    assert Counter(q['GT_relation_strict'] for q in pairs) == dict(same=1281, different=491, mixed=1645, unknown=597)
    latest = evidence['experiments']['fi1_relation_train200_dev55_20261015_v1']
    for arm in ('reference82', 'train200'):
        expected = latest['results']['results'][arm]
        accepted = [q for q in pairs if q[arm + '_state'] == 'accept']
        assert Counter(q[arm + '_state'] for q in pairs) == expected['decisions']
        assert Counter(q['GT_relation_strict'] for q in accepted) == expected['strict_accepted']
        assert Counter(q['GT_relation_previous'] for q in accepted) == expected['previous_accepted']
        assert sum(q[arm + '_emit'] for q in pairs) == expected['emitted']
        assert sum(q['good_endpoint_loss_gt02'] for q in accepted) == expected['good_endpoint_loss_gt02']
        emitted = [q for q in pairs if q[arm + '_emit']]
        close(sum(q['union_unknown'] for q in emitted) / sum(q['union_points'] for q in emitted),
              expected['annotation_unknown_fraction'])
        for scene in dev:
            used, hashes = set(), set()
            for q in sorted((p for p in pairs if p['scene'] == scene),
                            key=lambda q: (-q[arm + '_score'], q['pair_id'])):
                score = q[arm + '_score']
                endpoints = {q['left'], q['right']}
                accept = q['eligible'] and score >= .9 and not endpoints & used
                state = 'accept' if accept else 'reject' if q['eligible'] and score <= .1 else 'unknown'
                assert state == q[arm + '_state']
                if accept:
                    used.update(endpoints)
                emit = accept and not q['duplicate_original_NT'] and q['union_sha256'] not in hashes
                assert emit == q[arm + '_emit']
                if emit:
                    hashes.add(q['union_sha256'])
        hard = [q for q in pairs if q['intersection_over_union'] == 0 and q['GT_relation_strict'] in ('same', 'different')]
        by_class = {}
        for label in ('same', 'different'):
            rows = [q for q in hard if q['GT_relation_strict'] == label]
            losses = []
            for q in rows:
                score = min(max(q[arm + '_score'], 1e-12), 1 - 1e-12)
                losses.append(-math.log(score) if label == 'same' else -math.log1p(-score))
            by_class[label] = sum(losses) / len(losses)
            counts = latest['learning']['arms'][arm]['dev55']['disjoint']['scan'][
                'positive' if label == 'same' else 'negative']
            assert len(rows) == counts['rows']
            assert sum(q[arm + '_score'] >= .9 for q in rows) == counts['high90']
        close(sum(by_class.values()) / 2,
              latest['learning']['arms'][arm]['dev55']['disjoint']['scan']['balanced_logloss'])
        pool = [q for q in targets if q['mode'] == arm]
        assert len(pool) == expected['pool']['targets'] == 1185
        for q in pool:
            close(q['expanded'] - q['baseline'], q['delta'])
        assert sum(q['delta'] >= .02 - 1e-12 for q in pool) == expected['pool']['pool_gain02']
        close(sum(q['delta'] for q in pool), expected['pool']['pool_delta_sum'])
    supply = [q for q in targets if q['mode'] == 'all_proposed' and q['delta'] >= .02 - 1e-12]
    assert (len(supply), len({q['scene'] for q in supply}), len({physical(q['scene']) for q in supply})) == (21, 15, 12)
    safe = [q for q in pairs if q['safe_gain02']]
    assert len(safe) == 10 and all(q['eligible'] for q in safe)
    assert len({(q['scene'], q['same_target']) for q in safe}) == 9
    assert len({physical(q['scene']) for q in safe}) == 7
    assert sum(q['GT_relation_strict'] == 'same' for q in safe) == 7
    for q in safe:
        for arm in ('reference82', 'train200'):
            assert q[arm + '_score'] < .9 and not q[arm + '_emit']
    print('Verified source hashes, 200/55 isolation, 4014 pair diagnostics, matching,')
    print('3555 target rows and opportunity counts. Raw geometry/model replay requires local assets.')


if __name__ == '__main__':
    main()

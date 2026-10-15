#!/usr/bin/env python3
"""Independent postseal recomputation from full frozen NT, pair points and GT."""
from collections import Counter
import numpy as np
import run_fi1_pair_relation as r


def scores(pp, gt, ids, sizes, official=True):
    if official and len(pp) < 100:
        return np.zeros(len(ids))
    hits = np.array([np.count_nonzero(gt[pp] == g) for g in ids])
    return hits/(sizes+len(pp)-hits)


def identity(pp, gt, ids, sizes):
    value = scores(pp, gt, ids, sizes, official=False)
    j = int(np.argmax(value)) if value.max(initial=0) > 0 else None
    hit = int(np.count_nonzero(gt[pp] == ids[j])) if j is not None else 0
    stable = hit >= 20 and hit/len(pp) >= .6
    gids, counts = np.unique(gt[pp], return_counts=True)
    known = gids >= 1000
    mixed = int(np.count_nonzero(known & (counts >= 20))) >= 2
    dominant = int(np.argmax(np.where(known, counts, 0))) if known.any() else None
    clear = not mixed and dominant is not None and counts[dominant] >= 20 and counts[dominant]/len(pp) >= .9
    return dict(j=j, target=int(ids[j]) if stable else None, stable=stable,
                official_iou=float(scores(pp, gt, ids, sizes)[j]) if j is not None else 0.,
                strict='mixed' if mixed else 'clear' if clear else 'unknown',
                strict_target=int(gids[dominant]) if clear else None)


def main():
    r.guard(False)
    c = r.contract()
    for name in ('predictions_frozen.json', 'evaluation_frozen.json'):
        r.verify(r.a.read(r.OUT/name)['outputs'])
    jobs = r.a.read(r.a.OUT/'contract.json')['jobs']
    classes = r.a.read(r.a.OLD/'contract.json')['class_ids']
    result = r.a.read(r.OUT/'results.json')['results']
    allrows = {arm: {(q['scene'], q['pair_id']): q for q in r.a.read(r.OUT/'evaluation'/f'{arm}_pairs.json')} for arm in r.ARMS}
    alltargets = {arm: {(q['scene'], q['gt_id']): q for q in r.a.read(r.OUT/'evaluation'/f'{arm}_targets.json')} for arm in r.ARMS}
    census = {arm: Counter() for arm in r.ARMS}
    verified, supply, cases = 0, [], []
    for s in c['scenes']:
        gt = np.loadtxt(jobs[s]['job']['GT'], dtype=np.int64)
        ids, sizes = np.unique(gt, return_counts=True)
        valid = (ids >= 1000) & (sizes >= 100) & np.isin(ids//1000, classes)
        ids, sizes = ids[valid], sizes[valid]
        nodes = r.a.read(r.SOURCE/'predictions'/s/'nodes.json')
        pairs = r.a.read(r.SOURCE/'predictions'/s/'pairs.json')
        with np.load(r.SOURCE/'predictions'/s/'data.npz') as z:
            base = np.zeros(len(ids))
            for n in nodes:
                if n['in_original_NT']:
                    base = np.maximum(base, scores(z['n%04d' % n['node']], gt, ids, sizes))
            best = {arm: base.copy() for arm in r.ARMS}
            identities = {n['node']: identity(z['n%04d' % n['node']], gt, ids, sizes)
                          for n in nodes if n['final_track_ids']}
            for p in pairs:
                pp = z[p['pair_id']+'_union']
                value = scores(pp, gt, ids, sizes)
                left, right = identities[p['left']], identities[p['right']]
                relation = 'same' if left['stable'] and right['stable'] and left['target'] == right['target'] else 'different' if left['stable'] and right['stable'] else 'unresolved'
                strict = 'same' if left['strict'] == right['strict'] == 'clear' and left['strict_target'] == right['strict_target'] else 'different' if left['strict'] == right['strict'] == 'clear' else 'mixed' if 'mixed' in (left['strict'], right['strict']) else 'unknown'
                purity = np.mean(gt[pp] == left['target']) if relation == 'same' else None
                delta = float(value[left['j']]-base[left['j']]) if relation == 'same' else None
                safe = relation == 'same' and purity >= .9 and delta >= .02-1e-12
                loss = any(x['stable'] and x['official_iou'] >= .9 and x['official_iou']-value[x['j']] > .02+1e-12 for x in (left, right))
                if safe:
                    supply.append((s, p['pair_id']))
                for arm in r.ARMS:
                    q = allrows[arm][s, p['pair_id']]
                    assert q['GT_relation_previous'] == relation
                    assert q['GT_relation_strict'] == strict
                    assert q['safe_gain02'] == safe and q['good_endpoint_loss_gt02'] == loss
                    if relation == 'same':
                        assert abs(q['union_target_purity']-purity) < 1e-12
                        assert abs(q['same_target_pool_delta']-delta) < 1e-12
                    if q['emit']:
                        best[arm] = np.maximum(best[arm], value)
                    if q['state'] == 'accept':
                        census[arm]['accepted'] += 1
                        census[arm]['strict_'+strict] += 1
                        census[arm]['previous_'+relation] += 1
                        census[arm]['good_endpoint_loss_gt02'] += int(loss)
                verified += 1
            for arm in r.ARMS:
                for i, g in enumerate(ids):
                    q = alltargets[arm][s, int(g)]
                    assert abs(q['baseline']-base[i]) < 1e-12
                    assert abs(q['expanded']-best[arm][i]) < 1e-12
                    census[arm]['gain02_targets'] += int(best[arm][i]-base[i] >= .02-1e-12)
        print('outcome audit', s, verified, flush=True)
    assert len(supply) == 18 and verified == 4865
    for arm in r.ARMS:
        q = census[arm]
        assert q['accepted'] == result[arm]['decisions']['accept']
        assert q['gain02_targets'] == result[arm]['pool']['pool_gain02']
        assert q['good_endpoint_loss_gt02'] == result[arm]['good_endpoint_loss_gt02']
        for state, count in result[arm]['identity_by_state']['accept'].items():
            assert q['strict_'+state] == count
    r.a.seal(r.OUT/'outcome_audit.json', dict(complete=True, code_sha256=r.a.sha(__file__),
        all_frozen_pairs_recomputed=verified, valuable_supply_pairs=len(supply), census=census,
        full_NT_source='Every frozen source node marked in_original_NT; no new acceptance labels or old pool helper used.',
        uses_new_independent_IoU_and_identity_implementation=True, predictions_changed=False))


if __name__ == '__main__':
    main()

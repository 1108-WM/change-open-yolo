"""Independent readout/IoU audit and fixed-edit-domain ceiling; no source writes."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'research/geometry/code/tools'))
import run_fi1_nt_region_identity as diagnostic

BASE = ROOT / 'research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1'
DEST = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def vector(points, gt, ids, sizes):
    if len(points) < 100:
        return np.zeros(len(ids))
    g, n = np.unique(gt[points], return_counts=True)
    counts = dict(zip(g.tolist(), n.tolist()))
    hit = np.array([counts.get(int(i), 0) for i in ids])
    return hit / (len(points) + sizes - hit)

def main():
    c = read(BASE / 'contract.json')
    receipt = read(BASE / 'completion_verified.json')
    hashes = 0
    for mapping in [receipt['outputs'], c['outputs'], c['input_bindings'], c['GT_inputs'], read(BASE / 'models_frozen.json')['outputs'], read(BASE / 'predictions_frozen.json')['outputs']]:
        for p, h in mapping.items():
            assert sha(Path(p)) == h, p
            hashes += 1
    for p, h in c['code'].items():
        assert sha(BASE / 'code_snapshot' / Path(p).relative_to(ROOT)) == h
        assert sha(Path(p)) == h
    ha, hb = [read(BASE / 'models' / a / 'history.json') for a in ('A', 'B')]
    assert ha['initial_state_sha256'] == hb['initial_state_sha256']
    assert len(ha['epochs']) == len(hb['epochs']) == 64
    assert all(a['batch_order'] == b['batch_order'] for a, b in zip(ha['epochs'], hb['epochs']))
    assert sum(r['steps'] for r in ha['epochs']) == sum(r['steps'] for r in hb['epochs']) == 256
    modes = ('A', 'B', 'B_zero', 'B_shuffle_0', 'B_shuffle_1', 'B_shuffle_2', 'region_average')
    lookup = {(r['scene'], r['candidate_id'], r['mode']): r for r in rows(BASE / 'evaluation/parents.csv')}
    pool = {(r['scene'], r['mode'], int(r['gt_id'])): r for r in rows(BASE / 'evaluation/pool.csv')}
    per_parent = []
    totals = {m: dict(added_target=0, added_other=0, deleted_other=0, deleted_target=0, unknown_edits=0, parent_iou_sum=0., delta_sum=0., pool_gain02=0) for m in modes}
    checks = dict(pointsets=0, pool_rows=0, input_labels=0, query_top1=0)
    gains = {m: [] for m in modes}
    opportunity_sum = dict(addition=0, deletion=0)
    for s in c['scenes']:
        gt = np.loadtxt(c['jobs'][s]['GT'], dtype=np.int64)
        ids, sizes = np.unique(gt, return_counts=True)
        valid = (ids >= 1000) & (sizes >= 100) & np.isin(ids // 1000, c['class_ids'])
        ids, sizes = ids[valid], sizes[valid]
        _, nt = diagnostic.load_nt(c['jobs'][s])
        baseline = np.max([vector(p, gt, ids, sizes) for p in nt.values()], axis=0)
        best = {m: baseline.copy() for m in (*modes, 'original_NT')}
        with np.load(diagnostic.OUT / 'priors' / s / 'data.npz') as pr, np.load(diagnostic.SOURCE / 'features' / s / 'data.npz') as original, np.load(diagnostic.OUT / 'associations' / s / 'data.npz') as region:
            for r in [r for r in c['samples'] if r['scene'] == s]:
                cid, gid = r['candidate_id'], r['target']
                idx = int(np.flatnonzero(ids == gid)[0])
                with np.load(BASE / 'inputs' / s / (cid + '.npz')) as z:
                    parent, halo = z['parent'], z['halo']
                    member, seen = np.isin(halo, parent), z['visible'] >= 2
                    y = np.where(gt[halo] == gid, 0, np.where(gt[halo] >= 1000, 1, 2))
                    assert np.array_equal(y, z['y'])
                    assert np.array_equal(z['base'], original[cid + '_gd10_selected6_x'].astype(np.float32))
                    assert np.array_equal(z['base'][:, 9] > .5, member)
                    ix = np.searchsorted(pr['point_ids'], halo)
                    assert np.array_equal(pr['point_ids'][ix], halo)
                    assert np.array_equal(z['head'], pr['P1_head_roi'][ix])
                    inter = pr['P1_probability'][np.searchsorted(pr['point_ids'], parent)].sum(0, dtype=np.float64)
                    scores = inter / (len(parent) + pr['P1_full_volume'] - inter)
                    q = int(np.lexsort((np.arange(len(scores)), -scores))[0])
                    assert q == r['query_id']
                    assert np.array_equal(z['query'], pr['P1_probability'][ix, q])
                    checks['input_labels'] += 1
                    checks['query_top1'] += 1
                    opportunity_sum['addition'] += int((~member & seen & (y == 0)).sum())
                    opportunity_sum['deletion'] += int((member & seen & (y == 1)).sum())
                    before = vector(parent, gt, ids, sizes)
                    # Hard upper bound for this target: all editable true points in,
                    # all editable non-target points out, invisible membership fixed.
                    # Removing unknown GT points is allowed ONLY for this optimistic
                    # bound, never a deployable operation or training policy.
                    ideal = halo[(member & ~seen) | (seen & (gt[halo] == gid))]
                    assert len(ideal) >= 100  # Otherwise minimum size needs a separate bound.
                    known = halo[(member & ~(seen & (y == 1))) | (~member & seen & (y == 0))]
                    known_score = vector(known, gt, ids, sizes)[idx]
                    ideal_score = vector(ideal, gt, ids, sizes)[idx]
                    scores_by_mode = {}
                    for mode in modes:
                        if mode == 'region_average':
                            actual = region[cid + '_P1_region']
                        else:
                            arm = 'A' if mode == 'A' else 'B'
                            with np.load(BASE / 'predictions' / s / (cid + '_' + arm + '.npz')) as pred:
                                p, actual = pred[mode + '_probability'], pred[mode + '_points']
                            assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-6)
                            expected = halo[(member & ~(seen & (p[:, 1] >= .8))) | (~member & seen & (p[:, 0] >= .8))]
                            if len(expected) < 100:
                                expected = parent
                            assert np.array_equal(expected, actual)
                        assert np.array_equal(actual, np.unique(actual))
                        row = lookup[s, cid, mode]
                        add, delete = np.setdiff1d(actual, parent), np.setdiff1d(parent, actual)
                        counts = dict(added_target=int((gt[add] == gid).sum()), added_other=int(((gt[add] >= 1000) & (gt[add] != gid)).sum()), deleted_target=int((gt[delete] == gid).sum()), deleted_other=int(((gt[delete] >= 1000) & (gt[delete] != gid)).sum()))
                        counts['unknown_edits'] = len(add) + len(delete) - sum(counts.values())
                        for k, v in counts.items():
                            assert int(row[k]) == v
                            totals[mode][k] += v
                        after = vector(actual, gt, ids, sizes)
                        assert abs(float(row['parent_iou']) - before[idx]) < 1e-12
                        assert abs(float(row['repaired_iou']) - after[idx]) < 1e-12
                        assert after[idx] <= ideal_score + 1e-12
                        totals[mode]['parent_iou_sum'] += float(after[idx])
                        totals[mode]['delta_sum'] += float(after[idx] - before[idx])
                        scores_by_mode[mode] = float(after[idx])
                        best[mode] = np.maximum(best[mode], after)
                        checks['pointsets'] += 1
                    per_parent.append(dict(scene=s, candidate_id=cid, source=r['source'], category=r['category'], baseline=float(before[idx]), known_edit_feasible=float(known_score), unrestricted_edit_upper_bound=float(ideal_score), A_to_upper_bound=float(ideal_score-scores_by_mode['A']), **scores_by_mode))
        for mode, e in best.items():
            for i, gid in enumerate(ids):
                row = pool[s, mode, int(gid)]
                assert abs(float(row['baseline']) - baseline[i]) < 1e-12
                assert abs(float(row['expanded']) - e[i]) < 1e-12
                gain = bool(e[i] - baseline[i] >= .02-1e-12)
                assert ast.literal_eval(row['gain02']) == gain
                for t in (.5, .75, .9, .95):
                    assert ast.literal_eval(row[f'cross_{t}']) == bool(baseline[i] <= t < e[i])
                if mode in totals and gain:
                    totals[mode]['pool_gain02'] += 1
                    gains[mode].append([s, int(gid)])
                checks['pool_rows'] += 1
    n = len(per_parent)
    ceiling = dict(mean_A=float(np.mean([r['A'] for r in per_parent])), mean_B=float(np.mean([r['B'] for r in per_parent])), mean_known_edit_feasible=float(np.mean([r['known_edit_feasible'] for r in per_parent])), mean_unrestricted_edit_upper_bound=float(np.mean([r['unrestricted_edit_upper_bound'] for r in per_parent])), max_possible_mean_B_minus_actual_A=float(np.mean([r['A_to_upper_bound'] for r in per_parent])), registered_mean_advantage_gate=c['gates']['mean_B_minus_A_min'], parents_with_upper_bound_advantage_ge02=sum(r['A_to_upper_bound'] >= .02 for r in per_parent))
    result = dict(complete=True, hash_checks=hashes, code_snapshots=len(c['code']), checks=checks, same_initialization_and_256_updates=True, ceiling=ceiling, opportunities=opportunity_sum, totals=totals, pool_gain_targets=gains, per_parent=per_parent, mean_B_minus_A=float(np.mean([r['B']-r['A'] for r in per_parent])), mean_B_minus_zero=float(np.mean([r['B']-r['B_zero'] for r in per_parent])), mean_B_minus_shuffle=float(np.mean([r['B']-np.mean([r[f'B_shuffle_{i}'] for i in range(3)]) for r in per_parent])), limitations=['No training or GPU model replay in this audit; original replay receipt checked as an artifact.', 'NT loader reused, pointset readout and metrics independently implemented.', 'Ceiling is GT-only, specific to each frozen parent target, halo and visibility; not a prediction.', 'Original experiment seals and state files not modified.'])
    (DEST / 'audit_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_parent', 'totals', 'pool_gain_targets')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

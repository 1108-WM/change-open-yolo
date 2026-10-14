"""Read-only independent pointset/IoU replay and explicitly posthoc diagnostics."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'research/geometry/code/tools'))
import run_fi1_nt_region_identity as loader
BASE = ROOT / 'research/geometry/experiments/fi1_query_transfer_eval_20261014_v1'
DEST = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def iou(p, gt, ids, sizes):
    if len(p) < 100:
        return np.zeros(len(ids))
    g, n = np.unique(gt[p], return_counts=True)
    counts = dict(zip(g.tolist(), n.tolist()))
    hit = np.array([counts.get(int(g), 0) for g in ids])
    return hit / (len(p) + sizes - hit)

def main():
    c = read(BASE / 'contract.json')
    assert len(c['scenes']) == len({s.split('_')[0] for s in c['scenes']}) == 16
    assert not {s.split('_')[0] for s in c['scenes']} & {s.split('_')[0] for s in c['pilot_training_scenes']}
    hashes = 0
    for mapping in [read(BASE / 'completion_verified.json')['outputs'], c['model_bindings'], c['inputs'], c['GT_inputs']]:
        for p, h in mapping.items():
            assert sha(p) == h, p
            hashes += 1
    for p, h in c['code'].items():
        assert sha(BASE / 'code_snapshot' / Path(p).relative_to(ROOT)) == h
        assert sha(p) == h
    for name in ('structure', 'priors', 'associations', 'predictions'):
        for p, h in read(BASE / (name + '_frozen.json'))['outputs'].items():
            assert sha(p) == h
            hashes += 1
    for name in ('export_entry_addendum', 'host_wrapper_addendum'):
        for p, h in read(BASE / (name + '.json'))['code'].items():
            assert sha(p) == h
    lookup = {(r['scene'], r['candidate_id'], r['mode']): r for r in rows(BASE / 'evaluation/parents.csv')}
    pool = {(r['scene'], r['mode'], r['source'], int(r['gt_id'])): r for r in rows(BASE / 'evaluation/pool.csv')}
    opp = {(r['scene'], r['candidate_id']): r for r in rows(BASE / 'evaluation/opportunities.csv') if r['mode'] == 'B'}
    modes = list(c['modes'])
    extra = ['B_add_only', 'B_delete_only', 'B_query_agreement_05']
    all_modes = modes + extra
    metrics = {m: dict(stable=0, delta_sum=0., good=0, good_loss02=0, worst_good_loss=0., added_target=0, added_other=0, deleted_target=0, deleted_other=0, unknown_edits=0, edits=0, pool_gain02=0, pool_delta_sum=0., pool_targets=[], negative_scenes=[], good_loss_scenes=[]) for m in all_modes}
    counts = dict(pointsets=0, pool_rows=0, mapping_points=0, parents=0, stable=0, opportunities=0)
    scope_mislabels = []
    bad_confidence = dict(wrong_add_total=0, wrong_add_ge95=0, wrong_add_query_lt05=0, wrong_delete_total=0, wrong_delete_ge95=0, wrong_delete_query_ge05=0)
    opportunity = dict(addition=0, deletion=0, known_delta_sum=0.)
    witnesses = []
    for s in c['scenes']:
        gt = np.loadtxt(c['jobs'][s]['GT'], dtype=np.int64)
        ids, sizes = np.unique(gt, return_counts=True)
        valid = (ids >= 1000) & (sizes >= 100) & np.isin(ids // 1000, c['class_ids'])
        ids, sizes = ids[valid], sizes[valid]
        _, nt = loader.load_nt(c['jobs'][s])
        baseline = np.max([iou(p, gt, ids, sizes) for p in nt.values()], axis=0)
        best = {(m, src): baseline.copy() for m in ['original_NT'] + all_modes for src in ('all', 'native', 'track')}
        scene_delta = {m: 0. for m in all_modes}
        scene_good_bad = {m: False for m in all_modes}
        with np.load(BASE / 'structure' / s / 'data.npz') as z, np.load(BASE / 'priors' / s / 'data.npz') as pri, np.load(BASE / 'predictions' / s / 'data.npz') as pred, np.load(BASE / 'associations' / s / 'data.npz') as reg:
            probabilities = pri['P1_probability']
            raw = np.load(c['jobs'][s]['paths']['processed'], mmap_mode='r')
            xyz = np.floor(raw[:, :3] / .02).astype(np.int32)
            assert np.array_equal(pri['input_sparse_coordinates'][pri['original_to_sparse_row'], 1:], xyz)
            counts['mapping_points'] += len(xyz)
            links = {r['candidate_id']: r for r in read(BASE / 'associations' / s / 'links.json') if r['prior'] == 'P1'}
            for r in c['parents'][s]:
                cid, src = r['candidate_id'], r['source']
                parent, halo = z[cid + '_parent'], z[cid + '_halo']
                member, seen, labels = np.isin(halo, parent), z[cid + '_visible'] >= 2, z[cid + '_labels']
                before = iou(parent, gt, ids, sizes)
                idx = int(before.argmax()) if before.max(initial=0) > 0 else None
                gid = int(ids[idx]) if idx is not None else None
                hits = int((gt[parent] == gid).sum()) if gid is not None else 0
                stable = gid is not None and hits >= 20 and hits / len(parent) >= .6
                counts['parents'] += 1
                counts['stable'] += int(stable)
                ix = np.searchsorted(pri['point_ids'], halo)
                assert np.array_equal(pri['point_ids'][ix], halo)
                pix = np.searchsorted(pri['point_ids'], parent)
                inter = probabilities[pix].sum(0, dtype=np.float64)
                scores = inter / (len(parent) + pri['P1_full_volume'] - inter)
                qid = int(np.lexsort((np.arange(len(scores)), -scores))[0])
                assert qid == links[cid]['parent']['top'][0]
                q = probabilities[ix, qid]
                region_p = (np.bincount(labels, weights=q) / np.bincount(labels))[labels]
                if stable:
                    y = np.where(gt[halo] == gid, 0, np.where(gt[halo] >= 1000, 1, 2))
                    feasible = halo[(member & ~(seen & (y == 1))) | (~member & seen & (y == 0))]
                    upper = halo[(member & ~seen) | (seen & (y == 0))]
                    assert len(feasible) >= 100 and len(upper) >= 100
                    fi, ui = iou(feasible, gt, ids, sizes)[idx], iou(upper, gt, ids, sizes)[idx]
                    op = opp[s, cid]
                    assert abs(float(op['known_feasible_iou']) - fi) < 1e-12
                    assert abs(float(op['generous_feasible_iou']) - ui) < 1e-12
                    add_op = int((~member & seen & (y == 0)).sum())
                    del_op = int((member & seen & (y == 1)).sum())
                    assert add_op == int(op['addition_opportunity']) and del_op == int(op['deletion_opportunity'])
                    opportunity['addition'] += add_op
                    opportunity['deletion'] += del_op
                    opportunity['known_delta_sum'] += float(fi - before[idx])
                    counts['opportunities'] += 1
                scope = lookup[s, cid, 'B']['scope']
                g, n = np.unique(gt[parent], return_counts=True)
                dominant = int(g[n.argmax()])
                if scope == 'out_of_scope' and dominant // 1000 in c['class_ids']:
                    scope_mislabels.append([s, cid, dominant])
                bp = pred[cid + '_B_probability']
                for mode in all_modes:
                    if mode == 'region_average':
                        expected = halo[np.where(seen, region_p >= .5, member)]
                        actual = reg[cid + '_P1_region']
                    else:
                        p = bp if mode in extra else pred[cid + '_' + mode + '_probability']
                        add = ~member & seen & (p[:, 0] >= .8)
                        delete = member & seen & (p[:, 1] >= .8)
                        if mode == 'B_add_only':
                            delete[:] = False
                        if mode == 'B_delete_only':
                            add[:] = False
                        if mode == 'B_query_agreement_05':
                            add &= q >= .5
                            delete &= q < .5
                        expected = halo[(member & ~delete) | add]
                        actual = expected if mode in extra else pred[cid + '_' + mode + '_points']
                    if len(expected) < 100:
                        expected = parent
                    if mode in extra:
                        actual = expected
                    assert np.array_equal(expected, actual), (s, cid, mode)
                    assert np.array_equal(actual, np.unique(actual))
                    ad, de = np.setdiff1d(actual, parent), np.setdiff1d(parent, actual)
                    after = iou(actual, gt, ids, sizes)
                    edit = dict(added_target=0, added_other=0, deleted_target=0, deleted_other=0, unknown_edits=len(ad)+len(de))
                    if stable:
                        edit.update(added_target=int((gt[ad] == gid).sum()), added_other=int(((gt[ad] >= 1000) & (gt[ad] != gid)).sum()), deleted_target=int((gt[de] == gid).sum()), deleted_other=int(((gt[de] >= 1000) & (gt[de] != gid)).sum()))
                        edit['unknown_edits'] -= sum(edit[k] for k in ('added_target', 'added_other', 'deleted_target', 'deleted_other'))
                    if mode in modes:
                        row = lookup[s, cid, mode]
                        assert ast.literal_eval(row['stable']) == stable
                        assert int(row['added']) == len(ad) and int(row['deleted']) == len(de)
                        for k, v in edit.items():
                            if stable or k == 'unknown_edits':
                                assert int(row[k]) == v
                        if stable:
                            assert abs(float(row['parent_iou']) - before[idx]) < 1e-12
                            assert abs(float(row['repaired_iou']) - after[idx]) < 1e-12
                        counts['pointsets'] += 1
                    m = metrics[mode]
                    for k, v in edit.items():
                        m[k] += v
                    m['edits'] += len(ad) + len(de)
                    if stable:
                        delta = float(after[idx] - before[idx])
                        m['stable'] += 1
                        m['delta_sum'] += delta
                        scene_delta[mode] += delta
                        if before[idx] >= .9:
                            m['good'] += 1
                            m['good_loss02'] += int(delta < -.02-1e-12)
                            m['worst_good_loss'] = max(m['worst_good_loss'], -delta)
                            scene_good_bad[mode] |= delta < -.02-1e-12
                    if mode == 'B' and stable:
                        mask = np.isin(halo, actual)
                        wa = ~member & mask & (y == 1)
                        wd = member & ~mask & (y == 0)
                        bad_confidence['wrong_add_total'] += int(wa.sum())
                        bad_confidence['wrong_add_ge95'] += int((wa & (bp[:, 0] >= .95)).sum())
                        bad_confidence['wrong_add_query_lt05'] += int((wa & (q < .5)).sum())
                        bad_confidence['wrong_delete_total'] += int(wd.sum())
                        bad_confidence['wrong_delete_ge95'] += int((wd & (bp[:, 1] >= .95)).sum())
                        bad_confidence['wrong_delete_query_ge05'] += int((wd & (q >= .5)).sum())
                    if mode == 'region_average':
                        for i in np.flatnonzero((after > best[mode, 'all']) & (after-baseline >= .02-1e-12)):
                            witnesses.append(dict(scene=s, candidate=cid, source=src, stable=stable, scope=scope, target=int(ids[i]), baseline=float(baseline[i]), after=float(after[i])))
                    best[mode, 'all'] = np.maximum(best[mode, 'all'], after)
                    best[mode, src] = np.maximum(best[mode, src], after)
        for (mode, src), v in best.items():
            if mode not in extra:
                for i, gid in enumerate(ids):
                    row = pool[s, mode, src, int(gid)]
                    assert abs(float(row['baseline'])-baseline[i]) < 1e-12
                    assert abs(float(row['expanded'])-v[i]) < 1e-12
                    assert ast.literal_eval(row['gain02']) == bool(v[i]-baseline[i] >= .02-1e-12)
                    for t in (.5, .75, .9, .95):
                        assert ast.literal_eval(row[f'cross_{t}']) == bool(baseline[i] <= t < v[i])
                    counts['pool_rows'] += 1
            if mode != 'original_NT' and src == 'all':
                metrics[mode]['pool_gain02'] += int((v-baseline >= .02-1e-12).sum())
                metrics[mode]['pool_delta_sum'] += float((v-baseline).sum())
                metrics[mode]['pool_targets'] += [[s, int(ids[i])] for i in np.flatnonzero(v-baseline >= .02-1e-12)]
        for mode in all_modes:
            if scene_delta[mode] < 0:
                metrics[mode]['negative_scenes'].append(s)
            if scene_good_bad[mode]:
                metrics[mode]['good_loss_scenes'].append(s)
        print(s, counts['pointsets'], flush=True)
    for m in metrics.values():
        m['mean_delta'] = m['delta_sum']/m['stable']
    result = dict(complete=True, physical_scene_isolation=True, hash_checks=hashes, counts=counts, metrics=metrics, opportunities=opportunity, B_confident_errors=bad_confidence, region_gain_witnesses=witnesses, scope_mislabels=scope_mislabels, posthoc_rules=extra, limitations=['No GPU or model re-export/replay in this audit; original replay receipt is checked, not rerun.', 'Shared NT input loader only; readout, GT identity and IoU independently implemented.', 'Extra rules are posthoc sensitivity diagnostics on already inspected data, not new validation or deployable decisions.', 'No training, production changes, original seal changes, or threshold selection.'])
    (DEST/'audit_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'metrics'},ensure_ascii=False,indent=2))
    for name,m in metrics.items():
        print(name, {k:m[k] for k in ('mean_delta','good_loss02','worst_good_loss','added_target','added_other','deleted_other','deleted_target','unknown_edits','pool_gain02','pool_delta_sum')})

if __name__ == '__main__':
    main()

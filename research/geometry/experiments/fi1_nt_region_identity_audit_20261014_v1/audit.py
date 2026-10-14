"""Read-only replay of the sealed diagnostic; writes only this audit's result."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'research/geometry/code/tools'))
import run_fi1_nt_region_identity as source

BASE = source.OUT
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
def iou(points, gt, ids, sizes):
    if len(points) < 100:
        return np.zeros(len(ids))
    g, n = np.unique(gt[points], return_counts=True)
    counts = dict(zip(g.tolist(), n.tolist()))
    hit = np.array([counts.get(int(i), 0) for i in ids])
    return hit / (len(points) + sizes - hit)
def value(b, e):
    return (e-b >= .02-1e-12) | np.logical_or.reduce([(b <= t) & (e > t) for t in (.75, .9, .95)])

def main():
    c = read(BASE / 'contract.json')
    receipt = read(BASE / 'completion_verified.json')
    for p, h in receipt['outputs'].items():
        assert sha(Path(p)) == h, p
    for p, h in c['code'].items():
        snap = BASE / 'code_snapshot' / Path(p).relative_to(ROOT)
        assert sha(snap) == h, str(snap)
    for f, k in [('report.md', 'report_sha256'), ('decision.json', 'decision_sha256'), ('contract.json', 'stage0_contract_sha256')]:
        assert sha(BASE / f) == receipt[k]
    original = {(r['scene'], r['candidate_id'], r['mode']): r for r in rows(BASE / 'prior_diagnosis/parents.csv')}
    targets = {(r['scene'], r['mode'], int(r['gt_id'])): r for r in rows(BASE / 'prior_diagnosis/targets.csv')}
    counts = dict(probes=0, pool_targets=0, mapped_points=0)
    scope_mislabels = []
    gain_witnesses = []
    aggregates = {}
    ablations = {k: dict(value_targets=0, gain02=0, delta_sum=0.) for k in ('region_add_only', 'region_delete_only')}
    for s in c['scenes']:
        gt = np.loadtxt(c['jobs'][s]['GT'], dtype=np.int64)
        assert sha(Path(c['jobs'][s]['GT'])) == c['GT_inputs'][c['jobs'][s]['GT']]
        ids, sizes = np.unique(gt, return_counts=True)
        valid = (ids >= 1000) & (sizes >= 100) & np.isin(ids // 1000, c['class_ids'])
        ids, sizes = ids[valid], sizes[valid]
        _, nt = source.load_nt(c['jobs'][s])  # Shared input loader only; no diagnostic/scoring functions reused.
        baseline = np.max([iou(p, gt, ids, sizes) for p in nt.values()], axis=0)
        best = {m: baseline.copy() for m in c['modes']}
        extra = {m: baseline.copy() for m in ablations}
        witness = {}
        links = {(r['candidate_id'], r['prior']): r for r in read(BASE / 'associations' / s / 'links.json')}
        with np.load(BASE / 'structure' / s / 'data.npz') as z, np.load(BASE / 'priors' / s / 'data.npz') as prior, np.load(BASE / 'associations' / s / 'data.npz') as actions:
            probs = {f: prior[f + '_probability'] for f in ('P0', 'P1')}
            raw = np.load(c['jobs'][s]['paths']['processed'], mmap_mode='r')
            coords = np.floor(raw[:, :3] / .02).astype(np.int32)
            assert np.array_equal(prior['input_sparse_coordinates'][prior['original_to_sparse_row'], 1:], coords)
            counts['mapped_points'] += len(coords)
            for parent_row in c['parents'][s]:
                cid = parent_row['candidate_id']
                parent, halo = z[cid + '_parent'], z[cid + '_halo']
                member = np.isin(halo, parent)
                labels, seen = z[cid + '_labels'], z[cid + '_visible'] >= 2
                before = iou(parent, gt, ids, sizes)
                idx = int(before.argmax()) if before.max(initial=0) > 0 else None
                gid = int(ids[idx]) if idx is not None else None
                hit = np.count_nonzero(gt[parent] == gid) if gid is not None else 0
                stable = gid is not None and hit >= 20 and hit / len(parent) >= .6
                g, n = np.unique(gt[parent], return_counts=True)
                dominant = int(g[n.argmax()])
                ref_scope = original[s, cid, 'P1_region']['scope']
                if ref_scope == 'out_of_scope' and dominant // 1000 in c['class_ids']:
                    scope_mislabels.append(dict(scene=s, candidate=cid, dominant=dominant))
                ix = np.searchsorted(prior['point_ids'], halo)
                assert np.array_equal(prior['point_ids'][ix], halo)
                for mode in c['modes']:
                    family, kind = mode.split('_')
                    q = links[cid, family]['parent']['top'][0]
                    p = probs[family][ix, q]
                    if kind == 'region':
                        means = np.bincount(labels, weights=p) / np.bincount(labels)
                        p = means[labels]
                    expected = halo[np.where(seen, p >= .5, member)]
                    if len(expected) < 100:
                        expected = parent.copy()
                    actual = actions[cid + '_' + mode]
                    assert np.array_equal(expected, actual), (s, cid, mode)
                    row = original[s, cid, mode]
                    assert ast.literal_eval(row['stable']) == stable
                    add, delete = np.setdiff1d(actual, parent), np.setdiff1d(parent, actual)
                    assert len(add) == int(row['added']) and len(delete) == int(row['deleted'])
                    metrics = dict(added=len(add), deleted=len(delete), unknown_edits=len(add)+len(delete))
                    if stable:
                        metrics.update(added_target=int((gt[add] == gid).sum()), added_other=int(((gt[add] >= 1000) & (gt[add] != gid)).sum()), deleted_target=int((gt[delete] == gid).sum()), deleted_other=int(((gt[delete] >= 1000) & (gt[delete] != gid)).sum()))
                        metrics['unknown_edits'] -= sum(metrics[k] for k in ('added_target', 'added_other', 'deleted_target', 'deleted_other'))
                    for k, v in metrics.items():
                        assert int(row[k]) == v
                    key = mode + ':' + parent_row['source']
                    ag = aggregates.setdefault(key, {})
                    for k, v in metrics.items():
                        ag[k] = ag.get(k, 0) + v
                    after = iou(actual, gt, ids, sizes)
                    if stable:
                        assert abs(float(row['repaired_iou']) - after[idx]) < 1e-10
                    if mode == 'P1_region':
                        for i in np.flatnonzero(after > best[mode]):
                            witness[int(ids[i])] = dict(candidate=cid, source=parent_row['source'], scope=ref_scope)
                        for name, pts in [('region_add_only', np.union1d(parent, add)), ('region_delete_only', np.setdiff1d(parent, delete))]:
                            if len(pts) < 100:
                                pts = parent
                            extra[name] = np.maximum(extra[name], iou(pts, gt, ids, sizes))
                    best[mode] = np.maximum(best[mode], after)
                    counts['probes'] += 1
        for mode, e in best.items():
            for i, gid in enumerate(ids):
                row = targets[s, mode, int(gid)]
                assert abs(float(row['baseline']) - baseline[i]) < 1e-10
                assert abs(float(row['expanded']) - e[i]) < 1e-10
                assert ast.literal_eval(row['value']) == bool(value(baseline, e)[i])
                counts['pool_targets'] += 1
                if mode == 'P1_region' and value(baseline, e)[i]:
                    gain_witnesses.append(dict(scene=s, target=int(gid), baseline=float(baseline[i]), expanded=float(e[i]), **witness[int(gid)]))
        for name, e in extra.items():
            ablations[name]['value_targets'] += int(value(baseline, e).sum())
            ablations[name]['gain02'] += int((e-baseline >= .02-1e-12).sum())
            ablations[name]['delta_sum'] += float((e-baseline).sum())
        print(s, counts['probes'], flush=True)
    result = dict(complete=True, verification=counts, receipt_outputs_hashed=len(receipt['outputs']), code_snapshots_hashed=len(c['code']), scope_mislabels_in_this_sample=scope_mislabels, P1_region_gain_witnesses=gain_witnesses, edit_counts_by_source=aggregates, posthoc_edit_direction_ablations=ablations, limitations=['No GPU/model re-export or training rerun.', 'NT source loader reused; pointset readout, GT identity, counts and IoU independently implemented.', 'Posthoc direction ablations are diagnostic, not a selected deployment policy.', 'Original diagnostic artifacts were not modified.'])
    (DEST / 'audit_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

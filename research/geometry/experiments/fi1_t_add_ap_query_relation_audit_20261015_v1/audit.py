"""Read-only replay of saved experiment artifacts; writes only this audit directory."""
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'research/geometry/code/tools'))
import numpy as np

BASE = HERE.parent
AP = BASE / 'fi1_t_add_union_downstream_internal24_20261014_v1'
QR = BASE / 'fi1_query_relation_diagnostic_exposed52_20261014_v1'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def audit():
    import analyze_fi1_t_add_downstream as trace
    receipts = {}
    for root in (AP, QR):
        completion = read(root / 'completion_verified.json')
        for key, filename in [('report_sha256', 'report.md'), ('verification_sha256', 'verification.json'), ('supply_verification_sha256', 'supply_verification.json')]:
            if key in completion:
                assert sha(root / filename) == completion[key], (root, filename)
                receipts[str(root / filename)] = completion[key]
        for path, digest in completion.get('code', {}).items():
            assert sha(path) == digest, path
            receipts[path] = digest
    for path, digest in read(AP / 'verification.json')['outputs'].items():
        assert sha(path) == digest, path
        receipts[path] = digest
    for seal in ('predictions_frozen.json', 'matches_frozen.json'):
        for path, digest in read(AP / seal)['outputs'].items():
            assert sha(path) == digest, path
            receipts[path] = digest

    contract = read(AP / 'contract.json')
    summary = read(AP / 'evaluation_summary.json')
    all_changes = read(AP / 'diagnostics/target_changes.json')
    indexed = {(r['scene'], r['view'], r['arm'], r['reference'], round(r['threshold'], 4)): r for r in all_changes}
    target_changes = Counter()
    replay = {}
    ev = trace.run.ctl.base.evaluator()
    ca, _, _ = trace.run.ctl.class_agnostic_evaluator()
    for view, evaluator in [('semantic', ev), ('class_agnostic', ca)]:
        thresholds = list(map(float, evaluator.opt['overlaps']))
        aggregate = {a: {t: dict(slots=[], fn=Counter(), presence={l: (False, False) for l in evaluator.CLASS_LABELS}) for t in thresholds} for a in trace.run.ARMS}
        for s in contract['scenes']:
            coverage = {}
            for arm in trace.run.ARMS:
                match = read(AP / 'evaluation' / s / (view + '_' + arm + '_matches.json'))
                for t in thresholds:
                    slots, fn, presence, covered = trace.trace(match, evaluator, t)
                    coverage[arm, t] = covered
                    agg = aggregate[arm][t]
                    agg['slots'].extend(slots)
                    agg['fn'].update(fn)
                    for label, (hasgt, haspred) in presence.items():
                        oldgt, oldpred = agg['presence'][label]
                        agg['presence'][label] = (oldgt or hasgt, oldpred or haspred)
            for arm in trace.run.ARMS[1:]:
                references = {'NT', arm[0] + '_context'} if arm.endswith('append') else {'NT'}
                for reference in references:
                    for t in thresholds:
                        gained = sorted(coverage[arm, t] - coverage[reference, t])
                        lost = sorted(coverage[reference, t] - coverage[arm, t])
                        r = indexed[s, view, arm, reference, round(t, 4)]
                        assert gained == r['gained_targets'] and lost == r['lost_targets']
                        target_changes['comparisons'] += 1
                        target_changes['gained'] += len(gained)
                        target_changes['lost'] += len(lost)
            print('AP replay', view, s, flush=True)
        for arm in trace.run.ARMS:
            computed = np.stack([trace.integrate(**aggregate[arm][t], labels=evaluator.CLASS_LABELS) for t in thresholds], axis=1)[None]
            path = AP / 'evaluation' / (view + '_' + arm + '_AP.npy')
            assert sha(path) == summary['AP_arrays'][str(path)]
            original = np.load(path)
            assert np.array_equal(np.isnan(computed), np.isnan(original))
            error = float(np.nanmax(np.abs(computed - original)))
            assert error < 1e-12
            values = {'ap': float(np.nanmean(computed[:, :, np.array(thresholds) >= .5])), 'ap50': float(np.nanmean(computed[:, :, thresholds.index(.5)])), 'ap25': float(np.nanmean(computed[:, :, thresholds.index(.25)]))}
            for key, value in values.items():
                assert abs(value - summary['AP'][view][arm][key]) < 1e-12
            replay[view + '/' + arm] = dict(max_error=error, **values)

    opportunities = defaultdict(set)
    categories = Counter()
    for directory in sorted((QR / 'evaluation').iterdir()):
        if not directory.is_dir():
            continue
        for r in read(directory / 'pool.json'):
            if r['expanded'] - r['baseline'] >= .02 - 1e-12:
                opportunities[r['mode']].add((directory.name, r['gt_id']))
        categories.update(r['relation'] for r in read(directory / 'pairs.json'))
    pair_only = opportunities['best_candidate_anchor_pair'] - opportunities['best_candidate_single']
    assert not pair_only
    save('verification.json', dict(complete=True, receipt_hashes=receipts, AP_replay=replay,
        target_changes=dict(target_changes), pair_categories=dict(categories),
        target_sets={mode: sorted(s) for mode, s in opportunities.items()}, pair_only_targets=sorted(pair_only),
        scope='Reuses reviewed original trace/integration, independently reaggregates saved matches and target sets. No model inference, context-feature recomputation or frame evidence replay.'))

def supply():
    import supplement_fi1_query_relation_supply as verifier
    def redirect(path, value):
        relative = Path(path).relative_to(verifier.OUT)
        save('supply_' + '_'.join(relative.parts), value)
    verifier.seal = redirect
    verifier.main()

if __name__ == '__main__':
    {'audit': audit, 'supply': supply}[sys.argv[1]]()

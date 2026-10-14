#!/usr/bin/env python3
"""Frozen source-factorial U/U-prime ablation, including existing U-gate controls.

Reuse candidate aliases, own evidence, B26 and the original two-view evaluator.
No U-prime is discarded merely because its parent U fails output admission.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from tools import run_fi1_union_screened_factorial_ap_development200 as shared
from tools.fi1_core_ablation import source_pool, features_and_members, learned_selection
from tools.fi1_r0_r1_protocol import original_aliases, direct_score
from tools.fi1_r2_core import geometry_features
from tools.fi1_r3_core import pair_overlap, independent_context
from tools.fi1_union_admission_learning import admitted_pool
from tools.fi1_union_development_replay import Guard, project_multiclass
from tools.fi1_class_agnostic_ablation import collapse_members, remap_gt, class_agnostic_evaluator, project_matches
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json, write_json as write

base = shared.base
read = base.read
SOURCE = shared.SOURCE
CLASSIFIER = shared.CLASSIFIER
OUT = ROOT / '.local/fi1_union_refinement_ablation_development200_20261007_v1'
ARMS = ('NT', 'NTU', 'NTUp', 'Full', 'screened_NTU', 'screened_Full')
VIEWS = shared.VIEWS
CONTROLS = {'NT': 'NT', 'NTU': 'NTU', 'screened_NTU': 'classifier_nested'}
PAIRS = (('NT', 'NTU'), ('NT', 'NTUp'), ('NTU', 'Full'), ('NTUp', 'Full'),
         ('NT', 'Full'), ('NT', 'screened_NTU'), ('screened_NTU', 'screened_Full'),
         ('NTUp', 'screened_Full'), ('NT', 'screened_Full'), ('Full', 'screened_Full'))
SOURCES = {'NT': ('native', 'track'), 'NTU': ('native', 'track', 'pair_union'),
           'NTUp': ('native', 'track', 'refined_union'),
           'Full': ('native', 'track', 'pair_union', 'refined_union')}


def stage(out, name, **extra):
    write(out / 'execution.json', dict(status='running', stage=name, updated=time.time(), **extra))


def candidate_pool(geometry, arm, keep):
    if arm in SOURCES:
        return source_pool(geometry, SOURCES[arm])
    if arm not in ('screened_NTU', 'screened_Full'):
        raise ValueError('unknown ablation arm')
    if arm == 'screened_NTU':
        return admitted_pool(geometry, keep)
    # Remove only the rejected U source aliases, preserving independent U-prime
    # and any N/T aliases that share exactly the same pointset.
    available = {g['geometry_key'] for g in geometry
                 if any(a['source'] == 'pair_union' for a in original_aliases(g))}
    if not set(keep) <= available:
        raise ValueError('unknown U admission key')
    result = []
    for g in geometry:
        allowed = SOURCES['Full'] if g['geometry_key'] in keep else SOURCES['NTUp']
        result.extend(source_pool([g], allowed))
    return result


def refinement_links(geometry, keep):
    by_legacy = {}
    for g in geometry:
        for k in [g['geometry_key'], *g.get('legacy_geometry_keys', [])]:
            if k in by_legacy and by_legacy[k] != g['geometry_key']:
                raise ValueError('ambiguous legacy geometry identity')
            by_legacy[k] = g['geometry_key']
    by_key = {g['geometry_key']: g for g in geometry}
    result = []
    for g in geometry:
        sources = {a['source'] for a in original_aliases(g)}
        for p in g['dv3_plans']:
            if not p['append_only']:
                continue
            parent = by_legacy[p['original_union_geometry_key']]
            assert 'pair_union' in {a['source'] for a in original_aliases(by_key[parent])}
            result.append(dict(plan_key=p['plan_key'], geometry_key=g['geometry_key'], parent_U=parent,
                parent_U_admitted=parent in keep, sources=sorted(sources), point_count=g['point_count'],
                independent_geometry=sources == {'refined_union'}))
    return result


def effects(ap):
    return {after + '_minus_' + before: {k: 100 * (ap[after][k] - ap[before][k]) for k in ap[before]}
            for before, after in PAIRS}


def prepare(out):
    assert not out.exists(), 'fresh output required'
    assert sklearn.__version__ == '1.3.2'
    assert shutil.disk_usage(ROOT).free >= 20 * 1024**3
    p = read(SOURCE / 'contract.json')
    assert read(SOURCE / 'acceptance.json')['complete'] and read(CLASSIFIER / 'acceptance.json')['complete']
    assert len(p['scenes']) == len({s[:9] for s in p['scenes']}) == 200
    assert not {s[:9] for s in p['scenes']} & {s[:9] for s in p['reserve100']}
    guard = Guard(p)
    sys.addaudithook(guard.hook)
    inputs = {str(SOURCE / n): sha(SOURCE / n) for n in ('contract.json', 'acceptance.json', 'final_verification.json')}
    inputs[str(base.MODEL)] = base.MODEL_HASH
    jobs, labels = {}, {}
    cp = read(CLASSIFIER / 'contract.json')
    for s in p['scenes']:
        for n in ('geometry', 'points', 'evidence'):
            q = p['bindings'][s][n]
            inputs[q] = p['inputs'][q]
        q = CLASSIFIER / 'oof' / s / 'classifier_nested.json'
        # Bind the already frozen outer-fold choices, never refit or select thresholds.
        inputs[str(q)] = p['inputs'][str(q)]
        jobs[s] = dict(gate=str(q), controls={})
        pr = read(CLASSIFIER / 'predictions' / s / 'receipt.json')
        er = read(CLASSIFIER / 'evaluation' / s / 'receipt.json')
        for a, old in CONTROLS.items():
            q = CLASSIFIER / 'predictions' / s / (old + '_selected.json')
            inputs[str(q)] = pr['outputs'][str(q)]
            jobs[s]['controls'][a] = str(q)
            q = CLASSIFIER / 'evaluation' / s / (old + '_matches.json')
            labels[str(q)] = er['outputs'][str(q)]
    oldsummary = read(CLASSIFIER / 'evaluation/summary.json')
    for group in p['reporting_groups']:
        for old in CONTROLS.values():
            q = CLASSIFIER / 'evaluation' / (group + '_' + old + '_AP.npy')
            labels[str(q)] = oldsummary['AP_arrays'][str(q)]
    code = base.code_paths()
    for name in ('tools/run_fi1_union_refinement_ablation_development200.py',
                 'tools/verify_fi1_union_refinement_ablation_development200.py',
                 'tests/test_fi1_union_refinement_ablation_development200.py',
                 'tools/run_fi1_union_screened_factorial_ap_development200.py',
                 'tools/fi1_union_admission_learning.py', 'tools/fi1_union_development_replay.py',
                 'tools/fi1_class_agnostic_ablation.py', 'tools/run_fi1_union_loss_attribution_train90.py',
                 'tools/fi1_union_loss_attribution.py', 'tools/fi1_union_budget_sensitivity.py'):
        code[str(ROOT / name)] = sha(ROOT / name)
    assert sha(base.MODEL) == base.MODEL_HASH
    base.verify_files(inputs)
    c = dict(version=out.name, scenes=p['scenes'], reserve100=p['reserve100'], bindings=p['bindings'],
        reporting_groups=p['reporting_groups'], jobs=jobs, inputs=inputs, label_inputs=labels, code=code,
        arms=ARMS, views=VIEWS, comparisons=PAIRS, budget=800, B_model=str(base.MODEL), B_model_sha256=base.MODEL_HASH,
        definitions=dict(NT='N/T only', NTU='original N/T/U', NTUp='N/T and existing accepted U-prime; U generated internally only',
            Full='original N/T/U/U-prime', screened_NTU='prior fixed OOF U admission; N/T/U',
            screened_Full='prior fixed OOF U admission; ALL existing accepted U-prime retained independently of parent U'),
        source_ablation='Remove source aliases; preserve shared pointsets; rebuild own/source features, R1 eligibility, actual pool context, frozen B26 and K800.',
        class_agnostic='Collapse each selected K800 by exact pointset, maximum selected score, no refill; preserve distinct near duplicates.',
        scope='Exposed development200 diagnostics, B26 saw597; no generalization or statistical significance claim. Source effects include eligibility, source features, context and ranking.',
        gate='Reuse classifier_nested outer-fold kept keys exactly; no training or threshold search.',
        refinement='Reuse accepted original StageC pointsets and own semantics; no relaxed gate, new pointsets or visual inference.',
        GT_policy='All2400 outputs and inference verification sealed before development200 raw GT. Preserve original198 instance evaluation and ignore rules.',
        new_training=False, new_threshold=False, new_visual_inference=False, production_modified=False,
        reserve100_read=False, heldout_read=False, final_test_read=False)
    out.mkdir(parents=True)
    freeze_json(out / 'contract.json', c)
    for q in code:
        dest = out / 'code_snapshot' / Path(q).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(q, dest)
    write(out / 'execution.json', dict(status='prepared', stage='tests'))
    print('Frozen six-arm development200 source ablation; no label payload read.', flush=True)


def verify_inference(pool, points, evidence, point_count, members, x, scores, model):
    keys = [g['geometry_key'] for g in pool]
    own = {g['geometry_key']: geometry_features(g, evidence[g['geometry_key']]) for g in pool}
    assert np.array_equal(x[:, :20], np.asarray([own[m['geometry_key']][m['class_index']] for m in members]).reshape(-1, 20))
    iou, cover = pair_overlap([points[k] for k in keys], point_count)
    assert np.allclose(x[:, 20:], independent_context(members, keys, iou, cover), atol=1e-14, rtol=0)
    expected = set()
    for g in pool:
        if not g['point_count']:
            continue
        aliases = original_aliases(g)
        classes = {a['class_index'] for a in aliases if 0 <= a['class_index'] < 198}
        if any(a['source'] != 'native' for a in aliases):
            classes |= set(np.flatnonzero(direct_score(evidence[g['geometry_key']]) > 0).tolist())
        expected.update((g['geometry_key'], cl) for cl in classes)
    assert {shared.key(m) for m in members} == expected and len(members) == len(expected)
    assert np.array_equal(scores, np.clip(model.predict(x), 0, 1))


def predictions(out):
    c = read(out / 'contract.json')
    guard = Guard(c)
    sys.addaudithook(guard.hook)
    assert read(out / 'tests_verified.json')['passed']
    base.verify_files(c['inputs']); base.verify_files(c['code'])
    model = joblib.load(c['B_model'])
    outputs, receipts = {}, {}
    for i, s in enumerate(c['scenes'], 1):
        stage(out, 'predictions_and_independent_feature_checks', scene=s, completed=i-1, total=200)
        d = out / 'predictions' / s
        if (d / 'receipt.json').exists():
            r = read(d / 'receipt.json'); base.verify_files(r['outputs'])
        else:
            spec = c['bindings'][s]
            gs, points, evidence = base.load_geometry(spec)
            keep = set(read(c['jobs'][s]['gate'])['kept'])
            links = refinement_links(gs, keep)
            stats, local = {}, {}
            for a in ARMS:
                pool = candidate_pool(gs, a, keep)
                keys = {g['geometry_key'] for g in pool}
                members, x = features_and_members(pool, {k: points[k] for k in keys},
                    {k: evidence[k] for k in keys}, spec['point_count'])
                selected, scores = learned_selection(members, x, model)
                verify_inference(pool, points, evidence, spec['point_count'], members, x, scores, model)
                assert len(selected) == 800 and selected == base.rank([{**m, 'score': float(v)} for m, v in zip(members, scores)])[:800]
                if a in CONTROLS:
                    assert selected == read(c['jobs'][s]['controls'][a]), (s, a, 'control differs')
                for view, values in zip(VIEWS, (selected, collapse_members(selected))):
                    q = d / (view + '_' + a + '.json'); freeze_json(q, values); local[str(q)] = sha(q)
                sources = Counter(al['source'] for g in pool for al in original_aliases(g))
                stats[a] = dict(geometry=len(pool), members=len(members), selected_geometry=len({m['geometry_key'] for m in selected}),
                    source_aliases=dict(sources), features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest())
                for link in links:
                    key = link['geometry_key']
                    link.setdefault('outputs', {})[a] = dict(present_in_pool=key in keys,
                        selected_classes=[m['class_index'] for m in selected if m['geometry_key'] == key])
            q = d / 'refinement_links.json'; freeze_json(q, links); local[str(q)] = sha(q)
            r = dict(complete=True, scene=s, stats=stats, kept_U=len(keep), refinement_aliases=len(links),
                rejected_parent_refinement_aliases=sum(not x['parent_U_admitted'] for x in links),
                independent_context_features_and_eligibility=True, controls_exact=True, raw_GT_read=False, outputs=local)
            freeze_json(d / 'receipt.json', r)
        outputs.update(r['outputs']); receipts[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        if i % 10 == 0: print('Predictions', i, '/200', flush=True)
    assert not guard.gt_opened
    freeze_json(out / 'predictions_frozen.json', dict(complete=True, outputs=outputs, scene_receipts=receipts,
        predictions=200*len(ARMS)*len(VIEWS), control_selections_exact=600,
        independent_inference_verified=True, raw_GT_read=False, contract_sha256=sha(out / 'contract.json')))
    print('All2400 outputs sealed before GT.', flush=True)


def target_review(matches, ev, uid_mapping=None):
    # Reuse the tested four-arm frontier review twice, then rename comparisons.
    result = dict(GT=None, thresholds=None, counts={}, changed_targets={})
    for name, mapping in (('original', dict(A='NT', B='NTU', C='NTUp', D='Full')),
                          ('screened', dict(A='NT', B='screened_NTU', C='NTUp', D='screened_Full'))):
        review = shared.target_review({k: matches[v] for k, v in mapping.items()}, ev, uid_mapping)
        if result['GT'] is not None: assert result['GT'] == review['GT']
        result['GT'], result['thresholds'] = review['GT'], review['thresholds']
        for pair, counts in review['counts'].items():
            after, before = pair.split('_minus_')
            result['counts'][mapping[after] + '_minus_' + mapping[before]] = counts
        result['changed_targets'][name] = review['changed_targets']
    return result


def matching(out):
    c = read(out / 'contract.json')
    guard = Guard(c, raw_GT_allowed=True); sys.addaudithook(guard.hook)
    seal = read(out / 'predictions_frozen.json'); assert seal['complete'] and seal['predictions'] == 2400
    base.verify_files(seal['outputs']); base.verify_files(c['code'])
    guard.gt_enabled = True; base.verify_files(c['label_inputs'])
    ev = base.evaluator(); ca, valid, uid = class_agnostic_evaluator()
    total, receipts = 0, {}
    for i, s in enumerate(c['scenes'], 1):
        stage(out, 'original_matching', scene=s, completed=i-1, total=200)
        d = out / 'evaluation' / s
        if (d / 'receipt.json').exists():
            r = read(d / 'receipt.json'); base.verify_files(r['outputs'])
        else:
            spec = c['bindings'][s]
            assert sha(spec['gt_path']) == spec['gt_sha256']
            gt = np.loadtxt(spec['gt_path'], dtype=np.int64); assert len(gt) == spec['point_count']
            _, points = shared.load_points(spec); sizes = {k: len(p) for k, p in points.items()}
            selected = {v: {a: read(out / 'predictions' / s / (v + '_' + a + '.json')) for a in ARMS} for v in VIEWS}
            unique = {shared.key(m): m for a in ARMS for m in selected['semantic'][a]}
            universe = [unique[k] for k in sorted(unique)]
            template = shared.assign(ev, universe, points, spec['gt_path'], len(gt))
            semantic = {a: project_multiclass(template, universe, selected['semantic'][a], sizes, ev.CLASS_LABELS, ev.VALID_CLASS_IDS) for a in ARMS}
            edges = sum(shared.check_match(semantic[a], selected['semantic'][a], points, gt, ev) for a in ARMS)
            for a, old in CONTROLS.items():
                assert shared.canonical_match(semantic[a]) == shared.canonical_match(read(CLASSIFIER / 'evaluation' / s / (old + '_matches.json')))
            mapped = remap_gt(gt, valid, uid)
            ids, cnt = np.unique(gt, return_counts=True); ok = (ids >= 1000) & np.isin(ids // 1000, valid); ids, cnt = ids[ok], cnt[ok]
            assert sorted(np.unique(mapped[mapped > 0], return_counts=True)[1].tolist()) == sorted(cnt.tolist())
            assert np.array_equal(mapped == 0, ~((gt >= 1000) & np.isin(gt // 1000, valid)))
            reverse = {uid*1000+j+1: int(g) for j, g in enumerate(ids)}
            def mapped_loader(path):
                assert Path(path).resolve() == Path(spec['gt_path']).resolve()
                return mapped.copy()
            ca.util_3d.load_ids = mapped_loader
            cg = {m['geometry_key']: m for a in ARMS for m in selected['class_agnostic'][a]}
            cu = [cg[k] for k in sorted(cg)]
            ct = shared.assign(ca, cu, points, spec['gt_path'], len(gt), True)
            ck = [m['geometry_key'] for m in cu if sizes[m['geometry_key']] >= 100]
            agnostic = {a: project_matches(ct, ck, selected['class_agnostic'][a], ca.CLASS_LABELS[0]) for a in ARMS}
            edges += sum(shared.check_match(agnostic[a], selected['class_agnostic'][a], points, mapped, ca, True) for a in ARMS)
            for view, matches in zip(VIEWS, (semantic, agnostic)):
                for a in ARMS: write(d / (view + '_' + a + '_matches.json'), base.serializable_matches(matches[a]))
            reviews = dict(semantic=target_review(semantic, ev), class_agnostic=target_review(agnostic, ca, reverse))
            assert reviews['semantic']['GT'] == reviews['class_agnostic']['GT'] == int((cnt >= 100).sum())
            write(d / 'target_review.json', reviews)
            r = dict(complete=True, GT=int((cnt >= 100).sum()), GT_sha256=spec['gt_sha256'],
                direct_census_edges=edges, old_semantic_controls_exact=True,
                outputs={str(q): sha(q) for q in d.iterdir() if q.is_file()})
            freeze_json(d / 'receipt.json', r)
        total += r['GT']; receipts[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        if i % 10 == 0: print('Matches', i, '/200', flush=True)
    assert total == 4689
    freeze_json(out / 'matches_frozen.json', dict(complete=True, GT=total, scene_receipts=receipts,
        raw_GT_read=True, outputs_sealed_before_GT=True, reserve100_read=False, heldout_read=False, final_test_read=False))


def evaluate(out):
    c = read(out / 'contract.json'); guard = Guard(c); guard.gt_enabled = True; sys.addaudithook(guard.hook)
    seal = read(out / 'matches_frozen.json'); assert seal['complete']
    base.verify_files(seal['scene_receipts']); base.verify_files(c['code'])
    for s in c['scenes']: base.verify_files(read(out / 'evaluation' / s / 'receipt.json')['outputs'])
    ap = {g: {v: {} for v in VIEWS} for g in c['reporting_groups']}; arrays = {}
    for view in VIEWS:
        ev = base.evaluator() if view == 'semantic' else class_agnostic_evaluator()[0]
        for a in ARMS:
            matches = {s: read(out / 'evaluation' / s / (view + '_' + a + '_matches.json')) for s in c['scenes']}
            uuids = [p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps]
            assert len(uuids) == len(set(uuids))
            for group, scenes in c['reporting_groups'].items():
                stage(out, 'pooled_AP', view=view, arm=a, group=group)
                x, *_ = ev.evaluate_matches({s: matches[s] for s in scenes}); ap[group][view][a] = base.metrics(ev, x)
                dest = out / 'evaluation' / (group + '_' + view + '_' + a + '_AP.npy')
                np.save(dest, x, allow_pickle=False); arrays[str(dest)] = sha(dest)
                if view == 'semantic' and a in CONTROLS:
                    assert np.array_equal(x, np.load(CLASSIFIER / 'evaluation' / (group + '_' + CONTROLS[a] + '_AP.npy')), equal_nan=True)
            print(view, a, ap['development200'][view][a], flush=True)
    reviews = {s: read(out / 'evaluation' / s / 'target_review.json') for s in c['scenes']}
    counts = {g: {v: {pair: {t: dict(sum((Counter(reviews[s][v]['counts'][pair][t]) for s in ss), Counter()))
        for t in reviews[ss[0]][v]['counts'][pair]} for pair in reviews[ss[0]][v]['counts']} for v in VIEWS}
        for g, ss in c['reporting_groups'].items()}
    freeze_json(out / 'evaluation/summary.json', dict(complete=True, AP=ap, AP_units='fraction', AP_arrays=arrays,
        effects_pp={g: {v: effects(ap[g][v]) for v in VIEWS} for g in ap}, target_changes=counts,
        GT=4689, old_control_arrays_exact=9, scope=c['scope'], raw_GT_read=True,
        reserve100_read=False, heldout_read=False, final_test_read=False, production_modified=False))
    write(out / 'execution.json', dict(status='evaluated', stage='independent_verification'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'predictions', 'matching', 'evaluate'))
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1): globals()[args.phase](args.output.resolve())


if __name__ == '__main__': main()

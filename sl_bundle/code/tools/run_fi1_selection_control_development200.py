#!/usr/bin/env python3
"""Two-arm frozen NT selection control on the existing development200 only."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from tools import run_fi1_union_refinement_ablation_development200 as prior
from tools import run_fi1_union_loss_attribution_development200 as cache
from tools.fi1_core_ablation import source_pool
from tools.fi1_r0_r1_protocol import build_pair, original_aliases
from tools.fi1_r2_core import geometry_features
from tools.fi1_same_class_mask_nms import select
from tools.fi1_failure_attribution_core import geometry_overlaps, coverage_funnel
from tools.fi1_union_development_replay import Guard, project_multiclass
from tools.fi1_class_agnostic_ablation import collapse_members, remap_gt, class_agnostic_evaluator, project_matches
from tools.fi1_frontend_protocol import sha256_file as sha, freeze_json, write_json as write

base, shared, read = prior.base, prior.shared, prior.read
SOURCE = prior.OUT
CACHE = cache.OUT
OUT = ROOT / '.local/fi1_nt_selection_control_development200_20261009_v1'
ARMS = ('A', 'B')
VIEWS = ('semantic', 'class_agnostic')
TESTS = ('tests/test_fi1_same_class_mask_nms.py', 'tests/test_fi1_union_development_replay.py',
         'tests/test_fi1_class_agnostic_ablation.py')


def stage(out, name, **extra):
    write(out / 'execution.json', dict(status='running', stage=name, updated=time.time(),
                                     resume_required=True, **extra))


def source_tag(g):
    sources = {a['source'] for a in original_aliases(g)}
    assert sources and sources <= {'native', 'track'}
    return 'shared_NT' if len(sources) == 2 else ('N_only' if 'native' in sources else 'T_only')


def source_flow(left, right):
    return left + '->' + right


def prepare(out):
    assert not out.exists(), 'new isolated output required'
    assert sklearn.__version__ == '1.3.2'
    assert shutil.disk_usage(ROOT).free >= 50 * 1024**3
    p = read(SOURCE / 'contract.json')
    assert read(SOURCE / 'acceptance.json')['complete'] and read(CACHE / 'acceptance.json')['complete']
    assert len(p['scenes']) == len({s[:9] for s in p['scenes']}) == 200
    assert not {s[:9] for s in p['scenes']} & {s[:9] for s in p['reserve100']}
    guard = Guard(p)
    sys.addaudithook(guard.hook)
    inputs = {str(SOURCE / n): sha(SOURCE / n) for n in
              ('contract.json', 'acceptance.json', 'final_verification.json')}
    inputs.update({str(CACHE / n): sha(CACHE / n) for n in
                   ('contract.json', 'acceptance.json', 'final_verification.json')})
    inputs[p['B_model']] = p['B_model_sha256']
    jobs, labels = {}, {}
    for s in p['scenes']:
        for n in ('geometry', 'points', 'evidence'):
            q = p['bindings'][s][n]
            inputs[q] = p['inputs'][q]
        cr = CACHE / 'scenes' / s / 'receipt.json'
        r = read(cr)
        inputs[str(cr)] = sha(cr)
        jobs[s] = {}
        for n in ('nt_geometry.json', 'nt_members.jsonl', 'nt_features.npy', 'nt_scores.npy'):
            q = CACHE / 'scenes' / s / n
            inputs[str(q)] = r['outputs'][str(q)]
            jobs[s][n] = str(q)
        pr = SOURCE / 'predictions' / s / 'receipt.json'
        r = read(pr)
        inputs[str(pr)] = sha(pr)
        jobs[s]['prior_receipt'] = str(pr)
        for v in VIEWS:
            q = SOURCE / 'predictions' / s / (v + '_NT.json')
            inputs[str(q)] = r['outputs'][str(q)]
            jobs[s]['control_' + v] = str(q)
        r = read(SOURCE / 'evaluation' / s / 'receipt.json')
        for v in VIEWS:
            q = SOURCE / 'evaluation' / s / (v + '_NT_matches.json')
            labels[str(q)] = r['outputs'][str(q)]
            jobs[s]['matches_' + v] = str(q)
    labels[str(SOURCE / 'evaluation/summary.json')] = read(SOURCE / 'acceptance.json')['artifacts']['evaluation/summary.json']
    code = base.code_paths()
    for name in ('tools/run_fi1_selection_control_development200.py', 'tools/fi1_same_class_mask_nms.py',
                 'tools/verify_fi1_selection_control_development200.py', 'tests/test_fi1_same_class_mask_nms.py',
                 'tests/test_fi1_union_development_replay.py', 'tests/test_fi1_class_agnostic_ablation.py',
                 'tools/run_fi1_union_refinement_ablation_development200.py',
                 'tools/run_fi1_union_screened_factorial_ap_development200.py',
                 'tools/run_fi1_union_loss_attribution_development200.py',
                 'tools/run_fi1_union_loss_attribution_train90.py', 'tools/fi1_union_development_replay.py',
                 'tools/fi1_union_loss_attribution.py', 'tools/fi1_union_budget_sensitivity.py',
                 'tools/fi1_union_admission_learning.py', 'tools/fi1_class_agnostic_ablation.py',
                 'tools/run_fi1_nt_selection_control_development200.sh'):
        code[str(ROOT / name)] = sha(ROOT / name)
    # Reused scientific implementations must still match the prior frozen run.
    assert all(h == p['code'][q] for q, h in code.items() if q in p['code'])
    c = dict(version=out.name, scenes=p['scenes'], reserve100=p['reserve100'], bindings=p['bindings'],
             reporting_groups=p['reporting_groups'], jobs=jobs, inputs=inputs, label_inputs=labels, code=code,
             B_model=p['B_model'], B_model_sha256=p['B_model_sha256'], arms=ARMS, views=VIEWS, budget=800,
             definitions=dict(A='Frozen NT, original K800', B='Same NT/prebudget members/B26/context; same-class point IoU>0.95 NMS, refill up to800'),
             selection=dict(threshold_numerator=19, threshold_denominator=20, strict=True,
                            order='score descending, original-native alias preferred, geometry key, class index',
                            witnesses='first retained same-class member; suppressed rows never witness',
                            stop='800 accepted rows or exhausted legal members; never restore suppressed rows'),
             source_metadata='All surviving native/track aliases retained; N-only/T-only/shared-NT; cross-class hypotheses are not NMS duplicates',
             class_agnostic='Actual selected members collapsed by exact pointset/max selected score; no refill; distinct near duplicates retained',
             GT_policy='Seal all800 scene/arm/view outputs before raw/cached development200 labels. Original198-instance evaluator/strict overlaps/void/small-instance rules unchanged.',
             authorization='User requested execution of the NT two-arm control after reading the updated plan.',
             scope='Exposed development200, B26 saw597; mechanism evidence, no end-to-end unseen acceptance or significance claim.',
             new_training=False, new_visual_inference=False, new_geometry=False, threshold_search=False,
             reserve100_read=False, heldout_read=False, final_test_read=False, production_modified=False)
    base.verify_files(inputs)
    base.verify_files(code)
    out.mkdir(parents=True)
    freeze_json(out / 'contract.json', c)
    for q in code:
        dest = out / 'code_snapshot' / Path(q).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(q, dest)
    write(out / 'execution.json', dict(status='prepared', stage='tests', resume_required=True))
    print('Prepared NT two-arm contract; no label payload read.', flush=True)


def tests(out):
    c = read(out / 'contract.json')
    base.verify_files(c['code'])
    result = subprocess.run([sys.executable, '-m', 'pytest', *TESTS, '-q'], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out / 'tests.log').write_text(result.stdout)
    print(result.stdout, end='', flush=True)
    assert result.returncode == 0
    freeze_json(out / 'tests_verified.json', dict(passed=True, modules=TESTS, log_sha256=sha(out / 'tests.log')))


def predictions(out):
    c = read(out / 'contract.json')
    guard = Guard(c)
    sys.addaudithook(guard.hook)
    assert read(out / 'tests_verified.json')['passed']
    base.verify_files(c['inputs']); base.verify_files(c['code'])
    model = joblib.load(c['B_model'])
    outputs, receipts, total_members = {}, {}, 0
    start = time.monotonic()
    for n, s in enumerate(c['scenes'], 1):
        stage(out, 'predictions', scene=s, completed=n-1, total=200)
        d = out / 'predictions' / s
        if (d / 'receipt.json').exists():
            r = read(d / 'receipt.json'); base.verify_files(r['outputs'])
        else:
            begin = time.monotonic()
            spec, job = c['bindings'][s], c['jobs'][s]
            gs, all_points, evidence = base.load_geometry(spec)
            pool = source_pool(gs)
            assert pool == read(job['nt_geometry.json'])
            points = {g['geometry_key']: all_points[g['geometry_key']] for g in pool}
            members = base.rows(job['nt_members.jsonl'])
            rebuilt = build_pair(pool, {k: evidence[k] for k in points}, budget=800)['R1']['eligible']
            assert members == rebuilt
            x = np.load(job['nt_features.npy'], allow_pickle=False)
            scores = np.load(job['nt_scores.npy'], allow_pickle=False)
            stats = read(job['prior_receipt'])['stats']['NT']
            assert x.shape == (len(members), 26) and len(pool) == stats['geometry']
            assert len(members) == stats['members'] and np.isfinite(x).all()
            assert hashlib.sha256(x.tobytes()).hexdigest() == stats['features_sha256']
            assert hashlib.sha256(scores.tobytes()).hexdigest() == stats['scores_sha256']
            own = {g['geometry_key']: geometry_features(g, evidence[g['geometry_key']]) for g in pool}
            assert np.array_equal(x[:, :20], np.asarray([own[m['geometry_key']][m['class_index']] for m in members]))
            assert np.array_equal(scores, np.clip(model.predict(x), 0, 1))
            scored = [{**m, 'score': float(v)} for m, v in zip(members, scores)]
            a = base.rank(scored)[:800]
            assert a == read(job['control_semantic'])
            nms_start = time.monotonic()
            b, review = select(scored, points, spec['point_count'])
            nms_seconds = time.monotonic() - nms_start
            metadata = {g['geometry_key']: dict(source_tag=source_tag(g), aliases=original_aliases(g),
                        point_count=g['point_count'], points_sha256=g['points_sha256']) for g in pool}
            akeys, bkeys = {shared.key(m) for m in a}, {shared.key(m) for m in b}
            review.update(sources=metadata, entered=[m for m in b if shared.key(m) not in akeys],
                          exited=[m for m in a if shared.key(m) not in bkeys])
            classes = defaultdict(set)
            for m in members: classes[m['geometry_key']].add(m['class_index'])
            touched = set()
            for pair in review['near_geometry_pairs']:
                pair['shared_eligible_classes'] = sorted(classes[pair['left']] & classes[pair['right']])
                for cl in pair['shared_eligible_classes']:
                    touched.update(((pair['left'], cl), (pair['right'], cl)))
            counts = Counter(prebudget_members=len(members), geometry=len(pool), selected_A=len(a), selected_B=len(b),
                             suppressed=len(review['suppressed']), entered=len(review['entered']), exited=len(review['exited']),
                             examined=review['examined'], unexamined=review['unexamined'],
                             near_geometry_pairs=len(review['near_geometry_pairs']),
                             overlap_gt_half_pairs=len(review['overlap_gt_half_pairs']),
                             prebudget_near_same_class_members=len(touched),
                             original_selected_near_same_class_members=sum(shared.key(m) in touched for m in a))
            flows = Counter()
            for rec in review['suppressed']:
                m, w = rec['member'], rec['witness']
                flow = source_flow(metadata[m['geometry_key']]['source_tag'], metadata[w['geometry_key']]['source_tag'])
                rec['source_flow'] = flow
                rec['was_in_original_K800'] = shared.key(m) in akeys
                flows[flow] += 1
            review.update(counts=dict(counts), suppression_source_flow=dict(flows),
                          warning='Pair overlap/source aliases/multi-class rows are not evaluator duplicate FP counts.')
            local = {}
            for arm, selected in zip(ARMS, (a, b)):
                ca = collapse_members(selected)
                if arm == 'A': assert ca == read(job['control_class_agnostic'])
                for v, rows in zip(VIEWS, (selected, ca)):
                    q = d / (v + '_' + arm + '.json'); freeze_json(q, rows); local[str(q)] = sha(q)
            q = d / 'selection_review.json'; freeze_json(q, review); local[str(q)] = sha(q)
            r = dict(complete=True, scene=s, prebudget_members=len(members), controls_exact=True,
                     eligible_members_rebuilt=True, original_feature_and_score_hashes_exact=True,
                     B26_scores_recomputed_exact=True, own20_recomputed_exact=True,
                     context='exact prior independently-verified X26 hash; unchanged', raw_GT_read=False,
                     elapsed_seconds=time.monotonic()-begin, nms_seconds=nms_seconds,
                     counts=dict(counts), outputs=local)
            freeze_json(d / 'receipt.json', r)
        outputs.update(r['outputs']); receipts[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        total_members += r['prebudget_members']
        if n % 10 == 0: print('Predictions', n, '/200', flush=True)
    assert not guard.gt_opened
    freeze_json(out / 'predictions_frozen.json', dict(complete=True, predictions=800, outputs=outputs,
                scene_receipts=receipts, prebudget_members=total_members, controls_exact=400,
                raw_GT_read=False, contract_sha256=sha(out / 'contract.json'),
                invocation_elapsed_seconds=time.monotonic()-start,
                process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('All800 outputs sealed before labels.', flush=True)


def target_review(matches, ev, uid_mapping=None):
    # Reuse the tested maximum-confidence frontier diagnostic with duplicated
    # arms C=A,D=B. Retain only the actual A->B contrast.
    r = shared.target_review(dict(A=matches['A'], B=matches['B'], C=matches['A'], D=matches['B']), ev, uid_mapping)
    return dict(GT=r['GT'], thresholds=r['thresholds'], counts=r['counts']['B_minus_A'],
                changed_targets=[dict(gt_id=x['gt_id'], frontiers={a: x['frontiers'][a] for a in ARMS})
                                 for x in r['changed_targets']])


def range_review(points, gt, pool, selected, review, ev):
    ids, counts = np.unique(gt, return_counts=True)
    keep = (ids >= 1000) & (counts >= 100) & np.isin(ids // 1000, ev.VALID_CLASS_IDS)
    ids, counts = ids[keep], counts[keep]
    keys = [g['geometry_key'] for g in pool]
    lookup = {k: i for i, k in enumerate(keys)}
    iou, _, _ = geometry_overlaps([points[k] for k in keys], gt, ids, counts)
    class_map = {int(v): i for i, v in enumerate(ev.VALID_CLASS_IDS)}
    gtclasses = np.asarray([class_map[int(g)//1000] for g in ids], dtype=int)

    def description(m):
        gi = lookup[m['geometry_key']]
        same = gtclasses == m['class_index']
        best = int(np.argmax(iou[gi])) if len(ids) and np.max(iou[gi]) > 0 else None
        return dict(best_iou=float(iou[gi].max(initial=0)),
                    best_gt=None if best is None else int(ids[best]),
                    correct_class_best_iou=float(iou[gi, same].max(initial=0)))

    touched = {x for p in review['near_geometry_pairs'] for x in (p['left'], p['right'])}
    diagnostic = dict(suppressed=[dict(**x, member_range=description(x['member']),
                      witness_range=description(x['witness'])) for x in review['suppressed']],
                      entered=[dict(member=m, **description(m)) for m in review['entered']],
                      exited=[dict(member=m, **description(m)) for m in review['exited']])
    totals = {}
    for t in map(float, ev.opt['overlaps']):
        stat = {}
        for arm in ARMS:
            rows = selected[arm]
            states = Counter()
            untouched = Counter()
            selected_classes = [set() for _ in keys]
            for m in rows:
                d = description(m)
                state = 'correct_class_range' if d['correct_class_best_iou'] > t else (
                    'wrong_class_good_range' if d['best_iou'] > t else 'no_valid_range')
                states[state] += 1
                selected_classes[lookup[m['geometry_key']]].add(m['class_index'])
                if m['geometry_key'] not in touched: untouched[state] += 1
            funnel = coverage_funnel(iou, selected_classes, selected_classes, gtclasses, t)
            stat[arm] = dict(member_ranges=dict(states), geometry_without_any_gt95_pair=dict(untouched),
                             covered_targets=sum(x['state'] == 'selected_covered' for x in funnel),
                             potential_duplicate_correct_members=sum(x['duplicate_correct_members'] for x in funnel))
        ct = Counter()
        for x in diagnostic['suppressed']:
            a, b = x['member_range'], x['witness_range']
            ct['different_best_GT'] += a['best_gt'] is not None and b['best_gt'] is not None and a['best_gt'] != b['best_gt']
            ct['suppressed_correct_witness_not_correct'] += a['correct_class_best_iou'] > t >= b['correct_class_best_iou']
            ct['suppressed_better_correct_range'] += a['correct_class_best_iou'] > b['correct_class_best_iou']
        stat['suppression_descriptions'] = dict(ct)
        for tag in ('entered', 'exited'):
            stat[tag] = dict(Counter('correct_class_range' if x['correct_class_best_iou'] > t else
                ('wrong_class_good_range' if x['best_iou'] > t else 'no_valid_range') for x in diagnostic[tag]))
        totals[f'{t:.2f}'] = stat
    return dict(GT=len(ids), thresholds=totals, records=diagnostic,
                warning='GT used after sealed predictions only. Range/frontier counts are not TP/FP, independent AP losses or deployment labels.')


def matching(out):
    c = read(out / 'contract.json'); guard = Guard(c, raw_GT_allowed=True); sys.addaudithook(guard.hook)
    seal = read(out / 'predictions_frozen.json'); assert seal['complete'] and seal['predictions'] == 800
    verified = read(out / 'selection_verified.json')
    assert verified['complete'] and verified['prediction_seal_sha256'] == sha(out / 'predictions_frozen.json')
    base.verify_files(seal['outputs']); base.verify_files(c['code'])
    guard.gt_enabled = True; base.verify_files(c['label_inputs'])
    ev = base.evaluator(); ca, valid, uid = class_agnostic_evaluator()
    total, receipts = 0, {}
    for n, s in enumerate(c['scenes'], 1):
        stage(out, 'matching_and_posthoc_review', scene=s, completed=n-1, total=200)
        d = out / 'evaluation' / s
        if (d / 'receipt.json').exists():
            r = read(d / 'receipt.json'); base.verify_files(r['outputs'])
        else:
            spec = c['bindings'][s]; assert sha(spec['gt_path']) == spec['gt_sha256']
            gt = np.loadtxt(spec['gt_path'], dtype=np.int64); assert len(gt) == spec['point_count']
            gs, points = shared.load_points(spec); sizes = {k: len(p) for k, p in points.items()}
            selected = {v: {a: read(out / 'predictions' / s / (v + '_' + a + '.json')) for a in ARMS} for v in VIEWS}
            unique = {shared.key(m): m for a in ARMS for m in selected['semantic'][a]}
            universe = [unique[k] for k in sorted(unique)]
            template = shared.assign(ev, universe, points, spec['gt_path'], len(gt))
            semantic = {a: project_multiclass(template, universe, selected['semantic'][a], sizes,
                        ev.CLASS_LABELS, ev.VALID_CLASS_IDS) for a in ARMS}
            edges = sum(shared.check_match(semantic[a], selected['semantic'][a], points, gt, ev) for a in ARMS)
            assert shared.canonical_match(semantic['A']) == shared.canonical_match(read(c['jobs'][s]['matches_semantic']))
            mapped = remap_gt(gt, valid, uid)
            ids, cnt = np.unique(gt, return_counts=True); keep = (ids >= 1000) & np.isin(ids//1000, valid); ids, cnt = ids[keep], cnt[keep]
            assert sorted(np.unique(mapped[mapped > 0], return_counts=True)[1].tolist()) == sorted(cnt.tolist())
            assert np.array_equal(mapped == 0, ~((gt >= 1000) & np.isin(gt//1000, valid)))
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
            assert shared.canonical_match(agnostic['A']) == shared.canonical_match(read(c['jobs'][s]['matches_class_agnostic']))
            for v, matches in zip(VIEWS, (semantic, agnostic)):
                for a in ARMS: write(d / (v + '_' + a + '_matches.json'), base.serializable_matches(matches[a]))
            reviews = dict(semantic=target_review(semantic, ev), class_agnostic=target_review(agnostic, ca, reverse))
            assert reviews['semantic']['GT'] == reviews['class_agnostic']['GT'] == int((cnt >= 100).sum())
            write(d / 'target_review.json', reviews)
            pr = read(out / 'predictions' / s / 'selection_review.json')
            rr = range_review(points, gt, source_pool(gs), selected['semantic'], pr, ev)
            assert rr['GT'] == reviews['semantic']['GT']
            write(d / 'range_review.json', rr)
            r = dict(complete=True, GT=int((cnt >= 100).sum()), GT_sha256=spec['gt_sha256'],
                     direct_census_edges=edges, old_two_view_controls_exact=True,
                     outputs={str(q): sha(q) for q in d.iterdir() if q.is_file()})
            freeze_json(d / 'receipt.json', r)
        total += r['GT']; receipts[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        if n % 10 == 0: print('Matches', n, '/200', flush=True)
    assert total == 4689
    freeze_json(out / 'matches_frozen.json', dict(complete=True, GT=total, scene_receipts=receipts,
                raw_GT_read=True, outputs_sealed_before_GT=True, reserve100_read=False, heldout_read=False, final_test_read=False))


def evaluate(out):
    c = read(out / 'contract.json'); guard = Guard(c); guard.gt_enabled = True; sys.addaudithook(guard.hook)
    seal = read(out / 'matches_frozen.json'); assert seal['complete']
    base.verify_files(seal['scene_receipts']); base.verify_files(c['code'])
    for s in c['scenes']: base.verify_files(read(out / 'evaluation' / s / 'receipt.json')['outputs'])
    old = read(SOURCE / 'evaluation/summary.json')
    ap = {g: {v: {} for v in VIEWS} for g in c['reporting_groups']}; arrays = {}
    for v in VIEWS:
        ev = base.evaluator() if v == 'semantic' else class_agnostic_evaluator()[0]
        for a in ARMS:
            matches = {s: read(out / 'evaluation' / s / (v + '_' + a + '_matches.json')) for s in c['scenes']}
            uuids = [p['uuid'] for m in matches.values() for ps in m['pred'].values() for p in ps]
            assert len(uuids) == len(set(uuids))
            for g, scenes in c['reporting_groups'].items():
                stage(out, 'pooled_AP', view=v, arm=a, group=g)
                x, *_ = ev.evaluate_matches({s: matches[s] for s in scenes})
                ap[g][v][a] = base.metrics(ev, x)
                q = out / 'evaluation' / (g + '_' + v + '_' + a + '_AP.npy')
                np.save(q, x, allow_pickle=False); arrays[str(q)] = sha(q)
                if a == 'A':
                    q = SOURCE / 'evaluation' / (g + '_' + v + '_NT_AP.npy')
                    assert sha(q) == old['AP_arrays'][str(q)]
                    assert np.array_equal(x, np.load(q, allow_pickle=False), equal_nan=True)
            print(v, a, ap['development200'][v][a], flush=True)
    aggregate = {}
    for g, scenes in c['reporting_groups'].items():
        target = {v: {} for v in VIEWS}; counts = Counter(); flows = Counter(); ranges = {}; elapsed = nms_elapsed = 0
        for s in scenes:
            pr = read(out / 'predictions' / s / 'selection_review.json')
            counts.update(pr['counts']); flows.update(pr['suppression_source_flow'])
            elapsed += read(out / 'predictions' / s / 'receipt.json')['elapsed_seconds']
            nms_elapsed += read(out / 'predictions' / s / 'receipt.json')['nms_seconds']
            tr = read(out / 'evaluation' / s / 'target_review.json')
            for v in VIEWS:
                for t, ct in tr[v]['counts'].items(): target[v].setdefault(t, Counter()).update(ct)
            rr = read(out / 'evaluation' / s / 'range_review.json')
            for t, arms in rr['thresholds'].items():
                dest = ranges.setdefault(t, {})
                for a, vals in arms.items():
                    if a in ARMS:
                        dest.setdefault(a, Counter()).update({k: vals[k] for k in ('covered_targets', 'potential_duplicate_correct_members')})
                        for kind in ('member_ranges', 'geometry_without_any_gt95_pair'):
                            dest.setdefault(a + '_' + kind, Counter()).update(vals[kind])
                    else: dest.setdefault(a, Counter()).update(vals)
        aggregate[g] = dict(counts=dict(counts), suppression_source_flow=dict(flows),
                            target_changes={v: {t: dict(ct) for t, ct in target[v].items()} for v in VIEWS},
                            range_descriptions={t: {a: dict(ct) for a, ct in vals.items()} for t, vals in ranges.items()},
                            prediction_seconds=elapsed, nms_seconds=nms_elapsed)
    effects = {g: {v: {k: 100*(ap[g][v]['B'][k]-ap[g][v]['A'][k]) for k in ('ap', 'ap50', 'ap25')}
                  for v in VIEWS} for g in ap}
    freeze_json(out / 'evaluation/summary.json', dict(complete=True, AP=ap, AP_arrays=arrays,
                AP_units='fraction', effects_pp=effects, GT=4689, old_control_arrays_exact=6, scope=c['scope'],
                raw_GT_read=True, reserve100_read=False, heldout_read=False, final_test_read=False, production_modified=False))
    freeze_json(out / 'selection_review.json', dict(complete=True, groups=aggregate,
                warning='Counts describe ranges, overlap and confidence frontiers; not additive AP losses. Different-class rows are not suppressed.'))
    write(out / 'execution.json', dict(status='evaluated', stage='independent_verification', resume_required=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'tests', 'predictions', 'matching', 'evaluate'))
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    try:
        with threadpool_limits(limits=1): globals()[args.phase](out)
    except Exception as error:
        if out.exists():
            write(out / 'execution.json', dict(status='failed', stage=args.phase,
                  error_type=type(error).__name__, error=str(error), resume_required=True))
        raise


if __name__ == '__main__': main()

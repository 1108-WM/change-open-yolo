#!/usr/bin/env python3
"""One C/D comparison: same model, scene/class mass, inputs and actions."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import audit_fi1_t_source_chain as a
import run_fi1_relation_controls as previous
import fi1_relation_controls_core as metrics_core
import fi1_pair_relation_core as relation
import fi1_physical_relation_balance_core as weight_core

SOURCE = previous.OUT
BASE = previous.BASE
GEOMETRY = previous.GEOMETRY
HERE = Path(__file__).resolve().parent
OUT = a.EXP/'fi1_physical_relation_balance_exposed52_20261015_v1'
ARMS = ('C', 'D')
FILES = ('run_fi1_physical_relation_balance.py', 'fi1_physical_relation_balance_core.py',
         'verify_fi1_physical_relation_balance.py', 'test_fi1_physical_relation_balance.py')


def guard(no_gt=True, label_scenes=()):
    allowed = {str((BASE/'labels'/f'{s}.json').resolve()) for s in label_scenes}
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
        if writing and not p.is_relative_to(OUT):
            raise PermissionError('write outside new C/D experiment: '+str(p))
        if no_gt and any(x in str(p) for x in ('ground_truth', '/evaluation/', '/labels/',
            'postseal_cases', 'postseal_diagnostic', 'opportunity_diagnostics', '_evidence.csv', '/results.json')) and str(p) not in allowed:
            raise PermissionError('GT, held-out labels or diagnosis forbidden: '+str(p))
    sys.addaudithook(hook)


def contract():
    c = a.read(OUT/'contract.json')
    previous.prior.verify(c['code'])
    previous.prior.verify(c['source_files'])
    return c


def prepare():
    guard()
    a.reserve()
    assert not OUT.exists()
    parent = a.read(SOURCE/'contract.json')
    delivered = a.read(SOURCE/'delivery_verified.json')
    assert delivered['complete']
    source_files = {}
    for name in ('contract.json', 'predictions_frozen.json', 'evaluation_frozen.json',
                 'runtime.json', 'C_scores.npz', 'C_decisions.json'):
        path = SOURCE/name
        source_files[str(path)] = delivered['outputs'][str(path)]
    source_files[str(SOURCE/'delivery_verified.json')] = a.sha(SOURCE/'delivery_verified.json')
    for s in parent['scenes']:
        for name in ('pairs.json', 'unions.npz'):
            path = SOURCE/'predictions/C'/s/name
            source_files[str(path)] = delivered['outputs'][str(path)]
    for fold in range(5):
        for name in ('C.joblib', 'training.npz', 'receipt.json'):
            path = SOURCE/'models'/f'fold{fold}'/name
            source_files[str(path)] = delivered['outputs'][str(path)]
    for name in ('contract.json', 'features.npz', 'feature_rows.json', 'features_frozen.json', 'labels_frozen.json'):
        path = BASE/name
        source_files[str(path)] = parent['source_files'][str(path)]
    label_files = a.read(BASE/'labels_frozen.json')['outputs']
    for path, expected in parent['source_files'].items():
        if Path(path).is_relative_to(GEOMETRY):
            source_files[path] = expected
    for path in (a.OUT/'contract.json', a.OLD/'contract.json'):
        source_files[str(path)] = a.sha(path)
    previous.prior.verify(source_files)
    files = [HERE/name for name in FILES]
    files += [HERE/name for name in ('run_fi1_relation_controls.py', 'fi1_relation_controls_core.py',
              'fi1_pair_relation_core.py', 'audit_fi1_t_source_chain.py', 'evaluate_fi1_nt_region_identity.py')]
    a.seal(OUT/'contract.json', dict(version=OUT.name, time_unix=time.time(),
        code={str(p): a.sha(p) for p in files}, source_files=source_files, label_files=label_files,
        scenes=parent['scenes'], folds=parent['folds'], physical_scene_groups=parent['physical_scene_groups'],
        pairs=4865, feature_dim=154, model=parent['models']['C'], labels=parent['labels'],
        C='Byte-for-byte reuse sealed C models/scores/actions/unions; no C fitting.',
        D='Exactly five original physical-scene folds, identical clear training rows and model. Only within-scene/within-class duplicate relationship weights change. No weight clipping, renormalization that changes C class masses, sampling or search.',
        groups='Training scene and GT version namespace. Same: one physical GT target. Different: unordered pair of physical GT targets. Both endpoints must be clear under unchanged labels. Never join IDs across scenes/directories. Mixed/unknown excluded from fitting only.',
        formula='w_D(pair) = sum(w_C for this training scene and class) / number_of_distinct_physical_relations_in_this_scene_class / pair_count_in_this_physical_relation. C reference weights copied from corresponding original training.npz weights_C.',
        invariants='Per training scene sum(D)==sum(C), and separately within each scene SAME and DIFFERENT sum(D)==sum(C), within float64 numerical tolerance1e-11. Exact original row indices, labels, folds,154 features and C estimator parameters. No held-out GT used for grouping, weights or prediction.',
        actions=parent['actions'], construction=parent['construction'], gates=parent['gates'],
        diagnostics='Unchanged difficult stratum: zero overlap and clear identity. Unweighted same/different score distributions, balanced logloss, scene macro loss, AUC and fixed .9 high-score rates, per fold and pooled. Training aggregates4 occurrences per clear row; OOF one occurrence.',
        relation_success='OOF difficult balanced logloss strictly lower than C AND lower in>=3/5 folds AND clear-different high90 rate strictly lower AND clear-same high90 rate not lower. Report training metrics separately. No AUC or new criterion added to this requested diagnostic gate.',
        geometry_success='Separate unchanged complete-NT value and all-candidate safety checks; classification diagnostic success never implies geometry success.',
        opportunities='All18 valuable pairs only postseal diagnosis; no source grouping, hyperparameter, threshold or output change based on them. Actual fitting membership distinguished from mixed/unknown other-model predictions.',
        stop='If relation diagnosis fails, stop this weighting configuration; do not change around18 cases. If relation improves but unsafe, preserve evidence for future separate contamination study. If geometry jointly passes, prepare a proposal for unexposed physical-scene validation; no automatic validation/AP.',
        seal_order='Protocol -> each fold training-only groups/weights/model -> GT-free all-fold scores/actions/full unions seal -> label diagnostics and GT risk -> verification.',
        limits=dict(D_fit_folds=5, C_refit=False, capacity_search=False, weight_search=False, threshold_search=False,
            new_rules=False, new_features=False, new_frames=False, new_data=False, risk_network=False,
            point_edits=False, partner_change=False, AP=False, production_modified=False, old_seals_modified=False)))
    for path in files:
        shutil.copyfile(path, OUT/('code_'+path.name))
    (OUT/'tmp').mkdir()
    print('C/D protocol frozen; preserve scene AND class masses', flush=True)


def fit(fold):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    c = a.read(OUT/'contract.json')
    train_scenes = [s for s in c['scenes'] if c['folds'][s] != fold]
    guard(True, train_scenes)
    c = contract()
    rr = previous.rows()
    lookup, labels, used = {}, {}, {}
    for s in train_scenes:
        path = BASE/'labels'/f'{s}.json'
        assert a.sha(path) == c['label_files'][str(path)]
        used[str(path)] = a.sha(path)
        labels[s] = a.read(path)
        lookup.update({(s, q['pair_id']): q['relation'] for q in labels[s]['pairs']})
    idx = np.array([i for i, q in enumerate(rr) if lookup.get((q['scene'], q['pair_id'])) in ('same', 'different')])
    with np.load(SOURCE/'models'/f'fold{fold}'/'training.npz') as z:
        np.testing.assert_array_equal(idx, z['row_indices'])
        reference, y = z['weights_C'].copy(), z['y'].copy()
    keys, physical_y = weight_core.group_keys([rr[i] for i in idx], labels)
    np.testing.assert_array_equal(y, physical_y)
    np.testing.assert_array_equal(y, [int(lookup[rr[i]['scene'], rr[i]['pair_id']] == 'same') for i in idx])
    weight = weight_core.balance(keys, reference)
    masses, group_rows = [], []
    scenes = np.array([rr[i]['scene'] for i in idx])
    for s in sorted(set(scenes)):
        for category in (-1, 0, 1):
            mask = (scenes == s) & ((y == category) if category >= 0 else True)
            old_mass, new_mass = float(reference[mask].sum()), float(weight[mask].sum())
            assert abs(old_mass-new_mass) <= 1e-11
            masses.append(dict(scene=s, category=category, C_mass=old_mass, D_mass=new_mass, pairs=int(mask.sum())))
    for key, count in sorted(Counter(keys).items()):
        mask = np.array([k == key for k in keys])
        group_rows.append(dict(scene=key[0], same=key[1], targets=list(key[2:]), pairs=count,
            C_mass=float(reference[mask].sum()), D_mass=float(weight[mask].sum()), per_pair_D=float(weight[mask][0])))
    d = OUT/'models'/f'fold{fold}'
    d.mkdir(parents=True)
    np.savez_compressed(d/'training.npz', row_indices=idx, y=y, weights_C=reference, weights_D=weight,
                        group_keys=np.array([json.dumps(k, separators=(',', ':')) for k in keys]))
    a.seal(d/'physical_groups.json', dict(groups=group_rows, masses=masses, training_only=True, label_inputs=used))
    model = HistGradientBoostingClassifier(**c['model'])
    x, _ = previous.features()
    with threadpool_limits(4):
        model.fit(x[idx], y, sample_weight=weight)
    joblib.dump(model, d/'D.joblib')
    a.seal(d/'receipt.json', dict(fold=fold, train_scenes=train_scenes,
        test_scenes=[s for s in c['scenes'] if c['folds'][s] == fold], label_inputs=used,
        training_rows=idx.tolist(), fitting_pairs=len(idx), physical_groups=len(set(keys)),
        largest_group=max(Counter(keys).values()),
        max_scene_class_mass_error=max(abs(q['C_mass']-q['D_mass']) for q in masses),
        outputs={str(d/name): a.sha(d/name) for name in ('D.joblib', 'training.npz', 'physical_groups.json')}))
    print('D fit', fold, 'pairs', len(idx), 'groups', len(set(keys)),
          'max mass error', max(abs(q['C_mass']-q['D_mass']) for q in masses), flush=True)


def predict():
    import joblib
    from threadpoolctl import threadpool_limits
    guard()
    c = contract()
    x, _ = previous.features()
    rr = previous.rows()
    folds = np.array([c['folds'][q['scene']] for q in rr])
    scores = np.empty((5, len(rr)))
    outputs = {}
    for fold in range(5):
        d = OUT/'models'/f'fold{fold}'
        receipt = a.read(d/'receipt.json')
        previous.prior.verify(receipt['outputs'])
        outputs.update(receipt['outputs'])
        outputs[str(d/'receipt.json')] = a.sha(d/'receipt.json')
        with threadpool_limits(4):
            scores[fold] = joblib.load(d/'D.joblib').predict_proba(x)[:, 1]
    assert np.isfinite(scores).all()
    np.savez_compressed(OUT/'D_scores.npz', fold_all=scores, oof=scores[folds, np.arange(len(rr))])
    shutil.copyfile(SOURCE/'C_scores.npz', OUT/'C_scores.npz')
    for arm in ARMS:
        oof = np.load(OUT/f'{arm}_scores.npz')['oof']
        allpred = []
        for s in c['scenes']:
            d = OUT/'predictions'/arm/s
            d.mkdir(parents=True)
            if arm == 'C':
                for name in ('pairs.json', 'unions.npz'):
                    shutil.copyfile(SOURCE/'predictions/C'/s/name, d/name)
                pred = a.read(d/'pairs.json')
            else:
                ix = [i for i, q in enumerate(rr) if q['scene'] == s]
                pred = relation.decide([rr[i] for i in ix], oof[ix], [rr[i]['eligible'] for i in ix])
                emitted, arrays = set(), {}
                with np.load(GEOMETRY/'predictions'/s/'data.npz') as z:
                    for q in pred:
                        if q['state'] != 'accept':
                            continue
                        pp = z[q['pair_id']+'_union']
                        assert a.digest(pp) == q['union_sha256']
                        if q['duplicate_original_NT'] or q['union_sha256'] in emitted:
                            q['reason'] += '_duplicate_geometry_not_emitted'
                            continue
                        emitted.add(q['union_sha256']);arrays[q['pair_id']+'_union'] = pp;q['emit'] = True
                np.savez_compressed(d/'unions.npz', **arrays)
                a.seal(d/'pairs.json', pred)
            allpred.extend(pred)
            for name in ('pairs.json', 'unions.npz'):
                outputs[str(d/name)] = a.sha(d/name)
        a.seal(OUT/f'{arm}_decisions.json', allpred)
        if arm == 'C':
            assert a.sha(OUT/'C_decisions.json') == a.sha(SOURCE/'C_decisions.json')
            assert a.sha(OUT/'C_scores.npz') == a.sha(SOURCE/'C_scores.npz')
        for name in (f'{arm}_decisions.json', f'{arm}_scores.npz'):
            outputs[str(OUT/name)] = a.sha(OUT/name)
        print(arm, Counter(q['state'] for q in allpred), 'emit', sum(q['emit'] for q in allpred), flush=True)
    a.seal(OUT/'predictions_frozen.json', dict(complete=True, time_unix=time.time(),
        C_exact_reuse=True, D_models_fitted=5, inference_GT_used=False,
        physical_GT_groups_used_only_in_training_weights=True, outputs=outputs))


def diagnostics(c, rr, label, overlap):
    folds = np.array([c['folds'][q['scene']] for q in rr])
    known = label >= 0
    result = {}
    for arm in ARMS:
        score = np.load(OUT/f'{arm}_scores.npz')['fold_all']
        per_fold, pooled = {}, {}
        for part in ('train', 'heldout'):
            yy, pp, ss, oo = [], [], [], []
            for fold in range(5):
                ix = np.flatnonzero(known & ((folds != fold) if part == 'train' else (folds == fold)))
                scenes = np.array([rr[i]['scene'] for i in ix])
                per_fold.setdefault(str(fold), {})[part] = {group: metrics_core.metrics(label[ix[sub]], score[fold, ix[sub]], scenes[sub])
                    for group, sub in (('all', np.ones(len(ix), bool)), ('disjoint', ~overlap[ix]), ('overlap', overlap[ix]))}
                yy.extend(label[ix]);pp.extend(score[fold, ix]);ss.extend(scenes);oo.extend(overlap[ix])
            yy, pp, ss, oo = map(np.asarray, (yy, pp, ss, oo))
            pooled[part] = {group: metrics_core.metrics(yy[sub], pp[sub], ss[sub]) for group, sub in (
                ('all', np.ones(len(yy), bool)), ('disjoint', ~oo), ('overlap', oo))}
        result[arm] = dict(folds=per_fold, pooled=pooled)
    flags = {}
    for part in ('train', 'heldout'):
        cstat, dstat = (result[arm]['pooled'][part]['disjoint'] for arm in ARMS)
        better_folds = sum(result['D']['folds'][str(f)][part]['disjoint']['balanced_logloss'] <
                           result['C']['folds'][str(f)][part]['disjoint']['balanced_logloss'] for f in range(5))
        checks = dict(balanced_logloss_lower=dstat['balanced_logloss'] < cstat['balanced_logloss'],
            at_least3_folds_lower=better_folds >= 3,
            different_high90_lower=dstat['negative']['high90_fraction'] < cstat['negative']['high90_fraction'],
            same_high90_not_lower=dstat['positive']['high90_fraction'] >= cstat['positive']['high90_fraction'])
        flags[part] = dict(checks=checks, lower_balanced_logloss_folds=better_folds, passed=all(checks.values()))
    return dict(arms=result, D_vs_C=flags, relation_diagnosis_passed=flags['heldout']['passed'],
                repeated_training_occurrences_per_pair=4, not_independent_validation=True)


def evaluate():
    import evaluate_fi1_nt_region_identity as met
    guard(False)
    c = contract()
    previous.prior.verify(a.read(OUT/'predictions_frozen.json')['outputs'])
    previous.prior.verify(a.read(SOURCE/'evaluation_frozen.json')['outputs'])
    previous.prior.verify(c['label_files'])
    rr = previous.rows()
    _, overlap = previous.features()
    lookup, scene_labels = {}, {}
    for s in c['scenes']:
        scene_labels[s] = a.read(BASE/'labels'/f'{s}.json')
        lookup.update({(s, q['pair_id']): q['relation'] for q in scene_labels[s]['pairs']})
    label = np.array([1 if lookup[q['scene'], q['pair_id']] == 'same' else 0 if lookup[q['scene'], q['pair_id']] == 'different' else -1 for q in rr])
    diag = diagnostics(c, rr, label, overlap)
    a.seal(OUT/'learning_diagnostics.json', diag)
    keys, _ = weight_core.group_keys([q for q, y in zip(rr, label) if y >= 0], scene_labels)
    same = Counter(k for k in keys if k[1] == 1)
    different = Counter(k for k in keys if k[1] == 0)
    a.seal(OUT/'physical_relation_census.json', dict(postseal_only=True, same_pairs=sum(same.values()),
        same_targets=len(same), different_pairs=sum(different.values()), different_relations=len(different),
        largest_same_groups=[dict(scene=k[0], target=k[2], pairs=n) for k, n in same.most_common(10)],
        largest_different_groups=[dict(scene=k[0], targets=list(k[2:]), pairs=n) for k, n in different.most_common(10)],
        not_used_to_calculate_training_weights=True))
    oldrows = {(q['scene'], q['pair_id']): q for q in a.read(SOURCE/'evaluation/C_pairs.json')}
    oldtargets = a.read(SOURCE/'evaluation/C_targets.json')
    base = {(q['scene'], q['gt_id']): q['baseline'] for q in oldtargets}
    jobs = a.read(a.OUT/'contract.json')['jobs']
    old = a.read(a.OLD/'contract.json')
    outputs, results, joined_by_arm = {}, {}, {}
    for arm in ARMS:
        pred = a.read(OUT/f'{arm}_decisions.json')
        joined, targets = [], []
        for s in c['scenes']:
            gtpath = jobs[s]['job']['GT']
            assert a.sha(gtpath) == old['GT_inputs'][gtpath]
            gt = np.loadtxt(gtpath, dtype=np.int64)
            ids, sizes = met.valid_targets(gt, old['class_ids'])
            baseline = np.array([base[s, int(g)] for g in ids])
            best = baseline.copy()
            with np.load(GEOMETRY/'predictions'/s/'data.npz') as z:
                for q in [q for q in pred if q['scene'] == s]:
                    original = oldrows[s, q['pair_id']]
                    row = dict(**q)
                    for key in ('GT_relation_strict', 'GT_relation_previous', 'safe_gain02', 'same_target',
                        'same_target_pool_delta', 'union_target_purity', 'mixed_endpoint', 'good_endpoint_loss_gt02', 'union_GT'):
                        row[key] = original[key]
                    joined.append(row)
                    if q['emit']:
                        best = np.maximum(best, met.ious(z[q['pair_id']+'_union'], gt, ids, sizes))
            targets.extend(met.pool_rows(s, arm, baseline, best, ids))
        joined_by_arm[arm] = {(q['scene'], q['pair_id']): q for q in joined}
        accepted = [q for q in joined if q['state'] == 'accept']
        emitted = [q for q in accepted if q['emit']]
        safe = [q for q in emitted if q['safe_gain02']]
        unknown = sum(q['union_GT']['unknown'] for q in emitted)/max(1, sum(q['union_GT']['points'] for q in emitted))
        gates = dict(safe_gain_scenes=len({q['scene'] for q in safe}) >= 2,
            no_confirmed_cross_identity=not any(q['GT_relation_previous'] == 'different' for q in accepted),
            no_unresolved_accept=not any(q['GT_relation_previous'] == 'unresolved' for q in accepted),
            full_endpoint_identity=bool(accepted) and all(q['GT_relation_strict'] == 'same' for q in accepted),
            good_copy_safety=not any(q['good_endpoint_loss_gt02'] for q in accepted), annotation_unknown=unknown <= .1)
        results[arm] = dict(decisions=dict(Counter(q['state'] for q in joined)), emitted=len(emitted),
            strict_accepted=dict(Counter(q['GT_relation_strict'] for q in accepted)),
            previous_accepted=dict(Counter(q['GT_relation_previous'] for q in accepted)),
            accepted_mixed=sum(q['mixed_endpoint'] for q in accepted),
            good_endpoint_loss_gt02=sum(q['good_endpoint_loss_gt02'] for q in accepted),
            safe_gain_pairs=len(safe), safe_gain_targets=len({(q['scene'], q['same_target']) for q in safe}),
            safe_gain_scenes=sorted({q['scene'] for q in safe}), pool=met.pool_summary(targets),
            annotation_unknown_fraction=unknown, gates=gates, passed=all(gates.values()))
        a.seal(OUT/'evaluation'/f'{arm}_pairs.json', joined)
        a.seal(OUT/'evaluation'/f'{arm}_targets.json', targets)
        compact = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in q.items() if k != 'views'} for q in joined]
        with (OUT/f'{arm}_evidence.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
        for path in (OUT/'evaluation'/f'{arm}_pairs.json', OUT/'evaluation'/f'{arm}_targets.json', OUT/f'{arm}_evidence.csv'):
            outputs[str(path)] = a.sha(path)
    scores = {arm: np.load(OUT/f'{arm}_scores.npz')['fold_all'] for arm in ARMS}
    opportunities = []
    for i, q in enumerate(rr):
        original = oldrows[q['scene'], q['pair_id']]
        if not original['safe_gain02']:
            continue
        fold = c['folds'][q['scene']]
        other = [f for f in range(5) if f != fold]
        stats = {}
        for arm in ARMS:
            pred = joined_by_arm[arm][q['scene'], q['pair_id']]
            values = scores[arm][other, i]
            stats[arm] = dict(oof=float(scores[arm][fold, i]), state=pred['state'], emit=pred['emit'], reason=pred['reason'],
                actually_fitted_other_folds=other if label[i] >= 0 else [], other_fold_scores=values.tolist(),
                other_fold_mean=float(values.mean()), training_high90=int((values >= .9).sum()) if label[i] >= 0 else None)
        opportunities.append(dict(scene=q['scene'], pair_id=q['pair_id'], strict_relation=lookup[q['scene'], q['pair_id']],
                                  delta=original['same_target_pool_delta'], heldout_fold=fold, arms=stats))
    assert len(opportunities) == 18
    a.seal(OUT/'opportunity_diagnostics.json', dict(posthoc_only=True, opportunities=opportunities))
    relation_passed, geometry_passed = diag['relation_diagnosis_passed'], results['D']['passed']
    decision = 'prepare_unexposed_validation_proposal' if geometry_passed and relation_passed else (
        'preserve_weighting_for_separate_pollution_study' if relation_passed else 'stop_this_weighting_configuration')
    a.seal(OUT/'results.json', dict(complete=True, results=results, relation_diagnosis_passed=relation_passed,
        geometry_repair_passed=geometry_passed, decision=decision, no_AP=True,
        all52_exposed_development=True, D_folds_fitted=5, C_refit=False,
        no_search=True, no_production_change=True))
    for name in ('learning_diagnostics.json', 'physical_relation_census.json', 'opportunity_diagnostics.json', 'results.json'):
        outputs[str(OUT/name)] = a.sha(OUT/name)
    a.seal(OUT/'evaluation_frozen.json', dict(complete=True, prediction_sha256=a.sha(OUT/'predictions_frozen.json'), outputs=outputs))
    print('relation diagnosis', json.dumps(diag['D_vs_C'], ensure_ascii=False), flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print('DECISION', decision, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'fit', 'predict', 'evaluate'))
    p.add_argument('--fold', type=int, choices=range(5))
    args = p.parse_args()
    if args.mode == 'fit':
        assert args.fold is not None
        fit(args.fold)
    else:
        globals()[args.mode]()

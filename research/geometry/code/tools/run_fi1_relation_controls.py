#!/usr/bin/env python3
"""A sealed RGB-D model, B relationship weights, C tree capacity. No search."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import audit_fi1_t_source_chain as a
import run_fi1_pair_relation as prior
import fi1_pair_relation_core as relation
import fi1_relation_controls_core as core

BASE = prior.OUT
GEOMETRY = prior.SOURCE
HERE = Path(__file__).resolve().parent
OUT = a.EXP/'fi1_pair_relation_learning_controls_exposed52_20261015_v1'
ARMS = ('A', 'B', 'C')
FILES = ('run_fi1_relation_controls.py', 'fi1_relation_controls_core.py',
         'verify_fi1_relation_controls.py', 'test_fi1_relation_controls.py')


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
            raise PermissionError('write outside isolated controls: '+str(p))
        if no_gt and any(x in str(p) for x in ('ground_truth', '/evaluation/', '/labels/',
            'postseal_cases', 'postseal_diagnostic', '_evidence.csv', '/results.json')) and str(p) not in allowed:
            raise PermissionError('GT or held-out assessment forbidden: '+str(p))
    sys.addaudithook(hook)


def contract():
    c = a.read(OUT/'contract.json')
    prior.verify(c['code'])
    prior.verify(c['source_files'])
    return c


def features():
    with np.load(BASE/'features.npz') as z:
        return np.c_[z['geometry'], z['rgbd']], z['geometry'][:, 4] > 0


def rows():
    return a.read(BASE/'feature_rows.json')


def prepare():
    guard()
    a.reserve()
    assert not OUT.exists()
    bc = a.read(BASE/'contract.json')
    delivery = a.read(BASE/'delivery_verified.json')
    assert delivery['complete']
    source_files = {str(BASE/name): a.sha(BASE/name) for name in (
        'contract.json', 'delivery_verified.json', 'features_frozen.json', 'labels_frozen.json',
        'predictions_frozen.json', 'evaluation_frozen.json', 'runtime_frozen.json',
        'features.npz', 'feature_rows.json', 'geometry_rgbd_oof.npy', 'geometry_rgbd_decisions.json')}
    # Immutable label references are hashed here without opening label contents.
    label_files = a.read(BASE/'labels_frozen.json')['outputs']
    for fold in range(5):
        d = BASE/'models'/f'fold{fold}'
        for name in ('geometry_rgbd.joblib', 'receipt.json'):
            source_files[str(d/name)] = delivery['outputs'][str(d/name)]
    for s in bc['scenes']:
        for name in ('pairs.json', 'unions.npz'):
            p = BASE/'predictions/geometry_rgbd'/s/name
            source_files[str(p)] = delivery['outputs'][str(p)]
    geometry_seal = a.read(GEOMETRY/'predictions_frozen.json')
    source_files[str(GEOMETRY/'predictions_frozen.json')] = a.sha(GEOMETRY/'predictions_frozen.json')
    for s in bc['scenes']:
        for name in ('data.npz', 'nodes.json', 'pairs.json'):
            p = GEOMETRY/'predictions'/s/name
            source_files[str(p)] = geometry_seal['outputs'][str(p)]
    for path, expected in source_files.items():
        assert a.sha(path) == expected
    files = [HERE/name for name in FILES]
    files += [HERE/name for name in ('fi1_pair_relation_core.py', 'run_fi1_pair_relation.py',
              'audit_fi1_t_source_chain.py', 'evaluate_fi1_nt_region_identity.py')]
    capacity = dict(bc['model'], max_leaf_nodes=31, max_depth=6, min_samples_leaf=5)
    a.seal(OUT/'contract.json', dict(version=OUT.name, time_unix=time.time(),
        source_files=source_files, label_files=label_files, code={str(p): a.sha(p) for p in files},
        scenes=bc['scenes'], folds=bc['folds'], physical_scene_groups=bc['physical_scene_groups'],
        pairs=4865, feature_dim=154, candidate_source=str(GEOMETRY), frozen_RGBD_source=str(BASE),
        A='Reuse sealed geometry_rgbd models, scores, actions and complete unions. No refit. Derive missing in-training scores only by inference with those original models.',
        B='Same estimator/parameters and exact clear training rows as A; only weights differ. Each scene has equal total mass. Within a scene, equally divide mass across actually present overlap/disjoint x clear_same/clear_different cells; uniformly divide within cell. No unavailable cell invented; no oracle gain or 18-case selection.',
        C='Same original scene-normalized A weights and training rows. Only fixed tree capacity changes: max_leaf_nodes31, max_depth6, min_samples_leaf5. All other parameters, including100 iterations, learning rate and L2, identical to A. No capacity search.',
        models=dict(A=bc['model'], B=bc['model'], C=capacity),
        labels=bc['labels'], fitting='Same 5 physical-scene folds. Clear labels only for fitting. Mixed/unknown scored and evaluated in full. All52 scenes exposed development.',
        actions=bc['action'], construction=bc['construction'], gates=bc['gates'],
        metrics='For each arm save every fold score on all4865 rows. Evaluate actual fitting rows and held-out rows separately: disjoint/overlap x same/different, positive/negative score distributions and >=.9/<=.1 rates, unweighted/macro-scene/balanced logloss, Brier, AUC and pairAP. Fold aggregation includes4 in-training occurrences per clear pair and exactly1 held-out occurrence; repetitions are not independent samples. Report each fold too.',
        diagnostic_improvement='Fixed evidence flags versus A: lower hard(disjoint) balanced logloss, no lower same high90 rate, no higher different high90 rate, higher hard AUC. Require all4 plus balanced-logloss improvement in>=3/5 held-out folds for robust OOF diagnostic support. These diagnostics do not replace unchanged full-NT safety/value gates.',
        posthoc='All18 prior valuable supply pairs only after scores/actions frozen; report exact membership, held-out and4 other-fold scores, and whether each was actually fitted. Mixed/unknown opportunities never called training samples.',
        seal_order='Protocol first; train using only corresponding fitting-scene label contents; prediction reads no labels/GT/outcomes; seal all3 arms/all folds/actions/unions before train/OOF diagnostics and GT pool risk evaluation.',
        limits=dict(new_rules=False, new_features=False, new_frames=False, new_data=False,
                    risk_network=False, combined_BC=False, model_search=False, threshold_search=False,
                    halo_training=False, AP=False, production_modified=False, old_seals_modified=False)))
    for p in files:
        shutil.copyfile(p, OUT/('code_'+p.name))
    (OUT/'tmp').mkdir()
    print('protocol frozen; B weights only; C tree capacity only', flush=True)


def fit(fold):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    c = a.read(OUT/'contract.json')
    train_scenes = [s for s in c['scenes'] if c['folds'][s] != fold]
    guard(True, train_scenes)
    c = contract()
    rr = rows()
    lookup, used = {}, {}
    for s in train_scenes:
        p = BASE/'labels'/f'{s}.json'
        used[str(p)] = a.sha(p)
        assert used[str(p)] == c['label_files'][str(p)]
        lookup.update({(s, q['pair_id']): q['relation'] for q in a.read(p)['pairs']})
    idx = np.array([i for i, q in enumerate(rr) if lookup.get((q['scene'], q['pair_id'])) in ('same', 'different')])
    original = a.read(BASE/'models'/f'fold{fold}'/'receipt.json')
    assert idx.tolist() == original['training_rows']
    x, overlap = features()
    y = np.array([lookup[rr[i]['scene'], rr[i]['pair_id']] == 'same' for i in idx], int)
    ss = [rr[i]['scene'] for i in idx]
    wa = core.weights(ss, overlap[idx], y)
    wb = core.weights(ss, overlap[idx], y, balanced=True)
    dest = OUT/'models'/f'fold{fold}'
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'training.npz', row_indices=idx, y=y, weights_A=wa, weights_B=wb, weights_C=wa)
    for arm, weight in (('B', wb), ('C', wa)):
        model = HistGradientBoostingClassifier(**c['models'][arm])
        with threadpool_limits(4):
            model.fit(x[idx], y, sample_weight=weight)
        joblib.dump(model, dest/f'{arm}.joblib')
    cells = [core.cell(o, label) for o, label in zip(overlap[idx], y)]
    mass = {arm: {k: float(w[np.array(cells) == k].sum()) for k in sorted(set(cells))}
            for arm, w in (('A', wa), ('B', wb), ('C', wa))}
    a.seal(dest/'receipt.json', dict(fold=fold, train_scenes=train_scenes,
        test_scenes=[s for s in c['scenes'] if c['folds'][s] == fold], label_inputs=used,
        training_rows=idx.tolist(), counts=dict(Counter(cells)), weight_mass=mass,
        outputs={str(dest/name): a.sha(dest/name) for name in ('training.npz', 'B.joblib', 'C.joblib')}))
    print('fit', fold, len(idx), Counter(cells), mass, flush=True)


def predict():
    import joblib
    from threadpoolctl import threadpool_limits
    guard()
    c = contract()
    x, overlap = features()
    rr = rows()
    folds = np.array([c['folds'][q['scene']] for q in rr])
    scores = {arm: np.empty((5, len(rr))) for arm in ARMS}
    outputs = {}
    for fold in range(5):
        d = OUT/'models'/f'fold{fold}'
        receipt = a.read(d/'receipt.json')
        prior.verify(receipt['outputs'])
        outputs.update(receipt['outputs'])
        outputs[str(d/'receipt.json')] = a.sha(d/'receipt.json')
        for arm in ARMS:
            path = BASE/'models'/f'fold{fold}'/'geometry_rgbd.joblib' if arm == 'A' else d/f'{arm}.joblib'
            with threadpool_limits(4):
                scores[arm][fold] = joblib.load(path).predict_proba(x)[:, 1]
    for arm in ARMS:
        oof = scores[arm][folds, np.arange(len(rr))]
        assert np.isfinite(scores[arm]).all()
        if arm == 'A':
            np.testing.assert_array_equal(oof, np.load(BASE/'geometry_rgbd_oof.npy'))
        np.savez_compressed(OUT/f'{arm}_scores.npz', fold_all=scores[arm], oof=oof)
        outputs[str(OUT/f'{arm}_scores.npz')] = a.sha(OUT/f'{arm}_scores.npz')
        allpred = []
        for s in c['scenes']:
            dest = OUT/'predictions'/arm/s
            dest.mkdir(parents=True)
            if arm == 'A':
                src = BASE/'predictions/geometry_rgbd'/s
                for name in ('pairs.json', 'unions.npz'):
                    shutil.copyfile(src/name, dest/name)
                pred = a.read(dest/'pairs.json')
            else:
                ix = [i for i, q in enumerate(rr) if q['scene'] == s]
                local = [rr[i] for i in ix]
                pred = relation.decide(local, oof[ix], [q['eligible'] for q in local])
                emitted, pointsets = set(), {}
                with np.load(GEOMETRY/'predictions'/s/'data.npz') as z:
                    for q in pred:
                        if q['state'] != 'accept':
                            continue
                        pp = z[q['pair_id']+'_union']
                        assert a.digest(pp) == q['union_sha256']
                        if q['duplicate_original_NT'] or q['union_sha256'] in emitted:
                            q['reason'] += '_duplicate_geometry_not_emitted'
                            continue
                        emitted.add(q['union_sha256'])
                        pointsets[q['pair_id']+'_union'] = pp
                        q['emit'] = True
                a.seal(dest/'pairs.json', pred)
                np.savez_compressed(dest/'unions.npz', **pointsets)
            allpred.extend(pred)
            for name in ('pairs.json', 'unions.npz'):
                outputs[str(dest/name)] = a.sha(dest/name)
        a.seal(OUT/f'{arm}_decisions.json', allpred)
        if arm == 'A':
            assert a.sha(OUT/f'{arm}_decisions.json') == a.sha(BASE/'geometry_rgbd_decisions.json')
        outputs[str(OUT/f'{arm}_decisions.json')] = a.sha(OUT/f'{arm}_decisions.json')
        print(arm, Counter(q['state'] for q in allpred), 'emitted', sum(q['emit'] for q in allpred), flush=True)
    a.seal(OUT/'predictions_frozen.json', dict(complete=True, time_unix=time.time(),
        inference_GT_used=False, supervised_fitting=True, all_scores_and_actions_sealed_before_diagnosis=True,
        A_exact_reuse_verified=True, outputs=outputs))


def training_diagnostics(c, rr, label, overlap):
    folds = np.array([c['folds'][q['scene']] for q in rr])
    known = label >= 0
    allmetrics, pairs = {}, []
    for arm in ARMS:
        with np.load(OUT/f'{arm}_scores.npz') as z:
            score, oof = z['fold_all'], z['oof']
        metrics, aggregate = {}, {part: dict(y=[], score=[], scene=[], overlap=[]) for part in ('train', 'heldout')}
        for fold in range(5):
            metrics[str(fold)] = {}
            for part, mask in (('train', known & (folds != fold)), ('heldout', known & (folds == fold))):
                ix = np.flatnonzero(mask)
                ss = [rr[i]['scene'] for i in ix]
                metrics[str(fold)][part] = {group: core.metrics(label[ix[sub]], score[fold, ix[sub]], np.array(ss)[sub])
                    for group, sub in (('all', np.ones(len(ix), bool)), ('disjoint', ~overlap[ix]), ('overlap', overlap[ix]))}
                arow = aggregate[part]
                arow['y'].extend(label[ix]);arow['score'].extend(score[fold, ix]);arow['scene'].extend(ss);arow['overlap'].extend(overlap[ix])
        pooled = {}
        for part, values in aggregate.items():
            y, p, ss, oo = (np.asarray(values[k]) for k in ('y', 'score', 'scene', 'overlap'))
            pooled[part] = {group: core.metrics(y[sub], p[sub], ss[sub]) for group, sub in (
                ('all', np.ones(len(y), bool)), ('disjoint', ~oo), ('overlap', oo))}
        allmetrics[arm] = dict(folds=metrics, pooled=pooled)
    changes = {}
    for arm in ('B', 'C'):
        changes[arm] = {}
        for part in ('train', 'heldout'):
            base = allmetrics['A']['pooled'][part]['disjoint']
            variant = allmetrics[arm]['pooled'][part]['disjoint']
            flags = core.improvement(base, variant)
            improved_folds = sum(allmetrics[arm]['folds'][str(f)][part]['disjoint']['balanced_logloss'] <
                                 allmetrics['A']['folds'][str(f)][part]['disjoint']['balanced_logloss'] for f in range(5))
            changes[arm][part] = dict(flags=flags, lower_balanced_logloss_folds=improved_folds,
                diagnostic_supported=all(flags.values()) and improved_folds >= 3)
    return dict(arms=allmetrics, comparisons_to_A=changes,
        repeated_training_occurrences_per_clear_pair=4, heldout_occurrences_per_pair=1,
        pair_AP_is_not_instance_AP=True, diagnostic_flags_not_geometry_success=True)


def evaluate():
    import evaluate_fi1_nt_region_identity as met
    guard(False)
    c = contract()
    prior.verify(a.read(OUT/'predictions_frozen.json')['outputs'])
    prior.verify(a.read(BASE/'evaluation_frozen.json')['outputs'])
    prior.verify(c['label_files'])
    rr = rows()
    x, overlap = features()
    labels = {}
    for s in c['scenes']:
        labels.update({(s, q['pair_id']): q['relation'] for q in a.read(BASE/'labels'/f'{s}.json')['pairs']})
    label = np.array([1 if labels[q['scene'], q['pair_id']] == 'same' else 0 if labels[q['scene'], q['pair_id']] == 'different' else -1 for q in rr])
    diagnostics = training_diagnostics(c, rr, label, overlap)
    a.seal(OUT/'learning_diagnostics.json', diagnostics)
    source = {(q['scene'], q['pair_id']): q for q in a.read(BASE/'evaluation/geometry_rgbd_pairs.json')}
    source_targets = a.read(BASE/'evaluation/geometry_rgbd_targets.json')
    sourcebase = {(q['scene'], q['gt_id']): q['baseline'] for q in source_targets}
    jobs = a.read(a.OUT/'contract.json')['jobs']
    old = a.read(a.OLD/'contract.json')
    outputs, results, opportunities = {}, {}, []
    scoremat = {arm: np.load(OUT/f'{arm}_scores.npz')['fold_all'] for arm in ARMS}
    for arm in ARMS:
        pred = a.read(OUT/f'{arm}_decisions.json')
        joined, targets = [], []
        for s in c['scenes']:
            gtpath = jobs[s]['job']['GT']
            assert a.sha(gtpath) == old['GT_inputs'][gtpath]
            gt = np.loadtxt(gtpath, dtype=np.int64)
            ids, sizes = met.valid_targets(gt, old['class_ids'])
            baseline = np.array([sourcebase[s, int(g)] for g in ids])
            best = baseline.copy()
            with np.load(GEOMETRY/'predictions'/s/'data.npz') as z:
                for q in [q for q in pred if q['scene'] == s]:
                    oldrow = source[s, q['pair_id']]
                    row = dict(**q)
                    for key in ('GT_relation_strict', 'GT_relation_previous', 'safe_gain02', 'same_target',
                        'same_target_pool_delta', 'union_target_purity', 'mixed_endpoint', 'good_endpoint_loss_gt02', 'union_GT'):
                        row[key] = oldrow[key]
                    joined.append(row)
                    if q['emit']:
                        best = np.maximum(best, met.ious(z[q['pair_id']+'_union'], gt, ids, sizes))
            targets.extend(met.pool_rows(s, arm, baseline, best, ids))
        accepted = [q for q in joined if q['state'] == 'accept']
        emitted = [q for q in accepted if q['emit']]
        safe = [q for q in emitted if q['safe_gain02']]
        unknown = sum(q['union_GT']['unknown'] for q in emitted)/max(1, sum(q['union_GT']['points'] for q in emitted))
        gates = dict(safe_gain_scenes=len({q['scene'] for q in safe}) >= 2,
            no_confirmed_cross_identity=not any(q['GT_relation_previous'] == 'different' for q in accepted),
            no_unresolved_accept=not any(q['GT_relation_previous'] == 'unresolved' for q in accepted),
            full_endpoint_identity=bool(accepted) and all(q['GT_relation_strict'] == 'same' for q in accepted),
            good_copy_safety=not any(q['good_endpoint_loss_gt02'] for q in accepted),
            annotation_unknown=unknown <= .1)
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
        for p in (OUT/'evaluation'/f'{arm}_pairs.json', OUT/'evaluation'/f'{arm}_targets.json', OUT/f'{arm}_evidence.csv'):
            outputs[str(p)] = a.sha(p)
    for i, q in enumerate(rr):
        oldrow = source[q['scene'], q['pair_id']]
        if not oldrow['safe_gain02']:
            continue
        fold = c['folds'][q['scene']]
        fitted = label[i] >= 0
        stats = {}
        for arm in ARMS:
            other = [f for f in range(5) if f != fold]
            value = scoremat[arm][other, i]
            pred = next(z for z in a.read(OUT/f'{arm}_decisions.json') if z['scene'] == q['scene'] and z['pair_id'] == q['pair_id'])
            stats[arm] = dict(oof=float(scoremat[arm][fold, i]), state=pred['state'], emit=pred['emit'],
                actually_fitted_other_folds=other if fitted else [],
                other_fold_scores=value.tolist(), other_fold_mean=float(value.mean()),
                other_fold_min=float(value.min()), other_fold_max=float(value.max()),
                training_high90=int((value >= .9).sum()) if fitted else None)
        opportunities.append(dict(scene=q['scene'], pair_id=q['pair_id'], strict_relation=labels[q['scene'], q['pair_id']],
            heldout_fold=fold, delta=oldrow['same_target_pool_delta'], arms=stats))
    assert len(opportunities) == 18
    a.seal(OUT/'opportunity_diagnostics.json', dict(posthoc_only=True, opportunities=opportunities,
        actual_training_pairs=sum(q['strict_relation'] in ('same', 'different') for q in opportunities),
        excluded_mixed_unknown_pairs=sum(q['strict_relation'] not in ('same', 'different') for q in opportunities)))
    a.seal(OUT/'results.json', dict(complete=True, results=results,
        learning_direction_flags=diagnostics['comparisons_to_A'],
        independent_validation_allowed_to_consider=any(q['passed'] for q in results.values()),
        all52_exposed_development=True, no_AP=True, no_production_change=True,
        no_weight_or_capacity_or_threshold_search=True))
    for name in ('learning_diagnostics.json', 'opportunity_diagnostics.json', 'results.json'):
        outputs[str(OUT/name)] = a.sha(OUT/name)
    a.seal(OUT/'evaluation_frozen.json', dict(complete=True, outputs=outputs,
        prediction_seal_sha256=a.sha(OUT/'predictions_frozen.json')))
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print('learning directions', json.dumps(diagnostics['comparisons_to_A'], ensure_ascii=False), flush=True)


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

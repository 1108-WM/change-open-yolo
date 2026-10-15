#!/usr/bin/env python3
"""Fixed-pair relation learning using surface appearance and measured RGB-D."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

import audit_fi1_t_source_chain as a
import fi1_pair_relation_core as core

HERE = Path(__file__).resolve().parent
SOURCE = a.EXP/'fi1_cross_track_dual_hypothesis_exposed52_20261015_v2'
PREDECESSOR = a.EXP/'fi1_pair_relation_rgbd_exposed52_20261015_v1'
OUT = a.EXP/'fi1_pair_relation_rgbd_exposed52_20261015_v2'
ARMS = ('geometry', 'geometry_rgbd')
FILES = ('run_fi1_pair_relation.py', 'fi1_pair_relation_core.py',
         'verify_fi1_pair_relation.py', 'test_fi1_pair_relation.py')


def guard(no_gt=True, label_scenes=()):
    allowed = {str((OUT/'labels'/f'{s}.json').resolve()) for s in label_scenes}
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = isinstance(mode, str) and any(x in mode for x in 'wax+')
        writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not p.is_relative_to(OUT):
            raise PermissionError('write outside new experiment: '+str(p))
        if no_gt and any(x in str(p) for x in ('ground_truth', '/evaluation/', '/labels/',
                                              'postseal_cases', 'pair_evidence.csv')) and str(p) not in allowed:
            raise PermissionError('GT/assessment forbidden: '+str(p))
    sys.addaudithook(hook)


def verify(mapping):
    for p, h in mapping.items():
        assert a.sha(p) == h, p


def contract():
    c = a.read(OUT/'contract.json')
    verify(c['code'])
    verify(c['source_files'])
    return c


def prepare():
    guard()
    a.reserve()
    assert not OUT.exists()
    sc = a.read(a.OUT/'contract.json')
    scenes = sorted(sc['jobs'])
    groups = sorted({s.rsplit('_', 1)[0] for s in scenes},
                    key=lambda x: hashlib.sha256(('pair_relation_20261015/'+x).encode()).hexdigest())
    folds = {s: groups.index(s.rsplit('_', 1)[0]) % 5 for s in scenes}
    source_seal = a.read(SOURCE/'predictions_frozen.json')
    assert source_seal['counts']['proposed_pairs'] == 4865
    source_files = {str(SOURCE/q): a.sha(SOURCE/q) for q in
                    ('contract.json', 'predictions_frozen.json', 'delivery_verified.json')}
    source_files[str(a.OUT/'contract.json')] = a.sha(a.OUT/'contract.json')
    abort = a.read(PREDECESSOR/'aborted_before_GT.json')
    assert abort['GT_used'] is False and abort['training_started'] is False
    verify(abort['reusable_outputs'])
    reuse = abort['reusable_outputs']
    for name in ('contract.json', 'aborted_before_GT.json', 'abort_errata.json', 'runtime_frozen.json'):
        source_files[str(PREDECESSOR/name)] = a.sha(PREDECESSOR/name)
    c = dict(version=OUT.name, created=time.time(), scenes=scenes, folds=folds,
        physical_scene_groups=groups, code={str(HERE/q): a.sha(HERE/q) for q in FILES},
        source_files=source_files, candidate_count=4865,
        reused_GT_free_feature_files=reuse,
        predecessor='v1 stopped before labels/training at zero-proposal scene0161_00. Explicit empty arrays fixed; first25 complete feature scenes reused by hashes; measurements/protocol unchanged.',
        inputs='Exact frozen endpoints, 4865 proposals and full unions from source v2. No old decision, mask fit, old reject label, GT opportunity, or N containment requirement is a model input.',
        features='Geometry baseline versus same geometry plus symmetric whole-exclusive-region mesh RGB/normal summaries, bidirectional contact RGB/normal discontinuity, measured sensor-depth paths and sampled RGB across regions. At most3 existing reliable distinct-pose views selected by common exclusive visibility without GT. Missing measurements explicit; no mask-template identity requirement.',
        reliability=dict(min_visible_points=20, min_exclusive_fraction=.1,
                         views_max=3, distinct_translation_m=.2, distinct_rotation_degrees=15),
        labels='Full endpoint pointsets: clear if one annotated instance has >=20 points and >=.9 total purity and no second annotated instance has>=20 points. Same/different uses both clear endpoints. Mixed and unknown excluded from fitting but all receive scores/actions and risk assessment. No valid-class restriction on relation labels.',
        fitting='Five physical-scene folds. Fit4 folds, predict held-out fold. Equal total weight per training scene; natural within-scene prevalence. No threshold/model search or calibration. All52 scenes exposed development; OOF is not independent validation.',
        model=dict(max_iter=100, max_leaf_nodes=7, max_depth=3, min_samples_leaf=20,
                   learning_rate=.05, l2_regularization=5, early_stopping=False,
                   random_state=20261015),
        action='Fixed P>=.9 accept proposals, P<=.1 model rejection, otherwise unknown. Require nonempty exclusive sides and at least a measured 10cm contact or one reliable common view. Sort descending probability then pair ID for disjoint nonrecursive matching. High-score losers unknown. Reject is not a training negative or cannot-link constraint.',
        construction='Keep complete original NT. Emit exact frozen full union only when accepted, changed and not duplicate NT/earlier accepted union. No cropping, new candidates, recursion, query, replacement, tree, new frames or AP.',
        arms=list(ARMS), primary_arm='geometry_rgbd',
        gates=dict(safe_gain02_scenes_min=2, confirmed_cross_identity_max=0,
                   unresolved_accept_max=0, good_endpoint_loss_gt02_max=0,
                   annotation_unknown_fraction_max=.1, union_target_purity_min=.9),
        stop='One fixed comparison. Failure stops this configuration; no post-assessment retuning. Passing only authorizes proposing future physical-isolated validation, not AP execution.',
        seal_order='GT-free features sealed; supervised labels prepared; per-fold fit reads only training scene labels; held-out prediction reads no labels; all scores/actions sealed before outcome evaluation.',
        limits=dict(relation_training=True, halo_training=False, production_modified=False,
                    old_seals_modified=False, AP=False, new_frames=False, point_edits=False))
    a.seal(OUT/'contract.json', c)
    for name in FILES:
        shutil.copyfile(HERE/name, OUT/('code_'+name))
    print('contract frozen', Counter(folds.values()), flush=True)


def checked(path, expected, used):
    path = Path(path)
    key = str(path)
    if key not in used:
        used[key] = a.sha(path)
    assert used[key] == expected, path
    return path


def features():
    guard()
    c = contract()
    sc = a.read(a.OUT/'contract.json')
    frozen = a.read(SOURCE/'predictions_frozen.json')['outputs']
    allrows, allg, allv, outputs = [], [], [], {}
    for si, scene in enumerate(c['scenes']):
        a.reserve()
        d = SOURCE/'predictions'/scene
        dest = OUT/'features'/scene
        assert not dest.exists()
        dest.mkdir(parents=True)
        prior = PREDECESSOR/'features'/scene
        if str(prior/'receipt.json') in c['reused_GT_free_feature_files']:
            for name in ('features.npz', 'rows.json', 'receipt.json'):
                assert a.sha(prior/name) == c['reused_GT_free_feature_files'][str(prior/name)]
                shutil.copyfile(prior/name, dest/name)
                outputs[str(dest/name)] = a.sha(dest/name)
            verify(a.read(dest/'receipt.json')['inputs'])
            allrows += a.read(dest/'rows.json')
            with np.load(dest/'features.npz') as cached:
                allg.extend(cached['geometry'])
                allv.extend(cached['rgbd'])
            print('reused_GT_free_features', si+1, scene, flush=True)
            continue
        used = {}
        def source(name):
            p = d/name
            return checked(p, frozen[str(p)], used)
        pairs = a.read(source('pairs.json'))
        nodes = a.read(source('nodes.json'))
        vp = a.read(source('visibility.json'))
        receipt = a.read(source('receipt.json'))
        job = sc['jobs'][scene]['job']
        mp = job['paths']['manifest']
        manifest = a.read(checked(mp, receipt['inputs'][mp], used))
        raw = np.asarray(np.load(checked(job['paths']['processed'], manifest['files']['processed']['sha256'], used),
                                 mmap_mode='r')[:, :9])
        assert len(raw) == job['binding']['point_count']
        z = np.load(source('data.npz'))
        fv = z['frame_node_visible']
        fids = z['frame_ids']
        findices = z['frame_indices']
        frames = {int(f['frame_id']): f for f in manifest['frames']}
        assert [(int(v['frame_id']), v['frame_index']) for v in vp] == list(zip(map(int, fids), findices))
        intrinsic = np.loadtxt(checked(manifest['files']['intrinsics']['path'], manifest['files']['intrinsics']['sha256'], used))
        poses = {}
        def pose(fi):
            if fi not in poses:
                f = frames[int(fids[fi])]['pose']
                poses[fi] = np.loadtxt(checked(f['path'], f['sha256'], used))
            return poses[fi]
        pointsets = {n['node']: z['n%04d' % n['node']] for n in nodes if n['final_track_ids']}
        for n in nodes:
            if n['node'] in pointsets:
                assert a.digest(pointsets[n['node']]) == n['geometry_sha256']
        g, app, meta, needs = [], [], [], defaultdict(list)
        for pi, p in enumerate(pairs):
            pa, pb = pointsets[p['left']], pointsets[p['right']]
            assert a.digest(z[p['pair_id']+'_union']) == p['union_sha256']
            assert np.array_equal(np.union1d(pa, pb), z[p['pair_id']+'_union'])
            geom, visual, aa, bb, contact = core.surface(raw, pa, pb)
            shared = z[p['pair_id']+'_intersection_visible']
            va, vb = fv[:, p['left']]-shared, fv[:, p['right']]-shared
            reliable = (va >= 20) & (vb >= 20) & (va >= .1*len(aa)) & (vb >= .1*len(bb))
            selected = []
            if len(aa) and len(bb):
                fraction = np.minimum(va/len(aa), vb/len(bb))
                order = sorted(np.flatnonzero(reliable), key=lambda i: (-fraction[i], -min(va[i], vb[i]), int(fids[i])))
                for fi in order:
                    pp = pose(int(fi))
                    if all(core.distinct_pose(pp, pose(int(j))) for j in selected):
                        selected.append(int(fi))
                        if len(selected) == 3:
                            break
            row = dict(scene=scene, pair_id=p['pair_id'], left=p['left'], right=p['right'],
                       left_track_ids=p['left_track_ids'], right_track_ids=p['right_track_ids'],
                       union_sha256=p['union_sha256'], duplicate_original_NT=p['duplicate_original_NT'],
                       eligible=bool(len(aa) and len(bb) and (len(contact) or len(selected))),
                       exclusive_sizes=[len(aa), len(bb)], contacts=len(contact),
                       reliable_common_frames=int(reliable.sum()), selected_frame_ids=[int(fids[i]) for i in selected],
                       views=[])
            meta.append(row)
            g.append(np.r_[geom, int(reliable.sum()), len(selected)])
            app.append(visual)
            for fi in selected:
                needs[fi].append((pi, aa, bb, core.sample(contact, 128)))
        # Process one frame at a time; no dense RGB-D/visibility cache for the scene.
        projection = Counter()
        for fi in sorted(needs):
            fm = frames[int(fids[fi])]
            for key in ('color', 'depth'):
                checked(fm[key]['path'], fm[key]['sha256'], used)
            rgb = np.asarray(Image.open(fm['color']['path']).convert('RGB'))
            depth = np.asarray(Image.open(fm['depth']['path']), dtype=np.float32)/1000.
            h, w = depth.shape
            vispath = vp[fi]['path']
            visible = np.load(checked(vispath, receipt['inputs'][vispath], used), allow_pickle=False)
            k = core.adapt_intrinsic(intrinsic, (rgb.shape[1], rgb.shape[0]), (w, h))
            px, camera_depth, valid = core.project(raw[:, :3], pose(fi), k)
            in_image = valid & (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < w) & (px[:, 1] < h)
            projection['visible'] += int(visible.sum())
            projection['outside'] += int((visible & ~in_image).sum())
            ix = np.flatnonzero(visible & in_image)
            residual = np.abs(depth[px[ix, 1], px[ix, 0]]-camera_depth[ix])
            projection['depth_error_gt0501'] += int((residual > .0501).sum())
            assert projection['outside'] == 0
            for pi, aa, bb, contacts in needs[fi]:
                av, bv = aa[visible[aa]], bb[visible[bb]]
                assert min(len(av), len(bv)) >= 20
                path = core.depth_path_evidence(depth, px[av], px[bv])
                # RGB values measured in the same source frame as the depth paths.
                def color(ids):
                    pp = px[core.sample(ids)]
                    x = np.clip(pp[:, 0]*rgb.shape[1]/w, 0, rgb.shape[1]-1).astype(int)
                    y = np.clip(pp[:, 1]*rgb.shape[0]/h, 0, rgb.shape[0]-1).astype(int)
                    v = rgb[y, x]/255.
                    return np.r_[v.mean(0), v.std(0)]
                ca, cb = color(av), color(bv)
                cp = contacts[visible[contacts].all(axis=1)] if len(contacts) else contacts
                if len(cp):
                    lpx, rpx = px[cp[:, 0]], px[cp[:, 1]]
                    za, zb = depth[lpx[:, 1], lpx[:, 0]], depth[rpx[:, 1], rpx[:, 0]]
                    ok = (za > 0) & (zb > 0)
                    jumps = np.abs(za[ok]-zb[ok])
                    boundary = np.r_[len(cp), ok.mean(), np.quantile(jumps, [.1, .5, .9]) if len(jumps) else [np.nan]*3]
                else:
                    boundary = np.r_[0, 0, [np.nan]*3]
                values = np.r_[path, (ca+cb)/2, np.abs(ca-cb), boundary]
                meta[pi]['views'].append(dict(frame_id=int(fids[fi]), frame_index=int(findices[fi]),
                    visible_left=len(av), visible_right=len(bv), visible_contacts=len(cp), values=values.tolist()))
        vrows = []
        for row, visual in zip(meta, app):
            values = np.asarray([v['values'] for v in row['views']])
            if len(values):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    agg = np.r_[np.nanmean(values, axis=0), np.nanstd(values, axis=0)]
            else:
                agg = np.full(50, np.nan)
            vrows.append(np.r_[visual, len(values), agg])
        g, vrows = np.asarray(g).reshape(-1, 23), np.asarray(vrows).reshape(-1, 131)
        assert g.shape[1] == 23 and vrows.shape[1] == 131
        assert not np.isinf(g).any() and not np.isinf(vrows).any()
        np.savez_compressed(dest/'features.npz', geometry=g, rgbd=vrows)
        a.seal(dest/'rows.json', meta)
        a.seal(dest/'receipt.json', dict(inputs=used, projection=dict(projection), pairs=len(meta), GT_used=False))
        for name in ('features.npz', 'rows.json', 'receipt.json'):
            outputs[str(dest/name)] = a.sha(dest/name)
        allrows += meta
        allg.extend(g)
        allv.extend(vrows)
        print('features', si+1, len(c['scenes']), scene, len(meta), 'frames', len(needs), projection, flush=True)
    assert len(allrows) == c['candidate_count']
    np.savez_compressed(OUT/'features.npz', geometry=np.asarray(allg), rgbd=np.asarray(allv))
    a.seal(OUT/'feature_rows.json', allrows)
    for name in ('features.npz', 'feature_rows.json'):
        outputs[str(OUT/name)] = a.sha(OUT/name)
    a.seal(OUT/'features_frozen.json', dict(complete=True, GT_used=False, outputs=outputs,
                                         pairs=len(allrows), dimensions=dict(geometry=23, rgbd=131)))


def make_labels():
    guard(False)
    c = contract()
    verify(a.read(OUT/'features_frozen.json')['outputs'])
    sc = a.read(a.OUT/'contract.json')
    old = a.read(a.OLD/'contract.json')
    outputs, counts = {}, Counter()
    for scene in c['scenes']:
        gp = sc['jobs'][scene]['job']['GT']
        assert a.sha(gp) == old['GT_inputs'][gp]
        gt = np.loadtxt(gp, dtype=np.int64)
        rows = a.read(OUT/'features'/scene/'rows.json')
        with np.load(SOURCE/'predictions'/scene/'data.npz') as z:
            nodes = {n for r in rows for n in (r['left'], r['right'])}
            identities = {n: core.endpoint_label(*np.unique(gt[z['n%04d' % n]], return_counts=True)) for n in nodes}
        pairs = [dict(pair_id=r['pair_id'], relation=core.labels(identities[r['left']], identities[r['right']])) for r in rows]
        counts.update(p['relation'] for p in pairs)
        dest = OUT/'labels'/f'{scene}.json'
        a.seal(dest, dict(scene=scene, GT_path=gp, GT_sha256=a.sha(gp), nodes=identities, pairs=pairs))
        outputs[str(dest)] = a.sha(dest)
    a.seal(OUT/'labels_frozen.json', dict(complete=True, counts=dict(counts), outputs=outputs,
                                       not_prediction_inputs=True, features_seal_sha256=a.sha(OUT/'features_frozen.json')))
    print('label inventory', counts, flush=True)


def matrix():
    with np.load(OUT/'features.npz') as z:
        g, v = z['geometry'], z['rgbd']
    return dict(geometry=g, geometry_rgbd=np.c_[g, v])


def fit(fold):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    c = a.read(OUT/'contract.json')
    train_scenes = [s for s in c['scenes'] if c['folds'][s] != fold]
    guard(True, train_scenes)
    c = contract()
    verify(a.read(OUT/'features_frozen.json')['outputs'])
    rows = a.read(OUT/'feature_rows.json')
    lookup, label_inputs = {}, {}
    for s in train_scenes:
        p = OUT/'labels'/f'{s}.json'
        label_inputs[str(p)] = a.sha(p)
        lookup.update({(s, y['pair_id']): y['relation'] for y in a.read(p)['pairs']})
    idx = np.array([i for i, r in enumerate(rows) if lookup.get((r['scene'], r['pair_id'])) in ('same', 'different')])
    y = np.array([lookup[rows[i]['scene'], rows[i]['pair_id']] == 'same' for i in idx], dtype=int)
    scene_n = Counter(rows[i]['scene'] for i in idx)
    weights = np.array([1/scene_n[rows[i]['scene']] for i in idx])
    weights *= len(weights)/weights.sum()
    dest = OUT/'models'/f'fold{fold}'
    dest.mkdir(parents=True)
    for arm, x in matrix().items():
        model = HistGradientBoostingClassifier(**c['model'])
        with threadpool_limits(4):
            model.fit(x[idx], y, sample_weight=weights)
        joblib.dump(model, dest/f'{arm}.joblib')
    a.seal(dest/'receipt.json', dict(fold=fold, train_scenes=train_scenes,
        test_scenes=[s for s in c['scenes'] if c['folds'][s] == fold], label_inputs=label_inputs,
        training_rows=idx.tolist(), training_counts=dict(Counter(y.tolist())),
        outputs={str(dest/f'{arm}.joblib'): a.sha(dest/f'{arm}.joblib') for arm in ARMS}))
    print('fit', fold, len(idx), Counter(y.tolist()), flush=True)


def predict():
    import joblib
    from threadpoolctl import threadpool_limits
    guard()
    c = contract()
    verify(a.read(OUT/'features_frozen.json')['outputs'])
    rows = a.read(OUT/'feature_rows.json')
    folds = np.array([c['folds'][r['scene']] for r in rows])
    scores = {arm: np.full(len(rows), np.nan) for arm in ARMS}
    outputs = {}
    for fold in range(5):
        dest = OUT/'models'/f'fold{fold}'
        receipt = a.read(dest/'receipt.json')
        assert not set(receipt['train_scenes']) & set(receipt['test_scenes'])
        verify(receipt['outputs'])
        outputs.update(receipt['outputs'])
        outputs[str(dest/'receipt.json')] = a.sha(dest/'receipt.json')
        for arm, x in matrix().items():
            model = joblib.load(dest/f'{arm}.joblib')
            with threadpool_limits(4):
                scores[arm][folds == fold] = model.predict_proba(x[folds == fold])[:, 1]
    for arm in ARMS:
        assert np.isfinite(scores[arm]).all()
        np.save(OUT/f'{arm}_oof.npy', scores[arm])
        outputs[str(OUT/f'{arm}_oof.npy')] = a.sha(OUT/f'{arm}_oof.npy')
        allpred = []
        for scene in c['scenes']:
            ix = [i for i, r in enumerate(rows) if r['scene'] == scene]
            local = [rows[i] for i in ix]
            pred = core.decide(local, scores[arm][ix], [r['eligible'] for r in local])
            d = OUT/'predictions'/arm/scene
            d.mkdir(parents=True)
            emitted, arrays = set(), {}
            with np.load(SOURCE/'predictions'/scene/'data.npz') as z:
                for r in pred:
                    if r['state'] != 'accept':
                        continue
                    pp = z[r['pair_id']+'_union']
                    assert a.digest(pp) == r['union_sha256']
                    if r['duplicate_original_NT'] or r['union_sha256'] in emitted:
                        r['reason'] += '_duplicate_geometry_not_emitted'
                        continue
                    emitted.add(r['union_sha256'])
                    arrays[r['pair_id']+'_union'] = pp
                    r['emit'] = True
            np.savez_compressed(d/'unions.npz', **arrays)
            a.seal(d/'pairs.json', pred)
            for name in ('unions.npz', 'pairs.json'):
                outputs[str(d/name)] = a.sha(d/name)
            allpred += pred
        a.seal(OUT/f'{arm}_decisions.json', allpred)
        outputs[str(OUT/f'{arm}_decisions.json')] = a.sha(OUT/f'{arm}_decisions.json')
        print('predictions', arm, Counter(r['state'] for r in allpred), 'emit', sum(r['emit'] for r in allpred), flush=True)
    a.seal(OUT/'predictions_frozen.json', dict(complete=True, inference_GT_used=False,
        supervised_training=True, all_scores_out_of_physical_scene_fold=True,
        outputs=outputs, features_sha256=a.sha(OUT/'features_frozen.json'), time_unix=time.time()))


def evaluate():
    from sklearn.metrics import roc_auc_score, average_precision_score
    import evaluate_fi1_nt_region_identity as met
    import run_fi1_nt_region_identity as diag
    guard(False)
    c = contract()
    verify(a.read(OUT/'predictions_frozen.json')['outputs'])
    verify(a.read(OUT/'labels_frozen.json')['outputs'])
    rows = a.read(OUT/'feature_rows.json')
    lookup = {}
    for s in c['scenes']:
        lookup.update({(s, y['pair_id']): y['relation'] for y in a.read(OUT/'labels'/f'{s}.json')['pairs']})
    label = np.array([1 if lookup[r['scene'], r['pair_id']] == 'same' else 0 if lookup[r['scene'], r['pair_id']] == 'different' else -1 for r in rows])
    previous = {(r['scene'], r['pair_id']): r for r in a.read(SOURCE/'evaluation/pairs.json')}
    old = a.read(a.OLD/'contract.json')
    sc = a.read(a.OUT/'contract.json')
    result, outputs = {}, {}
    for arm in ARMS:
        pred = a.read(OUT/f'{arm}_decisions.json')
        score = np.load(OUT/f'{arm}_oof.npy')
        known = label >= 0
        metric = lambda ix: dict(pairs=int(ix.sum()), roc_auc=float(roc_auc_score(label[ix], score[ix])),
                                 pair_average_precision=float(average_precision_score(label[ix], score[ix])))
        metrics = dict(pooled=metric(known), folds={str(f): metric(known & np.array([c['folds'][r['scene']] == f for r in rows])) for f in range(5)})
        joined, targets = [], []
        for s in c['scenes']:
            gp = sc['jobs'][s]['job']['GT']
            assert a.sha(gp) == old['GT_inputs'][gp]
            gt = np.loadtxt(gp, dtype=np.int64)
            ids, sizes = met.valid_targets(gt, old['class_ids'])
            _, nt = diag.load_nt(sc['jobs'][s]['job'])
            baseline = met.pool_base(nt, gt, ids, sizes)
            best = baseline.copy()
            with np.load(SOURCE/'predictions'/s/'data.npz') as z:
                for r in [r for r in pred if r['scene'] == s]:
                    prior = previous[s, r['pair_id']]
                    row = dict(**r, GT_relation_strict=lookup[s, r['pair_id']],
                        GT_relation_previous=prior['relation'], safe_gain02=prior['safe_gain02'],
                        same_target=prior['same_target'], same_target_pool_delta=prior['same_target_pool_delta'],
                        union_target_purity=prior['union_target_purity'], mixed_endpoint=prior['mixed_endpoint'],
                        good_endpoint_loss_gt02=prior['good_endpoint_loss_gt02'], union_GT=prior['union_GT'],
                        old_state=prior['state'], old_merge_winning_frames=prior['merge_winning_frames'])
                    joined.append(row)
                    if r['emit']:
                        best = np.maximum(best, met.ious(z[r['pair_id']+'_union'], gt, ids, sizes))
            targets.extend(met.pool_rows(s, arm, baseline, best, ids))
        accepted = [r for r in joined if r['state'] == 'accept']
        emitted = [r for r in accepted if r['emit']]
        safe = [r for r in emitted if r['safe_gain02']]
        fraction_unknown = sum(r['union_GT']['unknown'] for r in emitted)/max(1, sum(r['union_GT']['points'] for r in emitted))
        gates = dict(safe_gain_scenes=len({r['scene'] for r in safe}) >= 2,
            no_confirmed_cross_identity=not any(r['GT_relation_previous'] == 'different' for r in accepted),
            no_unresolved_accept=not any(r['GT_relation_previous'] == 'unresolved' for r in accepted),
            full_endpoint_identity=bool(accepted) and all(r['GT_relation_strict'] == 'same' for r in accepted),
            good_copy_safety=not any(r['good_endpoint_loss_gt02'] for r in accepted),
            annotation_unknown=fraction_unknown <= .1)
        result[arm] = dict(metrics=metrics, decisions=dict(Counter(r['state'] for r in joined)),
            identity_by_state={state: dict(Counter(r['GT_relation_strict'] for r in joined if r['state'] == state)) for state in ('accept', 'reject', 'unknown')},
            previous_identity_by_state={state: dict(Counter(r['GT_relation_previous'] for r in joined if r['state'] == state)) for state in ('accept', 'reject', 'unknown')},
            emitted=len(emitted), safe_gain_pairs=len(safe), safe_gain_targets=len({(r['scene'], r['same_target']) for r in safe}),
            safe_gain_scenes=sorted({r['scene'] for r in safe}), accepted_mixed=sum(r['mixed_endpoint'] for r in accepted),
            good_endpoint_loss_gt02=sum(r['good_endpoint_loss_gt02'] for r in accepted), annotation_unknown_fraction=fraction_unknown,
            pool=met.pool_summary(targets), pool_gain02_scenes=sorted({r['scene'] for r in targets if r['gain02']}),
            gates=gates, passed=all(gates.values()),
            valuable_supply=[dict(scene=r['scene'], pair_id=r['pair_id'], score=r['score'], state=r['state'],
                emit=r['emit'], strict_relation=r['GT_relation_strict'], delta=r['same_target_pool_delta'],
                old_merge_winning_frames=r['old_merge_winning_frames']) for r in joined if r['safe_gain02']])
        a.seal(OUT/'evaluation'/f'{arm}_pairs.json', joined)
        a.seal(OUT/'evaluation'/f'{arm}_targets.json', targets)
        for name in (f'{arm}_pairs.json', f'{arm}_targets.json'):
            outputs[str(OUT/'evaluation'/name)] = a.sha(OUT/'evaluation'/name)
        with (OUT/f'{arm}_evidence.csv').open('x', newline='') as f:
            compact = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in r.items() if k != 'views'} for r in joined]
            w = csv.DictWriter(f, fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
        outputs[str(OUT/f'{arm}_evidence.csv')] = a.sha(OUT/f'{arm}_evidence.csv')
    primary = result['geometry_rgbd']
    a.seal(OUT/'results.json', dict(complete=True, exposed_development_only=True, results=result,
        decision='propose_isolated_validation' if primary['passed'] else 'stop_fixed_configuration',
        no_AP=True, no_production_change=True, no_old_reject_labels=True))
    outputs[str(OUT/'results.json')] = a.sha(OUT/'results.json')
    a.seal(OUT/'evaluation_frozen.json', dict(complete=True, prediction_sha256=a.sha(OUT/'predictions_frozen.json'), outputs=outputs))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'features', 'labels', 'fit', 'predict', 'evaluate'))
    p.add_argument('--fold', type=int, choices=range(5))
    args = p.parse_args()
    if args.mode == 'fit':
        assert args.fold is not None
        fit(args.fold)
    elif args.mode == 'labels':
        make_labels()
    else:
        globals()[args.mode]()

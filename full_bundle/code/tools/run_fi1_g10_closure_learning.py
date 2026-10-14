#!/usr/bin/env python3
"""Matched full597 B/Q fitting and dev55/full312 evaluation using FI1 kernels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from tools.run_fi1_g10_closure_assets import LOCAL, OUT, HISTORY, read, require, now
from tools.fi1_frontend_protocol import digest, freeze_json, sha256_file as sha, write_json as write
from tools.run_fi1_g10_matched_oof import load_q_runner, write_jsonl
from tools.run_fi1_train597 import compute, evidence_compatibility, validated_arrays, verify, files_at, resources
from tools.fi1_r0_r1_protocol import PROTOCOL, build_pair
from tools.fi1_r2_core import FEATURES, MODEL_PARAMS, geometry_features, select
from tools.audit_fi1_r2_dataset import features as independent_features
from tools.fi1_r3_core import EXTRA_FEATURES, pair_overlap, context_features, independent_context
from tools.fi1_geometry_feature_ablation import Q, extra_features
from tools.prepare_fi1_r3b_val312 import no_gt_io
from tools.run_fi1_r3b_val312 import validate_selected
from tools.analyze_fi1_r3 import configure_evaluator, compact_metrics, validate_ap_array
from evaluate.scannet200 import eval_semantic_instance as evaluator

DEST = OUT / 'learning'
REFERENCE = HISTORY / 'alpha_completed_v1/ncs_validation60'
OLD_TRAIN = OUT.parent / 'fi1_R3b_train597_20260917'


def status(stage, **kw):
    write(LOCAL / 'learning_execution.json', {'stage': stage, 'updated_at': now(), **kw})
    print(stage, kw, flush=True)


def initialize():
    audit = read(LOCAL / 'learning_code_audit.json')
    verify(audit['code'])
    inventory = read(LOCAL / 'inventory.json')
    identity = {'inventory_sha256': sha(LOCAL / 'inventory.json'), 'groups': inventory['groups'],
        'code': audit['code'], 'features_B': FEATURES + EXTRA_FEATURES, 'features_Q': FEATURES + EXTRA_FEATURES + Q,
        'model_parameters': MODEL_PARAMS, 'scene_budget': PROTOCOL['scene_budget'],
        'fit_group': 'train', 'GT_features': False, 'GT_training_labels_only': True,
        'Q_primary_B_control': True, 'validation_is_blind': False, 'full312_authorized': True,
        'parameter_search': False, 'frontend': 'grounded_sam_stride10', 'Dv3_retrained': False}
    freeze_json(DEST / 'identity.json', identity)
    return identity, inventory


def gt_binding(scene, group):
    """Use previously reconstructed/evaluated GT with its frozen checksum."""
    if group == 'train':
        p = OLD_TRAIN / 'dataset' / scene / 'receipt.json'
        r = read(p)
        files = r['inputs']
    elif group == 'dev':
        p = HISTORY / 'r2_v1/evaluation/gt_binding.json'
        require(sha(p) == read(HISTORY / 'r2_v1/evaluation/summary.json')['gt_binding_sha256'], 'dev GT binding changed')
        files = next(r['files'] for r in read(p)['rows'] if r['scene_name'] == scene)
    else:
        p = ROOT / '.local/fi1_r3b_val312_execution_20260914/continued_after_alpha_key_fix/evaluation' / scene / 'receipt.json'
        files = read(p)['inputs']
    candidates = {p: h for p, h in files.items() if p.endswith('/' + scene + '.txt') and 'ground_truth' in p}
    require(len(candidates) == 1, 'missing/ambiguous certified GT')
    verify(candidates)
    return Path(next(iter(candidates))), {**candidates, str(p): sha(p)}


def inference_features(geometry, points, evidence, count):
    members = build_pair(geometry, evidence, PROTOCOL['scene_budget'])['R1']['eligible']
    require(bool(members), 'empty member set')
    keys = [g['geometry_key'] for g in geometry]
    require(len(keys) == len(set(keys)) == len(points) == len(evidence) and set(keys) == set(evidence), 'geometry coverage differs')
    own = {}
    for g, p in zip(geometry, points):
        require(p.ndim == 1 and p.dtype.kind in 'iu' and np.array_equal(p, np.unique(p)), 'invalid point indices')
        require(len(p) == g['point_count'] and (not len(p) or (p[0] >= 0 and p[-1] < count)), 'point domain differs')
        require(hashlib.sha256(np.asarray(p, dtype='<i8').tobytes()).hexdigest() == g['points_sha256'], 'point digest differs')
        own[g['geometry_key']] = geometry_features(g, evidence[g['geometry_key']])
        require(np.allclose(own[g['geometry_key']], independent_features(g, evidence[g['geometry_key']]), atol=1e-12, rtol=0),
                'independent own features differ')
    x = np.asarray([own[m['geometry_key']][m['class_index']] for m in members], dtype=np.float64)
    iou, cover = pair_overlap(points, count)
    context = context_features(members, keys, iou, cover)
    require(np.allclose(context, independent_context(members, keys, iou, cover), atol=1e-14, rtol=0), 'context differs')
    return np.column_stack((x, context)), members


def prepare_scene(scene, identity, inventory, qrunner):
    job = inventory['jobs'][scene]
    dest = DEST / 'dataset' / scene
    rp = dest / 'receipt.json'
    if rp.exists():
        r = read(rp)
        require(r['identity_sha256'] == digest(identity), 'dataset identity changed')
        verify(r['inputs']); verify(files_at(dest, r['outputs']))
        return r
    source_audit = Path(job['folder']) / 'audit.json'
    a = read(source_audit)
    require(a['complete'] and a['scene'] == scene and a['strict_stride'] == 10, 'source asset audit incomplete')
    verify(a['inputs'])
    own = Path(a['own'])
    binding, evidence_root, views = own / 'geometry_binding', own / 'alpha_completed', own / 'views'
    br = read(binding / scene / 'receipt.json'); er = read(evidence_root / scene / 'receipt.json')
    bi, ei, vi = (read(p / 'identity.json') for p in (binding, evidence_root, views))
    reference = read(REFERENCE / 'identity.json')
    evidence_compatibility(bi, ei, vi, reference, read(Path(reference['views']) / 'identity.json'))
    require(br['identity_sha256'] == digest(bi) and er['identity_sha256'] == digest(ei), 'scene binding identity mismatch')
    gp, pp, ep = binding / scene / 'geometry.json', binding / scene / 'points.npz', evidence_root / scene / 'evidence.jsonl'
    inputs = {str(source_audit): sha(source_audit), str(gp): br['output_files']['geometry.json'],
              str(pp): br['output_files']['points.npz'], str(ep): er['outputs']['evidence.jsonl']}
    for p in (binding / 'identity.json', evidence_root / 'identity.json', views / 'identity.json',
              binding / scene / 'receipt.json', evidence_root / scene / 'receipt.json'):
        inputs[str(p)] = sha(p)
    processed = Path(job['data']) / scene / f'{scene[5:]}.npy'
    inputs[str(processed)] = sha(processed)
    verify(inputs)
    count = np.load(processed, mmap_mode='r').shape[0]
    geometry = read(gp)
    erows = [json.loads(s) for s in ep.read_text().splitlines()]
    evidence = {r['geometry_key']: r for r in erows}
    require(len(erows) == len(evidence), 'duplicate evidence keys')
    with np.load(pp, allow_pickle=False) as z:
        points = [z[g['array_key']] for g in geometry]
    x26, members = inference_features(geometry, points, evidence, count)
    data = {'x26': x26}
    if job['group'] == 'train':
        gt, gf = gt_binding(scene, 'train')
        inputs.update(gf)
        labels = np.loadtxt(gt, dtype=np.int64)
        require(len(labels) == count, 'GT/geometry point domain differs')
        supervised, m2, error = compute(geometry, points, evidence, labels)
        sx, sy, sw = validated_arrays(supervised)
        require(m2 == members and np.array_equal(sx, x26), 'training/inference feature parity failed')
        data.update(y=sy, weights=sw)
    index, quality_inputs = qrunner.quality_index(Path(a['backend']) / 'D' / scene / 'dv3', scene)
    inputs.update(quality_inputs)
    extra, stats = extra_features(scene, geometry, points, index)
    require(np.allclose(extra[:, :3], qrunner.independent_quality(geometry, index), atol=1e-14, rtol=0), 'quality parity differs')
    lookup = {g['geometry_key']: i for i, g in enumerate(geometry)}
    data['quality'] = extra[[lookup[m['geometry_key']] for m in members], :3]
    require(x26.shape == (len(members), 26) and data['quality'].shape == (len(members), 3), 'feature width differs')
    require(all(np.isfinite(x).all() for x in data.values()), 'nonfinite features/targets')
    verify(inputs)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'features.npz', **data)
    write_jsonl(dest / 'members.jsonl', members)
    r = {'complete': True, 'identity_sha256': digest(identity), 'scene': scene, 'group': job['group'],
         'members': len(members), 'point_count': count, 'geometry_file': str(gp), 'points_file': str(pp),
         'quality_statistics': stats, 'GT_read': job['group'] == 'train', 'inputs': inputs,
         'outputs': {p: sha(dest / p) for p in ('features.npz', 'members.jsonl')}}
    write(rp, r)
    return r


def prepare(group, pilot=False):
    if group != 'train':
        no_gt_io()
    identity, inventory = initialize()
    scenes = inventory['groups'][group][:3] if pilot else inventory['groups'][group]
    qrunner = load_q_runner(); receipts = {}
    for i, scene in enumerate(scenes, 1):
        prepare_scene(scene, identity, inventory, qrunner)
        p = DEST / 'dataset' / scene / 'receipt.json'
        receipts[str(p)] = sha(p)
        status('prepare_' + group, completed=i, total=len(scenes), scene=scene)
    write(DEST / ('pilot_dataset.json' if pilot else group + '_dataset.json'),
          {'complete': True, 'identity_sha256': digest(identity), 'scenes': scenes, 'receipts': receipts})


def arrays(scene):
    folder = DEST / 'dataset' / scene
    r = read(folder / 'receipt.json')
    verify(files_at(folder, r['outputs']))
    with np.load(folder / 'features.npz', allow_pickle=False) as z:
        data = dict(z)
    return r, data


def fit(pilot=False):
    identity, inventory = initialize()
    summary = read(DEST / ('pilot_dataset.json' if pilot else 'train_dataset.json'))
    scenes = inventory['groups']['train'][:3] if pilot else inventory['groups']['train']
    require(summary['complete'] and summary['scenes'] == scenes and summary['identity_sha256'] == digest(identity), 'fit dataset differs')
    verify(summary['receipts'])
    count = sum(read(Path(p))['members'] for p in summary['receipts'])
    modelroot = DEST / ('pilot_models' if pilot else 'models')
    modelroot.mkdir(parents=True, exist_ok=True)
    for head, width in [('B', 26), ('Q', 29)]:
        receipt = modelroot / f'{head}.json'
        if receipt.exists():
            r = read(receipt)
            require(r['identity_sha256'] == digest(identity) and r['dataset_sha256'] == sha(DEST / ('pilot_dataset.json' if pilot else 'train_dataset.json')), 'fit identity differs')
            verify(r['outputs']); continue
        required = max(8 * 1024**3, count * width * 8 * 5 + 12 * 1024**3)
        while resources()[0] < required:
            status('waiting_for_fit_memory', required_bytes=required)
            time.sleep(30)
        x = np.empty((count, width), dtype=np.float64)
        y = np.empty(count); w = np.empty(count); start = 0
        for scene in scenes:
            r, data = arrays(scene)
            require(r['group'] == 'train' and r['GT_read'], 'nontraining data in fit')
            end = start + len(data['x26'])
            x[start:end] = data['x26'] if head == 'B' else np.column_stack((data['x26'], data['quality']))
            y[start:end], w[start:end] = data['y'], data['weights']; start = end
        require(start == count, 'fit row count differs')
        w *= len(w) / w.sum()
        status('fitting', head=head, rows=count, scans=len(scenes), pilot=pilot)
        start_time = time.monotonic()
        model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(x, y, sample_weight=w)
        require(model.n_iter_ == 100 and model.n_features_in_ == width, 'fit incomplete')
        # Independently instantiate/refit; compare every row in bounded chunks.
        second = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(x, y.copy(), sample_weight=w.copy())
        mp, ap = modelroot / f'{head}.joblib', modelroot / f'{head}_independent.joblib'
        joblib.dump(model, mp); joblib.dump(second, ap)
        saved = joblib.load(mp)
        for i in range(0, count, 100000):
            chunk = x[i:i + 100000]
            require(np.array_equal(model.predict(chunk), saved.predict(chunk)), 'model persistence replay failed')
            require(np.allclose(model.predict(chunk), second.predict(chunk), atol=1e-12, rtol=0), 'independent refit failed')
        write(receipt, {'complete': True, 'identity_sha256': digest(identity),
            'dataset_sha256': sha(DEST / ('pilot_dataset.json' if pilot else 'train_dataset.json')),
            'fit_scenes': scenes, 'fit_rows': count, 'features': width, 'parameters': MODEL_PARAMS,
            'seconds': time.monotonic() - start_time, 'independent_refit_all_training_rows': True,
            'outputs': {str(p): sha(p) for p in (mp, ap)}})
        del x, y, w, model, second, saved
    write(modelroot / 'summary.json', {'complete': True, 'identity_sha256': digest(identity), 'pilot': pilot,
        'fit_scenes': scenes, 'receipts': {str(modelroot / f'{h}.json'): sha(modelroot / f'{h}.json') for h in ('B', 'Q')}})


def predict(group):
    require(group in ('dev', 'val'), 'prediction group must be held-out')
    no_gt_io()
    identity, inventory = initialize()
    fit_summary = read(DEST / 'models/summary.json')
    require(fit_summary['complete'] and not fit_summary['pilot'] and fit_summary['fit_scenes'] == inventory['groups']['train'], 'full fit required')
    verify(fit_summary['receipts'])
    ds = read(DEST / (group + '_dataset.json'))
    require(ds['complete'] and ds['scenes'] == inventory['groups'][group], 'incomplete prediction dataset')
    verify(ds['receipts'])
    models = {}
    for h in ('B', 'Q'):
        mr = read(DEST / 'models' / f'{h}.json'); verify(mr['outputs'])
        models[h] = (joblib.load(DEST / 'models' / f'{h}.joblib'), joblib.load(DEST / 'models' / f'{h}_independent.joblib'))
    receipts = {}
    for i, scene in enumerate(inventory['groups'][group], 1):
        r, data = arrays(scene)
        require(r['group'] == group and not r['GT_read'] and 'y' not in data, 'heldout features accessed GT')
        members = [json.loads(s) for s in (DEST / 'dataset' / scene / 'members.jsonl').read_text().splitlines()]
        dest = DEST / 'predictions' / scene
        inputs = {str(DEST / 'dataset' / scene / 'receipt.json'): sha(DEST / 'dataset' / scene / 'receipt.json'),
                  str(DEST / 'models/summary.json'): sha(DEST / 'models/summary.json')}
        rp = dest / 'receipt.json'
        if rp.exists():
            prior = read(rp)
            require(prior['inputs'] == inputs and prior['identity_sha256'] == digest(identity), 'prediction resume differs')
            verify(files_at(dest, prior['outputs']))
        else:
            dest.mkdir(parents=True, exist_ok=True)
            for h in ('B', 'Q'):
                x = data['x26'] if h == 'B' else np.column_stack((data['x26'], data['quality']))
                scores = np.clip(models[h][0].predict(x), 0, 1)
                audit_scores = np.clip(models[h][1].predict(x), 0, 1)
                require(np.allclose(scores, audit_scores, atol=1e-12, rtol=0), 'heldout refit replay differs')
                chosen = select(members, scores)
                require([(m['geometry_key'], m['class_index']) for m in chosen] ==
                        [(m['geometry_key'], m['class_index']) for m in select(members, audit_scores)],
                        'independent top600 differs')
                validate_selected(members, scores, chosen)
                np.save(dest / f'{h}_scores.npy', scores)
                write(dest / f'{h}_selected.json', chosen)
            write(rp, {'identity_sha256': digest(identity), 'inputs': inputs,
                'outputs': {p: sha(dest / p) for h in ('B', 'Q') for p in (f'{h}_scores.npy', f'{h}_selected.json')}})
        receipts[str(rp)] = sha(rp)
        status('predict_' + group, completed=i, total=len(inventory['groups'][group]), scene=scene)
    write(DEST / (group + '_predictions.json'), {'complete': True, 'identity_sha256': digest(identity),
        'scenes': inventory['groups'][group], 'receipts': receipts})


def evaluate(group):
    require(group in ('dev', 'val'), 'evaluation only on fixed holdouts')
    identity, inventory = initialize()
    ps = read(DEST / (group + '_predictions.json'))
    require(ps['complete'] and ps['scenes'] == inventory['groups'][group] and ps['identity_sha256'] == digest(identity), 'predictions incomplete')
    verify(ps['receipts'])
    # Both heads and predictions are frozen before opening any held-out GT.
    freeze_json(DEST / (group + '_evaluation_contract.json'), {'identity_sha256': digest(identity),
        'prediction_summary_sha256': sha(DEST / (group + '_predictions.json')),
        'model_summary_sha256': sha(DEST / 'models/summary.json'), 'heads': ['B', 'Q'],
        'primary': 'Q', 'search': False, 'scenes': inventory['groups'][group]})
    configure_evaluator()
    matches = {'B': {}, 'Q': {}}
    for i, scene in enumerate(inventory['groups'][group], 1):
        r, _ = arrays(scene)
        gt, inputs = gt_binding(scene, group)
        require(len(np.loadtxt(gt, dtype=np.int64)) == r['point_count'], 'evaluation point domain mismatch')
        pred = DEST / 'predictions' / scene
        pr = read(pred / 'receipt.json'); verify(files_at(pred, pr['outputs']))
        inputs.update({str(pred / 'receipt.json'): sha(pred / 'receipt.json'),
                       r['geometry_file']: r['inputs'][r['geometry_file']], r['points_file']: r['inputs'][r['points_file']]})
        verify(inputs)
        dest = DEST / 'evaluation' / group / scene; rp = dest / 'receipt.json'
        if rp.exists():
            prior = read(rp)
            require(prior['inputs'] == inputs and prior['identity_sha256'] == digest(identity), 'evaluation resume changed')
            verify(files_at(dest, prior['outputs']))
            for h in matches:
                with (dest / f'{h}.pkl').open('rb') as f:
                    matches[h][scene] = pickle.load(f)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            gs = {g['geometry_key']: g for g in read(r['geometry_file'])}
            with np.load(r['points_file'], allow_pickle=False) as pts:
                for h in matches:
                    chosen = read(pred / f'{h}_selected.json')
                    masks = np.zeros((r['point_count'], len(chosen)), dtype=bool)
                    for k, m in enumerate(chosen):
                        masks[pts[gs[m['geometry_key']]['array_key']], k] = True
                    gg, pp = evaluator.assign_instances_for_scan({'pred_masks': masks,
                        'pred_classes': np.asarray([m['class_index'] for m in chosen], dtype=np.int64),
                        'pred_scores': np.asarray([m['score'] for m in chosen])}, str(gt))
                    matches[h][scene] = {'gt': gg, 'pred': pp}
                    with (dest / f'{h}.pkl').open('wb') as f:
                        pickle.dump(matches[h][scene], f)
                    del masks
            write(rp, {'identity_sha256': digest(identity), 'inputs': inputs,
                'outputs': {f'{h}.pkl': sha(dest / f'{h}.pkl') for h in matches}})
        status('match_' + group, completed=i, total=len(inventory['groups'][group]), scene=scene)
    results = {}
    for h in matches:
        aps, *_ = evaluator.evaluate_matches(matches[h])
        validate_ap_array(aps); results[h] = compact_metrics(aps)
        np.save(DEST / 'evaluation' / f'{group}_{h}_AP.npy', aps)
    write(DEST / f'{group}_summary.json', {'complete': True, 'identity_sha256': digest(identity),
        'AP': results, 'AP_units': 'fraction', 'scenes': len(inventory['groups'][group]), 'train_scans': 597,
        'matched_G10_training': True, 'validation_is_blind': False, 'Q_primary_B_control': True})
    status('evaluated_' + group, AP=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fit', 'predict', 'evaluate'])
    parser.add_argument('--group', choices=['train', 'dev', 'val'], default='train')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.command == 'prepare': prepare(args.group, args.pilot)
        elif args.command == 'fit': fit(args.pilot)
        elif args.command == 'predict': predict(args.group)
        else: evaluate(args.group)

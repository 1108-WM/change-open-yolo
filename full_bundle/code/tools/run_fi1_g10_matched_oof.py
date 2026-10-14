#!/usr/bin/env python3
"""Five-fold A-control versus strict-G10 matched R3b B/Q experiment.

This is a thin orchestration layer over the existing audited kernels:

* ``run_fi1_train597.compute`` rebuilds the unchanged 26-dimensional R3b
  members, features, context, competition targets, and scene weights.
* the completed full597-Q runner supplies the unchanged three Q fields.
* ``HistGradientBoostingRegressor(**MODEL_PARAMS)`` and ``select`` remain the
  existing scoring and top-600 implementations.

Each fold fits four heads on the same 24 main scenes: A-B, A-Q, G10-B, and
G10-Q.  Only the six untouched G10 main scenes are evaluated.  The 30 OOF
predictions are pooled for the primary AP result.  Existing full597 heads are
not used as controls because they were trained on all 30 A-domain scenes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import contextlib
import datetime
import fcntl
import importlib.util
import json
import os
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

from tools.analyze_fi1_r3 import compact_metrics, configure_evaluator
from tools.fi1_geometry_feature_ablation import extra_features, matrix, rows
from tools.fi1_r0_r1_protocol import PROTOCOL
from tools.fi1_r2_core import MODEL_PARAMS, select
from tools.prepare_fi1_r0_r1_inputs import digest, read, sha
from tools.prepare_scannet200_train_scene import DEFAULT_LABEL_MAP
from tools.run_fi1_train597 import (
    compute,
    evidence_compatibility,
    files_at,
    reconstruct_gt,
    validated_arrays,
    verify,
    write,
)
from evaluate.scannet200 import eval_semantic_instance as ev

BASE = Path('/media/jia/软件1/scannet_additional_train_stream')
G10 = BASE / 'fi1_G10_matched_pilot30_20260919'
TRAIN = BASE / 'fi1_R3b_train597_20260917'
QROOT = ROOT / '.local/fi1_full597_Q_20260918'
Q_RUNNER = QROOT / 'run.py'
FOLDS = ROOT / '.local/fi1_g10_matched_pilot30_20260919_v1/selection/folds.json'
G10_EXECUTION = ROOT / '.local/fi1_g10_matched_pilot30_20260919_v1/execution.json'
OUT = ROOT / '.local/fi1_g10_matched_oof_20260919_v1'
REFERENCE_ALPHA = ROOT / '.local/fi1_r0_r4_preparation_20260913/alpha_completed_v1/ncs_validation60'

ARMS = ('A_B_control', 'A_Q_control', 'G10_B_matched', 'G10_Q_matched')
WIDTHS = {'A_B_control': 26, 'A_Q_control': 29,
          'G10_B_matched': 26, 'G10_Q_matched': 29}
DOMAINS = {'A_B_control': 'A', 'A_Q_control': 'A',
           'G10_B_matched': 'G10', 'G10_Q_matched': 'G10'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def load_q_runner():
    """Import the completed Q runner so quality extraction is not reimplemented."""
    spec = importlib.util.spec_from_file_location('fi1_full597_q_frozen_runner', Q_RUNNER)
    require(spec is not None and spec.loader is not None, 'cannot load frozen Q runner')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                            allow_nan=False) + '\n' for row in records))
    temporary.replace(path)


def fold_contract() -> dict:
    payload = read(FOLDS)
    require(len(payload['folds']) == 5, 'expected five folds')
    holdout = [scene for fold in payload['folds'] for scene in fold['holdout_scans']]
    require(len(holdout) == len(set(holdout)) == 30, 'OOF holdout coverage differs')
    require(len({scene.split('_')[0] for scene in holdout}) == 30,
            'OOF main-scene coverage differs')
    for index, fold in enumerate(payload['folds']):
        require(fold['fold'] == index, 'fold order differs')
        require(len(fold['fit_scans']) == 24 and len(fold['holdout_scans']) == 6,
                'fold size differs')
        require(set(fold['fit_scans']).isdisjoint(fold['holdout_scans']), 'fold leak')
        require(set(fold['fit_scans']) | set(fold['holdout_scans']) == set(holdout),
                'fold coverage differs')
        require(set(fold['fit_main_scenes']).isdisjoint(fold['holdout_main_scenes']),
                'main-scene leak')
    return payload


def initialize() -> tuple[dict, dict]:
    folds = fold_contract()
    frontend = read(G10_EXECUTION)
    require(frontend['status'] == 'complete' and len(frontend['completed']) == 30,
            'G10 pilot30 is not complete')
    require(all(row['complete'] and not row['training_run'] and not row['AP_run']
                for row in frontend['completed']), 'G10 frontend audit differs')
    sources = [Path(__file__), ROOT / 'tools/run_fi1_train597.py',
               ROOT / 'tools/fi1_geometry_feature_ablation.py',
               ROOT / 'tools/fi1_r0_r1_protocol.py', ROOT / 'tools/fi1_r2_core.py',
               ROOT / 'tools/fi1_r3_core.py', ROOT / 'tools/analyze_fi1_r3.py',
               ROOT / 'evaluate/scannet200/eval_semantic_instance.py', Q_RUNNER]
    inputs = {str(path): sha(path) for path in (FOLDS, G10_EXECUTION,
                                                TRAIN / 'identity.json',
                                                QROOT / 'identity.json')}
    identity = {
        'version': 'fi1_g10_matched_oof_v1',
        'folds': folds,
        'folds_sha256': sha(FOLDS),
        'arms': list(ARMS),
        'widths': WIDTHS,
        'model_parameters': MODEL_PARAMS,
        'target': 'unchanged R3b y_competition',
        'weights': 'unchanged per-scene weights; global fit mean weight 1',
        'scene_budget': PROTOCOL['scene_budget'],
        'primary_evaluation_domain': 'strict G10 heldout scenes only',
        'control': 'same 24 fit main scenes, A-domain arrays versus G10-domain arrays',
        'selection_rule': 'pooled 30-scene OOF main AP; AP50/AP25 auxiliary',
        'full597_heads_in_decision': False,
        'full597_exclusion_reason': 'full597 heads saw all 30 A-domain labels',
        'true_delta_ap_training': False,
        'Qwen': False,
        'new_visual_inference': False,
        'inputs': inputs,
        'code': {str(path): sha(path) for path in sources},
    }
    identity_path = OUT / 'identity.json'
    if identity_path.exists():
        require(read(identity_path) == identity, 'frozen OOF identity changed')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        write(identity_path, identity)
    return identity, folds


def a_scene(scene: str) -> dict:
    base_receipt_path = TRAIN / 'dataset' / scene / 'receipt.json'
    q_receipt_path = QROOT / 'dataset' / scene / 'receipt.json'
    base_receipt, q_receipt = read(base_receipt_path), read(q_receipt_path)
    require(base_receipt['complete'] and base_receipt['scene'] == scene,
            'A base receipt differs: ' + scene)
    require(q_receipt['scene'] == scene and q_receipt['members'] == base_receipt['members'],
            'A Q receipt differs: ' + scene)
    feature_path = TRAIN / 'dataset' / scene / 'features.npz'
    quality_path = QROOT / 'dataset' / scene / 'quality.npy'
    verify({str(feature_path): base_receipt['outputs']['features.npz'],
            str(quality_path): q_receipt['outputs']['quality.npy']})
    with np.load(feature_path, allow_pickle=False) as packed:
        data = {name: packed[name] for name in packed.files}
    x26, y, weights = validated_arrays(data)
    quality = np.load(quality_path, allow_pickle=False)
    x29 = np.column_stack((x26, quality))
    require(x29.shape == (len(x26), 29) and np.isfinite(x29).all(),
            'A Q matrix differs: ' + scene)
    return {'x26': x26, 'x29': x29, 'y': y, 'weights': weights,
            'members': len(x26)}


def g10_paths(scene: str) -> dict[str, Path]:
    root = G10 / 'scenes' / scene
    own = root / 'own/D'
    return {
        'root': root,
        'audit': root / 'audit.json',
        'binding': own / 'geometry_binding',
        'geometry': own / 'geometry_binding' / scene / 'geometry.json',
        'points': own / 'geometry_binding' / scene / 'points.npz',
        'evidence': own / 'alpha_completed' / scene / 'evidence.jsonl',
        'evidence_root': own / 'alpha_completed',
        'views': own / 'views',
        'quality_root': root / 'backend/D' / scene / 'dv3',
        'prepared': BASE / 'prepared' / scene / f'{scene[5:]}.npy',
        'gt': BASE / 'prepared/ground_truth' / f'{scene}.txt',
    }


def prepare_g10_scene(scene: str, identity: dict, qrunner) -> dict:
    destination = OUT / 'dataset/G10' / scene
    receipt_path = destination / 'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['identity_sha256'] == digest(identity), 'G10 prepared identity differs')
        verify(receipt['inputs'])
        verify(files_at(destination, receipt['outputs']))
        return receipt
    p = g10_paths(scene)
    audit = read(p['audit'])
    require(audit['complete'] and audit['scene'] == scene and audit['strict_stride'] == 10,
            'G10 source audit differs')
    binding_receipt = read(p['binding'] / scene / 'receipt.json')
    evidence_receipt = read(p['evidence'].parent / 'receipt.json')
    binding_identity = read(p['binding'] / 'identity.json')
    evidence_identity = read(p['evidence_root'] / 'identity.json')
    views_identity = read(p['views'] / 'identity.json')
    reference_evidence = read(REFERENCE_ALPHA / 'identity.json')
    reference_views = read(Path(reference_evidence['views']) / 'identity.json')
    evidence_compatibility(binding_identity, evidence_identity, views_identity,
                           reference_evidence, reference_views)
    require(binding_receipt['identity_sha256'] == digest(binding_identity),
            'G10 geometry receipt differs')
    require(evidence_receipt['identity_sha256'] == digest(evidence_identity),
            'G10 evidence receipt differs')
    bound = {
        str(p['audit']): sha(p['audit']),
        str(p['binding'] / scene / 'receipt.json'): sha(p['binding'] / scene / 'receipt.json'),
        str(p['evidence'].parent / 'receipt.json'): sha(p['evidence'].parent / 'receipt.json'),
        str(p['geometry']): binding_receipt['output_files']['geometry.json'],
        str(p['points']): binding_receipt['output_files']['points.npz'],
        str(p['evidence']): evidence_receipt['outputs']['evidence.jsonl'],
    }
    verify(bound)
    geometry = read(p['geometry'])
    evidence_rows = rows(p['evidence'])
    evidence = {row['geometry_key']: row for row in evidence_rows}
    require(len(evidence) == len(evidence_rows), 'duplicate G10 evidence keys')
    with np.load(p['points'], allow_pickle=False) as packed:
        points = [packed[row['array_key']] for row in geometry]
    gt_identity = reconstruct_gt(scene, BASE / 'scans', p['prepared'], DEFAULT_LABEL_MAP, p['gt'])
    bound.update(gt_identity['files'])
    data, members, error = compute(geometry, points, evidence,
                                   np.loadtxt(p['gt'], dtype=np.int64))
    quality_index, quality_files = qrunner.quality_index(p['quality_root'], scene)
    bound.update(quality_files)
    extra, quality_stats = extra_features(scene, geometry, points, quality_index)
    lookup = {row['geometry_key']: index for index, row in enumerate(geometry)}
    quality = extra[[lookup[member['geometry_key']] for member in members], :3]
    independent = qrunner.independent_quality(geometry, quality_index)
    require(np.allclose(extra[:, :3], independent, atol=1e-14, rtol=0),
            'independent G10 Q extraction differs')
    require(quality.shape == (len(members), 3) and np.isfinite(quality).all(),
            'G10 Q matrix differs')
    verify(bound)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / 'features.npz', **data)
    np.save(destination / 'quality.npy', quality, allow_pickle=False)
    write_jsonl(destination / 'members.jsonl', members)
    receipt = {
        'complete': True, 'identity_sha256': digest(identity), 'scene': scene,
        'members': len(members), 'geometries': len(geometry),
        'positive_targets': int(np.count_nonzero(data['y_competition'] > 0)),
        'quality': quality_stats, 'raw_gt_reconstruction_equal': True,
        'max_independent_feature_error': error,
        'geometry_file': str(p['geometry']), 'points_file': str(p['points']),
        'gt_file': str(p['gt']), 'inputs': bound,
        'outputs': {name: sha(destination / name) for name in
                    ('features.npz', 'quality.npy', 'members.jsonl')},
        'prepared_at': now(),
    }
    write(receipt_path, receipt)
    return receipt


def load_g10(scene: str, evaluation: bool = False) -> dict:
    folder = OUT / 'dataset/G10' / scene
    receipt = read(folder / 'receipt.json')
    verify(receipt['inputs'])
    verify(files_at(folder, receipt['outputs']))
    with np.load(folder / 'features.npz', allow_pickle=False) as packed:
        data = {name: packed[name] for name in packed.files}
    x26, y, weights = validated_arrays(data)
    quality = np.load(folder / 'quality.npy', allow_pickle=False)
    x29 = np.column_stack((x26, quality))
    require(x29.shape == (receipt['members'], 29), 'G10 feature shape differs')
    result = {'x26': x26, 'x29': x29, 'y': y, 'weights': weights,
              'members_count': receipt['members']}
    if evaluation:
        result['members'] = rows(folder / 'members.jsonl')
        result['geometry'] = read(receipt['geometry_file'])
        result['points_file'] = Path(receipt['points_file'])
        result['gt_file'] = Path(receipt['gt_file'])
    return result


def prepare_dataset(identity: dict, folds: dict) -> dict:
    qrunner = load_q_runner()
    scenes = [scene for fold in folds['folds'] for scene in fold['holdout_scans']]
    receipts = {}
    for index, scene in enumerate(scenes, 1):
        write(OUT / 'execution.json', {'status': 'preparing_G10', 'current': scene,
              'completed': index - 1, 'total': len(scenes), 'updated_at': now()})
        receipt = prepare_g10_scene(scene, identity, qrunner)
        receipts[str(OUT / 'dataset/G10' / scene / 'receipt.json')] = sha(
            OUT / 'dataset/G10' / scene / 'receipt.json')
        print('PREPARED G10', index, len(scenes), scene, receipt['members'], flush=True)
        if index == 1:
            # The single-scene gate exercises GT reconstruction, 26D features,
            # Q extraction, targets, members, and all binding checks.
            write(OUT / 'prepare_smoke_gate.json', {
                'complete': True, 'scene': scene, 'receipt': receipt,
                'identity_sha256': digest(identity), 'passed_at': now(),
            })
    verify(receipts)
    summary = {'complete': True, 'scenes': len(scenes),
               'members': sum(read(Path(path))['members'] for path in receipts),
               'receipts': receipts, 'identity_sha256': digest(identity)}
    write(OUT / 'dataset/summary.json', summary)
    return summary


def fit_one(arm: str, fit_scans: list[str], fold_root: Path) -> tuple[object, dict]:
    width, domain = WIDTHS[arm], DOMAINS[arm]
    counts = []
    for scene in fit_scans:
        data = a_scene(scene) if domain == 'A' else load_g10(scene)
        counts.append(data['members'] if domain == 'A' else data['members_count'])
    total = sum(counts)
    x = np.empty((total, width), dtype=np.float64)
    y = np.empty(total, dtype=np.float64)
    weights = np.empty(total, dtype=np.float64)
    offset = 0
    for scene in fit_scans:
        data = a_scene(scene) if domain == 'A' else load_g10(scene)
        xx = data['x26'] if width == 26 else data['x29']
        end = offset + len(xx)
        x[offset:end], y[offset:end], weights[offset:end] = xx, data['y'], data['weights']
        offset = end
    require(offset == total, 'fit row assembly differs')
    weights *= len(weights) / weights.sum()
    started = time.monotonic()
    model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(x, y, sample_weight=weights)
    require(model.n_features_in_ == width and model.n_iter_ == 100, 'model fit incomplete')
    model_root = fold_root / 'models'
    model_root.mkdir(parents=True, exist_ok=True)
    path = model_root / f'{arm}.joblib'
    joblib.dump(model, path)
    saved = joblib.load(path)
    require(np.array_equal(model.predict(x), saved.predict(x)), 'persisted model replay differs')
    row = {'arm': arm, 'domain': domain, 'feature_count': width,
           'fit_scans': len(fit_scans), 'fit_main_scenes': len({s.split('_')[0] for s in fit_scans}),
           'fit_rows': total, 'seconds': time.monotonic() - started,
           'model_sha256': sha(path), 'model_path': str(path)}
    del x, y, weights
    return saved, row


def predictions_for_scene(scene: str, models: dict[str, object]) -> tuple[dict, dict]:
    data = load_g10(scene, evaluation=True)
    selected = {}
    for arm, model in models.items():
        matrix_x = data['x26'] if WIDTHS[arm] == 26 else data['x29']
        scores = np.clip(model.predict(matrix_x), 0, 1)
        selected[arm] = select(data['members'], scores)
        require(len(selected[arm]) <= 600, 'prediction budget exceeded')
    geometry = {row['geometry_key']: row for row in data['geometry']}
    count = len(np.loadtxt(data['gt_file'], dtype=np.int64))
    scene_matches = {}
    with np.load(data['points_file'], allow_pickle=False) as packed:
        for arm, chosen in selected.items():
            masks = np.zeros((count, len(chosen)), dtype=bool)
            for index, member in enumerate(chosen):
                masks[packed[geometry[member['geometry_key']]['array_key']], index] = True
            gt_instances, pred_instances = ev.assign_instances_for_scan({
                'pred_masks': masks,
                'pred_classes': np.asarray([row['class_index'] for row in chosen], dtype=np.int64),
                'pred_scores': np.asarray([row['score'] for row in chosen], dtype=np.float64),
            }, str(data['gt_file']))
            scene_matches[arm] = {'gt': gt_instances, 'pred': pred_instances}
    return selected, scene_matches


def fit_evaluate_fold(identity: dict, fold: dict) -> dict:
    index = fold['fold']
    fold_root = OUT / f'fold{index}'
    summary_path = fold_root / 'summary.json'
    if summary_path.exists():
        summary = read(summary_path)
        require(summary['identity_sha256'] == digest(identity), 'fold identity differs')
        return summary
    models, fit_rows = {}, []
    for arm in ARMS:
        write(OUT / 'execution.json', {'status': 'fitting', 'fold': index, 'arm': arm,
              'completed_folds': index, 'total_folds': 5, 'updated_at': now()})
        models[arm], row = fit_one(arm, fold['fit_scans'], fold_root)
        fit_rows.append(row)
        print('FIT', index, arm, row['fit_rows'], row['seconds'], flush=True)
    matches = {arm: {} for arm in ARMS}
    prediction_receipts = {}
    for scene in fold['holdout_scans']:
        selected, scene_matches = predictions_for_scene(scene, models)
        destination = fold_root / 'predictions' / scene
        destination.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for arm in ARMS:
            path = destination / f'{arm}.json'
            write(path, selected[arm])
            outputs[path.name] = sha(path)
            matches[arm][scene] = scene_matches[arm]
        receipt = {'scene': scene, 'fold': index, 'identity_sha256': digest(identity),
                   'outputs': outputs, 'heldout_from_all_four_fits': True}
        write(destination / 'receipt.json', receipt)
        prediction_receipts[str(destination / 'receipt.json')] = sha(destination / 'receipt.json')
        print('EVALUATED FOLD SCENE', index, scene, flush=True)
    metrics = {}
    match_root = fold_root / 'matches'
    match_root.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        with (match_root / f'{arm}.pkl').open('wb') as stream:
            pickle.dump(matches[arm], stream)
        array, *_ = ev.evaluate_matches(matches[arm])
        np.save(fold_root / f'{arm}_AP.npy', array, allow_pickle=False)
        metrics[arm] = compact_metrics(array)
    summary = {
        'complete': True, 'fold': index, 'identity_sha256': digest(identity),
        'fit_scans': fold['fit_scans'], 'holdout_scans': fold['holdout_scans'],
        'fit_rows': fit_rows, 'AP': metrics,
        'prediction_receipts': prediction_receipts,
        'decision_use': 'correctness gate for fold0; final decision uses pooled 30-scene OOF AP',
    }
    write(summary_path, summary)
    return summary


def pooled_oof(identity: dict, folds: dict, fold_summaries: list[dict]) -> dict:
    configure_evaluator()
    pooled = {arm: {} for arm in ARMS}
    seen = set()
    for fold, summary in zip(folds['folds'], fold_summaries):
        require(summary['complete'] and summary['fold'] == fold['fold'], 'fold summary differs')
        for arm in ARMS:
            with (OUT / f"fold{fold['fold']}/matches/{arm}.pkl").open('rb') as stream:
                group = pickle.load(stream)
            require(set(group) == set(fold['holdout_scans']), 'fold match coverage differs')
            pooled[arm].update(group)
        require(seen.isdisjoint(fold['holdout_scans']), 'OOF scene duplicated')
        seen.update(fold['holdout_scans'])
    require(len(seen) == 30, 'pooled OOF coverage differs')
    metrics, outputs = {}, {}
    evaluation = OUT / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    for arm in ARMS:
        require(set(pooled[arm]) == seen, 'pooled arm coverage differs')
        array, *_ = ev.evaluate_matches(pooled[arm])
        metrics[arm] = compact_metrics(array)
        path = evaluation / f'{arm}_OOF_AP.npy'
        np.save(path, array, allow_pickle=False)
        outputs[str(path)] = sha(path)
    comparisons = {}
    for matched, control in (('G10_B_matched', 'A_B_control'),
                             ('G10_Q_matched', 'A_Q_control'),
                             ('G10_Q_matched', 'G10_B_matched'),
                             ('A_Q_control', 'A_B_control')):
        comparisons[f'{matched}_minus_{control}'] = {
            key: 100 * (metrics[matched][key] - metrics[control][key])
            for key in ('ap', 'ap50', 'ap25')
        }
    result = {
        'complete': True, 'identity_sha256': digest(identity),
        'OOF_scans': 30, 'OOF_main_scenes': 30, 'folds': 5,
        'AP_units': 'fraction', 'AP': metrics, 'delta_pp': comparisons,
        'primary_metric': 'pooled 30-scene OOF category-aware AP',
        'full597_heads_excluded_from_decision': True,
        'full597_exclusion_reason': identity['full597_exclusion_reason'],
        'outputs': outputs,
        'fold_summaries': {str(OUT / f"fold{row['fold']}/summary.json"):
                           sha(OUT / f"fold{row['fold']}/summary.json") for row in fold_summaries},
    }
    write(OUT / 'summary.json', result)
    return result


def run() -> None:
    identity, folds = initialize()
    verify(identity['inputs'])
    verify(identity['code'])
    dataset = prepare_dataset(identity, folds)
    require(dataset['complete'] and dataset['scenes'] == 30, 'G10 dataset incomplete')
    fold_summaries = []
    configure_evaluator()
    for fold in folds['folds']:
        summary = fit_evaluate_fold(identity, fold)
        fold_summaries.append(summary)
        if fold['fold'] == 0:
            require(summary['complete'] and set(summary['AP']) == set(ARMS),
                    'fold0 smoke gate failed')
            require(all(np.isfinite(value) for metrics in summary['AP'].values()
                        for value in metrics.values()), 'fold0 AP is not finite')
            write(OUT / 'fold0_gate.json', {'complete': True, 'fold_summary_sha256':
                  sha(OUT / 'fold0/summary.json'), 'gate': 'correctness only; no AP tuning',
                  'passed_at': now()})
        write(OUT / 'execution.json', {'status': 'fold_complete', 'completed_folds': fold['fold'] + 1,
              'total_folds': 5, 'updated_at': now()})
    result = pooled_oof(identity, folds, fold_summaries)
    verify(identity['inputs'])
    verify(identity['code'])
    write(OUT / 'execution.json', {'status': 'complete', 'completed_folds': 5,
          'total_folds': 5, 'finished_at': now(), 'AP': result['AP'],
          'delta_pp': result['delta_pp']})
    print(json.dumps(result['AP'], ensure_ascii=False, indent=2), flush=True)


def check() -> None:
    identity, folds = initialize()
    first = folds['folds'][0]
    print(json.dumps({'complete': True, 'identity_sha256': digest(identity),
                      'folds': [(f['fold'], len(f['fit_scans']), len(f['holdout_scans']))
                                for f in folds['folds']],
                      'fold0_fit': first['fit_scans'], 'fold0_holdout': first['holdout_scans']},
                     ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run'))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            {'check': check, 'run': run}[args.action]()
        except BaseException as error:
            write(OUT / 'failure.json', {'error': repr(error), 'time': now(), 'pid': os.getpid()})
            write(OUT / 'execution.json', {'status': 'failed', 'error': repr(error),
                  'updated_at': now()})
            raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build GT-free scene partitions and evaluate actions in one partition only.

This isolates the downstream label reference. Historical supervised FI1/B1
assets remain frozen; this is not a claim of whole-system out-of-fold fitting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fi1_shared.build_ncs1_scene_folds import build_folds
from tools.fi1_evaluator_delta_core import (
    ACTIONS, IDENTITY, CounterfactualEvaluator, configure_evaluator, finite,
    identity_equal, indexed_b1, integer, intervention, load_population, validate_prediction,
)
from tools.fi1_gain_attribution_core import Bindings, points_from_population, read_rows, sha, unique, write_json, write_rows

SEED = 'fi1-margin-main-scene-v1'
FOLDS = 5
SCENE_RE = re.compile(r'^(scene\d{4})_\d{2}$')
REFERENCE_VERSION = 'fi1_scene_reference_v1'
LABEL_VERSION = 'fi1_scene_counterfactual_v1'
ROLES = ('fit', 'heldout')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main_scene(scene):
    match = SCENE_RE.fullmatch(scene)
    if not match:
        raise ValueError(f'invalid scene name: {scene}')
    return match.group(1)


def make_split(scenes):
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError('empty or duplicate scene list')
    if len({main_scene(s) for s in scenes}) < FOLDS:
        raise ValueError('at least five main-scene groups required')
    groups = build_folds(sorted(scenes), FOLDS, SEED)
    return {'seed': SEED, 'fold_count': FOLDS, 'scenes': sorted(scenes),
            'heldout_folds': groups, 'assignment_inputs': 'scene names only; no GT or class statistics'}


def partition_scenes(split, fold, role):
    integer(fold, 'outer_fold_index', maximum=FOLDS - 1)
    if role not in ROLES:
        raise ValueError('unknown partition role')
    if split != make_split(split['scenes']):
        raise ValueError('split differs from deterministic main-scene grouping')
    heldout = set(split['heldout_folds'][fold])
    fit = set(split['scenes']) - heldout
    if {main_scene(s) for s in fit} & {main_scene(s) for s in heldout}:
        raise ValueError('main scene crosses fit/heldout boundary')
    return sorted(fit if role == 'fit' else heldout)


def reference_payload(ref):
    # Opposite-partition GT/prediction hashes never enter this identity.
    return {k: ref[k] for k in ('version', 'split_id', 'outer_fold_index', 'role',
                                'reference_scenes', 'class_names', 'predictions_sha256', 'geometry_index_sha256', 'cache_sha256')}


def validate_reference(ref):
    if ref.get('version') != REFERENCE_VERSION:
        raise ValueError('scene reference v1 required')
    split = ref['split']
    if ref['split_id'] != digest(split):
        raise ValueError('split_id differs from partition manifest')
    scenes = partition_scenes(split, ref['outer_fold_index'], ref['role'])
    if ref['reference_scenes'] != scenes or set(ref['cache_sha256']) != set(scenes):
        raise ValueError('reference scenes/cache scope differs from partition')
    if ref['reference_id'] != digest(reference_payload(ref)):
        raise ValueError('reference content identity differs')
    return scenes


def validate_partition_requests(requests, predictions, geometries, ref):
    scenes = set(validate_reference(ref))
    unique(requests, 'prediction_id')
    known = unique(predictions, 'prediction_id')
    pairs = set()
    for req in requests:
        validate_prediction(req, foreground=True)
        if req['scene_name'] not in scenes:
            raise ValueError('request outside partition; cross-role labels are forbidden')
        if req.get('action') not in ACTIONS:
            raise ValueError('explicit supported action required')
        geo = geometries[req['geometry_key']]
        if geo['scene_name'] != req['scene_name']:
            raise ValueError('request/geometry scene differs')
        integer(req['fold_index'], 'legacy fold_index', maximum=4)
        if req['fold_index'] != geo['legacy_fold_index']:
            raise ValueError('legacy source fold differs; do not overwrite it with the new split fold')
        if req['action'] == 'append_new':
            pair = (req['geometry_key'], req['class_index'])
            if pair in pairs:
                raise ValueError('duplicate new geometry-class action')
            pairs.add(pair)
            if (req.get('is_new_geometry') is not True or geo['is_new_geometry'] is not True or
                    geo['has_b1_members'] or req['source'] != geo['sources']):
                raise ValueError('new geometry/source contract differs')
        else:
            old = known.get(req['prediction_id'])
            if old is None or not identity_equal(old, req) or req['source'] != old['source']:
                raise ValueError('existing member/source differs')
        intervention(predictions, req, req['action'])


def load_partition(root, bindings, expected_role=None):
    """Read only this partition's predictions, geometry metadata and GT caches."""
    root = Path(root)
    ref = json.loads(bindings.check(root / 'reference.json').read_text())
    scenes = validate_reference(ref)  # reject wrong scope BEFORE opening any cache
    if expected_role is not None and ref['role'] != expected_role:
        raise ValueError('explicit role differs from reference; heldout is evaluation-only')
    predictions = read_rows(bindings.check(root / 'predictions.jsonl', ref['predictions_sha256']))
    unique(predictions, 'prediction_id')
    by_scene = defaultdict(list)
    for row in predictions:
        validate_prediction(row)
        if row['scene_name'] not in scenes:
            raise ValueError('prediction outside partition')
        by_scene[row['scene_name']].append(integer(row['column_index'], 'column_index'))
    for columns in by_scene.values():
        if columns != list(range(len(columns))):
            raise ValueError('partition changed B1 column order')
    geometries = unique(read_rows(bindings.check(root / 'geometry_index.jsonl', ref['geometry_index_sha256'])), 'geometry_key')
    new_fold = {s: i for i, group in enumerate(ref['split']['heldout_folds']) for s in group}
    for key, geo in geometries.items():
        if geo['scene_name'] not in scenes or geo['split_fold_index'] != new_fold[geo['scene_name']]:
            raise ValueError('geometry outside partition or split fold differs')
        integer(geo['legacy_fold_index'], 'legacy_fold_index', maximum=4)
        if key != f"{geo['scene_name']}:geometry:{geo['points_sha256']}":
            raise ValueError('geometry key/point hash differs')
    caches = {}
    for scene in scenes:
        path = Path(ref['attribution_root']) / 'scene_cache' / f'{scene}.json'
        cache = json.loads(bindings.check(path, ref['cache_sha256'][scene]).read_text())
        if cache['scene_name'] != scene:
            raise ValueError('cache scene identity differs')
        caches[scene] = cache
    for row in predictions:
        geo = geometries[row['geometry_key']]
        if geo['scene_name'] != row['scene_name'] or not geo['has_b1_members']:
            raise ValueError('B1 geometry identity differs')
    return ref, predictions, geometries, caches


def _load_frozen_metadata(args, bind):
    # No GT cache or AP array is opened while assigning folds/materializing references.
    root = Path(args.attribution_root)
    audit = json.loads(bind.check(args.attribution_audit).read_text())
    if audit.get('audit_valid') is not True or audit.get('all_seven_ap_arrays_exactly_replayed') is not True:
        raise ValueError('accepted historical attribution audit required')
    summary = json.loads(bind.check(root / 'summary.json', audit['summary_sha256']).read_text())
    if summary['ground_truth_scope'] != 'official100 only' or summary['scene_count'] != 100:
        raise ValueError('official100 metadata required')
    scenes = sorted(Path(k).stem for k in summary['hashes'] if k.startswith('scene_cache/') and k.endswith('.json'))
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise ValueError('exact100 cache metadata required')
    predictions = indexed_b1(read_rows(bind.check(args.b1_predictions, summary['hashes']['b1_frozen_predictions.jsonl'])))
    pop, pop_summary = load_population(args.population_root, bind)
    if {r['scene_name'] for r in pop.values()} != set(scenes) or {r['scene_name'] for r in predictions} - set(scenes):
        raise ValueError('population/B1 scene universe differs')
    return summary, scenes, predictions, pop, pop_summary


def prepare(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_state.json', {'status': 'running'})
    try:
        bind = Bindings()
        summary, scenes, predictions, pop, pop_summary = _load_frozen_metadata(args, bind)
        split = make_split(scenes)
        legacy = {}
        for row in pop.values():
            previous = legacy.setdefault(row['scene_name'], row['fold_index'])
            if previous != row['fold_index']:
                raise ValueError('inconsistent legacy scene folds')
        old_groups = defaultdict(dict)
        for scene, fold in legacy.items():
            old_groups[main_scene(scene)][scene] = fold
        crossings = {g: members for g, members in old_groups.items() if len(set(members.values())) > 1}
        new_fold = {s: i for i, members in enumerate(split['heldout_folds']) for s in members}
        b1keys = {r['geometry_key'] for r in predictions}
        index = [{**{k: row[k] for k in ('geometry_key', 'scene_name', 'points_file', 'points_file_sha256',
                   'points_sha256', 'point_count', 'sources', 'is_new_geometry')},
                  'has_b1_members': row['geometry_key'] in b1keys,
                  'legacy_fold_index': row['fold_index'], 'split_fold_index': new_fold[row['scene_name']]}
                 for row in pop.values()]
        actions = []
        for action, path in [('restore_existing', args.existing_requests), ('append_new', args.new_candidates)]:
            if path is not None:
                fields = (*IDENTITY, 'score', 'source', 'fold_index')
                for row in read_rows(bind.check(path)):
                    item = {k: row[k] for k in fields}
                    item['action'] = action
                    if action == 'append_new':
                        item.update(is_new_geometry=row['is_new_geometry'], candidate_score_source=row['candidate_score_source'])
                    actions.append(item)
        unique(actions, 'prediction_id')
        if {r['scene_name'] for r in actions} - set(scenes):
            raise ValueError('action outside official100')
        refs = []
        for fold in range(FOLDS):
            for role in ROLES:
                allowed = partition_scenes(split, fold, role)
                folder = out / f'fold_{fold}' / role
                folder.mkdir(parents=True)
                members = [r for r in predictions if r['scene_name'] in allowed]
                geos = [r for r in index if r['scene_name'] in allowed]
                selected = [r for r in actions if r['scene_name'] in allowed]
                write_rows(folder / 'predictions.jsonl', members)
                write_rows(folder / 'geometry_index.jsonl', geos)
                ref = {'version': REFERENCE_VERSION, 'split': split, 'split_id': digest(split),
                       'outer_fold_index': fold, 'role': role, 'reference_scenes': allowed,
                       'predictions_sha256': sha(folder / 'predictions.jsonl'),
                       'geometry_index_sha256': sha(folder / 'geometry_index.jsonl'),
                       'cache_sha256': {s: summary['hashes'][f'scene_cache/{s}.json'] for s in allowed},
                       'attribution_root': str(Path(args.attribution_root).resolve()),
                       'population_root': str(Path(args.population_root).resolve()),
                       'class_names': pop_summary['class_names'],
                       'historical_supervised_upstream_reused': True, 'whole_system_oof': False}
                ref['reference_id'] = digest(reference_payload(ref))
                validate_partition_requests(selected, members, unique(geos, 'geometry_key'), ref)
                write_json(folder / 'reference.json', ref)
                write_rows(folder / 'requests.jsonl', selected)
                refs.append({'outer_fold_index': fold, 'role': role, 'root': str(folder.relative_to(out)),
                             'scene_count': len(allowed), 'prediction_count': len(members), 'request_count': len(selected),
                             'reference_id': ref['reference_id']})
        write_json(out / 'split.json', split)
        write_json(out / 'summary.json', {'version': REFERENCE_VERSION, 'split_id': digest(split),
            'references': refs, 'legacy_crossfold_main_scenes': crossings,
            'gt_cache_payloads_read': False, 'ap_arrays_read': False,
            'fold_assignment_uses_gt_or_class_statistics': False, 'input_sha256': bind.files,
            'code_sha256': {name: sha(ROOT / name) for name in ('tools/fi1_scene_reference.py',
                'tools/build_ncs1_scene_folds.py', 'tools/fi1_evaluator_delta_core.py', 'tools/fi1_gain_attribution_core.py')},
            'hashes': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_state.json'},
            'training_ready': False, 'model_fitted': False, 'production_ready': False,
            'limitations': ['partition of downstream reference only; historical FI1/B1 is frozen',
                            'candidate sampling and complete feature consumption remain pending']})
        write_json(out / 'run_state.json', {'status': 'complete', 'summary_sha256': sha(out / 'summary.json')})
        print(json.dumps({'fold_sizes': [len(x) for x in split['heldout_folds']],
                          'legacy_crossfold_main_scene_count': len(crossings), 'reference_count': len(refs)}, indent=2))
    except Exception as exc:
        write_json(out / 'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


def evaluate(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'run_state.json', {'status': 'running'})
    try:
        bind = Bindings()
        ref, predictions, geos, caches = load_partition(args.reference_root, bind, expected_role=args.role)
        requests = read_rows(bind.check(args.requests))
        validate_partition_requests(requests, predictions, geos, ref)
        checked = set()
        for row in requests:
            key = row['geometry_key']; geo = geos[key]; cache = caches[row['scene_name']]
            if cache['points_sha256'][key] != geo['points_sha256'] or cache['geometry_stats'][key]['point_count'] != geo['point_count']:
                raise ValueError('partition geometry/cache identity differs')
            if key not in checked:
                points_from_population(Path(ref['population_root']), geo, bind)
                checked.add(key)
        from evaluate.scannet200 import eval_semantic_instance as ev
        configure_evaluator(ev)
        if list(ev.CLASS_LABELS) != ref['class_names']:
            raise ValueError('reference vocabulary differs from evaluator')
        engine = CounterfactualEvaluator(predictions, caches, ev)
        arrays = out / 'ap_arrays'; arrays.mkdir()
        np.save(out / 'baseline_ap.npy', engine.baseline_array, allow_pickle=False)
        rows = []
        for i, req in enumerate(requests):
            row, before, after = engine.evaluate(req, req['action'])
            row.update(version=LABEL_VERSION, evaluation_scope='partition_reference_scenes',
                       reference_id=ref['reference_id'], split_id=ref['split_id'],
                       outer_fold_index=ref['outer_fold_index'], partition_role=ref['role'],
                       label_usage='fit_only' if ref['role'] == 'fit' else 'evaluation_only',
                       array_file=f'ap_arrays/{i:04d}.npz')
            np.savez(out / row['array_file'], before=before, after=after)
            rows.append(row)
            print(f"fold {ref['outer_fold_index']} {ref['role']}: {i+1}/{len(requests)} actions, {len(caches)} scenes", flush=True)
        write_rows(out / 'actions.jsonl', requests)
        write_rows(out / 'evaluator_delta.jsonl', rows)
        write_json(out / 'reference.json', ref)
        names = ['tools/fi1_scene_reference.py', 'tools/build_ncs1_scene_folds.py', 'tools/fi1_evaluator_delta_core.py',
                 'tools/fi1_gain_attribution_core.py', 'evaluate/scannet200/eval_semantic_instance.py',
                 'evaluate/scannet200/util_3d.py', 'evaluate/scannet200/scannet_constants.py']
        write_json(out / 'summary.json', {'version': LABEL_VERSION, 'reference_id': ref['reference_id'],
            'split_id': ref['split_id'], 'outer_fold_index': ref['outer_fold_index'], 'partition_role': ref['role'],
            'reference_scenes': ref['reference_scenes'], 'reference_prediction_count': len(predictions),
            'evaluation_scope': 'partition_reference_scenes', 'candidate_count': len(rows),
            'label_usage': 'fit_only' if ref['role'] == 'fit' else 'evaluation_only',
            'thresholds': list(map(float, ev.opt['overlaps'])), 'input_sha256': bind.files,
            'hashes': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_state.json'},
            'code_sha256': {n: sha(ROOT / n) for n in names},
            'label_reference_isolated': True, 'whole_system_oof': False, 'training_ready': False,
            'model_fitted': False, 'production_ready': False,
            'limitations': ['no fitting or hyperparameter selection implemented',
                'historical supervised FI1/B1 frozen; no end-to-end OOF claim',
                'raw official100 GT unavailable; matching uses historically audited caches',
                'class absent from this whole partition has undefined AP; retain null targets',
                'for inner calibration, build new fit/calibration references; never reuse outer-fit targets']})
        write_json(out / 'run_state.json', {'status': 'complete', 'summary_sha256': sha(out / 'summary.json')})
    except Exception as exc:
        write_json(out / 'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('prepare', help='Build five grouped partitions without opening GT caches.')
    for name in ('b1-predictions', 'attribution-root', 'attribution-audit', 'population-root', 'output-root'):
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--existing-requests', type=Path)
    b.add_argument('--new-candidates', type=Path)
    e = sub.add_parser('evaluate', help='Read/evaluate only one explicit partition.')
    for name in ('reference-root', 'requests', 'output-root'):
        e.add_argument('--' + name, type=Path, required=True)
    e.add_argument('--role', choices=ROLES, required=True)
    args = p.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze R0/R1 scene roles and inspect existing inputs without running models.

This creates new preparation artifacts only. Historical plans are inventoried,
never relabelled as R0/R1 outputs or as a shared deployment model run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.fi1_shared.build_ncs1_scene_folds import build_folds

VERSION = 'fi1_r0_r1_preparation_v1'
SEED = 'fi1-r0-r4-train200-main-scene-20260913-v1'
ROLES = ('train', 'development_validation', 'diagnostic')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2,
                                   ensure_ascii=False, allow_nan=False) + '\n')


def main_scene(scene):
    if re.fullmatch(r'scene\d{4}_\d{2}', scene) is None:
        raise ValueError(f'invalid scan identity: {scene}')
    return scene.split('_')[0]


def read_scenes(path):
    rows = Path(path).read_text().split()
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f'empty or duplicate scan list: {path}')
    for scene in rows:
        main_scene(scene)
    return sorted(rows)


def make_partition(official, ncs_train, ncs_validation):
    sources = {'official100': official, 'ncs_train100': ncs_train,
               'ncs_validation60': ncs_validation}
    for name, rows in sources.items():
        if not rows or len(rows) != len(set(rows)):
            raise ValueError(f'empty or duplicate source scans: {name}')
        for scene in rows:
            main_scene(scene)
    if (set(official) & set(ncs_train) or
            (set(official) | set(ncs_train)) & set(ncs_validation)):
        raise ValueError('source scan lists overlap; do not silently deduplicate')
    train = sorted(official + ncs_train)
    train_mains = {main_scene(s) for s in train}
    roles = {'train': train,
             'development_validation': sorted(s for s in ncs_validation if main_scene(s) not in train_mains),
             'diagnostic': sorted(s for s in ncs_validation if main_scene(s) in train_mains)}
    if not roles['development_validation'] or len(train_mains) < 5:
        raise ValueError('insufficient independent groups')
    folds = build_folds(train, 5, SEED)
    if any(not f for f in folds):
        raise ValueError('empty training fold')
    return {'version': VERSION, 'roles': roles, 'seed': SEED,
            'training_heldout_folds': folds,
            'counts': {k: {'scans': len(v), 'main_scenes': len({main_scene(s) for s in v})}
                       for k, v in roles.items()},
            'source_scans': {k: sorted(v) for k, v in sources.items()},
            'assignment_uses': 'scan names only; no GT, class frequencies, or AP',
            'validation_interpretation': 'historically used development validation, not a blind test',
            'whole_system_oof': False,
            'upstream_nested_exclusion_verified': False}


def validate_partition(partition):
    source = partition['source_scans']
    expected = make_partition(source['official100'], source['ncs_train100'], source['ncs_validation60'])
    if partition != expected:
        raise ValueError('partition differs from deterministic source reconstruction')


def freeze(args):
    paths = {'official100': args.official, 'ncs_train100': args.ncs_train,
             'ncs_validation60': args.ncs_validation}
    lists = {k: read_scenes(p) for k, p in paths.items()}
    partition = make_partition(lists['official100'], lists['ncs_train100'], lists['ncs_validation60'])
    if partition['counts'] != {
        'train': {'scans': 200, 'main_scenes': 139},
        'development_validation': {'scans': 55, 'main_scenes': 29},
        'diagnostic': {'scans': 5, 'main_scenes': 4},
    }:
        raise ValueError('actual split differs from the user-approved 200/55/5 split')
    validate_partition(partition)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'partition.json', partition)
    for role, scenes in partition['roles'].items():
        (args.output / f'{role}.txt').write_text(''.join(s + '\n' for s in scenes))
    manifest = {'version': VERSION, 'status': 'user_authorized_frozen_split',
                'authorization': 'User confirmed 200 training / 55 development validation / 5 diagnostic and instructed updating state files then starting.',
                'partition_id': digest(partition),
                'source_identity': {k: {'path': str(p.resolve()), 'sha256': sha(p)} for k, p in paths.items()},
                'files': {p.name: sha(p) for p in sorted(args.output.iterdir())},
                'code': {str(Path(__file__).resolve()): sha(__file__),
                         str(ROOT / 'tools/build_ncs1_scene_folds.py'): sha(ROOT / 'tools/build_ncs1_scene_folds.py')},
                'new_training': False, 'new_gpu_inference': False, 'new_ap': False}
    write(args.output / 'manifest.json', manifest)
    return {'output': str(args.output.resolve()), 'counts': partition['counts'],
            'training_fold_scan_counts': [len(f) for f in partition['training_heldout_folds']]}


def load_partition(root):
    manifest = read(root / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha(root / name) != expected:
            raise ValueError(f'frozen split changed: {name}')
    partition = read(root / 'partition.json')
    validate_partition(partition)
    if digest(partition) != manifest['partition_id']:
        raise ValueError('partition identity differs')
    return partition


def file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'exists': path.is_file(),
            'sha256': sha(path) if path.is_file() else None,
            'bytes': path.stat().st_size if path.is_file() else None}


def ledger_counts(path):
    counts = Counter()
    member_counts = Counter()
    locators = {}
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            scene = row['scene_name']
            main_scene(scene)
            counts[scene] += 1
            member_counts[scene] += row.get('member_count', len(row.get('members', [])))
            for locator in [row.get('canonical_geometry_locator', row.get('geometry_locator_read_only', {}))]:
                for key in ('masks_path', 'points_path'):
                    if key in locator:
                        p = str(Path(locator[key]).resolve())
                        locators[p] = Path(p).is_file()
    return dict(counts), dict(member_counts), locators


def inspect_model(root):
    root = Path(root)
    summary = read(root / 'summary.json')
    models = []
    for record in summary['models']:
        identity = file_identity(root / record['model_file'])
        identity['expected_sha256'] = record['model_sha256']
        identity['hash_matches'] = identity['sha256'] == record['model_sha256']
        identity['model_kind'] = record['model_kind']
        models.append(identity)
    return {'summary': file_identity(root / 'summary.json'), 'models': models,
            'valid_seven_model_package': len(models) == 7 and all(m['hash_matches'] for m in models)
            and summary.get('training_dataset') == 'official100',
            'training_dataset_declared': summary.get('training_dataset'),
            'ground_truth_usage_declared': summary.get('ground_truth_usage'),
            'historical_input_hashes': summary.get('input_hashes', {}),
            'training_scene_binding_status': 'declared official100; transitive historical training dependency audit pending',
            'whole_system_oof': False}


def inventory(args):
    import numpy as np
    partition = load_partition(args.split)
    config = read(args.sources)
    if set(config['sources']) != set(partition['source_scans']):
        raise ValueError('source identity names differ')
    role_of = {scene: role for role, scenes in partition['roles'].items() for scene in scenes}
    source_reports, scene_rows = {}, []
    for name, spec in config['sources'].items():
        ledger_path = Path(spec['legacy_geometry_root']) / 'unique_geometry_ledger.jsonl'
        counts, members, locators = ledger_counts(ledger_path)
        if set(counts) != set(partition['source_scans'][name]):
            raise ValueError(f'{name}: ledger scan coverage differs from frozen source')
        summaries = {key: file_identity(Path(spec[key]) / 'summary.json') for key in
                     ('legacy_geometry_root', 'gvc_root', 'relation_root', 'champion_plan_root')}
        source_reports[name] = {'legacy_geometry_ledger': file_identity(ledger_path),
                                'summaries': summaries,
                                'geometry_locator_file_count': len(locators),
                                'missing_geometry_locator_files': sorted(p for p, exists in locators.items() if not exists),
                                'legacy_geometry_count': sum(counts.values()),
                                'legacy_member_count': sum(members.values()),
                                'historical_dv3_plan': None}
        if spec.get('historical_dv3_plan'):
            plan = Path(spec['historical_dv3_plan'])
            pc, _, pl = ledger_counts(plan)
            source_reports[name]['historical_dv3_plan'] = {
                'identity': file_identity(plan), 'scene_count': len(pc),
                'candidate_count': sum(pc.values()),
                'source_scan_coverage_exact': set(pc) == set(counts),
                'missing_geometry_locator_files': sorted(p for p, exists in pl.items() if not exists),
                'shared_deployment_model_run': False}
        for scene in partition['source_scans'][name]:
            prepared = Path(spec['prepared_root']) / scene
            records = Path(spec['records_root']) / scene
            files = {
                'processed': prepared / f'{scene[5:]}.npy',
                'ply': prepared / f'{scene}_vh_clean_2.ply',
                'intrinsics': prepared / 'intrinsics.txt',
                'prepare_manifest': prepared / 'stream_prepare_manifest.json',
                'native_masks': records / 'native_cache' / f'{scene}_pred_masks.npy',
                'native_classes': records / 'native_cache' / f'{scene}_pred_classes.npy',
                'native_scores': records / 'native_cache' / f'{scene}_pred_scores.npy',
                'tracks': Path(spec['track_root']) / scene / 'automatic_tracks.json',
                'observations': Path(spec['automatic_root']) / scene / 'automatic_observations.jsonl',
                'relations': Path(spec['relation_root']) / scene / 'relation_features_no_gt.jsonl',
            }
            present = {key: p.is_file() for key, p in files.items()}
            shape, class_count = None, None
            issues = [f'missing_{k}' for k, exists in present.items() if not exists]
            if all(present[k] for k in ('processed', 'native_masks', 'native_classes', 'native_scores')):
                # Only headers for processed XYZ/superpoints/masks; no ground-truth columns are consumed.
                processed = np.load(files['processed'], mmap_mode='r', allow_pickle=False)
                masks = np.load(files['native_masks'], mmap_mode='r', allow_pickle=False)
                classes = np.load(files['native_classes'], mmap_mode='r', allow_pickle=False)
                scores = np.load(files['native_scores'], mmap_mode='r', allow_pickle=False)
                shape, class_count = list(masks.shape), len(classes)
                if (processed.ndim != 2 or masks.ndim != 2 or processed.shape[0] != masks.shape[0]
                        or classes.shape != scores.shape or classes.ndim != 1 or masks.shape[1] != len(classes)):
                    issues.append('point_or_prediction_array_shape_mismatch')
                if not np.isfinite(scores).all() or not np.isin(classes, range(199)).all():
                    issues.append('invalid_native_class_or_score')
            scene_rows.append({'scene_name': scene, 'main_scene': main_scene(scene),
                               'role': role_of[scene], 'source_partition': name,
                               'legacy_geometry_count': counts[scene], 'legacy_member_count': members[scene],
                               'native_mask_shape': shape, 'native_prediction_count': class_count,
                               'files': {k: {'path': str(v.resolve()), 'exists': present[k]} for k, v in files.items()},
                               'frame_directory_counts': {k: len(list((prepared / k).iterdir())) if (prepared / k).is_dir() else 0
                                                          for k in ('color', 'depth', 'poses')},
                               'bbox_cache_directory': str((records / 'yoloworld_bboxes_2d').resolve()),
                               'bbox_cache_exists': (records / 'yoloworld_bboxes_2d').is_dir(),
                               'issues': issues,
                               'new_shared_dv3_geometry_ready': False,
                               'geometry_point_hashes_verified': False,
                               'new_own_semantic_evidence_ready': False})
    model = inspect_model(config['dv3_model_root'])
    required = {s: len([r for r in scene_rows if r['source_partition'] == s and not r['issues']])
                for s in config['sources']}
    summary = {'version': VERSION, 'partition_id': digest(partition),
               'sources_identity': file_identity(args.sources), 'source_reports': source_reports,
               'dv3_model': model, 'scene_count': len(scene_rows),
               'basic_array_and_required_file_checks_passed': required,
               'scene_issue_count': sum(bool(r['issues']) for r in scene_rows),
               'new_shared_dv3_geometry_ready': False, 'r0_r1_ap_ready': False,
               'new_training': False, 'new_gpu_inference': False, 'new_ap': False,
               'ground_truth_usage': 'none; prepared NPY metadata only, no label columns',
               'limitations': ['File presence and array shape are not point-order or content identity verification.',
                               'Historical official100/NCS OOF geometry is not one frozen full-model deployment run.',
                               'No new geometry or own semantic evidence was generated by this inventory.',
                               'Full upstream supervision dependency audit remains pending.'],
               'code_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'scene_inventory.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n'
                                                             for r in sorted(scene_rows, key=lambda r: r['scene_name'])))
    summary['scene_inventory_sha256'] = sha(args.output / 'scene_inventory.jsonl')
    write(args.output / 'summary.json', summary)
    return {k: summary[k] for k in ('scene_count', 'basic_array_and_required_file_checks_passed',
                                   'scene_issue_count', 'r0_r1_ap_ready')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--official', type=Path, required=True)
    freeze_parser.add_argument('--ncs-train', type=Path, required=True)
    freeze_parser.add_argument('--ncs-validation', type=Path, required=True)
    freeze_parser.add_argument('--output', type=Path, required=True)
    inventory_parser = sub.add_parser('inventory')
    inventory_parser.add_argument('--split', type=Path, required=True)
    inventory_parser.add_argument('--sources', type=Path, required=True)
    inventory_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args) if args.command == 'freeze' else inventory(args), ensure_ascii=False))


if __name__ == '__main__':
    main()

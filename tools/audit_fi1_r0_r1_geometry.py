#!/usr/bin/env python3
"""Bind actual D-v3 point sets and every legacy semantic alias, without GT/AP.

Each scene is independently resumable through hashed receipts. Input files are
read-only; output masks are unique sorted point sets, never dense class copies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_fi1_r0_r1_inputs import digest, load_partition, read, sha, write

VERSION = 'fi1_r0_r1_geometry_binding_v1'


def point_digest(points, algorithm='sha256'):
    return hashlib.new(algorithm, np.asarray(points, dtype='<i8').tobytes()).hexdigest()


def checked_points(raw, domain):
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu':
        raise ValueError('point indices must be a one-dimensional integer array')
    if raw.size and (raw.min() < 0 or raw.max() >= domain):
        raise ValueError('point indices outside scene domain')
    return np.unique(raw.astype('<i8'))


class Resolver:
    def __init__(self, domain):
        self.domain = domain
        self.files, self.arrays, self.points = {}, {}, {}

    def bind(self, path):
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'missing or nonabsolute input: {path}')
        name = str(path)
        if name not in self.files:
            self.files[name] = sha(path)
        return name

    def array(self, path):
        name = self.bind(path)
        if name not in self.arrays:
            self.arrays[name] = np.load(name, mmap_mode='r', allow_pickle=False)
        return self.arrays[name]

    def resolve(self, locator):
        key = digest(locator)
        if key in self.points:
            return self.points[key]
        if locator['kind'] == 'native_mask_column':
            masks = self.array(locator['masks_path'])
            column = locator['column_index']
            if (masks.ndim != 2 or masks.shape[0] != self.domain or type(column) is not int
                    or not 0 <= column < masks.shape[1]):
                raise ValueError('native mask shape/column differs')
            mask = masks[:, column]
            if not np.isin(mask, [0, 1]).all():
                raise ValueError('nonbinary native mask')
            points = np.flatnonzero(mask).astype('<i8')
        elif locator['kind'] == 'point_indices_npz':
            name = self.bind(locator['points_path'])
            with np.load(name, allow_pickle=False) as data:
                points = checked_points(data[locator.get('array_key', 'point_indices')], self.domain)
        else:
            raise ValueError('unknown geometry locator kind')
        self.points[key] = points
        return points


def check_claim(points, claimed, count, algorithm):
    if len(points) != count or point_digest(points, algorithm) != claimed:
        raise ValueError('actual point count/hash differs from declared geometry')


def audit_scene(scene, spec, legacy, plans):
    from plyfile import PlyData

    prepared = Path(spec['prepared_root']) / scene
    processed_path = prepared / f'{scene[5:]}.npy'
    processed = np.load(processed_path, mmap_mode='r', allow_pickle=False)
    if processed.ndim != 2 or processed.shape[1] < 3:
        raise ValueError('invalid prepared XYZ array')
    resolver = Resolver(len(processed))
    resolver.bind(processed_path)
    ply_path = prepared / f'{scene}_vh_clean_2.ply'
    resolver.bind(ply_path)
    vertex = PlyData.read(ply_path)['vertex']
    xyz = np.column_stack([vertex[k] for k in ('x', 'y', 'z')])
    if not np.isfinite(xyz).all() or not np.array_equal(xyz, processed[:, :3]):
        raise ValueError('PLY and prepared XYZ differ in content/order')
    native_root = Path(spec['records_root']) / scene / 'native_cache'
    native_masks_path = native_root / f'{scene}_pred_masks.npy'
    native_masks = resolver.array(native_masks_path)
    native_classes = resolver.array(native_root / f'{scene}_pred_classes.npy')
    native_scores = resolver.array(native_root / f'{scene}_pred_scores.npy')
    if (native_masks.shape != (len(xyz), len(native_classes)) or native_classes.ndim != 1
            or native_scores.shape != native_classes.shape or not np.isfinite(native_scores).all()):
        raise ValueError('original native prediction arrays differ')

    groups, points_by_hash, aliases, native_columns = {}, {}, {}, []

    def register(points, locator):
        h = point_digest(points)
        if h not in groups:
            groups[h] = {'geometry_key': f'{scene}:geometry:{h}', 'points_sha256': h,
                         'point_count': len(points), 'locators': [], 'legacy_members': [],
                         'dv3_plans': [], 'legacy_geometry_keys': []}
            points_by_hash[h] = points
        elif not np.array_equal(points_by_hash[h], points):
            raise ValueError('geometry hash collision')
        if locator not in groups[h]['locators']:
            groups[h]['locators'].append(locator)
        return groups[h]

    for row in legacy:
        if row['scene_name'] != scene or row['geometry_key'] in aliases:
            raise ValueError('duplicate/misassigned legacy geometry')
        loc = row['canonical_geometry_locator']
        points = resolver.resolve(loc)
        check_claim(points, row['geometry_hash'], row['point_count'], 'sha1')
        if row['geometry_key'] != f"{scene}:geometry:{row['geometry_hash']}":
            raise ValueError('legacy key differs from digest')
        group = register(points, loc)
        group['legacy_geometry_keys'].append(row['geometry_key'])
        aliases[row['geometry_key']] = group['points_sha256']
        if row['member_count'] != len(row['members']):
            raise ValueError('legacy member count differs')
        for member in row['members']:
            own = resolver.resolve(member['geometry_locator'])
            check_claim(own, member['geometry_hash'], member['point_count'], 'sha1')
            if member['scene_name'] != scene or not np.array_equal(own, points):
                raise ValueError('member does not match canonical point set')
            register(own, member['geometry_locator'])
            cls = member['frozen_class_index']
            if type(cls) is not int or not -1 <= cls <= 198:
                raise ValueError('invalid legacy class')
            if member['candidate_source'] == 'native':
                column = member['geometry_locator']['column_index']
                original = resolver.resolve({'kind': 'native_mask_column',
                    'masks_path': str(native_masks_path), 'column_index': column})
                if not np.array_equal(original, own) or int(native_classes[column]) != cls:
                    raise ValueError('native adapter differs from original mask/class')
                if not np.isclose(float(native_scores[column]), member['frozen_score'], rtol=0, atol=1e-7):
                    raise ValueError('native adapter differs from original score')
                provenance = member['semantic_provenance']
                classes = resolver.array(provenance['classes_path'])
                if int(classes[provenance['class_array_index']]) != cls:
                    raise ValueError('native semantic provenance differs')
                native_columns.append(column)
            group['legacy_members'].append(member)
    if sorted(native_columns) != list(range(native_masks.shape[1])):
        raise ValueError('legacy ledger drops/duplicates original native columns')

    seen, base_keys, refined_count = set(), set(), 0
    for plan in plans:
        if plan['scene_name'] != scene or plan['plan_key'] in seen:
            raise ValueError('duplicate/misassigned D-v3 plan')
        seen.add(plan['plan_key'])
        loc = plan['geometry_locator_read_only']
        points = resolver.resolve(loc)
        algorithm = {'sha1_point_indices': 'sha1', 'sha256_point_indices': 'sha256'}.get(plan['geometry_digest_algorithm'])
        if algorithm is None:
            raise ValueError('unknown D-v3 digest algorithm')
        check_claim(points, plan['geometry_digest'], plan['point_count'], algorithm)
        if plan['append_only']:
            parent = plan['original_union_geometry_key']
            if parent not in aliases or plan['candidate_source'] != 'refined_union':
                raise ValueError('unbound refined parent/source')
            refined_count += 1
        else:
            key = plan['geometry_key']
            if key in base_keys or aliases.get(key) != point_digest(points):
                raise ValueError('D-v3 base geometry differs from legacy input')
            base_keys.add(key)
        register(points, loc)['dv3_plans'].append(plan)
    if base_keys != set(aliases):
        raise ValueError('D-v3 plan does not retain exact legacy geometry set')
    ordered = [groups[k] for k in sorted(groups)]
    packed = {}
    for i, group in enumerate(ordered):
        group['array_key'] = f'g{i:05d}'
        packed[group['array_key']] = points_by_hash[group['points_sha256']]
    summary = {'scene_name': scene, 'point_count': len(xyz), 'legacy_geometry_count': len(legacy),
               'legacy_member_count': sum(len(r['members']) for r in legacy),
               'native_member_count': len(native_columns), 'dv3_plan_count': len(plans),
               'refined_plan_count': refined_count, 'unique_geometry_count': len(ordered),
               'novel_refined_geometry_count': len(groups) - len(set(aliases.values())),
               'xyz_content_and_order_equal': True, 'all_member_point_hashes_verified': True,
               'original_native_columns_preserved': True, 'ground_truth_read': False, 'ap_computed': False}
    return ordered, packed, resolver.files, summary


def grouped(path):
    result = defaultdict(list)
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            result[row['scene_name']].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', help='Explicit subset smoke; otherwise require complete source')
    args = parser.parse_args()
    partition = load_partition(args.split)
    config = read(args.sources)
    spec = config['sources'][args.source]
    ledger_path = Path(spec['legacy_geometry_root']) / 'unique_geometry_ledger.jsonl'
    legacy, plans = grouped(ledger_path), grouped(args.plan)
    expected = set(partition['source_scans'][args.source])
    scenes = sorted(args.scenes or expected)
    if len(scenes) != len(set(scenes)) or not set(scenes) <= expected or set(plans) != set(scenes):
        raise ValueError('requested scenes/plan/source coverage differs')
    identity = {'version': VERSION, 'partition_id': digest(partition), 'sources_sha256': sha(args.sources),
                'source': args.source, 'legacy_ledger_sha256': sha(ledger_path),
                'plan_path': str(args.plan.resolve()), 'plan_sha256': sha(args.plan),
                'code_sha256': sha(__file__), 'split_code_sha256': sha(ROOT/'tools/prepare_fi1_r0_r1_inputs.py'),
                'scenes': scenes, 'subset_smoke': bool(args.scenes)}
    args.output.mkdir(parents=True, exist_ok=True)
    ipath = args.output / 'identity.json'
    if ipath.exists():
        if read(ipath) != identity:
            raise ValueError('resume identity differs; use a new output directory')
    else:
        write(ipath, identity)
    summaries = []
    role_of = {s: role for role, scans in partition['roles'].items() for s in scans}
    for scene in scenes:
        target = args.output / scene
        receipt_path = target / 'receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['identity_sha256'] != digest(identity):
                raise ValueError('receipt identity mismatch')
            for name, expected_hash in {**receipt['input_files'], **{str(target/k): v for k,v in receipt['output_files'].items()}}.items():
                if sha(name) != expected_hash:
                    raise ValueError(f'resume file changed: {name}')
            summaries.append(receipt['summary'])
            continue
        target.mkdir(exist_ok=False)
        started = time.monotonic()
        rows, points, files, summary = audit_scene(scene, spec, legacy[scene], plans[scene])
        summary.update(role=role_of[scene], source=args.source, seconds=time.monotonic()-started)
        write(target/'geometry.json', rows)
        np.savez_compressed(target/'points.npz', **points)
        write(receipt_path, {'identity_sha256': digest(identity), 'input_files': files,
                            'output_files': {p.name: sha(p) for p in target.iterdir()}, 'summary': summary})
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    write(args.output/'summary.json', {'complete': True, 'identity': identity,
        'scene_count': len(summaries), 'roles': dict(Counter(s['role'] for s in summaries)),
        'totals': {key: sum(s[key] for s in summaries) for key in ('legacy_geometry_count',
            'legacy_member_count', 'native_member_count', 'dv3_plan_count', 'refined_plan_count',
            'unique_geometry_count', 'novel_refined_geometry_count')},
        'rows': summaries, 'ground_truth_read': False, 'ap_computed': False,
        'limitations': ['Point order is bound to existing native/prepared assets; this does not replay native model inference.',
                        'Upstream supervision isolation and own semantic evidence are separate pending audits.']})


if __name__ == '__main__':
    main()

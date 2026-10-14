"""Geometry-keyed evidence primitives; no labels, learned scores or AP evaluation."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.build_official100_fi1_b2_multisemantic import digest_record, sha256_file, unit_score, validate_b1
from tools.dm_sms_core import visible_ratio_multiscale_feature


VERSION = 'fi1_unified_evidence_v1'
SOURCES = {'native', 'track', 'pair_union'}


def point_hash(points):
    return hashlib.sha256(np.asarray(points, dtype='<i8').tobytes()).hexdigest()


class PointResolver:
    """Cache scene mask arrays and file digests, never class-dependent geometry."""
    def __init__(self):
        self.arrays = {}
        self.hashes = {}

    def resolve(self, locator):
        kind = locator.get('kind')
        field = 'masks_path' if kind == 'native_mask_column' else 'points_path'
        path = Path(locator[field])
        if not path.is_absolute() or kind not in {'native_mask_column', 'point_indices_npz'}:
            raise ValueError('unsupported/nonabsolute geometry locator')
        if path not in self.hashes:
            self.hashes[path] = sha256_file(path)
        if kind == 'native_mask_column':
            if path not in self.arrays:
                self.arrays[path] = np.load(path, mmap_mode='r', allow_pickle=False)
            masks = self.arrays[path]
            column = locator['column_index']
            if masks.ndim != 2 or type(column) is not int or not 0 <= column < masks.shape[1]:
                raise ValueError('invalid native column')
            mask = masks[:, column]
            if not np.isin(mask, [0, 1]).all():
                raise ValueError('native mask must be binary')
            points = np.flatnonzero(mask).astype('<i8')
        else:
            with np.load(path, allow_pickle=False) as payload:
                raw = payload[locator.get('array_key', 'point_indices')]
            if raw.ndim != 1 or raw.dtype.kind not in 'iu':
                raise ValueError('invalid point indices')
            if raw.size and (raw.min() < 0 or raw.max() > np.iinfo(np.int64).max):
                raise ValueError('point index out of int64 bounds')
            points = np.unique(raw.astype('<i8'))
        return points, {'path': str(path), 'sha256': self.hashes[path]}


def verify_identity(points, claimed_hash, count=None):
    accepted = {point_hash(points), hashlib.sha1(points.tobytes()).hexdigest()}
    if claimed_hash not in accepted or (count is not None and count != len(points)):
        raise ValueError('declared geometry differs from actual point set')


def build_inventory(b1, geometry_rows, class_count, resolver=None):
    """Retain every B1 column and every refined alias; group only exact point sets."""
    scenes = validate_b1(b1, class_count)
    resolver = resolver or PointResolver()
    groups, points_by_key, aliases, plans = {}, {}, {}, set()

    def register(scene, points, locator, binding, claimed):
        key = f'{scene}:geometry:{point_hash(points)}'
        alias = (scene, claimed)
        if alias in aliases and aliases[alias] != key:
            raise ValueError('geometry alias resolves inconsistently')
        aliases[alias] = key
        if key not in groups:
            groups[key] = {'geometry_key': key, 'scene_name': scene, 'points_sha256': point_hash(points),
                           'point_count': len(points), 'locators': [], 'b1_members': [], 'fi1_plan_members': [],
                           'sources': [], 'geometry_quality': None, 'geometry_quality_provenance': None,
                           'ground_truth_read': False, 'ap_computed': False}
            points_by_key[key] = points
        elif not np.array_equal(points_by_key[key], points):
            raise ValueError('geometry digest collision')
        source = {'locator': locator, **binding}
        if source not in groups[key]['locators']:
            groups[key]['locators'].append(source)
        return groups[key]

    for index, row in enumerate(b1):
        if row['candidate_source'] not in SOURCES:
            raise ValueError('unknown B1 source')
        points, binding = resolver.resolve(row['geometry_locator_read_only'])
        claimed = row['geometry_evidence_key'].rsplit(':', 1)[-1]
        verify_identity(points, claimed)
        group = register(row['scene_name'], points, row['geometry_locator_read_only'], binding, claimed)
        group['b1_members'].append({'input_index': index, 'column_index': row['column_index'],
            'semantic_hypothesis_key': row['semantic_hypothesis_key'], 'input_row_sha256': digest_record(row),
            'candidate_source': row['candidate_source'], 'candidate_id': row['candidate_id'],
            'class_index': row['class_index'], 'b1_score': row['score'],
            'fi1_geometry_score': row.get('fi1_geometry_score')})
        group['sources'].append(row['candidate_source'])
    for row in geometry_rows:
        members = [m for m in row['fi1_plan_members'] if m.get('append_only') is True]
        if not members:
            continue
        scene = row['scene_name']
        if scene not in scenes or row.get('ground_truth_read') is not False or row.get('ap_computed') is not False:
            raise ValueError('refined scene/no-GT contract differs')
        points, binding = resolver.resolve(row['geometry_locator_read_only'])
        verify_identity(points, row['geometry_hash'], row['point_count'])
        group = register(scene, points, row['geometry_locator_read_only'], binding, row['geometry_hash'])
        for member in members:
            if member['plan_key'] in plans:
                raise ValueError('duplicate refined plan')
            plans.add(member['plan_key'])
            match = re.fullmatch(re.escape(scene) + r':union:(\d+):refined', member['plan_key'])
            if (member['candidate_source'] != 'refined_union' or match is None
                    or type(member['candidate_id']) is not int or int(match.group(1)) != member['candidate_id']):
                raise ValueError('refined plan identity differs')
            for flag in ('candidate_deletion', 'geometry_mutation', 'class_mutation', 'ground_truth_read', 'ap_computed'):
                if member.get(flag) is not False:
                    raise ValueError('refined forbidden/missing flag: ' + flag)
            other, other_binding = resolver.resolve(member['geometry_locator_read_only'])
            verify_identity(other, member['geometry_hash'], member['point_count'])
            if not np.array_equal(points, other):
                raise ValueError('refined plan locator differs')
            register(scene, other, member['geometry_locator_read_only'], other_binding, member['geometry_hash'])
            unit_score(member['fi1_geometry_score'], 'FI1 marginal score')
            group['fi1_plan_members'].append(dict(member))
            group['sources'].append('refined_union')
    result = sorted(groups.values(), key=lambda r: (r['scene_name'], r['geometry_key']))
    for row in result:
        row['sources'] = sorted(set(row['sources']))
        row['is_new_geometry'] = not bool(row['b1_members'])
    verify_replay(result, b1)
    return result, points_by_key, resolver.hashes


def verify_replay(inventory, b1):
    members = sorted((m for r in inventory for m in r['b1_members']), key=lambda m: m['input_index'])
    if [m['input_index'] for m in members] != list(range(len(b1))):
        raise ValueError('B1 membership dropped/duplicated')
    for member, row in zip(members, b1):
        if member['input_row_sha256'] != digest_record(row):
            raise ValueError('B1 replay row changed')
        for out, original in [('column_index','column_index'), ('semantic_hypothesis_key','semantic_hypothesis_key'),
                              ('class_index','class_index'), ('b1_score','score')]:
            if member[out] != row[original]:
                raise ValueError('B1 replay field changed')


def baseline_maps(context, class_count):
    """Call the original painting implementation, aligned by explicit frame IDs."""
    import torch
    from utils import OpenYolo3D
    world = context['world']
    frames = [Path(p).stem for p in world.color_paths]
    if len(set(frames)) != len(frames) or set(context['predictions']) != set(frames):
        raise ValueError('YOLO cache frame coverage differs from projection frames')
    ordered = {}
    for frame in frames:
        pred = context['predictions'][frame]
        boxes = torch.as_tensor(pred['bbox']).cpu()
        labels = torch.as_tensor(pred['labels']).cpu()
        scores = torch.as_tensor(pred['scores']).cpu()
        if boxes.shape != (len(labels), 4) or labels.ndim != 1 or scores.shape != labels.shape:
            raise ValueError('YOLO bbox/label/score shape differs')
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError('nonfinite YOLO frame')
        if ((labels < 0) | (labels >= class_count) | (labels != labels.long())).any():
            raise ValueError('YOLO class outside vocabulary')
        ordered[frame] = {'bbox': boxes, 'labels': labels, 'scores': scores}
    holder = SimpleNamespace(world2cam=SimpleNamespace(height=int(world.depth_resolution[0]),
        width=int(world.depth_resolution[1])), scaling_params=context['scaling'])
    return OpenYolo3D.construct_label_maps(holder, ordered).numpy()


def yolo_evidence(points, context, maps, class_count, top_views):
    import torch
    from utils import get_visibility_mat
    from tools.build_z1_yoloworld_multiview_distribution_ledger import _frame_evidence
    if type(top_views) is not int or top_views < 1:
        raise ValueError('positive label-map view budget required')
    visibility = context['visibility']
    frames = [Path(p).stem for p in context['world'].color_paths]
    votes = np.zeros(class_count, dtype=np.int64)
    boxes = np.zeros(class_count, dtype=np.float64)
    views = []
    if len(points):
        if points[-1] >= visibility.shape[1]:
            raise ValueError('geometry exceeds projection point space')
        mask = torch.zeros((1, visibility.shape[1]), dtype=torch.bool)
        mask[0, points] = True
        selected = get_visibility_mat(mask, torch.from_numpy(visibility), topk=top_views)[0].numpy()
        for frame_index in np.flatnonzero(selected):
            visible = points[visibility[frame_index, points]]
            coords = context['projections'][frame_index, visible]
            if len(coords) and ((coords < 0).any() or (coords[:,0] >= maps.shape[2]).any() or (coords[:,1] >= maps.shape[1]).any()):
                raise ValueError('visible coordinates outside label map')
            labels = maps[frame_index, coords[:,1], coords[:,0]]
            counts = np.bincount(labels[labels >= 0].astype(int), minlength=class_count)
            votes += counts
            _, values = _frame_evidence(points, int(frame_index), context['projections'], visibility,
                context['scaling'], context['predictions'][frames[frame_index]], class_count)
            vector = np.zeros(class_count)
            for c, value in values.items():
                vector[c] = value
            weight = len(visible) / len(points)
            boxes += weight * vector
            views.append({'frame_id': frames[frame_index], 'frame_index': int(frame_index),
                'visible_point_count': len(visible), 'weight': weight,
                'label_votes': counts.tolist(), 'bbox_evidence': vector.tolist()})
    maximum = int(votes.max())
    total = float(boxes.sum())
    return {'views': views, 'label_map': {'available': bool(maximum), 'counts': votes.tolist(),
        'max_normalized': (votes / maximum).tolist() if maximum else None,
        'background_sentinel_score': 0.0 if maximum else 1.0,
        'missing_reason': None if maximum else 'no_labeled_visible_points'},
        'bbox': {'available': bool(total), 'weighted_evidence': boxes.tolist(),
        'distribution': (boxes / total).tolist() if total else None,
        'missing_reason': None if total else 'no_box_support'},
        'view_selection': 'original get_visibility_mat, CPU torch.topk; baseline configured budget',
        'top_views': top_views}


def aggregate_alpha(features, ratios, text_features):
    features = np.asarray(features, dtype=np.float32)
    text = np.asarray(text_features, dtype=np.float32)
    if features.ndim != 3 or features.shape[1] != 3 or text.ndim != 2 or features.shape[2] != text.shape[1]:
        raise ValueError('Alpha feature dimensions/three-scale contract differ')
    if not np.isfinite(features).all() or not np.isfinite(text).all():
        raise ValueError('nonfinite Alpha features')
    if not np.allclose(np.linalg.norm(features, axis=-1), 1, atol=2e-4) or not np.allclose(np.linalg.norm(text, axis=-1),1,atol=2e-4):
        raise ValueError('Alpha features are not L2 normalized')
    aggregate = visible_ratio_multiscale_feature(features, np.asarray(ratios))
    cosines = aggregate @ text.T
    per_scale = features @ text.T
    return {'available': True, 'cosine': cosines.tolist(),
            'view_scale_std': per_scale.reshape(-1, len(text)).std(axis=0).tolist(),
            'aggregate_feature': aggregate.tolist(), 'missing_reason': None}


def class_rows(inventory_row, evidence, labels):
    evidence_hash = digest_record(evidence)
    for c, name in enumerate(labels):
        yolo, alpha = evidence['yolo'], evidence['alpha']
        existing = [m for m in inventory_row['b1_members'] if m['class_index'] == c]
        yield {'geometry_key': inventory_row['geometry_key'], 'scene_name': inventory_row['scene_name'],
            'points_sha256': inventory_row['points_sha256'], 'class_index': c, 'class_name': name,
            'sources': inventory_row['sources'], 'existing_b1_members': existing,
            'label_map_available': yolo['label_map']['available'],
            'label_map_vote': yolo['label_map']['max_normalized'][c] if yolo['label_map']['available'] else None,
            'bbox_available': yolo['bbox']['available'],
            'bbox_probability': yolo['bbox']['distribution'][c] if yolo['bbox']['available'] else None,
            'alpha_available': alpha['available'], 'alpha_cosine': alpha['cosine'][c] if alpha['available'] else None,
            'alpha_view_scale_std': alpha['view_scale_std'][c] if alpha['available'] else None,
            'geometry_quality': inventory_row['geometry_quality'], 'score': None,
            'evidence_sha256': evidence_hash, 'ground_truth_read': False, 'ap_computed': False}

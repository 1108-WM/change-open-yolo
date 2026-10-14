"""Frozen-pipeline attribution primitives. No model fitting or parameter search."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_rows(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


class Bindings:
    def __init__(self):
        self.files = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('hash mismatch: ' + str(path))
        if str(path) in self.files and self.files[str(path)] != value:
            raise ValueError('input changed: ' + str(path))
        self.files[str(path)] = value
        return path


def point_hash(points):
    return hashlib.sha256(np.asarray(points, dtype='<i8').tobytes()).hexdigest()


def points_from_population(root, row, binding):
    with np.load(binding.check(root / row['points_file'], row['points_file_sha256']), allow_pickle=False) as f:
        p = f['point_indices']
    if (p.ndim != 1 or p.dtype.kind not in 'iu' or (p < 0).any() or
            not np.array_equal(p, np.unique(p)) or len(p) != row['point_count'] or
            point_hash(p) != row['points_sha256']):
        raise ValueError('population point identity differs')
    return p.astype('<i8', copy=False)


def unique(rows, key):
    out = {r[key]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError('duplicate ' + key)
    return out


def configure_evaluator(evaluator):
    # Match the existing project's ScanNet200 configuration, without modifying
    # the evaluator source or its threshold/ignore/AP implementation.
    from evaluate.scannet200 import scannet_constants as constants
    names = dict(zip(constants.VALID_CLASS_IDS_200, constants.CLASS_LABELS_200))
    evaluator.DATASET_NAME = 'scannet200'
    evaluator.VALID_CLASS_IDS = np.asarray(constants.VALID_CLASS_IDS_200_INST)
    evaluator.CLASS_LABELS = tuple(names[int(i)] for i in evaluator.VALID_CLASS_IDS)
    evaluator.ID_TO_LABEL = {int(i): names[int(i)] for i in evaluator.VALID_CLASS_IDS}
    evaluator.LABEL_TO_ID = {name: i for i, name in evaluator.ID_TO_LABEL.items()}
    return evaluator


def geometry_statistics(points_by_key, gt, evaluator):
    """Exact point-index histograms reused across semantic copies of a mask."""
    inventory = evaluator.util_3d.get_instances(gt, evaluator.VALID_CLASS_IDS,
                                               evaluator.CLASS_LABELS, evaluator.ID_TO_LABEL)
    counts = {int(r['instance_id']): int(r['vert_count']) for rows in inventory.values() for r in rows}
    valid_classes = set(map(int, evaluator.VALID_CLASS_IDS))
    stats = {}
    for key, points in points_by_key.items():
        if len(points) and points[-1] >= len(gt):
            raise ValueError('points exceed GT point space')
        ids, sizes = np.unique(gt[points], return_counts=True)
        histogram = {int(i): int(n) for i, n in zip(ids, sizes)}
        stats[key] = {'point_count': len(points),
                      'void_intersection': sum(n for i, n in histogram.items() if i // 1000 not in valid_classes),
                      'intersections': {str(i): n for i, n in histogram.items() if i in counts}}
    return inventory, stats


def sparse_assign(rows, stats, inventory, evaluator, scene):
    """Equivalent scan association; validated against the unmodified dense path.

    Keeps every prediction column, including duplicate geometry/class columns,
    zero scores, small GT instances and the original valid/void definitions.
    """
    gt2pred = copy.deepcopy(inventory)
    pred2gt = {label: [] for label in evaluator.CLASS_LABELS}
    for group in gt2pred.values():
        for g in group:
            g['matched_pred'] = []
    pred_id = 0
    for column, row in enumerate(rows):
        cid = int(row['class_index'])
        if cid not in evaluator.PRED_ID_TO_ID:
            raise ValueError('prediction class outside evaluator mapping')
        label_id = int(evaluator.PRED_ID_TO_ID[cid])
        if label_id not in evaluator.ID_TO_LABEL:
            continue
        label = evaluator.ID_TO_LABEL[label_id]
        s = stats[row['geometry_key']]
        if s['point_count'] < evaluator.opt['min_region_sizes'][0]:
            continue
        score = float(row['score'])
        if not np.isfinite(score):
            raise ValueError('nonfinite prediction score')
        instance = {'uuid': f'{scene}:{column}', 'pred_id': pred_id,
                    'label_id': label_id, 'vert_count': s['point_count'],
                    'confidence': score, 'void_intersection': s['void_intersection']}
        matched = []
        for g in gt2pred[label]:
            intersection = s['intersections'].get(str(g['instance_id']), 0)
            if intersection:
                gc = g.copy(); pc = instance.copy()
                gc['intersection'] = intersection; pc['intersection'] = intersection
                matched.append(gc); g['matched_pred'].append(pc)
        instance['matched_gt'] = matched
        pred2gt[label].append(instance)
        pred_id += 1
    return {'gt': gt2pred, 'pred': pred2gt}


def normalized_matches(match):
    # UUIDs are random in the original evaluator. Nested matched_pred lists in
    # matched_gt are unused aliases, not independent association information.
    g = {name: [{**{k: v for k, v in r.items() if k != 'matched_pred'},
                 'matched_pred': [{k: v for k, v in p.items() if k != 'uuid'} for p in r['matched_pred']]}
                for r in rows] for name, rows in match['gt'].items()}
    p = {name: [{**{k: v for k, v in r.items() if k not in ('uuid', 'matched_gt')},
                 'matched_gt': [{k: v for k, v in t.items() if k != 'matched_pred'} for t in r['matched_gt']]}
                for r in rows] for name, rows in match['pred'].items()}
    return {'gt': g, 'pred': p}


def dense_parity(rows, points, gt_path, sparse, evaluator):
    gt = evaluator.util_3d.load_ids(str(gt_path))
    masks = np.zeros((len(gt), len(rows)), dtype=bool)
    for j, row in enumerate(rows):
        masks[points[row['geometry_key']], j] = True
    prediction = {'pred_masks': masks,
                  'pred_classes': np.asarray([r['class_index'] for r in rows], dtype=np.int64),
                  'pred_scores': np.asarray([r['score'] for r in rows], dtype=np.float64)}
    g, p = evaluator.assign_instances_for_scan(prediction, str(gt_path))
    if normalized_matches(sparse) != normalized_matches({'gt': g, 'pred': p}):
        raise ValueError('sparse association differs from original dense evaluator')


def fixed_mask_swap(rows, mapping):
    return [{**r, 'geometry_key': mapping.get(r['geometry_key'], r['geometry_key'])} for r in rows]


def iou(stats, gt_row):
    intersection = stats['intersections'].get(str(gt_row['instance_id']), 0)
    return intersection / (stats['point_count'] + gt_row['vert_count'] - intersection)


def best_target(stats, valid_gt):
    hits = [g for g in valid_gt if stats['intersections'].get(str(g['instance_id']), 0)]
    return min(hits, key=lambda g: (-iou(stats, g), -stats['intersections'][str(g['instance_id'])],
                                    g['instance_id'])) if hits else None


def class_rank(values, cid):
    if values is None:
        return None
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (198,) or not np.isfinite(a).all():
        raise ValueError('full finite198 evidence required')
    return {'value': float(a[cid]), 'best_rank': int(1 + (a > a[cid]).sum()),
            'worst_rank': int((a >= a[cid]).sum())}


def alpha_visual_change(parent, child):
    if parent is None or child is None:
        return None
    pv, cv = parent['alpha_views'], child['alpha_views']
    common = sorted(pv.keys() & cv.keys())
    fields = ('image_sha256', 'sam_box_prompt_xyxy', 'sam_mask_sha256', 'feature_sha256', 'crops')
    return {'common_frames': len(common),
            'identical_complete_model_input_frames': sum(all(pv[f][k] == cv[f][k] for k in fields) for f in common),
            'identical_sam_frames': sum(pv[f]['sam_mask_sha256'] == cv[f]['sam_mask_sha256'] for f in common),
            'identical_aggregate_cosine': parent['alpha'] == child['alpha']}

"""Frozen R2 feature, supervision, weighting and selection definitions."""
from collections import Counter

import numpy as np
from scipy.special import softmax, xlogy

from tools.fi1_r0_r1_protocol import direct_score, original_aliases

FEATURES = [
    'direct_score', 'yolo_probability', 'alpha_probability', 'alpha_cosine',
    'alpha_view_scale_std', 'yolo_rank', 'alpha_rank', 'yolo_gap_to_max',
    'alpha_gap_to_max', 'yolo_entropy', 'alpha_entropy', 'yolo_alpha_js',
    'top1_agreement', 'log1p_point_count', 'source_native', 'source_track',
    'source_pair_union', 'source_refined_union', 'yolo_missing', 'alpha_missing',
]
MODEL_PARAMS = dict(loss='squared_error', learning_rate=0.05, max_iter=100,
                    max_leaf_nodes=7, max_depth=3, min_samples_leaf=100,
                    l2_regularization=1., early_stopping=False, random_state=20260913)
DETAILS = {
    'version': 'fi1_r2_quality_calibration_v1', 'features': FEATURES,
    'model': MODEL_PARAMS, 'feature_dtype': 'float64',
    'entropy_js': 'natural logs; JS and agreement zero if either modality missing',
    'rank': 'descending probability then ascending class index; zero-based /197',
    'gap': 'class probability minus largest probability in that modality',
    'weight': 'equal scene, then geometry, then class; normalize mean training-row weight to1 separately per fit',
    'target': 'maximum IoU with same-class valid GT; all predicted points included in union; size<100 target0',
    'prediction': 'clip to[0,1]; unchanged R1 pre-budget members; float64 top600 with original tie rule',
    'whole_system_oof': False, 'validation_is_blind': False,
}


def geometry_features(g, evidence):
    if g['geometry_key'] != evidence['geometry_key'] or g['points_sha256'] != evidence['points_sha256']:
        raise ValueError('geometry/evidence identity differs')
    direct = direct_score(evidence)
    x = np.zeros((198, len(FEATURES)), dtype=np.float64)
    x[:, 0] = direct
    distributions = []
    available = [evidence[k]['available'] for k in ('yolo', 'alpha')]
    for i, name in enumerate(('yolo', 'alpha')):
        x[:, 18+i] = not available[i]
        p = np.zeros(198)
        if available[i]:
            item = evidence[name]
            if name == 'yolo':
                p = np.asarray(item['votes'], dtype=np.float64)
                p = p/p.sum()
            else:
                cosine = np.asarray(item['cosine'], dtype=np.float64)
                p = softmax(100*(cosine-cosine.max()))
                std = np.asarray(item['view_scale_std'], dtype=np.float64)
                if std.shape != (198,) or not np.isfinite(std).all() or (std < 0).any():
                    raise ValueError('invalid Alpha scale standard deviation')
                x[:, 3], x[:, 4] = cosine, std
            ranks = np.empty(198, dtype=np.float64)
            ranks[np.lexsort((np.arange(198), -p))] = np.arange(198)/197.
            x[:, 5+i] = ranks
            x[:, 7+i] = p-p.max()
            x[:, 9+i] = -np.sum(xlogy(p, p))
        x[:, 1+i] = p
        distributions.append(p)
    if all(available):
        p, q = distributions
        m = (p+q)/2
        x[:, 11] = np.sum(xlogy(p, p)-xlogy(p, m)+xlogy(q, q)-xlogy(q, m))/2
        x[:, 12] = np.argmax(p) == np.argmax(q)
    x[:, 13] = np.log1p(g['point_count'])
    sources = {a['source'] for a in original_aliases(g)}
    for i, source in enumerate(('native', 'track', 'pair_union', 'refined_union')):
        x[:, 14+i] = source in sources
    if not np.isfinite(x).all():
        raise ValueError('nonfinite R2 feature')
    return x


def valid_instances(gt, dataset_class_ids, minimum=100):
    ids, counts = np.unique(gt, return_counts=True)
    valid = (ids >= 1000) & (counts >= minimum) & np.isin(ids//1000, dataset_class_ids)
    return dict(zip(ids[valid].tolist(), counts[valid].tolist()))


def geometry_targets(points, gt, instances, dataset_class_ids, minimum=100):
    points = np.asarray(points)
    if points.ndim != 1 or points.dtype.kind not in 'iu' or not np.array_equal(points, np.unique(points)):
        raise ValueError('invalid geometry indices')
    if len(points) and (points.min() < 0 or points.max() >= len(gt)):
        raise ValueError('geometry outside GT point domain')
    target = np.zeros(len(dataset_class_ids), dtype=np.float64)
    if len(points) < minimum:
        return target
    mapping = {int(c): i for i, c in enumerate(dataset_class_ids)}
    ids, overlaps = np.unique(gt[points], return_counts=True)
    for instance, intersection in zip(ids, overlaps):
        if int(instance) in instances:
            c = mapping[int(instance)//1000]
            target[c] = max(target[c], intersection/(len(points)+instances[int(instance)]-intersection))
    return target


def scene_weights(members):
    counts = Counter(r['geometry_key'] for r in members)
    if not counts:
        raise ValueError('empty scene members')
    return np.asarray([1/(len(counts)*counts[r['geometry_key']]) for r in members], dtype=np.float64)


def select(members, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(members),) or not np.isfinite(scores).all():
        raise ValueError('invalid prediction scores')
    if len({(r['geometry_key'], r['class_index']) for r in members}) != len(members):
        raise ValueError('duplicate eligible class member')
    updated = [{**r, 'score': float(s)} for r, s in zip(members, np.clip(scores, 0, 1))]
    return sorted(updated, key=lambda r: (-r['score'], not r['has_original_native_alias'],
                                          r['geometry_key'], r['class_index']))[:600]

"""Fixed, GT-free, same-class point-mask NMS after frozen B26 scoring.

The only overlap rule is point IoU > 19/20. Suppressed rows never become
witnesses, and the common budget is applied after suppression, with refill.
"""
from collections import defaultdict

import numpy as np

from tools.fi1_r3_core import point_matrix
from tools.fi1_r3b_delta_diagnostic_core import rank


def select(members, points, point_count, budget=800):
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    keys = sorted(points)
    lookup = {k: i for i, k in enumerate(keys)}
    identities = [(m['geometry_key'], m['class_index']) for m in members]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate geometry-class member')
    for m in members:
        if m['geometry_key'] not in lookup or type(m['class_index']) is not int or not 0 <= m['class_index'] < 198:
            raise ValueError('invalid member identity')
        if not np.isfinite(m['score']) or not 0 <= m['score'] <= 1:
            raise ValueError('invalid frozen B26 score')
    matrix = point_matrix([points[k] for k in keys], point_count)
    intersection = (matrix @ matrix.T).toarray()
    sizes = np.asarray([len(points[k]) for k in keys], dtype=np.int64)
    union = sizes[:, None] + sizes[None, :] - intersection
    # Integer comparison preserves the strict boundary, including exact .95.
    conflicts = (20 * intersection > 19 * union) & (union > 0)
    np.fill_diagonal(conflicts, False)
    ranked = rank(members)
    selected, suppressed = [], []
    kept_by_class = defaultdict(list)
    examined = 0
    for position, m in enumerate(ranked):
        examined += 1
        gi = lookup[m['geometry_key']]
        witness = next((w for w in kept_by_class[m['class_index']]
                        if conflicts[gi, lookup[w['geometry_key']]]), None)
        if witness is not None:
            wi = lookup[witness['geometry_key']]
            suppressed.append(dict(member=m, witness=witness, input_rank=position + 1,
                                   intersection=int(intersection[gi, wi]),
                                   union=int(union[gi, wi]), iou=float(intersection[gi, wi] / union[gi, wi])))
        else:
            selected.append(m)
            kept_by_class[m['class_index']].append(m)
            if len(selected) == budget:
                break
    near_pairs = [dict(left=keys[i], right=keys[j], intersection=int(intersection[i, j]),
                       union=int(union[i, j]), iou=float(intersection[i, j] / union[i, j]))
                  for i, j in zip(*np.nonzero(np.triu(conflicts, 1)))]
    overlapping_pairs = [dict(left=keys[i], right=keys[j], intersection=int(intersection[i, j]),
                             union=int(union[i, j]), iou=float(intersection[i, j] / union[i, j]))
                         for i, j in zip(*np.nonzero(np.triu((2 * intersection > union) & (union > 0), 1)))]
    return selected, dict(examined=examined, unexamined=len(ranked) - examined,
                         suppressed=suppressed, near_geometry_pairs=near_pairs,
                         overlap_gt_half_pairs=overlapping_pairs)

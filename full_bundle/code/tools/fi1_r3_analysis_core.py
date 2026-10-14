"""Post-hoc diagnostics only: no score, member or target modification."""
from collections import Counter
from copy import deepcopy

import numpy as np

from tools.fi1_r0_r1_protocol import original_aliases

SOURCES = ('native', 'track', 'pair_union', 'refined_union')


def member_key(row):
    return row['geometry_key'], int(row['class_index'])


def geometry_sources(geometry):
    sources = sorted({a['source'] for a in original_aliases(geometry)})
    if not sources or set(sources) - set(SOURCES):
        raise ValueError('missing or unrecognized geometry source')
    return sources


def duplicate_counts(members, geometry_keys, iou):
    """Count unordered same-class pairs and involved members; strict IoU > .5."""
    if len({member_key(r) for r in members}) != len(members):
        raise ValueError('duplicate member identity')
    lookup = {k: i for i, k in enumerate(geometry_keys)}
    by_class = {}
    for i, row in enumerate(members):
        by_class.setdefault(row['class_index'], []).append(i)
    degrees = np.zeros(len(members), dtype=np.int64)
    possible = 0
    for ids in by_class.values():
        gi = [lookup[members[i]['geometry_key']] for i in ids]
        strong = iou[np.ix_(gi, gi)] > .5
        np.fill_diagonal(strong, False)
        degrees[ids] = strong.sum(axis=1)
        possible += len(ids) * (len(ids) - 1) // 2
    return {
        'members': len(members), 'same_class_possible_pairs': possible,
        'pairs_iou_gt_half': int(degrees.sum() // 2),
        'members_with_neighbor': int((degrees > 0).sum()),
        'degree_histogram': dict(Counter(map(int, degrees))),
    }


def aggregate_duplicates(records):
    result = {k: sum(r[k] for r in records) for k in (
        'members', 'same_class_possible_pairs', 'pairs_iou_gt_half', 'members_with_neighbor')}
    hist = Counter()
    for r in records:
        hist.update({int(k): int(v) for k, v in r['degree_histogram'].items()})
    if sum(hist.values()) != result['members'] or sum(k*v for k, v in hist.items()) != 2*result['pairs_iou_gt_half']:
        raise ValueError('duplicate count accounting differs')
    result['degree_histogram'] = dict(sorted(hist.items()))
    n = result['members']
    result['fraction_members_with_neighbor'] = result['members_with_neighbor'] / n if n else 0.
    result['mean_neighbors'] = 2*result['pairs_iou_gt_half'] / n if n else 0.
    return result


def selection_change(before, after, sources):
    a, b = {member_key(r) for r in before}, {member_key(r) for r in after}
    if len(a) != len(before) or len(b) != len(after):
        raise ValueError('repeated selected identity')
    def counts(keys):
        return dict(Counter('+'.join(sources[g]) for g, _ in keys))
    return {'retained': len(a & b), 'removed': len(a-b), 'added': len(b-a),
            'removed_sources_exclusive': counts(a-b), 'added_sources_exclusive': counts(b-a)}


def filter_matches(match, keep):
    """Filter both directions of frozen associations; keep every GT instance.

    No budget refill. `keep` is a set of prediction UUIDs. AP changes from
    removing a source are conditional diagnostics, not additive contributions.
    """
    out = deepcopy(match)
    for label, preds in out['pred'].items():
        out['pred'][label] = [p for p in preds if p['uuid'] in keep]
    for instances in out['gt'].values():
        for gt in instances:
            gt['matched_pred'] = [p for p in gt['matched_pred'] if p['uuid'] in keep]
    return out


def weighted_errors(scores, maximum, competition, weights):
    if not (scores.shape == maximum.shape == competition.shape == weights.shape):
        raise ValueError('unaligned targets/scores')
    if not np.isclose(weights.sum(), 1.):
        raise ValueError('scene weights must sum to one')
    return {'mse_max_iou': float(weights @ ((scores-maximum)**2)),
            'mse_competition': float(weights @ ((scores-competition)**2))}

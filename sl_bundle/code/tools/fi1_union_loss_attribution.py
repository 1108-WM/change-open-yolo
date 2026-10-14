"""Frozen two-by-two interventions on NT/NTU members, not a deployable scorer."""
from __future__ import annotations

import numpy as np
from tools.fi1_r3b_delta_diagnostic_core import rank

ARMS = ('A', 'B', 'C', 'D')
THRESHOLDS = (.5, .75, .9)


def key(row):
    return row['geometry_key'], row['class_index']


def factorial_selections(nt, s0, ntu, s1, budget=800):
    """A=M0/s0; B=M0/s0+delta/s1; C=M0/s1; D=M1/s1.

    s1 must be computed on the whole NTU pool before restricting C's output.
    All newly eligible members participate in B/D, including new classes on an
    old physical mask. The original native tie flag cannot change on shared rows.
    """
    ntkeys = [key(m) for m in nt]; ntukeys = [key(m) for m in ntu]
    if len(set(ntkeys)) != len(ntkeys) or len(set(ntukeys)) != len(ntukeys):
        raise ValueError('duplicate member identity')
    if not set(ntkeys) <= set(ntukeys):
        raise ValueError('NT members must be a subset of NTU')
    for members, scores in ((nt, s0), (ntu, s1)):
        scores = np.asarray(scores)
        if scores.shape != (len(members),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('invalid frozen member scores')
    if type(budget) is not int or budget < 1:
        raise ValueError('invalid budget')
    old = {key(m): (m, float(s)) for m, s in zip(nt, s0)}
    updated = {key(m): (m, float(s)) for m, s in zip(ntu, s1)}
    for k in old:
        if old[k][0]['has_original_native_alias'] != updated[k][0]['has_original_native_alias']:
            raise ValueError('shared native tie identity changed')
    a = [{**m, 'score': s} for m, s in old.values()]
    c = [{**m, 'score': updated[k][1]} for k, (m, _) in old.items()]
    added = [{**m, 'score': s} for k, (m, s) in updated.items() if k not in old]
    d = [{**m, 'score': s} for m, s in updated.values()]
    return {arm: rank(members)[:budget] for arm, members in zip(ARMS, (a, a + added, c, d))}


def member_diagnostics(keys, sizes, iou, intersections, gt_classes, selected, original_members, nt_geometries):
    """Coverage and candidate descriptions, explicitly not evaluator TP/FP labels."""
    lookup = {k: n for n, k in enumerate(keys)}
    original = {key(m) for m in original_members}
    cls = np.asarray(gt_classes, dtype=np.int64)
    same = {int(c): np.flatnonzero(cls == c) for c in set(cls.tolist())}
    scope = ('all', 'original_member', 'new_member', 'new_geometry_member', 'new_class_on_existing_geometry')
    details = {name: dict(members=0, small_ignored=0, multi_significant_GT=0,
        thresholds={str(t): dict(correct_range=0, wrong_class_good_range=0, no_range_at_threshold=0,
                               unique_correct_GT=0, redundant_correct_members=0) for t in THRESHOLDS}) for name in scope}
    covered = {name: {str(t): set() for t in THRESHOLDS} for name in scope}
    for m in selected:
        i = lookup[m['geometry_key']]
        names = ['all', 'original_member' if key(m) in original else 'new_member']
        if key(m) not in original:
            names.append('new_class_on_existing_geometry' if m['geometry_key'] in nt_geometries else 'new_geometry_member')
        ci = same.get(m['class_index'], np.empty(0, dtype=np.int64))
        best_any = float(iou[i].max(initial=0))
        good_same = iou[i, ci]
        multi = int(((intersections[i] >= 100) & (intersections[i] >= .1 * sizes[i])).sum()) >= 2
        for name in names:
            row = details[name]; row['members'] += 1
            if sizes[i] < 100:
                row['small_ignored'] += 1
                continue
            row['multi_significant_GT'] += multi
            for t in THRESHOLDS:
                hit = ci[good_same > t]
                kind = 'correct_range' if len(hit) else 'wrong_class_good_range' if best_any > t else 'no_range_at_threshold'
                row['thresholds'][str(t)][kind] += 1
                covered[name][str(t)].update(map(int, hit))
    for name in scope:
        for t in THRESHOLDS:
            row = details[name]['thresholds'][str(t)]
            row['unique_correct_GT'] = len(covered[name][str(t)])
            row['redundant_correct_members'] = row['correct_range'] - row['unique_correct_GT']
    return details, covered['all']

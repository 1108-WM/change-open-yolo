"""Inference-only source removal and scoring for the frozen FI1 core ablation."""
from __future__ import annotations

import copy
import numpy as np

from tools.fi1_r0_r1_protocol import build_pair, original_aliases
from tools.fi1_r2_core import geometry_features
from tools.fi1_r3_core import pair_overlap, context_features
from tools.fi1_r3b_delta_diagnostic_core import rank

ARMS = ('full', 'nt_only', 'direct_semantics')
BUDGET = 800


def source_pool(geometry, allowed=('native', 'track')):
    """Remove aliases, not shared pointsets; rebuild membership from surviving aliases."""
    allowed = set(allowed)
    if not allowed <= {'native', 'track', 'pair_union', 'refined_union'}:
        raise ValueError('unknown source')
    result = []
    for original in geometry:
        original_aliases(original)  # Reject unrecognized or duplicated aliases.
        g = copy.deepcopy(original)
        g['legacy_members'] = [m for m in g['legacy_members'] if m['candidate_source'] in allowed]
        g['dv3_plans'] = [p for p in g['dv3_plans'] if
                          ('refined_union' if p['append_only'] else p['candidate_source']) in allowed]
        if original_aliases(g):
            result.append(g)
    return result


def features_and_members(geometry, points, evidence, point_count):
    keys = [g['geometry_key'] for g in geometry]
    if len(set(keys)) != len(keys) or set(keys) != set(points) or set(keys) != set(evidence):
        raise ValueError('geometry/evidence/points identity differs')
    members = build_pair(geometry, evidence, budget=BUDGET)['R1']['eligible']
    if not members:
        return members, np.empty((0, 26), dtype=np.float64)
    own = {g['geometry_key']: geometry_features(g, evidence[g['geometry_key']]) for g in geometry}
    x20 = np.asarray([own[m['geometry_key']][m['class_index']] for m in members])
    iou, coverage = pair_overlap([points[k] for k in keys], point_count)
    x26 = np.column_stack((x20, context_features(members, keys, iou, coverage)))
    if x26.shape != (len(members), 26) or not np.isfinite(x26).all():
        raise ValueError('invalid inference features')
    return members, x26


def learned_selection(members, x26, model):
    if getattr(model, 'n_features_in_', None) != 26:
        raise ValueError('requires the frozen B26 head')
    scores = np.clip(model.predict(x26), 0, 1) if members else np.empty(0)
    if scores.shape != (len(members),) or not np.isfinite(scores).all():
        raise ValueError('invalid prediction')
    return rank([{**m, 'score': float(s)} for m, s in zip(members, scores)])[:BUDGET], scores


def direct_selection(members):
    """Same full pre-budget members and existing fusion, no class regeneration."""
    return rank(members)[:BUDGET]

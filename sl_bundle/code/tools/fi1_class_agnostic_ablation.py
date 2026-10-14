"""Frozen geometry scoring and class-agnostic evaluation, without Torch."""
from __future__ import annotations

import numpy as np
from tools.fi1_r0_r1_protocol import original_aliases
from tools.fi1_r3b_delta_diagnostic_core import evaluator

ARMS = ('nt_only', 'ntu', 'full')
VIEWS = ('dv3_geometry', 'b26_output')
BUDGET = 800


def geometry_selection(pool, budget=BUDGET):
    """One actual pointset, maximum surviving frozen D-v3 plan score, fixed ties."""
    result = []
    for g in pool:
        plans = g['dv3_plans']
        if not plans:
            raise ValueError('geometry lacks frozen D-v3 score')
        scores = np.asarray([p['challenger_score'] for p in plans], dtype=np.float64)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('invalid frozen D-v3 score')
        result.append(dict(geometry_key=g['geometry_key'], score=float(scores.max()),
            has_native_source=any(a['source']=='native' for a in original_aliases(g))))
    if len({r['geometry_key'] for r in result}) != len(result):
        raise ValueError('duplicate actual geometry')
    return sorted(result, key=lambda r:(-r['score'],not r['has_native_source'],r['geometry_key']))[:budget]


def collapse_members(members):
    """Collapse already selected mask×class rows; keep max B26 score, no refill."""
    best = {}
    for m in members:
        if not np.isfinite(m['score']) or not 0 <= m['score'] <= 1:
            raise ValueError('invalid B26 score')
        k = m['geometry_key']
        if k not in best or (-m['score'],m['class_index']) < (-best[k]['score'],best[k]['class_index']):
            best[k] = m
    result = [dict(geometry_key=k,score=float(m['score']),winning_class_index=m['class_index'])
              for k,m in best.items()]
    return sorted(result,key=lambda r:(-r['score'],r['geometry_key']))


def remap_gt(gt, original_valid_ids, unified_id):
    """Keep all valid instance boundaries, including small ignored objects; void=0."""
    gt = np.asarray(gt, dtype=np.int64)
    ids = np.unique(gt)
    ids = ids[(ids >= 1000) & np.isin(ids//1000, original_valid_ids)]
    if len(ids) >= 1000:
        raise ValueError('class-agnostic instance encoding limit reached')
    mapped = np.zeros_like(gt)
    for n,gid in enumerate(ids,1):
        mapped[gt==gid] = int(unified_id)*1000+n
    return mapped


def class_agnostic_evaluator():
    """Original matching/AP function ASTs, one unified class; original198 valid GT."""
    ev = evaluator()
    original_valid = ev.VALID_CLASS_IDS.copy()
    unified_id = int(original_valid[0]); label = ev.CLASS_LABELS[0]
    ns = ev.assign_instances_for_scan.__globals__
    ns.update(VALID_CLASS_IDS=np.asarray([unified_id]),CLASS_LABELS=(label,),
              ID_TO_LABEL={unified_id:label},PRED_ID_TO_ID={0:unified_id})
    ev.__dict__.update(ns)
    return ev, original_valid, unified_id


def project_matches(template, geometry_keys, selected, label):
    """Subset/rescore original geometry assignments; retain all AP-used fields.

    Original assignment does not depend on prediction score. The redundant nested
    matched_pred field inside matched_gt is omitted; the AP function never reads it.
    """
    gt = [{**{k:v for k,v in g.items() if k!='matched_pred'},'matched_pred':[]}
          for g in template['gt'][label]]
    gt_lookup = {g['instance_id']:g for g in gt}
    lookup = dict(zip(geometry_keys,template['pred'][label]))
    result = []
    for m in selected:
        old = lookup.get(m['geometry_key'])
        if old is None: continue  # Original evaluator drops masks smaller than100.
        p = {k:v for k,v in old.items() if k!='matched_gt'}
        p['confidence']=m['score'];p['pred_id']=len(result);p['matched_gt']=[]
        for g in old['matched_gt']:
            p['matched_gt'].append({k:v for k,v in g.items() if k!='matched_pred'})
            gt_lookup[g['instance_id']]['matched_pred'].append(
                {**{k:v for k,v in p.items() if k!='matched_gt'},'intersection':g['intersection']})
        result.append(p)
    return dict(gt={label:gt},pred={label:result})

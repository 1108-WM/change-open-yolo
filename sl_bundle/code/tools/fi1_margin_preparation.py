"""GT-free representative action/feature preparation for frozen FI1/B1.

This module never imports the evaluator or opens a GT cache. Historical A
quality is permitted only as explicitly marked frozen-upstream evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.build_fi1_margin_dataset import overlapping_context
from tools.fi1_evaluator_delta_core import finite, integer
from tools.fi1_gain_attribution_core import points_from_population, unique
from tools.fi1_scene_reference import digest, main_scene

VERSION = 'fi1_margin_preparation_v1'
SEED = 'fi1-margin-representatives-v1'
SOURCES = ('track', 'pair_union', 'temporary_refined', 'refined_union')
CHANNELS = ('alpha', 'label_map', 'bbox')
VISUAL_FEATURES = (
    'log1p_point_count', 'b1_geometry_present', 'source_native', 'source_track',
    'source_pair_union', 'source_refined_union', 'source_temporary_refined',
    'log1p_geometric_neighbor_count', 'maximum_geometry_overlap_iou',
    'alpha_valid_view_fraction', 'alpha_mean_visible_ratio',
    'refinement_parent_retention_min', 'refinement_parent_retention_max',
    'independent_geometry_quality',
    *(f'{c}_{s}' for c in ('label_map', 'bbox', 'alpha')
      for s in ('available', 'neighbor_evidence_count', 'neighbor_missing_count')),
    *(f'{c}_{s}' for c in ('label_map', 'bbox', 'alpha')
      for s in ('class_support', 'gap_to_own_max', 'max_iou_weighted_neighbor_support', 'stronger_neighbor_iou_max')),
    'alpha_class_view_scale_std',
)
ARMS = ('existing_down', 'existing_keep', 'existing_up', 'new_q25', 'new_q50', 'new_q75')


def protocol():
    return {
        'version': VERSION, 'seed': SEED, 'geometries_per_split_fold_source': 2,
        'source_strata': list(SOURCES), 'minimum_points': 100,
        'geometry_selection': 'fixed hash; main-scene round robin within each split-fold/source; no quota borrowing',
        'eligibility': 'complete unique198 visual rows; >=1 available non-flat semantic channel for new geometry',
        'existing_hypotheses': 'all foreground B1 members of the selected source and geometry; retain duplicate columns',
        'new_hypotheses': 'union of top2 per informative channel, all foreground parent B1 classes, one hash control class',
        'ties': 'within-channel value descending then seeded geometry/class hash; unavailable/flat channels contribute none',
        'scores': {'existing_down': 'float32(clip(B1_score - 0.10,0,1))',
                   'existing_keep': 'exact original B1 score',
                   'existing_up': 'float32(clip(B1_score + 0.10,0,1))',
                   'new': 'float32 linear quantiles [.25,.50,.75] of same-class B1 scores in outer-fit; fallback all fit foreground'},
        'score_anchor_scope': 'fit-only no-GT prediction scores; apply the same anchors to both roles',
        'action_unit': 'one geometry-class/member at one score against unchanged full partition B1',
        'target': 'original evaluator class_ap(after)-class_ap(before); main thresholds strict > .50:.05:.90',
        'undefined_target': 'null when class absent from the entire reference partition; never zero-fill',
        'controls': 'existing_keep evaluated as a zero-change control; append skip has analytic delta0 and no fake prediction',
        'non_additive': True, 'score_grid_is_deployment_calibration': False,
        'gt_used_for_selection_or_features': False, 'new_upstream_training': False,
        'historical_quality_policy': 'frozen-upstream development feature; retain original false nested/use flags',
        'whole_system_oof': False, 'hyperparameter_search': False,
        'smoke_rule': 'fold0, first hash-ranked geometry per source per role; first hash-ranked hypothesis; all three score arms',
    }


def order_key(*parts):
    return digest([SEED, *parts])


def stream(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class VisualIndex:
    """Compact numeric arrays, with nulls preserved, instead of 452k Python dicts."""
    def __init__(self, rows, population, class_names):
        self.values = {}; self.evidence_hashes = {}; self.seen = {}
        if len(class_names) != 198 or len(set(class_names)) != 198:
            raise ValueError('unique198 vocabulary required')
        for row in rows:
            key = row['geometry_key']; pop = population[key]
            cid = integer(row['class_index'], 'class_index', maximum=197)
            if (row['scene_name'] != pop['scene_name'] or row['fold_index'] != pop['fold_index'] or
                    row['geometry_sha256'] != pop['points_sha256'] or row['class_name'] != class_names[cid]):
                raise ValueError('visual identity/vocabulary mismatch')
            if key not in self.values:
                self.values[key] = np.full((198, len(VISUAL_FEATURES)), np.nan)
                self.seen[key] = set(); self.evidence_hashes[key] = row['evidence_sha256']
            if cid in self.seen[key]:
                raise ValueError('duplicate visual geometry-class')
            if row['evidence_sha256'] != self.evidence_hashes[key]:
                raise ValueError('mixed geometry evidence provenance')
            self.seen[key].add(cid)
            f = row['features']
            if set(f) != set(VISUAL_FEATURES) or f['independent_geometry_quality'] is not None:
                raise ValueError('visual schema/quality placeholder differs')
            vals = []
            for name in VISUAL_FEATURES:
                v = f[name]
                if name.endswith('_available') or name.startswith('source_') or name == 'b1_geometry_present':
                    if type(v) is not bool:
                        raise ValueError('visual availability/source flag must be boolean')
                    vals.append(float(v))
                else:
                    vals.append(np.nan if v is None else finite(v, name))
            for c in CHANNELS:
                support = f[c + '_class_support']
                if f[c + '_available'] != (support is not None):
                    raise ValueError('channel availability and support disagree')
                if support is not None and not (-1 if c == 'alpha' else 0) <= support <= 1:
                    raise ValueError('semantic support outside native range')
            self.values[key][cid] = vals
        for key, seen in self.seen.items():
            if seen != set(range(198)):
                raise ValueError('complete unique198 visual rows required')
            for c in CHANNELS:
                if len(set(self.column(key, c + '_available'))) != 1:
                    raise ValueError('availability differs across classes')

    def column(self, key, name):
        return self.values[key][:, VISUAL_FEATURES.index(name)]

    def features(self, key, cid):
        return {name: None if np.isnan(v) else float(v) for name, v in zip(VISUAL_FEATURES, self.values[key][cid])}

    def informative(self, key, channel):
        v = self.column(key, channel + '_class_support')
        return bool(self.column(key, channel + '_available')[0] and np.isfinite(v).all() and np.ptp(v) > 0)


def representative_geometries(population, b1, visual, split, quota=2):
    integer(quota, 'quota', minimum=1)
    fold_of = {s: i for i, scenes in enumerate(split['heldout_folds']) for s in scenes}
    eligible_members = {(r['geometry_key'], r['source']) for r in b1 if r['class_index'] < 198}
    buckets = defaultdict(list)
    for key, p in population.items():
        if key not in visual.values or p['point_count'] < 100:
            continue
        if p['scene_name'] not in fold_of:
            raise ValueError('population scene outside split')
        for source in SOURCES:
            if source not in p['sources']:
                continue
            if source in ('track', 'pair_union'):
                eligible = not p['is_new_geometry'] and (key, source) in eligible_members
            else:
                eligible = p['is_new_geometry'] and not p['b1_members'] and any(visual.informative(key, c) for c in CHANNELS)
            if eligible:
                buckets[(fold_of[p['scene_name']], source)].append(key)
    chosen = []; inventory = []
    for fold in range(5):
        for source in SOURCES:
            groups = defaultdict(list)
            for key in buckets[(fold, source)]:
                groups[main_scene(population[key]['scene_name'])].append(key)
            for group in groups.values():
                group.sort(key=lambda k: order_key(fold, source, k))
            scenes = sorted(groups, key=lambda s: order_key(fold, source, s))
            selected = []
            for depth in range(max(map(len, groups.values()), default=0)):
                for scene in scenes:
                    if depth < len(groups[scene]) and len(selected) < quota:
                        selected.append(groups[scene][depth])
                if len(selected) == quota:
                    break
            chosen.extend({'geometry_key': key, 'source_stratum': source, 'split_fold_index': fold} for key in selected)
            inventory.append({'split_fold_index': fold, 'source': source, 'eligible_geometry_count': len(buckets[(fold, source)]),
                              'eligible_main_scene_count': len(groups), 'selected_count': len(selected), 'quota': quota})
    if len({r['geometry_key'] for r in chosen}) != len(chosen):
        raise ValueError('geometry selected into multiple source strata')
    return chosen, inventory


def parent_index(aliases, population):
    unique(aliases, 'alias_key')
    result = defaultdict(list)
    for a in aliases:
        child = population[a['geometry_key']]; parent = population[a['parent_geometry_key']]
        if (child['scene_name'] != parent['scene_name'] or a['scene_name'] != child['scene_name'] or
                a['fold_index'] != child['fold_index'] or parent['fold_index'] != child['fold_index'] or
                a['points_sha256'] != child['points_sha256'] or a['point_count'] != child['point_count'] or
                a['alias_key'] not in child['refinement_aliases']):
            raise ValueError('parent/alias identity mismatch')
        result[a['geometry_key']].append(a)
    for key, p in population.items():
        if set(p['refinement_aliases']) != {a['alias_key'] for a in result[key]}:
            raise ValueError('missing or extra refinement alias')
    return result


def candidates_from_selection(selected, population, b1, visual, parents):
    by_geometry = defaultdict(list)
    for r in b1:
        if r['class_index'] < 198:
            by_geometry[r['geometry_key']].append(r)
    result = []
    for s in selected:
        key = s['geometry_key']; p = population[key]
        if not p['is_new_geometry']:
            for row in by_geometry[key]:
                if row['source'] == s['source_stratum']:
                    result.append({**row, 'fold_index': p['fold_index'], 'source_stratum': s['source_stratum'],
                                   'hypothesis_reasons': ['existing_b1_member']})
            continue
        reasons = defaultdict(list)
        for channel in CHANNELS:
            if not visual.informative(key, channel):
                continue
            scores = visual.column(key, channel + '_class_support')
            ranked = sorted(range(198), key=lambda c: (-scores[c], order_key(key, channel, c)))
            for cid in ranked[:2]:
                # Zero-vote YOLO classes provide no positive support.
                if channel == 'alpha' or scores[cid] > 0:
                    reasons[cid].append(channel + '_top2')
        for a in parents[key]:
            for row in by_geometry[a['parent_geometry_key']]:
                reasons[row['class_index']].append('parent_b1_class')
        remaining = set(range(198)) - set(reasons)
        if remaining:
            cid = min(remaining, key=lambda c: order_key(key, 'control', c))
            reasons[cid].append('hash_control')
        for cid in sorted(reasons):
            result.append({'prediction_id': f'new:{key}:{cid}', 'scene_name': p['scene_name'],
                           'geometry_key': key, 'class_index': cid, 'source': p['sources'],
                           'fold_index': p['fold_index'], 'is_new_geometry': True,
                           'source_stratum': s['source_stratum'], 'hypothesis_reasons': sorted(set(reasons[cid]))})
    unique(result, 'prediction_id')
    return result


def verify_quality(keys, metadata, source_rows, population, get_points):
    by_geo = unique(metadata, 'geometry_key'); by_row = unique(source_rows, 'row_id')
    result = {}
    for key in sorted(keys):
        q = by_geo.get(key)
        if q is None:
            result[key] = None
            continue
        p = population[key]; source = by_row[q['source_row_id']]
        value = finite(q['legacy_stage_a_quality'], 'quality')
        if (not 0 <= value <= 1 or q['scene_name'] != p['scene_name'] or q['fold_index'] != p['fold_index'] or
                q['scene_name'] != source['scene_name'] or q['fold_index'] != source['fold_index'] or
                value != source['oof_unified_quality'] or q['candidate_source'] != source['candidate_source'] or
                q['candidate_source'] not in p['sources']):
            raise ValueError('quality provenance mismatch')
        if hashlib.sha1(get_points(key).tobytes()).hexdigest() != source['geometry_hash']:
            raise ValueError('quality not bound to exact points')
        result[key] = dict(q)
    return result


def expanded_features(candidates, population, b1, visual, parents, quality, get_points):
    by_geometry = defaultdict(list)
    for r in b1:
        if r['class_index'] < 198:
            by_geometry[r['geometry_key']].append(r)
    relations = {}; result = {}
    for c in candidates:
        key = c['geometry_key']; cid = c['class_index']
        f = visual.features(key, cid)
        if key not in relations:
            rows = []
            for a in parents[key]:
                pk = a['parent_geometry_key']; cp = get_points(key); pp = get_points(pk)
                intersection = len(np.intersect1d(cp, pp, assume_unique=True))
                removed = len(pp) - intersection; added = len(cp) - intersection
                if removed != a['removed_point_count'] or bool(np.array_equal(cp, pp)) != a['same_as_parent']:
                    raise ValueError('parent region differs from actual point sets')
                rows.append({'alias_key': a['alias_key'], 'parent_geometry_key': pk,
                             'retained_fraction': intersection / len(pp), 'removed_fraction': removed / len(pp),
                             'added_fraction': added / len(cp), 'intersection': intersection,
                             'removed_point_count': removed, 'added_point_count': added,
                             'accepted_by_historical_guard': a['accepted_by_historical_guard'],
                             'final_status': a['final_status']})
            relations[key] = rows
        # Deduplicate actual parents for evidence aggregation; keep every alias in provenance.
        rs = list({r['parent_geometry_key']: r for r in relations[key]}.values())
        pkeys = [r['parent_geometry_key'] for r in rs]
        for measure in ('retained_fraction', 'removed_fraction', 'added_fraction'):
            values = [r[measure] for r in rs]
            f['parent_' + measure + '_min'] = min(values, default=None)
            f['parent_' + measure + '_max'] = max(values, default=None)
        # Existing retention exports are checked, not silently trusted/replaced.
        for suffix in ('min', 'max'):
            expected = f['parent_retained_fraction_' + suffix]
            actual = f['refinement_parent_retention_' + suffix]
            if actual is not None and (expected is None or not np.isclose(actual, expected, rtol=0, atol=1e-12)):
                raise ValueError('visual parent retention differs from actual regions')
        f['parent_geometry_count'] = len(pkeys)
        q = quality[key]
        f['independent_geometry_quality'] = None if q is None else q['legacy_stage_a_quality']
        pq = [quality[k]['legacy_stage_a_quality'] for k in pkeys if quality[k] is not None]
        f['parent_quality_max'] = max(pq, default=None)
        f['parent_quality_available_count'] = len(pq)
        ps = [r['score'] for k in pkeys for r in by_geometry[k] if r['class_index'] == cid]
        f['parent_same_class_member_count'] = len(ps)
        f['parent_same_class_max_b1_score'] = max(ps, default=None)
        for channel in CHANNELS:
            evidence = [visual.features(k, cid)[channel + '_class_support'] for k in pkeys if k in visual.values]
            evidence = [v for v in evidence if v is not None]
            f['parent_' + channel + '_available_count'] = len(evidence)
            f['parent_' + channel + '_missing_count'] = len(pkeys) - len(evidence)
            f['parent_' + channel + '_support_max'] = max(evidence, default=None)
            own = f[channel + '_class_support']
            f['child_minus_parent_' + channel + '_support'] = None if own is None or not evidence else own - max(evidence)
        # Uniform schema: every base column has an explicit missing flag, including qualities.
        f.update({name + '_missing': float(value is None) for name, value in list(f.items())})
        result[c['prediction_id']] = {'features': f, 'evidence_sha256': visual.evidence_hashes[key],
            'quality_metadata': q, 'parent_relations': relations[key],
            'parent_quality_metadata': {k: quality[k] for k in pkeys},
            'parent_evidence_sha256': {k: visual.evidence_hashes.get(k) for k in pkeys}}
    return result


def fit_score_anchors(predictions, fit_scenes):
    scores = defaultdict(list); all_scores = []
    for row in predictions:
        if row['scene_name'] not in fit_scenes:
            raise ValueError('score anchors must receive only fit predictions')
        if row['class_index'] < 198:
            value = finite(row['score'])
            if not 0 <= value <= 1:
                raise ValueError('B1 score outside action grid bounds')
            scores[row['class_index']].append(value); all_scores.append(value)
    if not all_scores:
        raise ValueError('empty fit foreground scores')
    return {str(cid): {'scores': [float(np.float32(v)) for v in np.quantile(scores.get(cid, all_scores), [.25, .5, .75], method='linear')],
                       'same_class_count': len(scores[cid]), 'fallback_all_fit_foreground': cid not in scores or not scores[cid]}
            for cid in range(198)}


def action_requests(candidates, arm, anchors):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    result = []
    for c in candidates:
        new = c.get('is_new_geometry', False)
        if arm.startswith('new_') != new:
            continue
        row = dict(c)
        if new:
            index = ('new_q25', 'new_q50', 'new_q75').index(arm)
            row.update(action='append_new', score=anchors[str(c['class_index'])]['scores'][index],
                       candidate_score_source='outer_fit_same_class_B1_quantile')
        else:
            offset = {'existing_down': -.10, 'existing_keep': 0, 'existing_up': .10}[arm]
            new_score = c['score'] if offset == 0 else float(np.float32(np.clip(c['score'] + offset, 0, 1)))
            row.update(action='rescore_existing', new_score=new_score)
        row.update(score_arm=arm, action_id=digest([arm, c['prediction_id']]))
        result.append(row)
    unique(result, 'prediction_id')
    return result


def action_features(requests, b1, edges, population, expanded):
    contexts, peers = overlapping_context(requests, b1, edges, population)
    by_id = unique(b1, 'prediction_id')
    class_scores = defaultdict(list)
    for r in b1:
        class_scores[r['class_index']].append(r)
    result = []
    for c in requests:
        pid = c['prediction_id']; proposed = c.get('new_score', c['score'])
        same = [r['score'] for r in class_scores[c['class_index']] if r['prediction_id'] != pid]
        peer_scores = [by_id[x]['score'] for x in peers[pid]]
        f = {**expanded[pid]['features'], **contexts[pid],
             'candidate_original_score': None if c['action'] == 'append_new' else c['score'],
             'candidate_original_score_missing': float(c['action'] == 'append_new'),
             'proposed_score': proposed, 'action_is_append': float(c['action'] == 'append_new'),
             'score_change': None if c['action'] == 'append_new' else proposed - c['score'],
             'score_change_missing': float(c['action'] == 'append_new'),
             'same_class_reference_member_count': len(same),
             'same_class_reference_higher_proposed_fraction': sum(s > proposed for s in same) / max(1, len(same)),
             'same_class_reference_equal_proposed_count': sum(s == proposed for s in same),
             'same_class_overlap_higher_proposed_count': sum(s > proposed for s in peer_scores)}
        f['same_class_overlap_max_score_missing'] = float(f['same_class_overlap_max_score'] is None)
        result.append({'request': c, 'features': f, 'overlapping_b1_prediction_ids': peers[pid], 'labels': None})
    return result

"""Fixed R0/R1 direct semantic scoring; no learned weights or GT access.

Evidence acquisition must be complete even when a modality has no usable view.
Missing/uncomputed evidence is never silently treated as a valid zero vector.
"""
from collections import Counter

import numpy as np


PROTOCOL = {
    'version': 'fi1_r0_r1_direct_semantics_v1',
    'foreground_class_count': 198,
    'background_class_indexes': [-1, 198],
    'yolo': 'own-mask label-map votes / sum of foreground votes; frequency 10, top 40 visible views',
    'alpha': 'own-mask full 198 cosine vector; softmax(100 * cosine); 20 views, SAM ViT-B, three scales 0/.2/.4',
    'fusion': 'arithmetic mean of available modality distributions; zero if both acquired but unavailable',
    'R0': 'all legacy foreground classes and refined frozen classes; preserve original alias ledger',
    'R1': 'R0 plus every positive-score foreground class for geometry having a nonnative alias',
    'native_only_geometry_expansion': False,
    'duplicate_policy': 'one output per exact point-set/class; all original aliases retained separately',
    'scene_budget': 600,
    'tie_rule': 'descending float64 score, existing native alias first, geometry SHA256 ascending, class index ascending',
    'zero_score_legacy_foreground': 'retain as eligible until common budget; no foreground sentinel invented',
    'empty_geometry': 'ledger retained; no eligible evaluator output',
    'min_mask_points': 'no extra threshold; evaluator uses its existing minimum size',
    'nms': False,
    'geometry_quality_in_score': False,
    'qwen_used': False,
    'requires_complete_yolo_and_alpha_acquisition': True,
    'upstream_interpretation': 'fixed upstream development; not whole-system OOF',
}


def direct_score(evidence):
    """Return the same full vector regardless of R0/R1 class membership."""
    vectors = []
    for modality in ('yolo', 'alpha'):
        item = evidence[modality]
        if item.get('acquisition_complete') is not True:
            raise ValueError(f'{modality} acquisition incomplete')
        if type(item.get('available')) is not bool:
            raise ValueError('modality availability must be explicit')
        if not item['available']:
            if not item.get('missing_reason'):
                raise ValueError('unavailable modality lacks a reason')
            continue
        field = 'votes' if modality == 'yolo' else 'cosine'
        values = np.asarray(item[field], dtype=np.float64)
        if values.shape != (198,) or not np.isfinite(values).all():
            raise ValueError('invalid full-vocabulary evidence vector')
        if modality == 'yolo':
            if (values < 0).any() or (values != np.floor(values)).any() or values.sum() <= 0:
                raise ValueError('available YOLO votes must be positive integer counts')
            vector = values / values.sum()
        else:
            if (np.abs(values) > 1 + 1e-5).any():
                raise ValueError('Alpha cosine outside [-1, 1]')
            vector = np.exp(100. * (values - values.max()))
            vector /= vector.sum()
        vectors.append(vector)
    return np.mean(vectors, axis=0) if vectors else np.zeros(198, dtype=np.float64)


def original_aliases(geometry):
    result = []
    for member in geometry['legacy_members']:
        result.append({'alias_key': f"{member['candidate_source']}:{member['candidate_id']}",
            'source': member['candidate_source'], 'class_index': member['frozen_class_index']})
    for plan in geometry['dv3_plans']:
        if plan['append_only']:
            result.append({'alias_key': plan['plan_key'], 'source': 'refined_union',
                           'class_index': plan['frozen_class_index']})
    keys = [r['alias_key'] for r in result]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate original alias within geometry')
    if any(type(r['class_index']) is not int or not -1 <= r['class_index'] <= 198 for r in result):
        raise ValueError('invalid original class index')
    if any(r['source'] not in {'native', 'track', 'pair_union', 'refined_union'} for r in result):
        raise ValueError('unknown candidate source')
    return result


def build_pair(geometries, evidence_by_key, budget=600):
    """Build both arms at once to enforce shared evidence, geometry and scores."""
    if type(budget) is not int or budget < 1:
        raise ValueError('positive integer scene budget required')
    keys = [g['geometry_key'] for g in geometries]
    if len(keys) != len(set(keys)) or set(keys) != set(evidence_by_key):
        raise ValueError('geometry/evidence coverage differs')
    if len({k.split(':geometry:')[0] for k in keys}) > 1:
        raise ValueError('budget must be applied separately per scene')
    output = {'R0': [], 'R1': []}
    native_alias_keys, sentinel_count, empty_count = set(), 0, 0
    for geometry in sorted(geometries, key=lambda g: g['geometry_key']):
        key = geometry['geometry_key']
        evidence = evidence_by_key[key]
        if evidence.get('geometry_key') != key or evidence.get('points_sha256') != geometry['points_sha256']:
            raise ValueError('own evidence geometry identity differs')
        scores = direct_score(evidence)
        aliases = original_aliases(geometry)
        sentinel_count += sum(a['class_index'] in (-1,198) for a in aliases)
        if geometry['point_count'] == 0:
            empty_count += 1
            continue
        for a in aliases:
            if a['source'] == 'native' and 0 <= a['class_index'] < 198:
                native_alias_keys.add((key, a['alias_key']))
        existing = {a['class_index'] for a in aliases if 0 <= a['class_index'] < 198}
        expanded = set(existing)
        if any(a['source'] != 'native' for a in aliases):
            expanded.update(np.flatnonzero(scores > 0).tolist())
        for arm, classes in (('R0', existing), ('R1', expanded)):
            for cls in sorted(classes):
                original = [a for a in aliases if a['class_index'] == cls]
                native = [a['alias_key'] for a in original if a['source'] == 'native']
                output[arm].append({'geometry_key': key, 'class_index': cls,
                    'score': float(scores[cls]), 'has_original_native_alias': bool(native),
                    'native_alias_keys': native, 'original_aliases': original,
                    'expanded_class': cls not in existing})
    result = {}
    for arm, eligible in output.items():
        ranked = sorted(eligible, key=lambda r: (-r['score'], not r['has_original_native_alias'],
                                               r['geometry_key'], r['class_index']))
        selected = ranked[:budget]
        kept_native = {(r['geometry_key'], a) for r in selected for a in r['native_alias_keys']}
        pre_native_geometry = {k for k, _ in native_alias_keys}
        post_native_geometry = {k for k, _ in kept_native}
        result[arm] = {'eligible': ranked, 'selected': selected, 'audit': {
            'eligible_class_members': len(ranked), 'selected_class_members': len(selected),
            'eligible_geometry_count': len({r['geometry_key'] for r in ranked}),
            'selected_geometry_count': len({r['geometry_key'] for r in selected}),
            'eligible_native_alias_count': len(native_alias_keys),
            'selected_native_alias_count': len(kept_native),
            'budget_evicted_native_alias_count': len(native_alias_keys - kept_native),
            'eligible_native_geometry_count': len(pre_native_geometry),
            'selected_native_geometry_count': len(post_native_geometry),
            'budget_evicted_native_geometry_count': len(pre_native_geometry-post_native_geometry),
            'legacy_background_sentinel_alias_count': sentinel_count,
            'empty_geometry_count': empty_count,
            'selected_expanded_classes': sum(r['expanded_class'] for r in selected),
            'selected_original_alias_sources': dict(Counter(a['source'] for r in selected for a in r['original_aliases']))}}
    base = {(r['geometry_key'],r['class_index']): r['score'] for r in result['R0']['eligible']}
    expanded = {(r['geometry_key'],r['class_index']): r['score'] for r in result['R1']['eligible']}
    if any(expanded.get(k) != score for k, score in base.items()):
        raise ValueError('R0/R1 shared-member scores differ')
    return result

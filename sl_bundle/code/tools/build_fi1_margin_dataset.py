#!/usr/bin/env python3
"""Build a diagnostic bundle with separate features, identities and GT labels.

Only v2 full-reference evaluator labels are accepted. This deliberately does
not declare training readiness: official100 reference GT spans held-out folds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fi1_evaluator_delta_core import (
    VERSION, ACTIONS, IDENTITY, finite, identity_equal, indexed_b1, integer,
    load_population, validate_prediction,
)
from tools.fi1_gain_attribution_core import Bindings, read_rows, sha, unique, write_json

FEATURES = ('alpha_class_support', 'label_map_class_support', 'bbox_class_support',
            'alpha_gap_to_own_max', 'maximum_geometry_overlap_iou', 'log1p_geometric_neighbor_count')


def extract_visual_features(candidates, visual_rows, population, class_names):
    """Stream large evidence tables; require exact198 identities for selected geometries."""
    unique(candidates, 'prediction_id')
    needed = {r['geometry_key'] for r in candidates}
    members = {}
    for row in visual_rows:
        key = row['geometry_key']
        if key not in needed:
            continue
        pop = population[key]
        cid = integer(row['class_index'], 'class_index', maximum=197)
        pair = (key, cid)
        if pair in members:
            raise ValueError('duplicate visual geometry-class key')
        if (row['scene_name'] != pop['scene_name'] or row['fold_index'] != pop['fold_index'] or
                row['geometry_sha256'] != pop['points_sha256'] or row['class_name'] != class_names[cid]):
            raise ValueError('visual identity/vocabulary differs from population')
        members[pair] = row
    result = {}
    for key in needed:
        if {cid for g, cid in members if g == key} != set(range(198)):
            raise ValueError('complete unique198 visual class rows required')
    for candidate in candidates:
        validate_prediction(candidate, foreground=True)
        integer(candidate['fold_index'], 'fold_index', maximum=4)
        key = candidate['geometry_key']
        pop = population[key]
        if candidate['scene_name'] != pop['scene_name'] or candidate['fold_index'] != pop['fold_index']:
            raise ValueError('candidate/population scene or fold differs')
        row = members[(key, candidate['class_index'])]
        nested = row['features']
        # Explicit allowlist: label, score, IDs and GT statistics never become features.
        values = {name: finite(nested.get(name), name) for name in FEATURES}
        result[candidate['prediction_id']] = values
    return result


def validate_labels(candidates, deltas, summary):
    c = unique(candidates, 'prediction_id')
    d = unique(deltas, 'prediction_id')
    if set(c) != set(d):
        raise ValueError('candidate/delta identity sets differ')
    if summary.get('version') != VERSION or summary.get('evaluation_scope') != 'full_reference_scenes':
        raise ValueError('v2 full-reference labels required; legacy local AP is rejected')
    if summary.get('baseline_array_exactly_replayed') is not True:
        raise ValueError('reference control was not replayed')
    if summary.get('action') not in ACTIONS or not summary.get('reference_id'):
        raise ValueError('invalid reference/action')
    scenes = summary['reference_scenes']
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise ValueError('full official100 reference required')
    expected_thresholds = list(np.append(np.arange(.5, .95, .05), .25))
    if summary['thresholds'] != expected_thresholds:
        raise ValueError('evaluator thresholds differ')
    for pid, row in d.items():
        request = c[pid]
        validate_prediction(request, foreground=True)
        if request['scene_name'] not in scenes:
            raise ValueError('candidate scene outside label reference')
        if not identity_equal(request, row):
            raise ValueError('candidate/delta identity differs')
        if (row['version'] != VERSION or row['action'] != summary['action'] or
                row['evaluation_scope'] != summary['evaluation_scope'] or
                row['reference_scene_count'] != len(scenes) or row['reference_id'] != summary['reference_id']):
            raise ValueError('delta reference/action differs')
        score = request['new_score'] if row['action'] == 'rescore_existing' else request['score']
        if row['score_after'] != score:
            raise ValueError('candidate evaluated score differs')
        before_score = None if row['action'] == 'append_new' else request['score']
        if row['score_before'] != before_score:
            raise ValueError('candidate original score differs')
        before_count = integer(row['before_count'], 'before_count')
        after_count = integer(row['after_count'], 'after_count')
        if after_count - before_count != (0 if row['action'] == 'rescore_existing' else 1):
            raise ValueError('action member counts differ')
        for name in ('class_ap', 'class_ap50', 'overall_ap', 'overall_ap50'):
            b, a, delta = (row[k][name] for k in ('before', 'after', 'delta'))
            if b is None or a is None:
                if b is not None or a is not None or delta is not None:
                    raise ValueError('undefined AP must have undefined delta')
            else:
                b, a, delta = finite(b, name), finite(a, name), finite(delta, name)
                if not (0 <= b <= 1 and 0 <= a <= 1) or delta != a - b:
                    raise ValueError('invalid AP/delta values')
    return d


def overlapping_context(candidates, b1, edges, population):
    """No-GT spatial competition from real point-set overlap and B1 scores."""
    needed = {r['geometry_key'] for r in candidates}
    neighbors = {key: {key: 1.0} for key in needed}
    seen = set()
    for edge in edges:
        left, right = edge['left'], edge['right']
        if left not in needed and right not in needed:
            continue
        key = tuple(sorted((left, right)))
        if key in seen or left == right:
            raise ValueError('duplicate/invalid overlap edge')
        seen.add(key)
        lp, rp = population[left], population[right]
        if lp['scene_name'] != rp['scene_name'] or edge['scene_name'] != lp['scene_name']:
            raise ValueError('overlap edge crosses scenes')
        count = integer(edge['intersection'], 'intersection', minimum=1)
        if count > min(lp['point_count'], rp['point_count']):
            raise ValueError('overlap intersection exceeds geometry')
        value = finite(edge['iou'], 'edge_iou')
        expected = count / (lp['point_count'] + rp['point_count'] - count)
        if value != expected:
            raise ValueError('overlap IoU differs from point counts')
        if left in needed:
            neighbors[left][right] = value
        if right in needed:
            neighbors[right][left] = value
    groups = defaultdict(list)
    for row in b1:
        groups[(row['scene_name'], row['class_index'])].append(row)
    features = {}
    identities = {}
    for c in candidates:
        peers = [r for r in groups[(c['scene_name'], c['class_index'])]
                 if r['prediction_id'] != c['prediction_id'] and r['geometry_key'] in neighbors[c['geometry_key']]]
        overlaps = [neighbors[c['geometry_key']][r['geometry_key']] for r in peers]
        scores = [finite(r['score']) for r in peers]
        features[c['prediction_id']] = {
            'same_class_overlap_member_count': len(peers),
            'same_class_overlap_max_iou': max(overlaps, default=0.0),
            'same_class_overlap_max_score': max(scores, default=None),
            'same_class_overlap_higher_score_count': sum(s > c['score'] for s in scores),
            'same_class_overlap_equal_score_count': sum(s == c['score'] for s in scores),
        }
        identities[c['prediction_id']] = [r['prediction_id'] for r in peers]
    return features, identities


def build_bundle(candidates, deltas, summary, visual_features, context, peer_ids):
    labels = validate_labels(candidates, deltas, summary)
    rows = []
    for c in candidates:
        pid = c['prediction_id']
        d = labels[pid]
        features = {**visual_features[pid], **context[pid], 'candidate_input_score': finite(c['score'])}
        if d['action'] == 'rescore_existing':
            features['proposed_score'] = finite(c['new_score'])
        rows.append({'identity': {**{k: c[k] for k in IDENTITY}, 'fold_index': c['fold_index'],
                                  'source': c['source'], 'overlapping_b1_prediction_ids': peer_ids[pid]},
                     'features': features,
                     'labels': {k: d[k] for k in ('action', 'reference_id', 'score_before', 'score_after', 'before', 'after', 'delta')}})
    return {'version': 'fi1_margin_diagnostic_dataset_v2', 'rows': rows, 'row_count': len(rows),
            'required_feature_nonnull_counts': {k: sum(r['features'][k] is not None for r in rows) for k in FEATURES},
            'scene_count': len({c['scene_name'] for c in candidates}),
            'fold_counts': {str(f): sum(c['fold_index'] == f for c in candidates) for f in range(5)},
            'label_scope': 'full_official100_development_reference',
            'gt_used_for_features': False, 'feature_contract': 'explicit no-GT field allowlist; audited frozen inputs',
            'training_ready': False, 'model_fitted': False, 'production_ready': False,
            'readiness_reasons': ['no scene-isolated training/held-out label reference pools',
                'historical supervised upstream reused, not rebuilt OOF',
                'independent quality and parent-region evidence not fully consumed',
                'candidate score and sampling policy not yet calibrated/frozen for training']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'deltas', 'delta-summary', 'visual', 'b1', 'population-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bind = Bindings()
    summary = json.loads(bind.check(args.delta_summary).read_text())
    state = json.loads(bind.check(args.delta_summary.parent / 'run_state.json').read_text())
    if state['status'] != 'complete' or state['summary_sha256'] != sha(args.delta_summary):
        raise ValueError('evaluator run incomplete or summary changed')
    candidates = read_rows(bind.check(args.candidates, summary['hashes']['actions.jsonl']))
    deltas = read_rows(bind.check(args.deltas, summary['hashes']['evaluator_delta.jsonl']))
    validate_labels(candidates, deltas, summary)
    b1path = str(args.b1.resolve())
    if b1path not in summary['input_sha256']:
        raise ValueError('B1 not bound to evaluator labels')
    b1 = indexed_b1(read_rows(bind.check(args.b1, summary['input_sha256'][b1path])))
    for row in deltas:
        expected_before = len(b1) - (row['action'] == 'restore_existing')
        expected_after = len(b1) + (row['action'] == 'append_new')
        if (row['before_count'], row['after_count']) != (expected_before, expected_after):
            raise ValueError('label member counts differ from bound B1')
    pop, pop_summary = load_population(args.population_root, bind)
    pop_path = str((args.population_root / 'population.jsonl').resolve())
    if summary['input_sha256'].get(pop_path) != bind.files[pop_path]:
        raise ValueError('population differs from evaluated geometry source')
    vs = json.loads(bind.check(args.visual.parent / 'summary.json').read_text())
    visual_path = bind.check(args.visual, vs['hashes'][args.visual.name])
    with visual_path.open() as handle:
        features = extract_visual_features(candidates, (json.loads(s) for s in handle if s.strip()), pop, pop_summary['class_names'])
    edges = read_rows(bind.check(args.population_root / 'geometry_overlap_edges.jsonl', pop_summary['hashes']['geometry_overlap_edges.jsonl']))
    context, peer_ids = overlapping_context(candidates, b1, edges, pop)
    result = build_bundle(candidates, deltas, summary, features, context, peer_ids)
    result.update(input_sha256=bind.files, code_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__),
                  'tools/fi1_evaluator_delta_core.py': sha(ROOT / 'tools/fi1_evaluator_delta_core.py')})
    # Exclusive output creation preserves every historical artifact.
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'input_sha256', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    main()

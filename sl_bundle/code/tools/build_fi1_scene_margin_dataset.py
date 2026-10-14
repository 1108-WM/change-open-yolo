#!/usr/bin/env python3
"""Join one partition's action labels and no-GT features without crossing folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_fi1_margin_dataset import FEATURES, extract_visual_features, overlapping_context
from tools.fi1_evaluator_delta_core import ACTIONS, IDENTITY, finite, identity_equal, integer, validate_prediction
from tools.fi1_gain_attribution_core import Bindings, read_rows, sha, unique
from tools.fi1_scene_reference import LABEL_VERSION, ROLES, validate_reference, validate_partition_requests

DATASET_VERSION = 'fi1_scene_margin_dataset_v1'
METRICS = ('class_ap', 'class_ap50', 'overall_ap', 'overall_ap50')


def validate_scoped_labels(candidates, labels, summary, ref, expected_role):
    scenes = set(validate_reference(ref))
    if expected_role not in ROLES or ref['role'] != expected_role:
        raise ValueError('role mismatch: heldout labels are evaluation-only')
    if summary.get('version') != LABEL_VERSION or summary.get('evaluation_scope') != 'partition_reference_scenes':
        raise ValueError('partition labels required; full100/local AP labels cannot be reused')
    fields = {'reference_id': ref['reference_id'], 'split_id': ref['split_id'],
              'outer_fold_index': ref['outer_fold_index'], 'partition_role': ref['role']}
    if any(summary.get(k) != v for k, v in fields.items()) or summary['reference_scenes'] != ref['reference_scenes']:
        raise ValueError('label summary belongs to another partition')
    if summary['thresholds'] != list(np.append(np.arange(.5, .95, .05), .25)):
        raise ValueError('evaluator thresholds differ')
    expected_usage = 'fit_only' if expected_role == 'fit' else 'evaluation_only'
    if summary.get('label_usage') != expected_usage:
        raise ValueError('invalid label usage')
    c = unique(candidates, 'prediction_id'); d = unique(labels, 'prediction_id')
    if set(c) != set(d):
        raise ValueError('candidate/label identity sets differ')
    count = integer(summary['reference_prediction_count'], 'reference_prediction_count')
    for pid, label in d.items():
        candidate = c[pid]
        validate_prediction(candidate, foreground=True)
        if candidate['scene_name'] not in scenes or not identity_equal(candidate, label):
            raise ValueError('candidate/label scene or identity differs')
        if label.get('version') != LABEL_VERSION or label.get('evaluation_scope') != summary['evaluation_scope']:
            raise ValueError('non-partition label in dataset')
        if (any(label.get(k) != v for k, v in fields.items()) or label.get('label_usage') != expected_usage or
                label.get('reference_scene_count') != len(scenes)):
            raise ValueError('mixed reference/fold/role labels')
        action = candidate['action']
        if action not in ACTIONS or label['action'] != action:
            raise ValueError('candidate action differs')
        expected_before = count - (action == 'restore_existing')
        expected_after = count + (action == 'append_new')
        if (label['before_count'], label['after_count']) != (expected_before, expected_after):
            raise ValueError('action member counts differ')
        before_score = None if action == 'append_new' else candidate['score']
        after_score = candidate['new_score'] if action == 'rescore_existing' else candidate['score']
        finite(after_score)
        if label['score_before'] != before_score or label['score_after'] != after_score:
            raise ValueError('action scores differ')
        for metric in METRICS:
            b, a, delta = (label[k][metric] for k in ('before', 'after', 'delta'))
            if b is None or a is None:
                if b is not None or a is not None or delta is not None:
                    raise ValueError('undefined partition target must remain null')
            else:
                b, a, delta = finite(b), finite(a), finite(delta)
                if not (0 <= b <= 1 and 0 <= a <= 1) or delta != a - b:
                    raise ValueError('invalid AP delta')
    return d


def make_bundle(candidates, labels, summary, ref, expected_role, visual_features, contexts, peer_ids):
    by_id = validate_scoped_labels(candidates, labels, summary, ref, expected_role)
    split_fold = {s: i for i, fold in enumerate(ref['split']['heldout_folds']) for s in fold}
    rows = []
    for c in candidates:
        pid = c['prediction_id']; label = by_id[pid]
        features = {**visual_features[pid], **contexts[pid], 'candidate_input_score': finite(c['score'])}
        if c['action'] == 'rescore_existing':
            features['proposed_score'] = finite(c['new_score'])
        rows.append({'identity': {**{k: c[k] for k in IDENTITY},
            'source': c['source'], 'legacy_fold_index': c['fold_index'],
            'split_fold_index': split_fold[c['scene_name']], 'outer_fold_index': ref['outer_fold_index'],
            'partition_role': expected_role, 'overlapping_b1_prediction_ids': peer_ids[pid]},
            'features': features,
            'labels': {**{k: label[k] for k in ('action', 'reference_id', 'score_before', 'score_after', 'before', 'after', 'delta')},
                       'label_usage': label['label_usage'],
                       'target_status': 'class_absent_from_partition' if label['delta']['class_ap'] is None else 'defined'}})
    return {'version': DATASET_VERSION, 'reference_id': ref['reference_id'], 'split_id': ref['split_id'],
        'outer_fold_index': ref['outer_fold_index'], 'partition_role': expected_role, 'label_usage': summary['label_usage'],
        'rows': rows, 'row_count': len(rows), 'reference_scenes': ref['reference_scenes'],
        'defined_target_count': sum(r['labels']['target_status'] == 'defined' for r in rows),
        'required_feature_nonnull_counts': {k: sum(r['features'][k] is not None for r in rows) for k in FEATURES},
        'label_reference_isolated': True, 'gt_used_for_features': False, 'whole_system_oof': False,
        'training_ready': False, 'model_fitted': False, 'production_ready': False,
        'readiness_reasons': ['small fixed diagnostic sample; representative training sampling not frozen',
            'full independent quality and parent-region evidence not consumed',
            'historical FI1/B1 reused, not end-to-end OOF',
            'original real-scene GT point-level verification still pending',
            'inner tuning requires its own disjoint label references; outer-heldout is evaluation-only']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'label-root', 'visual', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--role', choices=ROLES, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bind = Bindings()
    ref = json.loads(bind.check(args.reference_root / 'reference.json').read_text())
    validate_reference(ref)
    if ref['role'] != args.role:
        raise ValueError('role mismatch before reading labels')
    summary = json.loads(bind.check(args.label_root / 'summary.json').read_text())
    state = json.loads(bind.check(args.label_root / 'run_state.json').read_text())
    if state['status'] != 'complete' or state['summary_sha256'] != sha(args.label_root / 'summary.json'):
        raise ValueError('label run incomplete or modified')
    if summary.get('reference_id') != ref['reference_id'] or summary.get('partition_role') != args.role:
        raise ValueError('label run belongs to another reference/role')
    label_ref = json.loads(bind.check(args.label_root / 'reference.json', summary['hashes']['reference.json']).read_text())
    if label_ref != ref:
        raise ValueError('reference changed since labels were computed')
    candidates = read_rows(bind.check(args.label_root / 'actions.jsonl', summary['hashes']['actions.jsonl']))
    labels = read_rows(bind.check(args.label_root / 'evaluator_delta.jsonl', summary['hashes']['evaluator_delta.jsonl']))
    validate_scoped_labels(candidates, labels, summary, ref, args.role)
    b1 = read_rows(bind.check(args.reference_root / 'predictions.jsonl', ref['predictions_sha256']))
    geos = unique(read_rows(bind.check(args.reference_root / 'geometry_index.jsonl', ref['geometry_index_sha256'])), 'geometry_key')
    if len(b1) != summary['reference_prediction_count']:
        raise ValueError('reference member count changed')
    validate_partition_requests(candidates, b1, geos, ref)
    # Reuse the existing no-GT feature contract with legacy fold as provenance.
    population = {key: {**g, 'fold_index': g['legacy_fold_index']} for key, g in geos.items()}
    visual_summary = json.loads(bind.check(args.visual.parent / 'summary.json').read_text())
    visual = bind.check(args.visual, visual_summary['hashes'][args.visual.name])
    with visual.open() as handle:
        features = extract_visual_features(candidates, (json.loads(s) for s in handle if s.strip()), population, ref['class_names'])
    pop_root = Path(ref['population_root'])
    pop_summary = json.loads(bind.check(pop_root / 'summary.json').read_text())
    edges_path = bind.check(pop_root / 'geometry_overlap_edges.jsonl', pop_summary['hashes']['geometry_overlap_edges.jsonl'])
    # Edge/visual tables have no GT; only selected scenes enter feature extraction.
    with edges_path.open() as handle:
        edges = (r for r in map(json.loads, handle) if r['scene_name'] in set(ref['reference_scenes']))
        contexts, peers = overlapping_context(candidates, b1, edges, population)
    result = make_bundle(candidates, labels, summary, ref, args.role, features, contexts, peers)
    result.update(input_sha256=bind.files, code_sha256={n: sha(ROOT / n) for n in (
        'tools/build_fi1_scene_margin_dataset.py', 'tools/build_fi1_margin_dataset.py', 'tools/fi1_scene_reference.py')})
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with args.output.open('x') as handle:
        handle.write(serialized)
    print(json.dumps({k: result[k] for k in ('outer_fold_index', 'partition_role', 'row_count', 'defined_target_count', 'training_ready')}, indent=2))


if __name__ == '__main__':
    main()

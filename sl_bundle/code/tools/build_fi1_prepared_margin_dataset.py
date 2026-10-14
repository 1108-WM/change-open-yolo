#!/usr/bin/env python3
"""Join frozen action features with labels from exactly their scene reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_fi1_scene_margin_dataset import validate_scoped_labels
from tools.fi1_evaluator_delta_core import IDENTITY, finite
from tools.fi1_gain_attribution_core import Bindings, read_rows, sha, unique
from tools.fi1_margin_preparation import VERSION
from tools.fi1_scene_reference import validate_reference

DATASET_VERSION = 'fi1_prepared_action_dataset_v1'


def join_prepared(features, candidates, labels, summary, ref, manifest, role):
    if (manifest['version'] != VERSION or manifest['reference_id'] != ref['reference_id'] or
            manifest['partition_role'] != role or manifest['outer_fold_index'] != ref['outer_fold_index'] or
            manifest['split_id'] != ref['split_id']):
        raise ValueError('prepared features belong to another reference/role/fold')
    by_id = unique([{'prediction_id': row['request']['prediction_id'], **row} for row in features], 'prediction_id')
    label_by_id = validate_scoped_labels(candidates, labels, summary, ref, role)
    output = []; schema = None
    split_fold = {s: i for i, scenes in enumerate(ref['split']['heldout_folds']) for s in scenes}
    for c in candidates:
        pid = c['prediction_id']; prepared = by_id.get(pid)
        if prepared is None or prepared['request'] != c or c['score_arm'] != manifest['score_arm']:
            raise ValueError('evaluated request/score is not the frozen prepared action')
        if prepared['labels'] is not None:
            raise ValueError('GT labels found in prepared features')
        f = prepared['features']
        if any(k in f for k in ('class_index', 'geometry_key', 'prediction_id', 'label', 'delta', 'target', 'fold_index')):
            raise ValueError('identity/target cannot be a feature')
        for value in f.values():
            if value is not None:
                finite(value, 'prepared feature')
        if schema is None:
            schema = sorted(f)
        elif sorted(f) != schema:
            raise ValueError('inconsistent feature schema')
        label = label_by_id[pid]
        output.append({'identity': {**{k: c[k] for k in IDENTITY},
            'action_id': c['action_id'], 'score_arm': c['score_arm'], 'source_stratum': c['source_stratum'],
            'hypothesis_reasons': c['hypothesis_reasons'], 'legacy_fold_index': c['fold_index'],
            'split_fold_index': split_fold[c['scene_name']],
            'outer_fold_index': ref['outer_fold_index'], 'partition_role': role,
            'overlapping_b1_prediction_ids': prepared['overlapping_b1_prediction_ids']},
            'features': f, 'labels': {**{k: label[k] for k in ('action', 'reference_id', 'score_before', 'score_after', 'before', 'after', 'delta')},
                'label_usage': label['label_usage'], 'target_class_ap_delta': label['delta']['class_ap'],
                'target_status': 'class_absent_from_partition' if label['delta']['class_ap'] is None else 'defined'}})
    return {'version': DATASET_VERSION, 'reference_id': ref['reference_id'], 'split_id': ref['split_id'],
        'outer_fold_index': ref['outer_fold_index'], 'partition_role': role, 'score_arm': manifest['score_arm'],
        'label_usage': summary['label_usage'], 'rows': output, 'row_count': len(output),
        'prepared_arm_row_count': len(features), 'all_prepared_arm_rows_labeled': len(output) == len(features),
        'feature_names': schema or [], 'defined_target_count': sum(r['labels']['target_status'] == 'defined' for r in output),
        'label_reference_isolated': True, 'whole_system_oof': False, 'gt_used_for_selection_or_features': False,
        'historical_supervised_quality_used': True, 'training_ready': False, 'model_fitted': False, 'production_ready': False,
        'limitations': ['pilot action data; sample coverage and joint output evaluation still required',
            'no whole-system OOF; historical supervised FI1/B1 reused',
            'raw point-level real-scene GT association not newly verified',
            'inner tuning needs its own fit/calibration label references', 'individual deltas are not additive']}


def build(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bind = Bindings(); folder = args.prepared_manifest.parent
    # Validate explicit role and partition identity before opening the label payload.
    manifest = json.loads(bind.check(args.prepared_manifest).read_text())
    if manifest['partition_role'] != args.role:
        raise ValueError('role mismatch before reading labels')
    prep_root = folder.parent.parent
    prep_summary = json.loads(bind.check(prep_root / 'summary.json').read_text())
    state = json.loads(bind.check(prep_root / 'run_state.json').read_text())
    if state != {'status': 'complete', 'summary_sha256': sha(prep_root / 'summary.json')}:
        raise ValueError('preparation incomplete or modified')
    bind.check(args.prepared_manifest, prep_summary['hashes'][str(args.prepared_manifest.relative_to(prep_root))])
    ref = json.loads(bind.check(Path(manifest['reference_root']) / 'reference.json').read_text())
    validate_reference(ref)
    if ref['role'] != args.role or manifest['reference_id'] != ref['reference_id']:
        raise ValueError('reference changed since preparation')
    summary = json.loads(bind.check(args.label_root / 'summary.json').read_text())
    label_state = json.loads(bind.check(args.label_root / 'run_state.json').read_text())
    if label_state != {'status': 'complete', 'summary_sha256': sha(args.label_root / 'summary.json')}:
        raise ValueError('label run incomplete or modified')
    label_ref = json.loads(bind.check(args.label_root / 'reference.json', summary['hashes']['reference.json']).read_text())
    if label_ref != ref:
        raise ValueError('label reference differs from preparation')
    features = read_rows(bind.check(folder / manifest['features_file'], manifest['features_sha256']))
    requests = read_rows(bind.check(folder / manifest['requests_file'], manifest['requests_sha256']))
    if [r['request'] for r in features] != requests:
        raise ValueError('feature and request files differ')
    for path, expected in ((prep_root / 'protocol.json', manifest['protocol_sha256']),
            (folder.parent / 'score_anchors.json', manifest['score_anchors_sha256']),
            (prep_root / 'candidate_evidence.jsonl', manifest['candidate_evidence_sha256'])):
        bind.check(path, expected)
    candidates = read_rows(bind.check(args.label_root / 'actions.jsonl', summary['hashes']['actions.jsonl']))
    labels = read_rows(bind.check(args.label_root / 'evaluator_delta.jsonl', summary['hashes']['evaluator_delta.jsonl']))
    result = join_prepared(features, candidates, labels, summary, ref, manifest, args.role)
    result.update(input_sha256=bind.files, code_sha256={n: sha(ROOT / n) for n in (
        'tools/build_fi1_prepared_margin_dataset.py', 'tools/build_fi1_scene_margin_dataset.py', 'tools/fi1_margin_preparation.py')})
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('partition_role', 'score_arm', 'row_count', 'defined_target_count', 'training_ready')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-manifest', 'label-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--role', choices=('fit', 'heldout'), required=True)
    build(p.parse_args())


if __name__ == '__main__':
    main()

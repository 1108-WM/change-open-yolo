"""Load a complete single-role action dataset and audit fit-side target coverage."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from tools.build_fi1_prepared_margin_dataset import DATASET_VERSION
from tools.fi1_evaluator_delta_core import finite
from tools.fi1_gain_attribution_core import Bindings, sha
from tools.fi1_margin_preparation import ARMS
from tools.fi1_scene_reference import validate_reference, main_scene

MANIFEST_VERSION = 'fi1_complete_action_dataset_v1'
FAMILIES = {'existing': ARMS[:3], 'new': ARMS[3:]}
# Reporting tolerance only: preserve original evaluator labels and fit targets.
SIGN_REPORT_ATOL = 1e-12


def validate_complete_rows(rows, ref, role, feature_names):
    if role != ref['role']:
        raise ValueError('dataset role differs from reference')
    groups = defaultdict(list); action_ids = set()
    for row in rows:
        identity, f, label = (row[k] for k in ('identity', 'features', 'labels'))
        pid = identity['prediction_id']; arm = identity['score_arm']
        if (identity['scene_name'] not in ref['reference_scenes'] or identity['partition_role'] != role or
                identity['outer_fold_index'] != ref['outer_fold_index'] or label['reference_id'] != ref['reference_id']):
            raise ValueError('row belongs to another partition')
        if label['label_usage'] != ('fit_only' if role == 'fit' else 'evaluation_only'):
            raise ValueError('invalid target usage')
        if sorted(f) != sorted(feature_names):
            raise ValueError('inconsistent feature columns')
        for value in f.values():
            if value is not None:
                finite(value, 'feature')
        if arm not in ARMS or identity['action_id'] in action_ids:
            raise ValueError('duplicate or unknown action')
        action_ids.add(identity['action_id'])
        if f['proposed_score'] != label['score_after'] or f['candidate_original_score'] != label['score_before']:
            raise ValueError('action scores and features differ')
        target = label['target_class_ap_delta']
        if target != label['delta']['class_ap']:
            raise ValueError('target differs from evaluator delta')
        b, a = label['before']['class_ap'], label['after']['class_ap']
        if target is None:
            if b is not None or a is not None or label['target_status'] != 'class_absent_from_partition':
                raise ValueError('undefined target must stay null')
        elif finite(target) != finite(a) - finite(b) or label['target_status'] != 'defined':
            raise ValueError('invalid target delta')
        family = 'new' if arm.startswith('new_') else 'existing'
        if label['action'] != ('append_new' if family == 'new' else 'rescore_existing'):
            raise ValueError('action/family mismatch')
        if family == 'existing':
            old = finite(label['score_before']); after = finite(label['score_after'])
            offset = {'existing_down': -.10, 'existing_keep': 0, 'existing_up': .10}[arm]
            expected = old if offset == 0 else float(np.float32(np.clip(old + offset, 0, 1)))
            if after != expected:
                raise ValueError('score differs from frozen action grid')
            if old == after and target not in (None, 0):
                raise ValueError('unchanged score has nonzero target')
        elif label['score_before'] is not None:
            raise ValueError('append action has an original B1 score')
        groups[pid].append(row)
    for pid, members in groups.items():
        names = {r['identity']['score_arm'] for r in members}
        family = 'new' if next(iter(names)).startswith('new_') else 'existing'
        if len(members) != 3 or names != set(FAMILIES[family]):
            raise ValueError('candidate does not have exactly its three score actions')
        identity = members[0]['identity']
        for r in members:
            if any(r['identity'][k] != identity[k] for k in ('scene_name', 'geometry_key', 'class_index',
                    'source_stratum', 'legacy_fold_index', 'split_fold_index', 'outer_fold_index')):
                raise ValueError('candidate identity differs across actions')
            if r['labels']['before'] != members[0]['labels']['before']:
                raise ValueError('candidate baseline metrics differ across actions')
        by_score = {}
        for r in members:
            score = r['labels']['score_after']; label = r['labels']
            if score in by_score and label['after'] != by_score[score]:
                raise ValueError('identical score actions have different AP')
            by_score[score] = label['after']
    return groups


def load_complete_dataset(manifest_path, expected_role, bind=None):
    """Reject the wrong role before reading any target-bearing dataset file."""
    bind = bind or Bindings(); path = Path(manifest_path)
    manifest = json.loads(bind.check(path).read_text())
    if manifest.get('version') != MANIFEST_VERSION or manifest.get('partition_role') != expected_role:
        raise ValueError('role/version mismatch before reading target files')
    ref = manifest['reference']; validate_reference(ref)
    if ref['role'] != expected_role or manifest['reference_id'] != ref['reference_id']:
        raise ValueError('manifest reference/role mismatch')
    if set(manifest['datasets']) != set(ARMS):
        raise ValueError('all six complete action datasets required')
    rows = []; schema = None
    for arm in ARMS:
        item = manifest['datasets'][arm]
        if Path(item['file']).name != item['file']:
            raise ValueError('dataset file must be local to manifest')
        data = json.loads(bind.check(path.parent / item['file'], item['sha256']).read_text())
        if (data['version'] != DATASET_VERSION or data['reference_id'] != ref['reference_id'] or
                data['partition_role'] != expected_role or data['outer_fold_index'] != ref['outer_fold_index'] or
                data['split_id'] != ref['split_id'] or data['score_arm'] != arm or
                data['all_prepared_arm_rows_labeled'] is not True or data['row_count'] != len(data['rows']) or
                data['prepared_arm_row_count'] != len(data['rows'])):
            raise ValueError('incomplete or mismatched arm dataset')
        if schema is None:
            schema = data['feature_names']
        elif schema != data['feature_names']:
            raise ValueError('feature schema differs across arms')
        if any(r['identity']['score_arm'] != arm for r in data['rows']):
            raise ValueError('mixed arms in dataset')
        rows.extend(data['rows'])
    validate_complete_rows(rows, ref, expected_role, schema)
    if len(rows) != manifest['row_count']:
        raise ValueError('total action count differs')
    return rows, ref, schema, bind


def effective_training_rows(rows, family):
    """Keep defined changing actions once per candidate/score; no synthetic labels."""
    if family not in FAMILIES:
        raise ValueError('unknown training family')
    output = []; seen = set()
    for r in rows:
        i, label = r['identity'], r['labels']
        if i['score_arm'] not in FAMILIES[family] or label['target_class_ap_delta'] is None:
            continue
        if family == 'existing' and label['score_after'] == label['score_before']:
            continue
        key = (i['prediction_id'], label['score_after'])
        if key not in seen:
            seen.add(key); output.append(r)
    return output


def sign_counts(rows):
    values = [r['labels']['target_class_ap_delta'] for r in rows]
    return {'rows': len(values), 'positive': sum(v is not None and v > 0 for v in values),
            'negative': sum(v is not None and v < 0 for v in values), 'zero': sum(v == 0 for v in values),
            'undefined': sum(v is None for v in values)}


def coverage_report(rows, ref, feature_names):
    result = {'partition_role': ref['role'], 'reference_id': ref['reference_id'],
        'reference_scene_count': len(ref['reference_scenes']), 'actions': sign_counts(rows),
        'candidate_count': len({r['identity']['prediction_id'] for r in rows}),
        'geometry_count': len({r['identity']['geometry_key'] for r in rows}),
        'scan_count': len({r['identity']['scene_name'] for r in rows}),
        'main_scene_count': len({main_scene(r['identity']['scene_name']) for r in rows}),
        'class_count': len({r['identity']['class_index'] for r in rows}),
        'families': {}, 'sources': {}, 'arms': {}, 'feature_count': len(feature_names),
        'missing_counts': {k: sum(r['features'][k] is None for r in rows) for k in feature_names},
        'constant_feature_count': sum(len({r['features'][k] for r in rows}) <= 1 for k in feature_names),
        'training_ready': False, 'production_ready': False, 'whole_system_oof': False}
    for arm in ARMS:
        result['arms'][arm] = sign_counts([r for r in rows if r['identity']['score_arm'] == arm])
    for source in sorted({r['identity']['source_stratum'] for r in rows}):
        result['sources'][source] = sign_counts([r for r in rows if r['identity']['source_stratum'] == source])
    for family in FAMILIES:
        effective = effective_training_rows(rows, family)
        signs = sign_counts(effective)
        positive_scenes = sorted({main_scene(r['identity']['scene_name']) for r in effective if r['labels']['target_class_ap_delta'] > 0})
        negative_scenes = sorted({main_scene(r['identity']['scene_name']) for r in effective if r['labels']['target_class_ap_delta'] < 0})
        signs.update(positive_main_scenes=positive_scenes, negative_main_scenes=negative_scenes,
            main_scene_count=len({main_scene(r['identity']['scene_name']) for r in effective}),
            distinct_targets=len({r['labels']['target_class_ap_delta'] for r in effective}),
            raw_delta_min=min((r['labels']['target_class_ap_delta'] for r in effective), default=None),
            raw_delta_max=max((r['labels']['target_class_ap_delta'] for r in effective), default=None))
        material_positive = [r for r in effective if r['labels']['target_class_ap_delta'] > SIGN_REPORT_ATOL]
        material_negative = [r for r in effective if r['labels']['target_class_ap_delta'] < -SIGN_REPORT_ATOL]
        signs.update(report_only_sign_atol=SIGN_REPORT_ATOL,
            roundoff_scale_nonzero_count=sum(0 < abs(r['labels']['target_class_ap_delta']) <= SIGN_REPORT_ATOL for r in effective),
            material_positive_actions=len(material_positive), material_negative_actions=len(material_negative),
            material_positive_candidates=len({r['identity']['prediction_id'] for r in material_positive}),
            material_positive_main_scenes=sorted({main_scene(r['identity']['scene_name']) for r in material_positive}))
        # Capability descriptions, not a claim that a sample-size threshold proves AP benefit.
        signs['target_has_both_gain_and_harm_examples'] = bool(positive_scenes and negative_scenes)
        signs['gain_evidence_spans_multiple_main_scenes'] = len(positive_scenes) > 1
        signs['material_gain_evidence_spans_multiple_main_scenes'] = len(signs['material_positive_main_scenes']) > 1
        signs['can_fit_nonconstant_diagnostic'] = signs['distinct_targets'] > 1
        result['families'][family] = signs
    return result
